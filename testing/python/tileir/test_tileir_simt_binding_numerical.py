"""Numerical tests for inner SIMT thread bindings and reshape views.

An explicit ``T.thread_binding(threadIdx.x)`` inside T.Kernel is a
single-axis T.Parallel over the lanes; conditional stores under it are TRUE
predicated stores (masked lanes untouched). ``T.reshape`` views share the
base buffer's tile through alias redirection.
"""

from __future__ import annotations

import pytest

import tilelang
import tilelang.testing
from tilelang import language as T

try:
    from cuda_tile._mlir import ir as _ir  # noqa: F401

    _HAS_CUDA_TILE = True
except ImportError:
    _HAS_CUDA_TILE = False

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)


def _skip_if_tileir_toolchain_unavailable():
    from tilelang.tileir.checks import TileIRDependencyError, check_tileir_available

    try:
        check_tileir_available()
    except TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_inner_simt_binding_predicated_store_numerical():
    """clean_logits_-style kernel: per-lane conditional -inf writes.

    Masked lanes (the valid [s, e) range) must be left UNTOUCHED — a
    select-zero store would clobber them with 0.
    """
    _skip_if_tileir_toolchain_unavailable()
    import torch

    ROWS, COLS, _THREADS = 64, 1024, 256

    @T.prim_func
    def kern(
        Data: T.Tensor((64, 1024), "float32"),
        S: T.Tensor((64,), "int32"),
        E: T.Tensor((64,), "int32"),
    ):
        with T.Kernel(64, threads=256) as bx:
            tx = T.thread_binding(0, 256, thread="threadIdx.x")
            s = S[bx]
            e = E[bx]
            for k_i in T.serial(1024 // 256):
                idx = k_i * 256 + tx
                if idx < s or idx >= e:
                    Data[bx, idx] = -T.infinity("float32")

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(0)
    data = torch.randn(ROWS, COLS, dtype=torch.float32, device="cuda")
    ref = data.clone()
    s = torch.randint(0, COLS // 2, (ROWS,), dtype=torch.int32, device="cuda")
    e = torch.randint(COLS // 2, COLS, (ROWS,), dtype=torch.int32, device="cuda")
    kernel(data, s, e)
    cols = torch.arange(COLS, device="cuda").unsqueeze(0)
    oob = (cols < s.unsqueeze(1)) | (cols >= e.unsqueeze(1))
    ref[oob] = float("-inf")
    torch.testing.assert_close(data, ref, rtol=0, atol=0)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_reshape_view_alias_numerical():
    """T.reshape view: writes through the view are visible via the base."""
    _skip_if_tileir_toolchain_unavailable()
    import torch

    @T.prim_func
    def kern(X: T.Tensor((64, 32), "float32"), Out: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((64, 32), "float32")
            frag3 = T.reshape(frag, (64, 8, 4))
            T.copy(X, frag)
            for i, j, k in T.Parallel(64, 8, 4):
                frag3[i, j, k] = frag3[i, j, k] * 2.0
            red = T.alloc_fragment((64,), "float32")
            T.reduce_sum(frag, red, dim=1)
            T.copy(red, Out)

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(0)
    x = torch.randn(64, 32, dtype=torch.float32, device="cuda")
    out = torch.empty(64, dtype=torch.float32, device="cuda")
    kernel(x, out)
    torch.testing.assert_close(out, (x * 2.0).sum(dim=1), rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    tilelang.testing.main()
