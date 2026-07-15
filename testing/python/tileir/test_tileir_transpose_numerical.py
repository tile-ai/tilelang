"""Numerical tests for transposed whole-tile reads.

A SHARED/REGISTER fragment indexed by parallel vars in a PERMUTED order
(``x_local[j, i]`` inside ``for i, j in T.Parallel(...)``) must permute the
loaded tile's axes to match the ``ordered_vars`` layout. Without the permute
the transpose is silently dropped: a square fragment passes the shape-check
and miscompiles; a rectangular one trips the downstream broadcast shape-check.
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
def test_transpose_whole_tile_read_square():
    """``xt[i, j] = x[j, i]`` over a square fragment — the square shape passes
    the shape-check, so a dropped transpose would silently miscompile."""
    _skip_if_tileir_toolchain_unavailable()
    import torch

    N = 64

    @T.prim_func
    def kern(X: T.Tensor((N, N), "float16"), Out: T.Tensor((N, N), "float16")):
        with T.Kernel(1, threads=128):
            x_local = T.alloc_fragment((N, N), "float16")
            xt = T.alloc_fragment((N, N), "float16")
            T.copy(X, x_local)
            for i, j in T.Parallel(N, N):
                xt[i, j] = x_local[j, i]
            T.copy(xt, Out)

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(0)
    x = torch.randn(N, N, dtype=torch.float16, device="cuda")
    out = torch.zeros(N, N, dtype=torch.float16, device="cuda")
    kernel(x, out)
    torch.testing.assert_close(out, x.t().contiguous(), rtol=0, atol=0)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_transpose_scale_whole_tile_read_rectangular():
    """The mamba_chunk_state pattern ``xt[i, j] = x[j, i] * scale[j]`` over a
    rectangular fragment — a dropped transpose trips the downstream broadcast
    shape-check at assembly time."""
    _skip_if_tileir_toolchain_unavailable()
    import torch

    M, K = 64, 32

    @T.prim_func
    def kern(X: T.Tensor((K, M), "float16"), S: T.Tensor((K,), "float16"), Out: T.Tensor((M, K), "float16")):
        with T.Kernel(1, threads=128):
            x_local = T.alloc_fragment((K, M), "float16")
            scale = T.alloc_fragment((K,), "float32")
            xt = T.alloc_fragment((M, K), "float16")
            T.copy(X, x_local)
            T.copy(S, scale)
            for i, j in T.Parallel(M, K):
                xt[i, j] = x_local[j, i] * scale[j]
            T.copy(xt, Out)

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(0)
    x = torch.randn(K, M, dtype=torch.float16, device="cuda")
    s = torch.randn(K, dtype=torch.float16, device="cuda")
    out = torch.zeros(M, K, dtype=torch.float16, device="cuda")
    kernel(x, s, out)
    ref = (x.t().float() * s.float()[None, :]).to(torch.float16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
