"""Numerical tests for data-dependent gather/scatter (GatherLoad + gather-form
AtomicRMW): per-element pointer tiles via load_ptr_tko / atomic_rmw_tko.

Covers the sparse_mla-style patterns:
  - ``dst[i, j] = Src[Indices[i], j]`` — a tile-valued row index.
  - ``T.atomic_add(Dst[Indices[i], j], Val[i, j])`` — data-dependent scatter-add.
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
def test_gather_load_row_indices_numerical():
    _skip_if_tileir_toolchain_unavailable()
    import torch

    ROWS, N, D = 512, 64, 128

    @T.prim_func
    def kern(
        Src: T.Tensor((512, 128), "float16"),
        Indices: T.Tensor((64,), "int32"),
        Out: T.Tensor((64, 128), "float16"),
    ):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((64, 128), "float16")
            for i, j in T.Parallel(64, 128):
                frag[i, j] = Src[Indices[i], j]
            T.copy(frag, Out)

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(0)
    src = torch.randn(ROWS, D, dtype=torch.float16, device="cuda")
    idx = torch.randint(0, ROWS, (N,), dtype=torch.int32, device="cuda")
    out = torch.empty(N, D, dtype=torch.float16, device="cuda")
    kernel(src, idx, out)
    torch.testing.assert_close(out, src[idx.long()], rtol=0, atol=0)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_gather_atomic_scatter_add_numerical():
    _skip_if_tileir_toolchain_unavailable()
    import torch

    ROWS, N, D = 256, 64, 128

    @T.prim_func
    def kern(
        Val: T.Tensor((64, 128), "float32"),
        Indices: T.Tensor((64,), "int32"),
        Dst: T.Tensor((256, 128), "float32"),
    ):
        with T.Kernel(4, threads=128):
            frag = T.alloc_fragment((64, 128), "float32")
            T.copy(Val, frag)
            for i, j in T.Parallel(64, 128):
                T.atomic_add(Dst[Indices[i], j], frag[i, j])

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(0)
    val = torch.randn(N, D, dtype=torch.float32, device="cuda")
    # Duplicate indices exercise atomic conflict resolution across the 4 CTAs.
    idx = torch.randint(0, ROWS, (N,), dtype=torch.int32, device="cuda")
    dst = torch.zeros(ROWS, D, dtype=torch.float32, device="cuda")
    kernel(val, idx, dst)
    ref = torch.zeros_like(dst)
    # 4 CTAs each add val once.
    for _ in range(4):
        ref.index_add_(0, idx.long(), val)
    torch.testing.assert_close(dst, ref, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    tilelang.testing.main()
