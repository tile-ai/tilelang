"""Numerical tests for SIMT-on-alloca buffer demotion.

SHARED buffers with data-dependent scatter/atomic indices (histogram bins)
cannot be SSA register tiles; they demote to ``alloca global`` scratch and
all access goes through the pointer gather/scatter/atomic machinery.
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
def test_alloca_histogram_scatter_add_numerical():
    """Shared histogram with data-dependent atomic bins demotes to alloca."""
    _skip_if_tileir_toolchain_unavailable()
    import torch

    N, RADIX, BLOCKS = 256, 64, 4

    @T.prim_func
    def kern(Idx: T.Tensor((256,), "int32"), Out: T.Tensor((64,), "int32")):
        with T.Kernel(4, threads=128):
            hist = T.alloc_shared((64,), "int32")
            idx_frag = T.alloc_fragment((256,), "int32")
            T.fill(hist, 0)
            T.copy(Idx, idx_frag)
            for i in T.Parallel(256):
                T.atomic_add(hist[idx_frag[i]], 1)
            for j in T.Parallel(64):
                T.atomic_add(Out[j], hist[j])

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(0)
    idx = torch.randint(0, RADIX, (N,), dtype=torch.int32, device="cuda")
    out = torch.zeros(RADIX, dtype=torch.int32, device="cuda")
    kernel(idx, out)
    # Each of the 4 blocks accumulates the same histogram into Out.
    ref = BLOCKS * torch.bincount(idx.long(), minlength=RADIX).to(torch.int32)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_alloca_return_prev_scatter_positions_numerical():
    """`pos = T.atomic_add(counter[bin], 1, return_prev=True)` as an RHS value:
    the shared counter demotes to alloca and each lane hitting a bin gets a
    distinct running position (stream-compaction slot)."""
    _skip_if_tileir_toolchain_unavailable()
    import torch

    N, RADIX = 128, 64

    @T.prim_func
    def kern(Idx: T.Tensor((128,), "int32"), Pos: T.Tensor((128,), "int32"), Cnt: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=128):
            counter = T.alloc_shared((64,), "int32")
            idx_frag = T.alloc_fragment((128,), "int32")
            T.fill(counter, 0)
            T.copy(Idx, idx_frag)
            for i in T.Parallel(128):
                Pos[i] = T.atomic_add(counter[idx_frag[i]], 1, return_prev=True)
            for j in T.Parallel(64):
                Cnt[j] = counter[j]

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(0)
    idx = torch.randint(0, RADIX, (N,), dtype=torch.int32, device="cuda")
    pos = torch.zeros(N, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(RADIX, dtype=torch.int32, device="cuda")
    kernel(idx, pos, cnt)

    ref_cnt = torch.bincount(idx.long(), minlength=RADIX).to(torch.int32)
    torch.testing.assert_close(cnt, ref_cnt, rtol=0, atol=0)
    # Per bin, the returned positions must be a permutation of {0..count-1}.
    for b in range(RADIX):
        lanes = (idx == b).nonzero(as_tuple=True)[0]
        assert sorted(pos[lanes].tolist()) == list(range(len(lanes)))


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_alloca_masked_scatter_add_numerical():
    """A parallel-`if` guard around a data-dependent atomic must predicate the
    scatter: lanes failing the mask contribute nothing."""
    _skip_if_tileir_toolchain_unavailable()
    import torch

    N, RADIX = 128, 64

    @T.prim_func
    def kern(Idx: T.Tensor((128,), "int32"), Cnt: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=128):
            counter = T.alloc_shared((64,), "int32")
            idx_frag = T.alloc_fragment((128,), "int32")
            T.fill(counter, 0)
            T.copy(Idx, idx_frag)
            for i in T.Parallel(128):
                if idx_frag[i] >= 0:
                    T.atomic_add(counter[idx_frag[i]], 1)
            for j in T.Parallel(64):
                Cnt[j] = counter[j]

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(1)
    idx = torch.randint(-8, RADIX, (N,), dtype=torch.int32, device="cuda")
    cnt = torch.zeros(RADIX, dtype=torch.int32, device="cuda")
    kernel(idx, cnt)

    ref = torch.bincount(idx[idx >= 0].long(), minlength=RADIX).to(torch.int32)
    torch.testing.assert_close(cnt, ref, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
