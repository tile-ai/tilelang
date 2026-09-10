"""Numerical tests for SIMT-on-alloca buffer demotion.

SHARED buffers with data-dependent scatter/atomic indices (histogram bins)
cannot be SSA register tiles; they demote to per-tile-block workspace and
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


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_scratch_dynamic_grid_cached_launch_and_streams(tmp_path):
    """Two typed scratch buffers stay private across a 3-D grid and launches."""
    _skip_if_tileir_toolchain_unavailable()
    import torch

    from dataclasses import replace
    from tilelang.jit.adapter.base import CachedTextSource
    from tilelang.jit.adapter.tileir.adapter import TileIRKernelAdapter

    blocks = T.dynamic("blocks")

    @T.prim_func
    def kern(
        Idx: T.Tensor((blocks, 128), "int32"),
        Out: T.Tensor((blocks, 2, 3, 64), "int32"),
        Weighted: T.Tensor((blocks, 2, 3, 64), "int64"),
    ):
        with T.Kernel(blocks, 2, 3, threads=128) as (bx, by, bz):
            hist = T.alloc_shared((64,), "int32")
            weighted = T.alloc_shared((64,), "int64")
            idx_frag = T.alloc_fragment((128,), "int32")
            T.fill(hist, 0)
            T.fill(weighted, 7)
            T.copy(Idx[bx, 0:128], idx_frag)
            for i in T.Parallel(128):
                T.atomic_add(hist[idx_frag[i]], 1 + by + 2 * bz)
                T.atomic_add(weighted[idx_frag[i]], T.int64(2))
            for j in T.Parallel(64):
                Out[bx, by, bz, j] = hist[j]
                Weighted[bx, by, bz, j] = weighted[j]

    compiled = tilelang.compile(kern, execution_backend="tileir")
    original = compiled.adapter
    artifact = original.tileir_artifact
    assert artifact.scratch_bytes_per_block == 64 * (4 + 8)
    with pytest.raises(ValueError, match="per-block scratch bytes"):
        original._validate_cached_artifact_abi(replace(artifact, scratch_bytes_per_block=0), original.prim_func)
    assert "alloca num_elem" not in compiled.get_kernel_source()
    cache_path = tmp_path / "kernel.tileir.json"
    cache_path.write_bytes(original._serialize_tileir_artifact(artifact))
    restored = TileIRKernelAdapter.from_database(
        params=original.params,
        result_idx=[],
        target=original.target,
        func_or_mod=kern,
        host_kernel_source=CachedTextSource(text=original.get_host_source()),
        device_kernel_source=CachedTextSource(text=compiled.get_kernel_source()),
        kernel_lib_path=str(cache_path),
    )
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    pending = []
    for count, stream in zip((2, 5), streams):
        idx = torch.randint(0, 64, (count, 128), dtype=torch.int32, device="cuda")
        out = torch.empty((count, 2, 3, 64), dtype=torch.int32, device="cuda")
        weighted = torch.empty_like(out, dtype=torch.int64)
        stream.wait_stream(torch.cuda.current_stream())
        # Use an explicit non-current stream and reuse the dispatcher with a
        # different dynamic grid. Each invocation must own its workspace.
        restored(idx, out, weighted, stream=stream)
        pending.append((stream, idx, out, weighted))
    for stream, idx, out, weighted in pending:
        stream.synchronize()
        counts = torch.stack([torch.bincount(row.long(), minlength=64) for row in idx])
        for by in range(2):
            for bz in range(3):
                torch.testing.assert_close(out[:, by, bz], ((1 + by + 2 * bz) * counts).int(), rtol=0, atol=0)
                torch.testing.assert_close(weighted[:, by, bz], 7 + 2 * counts, rtol=0, atol=0)

    # Capture retains the hidden workspace; repeated replays must initialize
    # it afresh rather than accumulate values left by a previous launch.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        restored(idx, out, weighted)
    for bin_index in (3, 19):
        idx.fill_(bin_index)
        graph.replay()
        torch.cuda.synchronize()
        counts = torch.zeros((idx.shape[0], 64), dtype=torch.int64, device="cuda")
        counts[:, bin_index] = 128
        for by in range(2):
            for bz in range(3):
                torch.testing.assert_close(out[:, by, bz], ((1 + by + 2 * bz) * counts).int(), rtol=0, atol=0)
                torch.testing.assert_close(weighted[:, by, bz], 7 + 2 * counts, rtol=0, atol=0)


@skip_no_cuda_tile
@tilelang.testing.requires_cuda
def test_scratch_multi_kernel_with_automatic_output():
    """Each stage receives its own hidden scratch after its public arguments."""
    _skip_if_tileir_toolchain_unavailable()
    import torch

    @T.prim_func
    def kern(Idx: T.Tensor((128,), "int32"), Out: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=128):
            first = T.alloc_shared((64,), "int32")
            indices = T.alloc_fragment((128,), "int32")
            T.fill(first, 0)
            T.copy(Idx, indices)
            for i in T.Parallel(128):
                T.atomic_add(first[indices[i]], 1)
            for i in T.Parallel(64):
                Out[i] = first[i]
        with T.Kernel(1, threads=128):
            second = T.alloc_shared((64,), "int32")
            indices = T.alloc_fragment((128,), "int32")
            T.fill(second, 0)
            T.copy(Idx, indices)
            for i in T.Parallel(128):
                T.atomic_add(second[indices[i]], 2)
            for i in T.Parallel(64):
                Out[i] = Out[i] + second[i]

    compiled = tilelang.compile(kern, out_idx=1, execution_backend="tileir")
    artifact = compiled.adapter.tileir_artifact
    assert tuple(kernel.scratch_bytes_per_block for kernel in artifact.kernels) == (256, 256)
    compiled.adapter._validate_cached_artifact_abi(artifact, compiled.adapter.prim_func)
    idx = torch.randint(0, 64, (128,), dtype=torch.int32, device="cuda")
    result = compiled(idx)
    reference = 3 * torch.bincount(idx.long(), minlength=64).int()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
