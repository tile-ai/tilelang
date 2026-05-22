"""Tests for distributed remote TMA copy address remapping."""

import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
import tilelang.testing
from testing.python.distributed._utils import distributed_test

os.environ.setdefault("NCCL_DEBUG", "WARN")

_M = 64
_N = 128
_BLOCK_M = 16
_BLOCK_N = 128
_THREADS = 128
_EDGE_M = 65


def _kernel_remote_simt_copy_edge_fallback(M: int, N: int, block_M: int, block_N: int, threads: int):
    @T.prim_func
    def main(dst: T.Tensor((M, N), "float32"), src: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (by, bx):
            rank = T.alloc_local((1,), "uint64")
            rank[0] = T.get_rank()
            T.copy(
                src[
                    by * block_M : (by + 1) * block_M,
                    bx * block_N : (bx + 1) * block_N,
                ],
                dst[
                    by * block_M : (by + 1) * block_M,
                    bx * block_N : (bx + 1) * block_N,
                ],
                src_pe=rank[0] ^ 1,
                disable_tma=True,
            )

    return main


def _kernel_remote_simt_copy_fallback(M: int, N: int, block_M: int, block_N: int, threads: int):
    @T.prim_func
    def main(dst: T.Tensor((M, N), "float32"), src: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (by, bx):
            rank = T.alloc_local((1,), "uint64")
            rank[0] = T.get_rank()
            T.copy(
                src[
                    by * block_M : (by + 1) * block_M,
                    bx * block_N : (bx + 1) * block_N,
                ],
                dst[
                    by * block_M : (by + 1) * block_M,
                    bx * block_N : (bx + 1) * block_N,
                ],
                src_pe=rank[0] ^ 1,
                disable_tma=True,
            )

    return main


def _kernel_remote_descriptor_tma_load(M: int, N: int, block_M: int, block_N: int, threads: int):
    @T.prim_func
    def main(out: T.Tensor((M, N // 2), "float32"), src: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as by:
            rank = T.alloc_local((1,), "uint64")
            rank[0] = T.get_rank()
            shared = T.alloc_shared((block_M, block_N // 2), "float32")
            mbar = T.alloc_barrier(128)
            T.tma_copy(
                src[by * block_M : (by + 1) * block_M, 0 : block_N // 2],
                shared,
                barrier=mbar,
                src_pe=rank[0] ^ 1,
            )
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(shared, out[by * block_M : (by + 1) * block_M, 0 : block_N // 2], disable_tma=True)

    return main


def _kernel_remote_descriptor_auto_tma_store(M: int, N: int, block_M: int, block_N: int, threads: int):
    @T.prim_func
    def main(dst: T.Tensor((M, N), "float32"), src: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as by:
            rank = T.alloc_local((1,), "uint64")
            rank[0] = T.get_rank()
            shared = T.alloc_shared((block_M, N // 2), "float32")
            T.copy(src[by * block_M : (by + 1) * block_M, 0 : N // 2], shared, disable_tma=True)
            T.sync_threads()
            T.copy(
                shared,
                dst[by * block_M : (by + 1) * block_M, 0 : N // 2],
                dst_pe=rank[0] ^ 1,
            )
            T.tma_store_wait()

    return main


def _kernel_remote_tma_load(M: int, N: int, block_M: int, block_N: int, threads: int):
    @T.prim_func
    def main(out: T.Tensor((M, N), "float32"), src: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (by, bx):
            rank = T.alloc_local((1,), "uint64")
            rank[0] = T.get_rank()
            shared = T.alloc_shared((block_M, block_N), "float32")
            mbar = T.alloc_barrier(128)
            T.tma_copy(
                src[by * block_M, bx * block_N],
                shared,
                barrier=mbar,
                src_pe=rank[0] ^ 1,
            )
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(shared, out[by * block_M, bx * block_N], disable_tma=True)

    return main


def _kernel_remote_tma_store(M: int, N: int, block_M: int, block_N: int, threads: int):
    @T.prim_func
    def main(dst: T.Tensor((M, N), "float32"), src: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (by, bx):
            rank = T.alloc_local((1,), "uint64")
            rank[0] = T.get_rank()
            shared = T.alloc_shared((block_M, block_N), "float32")
            T.copy(src[by * block_M, bx * block_N], shared, disable_tma=True)
            T.sync_threads()
            T.tma_copy(
                shared,
                dst[by * block_M, bx * block_N],
                dst_pe=rank[0] ^ 1,
            )
            T.tma_store_wait()

    return main


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_remote_descriptor_tma_codegen():
    for kernel_func in (
        _kernel_remote_descriptor_tma_load,
        _kernel_remote_descriptor_auto_tma_store,
    ):
        kernel = tilelang.compile(kernel_func(_M, _N, _BLOCK_M, _BLOCK_N, _THREADS))
        source = kernel.get_kernel_source()
        host_source = kernel.get_host_source()
        assert "_desc" in source
        assert "__tvm_tensormap_create_remote_tiled" in host_source


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test()
def test_remote_descriptor_tma_copy(local_rank: int, num_ranks: int):
    from tilelang.distributed.host import init_dist

    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**22,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )

    load_kernel = tilelang.compile(
        _kernel_remote_descriptor_tma_load(_M, _N, _BLOCK_M, _BLOCK_N, _THREADS),
        compile_once=True,
        compile_group=group,
    )
    store_kernel = tilelang.compile(
        _kernel_remote_descriptor_auto_tma_store(_M, _N, _BLOCK_M, _BLOCK_N, _THREADS),
        compile_once=True,
        compile_group=group,
    )
    if rank == 0:
        for kernel in (load_kernel, store_kernel):
            source = kernel.get_kernel_source()
            host_source = kernel.get_host_source()
            assert "_desc" in source
            assert "__tvm_tensormap_create_remote_tiled" in host_source

    load_kernel.initialize(allocator=allocator)
    store_kernel.initialize(allocator=allocator)

    src = tilelang.tensor((_M, _N), torch.float32, allocator=allocator).normal_()
    out = tilelang.tensor((_M, _N // 2), torch.float32, allocator=allocator).zero_()
    dst = tilelang.tensor((_M, _N), torch.float32, allocator=allocator).zero_()

    src_refs = [torch.empty_like(src) for _ in range(num_ranks)]
    dist.all_gather(src_refs, src, group)
    expected_peer_half = src_refs[rank ^ 1][:, : _N // 2]

    torch.cuda.synchronize()
    dist.barrier(group)
    load_kernel(out, src)
    torch.cuda.synchronize()
    dist.barrier(group)

    assert torch.allclose(expected_peer_half, out, atol=1e-6, rtol=1e-6), (
        f"rank {rank}: remote descriptor TMA load mismatch"
    )

    torch.cuda.synchronize()
    dist.barrier(group)
    store_kernel(dst, src)
    torch.cuda.synchronize()
    dist.barrier(group)

    assert torch.allclose(expected_peer_half, dst[:, : _N // 2], atol=1e-6, rtol=1e-6), (
        f"rank {rank}: remote descriptor TMA store mismatch"
    )
    assert torch.allclose(torch.zeros_like(dst[:, _N // 2 :]), dst[:, _N // 2 :]), (
        f"rank {rank}: remote descriptor TMA store touched unwritten columns"
    )

    dist.destroy_process_group()


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test()
def test_remote_simt_fallback_edge_tile(local_rank: int, num_ranks: int):
    from tilelang.distributed.host import init_dist

    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**22,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )

    kernel = tilelang.compile(
        _kernel_remote_simt_copy_edge_fallback(_EDGE_M, _N, _BLOCK_M, _BLOCK_N, _THREADS),
        compile_once=True,
        compile_group=group,
    )
    if rank == 0:
        source = kernel.get_kernel_source()
        assert "tl::remote_load" in source
        assert "tl::cp_block<" not in source
    kernel.initialize(allocator=allocator)

    src = tilelang.tensor((_EDGE_M, _N), torch.float32, allocator=allocator).normal_()
    out = tilelang.tensor((_EDGE_M, _N), torch.float32, allocator=allocator).zero_()

    torch.cuda.synchronize()
    dist.barrier(group)
    kernel(out, src)
    torch.cuda.synchronize()
    dist.barrier(group)

    src_refs = [torch.empty_like(src) for _ in range(num_ranks)]
    dist.all_gather(src_refs, src, group)
    expected = src_refs[rank ^ 1]
    assert torch.allclose(expected, out, atol=1e-6, rtol=1e-6), (
        f"rank {rank}: remote SIMT edge fallback mismatch"
    )

    dist.destroy_process_group()


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test()
def test_remote_tma_copy(local_rank: int, num_ranks: int):
    from tilelang.distributed.host import init_dist

    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**22,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )

    load_kernel = tilelang.compile(
        _kernel_remote_tma_load(_M, _N, _BLOCK_M, _BLOCK_N, _THREADS),
        compile_once=True,
        compile_group=group,
    )
    store_kernel = tilelang.compile(
        _kernel_remote_tma_store(_M, _N, _BLOCK_M, _BLOCK_N, _THREADS),
        compile_once=True,
        compile_group=group,
    )
    simt_kernel = tilelang.compile(
        _kernel_remote_simt_copy_fallback(_M, _N, _BLOCK_M, _BLOCK_N, _THREADS),
        compile_once=True,
        compile_group=group,
    )
    if rank == 0:
        for kernel in (load_kernel, store_kernel):
            source = kernel.get_kernel_source()
            assert "tl::get_remote_base_ptr" in source
            assert "tl::get_uintptr_t" in source
            assert "tl::tma_" in source
        source = simt_kernel.get_kernel_source()
        assert "tl::get_remote_base_ptr" in source
        assert "tl::get_uintptr_t" in source
        assert "tl::cp_block<" in source

    load_kernel.initialize(allocator=allocator)
    store_kernel.initialize(allocator=allocator)
    simt_kernel.initialize(allocator=allocator)

    src = tilelang.tensor((_M, _N), torch.float32, allocator=allocator).normal_()
    out = tilelang.tensor((_M, _N), torch.float32, allocator=allocator).zero_()
    dst = tilelang.tensor((_M, _N), torch.float32, allocator=allocator).zero_()
    simt_out = tilelang.tensor((_M, _N), torch.float32, allocator=allocator).zero_()

    torch.cuda.synchronize()
    dist.barrier(group)
    load_kernel(out, src)
    torch.cuda.synchronize()
    dist.barrier(group)

    src_refs = [torch.empty_like(src) for _ in range(num_ranks)]
    dist.all_gather(src_refs, src, group)
    expected = src_refs[rank ^ 1]
    assert torch.allclose(expected, out, atol=1e-6, rtol=1e-6), (
        f"rank {rank}: remote TMA load mismatch"
    )

    torch.cuda.synchronize()
    dist.barrier(group)
    store_kernel(dst, src)
    torch.cuda.synchronize()
    dist.barrier(group)

    src_refs = [torch.empty_like(src) for _ in range(num_ranks)]
    dist.all_gather(src_refs, src, group)
    expected = src_refs[rank ^ 1]
    assert torch.allclose(expected, dst, atol=1e-6, rtol=1e-6), (
        f"rank {rank}: remote TMA store mismatch"
    )

    torch.cuda.synchronize()
    dist.barrier(group)
    simt_kernel(simt_out, src)
    torch.cuda.synchronize()
    dist.barrier(group)

    src_refs = [torch.empty_like(src) for _ in range(num_ranks)]
    dist.all_gather(src_refs, src, group)
    expected = src_refs[rank ^ 1]
    assert torch.allclose(expected, simt_out, atol=1e-6, rtol=1e-6), (
        f"rank {rank}: remote SIMT fallback mismatch"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    tilelang.testing.main()
