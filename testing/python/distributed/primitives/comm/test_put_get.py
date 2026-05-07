"""Tests for distributed put/get primitives (block and warp granularity).

All four ops are tested in a single spawn session to avoid paying per-test
import / NCCL setup overhead (~10s per spawn).  pytest -k can select individual
ops via the test names reported by the worker.

Requirements: >= 2 GPUs, compute >= 9.0, TILELANG_USE_DISTRIBUTED=1.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
import tilelang.testing

os.environ.setdefault("NCCL_DEBUG", "WARN")

_USE_DISTRIBUTED = os.environ.get("TILELANG_USE_DISTRIBUTED", "0").lower() in (
    "1", "true", "on",
)

_M = 65536
_BLOCK_M = 4096
_THREADS = 128


def _skip_common():
    if not _USE_DISTRIBUTED:
        pytest.skip("Requires TILELANG_USE_DISTRIBUTED=1")
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Need >= 2 GPUs")


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------

def _kernel_get_block(M, num_rank, block_M, threads):
    @T.prim_func
    def main(dst: T.Tensor((M), "float32"), src: T.Tensor((M), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            T.get_block(
                src=T.address_of(src[bx * block_M]),
                dst=T.address_of(dst[bx * block_M]),
                size=block_M,
                src_pe=rank[0] ^ 1,
            )
            T.fence_sys()
    return main


def _kernel_get_warp(M, num_rank, block_M, threads):
    @T.prim_func
    def main(dst: T.Tensor((M), "float32"), src: T.Tensor((M), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            warp_idx = T.get_thread_binding(0) // 32
            warp_copy_size = T.ceildiv(block_M, threads // 32)
            warp_start = bx * block_M + warp_copy_size * warp_idx
            T.get_warp(
                src=T.address_of(src[warp_start]),
                dst=T.address_of(dst[warp_start]),
                size=warp_copy_size,
                src_pe=rank[0] ^ 1,
                unroll_factor=4,
            )
            T.fence_sys()
    return main


def _kernel_put_block(M, num_rank, block_M, threads):
    @T.prim_func
    def main(dst: T.Tensor((M), "float32"), src: T.Tensor((M), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            T.put_block(
                src=T.address_of(src[bx * block_M]),
                dst=T.address_of(dst[bx * block_M]),
                size=block_M,
                dst_pe=rank[0] ^ 1,
            )
    return main


def _kernel_put_warp(M, num_rank, block_M, threads):
    @T.prim_func
    def main(dst: T.Tensor((M), "bfloat16"), src: T.Tensor((M), "bfloat16")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            warp_idx = T.get_thread_binding(0) // 32
            warp_copy_size = T.ceildiv(block_M, threads // 32)
            warp_start = bx * block_M + warp_copy_size * warp_idx
            T.put_warp(
                src=T.address_of(src[warp_start]),
                dst=T.address_of(dst[warp_start]),
                size=warp_copy_size,
                dst_pe=rank[0] ^ 1,
                unroll_factor=4,
            )
    return main


# Pre-compile all kernels at import time (warms disk cache for children)
_OP_NAMES = ["get_block", "get_warp", "put_block", "put_warp"]
_KERNEL_FNS = [_kernel_get_block, _kernel_get_warp, _kernel_put_block, _kernel_put_warp]

for _fn in _KERNEL_FNS:
    tilelang.compile(_fn(_M, 2, _BLOCK_M, _THREADS))


# ---------------------------------------------------------------------------
# Worker: runs all four kernel tests in one spawn session
# ---------------------------------------------------------------------------

def _worker(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    rank, num_ranks, group = init_dist(local_rank, num_ranks)

    allocator = tilelang.get_allocator(
        size=2**25,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )

    for name, kernel_fn in zip(_OP_NAMES, _KERNEL_FNS):
        kernel = tilelang.compile(kernel_fn(_M, num_ranks, _BLOCK_M, _THREADS))
        kernel.initialize(allocator=allocator)

        dtype = torch.bfloat16 if name == "put_warp" else torch.float32
        src = tilelang.tensor((_M,), dtype, allocator=allocator).normal_()
        dst = tilelang.tensor((_M,), dtype, allocator=allocator)

        torch.cuda.synchronize()
        dist.barrier(group)
        kernel(dst, src)
        torch.cuda.synchronize()
        dist.barrier(group)

        dst_refs = [torch.empty_like(src) for _ in range(num_ranks)]
        dist.all_gather(dst_refs, src, group)
        expected = dst_refs[local_rank ^ 1]

        assert torch.allclose(expected, dst, atol=1e-6, rtol=1e-6), (
            f"rank {local_rank}: {name} mismatch"
        )

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------

@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_put_get():
    """Spawn once, test all four put/get primitives in sequence."""
    _skip_common()
    torch.multiprocessing.spawn(_worker, args=(2,), nprocs=2)


if __name__ == "__main__":
    tilelang.testing.main()
