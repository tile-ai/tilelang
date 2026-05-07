"""Tests for distributed remote scalar store via T.st(..., dst_pe=...).

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
    "1",
    "true",
    "on",
)

_M = 1024
_BLOCK_M = 128
_THREADS = 128


def _skip_common():
    if not _USE_DISTRIBUTED:
        pytest.skip("Requires TILELANG_USE_DISTRIBUTED=1")
    if torch.cuda.device_count() < 2:
        pytest.skip("Need >= 2 GPUs")


def _kernel_remote_st(M: int, block_M: int, threads: int):
    @T.prim_func
    def main(dst: T.Tensor((M,), "float32"), src: T.Tensor((M,), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            tx = T.get_thread_binding()
            offset = bx * block_M + tx
            T.st(dst[offset], src[offset], dst_pe=rank[0] ^ 1)

    return main


def _worker(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**20,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )

    kernel = tilelang.compile(_kernel_remote_st(_M, _BLOCK_M, _THREADS))
    kernel.initialize(allocator=allocator)

    src = tilelang.tensor((_M,), torch.float32, allocator=allocator).normal_()
    dst = tilelang.tensor((_M,), torch.float32, allocator=allocator).zero_()

    torch.cuda.synchronize()
    dist.barrier(group)
    kernel(dst, src)
    torch.cuda.synchronize()
    dist.barrier(group)

    src_refs = [torch.empty_like(src) for _ in range(num_ranks)]
    dist.all_gather(src_refs, src, group)
    expected = src_refs[rank ^ 1]

    assert torch.allclose(expected, dst, atol=1e-6, rtol=1e-6), (
        f"rank {rank}: remote T.st mismatch"
    )

    dist.destroy_process_group()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_remote_st():
    _skip_common()
    torch.multiprocessing.spawn(_worker, args=(2,), nprocs=2)


if __name__ == "__main__":
    tilelang.testing.main()
