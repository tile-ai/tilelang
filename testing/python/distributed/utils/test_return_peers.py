"""Tests for distributed allocator peer tensor views.

Requirements: >= 2 GPUs, TILELANG_USE_DISTRIBUTED=1.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.testing

os.environ.setdefault("NCCL_DEBUG", "WARN")

_USE_DISTRIBUTED = os.environ.get("TILELANG_USE_DISTRIBUTED", "0").lower() in (
    "1",
    "true",
    "on",
)

_M = 1024


def _skip_common():
    if not _USE_DISTRIBUTED:
        pytest.skip("Requires TILELANG_USE_DISTRIBUTED=1")
    if torch.cuda.device_count() < 2:
        pytest.skip("Need >= 2 GPUs")


def _worker(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    try:
        allocator = tilelang.get_allocator(
            size=2**20,
            device="cuda",
            is_distributed=True,
            local_rank=local_rank,
            num_local_ranks=num_ranks,
            group=group,
        )

        peers = tilelang.tensor((_M,), torch.float32, allocator=allocator, return_peers=True)
        assert len(peers) == num_ranks

        expected = torch.arange(10, dtype=torch.float32, device=f"cuda:{local_rank}") + 100
        if rank == 0:
            peers[1][:10] = expected.to(peers[1].device)

        torch.cuda.synchronize()
        dist.barrier(group)

        assert torch.equal(peers[1][:10].cpu(), expected.cpu()), f"rank {rank}: peer tensor write was not visible"
    finally:
        dist.destroy_process_group()


@tilelang.testing.requires_cuda
def test_return_peers():
    _skip_common()
    torch.multiprocessing.spawn(_worker, args=(2,), nprocs=2)


if __name__ == "__main__":
    tilelang.testing.main()
