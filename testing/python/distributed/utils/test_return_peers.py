"""Tests for distributed allocator peer tensor views.

Requirements: >= 2 GPUs, TILELANG_USE_DISTRIBUTED=1.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

import tilelang
from testing.python.distributed._utils import distributed_test

os.environ.setdefault("NCCL_DEBUG", "WARN")

_M = 1024


@distributed_test()
def test_return_peers(local_rank: int, num_ranks: int):
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


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
