"""Tests for T.get_rank() and T.get_num_ranks() primitives.

Both ops are tested in a single spawn session.  Kernels are pre-compiled at
module import time so children load from disk cache.

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

_THREADS = 32


def _skip_common():
    if not _USE_DISTRIBUTED:
        pytest.skip("Requires TILELANG_USE_DISTRIBUTED=1")
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Need >= 2 GPUs")


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------

def _kernel_get_rank():
    @T.prim_func
    def main(out: T.Tensor((1,), T.uint64)):
        with T.Kernel(1, threads=_THREADS) as (_bx,):
            rank = T.alloc_local([1], T.uint64)
            rank[0] = T.get_rank()
            if T.get_thread_binding(0) == 0:
                out[0] = rank[0]
    return main


def _kernel_get_num_ranks():
    @T.prim_func
    def main(out: T.Tensor((1,), T.uint64)):
        with T.Kernel(1, threads=_THREADS) as (_bx,):
            num_ranks = T.alloc_local([1], T.uint64)
            num_ranks[0] = T.get_num_ranks()
            if T.get_thread_binding(0) == 0:
                out[0] = num_ranks[0]
    return main


# Pre-compile at import time
tilelang.compile(_kernel_get_rank())
tilelang.compile(_kernel_get_num_ranks())


# ---------------------------------------------------------------------------
# Worker: tests both in one spawn session
# ---------------------------------------------------------------------------

_KERNEL_NAMES = ["get_rank", "get_num_ranks"]
_KERNELS = [_kernel_get_rank, _kernel_get_num_ranks]


def _worker(local_rank: int, num_local_ranks: int):
    from tilelang.distributed import init_dist

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    allocator = tilelang.get_allocator(
        size=2**20,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
    )

    for name, kernel_fn in zip(_KERNEL_NAMES, _KERNELS):
        kernel = tilelang.compile(kernel_fn())
        kernel.initialize(allocator=allocator)

        out = tilelang.tensor((1,), T.uint64, allocator=allocator)

        torch.cuda.synchronize()
        dist.barrier(group)
        kernel(out)
        torch.cuda.synchronize()
        dist.barrier(group)

        out_cpu = out.cpu()

        if name == "get_rank":
            assert out_cpu[0].item() == local_rank, (
                f"rank {local_rank}: get_rank() = {out_cpu[0].item()}, expected {local_rank}"
            )
        else:
            assert out_cpu[0].item() == num_ranks, (
                f"rank {local_rank}: get_num_ranks() = {out_cpu[0].item()}, expected {num_ranks}"
            )

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------

@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_basic():
    """Spawn once, test get_rank + get_num_ranks."""
    _skip_common()
    torch.multiprocessing.spawn(_worker, args=(2,), nprocs=2)


if __name__ == "__main__":
    tilelang.testing.main()
