"""
Multicast (NVSwitch SHARP) shared-memory operations.

Single-GPU tests run directly under pytest.
Multi-GPU tests use torch.multiprocessing.spawn and require >= 2 GPUs.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang.testing

os.environ.setdefault("NCCL_DEBUG", "WARN")

_USE_DISTRIBUTED = os.environ.get("TILELANG_USE_DISTRIBUTED", "0").lower() in (
    "1", "true", "on",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_multicast():
    from tilelang.distributed.shared_memory import _supports_multicast
    if not _supports_multicast():
        pytest.skip("NVSwitch multicast not supported on this hardware")


def _skip_common():
    if not _USE_DISTRIBUTED:
        pytest.skip("Requires TILELANG_USE_DISTRIBUTED=1")
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        pytest.skip(f"Need >= 2 GPUs, found {num_gpus}")


# ---------------------------------------------------------------------------
# Single-GPU tests
# ---------------------------------------------------------------------------

@tilelang.testing.requires_cuda
def test_supports_multicast():
    from tilelang.distributed.shared_memory import _supports_multicast
    result = _supports_multicast()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Multi-GPU worker functions (called by spawn)
# ---------------------------------------------------------------------------

def _worker_distributed_multicast_allocator(local_rank: int, num_ranks: int):
    """Create multicast buffer via BaseAllocator, verify P2P access through MC VAs."""
    from tilelang.distributed import init_dist
    from tilelang.utils.allocator import BaseAllocator

    tilelang.disable_cache()
    _, _, group = init_dist(local_rank, num_ranks)

    allocator = BaseAllocator(
        size=2**25,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
        use_vmm=True,
        mcast_size=1024 * 1024,  # 1 MB multicast buffer
    )

    assert allocator._initialized
    assert allocator.ptr != 0
    assert allocator._use_multicast

    # Allocate from MC buffer: mcast_t backed by MC VA, local_t by phys VA
    mcast_t, local_t = allocator._allocate_mcast_tensor((256,), torch.bfloat16)
    assert mcast_t.shape == (256,)

    # Each rank writes data to its local (physical) memory
    local_t.fill_(float(local_rank + 1))
    torch.cuda.synchronize()
    dist.barrier(group)

    # Read from MC VA — should be visible to all ranks
    mcast_val = mcast_t[0].item()
    assert mcast_val >= 1.0 and mcast_val <= float(num_ranks), (
        f"rank {local_rank}: mcast_t[0] = {mcast_val}, expected 1..{num_ranks}"
    )

    dist.barrier(group)
    allocator.close()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Multi-GPU pytest entry point
# ---------------------------------------------------------------------------

@tilelang.testing.requires_cuda
def test_distributed_multicast_allocator():
    _skip_if_no_multicast()
    _skip_common()
    num_gpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        _worker_distributed_multicast_allocator, args=(num_gpus,), nprocs=num_gpus
    )


if __name__ == "__main__":
    tilelang.testing.main()
