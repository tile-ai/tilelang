"""
Multicast (NVSwitch SHARP) shared-memory operations.

Single-GPU tests run directly under pytest.
Multi-GPU tests use torch.multiprocessing.spawn and require >= 2 GPUs.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
import tilelang.testing
from testing.python.distributed._utils import distributed_test

os.environ.setdefault("NCCL_DEBUG", "WARN")

_N = 256
_BLOCK_N = 256
_THREADS = 128


def _multimem_reduce_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((_N,), T.float32), out: T.Tensor((_N,), T.float32)):
        with T.Kernel(1, threads=_THREADS):
            tmp = T.alloc_fragment((_BLOCK_N,), T.float32)
            T.multimem_ld_reduce(
                mcast_buf[0:_BLOCK_N],
                tmp,
                reduce_op=T.MultimemReduceOp.ADD,
            )
            T.copy(tmp, out[0:_BLOCK_N])

    return main

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

@distributed_test(nprocs=None, require_multicast=True)
def test_distributed_multicast_allocator(local_rank: int, num_ranks: int):
    """Create multicast buffer via BaseAllocator and verify multimem access."""
    from tilelang.distributed.host import init_dist
    from tilelang.distributed.allocator import BaseAllocator

    _, _, group = init_dist(local_rank, num_ranks)

    allocator = BaseAllocator(
        size=2**25,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
        use_vmm=True,
        mcast_size=_N * torch.empty((), dtype=torch.float32).element_size(),
    )

    assert allocator._initialized
    assert allocator.ptr != 0
    assert allocator._use_multicast

    kernel = tilelang.compile(
        _multimem_reduce_kernel(),
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
        compile_once=True,
        compile_group=group,
    )
    kernel.initialize(allocator=allocator)

    # Allocate from MC buffer: mcast_t is backed by MC VA for multimem
    # instructions, local_t by this rank's physical VA for ordinary writes.
    mcast_t, local_t = allocator._allocate_mcast_tensor((_N,), torch.float32)
    assert mcast_t.shape == (_N,)

    local_t.fill_(float(local_rank + 1))
    out = tilelang.tensor((_N,), torch.float32, allocator=allocator).zero_()

    torch.cuda.synchronize()
    dist.barrier(group)

    kernel(mcast_t, out)
    torch.cuda.synchronize()
    dist.barrier(group)

    expected_sum = float(num_ranks * (num_ranks + 1) // 2)
    expected = torch.full((_N,), expected_sum, dtype=torch.float32, device=f"cuda:{local_rank}")
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)

    dist.barrier(group)
    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    tilelang.testing.main()
