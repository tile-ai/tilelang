"""Small multimem all-reduce correctness test."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
import tilelang.testing
from testing.python.distributed._utils import distributed_test

os.environ.setdefault("NCCL_DEBUG", "WARN")

_N = 1024
_BLOCK_N = 256
_THREADS = 128


def _multimem_allreduce_kernel(N: int, block_N: int, threads: int):
    @T.prim_func
    def main(mcast_buf: T.Tensor((N,), T.float32), out: T.Tensor((N,), T.float32)):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            tmp = T.alloc_fragment((block_N,), T.float32)
            T.multimem_ld_reduce(
                mcast_buf[bx * block_N : (bx + 1) * block_N],
                tmp,
                reduce_op=T.MultimemReduceOp.ADD,
            )
            T.copy(tmp, out[bx * block_N : (bx + 1) * block_N])

    return main


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test(nprocs=None, require_multicast=True)
def test_multimem(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    rank, _, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**22,
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
        mcast_size=_N * torch.empty((), dtype=torch.float32).element_size(),
    )

    kernel = tilelang.compile(
        _multimem_allreduce_kernel(_N, _BLOCK_N, _THREADS),
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
        compile_once=True,
        compile_group=group,
    )
    kernel.initialize(allocator=allocator)

    torch.manual_seed(100 + rank)
    local_data = torch.randn(_N, dtype=torch.float32, device=f"cuda:{local_rank}")
    mcast_buf, local_buf = allocator._allocate_mcast_tensor((_N,), torch.float32)
    local_buf.copy_(local_data)
    out = tilelang.tensor((_N,), torch.float32, allocator=allocator).zero_()

    torch.cuda.synchronize()
    dist.barrier(group)
    kernel(mcast_buf, out)
    torch.cuda.synchronize()
    dist.barrier(group)

    expected = local_data.clone()
    dist.all_reduce(expected, op=dist.ReduceOp.SUM, group=group)
    assert torch.allclose(expected, out, atol=1e-5, rtol=1e-5)

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
