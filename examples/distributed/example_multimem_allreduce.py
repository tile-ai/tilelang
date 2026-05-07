"""
Multimem allreduce example using NVSwitch multicast instructions.

Multi-process multi-GPU: each process manages one GPU, multicast handle
shared via fabric handles through torch.distributed.

Usage:
  export TILESCALE_USE_VMM=1
  export NCCL_IB_DISABLE=1
  export TILELANG_USE_DISTRIBUTED=1
  python examples/distributed/example_multimem_allreduce.py [--num-processes 8]

Requirements:
  - NVSwitch with multicast support (H100/B200 DGX)
"""

import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
from tilelang.distributed import init_dist
from tilelang.utils.allocator import get_allocator

tilelang.disable_cache()
os.environ["NCCL_DEBUG"] = "WARN"


def multimem_allreduce_kernel_one_shot(N, block_N, threads):
    @T.prim_func
    def main(
        mcast_buf: T.Tensor((N,), T.float32),
        result: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx,):
            result_local = T.alloc_fragment([block_N], T.float32)
            T.multimem_ld_reduce(
                mcast_buf[bx * block_N : (bx + 1) * block_N],
                result_local,
                reduce_op=T.MultimemReduceOp.ADD,
            )
            T.copy(result_local, result[bx * block_N : (bx + 1) * block_N])

    return main


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    N = args.N
    BLOCK_N = args.block_n
    threads = args.threads

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # Create allocator with integrated multicast buffer
    allocator = get_allocator(
        size=N * 4,  # float32 = 4 bytes
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
        mcast_size=N * 4,
    )

    # Compile kernel
    kernel = tilelang.compile(
        multimem_allreduce_kernel_one_shot(N, BLOCK_N, threads),
        pass_configs={"tl.disable_tma_lower": True},
    )
    if local_rank == 0 and args.print_source:
        print(kernel.get_kernel_source())

    # Random input per rank
    torch.manual_seed(42 + local_rank)
    local_data = torch.randn(N, dtype=torch.float32, device=f"cuda:{local_rank}")

    # Allocate from multicast buffer
    # mcast_tensor: MC VA for multimem instructions (read)
    # local_tensor: physical VA for writing data
    mcast_tensor, local_tensor = allocator._allocate_mcast_tensor((N,), torch.float32)

    # Write to physical memory (NOT the MC VA)
    local_tensor.copy_(local_data)
    torch.cuda.synchronize()
    dist.barrier(group)
    result = torch.empty(N, dtype=torch.float32, device=f"cuda:{local_rank}")
    kernel(mcast_tensor, result)
    torch.cuda.synchronize()

    # torch.distributed reference
    expected = local_data.clone()
    dist.all_reduce(expected, op=dist.ReduceOp.SUM, group=group)

    # Compare (fp32 should be exact or near-exact)
    atol = 1e-5
    max_diff = (result - expected).abs().max().item()
    passed = max_diff < atol

    if local_rank == 0:
        print(f"N={N}, num_ranks={num_ranks}, max_diff={max_diff:.4f}, atol={atol}")
    if passed:
        print(f"[rank {local_rank}] PASSED")
    else:
        print(f"[rank {local_rank}] FAILED (max_diff={max_diff:.4f})")

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--N", type=int, default=65536)
    parser.add_argument("--block_n", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--print_source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
