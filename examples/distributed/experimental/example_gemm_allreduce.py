"""
Fused GEMM + AllReduce using multimem.red — single kernel.

Each rank computes a partial GEMM (C_partial = A_local @ B), then uses
multimem.red to reduce partial results into the multicast buffer across
all ranks. A final multimem.ld_reduce reads back the fully reduced result.

Multi-process multi-GPU: each process manages one GPU, multicast handle
shared via fabric handles through torch.distributed.

Usage:
  export TILESCALE_USE_VMM=1
  export NCCL_IB_DISABLE=1
  export TILELANG_USE_DISTRIBUTED=1
  python examples/distributed/example_gemm_allreduce.py [--num-processes 8]

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

os.environ["NCCL_DEBUG"] = "WARN"


def gemm_allreduce_kernel(M, N, K, block_M, block_N, block_K, threads, dtype=T.float16):
    """Fused GEMM + multimem allreduce in a single kernel.

    Each rank computes C_partial[M,N] = A_local[M,K] @ B[K,N],
    then reduces into mcast_buf via multimem.red (accumulate without read-back).
    After barrier, a second kernel (or ld_reduce) retrieves the final result.

    For simplicity, this example does:
      1. GEMM into fragment
      2. multimem.red to push partial sum into mcast buffer
    A separate ld_reduce kernel reads the final reduced result.
    """
    accum_dtype = T.float32

    @T.prim_func
    def gemm_red(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        mcast_C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[k * block_K : (k + 1) * block_K, bx * block_N : (bx + 1) * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # Reduce partial GEMM result into multicast buffer
            T.multimem_red(
                C_local,
                mcast_C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
                reduce_op=T.MultimemReduceOp.ADD,
            )
            T.fence_sys()

    return gemm_red


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    M, N, K = args.M, args.N, args.K
    block_M, block_N, block_K = args.block_m, args.block_n, args.block_k
    threads = args.threads
    dtype = torch.float16

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # Multicast buffer for C (M x N, float32)
    mcast_bytes = M * N * 4
    allocator = get_allocator(
        size=mcast_bytes,
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
        mcast_size=mcast_bytes,
    )

    # Compile kernels
    gemm_red_func = gemm_allreduce_kernel(M, N, K, block_M, block_N, block_K, threads)
    gemm_red_kernel = tilelang.compile(
        gemm_red_func,
        pass_configs={"tl.disable_tma_lower": True},
    )

    if local_rank == 0 and args.print_source:
        print("=== GEMM + Red kernel ===")
        print(gemm_red_kernel.get_kernel_source())

    # Per-rank input
    torch.manual_seed(42 + rank)
    A_local = torch.randn(M, K, dtype=dtype, device=f"cuda:{local_rank}")
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{local_rank}")

    # Each rank computes local GEMM as reference (same precision as kernel)
    C_local_ref = (A_local @ B).float()

    # NCCL all_reduce as ground truth for the reduction
    expected = C_local_ref.clone()
    dist.all_reduce(expected, op=dist.ReduceOp.SUM, group=group)

    # Multicast buffer for partial sum accumulation
    mcast_C, local_C = allocator._allocate_mcast_tensor((M * N,), torch.float32)
    local_C.zero_()
    torch.cuda.synchronize()
    dist.barrier(group)

    # Run fused GEMM + multimem.red
    gemm_red_kernel(A_local, B, mcast_C.view(M, N))
    torch.cuda.synchronize()
    dist.barrier(group)

    # Read back reduced result from local physical memory
    result = local_C.view(M, N)
    torch.cuda.synchronize()

    # Compare: only validates multimem.red correctness (GEMM precision identical)
    atol = 1e-5
    max_diff = (result - expected).abs().max().item()
    passed = max_diff < atol

    if local_rank == 0:
        print(f"M={M}, N={N}, K={K}, num_ranks={num_ranks}, max_diff={max_diff:.6f}")
    if passed:
        print(f"[rank {local_rank}] PASSED")
    else:
        print(f"[rank {local_rank}] FAILED (max_diff={max_diff:.6f})")

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--block_m", type=int, default=128)
    parser.add_argument("--block_n", type=int, default=256)
    parser.add_argument("--block_k", type=int, default=64)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--print_source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
