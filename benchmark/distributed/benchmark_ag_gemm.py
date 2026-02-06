"""Bugfix first:
Triton-distributed/python/triton_dist/kernels/nvidia/allgather_gemm.py:566
```python
M = M_per_rank * ctx.num_ranks
```
should be:
```python
M = M_per_rank * num_ranks
```
"""

# TODO: further tune the performance

import argparse
import torch
import torch.distributed as dist
import pynvshmem
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.distributed import init_distributed, dtype_map, perf_fn
from triton_dist.kernels.nvidia.allgather_gemm import ag_gemm, create_ag_gemm_context
from functools import partial

tilelang.disable_cache()


@tilelang.jit(
    out_idx=-1,
    pass_configs={"tl.disable_rdc": True},
    # FIXME: https://github.com/tile-ai/tilelang/issues/659
)
def matmut_transpose(
    rank, num_ranks, M, N_per_rank, K, block_M, block_N, block_K, dtype="float16", threads=256, persistent=False
) -> tilelang.JITKernel:
    accum_dtype = "float32"
    signal_dtype = "uint64"  # NVSHMEM requires uint64 for signal

    assert M % block_M == 0 and N_per_rank % block_N == 0 and K % block_K == 0
    M_blocks, N_blocks, K_stages = T.ceildiv(M, block_M), T.ceildiv(N_per_rank, block_N), T.ceildiv(K, block_K)
    M_blocks_per_rank = M_blocks // num_ranks

    sm_num = driver.get_num_sms()  # Get # of SMs for persistent kernel

    @T.prim_func
    def nonpersistent_kernel(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((N_per_rank, K), dtype),  # type: ignore
        signal: T.Tensor((num_ranks), signal_dtype),  # type: ignore
        C: T.Tensor((M, N_per_rank), dtype),  # type: ignore
    ):
        with T.Kernel(N_blocks, M_blocks, threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            # thread-block swizzle for allgather
            T.use_swizzle(10, order="column", offset=rank * M_blocks_per_rank)

            T.clear(C_local)

            src_rank = by // M_blocks_per_rank
            T.signal_wait_until(T.address_of(signal[src_rank]), T.CmpType.EQ, 1)
            for k in T.Pipelined(K_stages, num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    @T.prim_func
    def persistent_kernel(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((N_per_rank, K), dtype),  # type: ignore
        signal: T.Tensor((num_ranks), signal_dtype),  # type: ignore
        C: T.Tensor((M, N_per_rank), dtype),  # type: ignore
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            for bx, by in T.Persistent([M_blocks, N_blocks], sm_num, block_id):
                T.clear(C_local)

                src_rank = bx // M_blocks_per_rank
                T.signal_wait_until(T.address_of(signal[src_rank]), T.CmpType.EQ, 1)

                for k in T.Pipelined(K_stages, num_stages=3):
                    T.copy(A[bx * block_M, k * block_K], A_shared)
                    T.copy(B[by * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_M, by * block_N])

    return persistent_kernel if persistent else nonpersistent_kernel


def overlapped_ag_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    rank: int,
    num_ranks: int,
    persistent: bool = False,
) -> torch.Tensor:
    """
    Overlapped AllGather-GEMM.
    Args:
        A: local input of shape (M_per_rank, K)
        B: local weight of shape (N_per_rank, K)
        rank: current rank
        num_ranks: total number of ranks
        persistent: whether to use persistent GEMM consumers
    Returns:
        Output of shape (M, N_per_rank)
    """

    M_per_rank, K = A.shape
    N_per_rank, _ = B.shape
    assert A.shape[1] == B.shape[1], "A and B must have the same inner dimension"
    M = M_per_rank * num_ranks

    # Prepare kernel and buffers
    consumer = matmut_transpose(
        rank=rank,
        num_ranks=num_ranks,
        M=M,
        N_per_rank=N_per_rank,
        K=K,
        block_M=128,
        block_N=256,
        block_K=64,
        dtype=dtype,
        threads=threads,
        persistent=persistent,
    )
    if RANK == 0 and args.print_source:
        print("We currently use cp-engine for producer, print consumer kernel code only...")
        print(consumer.get_kernel_source())

    ag_buffer = pynvshmem.nvshmem_create_tensor_list_intra_node(
        shape=[M, K],
        dtype=A.dtype,
    )
    signal_buffer = torch.zeros([num_ranks], dtype=torch.uint64, device="cuda")

    # We place copy-based AllGather and GEMM on two streams to implement inter-op comm-comp overlapping
    ag_stream = torch.cuda.current_stream()
    gemm_stream = torch.cuda.Stream(priority=-1)
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)
    gemm_stream.wait_stream(current_stream)

    with torch.cuda.stream(ag_stream):
        ag_buffer[rank][rank * M_per_rank : (rank + 1) * M_per_rank, :].copy_(A)
        pynvshmem.write64_on_stream(signal_buffer[rank], 1, ag_stream)
        pynvshmem.nvshmemx_barrier_all_on_stream(ag_stream.cuda_stream)  # Ensure visible to all ranks
        rank_orders = [(rank + i) % num_ranks for i in range(1, num_ranks)]
        for src_rank in rank_orders:
            dst = ag_buffer[rank][src_rank * M_per_rank : (src_rank + 1) * M_per_rank, :]
            src = ag_buffer[src_rank][src_rank * M_per_rank : (src_rank + 1) * M_per_rank, :]
            dst.copy_(src)
            pynvshmem.write64_on_stream(signal_buffer[src_rank], 1, ag_stream)

    with torch.cuda.stream(gemm_stream):
        out = consumer(ag_buffer[rank], B, signal_buffer)

    current_stream.wait_stream(ag_stream)
    current_stream.wait_stream(gemm_stream)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=49152)
    parser.add_argument("--K", type=int, default=12288)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--threads", type=int, default=256, help="number of threads in a block")
    parser.add_argument("--persistent", action="store_true", default=False, help="use persistent GEMM consumers")
    parser.add_argument("--print_source", action="store_true", help="print kernel source code")
    parser.add_argument("--warmup", type=int, default=5, help="number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=10, help="number of repeat iterations")
    return parser.parse_args()


if __name__ == "__main__":
    assert torch.cuda.get_device_capability()[0] >= 9, "❗This benchmark requires sm_90 or higher"

    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP = init_distributed(return_tp_group=True)
    assert WORLD_SIZE <= 8, "This benchmark is designed for intra-node AG-GEMM"

    args = parse_args()
    M, N, K, dtype, threads, warmup, repeat = args.M, args.N, args.K, args.dtype, args.threads, args.warmup, args.repeat
    PE_num = WORLD_SIZE
    assert M % PE_num == 0 and N % PE_num == 0, "M and N must be divisible by PE_num"
    M_per_rank, N_per_rank = M // PE_num, N // PE_num
    torch_dtype = dtype_map[dtype]

    ## Inputs: A (M_per_rank, K), B (N_per_rank, K)
    ## Output: ag(A) @ B.T (M, N_per_rank)

    A = torch.randn([M_per_rank, K], dtype=torch_dtype, device="cuda")
    B = torch.randn([N_per_rank, K], dtype=torch_dtype, device="cuda")

    # Benchmark Torch (non-overlapped baseline)
    def torch_ag_gemm():
        ag_buffer = torch.empty([M, K], dtype=torch_dtype, device="cuda")
        dist.all_gather_into_tensor(ag_buffer, A, TP_GROUP)
        return ag_buffer @ B.T

    dist.barrier(TP_GROUP)
    torch_out, torch_t = perf_fn(torch_ag_gemm, warmup, repeat)
    print(f"rank {RANK} torch AG-GEMM avg time: {torch_t} ms")

    # Benchmark Triton-dist (overlapped)
    ag_intranode_stream = torch.cuda.Stream(priority=-1)

    ctx = create_ag_gemm_context(A, B, RANK, PE_num, max_M=M, for_correctness=False, ag_intranode_stream=ag_intranode_stream)

    def triton_ag_gemm(persistent, autotune):
        return ag_gemm(A, B, ctx=ctx, rank=RANK, num_ranks=PE_num, persistent=persistent, autotune=autotune)

    dist.barrier(TP_GROUP)
    triton_ag_gemm = partial(triton_ag_gemm, persistent=False, autotune=False)
    tt_out, tt_t = perf_fn(triton_ag_gemm, warmup, repeat)
    print(f"rank {RANK} triton AG-GEMM avg time: {tt_t} ms")

    # Benchmark Tilelang-dist (overlapped)
    if args.persistent:
        print("Use persistent GEMM consumers...")
    else:
        print("Use non-persistent GEMM consumers...")

    def tilelang_ag_gemm():
        return overlapped_ag_gemm(A, B, rank=RANK, num_ranks=PE_num, persistent=args.persistent)

    dist.barrier(TP_GROUP)
    tl_out, tl_t = perf_fn(tilelang_ag_gemm, warmup, repeat)
    print(f"rank {RANK} tilelang AG-GEMM avg time: {tl_t} ms")

    # Check correctness
    assert torch.allclose(tl_out, torch_out, atol=1e-2, rtol=1e-2), f"max error: {(tl_out - torch_out).abs().max()}"
    print(f"rank {RANK} check passed.✅")

    dist.destroy_process_group()
