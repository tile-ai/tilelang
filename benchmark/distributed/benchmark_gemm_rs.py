# Currently we only implement in Tilelang
# TODO: add Triton-dist v3.4 impl
# TODO: further tune the performance

import argparse
import torch
import torch.distributed as dist
import pynvshmem
import tilelang
import tilelang.language as T

# from tilelang.carver.arch import driver
from tilelang.distributed import init_distributed, dtype_map, perf_fn

tilelang.disable_cache()


@tilelang.jit(
    pass_configs={"tl.disable_rdc": True}
    # FIXME: https://github.com/tile-ai/tilelang/issues/659
)
def fused_gemm_scatter(
    rank, num_ranks, M, N, K_per_rank, block_M, block_N, block_K, dtype="float16", threads=128, persistent=False
) -> tilelang.JITKernel:
    accum_dtype = "float32"

    assert M % block_M == 0 and N % block_N == 0 and K_per_rank % block_K == 0
    M_blocks, N_blocks, K_stages = T.ceildiv(M, block_M), T.ceildiv(N, block_N), T.ceildiv(K_per_rank, block_K)
    M_blocks_per_rank = M_blocks // num_ranks

    # sm_num = driver.get_num_sms()  # Get # of SMs for persistent kernel

    @T.prim_func
    def nonpersistent_kernel(
        A: T.Tensor((M, K_per_rank), dtype),  # type: ignore
        B: T.Tensor((N, K_per_rank), dtype),  # type: ignore
        C: T.Tensor((M_blocks, N_blocks, block_M, block_N), dtype),  # type: ignore
    ):
        with T.Kernel(N_blocks, M_blocks, threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            # thread-block swizzle for allgather
            T.use_swizzle(M_blocks, order="column")

            T.clear(C_local)

            for k in T.Pipelined(K_stages, num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by, bx, :, :])
            peer = by // M_blocks_per_rank
            T.putmem_nbi_block(
                T.address_of(C[by, bx, 0, 0]), T.address_of(C[by, bx, 0, 0]), block_M * block_N * dtype_map[dtype].itemsize, peer
            )

    assert not persistent
    return nonpersistent_kernel


# https://github.com/bytedance/flux/blob/main/docs/design.md
def overlapped_gemm_rs(
    input: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    num_ranks: int,
    persistent: bool = False,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
) -> torch.Tensor:
    """Overlapped GEMM with Reduce-Scatter using Tilelang.
    Args:
        input (torch.Tensor): Input tensor of shape (M, K_per_rank).
        weight (torch.Tensor): Weight tensor of shape (N, K_per_rank).
        rank (int): Current rank.
        num_ranks (int): Total number of ranks.
        persistent (bool): Whether to use persistent GEMM producers.
    Returns:
        torch.Tensor: Output tensor of shape (M_per_rank, N).
    """

    M, K_per_rank = input.shape
    N, _ = weight.shape
    assert weight.shape[1] == K_per_rank, "Weight tensor's second dimension must match K_per_rank"
    M_per_rank = M // num_ranks
    M_blocks, N_blocks = M // block_M, N // block_N

    # Prepare kernels and buffers
    fused_gemm_scatter_kernel = fused_gemm_scatter(
        rank=rank,
        num_ranks=num_ranks,
        M=M,
        N=N,
        K_per_rank=K_per_rank,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype=dtype,
        threads=threads,
        persistent=persistent,
    )

    gemm_output = pynvshmem.nvshmem_create_tensor_list_intra_node([M_blocks, N_blocks, block_M, block_N], dtype=input.dtype)
    output = torch.empty((M_per_rank, N), dtype=input.dtype, device="cuda")
    fused_gemm_scatter_kernel(input, weight, gemm_output[rank])
    dist.barrier(TP_GROUP)
    output = gemm_output[rank].transpose(1, 2).view((num_ranks, M_per_rank, N)).sum(0)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=16384)
    parser.add_argument("--N", type=int, default=12288)
    parser.add_argument("--K", type=int, default=49152)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--threads", type=int, default=128, help="number of threads in a block")
    parser.add_argument("--persistent", action="store_true", default=False, help="use persistent GEMM producers")
    parser.add_argument("--print_source", action="store_true", help="print kernel source code")
    parser.add_argument("--warmup", type=int, default=5, help="number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=10, help="number of repeat iterations")
    return parser.parse_args()


if __name__ == "__main__":
    assert torch.cuda.get_device_capability()[0] >= 9, "❗This benchmark requires sm_90 or higher"

    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP = init_distributed(return_tp_group=True)
    assert WORLD_SIZE <= 8, "This benchmark is designed for intra-node GEMM-RS"

    args = parse_args()
    M, N, K, dtype, threads, warmup, repeat = args.M, args.N, args.K, args.dtype, args.threads, args.warmup, args.repeat
    PE_num = WORLD_SIZE
    assert M % PE_num == 0 and K % PE_num == 0, "M and K must be divisible by PE_num"
    M_per_rank, K_per_rank = M // PE_num, K // PE_num
    torch_dtype = dtype_map[dtype]

    ## Inputs: input (M, K_per_rank), weight (N, K_per_rank)
    ## Output: rs(input@weight.T) (M_per_rank, N)

    input = torch.randn([M, K_per_rank], dtype=torch_dtype, device="cuda")
    weight = torch.randn([N, K_per_rank], dtype=torch_dtype, device="cuda")

    # Benchmark Torch (non-overlapped baseline)
    def torch_gemm_rs():
        local_output = input @ weight.T
        rs_output = torch.empty((M // PE_num, N), dtype=torch_dtype, device="cuda")
        dist.reduce_scatter_tensor(rs_output, local_output, group=TP_GROUP)
        return rs_output

    dist.barrier(TP_GROUP)
    torch_out, torch_t = perf_fn(torch_gemm_rs, warmup, repeat)
    print(f"rank {RANK} torch GEMM-RS avg time: {torch_t} ms")

    # TODO(wt) Add Triton-dist baseline (overlapped)

    # Benchmark Tilelang-dist (overlapped)
    if args.persistent:
        print("Use persistent GEMM producers...")
    else:
        print("Use non-persistent GEMM producers...")

    def tilelang_gemm_rs():
        return overlapped_gemm_rs(input, weight, rank=RANK, num_ranks=PE_num, persistent=args.persistent)

    dist.barrier(TP_GROUP)
    tl_out, tl_t = perf_fn(tilelang_gemm_rs, warmup, repeat)
    print(f"rank {RANK} tilelang GEMM avg time: {tl_t} ms")

    # Check correctness
    assert torch.allclose(tl_out, torch_out, atol=1e-2, rtol=1e-2), f"max error: {(tl_out - torch_out).abs().max()}"
    print(f"rank {RANK} check passed.✅")

    dist.destroy_process_group()
