import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.distributed import init_dist, do_bench

os.environ.setdefault("NCCL_DEBUG", "ERROR")


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
    compile_once=True,
)
def gemm_rs_specialized_kernel(
    M,
    N,
    K,
    num_ranks,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    group_size_m: int,
    pipeline_stages: int,
    use_tma_epilogue: bool,
    dtype=T.bfloat16,
):
    sm_num = driver.get_num_sms()
    K_per_rank = K // num_ranks
    M_per_rank = M // num_ranks
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    k_blocks = T.ceildiv(K_per_rank, block_K)
    blocks_per_rank = m_blocks // num_ranks
    total_tiles = m_blocks * n_blocks
    waves = T.ceildiv(total_tiles, sm_num)
    accum_dtype = T.float32

    store_block_M = block_M // 2   # avoid SMEM spill when #stages = 4

    def tile_coords(tile_id, local_rank):
        rotation = ((local_rank + 1) * blocks_per_rank * n_blocks) % total_tiles
        rotated = (tile_id + rotation) % total_tiles
        super_rows = (m_blocks // group_size_m) * group_size_m
        final_rows = m_blocks - super_rows
        final_rows_safe = T.max(final_rows, 1)
        super_tiles = group_size_m * n_blocks
        is_super_tile = rotated < super_rows * n_blocks
        remainder_id = rotated - super_rows * n_blocks

        by = T.if_then_else(
            is_super_tile,
            group_size_m * (rotated // super_tiles) + rotated % group_size_m,
            super_rows + remainder_id % final_rows_safe,
        )
        bx = T.if_then_else(
            is_super_tile,
            (rotated % super_tiles) // group_size_m,
            remainder_id // final_rows_safe,
        )
        return by, bx

    @T.macro
    def materialized_tile_coords(tile_id, local_rank):
        by_expr, bx_expr = tile_coords(tile_id, local_rank)
        by = T.alloc_var(T.int32)
        bx = T.alloc_var(T.int32)
        by = by_expr
        bx = bx_expr
        return by, bx
        
    @T.prim_func
    def main(
        A: T.Tensor((M, K_per_rank), dtype),
        B: T.Tensor((K_per_rank, N), dtype),
        C: T.Tensor((M_per_rank, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as bid:
            local_rank = T.get_rank()
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((store_block_M, block_N), dtype)

            for w in T.serial(waves):
                tile_id = bid + w * sm_num
                if tile_id < total_tiles:
                    by, bx = materialized_tile_coords(tile_id, local_rank)
                    dst_rank = by // blocks_per_rank
                    local_by = by - dst_rank * blocks_per_rank

                    T.clear(C_local)
                    for k in T.Pipelined(k_blocks, num_stages=pipeline_stages):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)

                    # Fused reduce-scatter
                    for row_offset in T.serial(0, block_M, store_block_M):
                        T.copy(C_local[row_offset, 0], C_shared)
                        if use_tma_epilogue:
                            # TMA store-add into peer C.
                            T.atomic_add(
                                C[
                                    local_by * block_M + row_offset : local_by * block_M + row_offset + store_block_M,
                                    bx * block_N : (bx + 1) * block_N,
                                ],
                                C_shared,
                                use_tma=True,
                                dst_pe=dst_rank,
                            )
                        else:
                            # vectorized bf16x2 red.global.add.
                            tid = T.get_thread_binding()
                            for idx in T.serial(tid, store_block_M * block_N // 2, threads):
                                elem = idx * 2
                                T.atomic_addx2(
                                    C[local_by * block_M + row_offset + elem // block_N, bx * block_N + elem % block_N],
                                    C_shared[elem // block_N, elem % block_N],
                                    dst_pe=dst_rank,
                                )

    return main


def torch_gemm_rs(group: torch.distributed.ProcessGroup, A: torch.Tensor, B: torch.Tensor, num_ranks: int):
    partial = torch.matmul(A, B)
    output = torch.empty((A.shape[0] // num_ranks, B.shape[1]), dtype=partial.dtype, device=A.device)
    dist.reduce_scatter_tensor(output, partial, group=group)
    return output


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.bfloat16
    M, N, K = args.M, args.N, args.K
    block_M, block_N, block_K = args.block_m, args.block_n, args.block_k

    assert num_local_ranks == 8, "remote TMA descriptor selection is currently specialized for 8 local ranks"
    assert M % (num_local_ranks * block_M) == 0
    assert N % block_N == 0
    assert K % (num_local_ranks * block_K) == 0

    M_per_rank = M // num_local_ranks
    K_per_rank = K // num_local_ranks

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks, "only support single-node launch for now"

    allocator = tilelang.get_allocator(
        size=2**35,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
    )

    kernel = gemm_rs_specialized_kernel(
        M,
        N,
        K,
        num_local_ranks,
        block_M,
        block_N,
        block_K,
        args.threads,
        args.group_size_m,
        args.pipeline_stages,
        args.tma_epilogue,
    )
    kernel.initialize(allocator=allocator)

    if local_rank == 0 and args.print_source:
        print(kernel.get_kernel_source())

    torch.manual_seed(42 + local_rank)
    A = tilelang.tensor((M, K_per_rank), dtype, allocator=allocator).normal_() / (K_per_rank**0.25)
    B = tilelang.tensor((K_per_rank, N), dtype, allocator=allocator).normal_() / (K_per_rank**0.25)
    C_local = tilelang.tensor((M_per_rank, N), dtype, allocator=allocator)
    C_local.zero_()
    dist.barrier(group)

    def reset_output():
        C_local.zero_()
        torch.cuda.synchronize()
        dist.barrier(group)

    def run_kernel():
        kernel(A, B, C_local)

    reset_output()
    run_kernel()
    torch.cuda.synchronize()
    dist.barrier(group)

    if args.check:
        torch_out = torch_gemm_rs(group, A, B, num_local_ranks)
        torch.cuda.synchronize()
        max_diff = (torch_out.float() - C_local.float()).abs().max().item()
        passed = torch.allclose(torch_out, C_local, atol=args.atol, rtol=args.rtol)
        print(f"rank {local_rank} check {'passed' if passed else 'failed'}. max_diff={max_diff}")
        dist.barrier(group)

    for _ in range(args.warmup):
        reset_output()
        run_kernel()
    reset_output()
    tl_t = do_bench(
        run_kernel,
        warmup=0,
        rep=args.rep,
        post_fn=reset_output,
        group=group,
    )

    if local_rank == 0:
        tflops = 2 * M * N * K_per_rank / 1e9 / tl_t
        print(f"tilelang specialized gemm_rs time: {tl_t:.3f} ms, TFLOPS/GPU: {tflops:.2f}")

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--M", type=int, default=32768)
    parser.add_argument("--N", type=int, default=6144)
    parser.add_argument("--K", type=int, default=16384, help="Total K across ranks")
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--group-size-m", type=int, default=12)
    parser.add_argument("--pipeline-stages", type=int, default=4)
    epilogue = parser.add_mutually_exclusive_group()
    epilogue.add_argument(
        "--tma-epilogue",
        dest="tma_epilogue",
        action="store_true",
        help="Use TMA store-add epilogue (default)",
    )
    parser.set_defaults(tma_epilogue=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes)
