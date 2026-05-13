"""BF16 GEMM + NVSwitch multimem allreduce.

This is the stable TileLang version of the ThunderKittens GEMM-AR data flow:
write each rank's GEMM partial into multicast-backed C, then allreduce C with
multimem.ld_reduce + multimem.st. The allreduce is a separate kernel for now;
that keeps the user-facing code simple and reuses the same robust primitive as
the standalone allreduce example.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.distributed import do_bench, init_dist
from tilelang.utils.allocator import get_allocator

os.environ.setdefault("NCCL_DEBUG", "ERROR")
tilelang.enable_cache()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def gemm_partial_kernel(
    M,
    N,
    K,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    pipeline_stages: int,
    dtype=T.bfloat16,
):
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=pipeline_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def multimem_allreduce_one_shot_kernel(total_elems, block_E, threads, dtype=T.bfloat16):
    @T.prim_func
    def main(
        mcast_buf: T.Tensor((total_elems,), dtype),
        result: T.Tensor((total_elems,), dtype),
    ):
        with T.Kernel(T.ceildiv(total_elems, block_E), threads=threads) as bx:
            offset = bx * block_E
            acc = T.alloc_fragment((block_E,), dtype)
            T.multimem_ld_reduce(
                mcast_buf[offset : offset + block_E],
                acc,
                reduce_op=T.MultimemReduceOp.ADD,
            )
            T.copy(acc, result[offset : offset + block_E])

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
    compile_once=True,
)
def gemm_ar_sm_specialized_kernel(
    M,
    N,
    K,
    num_ranks,
    num_comm_sms: int,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    group_size_m: int,
    pipeline_stages: int,
    ar_block_e: int,
    dtype=T.bfloat16,
):
    sm_num = driver.get_num_sms()
    num_comp_sms = sm_num - num_comm_sms
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    k_blocks = T.ceildiv(K, block_K)
    total_tiles = m_blocks * n_blocks
    waves = T.ceildiv(total_tiles, num_comp_sms)
    ar_rows = ar_block_e // block_N
    ar_row_chunks = block_M // ar_rows
    store_block_M = block_M // 2
    accum_dtype = T.float32

    def tile_coords(tile_id):
        super_rows = (m_blocks // group_size_m) * group_size_m
        final_rows = m_blocks - super_rows
        final_rows_safe = T.max(final_rows, 1)
        super_tiles = group_size_m * n_blocks
        is_super_tile = tile_id < super_rows * n_blocks
        remainder_id = tile_id - super_rows * n_blocks
        by = T.if_then_else(
            is_super_tile,
            group_size_m * (tile_id // super_tiles) + tile_id % group_size_m,
            super_rows + remainder_id % final_rows_safe,
        )
        bx = T.if_then_else(
            is_super_tile,
            (tile_id % super_tiles) // group_size_m,
            remainder_id // final_rows_safe,
        )
        return by, bx

    @T.macro
    def materialized_tile_coords(tile_id):
        by_expr, bx_expr = tile_coords(tile_id)
        by = T.alloc_var(T.int32)
        bx = T.alloc_var(T.int32)
        by = by_expr
        bx = bx_expr
        return by, bx

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C_local_out: T.Tensor((M, N), dtype),
        mcast_C: T.Tensor((M, N), dtype),
        mcast_signal: T.Tensor((total_tiles,), T.uint32),
        local_signal: T.Tensor((total_tiles,), T.uint32),
    ):
        with T.Kernel(sm_num, threads=threads) as bid:
            local_rank = T.get_rank()
            if bid < num_comp_sms:
                with T.sm_specialize_scope(auto_ws=True):
                    A_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_shared = T.alloc_shared((block_K, block_N), dtype)
                    C_shared = T.alloc_shared((store_block_M, block_N), dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    for w in T.serial(waves):
                        tile_id = bid + w * num_comp_sms
                        if tile_id < total_tiles:
                            by, bx = materialized_tile_coords(tile_id)
                            T.clear(C_local)
                            for k in T.Pipelined(k_blocks, num_stages=pipeline_stages):
                                T.copy(A[by * block_M, k * block_K], A_shared)
                                T.copy(B[k * block_K, bx * block_N], B_shared)
                                T.gemm(A_shared, B_shared, C_local)
                            # Match TK's store path: consumers materialize C in
                            # shared memory, then use TMA stores before
                            # signaling the communicator. Store half a tile at
                            # a time to keep shared memory under the H100 limit.
                            for row_offset in T.serial(0, block_M, store_block_M):
                                T.copy(C_local[row_offset, 0], C_shared)
                                T.sync_threads()
                                T.copy(C_shared, C_local_out[by * block_M + row_offset, bx * block_N])
                                T.sync_threads()
                            T.fence_sys()
                            if T.get_thread_binding(0) == 0:
                                T.multimem_signal_add(mcast_signal[tile_id], 1)
            else:
                with T.sm_specialize_scope(auto_ws=False):
                    comm_sm_id = bid - num_comp_sms
                    acc = T.alloc_fragment((ar_rows, block_N), dtype)
                    for task_iter in T.serial(T.ceildiv(total_tiles, num_comm_sms * num_ranks)):
                        tile_id = (task_iter * num_comm_sms + comm_sm_id) * num_ranks + local_rank
                        if tile_id < total_tiles:
                            by, bx = materialized_tile_coords(tile_id)
                            if T.get_thread_binding(0) == 0:
                                T.wait_ge(
                                    local_signal[tile_id],
                                    num_ranks,
                                    scope=T.WaitScope.SYS,
                                    semantics=T.WaitSemantics.ACQUIRE,
                                )
                            T.sync_threads()
                            for row_chunk in T.serial(ar_row_chunks):
                                row = row_chunk * ar_rows
                                T.multimem_ld_reduce(
                                    mcast_C[
                                        by * block_M + row : by * block_M + row + ar_rows,
                                        bx * block_N : (bx + 1) * block_N,
                                    ],
                                    acc,
                                    reduce_op=T.MultimemReduceOp.ADD,
                                )
                                T.multimem_st(
                                    acc,
                                    mcast_C[
                                        by * block_M + row : by * block_M + row + ar_rows,
                                        bx * block_N : (bx + 1) * block_N,
                                    ],
                                )

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def reset_gemm_ar_specialized_state_kernel(num_tiles, threads=256):
    @T.prim_func
    def main(local_signal: T.Tensor((num_tiles,), T.uint32)):
        with T.Kernel(T.ceildiv(num_tiles, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            idx = bx * threads + tid
            if idx < num_tiles:
                local_signal[idx] = 0

    return main


def torch_gemm_ar(group: torch.distributed.ProcessGroup, A: torch.Tensor, B: torch.Tensor):
    C = torch.matmul(A, B)
    dist.all_reduce(C, op=dist.ReduceOp.SUM, group=group)
    return C


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.bfloat16
    M, N, K = args.M, args.N, args.K
    block_M, block_N, block_K = args.block_m, args.block_n, args.block_k

    assert M % block_M == 0, "M must be divisible by block-m"
    assert N % block_N == 0, "N must be divisible by block-n"
    assert K % block_K == 0, "K must be divisible by block-k"
    assert M * N % args.ar_block_e == 0, "M*N must be divisible by ar-block-e"
    assert args.ar_block_e % 512 == 0, "ar-block-e must be a multiple of 512 for bf16x2 multimem vectorization"
    assert args.ar_block_e % block_N == 0, "ar-block-e must contain an integer number of tile rows"
    assert block_M % (args.ar_block_e // block_N) == 0, "block-m must be divisible by ar-block-e/block-n"
    assert args.two_kernel or 0 < args.num_comm_sms < driver.get_num_sms(), (
        "num-comm-sms must leave at least one compute SM"
    )
    assert block_M * block_N % args.ar_block_e == 0, "block_m*block_n must be divisible by ar-block-e"

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks, "only support single-node launch for now"

    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    num_tiles = (M // block_M) * (N // block_N)
    signal_bytes = torch.empty((), dtype=torch.uint32).element_size()
    allocator = get_allocator(
        size=max(M * K * dtype_bytes + K * N * dtype_bytes + M * N * dtype_bytes * 4, 2**28),
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
        mcast_size=M * N * dtype_bytes + num_tiles * signal_bytes + 4096,
    )

    if args.two_kernel:
        gemm_kernel = gemm_partial_kernel(
            M,
            N,
            K,
            block_M,
            block_N,
            block_K,
            args.threads,
            args.pipeline_stages,
        )
        ar_kernel = multimem_allreduce_one_shot_kernel(M * N, args.ar_block_e, args.threads)
        gemm_kernel.compile_group = group
        ar_kernel.compile_group = group
        gemm_kernel.initialize(allocator=allocator)
        ar_kernel.initialize(allocator=allocator)
        reset_kernel = None
    else:
        gemm_kernel = gemm_ar_sm_specialized_kernel(
            M,
            N,
            K,
            num_local_ranks,
            args.num_comm_sms,
            block_M,
            block_N,
            block_K,
            args.threads,
            args.group_size_m,
            args.pipeline_stages,
            args.ar_block_e,
        )
        reset_kernel = reset_gemm_ar_specialized_state_kernel(num_tiles, args.threads)
        gemm_kernel.compile_group = group
        reset_kernel.compile_group = group
        gemm_kernel.initialize(allocator=allocator)
        reset_kernel.initialize(allocator=allocator)
        ar_kernel = None

    if local_rank == 0 and args.print_source:
        print(gemm_kernel.get_kernel_source())
        if ar_kernel is not None:
            print(ar_kernel.get_kernel_source())
        if reset_kernel is not None:
            print(reset_kernel.get_kernel_source())

    torch.manual_seed(42 + local_rank)
    A = tilelang.tensor((M, K), dtype, allocator=allocator).normal_() / (K**0.25)
    B = tilelang.tensor((K, N), dtype, allocator=allocator).normal_() / (K**0.25)
    mcast_C_flat, local_C_flat = allocator._allocate_mcast_tensor((M * N,), dtype)
    local_C = local_C_flat.view(M, N)
    mcast_signal, local_signal = allocator._allocate_mcast_tensor((num_tiles,), torch.uint32)
    if args.two_kernel:
        output_flat = tilelang.tensor((M * N,), dtype, allocator=allocator)
        output = output_flat.view(M, N)
    else:
        output_flat = local_C_flat
        output = local_C

    torch.cuda.synchronize()
    dist.barrier(group)

    def reset_output():
        # GEMM and allreduce overwrite the full output tile space. Keep reset
        # aligned with TK's epilogue: only restore the inter-SM signal state.
        if not args.two_kernel:
            reset_kernel(local_signal)
        torch.cuda.synchronize()
        dist.barrier(group)

    def run_kernel():
        if args.two_kernel:
            gemm_kernel(A, B, local_C)
            torch.cuda.synchronize()
            dist.barrier(group)
            ar_kernel(mcast_C_flat, output_flat)
        else:
            mcast_C = mcast_C_flat.view(M, N)
            gemm_kernel(A, B, local_C, mcast_C, mcast_signal, local_signal)
        return output

    reset_output()
    tilelang_C = run_kernel()
    torch.cuda.synchronize()
    dist.barrier(group)

    if args.check:
        torch_C = torch_gemm_ar(group, A, B)
        torch.cuda.synchronize()
        max_diff = (torch_C.float() - tilelang_C.float()).abs().max().item()
        passed = torch.allclose(torch_C, tilelang_C, atol=args.atol, rtol=args.rtol)
        print(f"rank {local_rank} check {'passed' if passed else 'failed'}. max_diff={max_diff}")
        dist.barrier(group)

    reset_output()
    if args.include_reset_in_bench:
        def bench_kernel():
            run_kernel()
            # Match TK's timed entrypoint: launch the main kernel followed by
            # the device-side epilogue/reset kernel, without adding host
            # synchronize or process-group barriers inside the measured body.
            if reset_kernel is not None:
                reset_kernel(local_signal)

        tl_t = do_bench(
            bench_kernel,
            warmup=args.warmup,
            rep=args.rep,
            group=group,
        )
    else:
        tl_t = do_bench(
            run_kernel,
            warmup=args.warmup,
            rep=args.rep,
            post_fn=reset_output,
            group=group,
        )
    if local_rank == 0:
        tflops = 2 * M * N * K / 1e9 / tl_t
        print(f"tilelang gemm_ar time: {tl_t:.3f} ms, TFLOPS/GPU: {tflops:.2f}")

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--pipeline-stages", type=int, default=4)
    parser.add_argument("--ar-block-e", type=int, default=4096)
    parser.add_argument("--num-comm-sms", type=int, default=16)
    parser.add_argument("--group-size-m", type=int, default=12)
    parser.add_argument("--two-kernel", action="store_true")
    parser.add_argument("--include-reset-in-bench", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
