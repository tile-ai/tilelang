"""Allgather GEMM intranode example (SM-specialized version)"""

import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.distributed import init_dist, do_bench
from tilelang.utils.allocator import get_allocator

os.environ.setdefault("NCCL_DEBUG", "ERROR")
tilelang.enable_cache()

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
    compile_once=True,
)
def ag_gemm_sm_specialized_kernel(
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
    use_tma_store: bool,
    dtype=T.bfloat16,
):
    sm_num = driver.get_num_sms()
    num_comp_sms = sm_num - num_comm_sms
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N_per_rank, block_N)
    local_m_blocks = T.ceildiv(M_per_rank, block_M)
    k_blocks = T.ceildiv(K, block_K)
    comm_block_M = block_M * 2
    comm_block_K = block_K * 2
    store_block_M = block_M // 2
    comm_chunks = 2
    local_comm_m_blocks = T.ceildiv(M_per_rank, comm_block_M)
    comm_m_blocks = T.ceildiv(M, comm_block_M)
    comm_k_blocks = T.ceildiv(K, comm_block_K)
    comm_tasks_per_rank = local_comm_m_blocks * comm_k_blocks
    total_tiles = m_blocks * n_blocks
    waves = T.ceildiv(total_tiles, num_comp_sms)
    accum_dtype = T.float32
    comm_workers_per_signal = T.min(num_comm_sms * comm_chunks, comm_k_blocks)
    local_tiles = local_m_blocks * n_blocks

    def tile_coords(tile_id, local_rank):
        """Local-first tile scheduler.

        Local tiles use Super-M grouping for B-tile reuse. Remote tiles then
        walk peer ranks round-robin inside each remote row shard.
        """
        is_local = tile_id < local_tiles
        local_super_rows = (local_m_blocks // group_size_m) * group_size_m
        final_rows = local_m_blocks - local_super_rows
        final_rows_safe = T.max(final_rows, 1)
        super_tiles = group_size_m * n_blocks

        is_super_tile = tile_id < local_super_rows * n_blocks
        local_remainder_id = tile_id - local_super_rows * n_blocks
        local_by = T.if_then_else(
            is_super_tile,
            group_size_m * (tile_id // super_tiles) + tile_id % group_size_m,
            local_super_rows + local_remainder_id % final_rows_safe,
        )
        local_bx = T.if_then_else(
            is_super_tile,
            (tile_id % super_tiles) // group_size_m,
            local_remainder_id // final_rows_safe,
        )

        remote_tile_id = tile_id - local_tiles
        target_shard = remote_tile_id // ((num_ranks - 1) * n_blocks)
        idx_in_shard = remote_tile_id % ((num_ranks - 1) * n_blocks)
        peer_rank_offset = idx_in_shard % (num_ranks - 1)
        peer_rank = peer_rank_offset + T.if_then_else(peer_rank_offset >= local_rank, 1, 0)
        remote_by = peer_rank * local_m_blocks + target_shard
        remote_bx = idx_in_shard // (num_ranks - 1)

        by = T.if_then_else(is_local, local_rank * local_m_blocks + local_by, remote_by)
        bx = T.if_then_else(is_local, local_bx, remote_bx)
        return is_local, by, bx, by - local_rank * local_m_blocks

    @T.macro
    def materialized_tile_coords(tile_id, local_rank):
        _, by_expr, bx_expr, local_by_expr = tile_coords(tile_id, local_rank)
        by = T.alloc_var(T.int32)
        bx = T.alloc_var(T.int32)
        local_by = T.alloc_var(T.int32)
        by = by_expr
        bx = bx_expr
        local_by = local_by_expr
        return by, bx, local_by

    @T.macro
    def materialized_global_tile_coords(tile_id, local_rank):
        _, by_expr, bx_expr, _ = tile_coords(tile_id, local_rank)
        by = T.alloc_var(T.int32)
        bx = T.alloc_var(T.int32)
        by = by_expr
        bx = bx_expr
        return by, bx

    @T.prim_func
    def main(
        A_local: T.Tensor((M_per_rank, K), dtype),
        B: T.Tensor((K, N_per_rank), dtype),
        mcast_A: T.Tensor((M, K), dtype),
        gathered_A: T.Tensor((M, K), dtype),
        mcast_signal: T.Tensor((comm_m_blocks,), T.uint32),
        local_signal: T.Tensor((comm_m_blocks,), T.uint32),
        barriers: T.Tensor((2, num_ranks), T.int32),
        C: T.Tensor((M, N_per_rank), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as bid:
            local_rank = T.get_rank()
            T.barrier_blocks(barriers[0, 0])
            if bid < num_comp_sms:
                with T.sm_specialize_scope(auto_ws=True):
                    A_comp_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_comp_shared = T.alloc_shared((block_K, block_N), dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    if use_tma_store:
                        # Stage one consumer warpgroup's rows at a time. A
                        # 128x256 TMA epilogue would exceed Hopper SMEM budget
                        # after auto warp-specialized input staging.
                        C_shared = T.alloc_shared((store_block_M, block_N), dtype)
                        C_local_cast = T.alloc_fragment((block_M, block_N), dtype)
                    for w in T.serial(waves):
                        tile_id = bid + w * num_comp_sms
                        if tile_id < total_tiles:
                            is_local_by, by_expr, _, _ = tile_coords(tile_id, local_rank)
                            load_by, load_bx, load_local_by = materialized_tile_coords(
                                tile_id, local_rank
                            )
                            if not is_local_by:
                                T.wait_ge(local_signal[by_expr // 2], comm_workers_per_signal)

                            T.clear(C_local)
                            for k in T.Pipelined(k_blocks, num_stages=pipeline_stages):
                                if is_local_by:
                                    T.copy(A_local[load_local_by * block_M, k * block_K,], A_comp_shared)
                                else:
                                    T.copy(
                                        gathered_A[load_by * block_M, k * block_K,], A_comp_shared)
                                T.copy(
                                    B[k * block_K, load_bx * block_N,], B_comp_shared)
                                T.gemm(A_comp_shared, B_comp_shared, C_local)
                            store_by, store_bx = materialized_global_tile_coords(tile_id, local_rank)
                            if use_tma_store:
                                T.copy(C_local, C_local_cast)
                                for row_offset in T.serial(0, block_M, store_block_M):
                                    for i, j in T.Parallel(store_block_M, block_N):
                                        C_shared[i, j] = C_local_cast[row_offset + i, j]
                                    T.sync_threads(0, threads)
                                    T.copy(
                                        C_shared[:, :],
                                        C[
                                            store_by * block_M + row_offset : store_by * block_M + row_offset + store_block_M,
                                            store_bx * block_N : (store_bx + 1) * block_N,
                                        ],
                                    )
                                    T.sync_threads(0, threads)
                            else:
                                T.copy(C_local, C[store_by * block_M, store_bx * block_N,])
            else:
                with T.sm_specialize_scope(auto_ws=False):
                    A_comm_shared = T.alloc_shared((comm_chunks, comm_block_M, comm_block_K), dtype)
                    A_comm_ready = T.alloc_barrier([1] * comm_chunks)
                    comm_sm_id = bid - num_comp_sms
                    warp_id = T.get_warp_idx_sync()
                    lane_id = T.get_lane_idx()
                    comm_chunk = warp_id % comm_chunks
                    if warp_id < comm_chunks:
                        # Each participating warp owns one comm chunk: it issues
                        # an independent TMA load/store pair into its own shared
                        # buffer and mbarrier.
                        for local_task_iter in T.serial(T.ceildiv(comm_tasks_per_rank, num_comm_sms * comm_chunks)):
                            local_task_id = (
                                (local_task_iter * num_comm_sms + comm_sm_id) * comm_chunks + comm_chunk
                            )
                            if local_task_id < comm_tasks_per_rank:
                                local_comm_by = local_task_id // comm_k_blocks
                                k = local_task_id - local_comm_by * comm_k_blocks
                                global_comm_by = local_rank * local_comm_m_blocks + local_comm_by
                                local_m_start = local_comm_by * comm_block_M
                                global_m_start = global_comm_by * comm_block_M
                                T.tma_copy(
                                    A_local[
                                        local_m_start : local_m_start + comm_block_M,
                                        k * comm_block_K : (k + 1) * comm_block_K,
                                    ],
                                    A_comm_shared[comm_chunk, :, :],
                                    barrier=A_comm_ready[comm_chunk],
                                    leader_thread_extent=32,
                                )
                                if lane_id == 0:
                                    T.barrier_arrive(A_comm_ready[comm_chunk])
                                T.mbarrier_wait_parity(A_comm_ready[comm_chunk], local_task_iter & 1)
                                T.tma_copy(
                                    A_comm_shared[comm_chunk, :, :],
                                    mcast_A[
                                        global_m_start : global_m_start + comm_block_M,
                                        k * comm_block_K : (k + 1) * comm_block_K,
                                    ],
                                    leader_thread_extent=32,
                                )
                                T.tma_store_wait(0, False)
                                if k + num_comm_sms * comm_chunks >= comm_k_blocks:
                                    if lane_id == 0:
                                        T.multimem_signal_add(mcast_signal[global_comm_by], 1)

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def reset_ag_gemm_specialized_state_kernel(signal_blocks, num_ranks, threads=256):
    @T.prim_func
    def main(
        local_signal: T.Tensor((signal_blocks,), T.uint32),
        barriers: T.Tensor((2, num_ranks), T.int32),
        do_barrier: T.bool,
    ):
        with T.Kernel(1, threads=threads):
            tid = T.get_thread_binding(0)
            if do_barrier:
                T.sync_blocks(barriers[1, 0])
            for i in T.serial(T.ceildiv(signal_blocks, threads)):
                signal_idx = i * threads + tid
                if signal_idx < signal_blocks:
                    local_signal[signal_idx] = 0
            if tid < num_ranks:
                barriers[0, tid] = 0
            if do_barrier:
                T.fence_sys()
                T.barrier_blocks(barriers[1, 0])

    return main


def torch_ag_gemm(group: torch.distributed.ProcessGroup, A: torch.Tensor, B: torch.Tensor, ag_out: torch.Tensor):
    torch.distributed.all_gather_into_tensor(ag_out, A, group)
    return torch.matmul(ag_out, B)


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.bfloat16
    M, N, K = args.M, args.N, args.K
    block_M, block_N, block_K = args.block_m, args.block_n, args.block_k
    threads = args.threads
    num_comm_sms = args.num_comm_sms

    assert M % num_local_ranks == 0, "M must be divisible by num-processes"
    assert N % num_local_ranks == 0, "N must be divisible by num-processes"
    assert (M // num_local_ranks) % block_M == 0, "M_per_rank must be divisible by block_m"
    assert (M // num_local_ranks) % (2 * block_M) == 0, "M_per_rank must be divisible by 2 * block_m"
    assert (N // num_local_ranks) % block_N == 0, "N_per_rank must be divisible by block_n"
    assert K % block_K == 0, "K must be divisible by block_k"
    assert 0 < num_comm_sms < driver.get_num_sms(), "num_comm_sms must leave at least one compute SM"

    M_per_rank = M // num_local_ranks
    N_per_rank = N // num_local_ranks
    m_blocks = M // block_M
    signal_blocks = (M + block_M * 2 - 1) // (block_M * 2)

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks, "only support single-node launch for now"

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    signal_bytes = torch.tensor([], dtype=torch.uint32).element_size()
    # _allocate_mcast_tensor uses aligned bump allocation; keep room for padding
    # between the gathered A buffer and the signal buffer.
    mcast_bytes = M * K * dtype_bytes + signal_blocks * signal_bytes + 4096
    regular_bytes = (
        M_per_rank * K * dtype_bytes
        + K * N_per_rank * dtype_bytes
        + M * N_per_rank * dtype_bytes
        + 2 * num_local_ranks * torch.tensor([], dtype=torch.int32).element_size()
    )
    allocator = get_allocator(
        size=max(2**30, regular_bytes + 2**28),
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
        use_vmm=True,
        mcast_size=mcast_bytes,
    )

    kernel = ag_gemm_sm_specialized_kernel(
        M,
        N,
        K,
        num_local_ranks,
        num_comm_sms,
        block_M,
        block_N,
        block_K,
        threads,
        args.group_size_m,
        args.pipeline_stages,
        args.use_tma_store,
    )
    reset_kernel = reset_ag_gemm_specialized_state_kernel(signal_blocks, num_local_ranks, threads)
    kernel.initialize(allocator=allocator)
    reset_kernel.initialize(allocator=allocator)
    if local_rank == 0 and args.print_source:
        print(kernel.get_kernel_source())
        print(reset_kernel.get_kernel_source())

    torch.manual_seed(42 + local_rank)
    A = tilelang.tensor((M_per_rank, K), dtype, allocator=allocator).normal_()
    B = tilelang.tensor((K, N_per_rank), dtype, allocator=allocator).normal_()
    C = tilelang.tensor((M, N_per_rank), dtype, allocator=allocator)
    barriers = tilelang.tensor((2, num_local_ranks), torch.int32, allocator=allocator).zero_()

    mcast_A_flat, gathered_A_flat = allocator._allocate_mcast_tensor((M * K,), dtype)
    mcast_signal, local_signal = allocator._allocate_mcast_tensor((signal_blocks,), torch.uint32)
    mcast_A = mcast_A_flat.view(M, K)
    gathered_A = gathered_A_flat.view(M, K)
    dist.barrier(group)
    reset_kernel(local_signal, barriers, False)
    torch.cuda.synchronize()
    dist.barrier(group)

    def ag_gemm_op():
        kernel(A, B, mcast_A, gathered_A, mcast_signal, local_signal, barriers, C)
        reset_kernel(local_signal, barriers, True)
        return C

    tilelang_C = ag_gemm_op()
    torch.cuda.synchronize()
    dist.barrier(group)

    torch_ag_buffer = torch.empty((M, K), dtype=dtype, device=f"cuda:{local_rank}")
    torch_C = torch_ag_gemm(group, A, B, torch_ag_buffer)

    if torch.allclose(torch_C, tilelang_C, atol=1e-2, rtol=1e-2):
        print(f"rank {local_rank} check passed. ✅")
    else:
        max_diff = (torch_C - tilelang_C).abs().max().item()
        ag_max_diff = (torch_ag_buffer - gathered_A).abs().max().item()
        print(f"rank {local_rank} check failed. ❌")
        print(f"max_diff={max_diff}, ag_max_diff={ag_max_diff}")

    tl_t = do_bench(
        ag_gemm_op,
        warmup=args.warmup,
        rep=args.rep,
        group=group,
    )
    if local_rank == 0:
        print(f"tilelang specialized ag_gemm time: {tl_t:.2f} ms, TFLOPS: {2 * M * N * K / 1e9 / tl_t / num_local_ranks:.2f}")

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--M", type=int, default=32768)
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--num-comm-sms", type=int, default=4)
    parser.add_argument("--group-size-m", type=int, default=12)
    parser.add_argument("--pipeline-stages", type=int, default=4)
    epilogue = parser.add_mutually_exclusive_group()
    epilogue.add_argument("--use-tma-store", dest="use_tma_store", action="store_true")
    epilogue.add_argument("--no-tma-store", dest="use_tma_store", action="store_false")
    parser.set_defaults(use_tma_store=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
