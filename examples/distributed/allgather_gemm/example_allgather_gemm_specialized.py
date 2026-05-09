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

@tilelang.jit(compile_once=True)
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
    comm_tile_elems = comm_block_M * comm_block_K
    comm_threads = threads + 128
    local_comm_m_blocks = T.ceildiv(M_per_rank, comm_block_M)
    comm_k_blocks = T.ceildiv(K, comm_block_K)
    total_tiles = m_blocks * n_blocks
    waves = T.ceildiv(total_tiles, num_comp_sms)
    GROUP_SIZE_M = 8
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A_local: T.Tensor((M_per_rank, K), dtype),
        B: T.Tensor((K, N_per_rank), dtype),
        mcast_A: T.Tensor((M, K), dtype),
        gathered_A: T.Tensor((M, K), dtype),
        mcast_signal: T.Tensor((m_blocks,), T.uint32),
        local_signal: T.Tensor((m_blocks,), T.uint32),
        barriers: T.Tensor((2, num_ranks), T.int32),
        C: T.Tensor((M, N_per_rank), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as bid:
            local_rank = T.get_rank()
            tid = T.get_thread_binding(0)
            T.barrier_blocks(barriers[0, 0])
            if bid < num_comp_sms:
                with T.sm_specialize_scope(auto_ws=True):
                    A_comp_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_comp_shared = T.alloc_shared((block_K, block_N), dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    for w in T.serial(waves):
                        tile_id = bid + w * num_comp_sms
                        if tile_id < total_tiles:
                            num_pid_in_group = GROUP_SIZE_M * n_blocks
                            group_id = tile_id // num_pid_in_group
                            first_pid_m = group_id * GROUP_SIZE_M
                            group_size_m = T.min(m_blocks - first_pid_m, GROUP_SIZE_M)
                            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                            pid_n = (tile_id % num_pid_in_group) // group_size_m
                            T.wait_eq(local_signal[pid_m], 1)

                            T.clear(C_local)
                            for k in T.Pipelined(k_blocks, num_stages=3):
                                T.copy(
                                    gathered_A[
                                        pid_m * block_M : (pid_m + 1) * block_M,
                                        k * block_K : (k + 1) * block_K,
                                    ],
                                    A_comp_shared,
                                )
                                T.copy(
                                    B[
                                        k * block_K : (k + 1) * block_K,
                                        pid_n * block_N : (pid_n + 1) * block_N,
                                    ],
                                    B_comp_shared,
                                )
                                T.gemm(A_comp_shared, B_comp_shared, C_local)
                            T.copy(
                                C_local,
                                C[pid_m * block_M : (pid_m + 1) * block_M, pid_n * block_N : (pid_n + 1) * block_N],
                            )
            else:
                with T.sm_specialize_scope(auto_ws=False):
                    comm_sm_id = bid - num_comp_sms
                    for local_m in T.serial(T.ceildiv(local_comm_m_blocks, num_comm_sms)):
                        local_comm_pid_m = comm_sm_id + local_m * num_comm_sms
                        if local_comm_pid_m < local_comm_m_blocks:
                            global_comm_pid_m = local_rank * local_comm_m_blocks + local_comm_pid_m
                            local_m_start = local_comm_pid_m * comm_block_M
                            global_m_start = global_comm_pid_m * comm_block_M
                            for k in T.serial(comm_k_blocks):
                                for elem in T.serial(T.ceildiv(comm_tile_elems, comm_threads)):
                                    offset = elem * comm_threads + tid
                                    if offset < comm_tile_elems:
                                        mi = offset // comm_block_K
                                        ki = offset % comm_block_K
                                        mcast_A[global_m_start + mi, k * comm_block_K + ki] = A_local[
                                            local_m_start + mi, k * comm_block_K + ki
                                        ]

                            T.sync_threads()
                            T.fence_sys()
                            if tid == 0:
                                T.multimem_signal(mcast_signal[global_comm_pid_m * 2], 1)
                                if global_comm_pid_m * 2 + 1 < m_blocks:
                                    T.multimem_signal(mcast_signal[global_comm_pid_m * 2 + 1], 1)

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def reset_ag_gemm_specialized_state_kernel(m_blocks, num_ranks, threads=256):
    @T.prim_func
    def main(
        local_signal: T.Tensor((m_blocks,), T.uint32),
        barriers: T.Tensor((2, num_ranks), T.int32),
        do_barrier: T.bool,
    ):
        with T.Kernel(1, threads=threads):
            tid = T.get_thread_binding(0)
            if do_barrier:
                T.sync_blocks(barriers[1, 0])
            for i in T.serial(T.ceildiv(m_blocks, threads)):
                signal_idx = i * threads + tid
                if signal_idx < m_blocks:
                    local_signal[signal_idx] = 0
            if tid < num_ranks:
                barriers[0, tid] = 0
            if do_barrier:
                T.fence_sys()
                T.barrier_blocks(barriers[1, 0])

    return main


def ag_gemm_op(
    A,
    B,
    mcast_A,
    gathered_A,
    mcast_signal,
    local_signal,
    barriers,
    C,
    kernel,
    reset_kernel,
):
    kernel(A, B, mcast_A, gathered_A, mcast_signal, local_signal, barriers, C)
    reset_kernel(local_signal, barriers, True)
    return C


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
    assert (N // num_local_ranks) % block_N == 0, "N_per_rank must be divisible by block_n"
    assert K % block_K == 0, "K must be divisible by block_k"
    assert 0 < num_comm_sms < driver.get_num_sms(), "num_comm_sms must leave at least one compute SM"

    M_per_rank = M // num_local_ranks
    N_per_rank = N // num_local_ranks
    m_blocks = M // block_M

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks, "only support single-node launch for now"

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    signal_bytes = torch.tensor([], dtype=torch.uint32).element_size()
    # _allocate_mcast_tensor uses aligned bump allocation; keep room for padding
    # between the gathered A buffer and the signal buffer.
    mcast_bytes = M * K * dtype_bytes + m_blocks * signal_bytes + 4096
    allocator = get_allocator(
        size=2**30,
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
    )
    reset_kernel = reset_ag_gemm_specialized_state_kernel(m_blocks, num_local_ranks, threads)
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
    mcast_signal, local_signal = allocator._allocate_mcast_tensor((m_blocks,), torch.uint32)
    mcast_A = mcast_A_flat.view(M, K)
    gathered_A = gathered_A_flat.view(M, K)
    dist.barrier(group)
    reset_kernel(local_signal, barriers, False)
    torch.cuda.synchronize()
    dist.barrier(group)
    tilelang_C = ag_gemm_op(
        A,
        B,
        mcast_A,
        gathered_A,
        mcast_signal,
        local_signal,
        barriers,
        C,
        kernel,
        reset_kernel,
    )
    torch.cuda.synchronize()
    dist.barrier(group)

    torch_ag_buffer = torch.empty((M, K), dtype=dtype, device=f"cuda:{local_rank}")
    torch_C = torch_ag_gemm(group, A, B, torch_ag_buffer)

    if torch.allclose(torch_C, tilelang_C, atol=1e-2, rtol=1e-2):
        print(f"rank {local_rank} check passed.")
    else:
        max_diff = (torch_C - tilelang_C).abs().max().item()
        ag_max_diff = (torch_ag_buffer - gathered_A).abs().max().item()
        print(f"rank {local_rank} check failed. max_diff={max_diff}, ag_max_diff={ag_max_diff}")

    tl_t = do_bench(
        lambda: ag_gemm_op(
            A,
            B,
            mcast_A,
            gathered_A,
            mcast_signal,
            local_signal,
            barriers,
            C,
            kernel,
            reset_kernel,
        ),
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
