import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.distributed import init_dist
from tilelang.distributed import perf_fn
from tilelang.utils.allocator import get_allocator

tilelang.disable_cache()
os.environ["NCCL_DEBUG"] = "WARN"


@tilelang.jit
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
    dtype=T.float16,
):
    sm_num = driver.get_num_sms()
    num_comp_sms = sm_num - num_comm_sms
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N_per_rank, block_N)
    local_m_blocks = T.ceildiv(M_per_rank, block_M)
    k_blocks = T.ceildiv(K, block_K)
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
        grid_barrier: T.Tensor((num_ranks,), T.int32),
        C: T.Tensor((M, N_per_rank), dtype),
        local_rank: T.int32,
    ):
        with T.Kernel(sm_num, threads=threads) as bid:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            tid = T.get_thread_binding(0)

            if bid == 0:
                for i in T.serial(T.ceildiv(m_blocks, threads)):
                    signal_idx = i * threads + tid
                    if signal_idx < m_blocks:
                        local_signal[signal_idx] = 0
            T.fence_sys()
            T.barrier_blocks(grid_barrier)

            if bid < num_comp_sms:
                for w in T.serial(waves):
                    tile_id = bid + w * num_comp_sms
                    if tile_id < total_tiles:
                        num_pid_in_group = GROUP_SIZE_M * n_blocks
                        group_id = tile_id // num_pid_in_group
                        first_pid_m = group_id * GROUP_SIZE_M
                        group_size_m = T.min(m_blocks - first_pid_m, GROUP_SIZE_M)
                        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                        pid_n = (tile_id % num_pid_in_group) // group_size_m

                        if tid == 0:
                            T.wait_eq(local_signal[pid_m], 1)

                        T.clear(C_local)
                        for k in T.Pipelined(k_blocks, num_stages=3):
                            T.copy(gathered_A[pid_m * block_M, k * block_K], A_shared)
                            T.copy(B[k * block_K, pid_n * block_N], B_shared)
                            T.gemm(A_shared, B_shared, C_local)
                        T.copy(C_local, C_shared)
                        T.copy(C_shared, C[pid_m * block_M, pid_n * block_N])
            else:
                loaded = T.alloc_barrier([256])
                parity = 0
                comm_sm_id = bid - num_comp_sms
                for local_m in T.serial(T.ceildiv(local_m_blocks, num_comm_sms)):
                    local_pid_m = comm_sm_id + local_m * num_comm_sms
                    if local_pid_m < local_m_blocks:
                        global_pid_m = local_rank * local_m_blocks + local_pid_m
                        for k in T.serial(k_blocks):
                            T.tma_load(A_local[local_pid_m * block_M, k * block_K], A_shared)
                            T.mbarrier_arrive(loaded)
                            T.mbarrier_wait_parity(loaded, parity)
                            parity = (parity + 1) % 2
                            T.copy(
                                A_shared,
                                mcast_A[global_pid_m * block_M, k * block_K],
                            )  # TODO(wt): Change to canonical mcast tma store later

                        T.fence_sys()
                        if tid == 0:
                            T.multimem_signal(mcast_signal[global_pid_m], 1)

    return main


def ag_gemm_op(
    A,
    B,
    mcast_A,
    gathered_A,
    mcast_signal,
    local_signal,
    grid_barrier,
    C,
    kernel,
    local_rank,
):
    kernel(A, B, mcast_A, gathered_A, mcast_signal, local_signal, grid_barrier, C, local_rank)
    return C


def torch_ag_gemm(group: torch.distributed.ProcessGroup, A: torch.Tensor, B: torch.Tensor, ag_out: torch.Tensor):
    torch.distributed.all_gather_into_tensor(ag_out, A, group)
    return torch.matmul(ag_out, B)


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.float16
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
    kernel.initialize(allocator=allocator)
    if local_rank == 0 and args.print_source:
        print(kernel.get_kernel_source())

    torch.manual_seed(42 + local_rank)
    A = tilelang.tensor((M_per_rank, K), dtype, allocator=allocator).normal_()
    B = tilelang.tensor((K, N_per_rank), dtype, allocator=allocator).normal_()
    C = tilelang.tensor((M, N_per_rank), dtype, allocator=allocator)
    grid_barrier = tilelang.tensor((num_local_ranks,), torch.int32, allocator=allocator).zero_()

    mcast_A_flat, gathered_A_flat = allocator._allocate_mcast_tensor((M * K,), dtype)
    mcast_signal, local_signal = allocator._allocate_mcast_tensor((m_blocks,), torch.uint32)
    mcast_A = mcast_A_flat.view(M, K)
    gathered_A = gathered_A_flat.view(M, K)

    dist.barrier(group)
    tilelang_C = ag_gemm_op(A, B, mcast_A, gathered_A, mcast_signal, local_signal, grid_barrier, C, kernel, local_rank)
    torch.cuda.synchronize()
    dist.barrier(group)

    torch_ag_buffer = torch.empty((M, K), dtype=dtype, device=f"cuda:{local_rank}")
    torch_C = torch_ag_gemm(group, A, B, torch_ag_buffer)

    if torch.allclose(torch_C, tilelang_C, atol=1e-2, rtol=1e-2):
        print(f"rank {local_rank} check passed.")
    else:
        max_diff = (torch_C - tilelang_C).abs().max().item()
        print(f"rank {local_rank} check failed. max_diff={max_diff}")

    tl_t = perf_fn(
        lambda: ag_gemm_op(A, B, mcast_A, gathered_A, mcast_signal, local_signal, grid_barrier, C, kernel, local_rank),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"rank {local_rank} tilelang specialized ag_gemm time: {tl_t:.2f} ms, TFLOPS: {2 * M * N * K / 1e9 / tl_t / num_local_ranks:.2f}")

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
    parser.add_argument("--num-comm-sms", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
