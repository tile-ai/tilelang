from __future__ import annotations

import tilelang
import tilelang.language as T
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
from tilelang.distributed import init_dist
from tilelang.distributed import perf_fn
from reduce_scatter import reduce_scatter_2d_op, create_reduce_scater_2d_ctx

tilelang.disable_cache()


@tilelang.jit
def gemm_kernel(
    M, N, K, local_rank, num_local_rank, block_M, block_N, block_K, threads, persistent=False, dtype="float16", accum_dtype="float"
):
    M_per_rank = T.ceildiv(M, num_local_rank)
    GROUP_SIZE_M = 8

    def swizzle_2d(tile_id, num_pid_m, num_pid_n):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = T.min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    @T.prim_func
    def main(
        A: T.Tensor((M, K // num_local_rank), dtype),
        B: T.Tensor((K // num_local_rank, N), dtype),
        scatter_signal_buf: T.Tensor((num_local_rank), "uint32"),
        counter_signal_buf: T.Tensor((num_local_rank), "uint32"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), threads=threads) as (bid):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            val = T.alloc_local((1,), "uint32")

            num_pid_m = T.ceildiv(M, block_M)
            num_pid_n = T.ceildiv(N, block_N)

            pid_m_, pid_n = swizzle_2d(bid, num_pid_m, num_pid_n)
            pid_m_offset = (local_rank + 1) * M_per_rank // block_M
            pid_m = (pid_m_ + pid_m_offset) % num_pid_m

            tid = T.get_thread_binding(0)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K // num_local_rank, block_K), num_stages=3):
                T.copy(A[pid_m * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, pid_n * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[pid_m * block_M, pid_n * block_N])

            # inc barrier
            segment_start = pid_m * block_M // M_per_rank
            segment_end = (T.min((pid_m + 1) * block_M, M) - 1) // M_per_rank
            segment = segment_start + tid
            if segment <= segment_end:
                m_start = M_per_rank * segment
                m_end = M_per_rank * (segment + 1) - 1
                tiled_m_start = m_start // block_M
                tiled_m_end = m_end // block_M
                tiled_m_size = tiled_m_end - tiled_m_start + 1
                val[0] = T.atom_add(counter_signal_buf[segment], 1, scope="gpu", sem="release")
                if T.Cast("int32", val[0]) == num_pid_n * tiled_m_size - 1:
                    T.st(scatter_signal_buf[segment], 1, scope="gpu", sem="release")

    return main


def gemm_rs_op(A, B, C, output, ctx, gemm_kernel, gemm_stream, rs_stream, local_rank, print_source=False):
    current_stream = torch.cuda.current_stream()
    rs_stream.wait_stream(gemm_stream)

    with torch.cuda.stream(gemm_stream):
        gemm_kernel(A, B, ctx.scatter_signal_bufs[local_rank], ctx.counter_bufs[local_rank], C)

    if print_source and local_rank == 1:
        print(gemm_kernel.get_kernel_source())

    with torch.cuda.stream(rs_stream):
        # don't allocate memory on other stream: error-prune
        output = reduce_scatter_2d_op(C, ctx, output)
    gemm_stream.wait_stream(rs_stream)
    current_stream.wait_stream(rs_stream)

    return output


def torch_gemm_rs(
    pg: torch.distributed.ProcessGroup,
    input: torch.Tensor,  # [M, local_k]
    weight: torch.Tensor,  # [local_K, N]
    bias: torch.Tensor | None,
    num_local_ranks: int,
):
    M, local_K = input.shape
    N = weight.shape[1]
    output = torch.matmul(input, weight)
    if bias:
        output = output + bias
    rs_output = torch.empty((M // num_local_ranks, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output, group=pg)
    return rs_output


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.float16
    M = args.M if args else 8192
    N = args.N if args else 8192
    K = args.K if args else 8192
    persistent = args.persistent
    M_per_rank = M // num_local_ranks
    K_per_rank = K // num_local_ranks

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    threads = 256

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks, "only support single node for now"
    allocator = tilelang.get_allocator(
        size=2**30, device="cuda", is_distributed=True, local_rank=local_rank, num_local_ranks=num_local_ranks, group=group
    )
    gemm_func = gemm_kernel(M, N, K, local_rank, num_local_ranks, BLOCK_M, BLOCK_N, BLOCK_K, threads, persistent)
    gemm_func.initialize(allocator=allocator)

    A = tilelang.tensor((M, K_per_rank), dtype, allocator=allocator).normal_() / 10
    B = tilelang.tensor((K_per_rank, N), dtype, allocator=allocator).normal_() / 10
    C = tilelang.tensor((M, N), dtype, allocator=allocator)
    output = tilelang.tensor((M_per_rank, N), dtype, allocator=allocator)
    gemm_stream = torch.cuda.Stream()
    rs_stream = torch.cuda.Stream(priority=-1)
    ctx = create_reduce_scater_2d_ctx(
        M, N, local_rank, num_local_ranks, num_local_ranks, dtype, allocator, overlap_with_gemm=True, num_reduction_sms=15
    )

    dist.barrier()

    tilelang_out = gemm_rs_op(A, B, C, output, ctx, gemm_func, gemm_stream, rs_stream, local_rank, print_source=True)
    torch_out = torch_gemm_rs(group, A, B, None, num_local_ranks)

    atol = 1e-2
    rtol = 1e-2
    if torch.allclose(torch_out, tilelang_out, atol=atol, rtol=rtol):
        print(f"rank {local_rank} check passed.✅")
    else:
        print(f"rank {local_rank} check failed.❌")
        print(f"torch_out: {torch_out}, tilelang_out: {tilelang_out}")

    _, tl_t = perf_fn(lambda: gemm_rs_op(A, B, C, output, ctx, gemm_func, gemm_stream, rs_stream, local_rank), warmup=5, rep=5)

    print(f"rank {local_rank} tilelang gemm_rs time: {tl_t:.2f} ms, TFLOPS: {2 * M * N * K / 1e9 / (tl_t) / num_local_ranks:.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2, help="Number of processes to spawn (default: 2)")
    parser.add_argument("--M", type=int, default=8192, help="M dimension")
    parser.add_argument("--N", type=int, default=8192, help="N dimension")
    parser.add_argument("--K", type=int, default=29568, help="K dimension")
    parser.add_argument("--persistent", action="store_true", help="Use persistent kernel")
    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
