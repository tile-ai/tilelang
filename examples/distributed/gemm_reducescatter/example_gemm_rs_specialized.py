import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.distributed import init_dist
from tilelang.distributed.shared_memory import tensor_from_ptr

os.environ.setdefault("NCCL_DEBUG", "ERROR")
tilelang.enable_cache()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
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

    def tile_coords(tile_id, local_rank):
        """TK-style rotated Super-M scheduler for B tile locality."""
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

    @T.macro
    def atomic_add_shared_output(C_dst, local_by, bx, C_shared):
        # Match TK/blackwell-numa: first materialize the accumulator tile in
        # shared memory, then issue bf16x2 global reductions from that row-major
        # staging buffer. This avoids keeping the converted C fragment live
        # across the full peer atomic epilogue.
        tid = T.get_thread_binding()
        for idx in T.serial(tid, block_M * block_N // 2, threads):
            elem = idx * 2
            T.atomic_addx2(
                C_dst[local_by * block_M + elem // block_N, bx * block_N + elem % block_N],
                C_shared[elem // block_N, elem % block_N],
            )

    @T.macro
    def tma_atomic_add_shared_output(C_dst, local_by, bx, C_shared):
        T.atomic_add(
            C_dst[
                local_by * block_M : (local_by + 1) * block_M,
                bx * block_N : (bx + 1) * block_N,
            ],
            C_shared,
            use_tma=True,
        )

    @T.macro
    def reduce_shared_output(C_dst, local_by, bx, C_shared):
        if use_tma_epilogue:
            tma_atomic_add_shared_output(C_dst, local_by, bx, C_shared)
        else:
            atomic_add_shared_output(C_dst, local_by, bx, C_shared)

    @T.prim_func
    def main(
        A: T.Tensor((M, K_per_rank), dtype),
        B: T.Tensor((K_per_rank, N), dtype),
        C0: T.Tensor((M_per_rank, N), dtype),
        C1: T.Tensor((M_per_rank, N), dtype),
        C2: T.Tensor((M_per_rank, N), dtype),
        C3: T.Tensor((M_per_rank, N), dtype),
        C4: T.Tensor((M_per_rank, N), dtype),
        C5: T.Tensor((M_per_rank, N), dtype),
        C6: T.Tensor((M_per_rank, N), dtype),
        C7: T.Tensor((M_per_rank, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as bid:
            local_rank = T.get_rank()
            with T.sm_specialize_scope(auto_ws=True):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

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

                        C_shared = T.alloc_shared((block_M, block_N), dtype)
                        T.copy(C_local, C_shared)
                        T.sync_threads(3, threads)
                        if dst_rank == 0:
                            reduce_shared_output(C0, local_by, bx, C_shared)
                        elif dst_rank == 1:
                            reduce_shared_output(C1, local_by, bx, C_shared)
                        elif dst_rank == 2:
                            reduce_shared_output(C2, local_by, bx, C_shared)
                        elif dst_rank == 3:
                            reduce_shared_output(C3, local_by, bx, C_shared)
                        elif dst_rank == 4:
                            reduce_shared_output(C4, local_by, bx, C_shared)
                        elif dst_rank == 5:
                            reduce_shared_output(C5, local_by, bx, C_shared)
                        elif dst_rank == 6:
                            reduce_shared_output(C6, local_by, bx, C_shared)
                        else:
                            reduce_shared_output(C7, local_by, bx, C_shared)
                        T.sync_threads(3, threads)

    return main


def torch_gemm_rs(group: torch.distributed.ProcessGroup, A: torch.Tensor, B: torch.Tensor, num_ranks: int):
    partial = torch.matmul(A, B)
    output = torch.empty((A.shape[0] // num_ranks, B.shape[1]), dtype=partial.dtype, device=A.device)
    dist.reduce_scatter_tensor(output, partial, group=group)
    return output


def benchmark_tk_aligned(fn, prepare_fn, group, *, warmup: int, rep: int):
    torch.cuda.synchronize()
    dist.barrier(group)
    for _ in range(warmup):
        prepare_fn()
        fn()
    torch.cuda.synchronize()
    dist.barrier(group)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        prepare_fn()
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    local_times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_events, end_events)],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(local_times, op=dist.ReduceOp.MAX, group=group)
    dist.barrier(group)
    return local_times.mean().item()


def make_peer_views(local_tensor: torch.Tensor, allocator, shape, dtype, local_rank: int, num_ranks: int):
    dtype_str = str(dtype).split(".")[-1]
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    base_ptr = int(allocator._base_ptr.value)
    local_ptr = local_tensor.data_ptr()
    offset = local_ptr - base_ptr
    expected_bytes = 1
    for extent in shape:
        expected_bytes *= extent
    expected_bytes *= elem_bytes
    if offset < 0 or offset + expected_bytes > allocator.size:
        raise RuntimeError("local tensor is not inside the distributed allocator arena")

    peer_views = []
    for rank in range(num_ranks):
        if rank == local_rank:
            peer_views.append(local_tensor)
        else:
            peer_ptr = int(allocator._buffer_ptrs[rank]) + offset
            peer_views.append(tensor_from_ptr(peer_ptr, shape, dtype_str, allocator._device, False))
    return peer_views


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.bfloat16
    M, N, K = args.M, args.N, args.K
    block_M, block_N, block_K = args.block_m, args.block_n, args.block_k

    assert num_local_ranks == 8, "this specialized example currently passes 8 peer C tensors explicitly"
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
    C_peers = make_peer_views(C_local, allocator, (M_per_rank, N), dtype, local_rank, num_local_ranks)
    C_local.zero_()
    dist.barrier(group)

    def prepare():
        C_local.zero_()
        torch.cuda.synchronize()
        dist.barrier(group)

    def run_kernel():
        kernel(A, B, *C_peers)

    prepare()
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

    tl_t = benchmark_tk_aligned(run_kernel, prepare, group, warmup=args.warmup, rep=args.rep)

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
    parser.add_argument("--pipeline-stages", type=int, default=3)
    parser.add_argument("--tma-epilogue", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes)
