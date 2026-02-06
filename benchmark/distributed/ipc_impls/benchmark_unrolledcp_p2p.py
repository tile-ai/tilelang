import os
import tilelang
import tilelang.language as T
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
from tilelang.distributed import init_dist, perf_fn

tilelang.disable_cache()
os.environ["NCCL_DEBUG"] = "WARN"


def ipc_kernel_push(size, threads, unroll_factor):
    @T.prim_func
    def ipc_push(
        dst: T.Tensor((size), "float32"),  # type: ignore
        src: T.Tensor((size), "float32"),  # type: ignore
    ):
        with T.Kernel(1, threads=threads):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            warp_idx = T.get_thread_binding(0) // 32
            warp_copy_size = T.ceildiv(size, threads // 32)
            warp_start = warp_copy_size * warp_idx
            T.put_warp(
                src=T.address_of(src[warp_start]),
                dst=T.address_of(dst[warp_start]),
                size=warp_copy_size,
                dst_pe=rank[0] ^ 1,
                unroll_factor=unroll_factor,
            )
            T.fence_sys()

    return ipc_push


def ipc_kernel_pull(size, threads, unroll_factor):
    @T.prim_func
    def ipc_pull(
        dst: T.Tensor((size), "float32"),  # type: ignore
        src: T.Tensor((size), "float32"),  # type: ignore
    ):
        with T.Kernel(1, threads=threads):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            warp_idx = T.get_thread_binding(0) // 32
            warp_copy_size = T.ceildiv(size, threads // 32)
            warp_start = warp_copy_size * warp_idx
            T.get_warp(
                src=T.address_of(src[warp_start]),
                dst=T.address_of(dst[warp_start]),
                size=warp_copy_size,
                src_pe=rank[0] ^ 1,
                unroll_factor=unroll_factor,
            )
            T.fence_sys()

    return ipc_pull


def benchmark_ipc_bw(rank: int, num_ranks: int, group: dist.ProcessGroup, size: int, args: argparse.Namespace, allocator):
    assert num_ranks == 2, "this benchmark only supports 2 ranks"
    assert args.threads % 32 == 0, "threads must be divisible by 32"

    kernel = tilelang.compile(ipc_kernel_push(size, args.threads, args.unroll_factor))
    kernel.initialize(allocator=allocator)
    src = tilelang.tensor((size,), torch.float32, allocator=allocator).random_()
    dst = tilelang.tensor((size,), torch.float32, allocator=allocator)

    def push_fn():
        kernel(dst, src)

    dist.barrier(group)
    torch.cuda.synchronize()
    _, t_push = perf_fn(push_fn, args.warmup, args.repeat)  # 1st returned value is output
    bw_push = (size * 4 * 1e-9) / (t_push * 1e-3)

    dist.barrier(group)

    # Reuse allocator and tensors
    kernel = tilelang.compile(ipc_kernel_pull(size, args.threads, args.unroll_factor))
    kernel.initialize(allocator=allocator)

    def pull_fn():
        kernel(dst, src)

    dist.barrier(group)
    torch.cuda.synchronize()
    _, t_pull = perf_fn(pull_fn, args.warmup, args.repeat)  # 1st returned value is output
    bw_pull = (size * 4 * 1e-9) / (t_pull * 1e-3)

    dist.barrier(group)

    return bw_push, bw_pull


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    allocator = tilelang.get_allocator(
        size=2**30, device="cuda", is_distributed=True, local_rank=rank, num_local_ranks=num_ranks, group=group
    )

    for log_size in range(9, 21):
        size = 2**log_size
        push_bw, pull_bw = benchmark_ipc_bw(rank, num_ranks, group, size, args, allocator)
        if rank == 0:
            print(f"size={size * 4} bytes, ipc push bw: {push_bw:.4f} GB/s, ipc pull bw: {pull_bw:.4f} GB/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=50, help="number of repeat iterations (default: 50)")
    parser.add_argument("--threads", type=int, default=128, help="Threads per block (default: 128)")
    parser.add_argument("--unroll-factor", type=int, default=4, help="Unroll factor (default: 4)")
    args = parser.parse_args()
    nprocs = 2

    torch.multiprocessing.spawn(main, args=(nprocs, args), nprocs=nprocs)
