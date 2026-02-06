# This benchmark aims to measure the bandwidth of NVHSMEM-based communication.
# We launch only one block on each rank to avoid NVLink bandwidth as the bottleneck.

# Usage: GPUS=2 bash tilelang/distributed/launch.sh benchmark/distributed/benchmark_nvshmem_p2p.py

import os
import tilelang
import tilelang.language as T
import argparse
import torch
import torch.distributed as dist
from tilelang.distributed import init_distributed, perf_fn
import pynvshmem

os.environ["NCCL_DEBUG"] = "WARN"


def nvshmem_kernel_push(size, threads):
    @T.prim_func
    def nvshmem_push(
        dst: T.Tensor((size), "float32"),  # type: ignore
        src: T.Tensor((size), "float32"),  # type: ignore
    ):
        with T.Kernel(1, threads=threads):
            T.putmem_block(
                T.address_of(dst),
                T.address_of(src),
                size * 4,
                T.get_pe() ^ 1,
            )
            T.fence_sys()

    return nvshmem_push


def nvshmem_kernel_pull(size, threads):
    @T.prim_func
    def nvshmem_pull(
        dst: T.Tensor((size), "float32"),  # type: ignore
        src: T.Tensor((size), "float32"),  # type: ignore
    ):
        with T.Kernel(1, threads=threads):
            T.getmem_block(
                T.address_of(dst),
                T.address_of(src),
                size * 4,
                T.get_pe() ^ 1,
            )
            T.fence_sys()

    return nvshmem_pull


def benchmark_nvshmem_bw(rank: int, num_ranks: int, group: dist.ProcessGroup, size: int, args: argparse.Namespace):
    assert num_ranks == 2, "this benchmark only supports 2 ranks"
    assert args.threads % 32 == 0, "threads must be divisible by 32"

    kernel = tilelang.compile(nvshmem_kernel_push(size, args.threads))
    src = pynvshmem.nvshmem_create_tensor([size], torch.float32)
    dst = pynvshmem.nvshmem_create_tensor([size], torch.float32)

    def push_fn():
        kernel(dst, src)

    dist.barrier(group)
    torch.cuda.synchronize()
    _, t_push = perf_fn(push_fn, args.warmup, args.repeat)  # 1st returned value is output
    bw_push = (size * 4 * 1e-9) / (t_push * 1e-3)

    dist.barrier(group)

    # Reuse allocator and tensors
    kernel = tilelang.compile(nvshmem_kernel_pull(size, args.threads))

    def pull_fn():
        kernel(dst, src)

    dist.barrier(group)
    torch.cuda.synchronize()
    _, t_pull = perf_fn(pull_fn, args.warmup, args.repeat)  # 1st returned value is output
    bw_pull = (size * 4 * 1e-9) / (t_pull * 1e-3)

    dist.barrier(group)

    return bw_push, bw_pull


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=50, help="number of repeat iterations (default: 50)")
    parser.add_argument("--threads", type=int, default=128, help="Threads per block (default: 128)")
    args = parser.parse_args()

    num_ranks, rank, _, group = init_distributed(return_tp_group=True)
    for log_size in range(9, 21):
        size = 2**log_size
        push_bw, pull_bw = benchmark_nvshmem_bw(rank, num_ranks, group, size, args)
        if rank == 0:
            print(f"size={size * 4} bytes, nvshmem push bw: {push_bw:.4f} GB/s, nvshmem pull bw: {pull_bw:.4f} GB/s")

    dist.destroy_process_group()
