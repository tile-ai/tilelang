import os
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
import torch
import torch.distributed as dist
import torch.multiprocessing
import argparse
from tilelang.distributed import init_dist

tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


@tilelang.jit(
    out_idx=-1, pass_configs={
        "tl.disable_warp_specialized": True,
        "tl.disable_tma_lower": True
    })
def get_test_barrierall_sys_kernel(num_ranks: int, blocks: int, threads: int):

    @T.prim_func
    def main(
            A: T.Tensor([threads], "int32"),  # type: ignore
            barrier: T.Tensor([num_ranks], "int32"),  # type: ignore
            B: T.Tensor([blocks, threads], "int32"),  # type: ignore
    ):
        with T.Kernel(blocks, threads=threads) as bid:
            tid = T.get_thread_binding()
            rank = T.alloc_local([1], "int32")
            rank[0] = T.get_rank()
            val = T.alloc_local([1], "int32")
            val[0] = 1
            T.atomic_add(A[tid], val[0])

            T.barrier_blocks(barrier)

            if tid < 32:
                T.put_warp(
                    src=T.address_of(A),
                    dst=T.address_of(B[bid, 0]),
                    size=threads,
                    dst_pe=rank[0] ^ 1,
                    unroll_factor=4)

    return main


def main(local_rank: int, num_ranks: int, args: argparse.Namespace):
    blocks, threads = args.blocks, args.threads

    _, _, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**20,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group)
    kernel = get_test_barrierall_sys_kernel(num_ranks, blocks, threads)
    kernel.initialize(allocator=allocator)

    A = tilelang.tensor([threads], torch.int32, allocator=allocator).zero_()
    barrier = tilelang.tensor([num_ranks], torch.int32, allocator=allocator).zero_()
    torch.cuda.synchronize()
    dist.barrier(group)
    output = kernel(A, barrier)
    torch.cuda.synchronize()
    dist.barrier(group)
    if torch.all(output == blocks):
        print(f"rank {local_rank} check passed.✅")
    else:
        print(f"rank {local_rank} check failed.❌")
        print(f"output: {output}")

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-processes', type=int, default=2, help='Number of processes to spawn (default: 2)')
    parser.add_argument('--blocks', type=int, default=64, help='Number of blocks (default: 64)')
    parser.add_argument('--threads', type=int, default=128, help='Number of threads (default: 128)')
    parser.add_argument(
        '--print-source', action='store_true', help='Print the source code of the kernel')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    num_processes = args.num_processes
    assert args.blocks <= driver.get_num_sms(
    ), f'Launched {args.blocks} blocks, which is larger than the number of SM ({driver.get_num_sms()}) on the current device and may cause deadlock!'

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
