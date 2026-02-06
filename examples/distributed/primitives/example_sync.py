import os
import tilelang
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
from tilelang.distributed import init_dist

tilelang.disable_cache()
os.environ["NCCL_DEBUG"] = "WARN"  # silence NCCL log


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    M = args.M if args else 65536
    assert num_local_ranks == 2, "this example only supports 2 ranks copying to each other"

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    allocator = tilelang.get_allocator(
        size=2**25, device="cuda", is_distributed=True, local_rank=local_rank, num_local_ranks=num_local_ranks, group=group
    )

    dst = tilelang.tensor((M), torch.float32, allocator=allocator)
    srcs = tilelang.tensor((M), torch.float32, allocator=allocator, return_peers=True)

    print(f"Before: rank {rank}; src: {srcs[rank]}")
    if rank == 0:
        srcs[1][0:10] = torch.arange(10, dtype=torch.float32) + 100

    dist.barrier(group)
    print(f"After: rank {rank}; src: {srcs[rank]}")
    print(f"After: rank {rank}; dst: {dst}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2, help="Number of processes to spawn (default: 2)")
    parser.add_argument("--M", type=int, default=65536, help="M dimension")
    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
