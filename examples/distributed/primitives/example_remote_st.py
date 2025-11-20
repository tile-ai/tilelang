import os
import tilelang
import tilelang.language as T
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
from tilelang.distributed import init_dist

tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


def kernel_(M, num_rank, block_M, threads):

    @T.prim_func
    def main(
            dst: T.Tensor((M), "float32"),
            src: T.Tensor((M), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            num_rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            num_rank[0] = T.get_num_ranks()
            tx = T.get_thread_binding()
            T.st(dst[bx * block_M + tx], src[bx * block_M + tx], dst_pe=1-rank[0])

    return main


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    M = args.M
    BLOCK_M = threads = 128
    assert num_local_ranks == 2, "this example only supports 2 ranks copying to each other"

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    allocator = tilelang.get_allocator(
        size=2**25,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group)
    kernel = tilelang.compile(kernel_(M, num_ranks, BLOCK_M, threads))
    kernel.initialize(allocator=allocator)
    if local_rank == 0:
        print(kernel.get_kernel_source())

    src = tilelang.tensor((M), torch.float32, allocator=allocator).normal_()
    dst = tilelang.tensor((M), torch.float32, allocator=allocator)

    torch.cuda.synchronize()
    torch.distributed.barrier(group)
    kernel(dst, src)
    torch.cuda.synchronize()
    torch.distributed.barrier(group)

    dst_torchs = [torch.empty_like(src) for _ in range(num_local_ranks)]
    dist.all_gather(dst_torchs, src, group)
    dst_torch = dst_torchs[local_rank ^ 1]

    if torch.allclose(dst_torch, dst, atol=1e-6, rtol=1e-6):
        print(f"rank {local_rank} check passed.✅")
    else:
        print(f"rank {local_rank} check failed.❌")
        print(f"dst_torch: {dst_torch}, dst: {dst}")
        raise ValueError("Test failed")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-processes', type=int, default=2, help='Number of processes to spawn (default: 2)')
    parser.add_argument('--M', type=int, default=1024, help='M dimension')
    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
