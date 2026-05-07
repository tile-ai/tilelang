import tilelang
import tilelang.testing
import tilelang.language as T
import torch
import torch.distributed as dist
import torch.multiprocessing
import argparse
from tilelang.distributed import init_dist


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True})
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
                T.put_warp(src=T.address_of(A), dst=T.address_of(B[bid, 0]), size=threads, dst_pe=rank[0] ^ 1, unroll_factor=4)

    return main


def main(local_rank: int, num_ranks: int, args: argparse.Namespace):
    blocks, threads = args.blocks, args.threads

    _, _, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**20, device="cuda", is_distributed=True, local_rank=local_rank, num_local_ranks=num_ranks, group=group
    )
    kernel = get_test_barrierall_sys_kernel(num_ranks, blocks, threads)
    kernel.initialize(allocator=allocator)

    A = tilelang.tensor([threads], torch.int32, allocator=allocator).zero_()
    barrier = tilelang.tensor([num_ranks], torch.int32, allocator=allocator).zero_()
    output = tilelang.tensor([blocks, threads], torch.int32, allocator=allocator).zero_()
    torch.cuda.synchronize()
    dist.barrier(group)
    kernel(A, barrier, output)
    torch.cuda.synchronize()
    dist.barrier(group)
    if torch.all(output == blocks):
        print(f"rank {local_rank} check passed.✅")
    else:
        print(f"rank {local_rank} check failed.❌")
        print(f"output: {output}")

    dist.destroy_process_group()


@tilelang.testing.requires_cuda
def test_barrierall_sys():
    # Trigger pre-compile
    kernel = get_test_barrierall_sys_kernel(2, 64, 128)  # noqa: F841
    torch.multiprocessing.spawn(main, args=(2, argparse.Namespace(blocks=64, threads=128)), nprocs=2)


if __name__ == "__main__":
    tilelang.testing.main()
