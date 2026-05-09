"""Tests for distributed T.wait_eq + remote T.st signaling."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
import tilelang.testing
from testing.python.distributed._utils import distributed_test

os.environ.setdefault("NCCL_DEBUG", "WARN")


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def _signal_wait_kernel():
    @T.prim_func
    def main(
        signal: T.Tensor((2,), T.uint32),
        data: T.Tensor((1,), T.int32),
        out: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(1, threads=32):
            rank = T.alloc_local((1,), T.uint64)
            rank[0] = T.get_rank()
            tid = T.get_thread_binding()
            if rank[0] == 0:
                if tid == 0:
                    T.st(data[0], 123, scope="sys", sem="release", dst_pe=1)
                    T.st(signal[0], 1, scope="sys", sem="release", dst_pe=1)
            else:
                if tid == 0:
                    T.wait_eq(signal[0], 1)
                    out[0] = data[0]

    return main


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test()
def test_signal_wait(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    _, _, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**20,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )
    kernel = _signal_wait_kernel()
    kernel.initialize(allocator=allocator)

    signal = tilelang.tensor((num_ranks,), torch.uint32, allocator=allocator).zero_()
    data = tilelang.tensor((1,), torch.int32, allocator=allocator).zero_()
    out = tilelang.tensor((1,), torch.int32, allocator=allocator).zero_()

    torch.cuda.synchronize()
    dist.barrier(group)
    kernel(signal, data, out)
    torch.cuda.synchronize()
    dist.barrier(group)

    if local_rank == 1:
        assert out.cpu()[0].item() == 123

    dist.destroy_process_group()


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
