"""Small distributed remote-copy correctness test."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
import tilelang.testing
from testing.python.distributed._utils import distributed_test

os.environ.setdefault("NCCL_DEBUG", "WARN")

_M = 1024
_BLOCK_M = 128
_THREADS = 128


def _kernel_remote_copy(M: int, block_M: int, threads: int):
    @T.prim_func
    def main(dst: T.Tensor((M,), "float32"), src: T.Tensor((M,), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            rank = T.alloc_local((1,), "uint64")
            rank[0] = T.get_rank()
            T.put_block(
                src=T.address_of(src[bx * block_M]),
                dst=T.address_of(dst[bx * block_M]),
                size=block_M,
                dst_pe=rank[0] ^ 1,
            )

    return main


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test()
def test_remote_copy(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**20,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )

    kernel = tilelang.compile(
        _kernel_remote_copy(_M, _BLOCK_M, _THREADS),
        compile_once=True,
        compile_group=group,
    )
    if rank == 0:
        source = kernel.get_kernel_source()
        assert "tl::get_remote_base_ptr" in source
        assert "tl::get_uintptr_t" in source
    kernel.initialize(allocator=allocator)

    src = tilelang.tensor((_M,), torch.float32, allocator=allocator).normal_()
    dst = tilelang.tensor((_M,), torch.float32, allocator=allocator).zero_()

    torch.cuda.synchronize()
    dist.barrier(group)
    kernel(dst, src)
    torch.cuda.synchronize()
    dist.barrier(group)

    src_refs = [torch.empty_like(src) for _ in range(num_ranks)]
    dist.all_gather(src_refs, src, group)
    expected = src_refs[rank ^ 1]
    assert torch.allclose(expected, dst, atol=1e-6, rtol=1e-6)

    dist.destroy_process_group()


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
