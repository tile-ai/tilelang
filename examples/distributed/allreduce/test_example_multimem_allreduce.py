from __future__ import annotations

import argparse

import tilelang.testing
from testing.python.distributed._utils import distributed_test

import example_multimem_allreduce


@distributed_test(nprocs=8, require_multicast=True)
def test_example_multimem_allreduce(local_rank: int, num_ranks: int):
    args = argparse.Namespace(
        num_processes=num_ranks,
        N=65536,
        block_n=512,
        threads=256,
        dtype="bf16",
        strategy="both",
        warmup=1,
        rep=1,
        atol=3e-2,
        rtol=3e-2,
        check=True,
        print_source=False,
    )
    example_multimem_allreduce.main(local_rank, num_ranks, args)


@distributed_test(nprocs=8, require_multicast=True)
def test_example_multimem_allreduce_tma_copy(local_rank: int, num_ranks: int):
    args = argparse.Namespace(
        num_processes=num_ranks,
        N=65536,
        block_n=512,
        threads=256,
        dtype="bf16",
        strategy="two_shot_tma_copy",
        warmup=1,
        rep=1,
        atol=3e-2,
        rtol=3e-2,
        check=True,
        print_source=False,
    )
    example_multimem_allreduce.main(local_rank, num_ranks, args)


if __name__ == "__main__":
    tilelang.testing.main()
