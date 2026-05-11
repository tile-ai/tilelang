from __future__ import annotations

import argparse

import tilelang.testing
from testing.python.distributed._utils import distributed_test

import example_allgather_gemm_overlapped
import example_allgather_gemm_specialized


@distributed_test(nprocs=8)
def test_example_allgather_gemm_overlapped(local_rank: int, num_ranks: int):
    args = argparse.Namespace(
        num_processes=num_ranks,
        M=1024,
        N=2048,
        K=256,
        block_m=128,
        block_n=256,
        block_k=64,
        threads=256,
        group_size_m=8,
        warmup=1,
        rep=1,
        persistent=False,
        print_source=False,
    )
    example_allgather_gemm_overlapped.main(local_rank, num_ranks, args)


@distributed_test(nprocs=8)
def test_example_allgather_gemm_specialized(local_rank: int, num_ranks: int):
    args = argparse.Namespace(
        num_processes=num_ranks,
        M=2048,
        N=2048,
        K=256,
        block_m=128,
        block_n=256,
        block_k=64,
        threads=256,
        num_comm_sms=4,
        group_size_m=12,
        pipeline_stages=4,
        use_tma_store=False,
        warmup=1,
        rep=1,
        print_source=False,
    )
    example_allgather_gemm_specialized.main(local_rank, num_ranks, args)


if __name__ == "__main__":
    tilelang.testing.main()
