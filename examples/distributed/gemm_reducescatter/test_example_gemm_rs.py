from __future__ import annotations

import argparse

import tilelang.testing
from testing.python.distributed._utils import distributed_test

import example_gemm_rs_overlapped
import example_gemm_rs_specialized


@distributed_test(nprocs=8)
def test_example_gemm_rs_overlapped(local_rank: int, num_ranks: int):
    args = argparse.Namespace(
        num_processes=num_ranks,
        M=1024,
        N=512,
        K=512,
        persistent=False,
        print_source=False,
        warmup=1,
        rep=1,
    )
    example_gemm_rs_overlapped.main(local_rank, num_ranks, args)


@distributed_test(nprocs=8)
def test_example_gemm_rs_specialized(local_rank: int, num_ranks: int):
    args = argparse.Namespace(
        num_processes=num_ranks,
        M=1024,
        N=512,
        K=512,
        block_m=128,
        block_n=256,
        block_k=64,
        threads=256,
        group_size_m=12,
        pipeline_stages=4,
        tma_epilogue=True,
        warmup=1,
        rep=1,
        atol=0.5,
        rtol=1e-1,
        check=True,
        print_source=False,
    )
    example_gemm_rs_specialized.main(local_rank, num_ranks, args)


if __name__ == "__main__":
    tilelang.testing.main()
