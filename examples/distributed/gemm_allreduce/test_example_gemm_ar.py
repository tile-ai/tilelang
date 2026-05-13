from __future__ import annotations

import argparse

import tilelang.testing
from testing.python.distributed._utils import distributed_test

import example_gemm_ar_specialized


@distributed_test(nprocs=8, require_multicast=True)
def test_example_gemm_ar_specialized(local_rank: int, num_ranks: int):
    args = argparse.Namespace(
        num_processes=num_ranks,
        M=512,
        N=512,
        K=128,
        block_m=128,
        block_n=256,
        block_k=64,
        threads=256,
        pipeline_stages=4,
        ar_block_e=2048,
        num_comm_sms=4,
        group_size_m=12,
        two_kernel=False,
        warmup=1,
        rep=1,
        atol=0.5,
        rtol=1e-1,
        check=True,
        print_source=False,
    )
    example_gemm_ar_specialized.main(local_rank, num_ranks, args)


if __name__ == "__main__":
    tilelang.testing.main()
