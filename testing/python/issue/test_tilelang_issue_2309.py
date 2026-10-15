# Copyright (c) 2025 Tile-AI Corporation.
# Licensed under the MIT License.
"""Tests for sibling pipeline shared buffer handling (issue #2309).

When two sibling T.Pipelined loops share the same alloc_shared buffer but use
different num_stages, the allocation must accommodate every loop's versions
while each loop retains its own version indices. All views must agree on the
allocation shape before layout inference.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


def run_asymmetric_pipeline(stages):
    """Sibling pipelines sharing alloc_shared buffers with different num_stages."""
    M, N, K = 128, 128, 256 * len(stages)
    BM, BN, BK = 64, 64, 32
    first, second = stages[:2]
    third = stages[2] if len(stages) == 3 else 0
    K_chunk = 256

    @T.prim_func
    def asymmetric_stages(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_shared = T.alloc_shared((BM, BK), T.float16)
            B_shared = T.alloc_shared((BK, BN), T.float16)
            C_local = T.alloc_fragment((BM, BN), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(K_chunk // BK, num_stages=first):
                T.copy(A[by * BM, ko * BK], A_shared)
                T.copy(B[ko * BK, bx * BN], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for ko in T.Pipelined(K_chunk // BK, num_stages=second):
                T.copy(A[by * BM, K_chunk + ko * BK], A_shared)
                T.copy(B[K_chunk + ko * BK, bx * BN], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            if third:
                for ko in T.Pipelined(K_chunk // BK, num_stages=third):
                    T.copy(A[by * BM, 2 * K_chunk + ko * BK], A_shared)
                    T.copy(B[2 * K_chunk + ko * BK, bx * BN], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * BM, bx * BN])

    kernel = tilelang.compile(
        asymmetric_stages,
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )

    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.float16)
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "stages",
    [(2, 4), (4, 2), (3, 1), (1, 3), (3, 3), (2, 4, 3), (2, 3, 4), (1, 3, 1), (4, 1, 2)],
    ids=lambda stages: "-".join(map(str, stages)),
)
def test_sibling_pipeline_different_num_stages(stages):
    """Reproducer for issue #2309: sibling pipelines with different num_stages
    sharing alloc_shared buffers should compile and run correctly."""
    run_asymmetric_pipeline(stages)


if __name__ == "__main__":
    tilelang.testing.main()
