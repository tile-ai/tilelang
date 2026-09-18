import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_zz_layout

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


_SHAPES = [
    (256, 128, 128),
    (128, 128, 128),
    (256, 256, 128),
    (256, 256, 256),
]


@target("Sunmmio")
def summa_matmul(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    block_K=32,
    dtype="bfloat16",
    accum_dtype="float32",
):
    shard_policy = T.placement.full_shard(0, 1)

    A_shape = (M, K)
    B_shape = (K, N)
    C_shape = (M, N)
    A_layout = make_zz_layout(A_shape, [0, 1], (32, 32))
    B_layout = make_zz_layout(B_shape, [0, 1], (32, 32))
    C_layout = make_zz_layout(C_shape, [0, 1], (32, 32))

    @T.prim_func
    def kernel(
        A: T.MeshTensor(A_shape, shard_policy, dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor(B_shape, shard_policy, dtype, layout=B_layout),  # type: ignore
        C: T.MeshTensor(C_shape, shard_policy, accum_dtype, layout=C_layout),  # type: ignore
    ):
        with T.Kernel() as _cid:
            sharded_M, _ = A.local_shape
            _, sharded_N = B.local_shape
            core_row = _cid // T.ncols()
            core_col = _cid % T.ncols()

            A_broadcast = T.alloc_shared((block_M, block_K), dtype, scope="shared.rsram")
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_local)
                    K_steps = T.ceildiv(K, block_K)

                    for k_tile in range(K_steps):
                        a_src_col = k_tile % T.ncols()
                        b_src_row = k_tile % T.nrows()
                        a_local_k = (k_tile // T.ncols()) * block_K
                        b_local_k = (k_tile // T.nrows()) * block_K

                        T.comm.broadcast(
                            A[
                                bx * block_M : bx * block_M + block_M,
                                a_local_k : a_local_k + block_K,
                            ],
                            A_broadcast,
                            (core_row, a_src_col),
                            direction="h",
                        )
                        T.copy(A_broadcast, A_shared)
                        T.comm.broadcast(
                            B[
                                b_local_k : b_local_k + block_K,
                                by * block_N : by * block_N + block_N,
                            ],
                            B_shared,
                            (b_src_row, core_col),
                            direction="v",
                        )
                        # Sunmmio MMA consumes B in the transposed operand mode;
                        # algorithmically this is still the SUMMA B(k, j) panel.
                        T.gemm(A_shared, B_shared, C_local)

                    T.copy(C_local, C[bx * block_M, by * block_N])

    return kernel


@pytest.mark.parametrize("M,N,K", _SHAPES)
def test_summa_matmul_codegen_validates_with_npuir_opt(tmp_path, M, N, K):
    src = validate_sunmmio_codegen_with_npuir_opt(
        summa_matmul(M=M, N=N, K=K),
        tmp_path,
        mlir_filename=f"summa_matmul_m{M}_n{N}_k{K}_suvm.mlir",
        expected_tokens=(
            "suvm.copy_async",
            "suvm.mcast_tok",
            "suvm.tc.mma",
            "suvm.sync",
        ),
    )

    assert "sunmmio.fake" not in src
    assert src.count("suvm.mcast_tok") >= 2
    assert src.count("suvm.tc.mma") >= 1
    assert src.count("suvm.sync") >= 3
    assert "!suvm.token" not in src


def test_summa_matmul_codegen_contains_broadcast_sync_sequence(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        summa_matmul(),
        tmp_path,
        mlir_filename="summa_matmul_sync_suvm.mlir",
        expected_tokens=(
            "suvm.barrier.init",
            "suvm.barrier.arrive_and_wait",
            "suvm.mcast_tok",
            "suvm.tc.mma",
            "suvm.copy_async",
        ),
    )

    assert_source_contains(src, ("suvm.sync", "suvm.tile.fill"))


if __name__ == "__main__":
    tilelang.testing.main()
