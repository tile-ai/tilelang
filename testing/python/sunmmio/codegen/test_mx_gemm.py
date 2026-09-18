import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import (
    make_mx_row_major_layout,
    make_mxznz_layout,
    make_mxzz_layout,
    make_zz_layout,
)

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ.setdefault("SUNMMIO_TEST_LOG_IR", "0")

MX_DTYPE_CASES = (
    pytest.param(T.mxfp8, "!suvm.mxfp8", 256, 64, id="mxfp8"),
    pytest.param(T.mxfp4, "!suvm.mxfp4", 128, 32, id="mxfp4"),
)


def _is_mx_dtype(dtype):
    return str(T.dtype(dtype)) in {"custom[mxfp8]8", "custom[mxfp4]4"}


def _dtype_filename(dtype):
    return str(T.dtype(dtype)).replace("[", "_").replace("]", "").replace(" ", "_")


@target("Sunmmio")
def matmul_mx_operand_kernel(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    block_K=32,
    a_dtype=T.bfloat16,
    b_dtype=T.bfloat16,
    a_layout_kind="default",
    b_layout_kind="default",
    accum_dtype=T.bfloat16,
):
    shard_policy = T.placement.full_shard(0, 1)
    if _is_mx_dtype(a_dtype):
        A_layout = make_mxzz_layout((M, K), dtype=a_dtype) if a_layout_kind == "mxzz" else make_mx_row_major_layout((M, K), dtype=a_dtype)
    else:
        A_layout = make_zz_layout((M, K), [0, 1], (32, 32))

    if _is_mx_dtype(b_dtype):
        if b_layout_kind == "mxznz":
            B_layout = make_mxznz_layout((K, N), dtype=b_dtype)
        elif b_layout_kind == "mxzz":
            B_layout = make_mxzz_layout((K, N), dtype=b_dtype)
        elif b_layout_kind == "default":
            B_layout = make_mx_row_major_layout((K, N), dtype=b_dtype)
        else:
            raise ValueError(f"Unsupported user-visible MX B layout kind: {b_layout_kind}")
    else:
        B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), shard_policy, a_dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor((K, N), shard_policy, b_dtype, layout=B_layout),  # type: ignore
        C: T.MeshTensor((M, N), shard_policy, accum_dtype, layout=C_layout),  # type: ignore
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape

            A_shared_dist = T.alloc_shared((block_M, block_K * T.ncols()), a_dtype)
            B_shared_dist = T.alloc_shared((block_K * T.nrows(), block_N), b_dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_shared)
                    for k in T.serial(T.ceildiv(sharded_K, block_K)):
                        T.comm.all_gather(
                            A[
                                bx * block_M : (bx + 1) * block_M,
                                k * block_K : (k + 1) * block_K,
                            ],
                            A_shared_dist,
                            direction="horizontal",
                            axis=-1,
                        )
                        T.comm.all_gather(
                            B[
                                k * block_K : (k + 1) * block_K,
                                by * block_N : (by + 1) * block_N,
                            ],
                            B_shared_dist,
                            direction="vertical",
                            axis=0,
                        )
                        T.gemm(A_shared_dist, B_shared_dist, C_shared)

                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return main


@pytest.mark.parametrize("b_dtype,mx_token,n,block_n", MX_DTYPE_CASES)
def test_mx_gemm_only_b_codegen_validates(tmp_path, b_dtype, mx_token, n, block_n):
    src = validate_sunmmio_codegen_with_npuir_opt(
        matmul_mx_operand_kernel(
            K=512,
            N=n,
            block_N=block_n,
            block_K=128,
            a_dtype=T.bfloat16,
            b_dtype=b_dtype,
            b_layout_kind="mxznz",
        ),
        tmp_path,
        mlir_filename=f"{_dtype_filename(b_dtype)}_only_b_matmul_suvm.mlir",
        expected_tokens=(
            mx_token,
            "suvm.mcast_tok",
            "suvm.tc.mma",
            "#suvm.memory_space<asram>",
            "#suvm.memory_space<wsram>",
        ),
    )

    assert "sunmmio.fake" not in src
    assert_source_contains(src, (mx_token, "suvm.mcast_tok", "suvm.tc.mma", "suvm.sync"))


@pytest.mark.parametrize("a_dtype,mx_token,n,block_n", MX_DTYPE_CASES)
def test_mx_gemm_row_major_a_uses_layout_transform_codegen_validates(tmp_path, a_dtype, mx_token, n, block_n):
    src = validate_sunmmio_codegen_with_npuir_opt(
        matmul_mx_operand_kernel(
            K=512,
            N=n,
            block_N=block_n,
            block_K=128,
            a_dtype=a_dtype,
            b_dtype=a_dtype,
            b_layout_kind="mxznz",
        ),
        tmp_path,
        mlir_filename=f"{_dtype_filename(a_dtype)}_row_major_a_matmul_suvm.mlir",
        expected_tokens=(
            mx_token,
            "suvm.transform_layout_async",
            "suvm.mcast_tok",
            "suvm.tc.mma",
            "#suvm.memory_space<asram>",
            "#suvm.memory_space<rsram>",
        ),
    )

    assert "sunmmio.fake" not in src
    assert_source_contains(src, (mx_token, "suvm.transform_layout_async", "suvm.tc.mma"))


@pytest.mark.parametrize("mx_dtype,mx_token,n,block_n", MX_DTYPE_CASES)
def test_mx_gemm_both_a_b_codegen_validates(tmp_path, mx_dtype, mx_token, n, block_n):
    src = validate_sunmmio_codegen_with_npuir_opt(
        matmul_mx_operand_kernel(
            K=512,
            N=n,
            block_N=block_n,
            block_K=128,
            a_dtype=mx_dtype,
            b_dtype=mx_dtype,
            a_layout_kind="mxzz",
            b_layout_kind="mxznz",
        ),
        tmp_path,
        mlir_filename=f"{_dtype_filename(mx_dtype)}_both_a_b_matmul_suvm.mlir",
        expected_tokens=(
            mx_token,
            "suvm.mcast_tok",
            "suvm.tc.mma",
            "suvm.copy_async",
            "#suvm.memory_space<asram>",
            "#suvm.memory_space<wsram>",
        ),
    )

    assert "sunmmio.fake" not in src
    assert_source_contains(src, (mx_token, "suvm.mcast_tok", "suvm.tc.mma", "suvm.sync"))


if __name__ == "__main__":
    tilelang.testing.main()
