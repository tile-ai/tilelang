"""Tests for T.gemm on CPU target (GemmScalar path).

Verifies that T.gemm works correctly with target="c" for various
matrix sizes, block sizes, and transpose combinations.
"""

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang import tvm as tvm
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, trans_A=False, trans_B=False, dtype=T.float32, accum_dtype=T.float32):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_local_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_local_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, dtype),
        B: T.Tensor(B_shape, dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_local = T.alloc_local(A_local_shape, dtype)
            B_local = T.alloc_local(B_local_shape, dtype)
            C_local = T.alloc_local((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.serial(T.ceildiv(K, block_K)):
                if trans_A:
                    T.copy(A[ko * block_K, by * block_M], A_local)
                else:
                    T.copy(A[by * block_M, ko * block_K], A_local)
                if trans_B:
                    T.copy(B[bx * block_N, ko * block_K], B_local)
                else:
                    T.copy(B[ko * block_K, bx * block_N], B_local)
                T.gemm(A_local, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def ref_matmul(A, B, trans_A, trans_B):
    if trans_A:
        A = A.T
    if trans_B:
        B = B.T
    return torch.matmul(A.float(), B.float()).to(A.dtype)


def run_gemm_codegen(M, N, K, block_M, block_N, block_K, trans_A=False, trans_B=False):
    func = matmul(M, N, K, block_M, block_N, block_K, trans_A, trans_B)
    with tvm.target.Target("c"):
        artifact = tilelang.lower(func, target="c", target_host="c")
    code = artifact.kernel_source
    assert code is not None, "Code generation failed"
    assert "matmul" in code or "main" in code, "Generated code missing kernel function"
    return code


def run_gemm_compile(M, N, K, block_M, block_N, block_K, trans_A=False, trans_B=False, dtype=T.float32):
    func = matmul(M, N, K, block_M, block_N, block_K, trans_A, trans_B, dtype=dtype)
    with tvm.target.Target("c"):
        kernel = tilelang.compile(func, out_idx=[2], target="c", target_host="c", execution_backend="cython")

    torch_dtype = torch.__getattribute__(dtype)
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A = torch.randn(A_shape, dtype=torch_dtype)
    B = torch.randn(B_shape, dtype=torch_dtype)

    C = kernel(A, B)
    C_ref = ref_matmul(A, B, trans_A, trans_B)

    tilelang.testing.torch_assert_close(C, C_ref, atol=1e-2, rtol=1e-2)


# --- Codegen tests ---


def test_codegen_basic():
    run_gemm_codegen(128, 128, 128, 64, 64, 64)


def test_codegen_rectangular():
    run_gemm_codegen(256, 512, 128, 64, 64, 64)


def test_codegen_trans_A():
    run_gemm_codegen(128, 128, 128, 64, 64, 64, trans_A=True)


def test_codegen_trans_B():
    run_gemm_codegen(128, 128, 128, 64, 64, 64, trans_B=True)


# --- Compile + correctness tests ---


@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K",
    [
        (128, 128, 128, 64, 64, 64),
        (256, 256, 256, 64, 64, 64),
        (256, 512, 128, 64, 64, 64),
        (512, 512, 512, 128, 128, 128),
    ],
)
def test_gemm_f32_nn(M, N, K, block_M, block_N, block_K):
    run_gemm_compile(M, N, K, block_M, block_N, block_K)


def test_gemm_f32_tn():
    run_gemm_compile(128, 128, 128, 64, 64, 64, trans_A=True)


def test_gemm_f32_nt():
    run_gemm_compile(128, 128, 128, 64, 64, 64, trans_B=True)


def test_gemm_f32_tt():
    run_gemm_compile(128, 128, 128, 64, 64, 64, trans_A=True, trans_B=True)


if __name__ == "__main__":
    tilelang.testing.main()
