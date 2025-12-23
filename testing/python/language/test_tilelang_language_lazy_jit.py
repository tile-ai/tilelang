import tilelang.testing
import tilelang
import tilelang.language as T
from itertools import product
import torch


def test_jit2_gemm():
    @tilelang.lazy_jit(verbose=True)
    def gemm(
        A,
        B,
        C,
        dtype: T.dtype = T.float16,
        accum_dtype: T.dtype = T.float32,
        block_M: int = 64,
        block_N: int = 64,
        block_K: int = 64,
    ):
        M, N, K = T.const("M N K")

        A: T.Tensor[[M, K], dtype]
        B: T.Tensor[[K, N], dtype]
        C: T.Tensor[[M, N], dtype]

        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N)) as (by, bx):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    A = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    B = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    C = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    gemm(A, B, C)
    C_ref = A @ B
    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=1e-2)


def test_jit2_gemm_ptr():
    @tilelang.lazy_jit
    def gemm_ptr(
        A: T.ptr,
        B: T.ptr,
        C: T.ptr,
        M: int,
        N: int,
        K: int,
        dtype: T.dtype,
        out_dtype: T.dtype,
        block_M: int = 64,
        block_N: int = 64,
        block_K: int = 32,
    ):
        A = T.make_tensor(A, (M, K), dtype)
        B = T.make_tensor(B, (K, N), dtype)
        C = T.make_tensor(C, (M, N), out_dtype)
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * block_M, by * block_N])

    prod = product([T.float16, T.float32], [T.float32])
    gemm_ptr.par_compile(
        [
            {"A": T.ptr(), "B": T.ptr(), "C": T.ptr(), "M": 1024, "N": 1024, "K": 1024, "dtype": in_dtype, "out_dtype": out_dtype}
            for in_dtype, out_dtype in prod
        ]
    )
    for in_dtype, out_dtype in prod:
        in_dtype = in_dtype.as_torch()
        out_dtype = out_dtype.as_torch()
        A = torch.randn(1024, 1024, dtype=in_dtype, device="cuda")
        B = torch.randn(1024, 1024, dtype=in_dtype, device="cuda")
        C_ref = out_dtype(A @ B)
        C = torch.empty(1024, 1024, dtype=out_dtype, device="cuda")
        gemm_ptr(A, B, C, 1024, 1024, 1024, in_dtype, out_dtype)
        torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=1e-2)


def test_jit2_many_annot():
    @T.macro
    def copy_impl(A, B):
        M, N = A.shape
        M_, N_ = B.shape
        assert M == M_, f"M mismatch {M} {M_}"
        assert N == N_, f"N mismatch {N} {N_}"
        # assert tuple(A.shape) == tuple(B.shape), f"Invalid tensor shape: {A.shape}, {B.shape}"
        with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
            T.copy(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128], B[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])

    @tilelang.lazy_jit
    def copy1(A, B):
        N, M = T.const("N, M")
        A: T.Tensor[[N, M], T.float32]
        B: T.Tensor[[N, M], T.float32]
        copy_impl(A, B)

    @tilelang.lazy_jit
    def copy2(
        A: T.Tensor[[128, 128], T.float32],
        B: T.Tensor[[128, 128], T.float32],
    ):
        copy_impl(A, B)

    @tilelang.lazy_jit
    def copy3(A, B):
        N = T.const("N")
        A: T.Tensor[[N, 128], T.float32]
        B: T.Tensor[[N, 128], T.float32]
        copy_impl(A, B)

    @tilelang.lazy_jit
    def copy4(A, B):
        N = T.dynamic("N")
        M = T.const("M")
        A: T.Tensor[[N, M], T.float32]
        B: T.Tensor[[N, M], T.float32]
        copy_impl(A, B)

    @tilelang.lazy_jit
    def copy5(A, B):
        N, M, N_, M_ = T.const("N, M, N_, M_")
        A: T.StridedTensor[[N, M], [N_, M_], T.float32]
        B: T.StridedTensor[[N, M], [N_, M_], T.float32]
        copy_impl(A, B)

    @tilelang.lazy_jit
    def copy6(A, B):
        N = T.dynamic("N")
        M, N_, M_ = T.const("M, N_, M_")
        A: T.StridedTensor[[N, M], [N_, M_], T.float32]
        B: T.StridedTensor[[N, M], [N_, M_], T.float32]
        copy_impl(A, B)

    tilelang.par_compile([copy.get_tir(T.Tensor((128, 128))) for copy in [copy1, copy2, copy3, copy4]])

    for copy in [copy1, copy2, copy3, copy4]:
        A = torch.randn(128, 128, device="cuda")
        B = torch.empty(128, 128, device="cuda")
        copy(A, B)
        assert torch.equal(B, A)

    for copy in [copy5, copy6]:
        A = torch.randn(128, 2, 128, 2, device="cuda")
        B = torch.randn(128, 2, 128, 2, device="cuda")
        copy(A[:, 0, :, 0], B[:, 0, :, 0])
        assert torch.equal(A[:, 0, :, 0], B[:, 0, :, 0])


def test_jit2_return():
    @T.macro
    def copy_impl(A):
        M, N = A.shape
        B = T.empty(M, N, dtype=A.dtype)
        M, N = A.shape
        M_, N_ = B.shape
        assert M == M_, f"M mismatch {M} {M_}"
        assert N == N_, f"N mismatch {N} {N_}"
        # assert tuple(A.shape) == tuple(B.shape), f"Invalid tensor shape: {A.shape}, {B.shape}"
        with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
            T.copy(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128], B[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
        return B

    @tilelang.lazy_jit
    def copy1(A):
        M, N = T.const("M, N")
        A: T.Tensor[[M, N], T.float32]
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy2(A):
        A: T.Tensor[[128, 128], T.float32]
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy3(A):
        N = T.const("N")
        A: T.Tensor[[N, 128], T.float32]
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy4(A):
        N = T.dynamic("N")
        M = T.const("M")
        A: T.Tensor[[N, M], T.float32]
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy5(A):
        N, M, N_, M_ = T.const("N, M, N_, M_")
        A: T.StridedTensor[[N, M], [N_, M_], T.float32]
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy6(A):
        N = T.dynamic("N")
        M, N_, M_ = T.const("M, N_, M_")
        A: T.StridedTensor[[N, M], [N_, M_], T.float32]
        return copy_impl(A)

    tilelang.par_compile([copy.get_tir(T.Tensor((128, 128))) for copy in [copy1, copy2, copy3, copy4]])

    for copy in [copy1, copy2, copy3, copy4]:
        A = torch.randn(128, 128, device="cuda")
        B = copy(A)
        assert torch.equal(B, A)

    for copy in [copy5, copy6]:
        A = torch.randn(128, 2, 128, 2, device="cuda")
        B = copy(A[:, 0, :, 0])
        assert torch.equal(A[:, 0, :, 0], B)


if __name__ == "__main__":
    tilelang.testing.main()
