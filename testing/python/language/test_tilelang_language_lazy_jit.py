from dataclasses import dataclass, field
import tilelang.testing
import tilelang
import tilelang.language as T
from typing import Any
from itertools import product
import torch


def _gemm_impl():
    @T.macro
    def gemm_impl(
        A: T.Tensor[[int, int], Any],
        B: T.Tensor[[int, int], Any],
        C: T.Tensor[[int, int], Any],
        out_dtype: T.dtype,
        block_M: int,
        block_N: int,
        block_K: int,
    ):
        dtype = A.dtype
        M, K = A.shape
        K, N = B.shape
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

    return gemm_impl


def test_jit2_gemm_annot():
    @tilelang.lazy_jit
    def gemm(
        A: T.Tensor[[int, int], Any],
        B: T.Tensor[[int, int], Any],
        out_dtype: T.dtype = T.float32,
        block_M: int = 64,
        block_N: int = 64,
        block_K: int = 32,
    ):
        M, K = A.shape
        K, N = B.shape
        C = T.empty(M, N, dtype=out_dtype)
        _gemm_impl()(A, B, C, out_dtype, block_M, block_N, block_K)
        return C

    prod = product([T.float16, T.float32], [T.float32])
    gemm.par_compile(
        [
            {"A": T.Tensor((1024, 1024), dtype=in_dtype), "B": T.Tensor((1024, 1024), dtype=in_dtype), "out_dtype": out_dtype}
            for in_dtype, out_dtype in prod
        ]
    )

    for in_dtype, out_dtype in prod:
        in_dtype = in_dtype.as_torch()
        out_dtype = out_dtype.as_torch()
        A = torch.randn(1024, 1024, dtype=in_dtype, device="cuda")
        B = torch.randn(1024, 1024, dtype=in_dtype, device="cuda")
        C_ref = out_dtype(A @ B)
        C = gemm(A, B)
        torch.testing.assert_close(C, C_ref, rtol=1e-2, atol=1e-2)


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
        _gemm_impl()(A, B, C, out_dtype, block_M, block_N, block_K)

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


def test_jit2_annot():
    from tilelang.language.v2.annot import Annot, ArgVarTable
    from tilelang.language.v2.builder import Builder
    import traceback

    @dataclass
    class AnnotTest:
        annot: Annot
        promote: Any
        match_ok: list[Any] = field(default_factory=list)
        match_ng: list[Any] = field(default_factory=list)

    tests = [
        AnnotTest(
            annot=T.Tensor[[int, int], T.float32],
            promote=False,
            match_ok=[torch.randn(1, 1, dtype=torch.float32), T.Tensor((1, 1), dtype=T.float32)],
            match_ng=[
                torch.randn(1, 1, dtype=torch.float16),
                T.Tensor(1, dtype=T.float32),
                T.Tensor((1, 1), dtype=T.float16),
            ],
        ),
        AnnotTest(
            annot=T.Tensor[[int], Any],
            promote=False,
            match_ok=[
                torch.randn(12, dtype=torch.float32),
                torch.randn(12, dtype=torch.float16),
                T.Tensor((1,), dtype=T.float32),
                T.Tensor((1,), dtype=T.float16),
            ],
            match_ng=[torch.randn((1, 1), dtype=torch.float32), T.Tensor((1, 1), dtype=T.float16)],
        ),
        AnnotTest(
            annot=T.Tensor[[int, 1], Any],
            promote=False,
            match_ok=[
                torch.randn(12, 1, dtype=torch.float32),
                torch.randn(12, 1, dtype=torch.float16),
                T.Tensor((12, 1), T.float32),
                T.Tensor((12, 1), T.float16),
            ],
            match_ng=[torch.randn(12, 12, dtype=torch.float32), T.Tensor((12, 12), T.float32)],
        ),
        AnnotTest(
            annot=T.Tensor[[T.dyn, 1], Any],
            promote=False,
            match_ok=[
                torch.randn(12, 1, dtype=torch.float32),
                torch.randn(12, 1, dtype=torch.float16),
                T.Tensor((12, 1), T.float32),
                T.Tensor((12, 1), T.float16),
            ],
            match_ng=[torch.randn(12, 12, dtype=torch.float32), T.Tensor((12, 12), T.float32)],
        ),
        AnnotTest(
            annot=T.Tensor[[1024, 1024], T.float32],
            promote=True,
        ),
        AnnotTest(annot=T.dyn[int, "X"], promote=False, match_ok=[1, 2, 3, 4]),
        AnnotTest(annot=T.dyn, promote=False, match_ok=[1, 2, 3, 4]),
    ]

    for test in tests:
        promote = test.annot.promote()
        promoted = promote is not None
        if promoted != test.promote:
            raise AssertionError(f"Promote mismatch for {test.annot}: expected {test.promote}, got {promoted}")
        with Builder().prim_func("_test"):
            for match_ok in test.match_ok:
                try:
                    vt = ArgVarTable()
                    test.annot.create_prim_func_arg("arg", match_ok, vt)
                except Exception as e:
                    traceback.print_exc()
                    raise AssertionError(f"Match failed for {test.annot} with value {match_ok}: {e}") from e
            for match_ng in test.match_ng:
                try:
                    vt = ArgVarTable()
                    test.annot.create_prim_func_arg("arg", match_ng, vt)
                    raise AssertionError(f"Match unexpectedly succeeded for {test.annot} with value {match_ng}")
                except Exception:
                    pass


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
    def copy1(
        A: T.Tensor[[int, int], T.float32],
        B: T.Tensor[[int, int], T.float32],
    ):
        copy_impl(A, B)

    @tilelang.lazy_jit
    def copy2(
        A: T.Tensor[[128, 128], T.float32],
        B: T.Tensor[[128, 128], T.float32],
    ):
        copy_impl(A, B)

    @tilelang.lazy_jit
    def copy3(
        A: T.Tensor[[int, 128], T.float32],
        B: T.Tensor[[int, 128], T.float32],
    ):
        copy_impl(A, B)

    @tilelang.lazy_jit
    def copy4(
        A: T.Tensor[[T.dyn, int], T.float32],
        B: T.Tensor[[T.dyn, int], T.float32],
    ):
        copy_impl(A, B)

    @tilelang.lazy_jit
    def copy5(
        A: T.StridedTensor[[int, int], [int, int], T.float32],
        B: T.StridedTensor[[int, int], [int, int], T.float32],
    ):
        copy_impl(A, B)

    @tilelang.lazy_jit
    def copy6(
        A: T.StridedTensor[[T.dyn, int], [int, int], T.float32],
        B: T.StridedTensor[[T.dyn, int], [int, int], T.float32],
    ):
        copy_impl(A, B)

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
    def copy0(A: T.Tensor[[int, int], Any]):
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy1(
        A: T.Tensor[[int, int], T.float32],
    ):
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy2(
        A: T.Tensor[[128, 128], T.float32],
    ):
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy3(
        A: T.Tensor[[int, 128], T.float32],
    ):
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy4(
        A: T.Tensor[[T.dyn, int], T.float32],
    ):
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy5(
        A: T.StridedTensor[[int, int], [int, int], T.float32],
    ):
        return copy_impl(A)

    @tilelang.lazy_jit
    def copy6(
        A: T.StridedTensor[[T.dyn, int], [int, int], T.float32],
    ):
        return copy_impl(A)

    for copy in [copy0, copy1, copy2, copy3, copy4]:
        A = torch.randn(128, 128, device="cuda")
        B = copy(A)
        assert torch.equal(B, A)
    for copy in [copy5, copy6]:
        A = torch.randn(128, 2, 128, 2, device="cuda")
        B = copy(A[:, 0, :, 0])
        assert torch.equal(A[:, 0, :, 0], B)


def test_jit2_deepseek_deepgemm():
    @tilelang.lazy_jit
    def deep_gemm(
        A: T.Tensor[[int, int], T.float8_e4m3fn],
        B: T.Tensor[[int, int], T.float8_e4m3fn],
        scales_a: T.Tensor[[int, int], T.float32],
        scales_b: T.Tensor[[int, int], T.float32],
        out_dtype: T.dtype = T.bfloat16,
        accum_dtype: T.dtype = T.float32,
        block_N: int = 128,
        block_M: int = 128,
        block_K: int = 128,
    ):
        # A: [M, K]
        # B: [N, K]
        # scales_a: [M, K // 128]
        # scales_b: [N, K // 128]
        # C: [M, N]

        group_size = 128
        in_dtype = A.dtype
        M, K = A.shape
        N, K = B.shape
        C = T.empty(M, N, dtype=out_dtype)

        assert out_dtype in [T.bfloat16, T.float32], f"Expect out_dtype to be one of [T.float16, T.float32], got {out_dtype}"
        assert scales_a.shape == [M, T.ceildiv(K, group_size)], f"Expect scales_a shape to be f{[M, T.ceildiv(K, group_size)]}"
        assert scales_b.shape == [N, T.ceildiv(K, group_size)], f"Expect scales_b shape to be f{[N, T.ceildiv(K, group_size)]}"

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            scale_C_shared = T.alloc_shared((block_M,), T.float32)
            C_local = T.alloc_fragment((block_M, block_K), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                Scale_B = scales_b[bx * block_N // group_size, k]
                for i in T.Parallel(block_M):
                    scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * scale_C_shared[i]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

        return C


#     def ceildiv(a, b):
#         return (a + b - 1) // b

#     def ref_deepgemm_fp8(A_fp8, B_fp8, A_scale, B_scale, out_dtype):
#         # A_scale: (M, K//128)       ==>   (M//128, K//128, 128)
#         # B_scale: (N//128, K//128)  ==>   (N//128, K//128, 128)
#         # A_fp8: (M, K)
#         # B_fp8: (N, K)
#         # out_dtype: float16 or float32
#         # return C: (M, N)
#         M, N, K = A_fp8.shape[0], B_fp8.shape[0], A_fp8.shape[1]
#         A_scales = A_scale.view(M // 128, 128, K // 128).permute(0, 2, 1)
#         B_scales = B_scale.repeat_interleave(128, dim=1).view(N // 128, K // 128, 128)
#         C = torch.zeros(M, N, device="cuda", dtype=out_dtype)
#         c_acc = torch.zeros(128, 128, device="cuda", dtype=torch.float32)
#         for i in range(ceildiv(M, 128)):
#             for j in range(ceildiv(N, 128)):
#                 c_acc.zero_()
#                 for k in range(ceildiv(K, 128)):
#                     c = torch._scaled_mm(
#                         A_fp8[i * 128:(i + 1) * 128, k * 128:(k + 1) * 128],
#                         B_fp8[j * 128:(j + 1) * 128, k * 128:(k + 1) * 128].T,
#                         scale_a=A_scales[i, k].view(128, 1).contiguous(),
#                         scale_b=B_scales[j, k].view(1, 128).contiguous(),
#                         out_dtype=torch.bfloat16)
#                     c_acc += c.to(torch.float32)
#                 C[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = c_acc.to(out_dtype)
#         return C

#     M, N, K = 1024, 1024, 8192
#     A = torch.randn((M, K), dtype=torch.float8_e4m3fn, )

if __name__ == "__main__":
    tilelang.testing.main()
