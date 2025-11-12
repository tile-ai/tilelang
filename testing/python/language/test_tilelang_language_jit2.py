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
    @tilelang.jit2
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

    prod = product(
        [T.float16, T.float32],
        [T.float32]
    )
    gemm.par_compile([
        {
            'A': T.Tensor((1024, 1024), dtype=in_dtype),
            'B': T.Tensor((1024, 1024), dtype=in_dtype),
            'out_dtype': out_dtype
        }
        for in_dtype, out_dtype in prod
    ])

    for in_dtype, out_dtype in prod:
        in_dtype = in_dtype.torch()
        out_dtype = out_dtype.torch()
        A = torch.randn(1024, 1024, dtype=in_dtype, device='cuda')
        B = torch.randn(1024, 1024, dtype=in_dtype, device='cuda')
        C_ref = out_dtype(A @ B)
        C = gemm(A, B)
        torch.testing.assert_close(C, C_ref, rtol=1e-2, atol=1e-2)


def test_jit2_gemm_ptr():
    @tilelang.jit2
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

    prod = product(
        [T.float16, T.float32],
        [T.float32]
    )
    gemm_ptr.par_compile([
        {
            'A': T.ptr(),
            'B': T.ptr(),
            'C': T.ptr(),
            'M': 1024,
            'N': 1024,
            'K': 1024,
            'dtype': in_dtype,
            'out_dtype': out_dtype
        }
        for in_dtype, out_dtype in prod
    ])
    for in_dtype, out_dtype in prod:
        in_dtype = in_dtype.torch()
        out_dtype = out_dtype.torch()
        A = torch.randn(1024, 1024, dtype=in_dtype, device='cuda')
        B = torch.randn(1024, 1024, dtype=in_dtype, device='cuda')
        C_ref = out_dtype(A @ B)
        C = torch.empty(1024, 1024, dtype=out_dtype, device='cuda')
        gemm_ptr(A, B, C, 1024, 1024, 1024, in_dtype, out_dtype)
        torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=1e-2)


def test_jit2_annot():
    from tilelang.language.v2.annot import Annot, ArgVarTable
    from tilelang.language.v2.builder import Builder

    @dataclass
    class AnnotTest:
        annot: Annot
        promote: Any
        match_ok: list[Any] = field(default_factory=list)
        match_ng: list[Any] = field(default_factory=list)

    tests = [
        AnnotTest(
            annot = T.Tensor[[int, int], T.float32],
            promote = False,
            match_ok = [
                torch.randn(1, 1, dtype=torch.float32),
                T.Tensor((1, 1), dtype=T.float32)
            ],
            match_ng = [
                torch.randn(1, 1, dtype=torch.float16),
                T.Tensor(1, dtype=T.float32),
                T.Tensor((1, 1), dtype=T.float16),
            ],
        ),
        AnnotTest(
            annot = T.Tensor[[int], Any],
            promote=False,
            match_ok = [
                torch.randn(12, dtype=torch.float32),
                torch.randn(12, dtype=torch.float16),
                T.Tensor((1,), dtype=T.float32),
                T.Tensor((1,), dtype=T.float16),
            ],
            match_ng = [
                torch.randn((1, 1), dtype=torch.float32),
                T.Tensor((1, 1), dtype=T.float16)
            ]
        ),
        AnnotTest(
            annot = T.Tensor[[int, 1], Any],
            promote=False,
            match_ok = [
                torch.randn(12, 1, dtype=torch.float32),
                torch.randn(12, 1, dtype=torch.float16),
                T.Tensor((12, 1), T.float32),
                T.Tensor((12, 1), T.float16),
            ],
            match_ng = [
                torch.randn(12, 12, dtype=torch.float32),
                T.Tensor((12, 12), T.float32)
            ]
        ),
        AnnotTest(
            annot = T.Tensor[[T.dyn, 1], Any],
            promote = False,
            match_ok = [
                torch.randn(12, 1, dtype=torch.float32),
                torch.randn(12, 1, dtype=torch.float16),
                T.Tensor((12, 1), T.float32),
                T.Tensor((12, 1), T.float16),
            ],
            match_ng = [
                torch.randn(12, 12, dtype=torch.float32),
                T.Tensor((12, 12), T.float32)
            ]
        ),
        AnnotTest(
            annot = T.Tensor[[1024, 1024], Any],
            promote=True,
        ),
        AnnotTest(
            annot = T.dyn[int],
            promote = False,
            match_ok = [1, 2, 3, 4]
        ),
        AnnotTest(
            annot = T.dyn,
            promote = False,
            match_ok = [1, 2, 3, 4]
        )
    ]

    for test in tests:
        promote = test.annot.promote()
        promoted = promote is not None
        if promoted != test.promote:
            raise AssertionError(f'Promote mismatch for {test.annot}: expected {test.promote}, got {promoted}')
        with Builder().prim_func('_test'):
            for match_ok in test.match_ok:
                try:
                    vt = ArgVarTable()
                    test.annot.create_prim_func_arg('arg', match_ok, vt)
                except Exception as e:
                    raise AssertionError(f'Match failed for {test.annot} with value {match_ok}: {e}') from e
            for match_ng in test.match_ng:
                try:
                    vt = ArgVarTable()
                    test.annot.create_prim_func_arg('arg', match_ng, vt)
                    raise AssertionError(f'Match unexpectedly succeeded for {test.annot} with value {match_ng}')
                except Exception as e:
                    pass


if __name__ == '__main__':
    tilelang.testing.main()
