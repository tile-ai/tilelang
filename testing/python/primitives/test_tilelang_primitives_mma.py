# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
from tilelang import primitives as P


def matmul_ssr(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    shared_scope = "shared"  # or "shared.dyn" for dynamic shared memory
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, in_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[ko * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                else:
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                P.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_matmul_ssr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul_ssr(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)
    print(mod.get_kernel_source())

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    mod.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_gemm_f16f16f16_nt_ssr():
    run_matmul_ssr(
        16, 16, 16, False, True, "float16", "float16", "float16", 16, 16, 16, 0, num_threads=32)
    run_matmul_ssr(
        128, 128, 128, False, True, "float16", "float16", "float16", 32, 32, 32, 0, num_threads=64)
    run_matmul_ssr(
        1024,
        1024,
        1024,
        False,
        True,
        "float16",
        "float16",
        "float16",
        128,
        128,
        32,
        2,
        num_threads=128)


def matmul_rsr(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    A_local_shape = A_shared_shape
    shared_scope = "shared"  # or "shared.dyn" for dynamic shared memory
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, in_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            A_local = T.alloc_fragment(A_local_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[ko * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                else:
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.copy(A_shared, A_local)
                P.gemm(A_local, B_shared, C_local, trans_A, trans_B)
                # T.gemm(A_local, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_matmul_rsr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul_rsr(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)
    print(mod.get_kernel_source())

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    mod.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


# TODO(lei): Fix the test case in future release
# Now it has some bugs related to is_m_first
# def test_gemm_f16f16f16_nt_rsr():
#     run_matmul_rsr(
#         1024,
#         1024,
#         1024,
#         False,
#         True,
#         "float16",
#         "float16",
#         "float16",
#         128,
#         128,
#         32,
#         0,
#         num_threads=128,
#     )


def matmul_rrr(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    A_local_shape = A_shared_shape
    B_local_shape = B_shared_shape
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, in_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            A_local = T.alloc_fragment(A_local_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            B_local = T.alloc_fragment(B_local_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                    T.copy(A_shared, A_local)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(A_shared, A_local)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.copy(B_shared, B_local)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.copy(B_shared, B_local)
                P.gemm(A_local, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_matmul_rrr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul_rrr(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    mod.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


# def test_gemm_f16f16f16_nt_rrr():
#     run_matmul_rrr(
#         1024,
#         1024,
#         1024,
#         False,
#         True,
#         "float16",
#         "float16",
#         "float16",
#         128,
#         128,
#         32,
#         2,
#     )

if __name__ == "__main__":
    # tilelang.testing.main()
    run_matmul_rsr(
        128,
        128,
        128,
        False,
        True,
        "float16",
        "float16",
        "float16",
        128,
        128,
        32,
        0,
        num_threads=128,
    )
