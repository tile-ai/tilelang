# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def matmul(
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
    k_pack=1,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    vec_size = 4 * k_pack

    @T.prim_func
    def main(A: T.Tensor(A_shape, in_dtype), B: T.Tensor(B_shape, in_dtype), C: T.Tensor(
        (M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared, coalesced_width=vec_size)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=vec_size)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared, coalesced_width=vec_size)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared, coalesced_width=vec_size)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B, k_pack=k_pack)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
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
    num_stages=0,
    num_threads=128,
    k_pack=1,
):
    program = matmul(
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
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        return (A @ B).to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_rocm
def test_gemm_f16f32f32_nt():
    run_gemm(1024, 1024, 1024, False, False, "float16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, False, True, "float16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, True, "float16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, False, "float16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, False, True, "float16", "float32", "float32", 128, 128, 32, k_pack=2)


@tilelang.testing.requires_rocm
def test_gemm_bf16f32f32_nt():
    run_gemm(1024, 1024, 1024, False, False, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, False, True, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, True, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, False, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm(
        1024, 1024, 1024, False, True, "bfloat16", "float32", "float32", 128, 128, 32, k_pack=2)


@tilelang.testing.requires_rocm
def test_gemm_bf16bf16f32():
    run_gemm(1024, 1024, 1024, False, False, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, False, True, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, True, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, False, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm(
        1024, 1024, 1024, False, True, "bfloat16", "bfloat16", "float32", 128, 128, 32, k_pack=2)


def matmul_rs(
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
    k_pack=1,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    vec_size = 4 * k_pack

    @T.prim_func
    def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared, coalesced_width=vec_size)
                    T.copy(A_shared, A_local)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=vec_size)
                    T.copy(A_shared, A_local)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared, coalesced_width=vec_size)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared, coalesced_width=vec_size)
                T.gemm(A_local, B_shared, C_local, trans_A, trans_B, k_pack=k_pack)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rs(
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
    num_stages=0,
    num_threads=128,
    k_pack=1,
):
    program = matmul_rs(
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
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        return (A @ B).to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_rocm
def test_gemm_rs_f16f32f32_nt():
    run_gemm_rs(1024, 1024, 1024, False, False, "float16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, False, True, "float16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, True, "float16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, False, "float16", "float32", "float32", 128, 128, 32)


@tilelang.testing.requires_rocm
def test_gemm_rs_bf16f32f32_nt():
    run_gemm_rs(1024, 1024, 1024, False, False, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, False, True, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, True, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, False, "bfloat16", "float32", "float32", 128, 128, 32)


@tilelang.testing.requires_rocm
def test_gemm_rs_bf16bf16f32_nt():
    run_gemm_rs(1024, 1024, 1024, False, False, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, False, True, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, True, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, False, "bfloat16", "bfloat16", "float32", 128, 128, 32)


if __name__ == "__main__":
    tilelang.testing.main()
