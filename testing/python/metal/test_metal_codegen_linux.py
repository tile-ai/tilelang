"""Test Metal code generation on any platform (including Linux).

These tests verify that TileLang can compile kernels down to Metal shader
source code without requiring a Metal runtime or macOS.
"""

import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float32, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                for i, j in T.Parallel(block_M, block_N):
                    for k in T.Serial(block_K):
                        C_local[i, j] += A_shared[i, k] * B_shared[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return main


def matmul_with_t_gemm(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float32,
    accum_dtype=T.float32,
    transpose_B=False,
    num_stages=0,
    threads=128,
):
    B_shape = (N, K) if transpose_B else (K, N)
    B_shared_shape = (block_N, block_K) if transpose_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor(B_shape, dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared(B_shared_shape, dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                if transpose_B:
                    T.copy(B[bx * block_N, ko * block_K], B_shared, coalesced_width=2)
                else:
                    T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                T.gemm(A_shared, B_shared, C_local, transpose_B=transpose_B)

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return main


def attention_like_with_t_gemm(
    block_M,
    block_N,
    head_dim,
    dtype=T.float32,
    accum_dtype=T.float32,
):
    @T.prim_func
    def main(
        Q: T.Tensor((block_M, head_dim), dtype),
        K: T.Tensor((block_N, head_dim), dtype),
        V: T.Tensor((block_N, head_dim), dtype),
        O: T.Tensor((block_M, head_dim), dtype),
    ):
        with T.Kernel(1, threads=128):
            Q_shared = T.alloc_shared((block_M, head_dim), dtype)
            K_shared = T.alloc_shared((block_N, head_dim), dtype)
            V_shared = T.alloc_shared((block_N, head_dim), dtype)
            scores = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc = T.alloc_fragment((block_M, head_dim), accum_dtype)
            row_max = T.alloc_fragment((block_M,), accum_dtype)
            row_sum = T.alloc_fragment((block_M,), accum_dtype)

            T.copy(Q, Q_shared)
            T.copy(K, K_shared)
            T.copy(V, V_shared)
            T.clear(scores)
            T.clear(acc)
            T.gemm(Q_shared, K_shared, scores, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                scores[i, j] = T.if_then_else(i >= j, scores[i, j], -T.infinity(accum_dtype))

            T.reduce_max(scores, row_max, dim=1, clear=True)
            for i, j in T.Parallel(block_M, block_N):
                scores[i, j] = T.exp2(scores[i, j] - row_max[i])

            T.reduce_sum(scores, row_sum, dim=1, clear=True)
            T.gemm(scores, V_shared, acc)

            for i, d in T.Parallel(block_M, head_dim):
                O[i, d] = acc[i, d] / row_sum[i]

    return main


def mixed_dynamic_static_shared_merge():
    @T.prim_func
    def main(
        A: T.Tensor((32,), T.float32),
        B: T.Tensor((32,), T.float32),
        C: T.Tensor((32,), T.float32),
    ):
        with T.Kernel(1, threads=128):
            dyn = T.alloc_shared((32,), T.float32, scope="shared.dyn")
            stat_a = T.alloc_shared((16,), T.float32, scope="shared")
            stat_b = T.alloc_shared((16,), T.float32, scope="shared")

            for i in T.Parallel(32):
                dyn[i] = A[i]
            for i in T.Parallel(16):
                stat_a[i] = dyn[i] + B[i]
                stat_b[i] = dyn[i + 16] + B[i + 16]
            for i in T.Parallel(16):
                C[i] = stat_a[i]
                C[i + 16] = stat_b[i]

    return main


def single_dynamic_shared_no_merge():
    @T.prim_func
    def main(
        A: T.Tensor((32,), T.float32),
        C: T.Tensor((32,), T.float32),
    ):
        with T.Kernel(1, threads=32):
            dyn = T.alloc_shared((32,), T.float32, scope="shared.dyn")

            for i in T.Parallel(32):
                dyn[i] = A[i]
            for i in T.Parallel(32):
                C[i] = dyn[i]

    return main


def assert_metal_codegen(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float32,
    accum_dtype=T.float32,
):
    func = matmul(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")

    src_code = artifact.kernel_source
    assert src_code is not None
    assert "metal" in src_code or "kernel void" in src_code


def assert_attention_like_metal_codegen(
    block_M,
    block_N,
    head_dim,
    dtype=T.float32,
    accum_dtype=T.float32,
):
    func = attention_like_with_t_gemm(
        block_M,
        block_N,
        head_dim,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
    with tvm.transform.PassContext():
        artifact = tilelang.lower(func, target="metal")

    src_code = artifact.kernel_source
    assert src_code is not None
    assert "threadgroup" in src_code
    assert "kernel void" in src_code


def assert_t_gemm_metal_codegen(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float32,
    accum_dtype=T.float32,
    transpose_B=False,
    num_stages=0,
    threads=128,
):
    func = matmul_with_t_gemm(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
        transpose_B=transpose_B,
        num_stages=num_stages,
        threads=threads,
    )
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")

    src_code = artifact.kernel_source
    assert src_code is not None
    assert "metal" in src_code or "kernel void" in src_code
    assert "threadIdx.x) == 0" not in src_code


def assert_mixed_shared_merge_metal_codegen():
    func = mixed_dynamic_static_shared_merge()
    with tvm.transform.PassContext(config={"tir.merge_static_smem": True}), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")

    src_code = artifact.kernel_source
    assert src_code is not None
    assert "kernel void" in src_code
    assert "buf_dyn_shmem[32]" in src_code


def assert_single_dynamic_shared_no_merge_metal_codegen():
    func = single_dynamic_shared_no_merge()
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")

    src_code = artifact.kernel_source
    assert src_code is not None
    assert "kernel void" in src_code
    assert "buf_dyn_shmem[32]" in src_code
    assert "buf_dyn_shmem[128]" not in src_code


def test_metal_codegen_float32():
    assert_metal_codegen(1024, 1024, 1024, 16, 16, 16)


def test_metal_codegen_float16():
    assert_metal_codegen(1024, 1024, 1024, 16, 16, 16, dtype=T.float16)


def test_metal_codegen_int32():
    assert_metal_codegen(1024, 1024, 1024, 16, 16, 16, dtype=T.int32)


def test_t_gemm_metal_codegen_float32():
    assert_t_gemm_metal_codegen(128, 128, 128, 16, 16, 16)


def test_t_gemm_metal_codegen_float16_accum_float32():
    assert_t_gemm_metal_codegen(64, 64, 64, 16, 16, 16, dtype=T.float16, accum_dtype=T.float32)


def test_t_gemm_metal_codegen_transpose_b_float32():
    assert_t_gemm_metal_codegen(128, 128, 128, 16, 16, 16, transpose_B=True)


def test_t_gemm_metal_codegen_pipelined_float32():
    assert_t_gemm_metal_codegen(64, 64, 64, 16, 16, 16, num_stages=2)


def test_t_gemm_metal_codegen_single_thread_float32():
    assert_t_gemm_metal_codegen(16, 16, 16, 16, 16, 16, threads=1)


def test_t_gemm_attention_like_metal_codegen_float32():
    assert_attention_like_metal_codegen(16, 16, 16)


def test_mixed_shared_merge_metal_codegen():
    assert_mixed_shared_merge_metal_codegen()


def test_single_dynamic_shared_no_merge_metal_codegen():
    assert_single_dynamic_shared_no_merge_metal_codegen()


if __name__ == "__main__":
    tilelang.testing.main()
