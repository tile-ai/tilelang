from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


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
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
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
    )

    kernel = tilelang.compile(program, out_idx=[2])
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if in_dtype == T.float32:
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 meas
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_gemm_f16f16f16_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


def test_gemm_f16f16f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
    )


def test_gemm_bf16bf16f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
    )


def test_gemm_f32f32f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float32,
        T.float32,
        T.float32,
        64,
        128,
        32,
    )


def test_gemm_f16f16f16_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


def test_gemm_f16f16f16_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


def test_gemm_i8i8i32_nt():
    run_gemm(512, 1024, 768, False, True, T.int8, T.int8, T.int32, 128, 128, 64)


def test_gemm_i8i8i32_tn():
    run_gemm(512, 1024, 768, True, False, T.int8, T.int8, T.int32, 128, 128, 64)


def test_gemm_f64f64f64_nt():
    run_gemm(512, 512, 512, False, True, T.float64, T.float64, T.float64, 64, 32, 16)


def test_gemm_f32f32f32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.float32,
        T.float32,
        T.float32,
        64,
        128,
        32,
    )


def test_gemm_f32f32f32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.float32,
        T.float32,
        T.float32,
        64,
        128,
        32,
    )


def test_pad_aligned_f16f16f16_nn():
    run_gemm(
        512 - 8,
        1024 - 32,
        768 - 24,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        256,
        32,
        2,
    )


def test_pad_f16f16f16_nn():
    run_gemm(
        512 - 9,
        1024 - 7,
        768 - 5,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        256,
        32,
        2,
    )


def test_pad_f16f16f32_nn():
    run_gemm(
        512 + 19,
        1024 + 17,
        768 + 15,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        64,
        32,
    )


def matmul_sr(
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

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            B_local = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.copy(B_shared, B_local)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.copy(B_shared, B_local)
                T.gemm(A_shared, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_sr(
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
    num_stages=1,
    num_threads=128,
):
    program = matmul_sr(
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

    kernel = tilelang.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        A = A.to(torch.float)
        B = B.to(torch.float)
        C = torch.matmul(A, B)
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


# WGMMA only supports B in shared
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_gemm_f16f16f16_sr():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


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
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope="shared")
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope="shared")
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
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_local, B_shared, C_local, trans_A, trans_B)
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
    num_stages=1,
    num_threads=128,
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
    )

    kernel = tilelang.compile(program, out_idx=[2])
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


# Register source A operand GMMAs must have K-major A layout.
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_gemm_f16f16f16_rs():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


if __name__ == "__main__":
    tilelang.testing.main()
