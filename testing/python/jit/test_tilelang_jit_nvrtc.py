from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing
import tilelang
import torch
from tilelang.utils.tensor import map_torch_type


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
    num_stages=3,
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

    stramp = "&*(XS)"

    @tvm.register_global_func("tilelang_callback_cuda_postproc", override=True)
    def tilelang_callback_cuda_postproc(code, _):
        code = f"// {stramp}\n" + code
        return code

    matmul_kernel = tilelang.compile(program, out_idx=-1, execution_backend="nvrtc")

    kernel_source = matmul_kernel.get_kernel_source()

    assert stramp in kernel_source, f"Expected {stramp} in the kernel source"


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
        256,
        32,
        2,
    )


def matmu_jit_kernel(
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


def run_gemm_jit_kernel(
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
    program = matmu_jit_kernel(
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

    matmul_kernel = tilelang.compile(program, out_idx=-1, execution_backend="nvrtc")

    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)

    A = torch.randn(M, K, dtype=in_dtype).cuda()
    B = torch.randn(K, N, dtype=in_dtype).cuda()

    if trans_A:
        A = A.T
    if trans_B:
        B = B.T

    def ref_program(A, B):
        import torch

        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(out_dtype)
        return C

    ref_C = ref_program(A, B)
    C = matmul_kernel(A, B)

    tilelang.testing.torch_assert_close(C, ref_C, atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.05)


def test_gemm_jit_kernel():
    run_gemm_jit_kernel(
        512,
        1024,
        768,
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


def run_nvrtc_kernel_do_bench(
    M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages=3, num_threads=128
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

    matmul_kernel = tilelang.compile(program, execution_backend="nvrtc")

    profiler = matmul_kernel.get_profiler()

    nvrtc_latency = profiler.do_bench(func=matmul_kernel)
    print(f"NVRTC Latency: {nvrtc_latency} ms")

    assert nvrtc_latency is not None

    tvm_latency = profiler.do_bench()
    print(f"TVM Latency: {tvm_latency} ms")

    assert tvm_latency is not None


def test_nvrtc_kernel_do_bench():
    run_nvrtc_kernel_do_bench(512, 1024, 768, False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)


def run_nvrtc_kernel_multi_stream(
    M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages=3, num_threads=128
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

    matmul_kernel = tilelang.compile(program, execution_backend="nvrtc")
    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)
    tensor_a = torch.randn(M, K, dtype=in_dtype).cuda()
    tensor_b = torch.randn(K, N, dtype=in_dtype).cuda()

    if trans_A:
        tensor_a = tensor_a.T
    if trans_B:
        tensor_b = tensor_b.T
    tensor_c = torch.randn(M, N, dtype=out_dtype).cuda()

    num_streams = 4
    for _ in range(num_streams):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            matmul_kernel(tensor_a, tensor_b, tensor_c)


def test_nvrtc_kernel_multi_stream():
    run_nvrtc_kernel_multi_stream(512, 1024, 768, False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)


def run_nvrtc_dynamic_shape(
    M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages=3, num_threads=128
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

    matmul_kernel = tilelang.compile(program, execution_backend="nvrtc")
    if isinstance(M, T.Var):
        M = 1024
    if isinstance(N, T.Var):
        N = 1024
    if isinstance(K, T.Var):
        K = 768

    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)

    tensor_a = torch.randn(M, K, dtype=in_dtype).cuda()
    tensor_b = torch.randn(K, N, dtype=in_dtype).cuda()

    if trans_A:
        tensor_a = tensor_a.T
    if trans_B:
        tensor_b = tensor_b.T
    tensor_c = torch.randn(M, N, dtype=out_dtype).cuda()

    matmul_kernel(tensor_a, tensor_b, tensor_c)

    tensor_ref_c = torch.matmul(tensor_a.to(torch.float), tensor_b.to(torch.float)).to(out_dtype)
    tilelang.testing.torch_assert_close(tensor_c, tensor_ref_c, atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.05)


def test_nvrtc_dynamic_shape():
    run_nvrtc_dynamic_shape(T.dynamic("m"), 1024, 768, False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)

    run_nvrtc_dynamic_shape(T.dynamic("m"), T.dynamic("n"), 768, False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)

    run_nvrtc_dynamic_shape(T.dynamic("m"), T.dynamic("n"), T.dynamic("k"), False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)


def check_hopper():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    compute_capability = props.major, props.minor
    return compute_capability == (9, 0)


def convolution_im2col(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages, threads, dtype=T.float16, accum_dtype=T.float32):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

    @T.prim_func
    def main(
        data: T.Tensor((N, H, W, C), dtype),
        kernel: T.Tensor((KH, KW, C, F), dtype),
        out: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=threads) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=num_stages):
                T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return main


def run_nvrtc_im2col_tma_desc(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages=3, num_threads=256):
    """Test im2col TMA descriptor functionality in NVRTC backend."""
    program = convolution_im2col(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages, num_threads)

    conv_kernel = tilelang.compile(program, out_idx=-1, execution_backend="nvrtc")

    a = torch.randn(N, H, W, C).cuda().half()
    b = torch.randn(K, K, C, F).cuda().half()

    out_c = conv_kernel(a, b)

    # Reference implementation using torch.conv2d
    def ref_program(A, B):
        A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
        B = B.permute(3, 2, 0, 1)  # H, W, C, F -> F, C, H, W
        C = torch.conv2d(A, B, stride=S, padding=P, dilation=D)
        C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
        return C

    ref_c = ref_program(a, b)
    tilelang.testing.torch_assert_close(out_c, ref_c, atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.05)


def test_nvrtc_im2col_tma_desc():
    """Test im2col TMA descriptor with NVRTC backend."""
    if not check_hopper():
        import pytest

        pytest.skip("Test requires Hopper GPU (compute capability 9.0)")

    # Small test case for im2col TMA descriptor
    run_nvrtc_im2col_tma_desc(
        N=4, C=64, H=32, W=32, F=64, K=3, S=1, D=1, P=1, block_M=64, block_N=128, block_K=32, num_stages=3, num_threads=256
    )


def test_nvrtc_l2_persistent_map():
    """Test L2 persistent cache annotation with elementwise add."""
    from tilelang.language import annotate_l2_hit_ratio

    M = 1024
    N = 1024

    @tilelang.jit(out_idx=[-1], execution_backend="nvrtc")
    def elementwise_add_with_l2_cache(
        M,
        N,
        block_size=256,
        dtype=T.float32,
    ):
        @T.prim_func
        def kernel(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(M * N // block_size, threads=block_size) as bx:
                # Annotate L2 persistent cache for buffer B
                # B will be accessed multiple times and benefit from L2 caching
                annotate_l2_hit_ratio({B: 0.8})

                for i in T.serial(block_size):
                    idx = bx * block_size + i
                    if idx < M * N:
                        row = idx // N
                        col = idx % N
                        C[row, col] = A[row, col] + B[row, col]

        return kernel

    # Compile the kernel
    kernel = elementwise_add_with_l2_cache(M, N)

    # Create test tensors
    a = torch.randn(M, N, dtype=torch.float32).cuda()
    b = torch.randn(M, N, dtype=torch.float32).cuda()

    # Run kernel with out_idx=[-1], C is returned not passed in
    c = kernel(a, b)

    # Verify correctness
    ref_c = a + b
    tilelang.testing.torch_assert_close(c, ref_c, atol=1e-5, rtol=1e-5)

    print("L2 persistent map test passed!")


if __name__ == "__main__":
    tilelang.testing.main()
