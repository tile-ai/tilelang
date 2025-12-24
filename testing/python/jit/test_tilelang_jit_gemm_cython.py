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

    matmul_kernel = tilelang.compile(program, out_idx=-1, execution_backend="cython")

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


def run_cython_kernel_do_bench(
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

    cython_matmul_kernel = tilelang.compile(program, execution_backend="cython")

    cython_profiler = cython_matmul_kernel.get_profiler()

    cython_latency = cython_profiler.do_bench(func=cython_matmul_kernel)
    print(f"cython Latency: {cython_latency} ms")

    tvm_latency = cython_profiler.do_bench()
    print(f"TVM Latency: {tvm_latency} ms")

    assert tvm_latency is not None
    assert cython_latency is not None


def test_cython_kernel_do_bench():
    run_cython_kernel_do_bench(512, 1024, 768, False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)


def run_cython_kernel_multi_stream(
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

    matmul_kernel = tilelang.compile(program, execution_backend="cython")

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


def test_cython_kernel_multi_stream():
    run_cython_kernel_multi_stream(512, 1024, 768, False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)


def run_cython_dynamic_shape(
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

    matmul_kernel = tilelang.compile(program, execution_backend="cython")
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


def test_cython_dynamic_shape():
    run_cython_dynamic_shape(T.dynamic("m"), 1024, 768, False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)

    run_cython_dynamic_shape(T.dynamic("m"), T.dynamic("n"), 768, False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)

    run_cython_dynamic_shape(T.dynamic("m"), T.dynamic("n"), T.dynamic("k"), False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)


def run_cython_dynamic_shape_with_out_idx(
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

    matmul_kernel = tilelang.compile(program, execution_backend="cython", out_idx=-1)
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

    tensor_c = matmul_kernel(tensor_a, tensor_b)

    tensor_ref_c = torch.matmul(tensor_a.to(torch.float), tensor_b.to(torch.float)).to(out_dtype)

    tilelang.testing.torch_assert_close(tensor_c, tensor_ref_c, atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.05)


def test_cython_dynamic_shape_with_out_idx():
    run_cython_dynamic_shape_with_out_idx(T.dynamic("m"), 1024, 768, False, False, T.float16, T.float16, T.float16, 128, 256, 32, 2)


def matmul_int_variable(
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
        offset: T.int32,
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
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] + offset
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_matmul_int_variable(M, N, K, block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, num_stages, threads):
    program = matmul_int_variable(
        M, N, K, block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, num_stages, threads
    )
    matmul_kernel = tilelang.compile(program, execution_backend="cython", out_idx=2)

    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)

    tensor_a = torch.randn(M, K, dtype=in_dtype).cuda()
    tensor_b = torch.randn(K, N, dtype=in_dtype).cuda()

    tensor_c = matmul_kernel(tensor_a, tensor_b, 1)

    tensor_ref_c = torch.matmul(tensor_a, tensor_b).to(out_dtype) + 1
    tilelang.testing.torch_assert_close(tensor_c, tensor_ref_c, rtol=1e-2, atol=1e-2)


def test_matmul_int_variable():
    run_matmul_int_variable(1024, 1024, 1024, 128, 128, 32, False, False, T.float16, T.float16, T.float32, 0, 128)


def matmul_float_variable(
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
        offset: T.float32,
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
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] + offset
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_matmul_float_variable(M, N, K, block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, num_stages, threads):
    program = matmul_float_variable(
        M, N, K, block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, num_stages, threads
    )
    matmul_kernel = tilelang.compile(program, execution_backend="cython", out_idx=2)

    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)

    tensor_a = torch.randn(M, K, dtype=in_dtype).cuda()
    tensor_b = torch.randn(K, N, dtype=in_dtype).cuda()

    tensor_c = matmul_kernel(tensor_a, tensor_b, 1.0)

    tensor_ref_c = torch.matmul(tensor_a, tensor_b).to(out_dtype) + 1.0
    tilelang.testing.torch_assert_close(tensor_c, tensor_ref_c, rtol=1e-2, atol=1e-2)


def test_matmul_float_variable():
    run_matmul_float_variable(1024, 1024, 1024, 128, 128, 32, False, False, T.float16, T.float16, T.float32, 0, 128)


if __name__ == "__main__":
    tilelang.testing.main()
