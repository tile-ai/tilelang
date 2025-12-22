import tilelang
import tilelang.language as T
import torch
import tilelang.testing
import pytest

tilelang.testing.set_random_seed()


def _require_cuda_tensor(shape, dtype=torch.float32):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        return torch.randn(*shape, device="cuda", dtype=dtype)
    except RuntimeError as err:
        pytest.skip(f"CUDA runtime unavailable: {err}")


"""
Nested Parallel cases:

T.Parallel
    T.Parallel

Rule:
    - continuous parallels is allowed and will be merged into one T.Parallel.
    - Non-continuous (e.g. with some statements in the outer-loop) are forbidden.
"""


@tilelang.jit(out_idx=[1])
def nested_continuous_parallels(length=256, block=16, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length // block):
                for j in T.Parallel(block):
                    B[i * block + j] = A[i * block + j] + 1.0

    return main


@tilelang.jit(out_idx=[1])
def nested_triple_continuous_parallels(length=256, block1=8, block2=2, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length // block1 // block2):
                for j in T.Parallel(block1):
                    for k in T.Parallel(block2):
                        B[i * block1 * block2 + j * block2 + k] = A[i * block1 * block2 + j * block2 + k] + 1.0

    return main


@tilelang.jit(out_idx=[1])
def nested_noncontinuous_parallels(length=256, block=16, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length // block):
                B[i] = 0
                for j in T.Parallel(block):
                    B[i * block + j] = A[i * block + j] + 1.0

    return main


def test_nested_parallels():
    kernel1 = nested_continuous_parallels(length=256, block=16)
    kernel2 = nested_triple_continuous_parallels(length=256, block1=8, block2=2)
    data = _require_cuda_tensor((256,), torch.float32)
    result1 = kernel1(data)
    result2 = kernel2(data)
    torch.testing.assert_close(result1, data + 1.0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result2, data + 1.0, atol=1e-5, rtol=1e-5)

    # This is invalid
    with pytest.raises(ValueError):
        nested_noncontinuous_parallels(length=256, block=16)


"""
Nested Pipeline cases:

T.Pipeline
    T.Pipeline

is OK.
"""


def matmul_nested_pipelines(
    M, N, K, block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype, accum_dtype, threads, order, stage, extra_pipeline_repeats
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
            for _ in T.Pipelined(extra_pipeline_repeats):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), order=order, stage=stage):
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


def run_gemm_nested_pipelines(
    order,
    stage,
    extra_pipeline_repeats,
):
    M = 1024
    N = 1024
    K = 1024
    block_M = 128
    block_N = 128
    block_K = 32
    trans_A = False
    trans_B = False
    in_dtype = T.float16
    out_dtype = T.float16
    dtypeAccum = T.float32
    num_threads = 128
    program = matmul_nested_pipelines(
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
        num_threads,
        order,
        stage,
        extra_pipeline_repeats,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
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


def test_nested_pipelines():
    run_gemm_nested_pipelines(order=[0, 1, 2], stage=[0, 0, 1], extra_pipeline_repeats=3)


"""
Nested serial cases:

T.serial
    T.serial

is OK.
"""


@tilelang.jit(out_idx=[1])
def nested_continuous_serials(length=256, block=16, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.serial(length // block):
                for j in T.serial(block):
                    B[i * block + j] = A[i * block + j] + 1.0

    return main


@tilelang.jit(out_idx=[1])
def nested_noncontinuous_serials(length=256, block=16, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.serial(length // block):
                B[i] = 0
                for j in T.serial(block):
                    B[i * block + j] = A[i * block + j] + 1.0

    return main


def test_nested_serials():
    kernel1 = nested_continuous_serials(length=256, block=16)
    data = _require_cuda_tensor((256,), torch.float32)
    result1 = kernel1(data)
    torch.testing.assert_close(result1, data + 1.0, atol=1e-5, rtol=1e-5)

    # This is valid
    nested_noncontinuous_serials(length=256, block=16)


"""
Mixed serial and Parallel loops:

(S-P)
T.serial
    T.Parallel

(P-S)
T.Parallel
    T.serial

Rule:
    - No Parallel - * - Parallel
"""


@tilelang.jit(out_idx=[1])
def nested_continuous_sp(length=256, block=16, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.serial(length // block):
                for j in T.Parallel(block):
                    B[i * block + j] = A[i * block + j] + 1.0

    return main


@tilelang.jit(out_idx=[1])
def nested_continuous_ps(length=256, block=16, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length // block):
                for j in T.serial(block):
                    B[i * block + j] = A[i * block + j] + 1.0

    return main


@tilelang.jit(out_idx=[1])
def nested_continuous_psp(length=256, block1=8, block2=2, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length // block1 // block2):
                for j in T.serial(block1):
                    for k in T.Parallel(block2):
                        B[i * block1 * block2 + j * block2 + k] = A[i * block1 * block2 + j * block2 + k] + 1.0

    return main


@tilelang.jit(out_idx=[1])
def nested_continuous_sps(length=256, block1=8, block2=2, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.serial(length // block1 // block2):
                for j in T.Parallel(block1):
                    for k in T.serial(block2):
                        B[i * block1 * block2 + j * block2 + k] = A[i * block1 * block2 + j * block2 + k] + 1.0

    return main


def test_mixed_sp():
    kernel1 = nested_continuous_sp(length=256, block=16)
    kernel2 = nested_continuous_ps(length=256, block=16)
    data = _require_cuda_tensor((256,), torch.float32)
    result1 = kernel1(data)
    result2 = kernel2(data)
    torch.testing.assert_close(result1, data + 1.0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result2, data + 1.0, atol=1e-5, rtol=1e-5)

    # This should be invalid (Undefined behaviour)
    with pytest.raises(ValueError):
        nested_continuous_psp(length=256, block1=16, block2=8)

    kernel3 = nested_continuous_sps(length=256, block1=8, block2=2)
    result3 = kernel3(data)
    torch.testing.assert_close(result3, data + 1.0, atol=1e-5, rtol=1e-5)


"""
Mixed Pipelined and Parallel loops:

(Pi-Pa)
T.Pipelined
    T.Parallel

(Pa-Pi)
T.Parallel
    T.Pipelined

Rule:
    - Pi-Pa is ok where Pa-Pi is not allowed.
    - For more nested cases, refer to the rule of T.Parallel.
"""


def matmul_nested_pipa(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
    order,
    stage,
):
    A_shape = (M, K)
    B_shape = (K, N)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_K, block_N)

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
            for k in T.Pipelined(T.ceildiv(K, block_K), order=order, stage=stage):
                for i, j in T.Parallel(block_M, block_K):
                    A_shared[i, j] = A[by * block_M + i, k * block_K + j]
                for i, j in T.Parallel(block_K, block_N):
                    B_shared[i, j] = B[k * block_K + i, bx * block_N + j]

                # T.copy(A[by * block_M, k * block_K], A_shared)
                # T.copy(B[k * block_K, bx * block_N], B_shared)

                T.gemm(A_shared, B_shared, C_local, False, False)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_nested_papipa(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
    order,
    stage,
):
    A_shape = (M, K)
    B_shape = (K, N)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_K, block_N)

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
            for _ in T.Parallel(1):
                for k in T.Pipelined(T.ceildiv(K, block_K), order=order, stage=stage):
                    for i, j in T.Parallel(block_M, block_K):
                        A_shared[i, j] = A[by * block_M + i, k * block_K + j]
                    for i, j in T.Parallel(block_K, block_N):
                        B_shared[i, j] = B[k * block_K + i, bx * block_N + j]

                    # T.copy(A[by * block_M, k * block_K], A_shared)
                    # T.copy(B[k * block_K, bx * block_N], B_shared)

                    T.gemm(A_shared, B_shared, C_local, False, False)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_mixed_pp(
    order,
    stage,
):
    M = 1024
    N = 1024
    K = 1024
    block_M = 128
    block_N = 128
    block_K = 32
    in_dtype = T.float16
    out_dtype = T.float16
    dtypeAccum = T.float32
    num_threads = 128

    program = matmul_nested_pipa(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_threads,
        order,
        stage,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if in_dtype == T.float32:
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 meas
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)

    program1 = matmul_nested_papipa(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_threads,
        order,
        stage,
    )
    with pytest.raises(ValueError):
        tilelang.compile(
            program1,
            out_idx=[2],
            pass_configs={
                tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
                tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            },
        )


def test_mixed_pp():
    run_gemm_mixed_pp(order=[0, 1, 2], stage=[0, 0, 1])


"""
TiledOp in a T.Parallel is also not permitted.
"""


def matmul_with_parallel(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
    order,
    stage,
):
    A_shape = (M, K)
    B_shape = (K, N)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_K, block_N)

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
            for k in T.Pipelined(T.ceildiv(K, block_K), order=order, stage=stage):
                for i, j in T.Parallel(block_M, block_K):
                    A_shared[i, j] = A[by * block_M + i, k * block_K + j]
                for i, j in T.Parallel(block_K, block_N):
                    B_shared[i, j] = B[k * block_K + i, bx * block_N + j]

                # T.copy(A[by * block_M, k * block_K], A_shared)
                # T.copy(B[k * block_K, bx * block_N], B_shared)

                for _ in T.Parallel(1):
                    T.gemm(A_shared, B_shared, C_local, False, False)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_tiled_op_with_parallel(
    order,
    stage,
):
    M = 1024
    N = 1024
    K = 1024
    block_M = 128
    block_N = 128
    block_K = 32
    in_dtype = T.float16
    out_dtype = T.float16
    dtypeAccum = T.float32
    num_threads = 128

    program = matmul_nested_pipa(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_threads,
        order,
        stage,
    )

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if in_dtype == T.float32:
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 meas
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)

    program1 = matmul_with_parallel(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_threads,
        order,
        stage,
    )
    with pytest.raises(ValueError):
        tilelang.compile(
            program1,
            out_idx=[2],
            pass_configs={
                tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
                tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            },
        )


@tilelang.jit(out_idx=[1])
def tir_op_with_parallel(length=256, block=16, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length // block):
                for j in T.Parallel(block):
                    B[i * block + j] = T.max(A[i * block + j], 0.0)

    return main


@tilelang.jit(out_idx=[1])
def customize_op_with_parallel(length=256, block=16, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length // block):
                for j in T.Parallel(block):
                    B[i * block + j] = A[i * block + j]
                    T.atomic_add(B[i * block + j], 1.0)

    return main


def test_tiled_op_with_parallel():
    run_gemm_tiled_op_with_parallel(order=[0, 1, 2], stage=[0, 0, 1])

    kernel1 = tir_op_with_parallel(length=256, block=16)
    data = _require_cuda_tensor((256,), torch.float32)
    result1 = kernel1(data)
    torch.testing.assert_close(result1, torch.relu(data), atol=1e-5, rtol=1e-5)
    kernel2 = customize_op_with_parallel(length=256, block=16)
    result2 = kernel2(data)
    torch.testing.assert_close(result2, data + 1, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
