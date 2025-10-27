from tilelang import tvm as tvm
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
                T.gemm_v2(A_shared, B_shared, C_local, trans_A, trans_B)
                # T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_ss(
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
    num_threads=256,
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

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

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

    latency = profiler.do_bench(profiler.func, warmup=100)
    print(f"GEMM SS latency: {latency} ms")


def test_gemm_ss():
    # GEMM tests for float16
    run_gemm_ss(1024, 1024, 1024, False, False, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_ss(1024, 1024, 1024, False, True, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_ss(1024, 1024, 1024, True, False, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_ss(1024, 1024, 1024, True, True, "float16", "float16", "float32", 128, 128, 32)

    # GEMM tests for int8 tests
    run_gemm_ss(1024, 1024, 1024, False, True, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_ss(1024, 1024, 1024, False, False, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_ss(1024, 1024, 1024, True, False, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_ss(1024, 1024, 1024, True, True, "int8", "int8", "int32", 128, 128, 32)


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
    A_frag_shape = A_shared_shape

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
            A_frag = T.alloc_fragment(A_frag_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
            })
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.copy(A_shared, A_frag)
                T.gemm_v2(A_frag, B_shared, C_local, trans_A, trans_B)
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
    num_stages=3,
    num_threads=256,
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

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

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


def test_gemm_rs():
    # GEMM tests for float16
    run_gemm_rs(1024, 1024, 1024, False, False, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, False, True, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, False, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, True, "float16", "float16", "float32", 128, 128, 32)

    # GEMM tests for int8 tests
    run_gemm_rs(1024, 1024, 1024, False, True, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, False, False, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, False, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, True, "int8", "int8", "int32", 128, 128, 32)


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
    B_frag_shape = B_shared_shape

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
            B_frag = T.alloc_fragment(B_frag_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            T.annotate_layout({
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.copy(B_shared, B_frag)
                T.gemm_v2(A_shared, B_frag, C_local, trans_A, trans_B)
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
    num_stages=3,
    num_threads=256,
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

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

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


def test_gemm_sr():
    # GEMM tests for float16
    run_gemm_sr(1024, 1024, 1024, False, False, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_sr(1024, 1024, 1024, False, True, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_sr(1024, 1024, 1024, True, False, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_sr(1024, 1024, 1024, True, True, "float16", "float16", "float32", 128, 128, 32)

    # GEMM tests for int8 tests
    run_gemm_sr(1024, 1024, 1024, False, True, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_sr(1024, 1024, 1024, False, False, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_sr(1024, 1024, 1024, True, False, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_sr(1024, 1024, 1024, True, True, "int8", "int8", "int32", 128, 128, 32)


def matmul_rr(
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
    A_frag_shape = A_shared_shape
    B_frag_shape = B_shared_shape

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
            A_frag = T.alloc_fragment(A_frag_shape, in_dtype)
            B_frag = T.alloc_fragment(B_frag_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.copy(A_shared, A_frag)
                T.copy(B_shared, B_frag)
                T.gemm_v2(A_frag, B_frag, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rr(
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
    num_threads=256,
):
    program = matmul_rr(
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

    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    print(program)

    print(kernel.get_kernel_source())

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

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


def test_gemm_rr():
    # GEMM tests for float16
    run_gemm_rr(1024, 1024, 1024, False, False, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_rr(1024, 1024, 1024, False, True, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_rr(1024, 1024, 1024, True, False, "float16", "float16", "float32", 128, 128, 32)
    run_gemm_rr(1024, 1024, 1024, True, True, "float16", "float16", "float32", 128, 128, 32)

    # GEMM tests for int8 tests
    run_gemm_rr(1024, 1024, 1024, False, True, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_rr(1024, 1024, 1024, False, False, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_rr(1024, 1024, 1024, True, False, "int8", "int8", "int32", 128, 128, 32)
    run_gemm_rr(1024, 1024, 1024, True, True, "int8", "int8", "int32", 128, 128, 32)


if __name__ == "__main__":
    tilelang.testing.main()
