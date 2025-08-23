from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl


def reshape_test(N, M, dtype):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N // M, M), dtype),
    ):
        with T.Kernel(1) as _:
            A_reshaped = T.reshape(A, [N // M, M])
            T.copy(A_reshaped, B)

    return main


def run_reshape(N, M, dtype):
    program = reshape_test(N, M, dtype)
    # TODO(lei): reshape cannot apply shared memory
    # layout transform propagation
    jit_kernel = tl.compile(
        program,
        out_idx=-1,
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.reshape(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_smem():
    # Test reshape
    run_reshape(1024, 32, "float32")
    run_reshape(2048, 64, "float16")


def reshape_test_smem_1d_2_2d(N, M, dtype):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N // M, M), dtype),
    ):
        with T.Kernel(1) as _:
            A_shared = T.alloc_shared((N,), dtype)
            for i in T.Parallel(N):
                A_shared[i] = A[i]

            A_smem_reshaped = T.reshape(A_shared, [N // M, M])
            T.copy(A_smem_reshaped, B)

    return main


def run_reshape_smem_1d_2_2d(N, M, dtype):
    program = reshape_test_smem_1d_2_2d(N, M, dtype)
    # TODO(lei): reshape cannot apply shared memory
    # layout transform propagation
    jit_kernel = tl.compile(
        program,
        out_idx=-1,
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.reshape(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_smem_1d_2_2d():
    run_reshape_smem_1d_2_2d(1024, 32, "float32")
    run_reshape_smem_1d_2_2d(2048, 64, "float16")


def reshape_test_smem_2d_2_1d(N, M, dtype):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Tensor((N // M, M), dtype),
            B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1) as _:
            A_shared = T.alloc_shared((N // M, M), dtype)
            for i, j in T.Parallel(N // M, M):
                A_shared[i, j] = A[i, j]

            A_smem_reshaped = T.reshape(A_shared, [N])
            T.copy(A_smem_reshaped, B)

    return main


def run_reshape_smem_2d_2_1d(N, M, dtype):
    program = reshape_test_smem_2d_2_1d(N, M, dtype)
    # TODO(lei): reshape cannot apply shared memory
    # layout transform propagation
    jit_kernel = tl.compile(
        program,
        out_idx=-1,
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.reshape(N)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_smem_2d_2_1d():
    run_reshape_smem_2d_2_1d(1024, 32, "float32")
    run_reshape_smem_2d_2_1d(2048, 64, "float16")


if __name__ == "__main__":
    tilelang.testing.main()
