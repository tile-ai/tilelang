# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl


def reshape_test(N, M, dtype):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer((N,), dtype),
            B: T.Buffer((N // M, M), dtype),
    ):
        with T.Kernel(1) as _:
            A_reshaped = T.reshape(A, [N // M, M])
            T.copy(A_reshaped, B)

    return main


def run_reshape(N, M, dtype):
    program = reshape_test(N, M, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.reshape(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_smem():
    # Test reshape
    run_reshape(1024, 32, "float32")
    run_reshape(2048, 64, "float16")


def reshape_test_smem(N, M, dtype):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer((N,), dtype),
            B: T.Buffer((N // M, M), dtype),
    ):
        with T.Kernel(1) as _:
            A_shared = T.alloc_shared((N,), dtype)
            for i in range(N):
                A_shared[i] = A[i]

            A_smem_reshaped = T.reshape(A_shared, [N // M, M])
            for i in range(N // M):
                for j in range(M):
                    B[i, j] = A_smem_reshaped[i, j]

    return main


def run_reshape_smem(N, M, dtype):
    program = reshape_test_smem(N, M, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.reshape(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_smem_shared():
    run_reshape_smem(1024, 32, "float32")
    run_reshape_smem(2048, 64, "float16")


if __name__ == "__main__":
    tilelang.testing.main()
