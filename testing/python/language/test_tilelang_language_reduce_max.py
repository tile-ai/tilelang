# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl


def reduce_max_test(M, N, dtype="float16"):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer((M, N), dtype),
            B: T.Buffer((M,), dtype),
    ):
        with T.Kernel(1) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            # Copy input to local
            T.copy(A, A_local)
            # Perform reduce_max operation
            T.reduce_max(A_local, B_local, dim=1)
            # Copy result back
            T.copy(B_local, B)

    return main


def run_reduce_max(M, N, dtype="float16"):
    program = reduce_max_test(M, N, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.max(dim=1).values

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reduce_max():
    # Test different sizes
    run_reduce_max(256, 256)
    run_reduce_max(512, 128)
    run_reduce_max(128, 512)

    # Test different dtypes
    run_reduce_max(256, 256, "float32")
    run_reduce_max(256, 256, "float16")


if __name__ == "__main__":
    tilelang.testing.main()
