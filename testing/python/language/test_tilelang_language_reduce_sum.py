# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl

tilelang.disable_cache()


def reduce_sum_test(M, N, dtype="float16"):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            # Copy input to local
            T.copy(A, A_local)
            # Perform reduce_sum operation
            T.reduce_sum(A_local, B_local, dim=0)
            # Copy result back
            T.copy(B_local, B)

    return main


def run_reduce_sum(M, N, dtype="float16"):
    program = reduce_sum_test(M, N, dtype)
    jit_kernel = tl.compile(program, out_idx=-1)
    print(jit_kernel.get_kernel_source())
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.sum(dim=0)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reduce_sum():
    # Test different sizes
    run_reduce_sum(256, 256)
    run_reduce_sum(512, 128)
    run_reduce_sum(128, 512)

    # Test different dtypes
    run_reduce_sum(256, 256, "float32")
    run_reduce_sum(256, 256, "float16")


if __name__ == "__main__":
    tilelang.testing.main()
