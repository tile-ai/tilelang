# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl


def view_test(N, M, dtype, new_dtype=None):
    import tilelang.language as T

    new_shape = [N // M, M]
    if new_dtype:
        from tvm import DataType
        dtype_src = DataType(dtype)
        dtype_dst = DataType(new_dtype)
        src_bits = dtype_src.bits
        dst_bits = dtype_dst.bits
        scale = src_bits / dst_bits
        new_shape[-1] = int(M * scale)

    @T.prim_func
    def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor(new_shape, new_dtype if new_dtype else dtype),
    ):
        with T.Kernel(1) as _:
            A_viewed = T.view(A, new_shape, dtype=new_dtype)
            T.copy(A_viewed, B)

    return main


def run_view(N, M, dtype, new_dtype=None):
    program = view_test(N, M, dtype, new_dtype)
    jit_kernel = tl.compile(program, out_idx=-1)
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        if new_dtype:
            from tilelang.utils.tensor import map_torch_type
            torch_dtype = map_torch_type(new_dtype)
            return A.view(N // M, M).view(dtype=torch_dtype)
        return A.view(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_view():

    # Test view with same dtype
    run_view(1024, 32, "float32")
    run_view(2048, 64, "float16")

    # Test view with dtype conversion
    run_view(1024, 32, "float32", "float16")
    run_view(2048, 64, "float16", "float32")


if __name__ == "__main__":
    tilelang.testing.main()
