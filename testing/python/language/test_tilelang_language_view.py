import tilelang.language as T
from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import pytest
import torch


def view_test(N, M, dtype, new_dtype=None):
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
            torch_dtype = T.dtype(new_dtype).as_torch()
            return A.view(N // M, M).view(dtype=torch_dtype)
        return A.view(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_view():
    # Test view with same dtype
    run_view(1024, 32, T.float32)
    run_view(2048, 64, T.float16)

    # Test view with dtype conversion
    run_view(1024, 32, T.float32, T.float16)
    run_view(2048, 64, T.float16, T.float32)


def view_shape_mismatch_test(N, M, dtype, new_dtype=None):
    new_shape = [N // M, M + 1]
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


def test_view_shape_mismatch():
    with pytest.raises(AssertionError):
        view_shape_mismatch_test(1024, 32, T.float32)


def test_view_subbyte_dtype_change():
    A = tvm.tir.decl_buffer((16, 32), "float4_e2m1fn", name="A")
    A_viewed = T.view(A, (16, 16), dtype=T.uint8)
    assert str(A_viewed.dtype) == "uint8"
    assert tuple(int(dim) for dim in A_viewed.shape) == (16, 16)
    assert A_viewed.data.same_as(A.data)


def fp4_to_uint8_view_test(rows_per_cta=16, mask_k=256):
    @T.prim_func
    def main(
        A: T.Tensor((rows_per_cta, mask_k), T.bfloat16),
        B: T.Tensor((rows_per_cta, mask_k // 2), T.uint8),
    ):
        with T.Kernel(1, threads=256) as _:
            A_frag = T.alloc_fragment((rows_per_cta, mask_k), T.bfloat16)
            B_shared_fp4 = T.alloc_shared((rows_per_cta, mask_k), T.float4_e2m1fn)
            B_shared_uint8 = T.view(B_shared_fp4, (rows_per_cta, mask_k // 2), dtype=T.uint8)

            T.copy(A, A_frag)
            for i, j in T.Parallel(rows_per_cta, mask_k):
                B_shared_fp4[i, j] = T.cast(A_frag[i, j], T.float4_e2m1fn)
            T.copy(B_shared_uint8, B)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_view_shared_fp4_to_uint8_compile():
    program = fp4_to_uint8_view_test()
    kernel = tl.compile(program, out_idx=-1)
    src = kernel.get_kernel_source()
    assert "fp4_e2" in src

    dummy_input = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    output = kernel(dummy_input)
    assert output.shape == (16, 128)
    assert output.dtype == torch.uint8


if __name__ == "__main__":
    tilelang.testing.main()
