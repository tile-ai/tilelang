from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing
import tilelang
import torch
import pytest


@pytest.mark.parametrize("present", [(True, True), (True, False), (False, True), (False, False)])
def test_cython_dynamic_symbol_shared_by_shape_and_stride(present):
    """Each non-null carrier must retain whether it supplies a shape or stride."""
    from tilelang.engine.param import KernelParam
    from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter, CythonKernelWrapper

    size = tvm.tirx.Var("size", "int64")
    a_buffer = tvm.tirx.decl_buffer((2, 3), "float32", name="A", strides=(size, 1))
    b_buffer = tvm.tirx.decl_buffer((size,), "float32", name="B")
    func = tvm.tirx.PrimFunc(
        [a_buffer.data, b_buffer.data], tvm.tirx.Evaluate(0), buffer_map={a_buffer.data: a_buffer, b_buffer.data: b_buffer}
    )
    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({"main": func})
    adapter.result_idx = []
    adapter.target = tvm.target.Target("c")

    class CaptureLibrary:
        def call(self, *args):
            self.symbol = args[2].value
            return 0

    lib = CaptureLibrary()
    params = [KernelParam.from_buffer(buffer) for buffer in (a_buffer, b_buffer)]
    wrapper = CythonKernelWrapper([], params, lib)
    wrapper.set_dynamic_symbolic_map(adapter._process_dynamic_symbolic())
    wrapper.set_dynamic_symbolic_sources(adapter._process_dynamic_symbolic_sources())
    wrapper.set_ptr_map(adapter._process_ptr_map())
    wrapper.set_buffer_dtype_map(adapter._process_buffer_dtype())
    wrapper.set_buffer_device_map(adapter._process_buffer_device())
    shapes, strides, contiguous = adapter._process_static_buffer_infos()
    wrapper.set_static_shape_map(shapes)
    wrapper.set_static_strides_map(strides)
    wrapper.set_static_contiguous_list(contiguous)

    a = torch.empty_strided((2, 3), (4, 1)) if present[0] else None
    b = torch.empty((4,)) if present[1] else None
    wrapper.forward([a, b], stream=0)
    assert lib.symbol == (4 if any(present) else 0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_cython_pdl():
    """Test pdl."""

    N = 64

    @tilelang.jit(execution_backend="cython")
    def multi_kernels_with_pdl(N, block_size=256, dtype=T.float32):
        @T.prim_func
        def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
            C: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < N:
                        B[idx] = A[idx] + 1.0
                T.pdl_trigger()

            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx2,):
                T.pdl_sync()
                for i in T.Parallel(block_size):
                    idx = bx2 * block_size + i
                    if idx < N:
                        C[idx] = B[idx] * 2.0

        return main

    # Compile the kernel
    kernel = multi_kernels_with_pdl(N)

    # Create test tensors
    a = torch.randn(N, dtype=torch.float32).cuda()
    b = torch.randn(N, dtype=torch.float32).cuda()
    c = torch.randn(N, dtype=torch.float32).cuda()

    ref_b = a + 1.0
    ref_c = ref_b * 2.0

    kernel(a, b, c)

    # Verify correctness

    tilelang.testing.torch_assert_close(b, ref_b, atol=1e-5, rtol=1e-5)
    tilelang.testing.torch_assert_close(c, ref_c, atol=1e-5, rtol=1e-5)

    print("pdl test passed!")


@tilelang.testing.requires_cuda
def test_cython_dynamic_shape_output_before_input():
    """An output may precede the input that supplies its symbolic dimension.

    The symbolic-dimension map used to be built over every parameter in signature
    order, outputs included, and the caller assembled its tensor list in a single
    pass. When the output came first it was therefore recorded as the owner of its
    own dimension and then read its own not-yet-filled slot.
    """
    N = T.dynamic("N")

    @tilelang.jit(out_idx=[0], execution_backend="cython")
    def kernel():
        @T.prim_func
        def main(
            B: T.Tensor((N,), "float32"),
            A: T.Tensor((N,), "float32"),
        ):
            with T.Kernel(1, threads=128):
                for i in T.Parallel(N):
                    B[i] = A[i] + T.float32(9)

        return main

    a = torch.arange(128, dtype=torch.float32).cuda()
    out = kernel()(a)
    tilelang.testing.torch_assert_close(out, a + 9, atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_cython_bfloat16_scalar_parameter():
    """A bfloat16 scalar parameter is marshalled through the Cython backend."""

    @tilelang.jit(execution_backend="cython")
    def kernel():
        @T.prim_func
        def main(A: T.Tensor((128,), T.bfloat16), scalar: T.bfloat16, B: T.Tensor((128,), T.bfloat16)):
            with T.Kernel(1, threads=128):
                for i in T.Parallel(128):
                    B[i] = A[i] + scalar

        return main

    a = torch.arange(128, dtype=torch.float32).to(torch.bfloat16).cuda()
    out = kernel()(a, 1.5)
    tilelang.testing.torch_assert_close(out, a + 1.5, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
