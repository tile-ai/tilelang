import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.transform import DecoupleTypeCast


def _check(original, transformed):
    """Apply DecoupleTypeCast pass and check IR matches expected output."""
    mod = tvm.IRModule.from_expr(original.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)

    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))

    tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def test_local_to_memory():
    """Test local → memory: compute to cast buffer, then copy to memory."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        b_local_cast = T.decl_buffer((16,), T.float4_e2m1fn, scope="local")
        for i in T.vectorized(16):
            b_local_cast[i] = T.cast(b_frag[i], T.float4_e2m1fn)
        for i_copy in T.vectorized(16):
            b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_memory_to_local():
    """Test memory → local: copy from memory to cast buffer, then compute."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        b_local_cast = T.decl_buffer((16,), T.float4_e2m1fn, scope="local")
        for i in T.vectorized(16):
            b_local_cast[i] = b_frag[i]
        for i_copy in T.vectorized(16):
            b[i_copy] = b_local_cast[i_copy]

    _check(before, after)


def test_no_transform_same_dtype():
    """Test no transformation when dtypes are the same."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float32]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float32]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]

    _check(before, after)


def test_no_transform_local_to_local():
    """Test no transformation for local → local (both are local buffers)."""

    @T.prim_func
    def before():
        a_frag = T.alloc_local((16,), T.float32)
        b_frag = T.alloc_local((16,), T.float4_e2m1fn)
        for i in T.vectorized(16):
            b_frag[i] = a_frag[i]

    @T.prim_func
    def after():
        a_frag = T.alloc_local((16,), T.float32)
        b_frag = T.alloc_local((16,), T.float4_e2m1fn)
        for i in T.vectorized(16):
            b_frag[i] = T.cast(a_frag[i], T.float4_e2m1fn)

    _check(before, after)


def test_no_transform_if_then_else_condition():
    """Test no transformation when different dtype is only in if_then_else condition.

    The condition part of if_then_else doesn't participate in type casting,
    so a global/shared buffer load with different dtype in condition should
    not trigger cast buffer insertion.
    """

    @T.prim_func
    def before(cond_buf: T.Tensor[(1,), T.int32]):
        acc = T.alloc_local((8,), T.float32)
        for i in T.vectorized(8):
            # cond_buf is int32, acc is float32, but cond_buf is only in condition
            acc[i] = T.if_then_else(cond_buf[0] > 0, acc[i] * 2.0, acc[i])

    @T.prim_func
    def after(cond_buf: T.Tensor[(1,), T.int32]):
        acc = T.alloc_local((8,), T.float32)
        for i in T.vectorized(8):
            # Should remain unchanged - no cast buffer needed
            acc[i] = T.if_then_else(cond_buf[0] > 0, acc[i] * T.float32(2), acc[i])

    _check(before, after)


# =============================================================================
# CUDA Codegen Tests
# =============================================================================


@tilelang.testing.requires_cuda
def test_codegen_local_to_memory():
    """Test CUDA codegen for local → memory with vectorized copy."""

    @tilelang.jit
    def kernel_fn():
        b = T.empty((16,), dtype="float4_e2m1fn")
        with T.Kernel(1, threads=32):
            b_frag = T.alloc_local((16,), T.float32)
            for i in T.vectorized(16):
                b[i] = b_frag[i]
        return b

    kernel = kernel_fn.compile()
    source = kernel.get_kernel_source()

    # Should have local cast buffer
    assert "b_local_cast" in source, "Expected local cast buffer in generated code"
    # Should have vectorized copy (fp4_e2_16_t is 16 fp4 elements = 64 bits)
    assert "fp4_e2_16_t" in source, "Expected vectorized fp4 copy in generated code"


@tilelang.testing.requires_cuda
def test_codegen_memory_to_local():
    """Test CUDA codegen for memory → local with vectorized copy."""

    @tilelang.jit
    def kernel_fn():
        b = T.empty((16,), dtype="float4_e2m1fn")
        with T.Kernel(1, threads=32):
            a_frag = T.alloc_local((16,), T.float32)
            for i in T.vectorized(16):
                a_frag[i] = b[i]
        return b

    kernel = kernel_fn.compile()
    source = kernel.get_kernel_source()

    # Should have local cast buffer
    assert "b_local_cast" in source, "Expected local cast buffer in generated code"


@tilelang.testing.requires_cuda
def test_codegen_fp8_local_to_memory():
    """Test CUDA codegen for fp8 local → memory."""

    @tilelang.jit
    def kernel_fn():
        b = T.empty((16,), dtype="float8_e4m3fn")
        with T.Kernel(1, threads=32):
            b_frag = T.alloc_local((16,), T.float32)
            for i in T.vectorized(16):
                b[i] = b_frag[i]
        return b

    kernel = kernel_fn.compile()
    source = kernel.get_kernel_source()

    # Should have local cast buffer
    assert "b_local_cast" in source, "Expected local cast buffer in generated code"
    # Should have fp8 conversion (uses __nv_cvt for fp8)
    assert "fp8" in source and "cvt" in source, "Expected fp8 conversion"


@tilelang.testing.requires_cuda
def test_codegen_no_cast_buffer_same_dtype():
    """Test no cast buffer when dtypes are the same."""

    @tilelang.jit
    def kernel_fn():
        @T.prim_func
        def kernel(b: T.Tensor[(16,), T.float32]):
            with T.Kernel(1, threads=32):
                b_frag = T.alloc_local((16,), T.float32)
                for i in T.vectorized(16):
                    b[i] = b_frag[i]

        return kernel

    kernel = kernel_fn()
    source = kernel.get_kernel_source()

    # Should NOT have local cast buffer when dtypes match
    assert "local_cast" not in source, "Should not have cast buffer when dtypes match"


if __name__ == "__main__":
    test_no_transform_if_then_else_condition()
