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


def test_rmw_same_buffer_different_indices():
    """RMW with different indices into the same buffer: a[i] = a[i] + a[i+32].

    Both loads and the store target the same buffer but at different index
    expressions. Each unique (buffer, indices) pair should get its own cast
    buffer, and the RMW load `a[i]` should read from the same cast buffer the
    store writes to (so the read-side copy-from and the write-side copy-to
    share that buffer).
    """

    @T.prim_func
    def before(a: T.Tensor[(64,), T.float8_e4m3fn]):
        for i in T.vectorized(32):
            a[i] = T.cast(
                T.cast(a[i], T.float32) + T.cast(a[i + 32], T.float32),
                T.float8_e4m3fn,
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = DecoupleTypeCast()(mod)

    # Sanity checks: pass ran, two distinct cast buffers were created, and the
    # RMW load site no longer references `a` directly in the compute body.
    text = mod["main"].script()
    assert "a_local_cast" in text, "Expected cast buffer for store-side of a[i]"
    assert "a_local_cast_1" in text, "Expected second cast buffer for a[i+32]"


def test_local_to_memory_with_let_stmt():
    """Test local → memory transform still triggers through LetStmt-bound loads."""

    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            factor = scale[i]
            b[i] = a_frag[i] * factor

    @T.prim_func
    def after(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        b_local_cast = T.decl_buffer((16,), T.float8_e4m3fn, scope="local")
        for i in T.vectorized(16):
            b_local_cast[i] = T.cast(a_frag[i] * scale[i], T.float8_e4m3fn)
        for i_copy in T.vectorized(16):
            b[i_copy] = b_local_cast[i_copy]

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


# =============================================================================
# End-to-end correctness + vectorization tests for DecoupleTypeCast
# =============================================================================


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10)
def test_e2e_bf16_global_to_frag():
    """bf16 global -> float32 frag -> bf16 global: roundtrip should be lossless.

    With 1024 bf16 elements and 64 threads, each thread handles 16 bf16 = 256 bits,
    so the kernel should use 256-bit load/store (load_global_256 / store_global_256).
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.bfloat16),
            B: T.Tensor((1024,), dtype=T.bfloat16),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float32)
                T.copy(A, a_frag)
                T.copy(a_frag, B)

        return main

    kernel = kernel_fn()

    # Check vectorization: 256-bit load/store
    source = kernel.get_kernel_source()
    assert "load_global_256" in source, "Expected 256-bit global load"
    assert "store_global_256" in source, "Expected 256-bit global store"

    # Correctness
    a = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8)
def test_e2e_bf16_global_shared_frag():
    """bf16 global -> shared -> float32 frag -> bf16 global: roundtrip should be lossless.

    Shared memory path uses TMA for global->shared, then 128-bit for shared->local.
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.bfloat16),
            B: T.Tensor((1024,), dtype=T.bfloat16),
        ):
            with T.Kernel(1, threads=64):
                a_shared = T.alloc_shared((1024,), dtype=T.bfloat16)
                a_frag = T.alloc_fragment((1024,), dtype=T.float32)
                T.copy(A, a_shared)
                T.copy(a_shared, a_frag)
                T.copy(a_frag, B)

        return main

    kernel = kernel_fn()

    # Check: shared path should NOT use 256-bit (shared doesn't support it)
    source = kernel.get_kernel_source()
    assert "uint4" in source, f"Expected uint4 store in {source}"

    # Correctness
    a = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_e2e_fp8_global_to_frag():
    """fp8 global -> float32 frag -> fp8 global: roundtrip should be lossless.

    Verifies that cast constraints do not pollute the memory access layout.
    With 1024 fp8 elements and 64 threads, each thread handles 16 fp8 = 128 bits,
    so the kernel should use fp8_e4_16_t (128-bit) loads/stores.
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.float8_e4m3fn),
            B: T.Tensor((1024,), dtype=T.float8_e4m3fn),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float32)
                T.copy(A, a_frag)
                T.copy(a_frag, B)

        return main

    kernel = kernel_fn()
    source = kernel.get_kernel_source()
    assert "fp8_e4_16_t" in source, (
        "Expected fp8_e4_16_t (128-bit) loads/stores for N=1024. Cast constraints may be polluting layout decisions."
    )

    a = (torch.randn(1024, device="cuda", dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn)
    b = kernel(a)
    torch.testing.assert_close(
        b.to(torch.float32),
        a.to(torch.float32),
        rtol=0,
        atol=0,
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_e2e_fp8_manual_decouple():
    """fp8 with manually decoupled copy stages: same result as auto-decoupled.

    Tests: fp8 global -> fp8 frag -> float32 frag -> fp8 frag -> fp8 global
    """
    import torch

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.float8_e4m3fn),
            B: T.Tensor((1024,), dtype=T.float8_e4m3fn),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float8_e4m3fn)
                b_frag = T.alloc_fragment((1024,), dtype=T.float32)
                c_frag = T.alloc_fragment((1024,), dtype=T.float8_e4m3fn)
                T.copy(A, a_frag)
                T.copy(a_frag, b_frag)
                T.copy(b_frag, c_frag)
                T.copy(c_frag, B)

        return main

    kernel = kernel_fn()

    # Check vectorization
    source = kernel.get_kernel_source()
    assert "fp8_e4_16_t" in source, "Expected fp8_e4_16_t in kernel source"

    # Correctness
    a = (torch.randn(1024, device="cuda", dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn)
    b = kernel(a)
    torch.testing.assert_close(
        b.to(torch.float32),
        a.to(torch.float32),
        rtol=0,
        atol=0,
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_e2e_scalar_load_no_cast_buffer():
    """Test that scalar memory load (b[0]) is not decoupled into a cast buffer.

    When a vectorized loop stores to global with a scalar memory load in the
    expression (e.g. c[i] = a_local[i] * b[0]), the scalar load's index does
    not depend on the loop variable. It should remain in the compute loop as
    a broadcast, not be extracted into a local cast buffer.

    Previously this caused float32x32 codegen errors because both
    VectorizePlanner and DecoupleTypeCast treated b[0] as a vector memory
    access.
    """

    @tilelang.jit
    def kernel_fn():
        @T.prim_func
        def main(
            a: T.Tensor[(32,), T.float8_e4m3fn],
            b: T.Tensor[(1,), T.float32],
            c: T.Tensor[(32,), T.float8_e4m3fn],
        ):
            with T.Kernel(1, threads=32):
                a_local = T.alloc_local((32,), T.float8_e4m3fn)
                T.copy(a, a_local)

                for i in T.vectorized(32):
                    c[i] = a_local[i] * b[0]

        return main

    kernel = kernel_fn()
    source = kernel.get_kernel_source()

    assert "c_local_cast" in source, "Expected c_local_cast for store-side decoupling"
    assert "b_local_cast" not in source, "Scalar load b[0] should not get a cast buffer"


if __name__ == "__main__":
    test_no_transform_if_then_else_condition()
    test_e2e_scalar_load_no_cast_buffer()
    test_e2e_bf16_global_to_frag()
    test_e2e_bf16_global_shared_frag()
    test_e2e_fp8_global_to_frag()
    test_e2e_fp8_manual_decouple()
