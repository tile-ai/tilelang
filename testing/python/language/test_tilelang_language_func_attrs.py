"""Test T.annotate_compile_flags, T.annotate_pass_configs, and out_idx via PrimFunc attrs."""

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.transform import PassConfigKey


def test_out_idx_via_attr_lazy():
    """out_idx should be stored as PrimFunc attr when using T.empty + return."""

    @T.prim_func
    def kernel(A):
        A: T.Tensor[[128, 128], T.float32]
        B = T.empty([128, 128], T.float32)
        with T.Kernel(1):
            for i in T.serial(128):
                for j in T.serial(128):
                    B[i, j] = A[i, j] + 1.0
        return B

    assert "tilelang_out_idx" in kernel.attrs
    assert list(kernel.attrs["tilelang_out_idx"]) == [-1]

    compiled = tilelang.compile(kernel)
    a = torch.randn(128, 128, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a + 1.0)

    rt_mod = compiled.adapter.get_exportable_executable().jit()
    with pytest.raises(AttributeError):
        rt_mod.get_function(f"{kernel.attrs['global_symbol']}_auto_output", query_imports=True)


@tilelang.testing.requires_cuda
def test_empty_dynamic_shape_is_allocated_by_tvm_ffi():
    """T.empty shape expressions should be evaluated by the packed ABI binder."""

    @tilelang.jit
    def kernel(A):
        m = T.dynamic("m")
        A: T.Tensor[[m], T.float32]
        B = T.empty([m * 2 + 1], T.float32)
        with T.Kernel(1, threads=1):
            for i in T.serial(m * 2 + 1):
                B[i] = A[i % m] + 1.0
        return B

    a = torch.arange(37, dtype=torch.float32, device="cuda")
    b = kernel(a)

    assert b.shape == (75,)
    torch.testing.assert_close(b, a.repeat(3)[:75] + 1.0)


@tilelang.testing.requires_cuda
def test_empty_dynamic_shape_from_scalar_uses_ffi_allocator_anchor():
    """Scalar-only kernels should still install Torch's FFI allocator."""

    @tilelang.jit
    def kernel(m: T.int32):
        B = T.empty([m + 1], T.float32)
        with T.Kernel(1, threads=1):
            for i in T.serial(m + 1):
                B[i] = T.cast(i, T.float32)
        return B

    b = kernel(23)

    assert b.shape == (24,)
    torch.testing.assert_close(b, torch.arange(24, dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize(
    ("logical_dtype", "expected_torch_dtype", "expected_shape"),
    [
        (T.int4, torch.int8, (3, 8)),
        (
            T.float4_e2m1fn,
            getattr(torch, "float4_e2m1fn_x2", torch.int8),
            (3, 8),
        ),
        pytest.param(
            T.float4_e2m1fnx2,
            getattr(torch, "float4_e2m1fn_x2", torch.int8),
            (3, 16),
            marks=pytest.mark.skipif(
                not hasattr(torch, "float4_e2m1fn_x2"),
                reason="PyTorch float4_e2m1fn_x2 dtype is unavailable",
            ),
        ),
    ],
)
@tilelang.testing.requires_cuda
def test_empty_subbyte_output_uses_torch_storage_dtype(
    logical_dtype,
    expected_torch_dtype,
    expected_shape,
):
    """FFI allocation should expose packed Torch storage for sub-byte TIR."""

    @T.prim_func
    def kernel():
        output = T.empty((3, 16), logical_dtype)
        with T.Kernel(1, threads=1):
            T.evaluate(0)
        return output

    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi")
    output = compiled()

    assert compiled.adapter._ffi_callee_allocated_output_abi
    assert "TVMFFIEnvTensorAlloc" in compiled.get_host_source()
    assert output.dtype == expected_torch_dtype
    assert output.shape == expected_shape


@tilelang.testing.requires_cuda
def test_empty_dynamic_subbyte_output_packs_symbolic_final_dimension():
    """Storage-shape packing should remain symbolic until the packed call."""

    @T.prim_func
    def kernel(n: T.int32):
        output = T.empty((3, n * 2), T.float4_e2m1fn)
        with T.Kernel(1, threads=1):
            T.evaluate(0)
        return output

    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi")
    output = compiled(11)

    expected_dtype = getattr(torch, "float4_e2m1fn_x2", torch.int8)
    assert compiled.adapter.dynamic_symbolic_map is None
    assert output.dtype == expected_dtype
    assert output.shape == (3, 11)


@tilelang.testing.requires_cuda
def test_multiple_empty_outputs_are_returned_from_tvm_ffi():
    """Multiple FFI-allocated outputs should preserve TileLang's list API."""

    @tilelang.jit
    def kernel(A):
        m = T.dynamic("m")
        A: T.Tensor[[m], T.float32]
        B = T.empty([m], T.float32)
        C = T.empty([m + 1], T.float32)
        with T.Kernel(1, threads=1):
            for i in T.serial(m):
                B[i] = A[i] + 1.0
            for i in T.serial(m + 1):
                C[i] = T.cast(i, T.float32)
        return B, C

    a = torch.arange(29, dtype=torch.float32, device="cuda")
    outputs = kernel(a)

    assert isinstance(outputs, list)
    assert [tuple(value.shape) for value in outputs] == [(29,), (30,)]
    torch.testing.assert_close(outputs[0], a + 1.0)
    torch.testing.assert_close(outputs[1], torch.arange(30, dtype=torch.float32, device="cuda"))


def test_all_attrs_together_lazy():
    """annotate_pass_configs, annotate_compile_flags, and out_idx should all work together."""

    @T.prim_func
    def kernel(A):
        A: T.Tensor[[64, 64], T.float32]
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        T.annotate_compile_flags(["--use_fast_math"])
        B = T.empty([64, 64], T.float32)
        with T.Kernel(1):
            for i in T.serial(64):
                for j in T.serial(64):
                    B[i, j] = A[i, j] * 2.0
        return B

    attrs = kernel.attrs
    assert "tilelang_out_idx" in attrs
    assert "tilelang_pass_configs" in attrs
    assert "tilelang_compile_flags" in attrs

    compiled = tilelang.compile(kernel)
    a = torch.randn(64, 64, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a * 2.0)


def test_eager_mode_attrs():
    """Eager mode should support annotate_pass_configs and out_idx via T.empty."""

    @tilelang.jit
    def kernel(A):
        M, N = T.const("M N")
        A: T.Tensor[[M, N], T.float32]
        B = T.empty([M, N], T.float32)
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        with T.Kernel(1):
            for i in T.serial(M):
                for j in T.serial(N):
                    B[i, j] = A[i, j] + 1.0
        return B

    a = torch.randn(32, 32, device="cuda")
    result = kernel(a)
    torch.testing.assert_close(result, a + 1.0)


def test_out_idx_conflict_detection():
    """Specifying both T.empty return and external out_idx should raise ValueError."""

    @T.prim_func
    def kernel(A):
        A: T.Tensor[[32, 32], T.float32]
        B = T.empty([32, 32], T.float32)
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j]
        return B

    with pytest.raises(ValueError, match="Out index conflict"):
        tilelang.compile(kernel, out_idx=[-1])


@tilelang.testing.requires_cuda
def test_manual_out_idx_multiple_dynamic_outputs_are_allocated_by_tvm_ffi():
    """Manual output indices should share T.empty's native allocation ABI."""

    @T.prim_func
    def kernel(A, B, C):
        m = T.dynamic("m")
        A: T.Tensor[[m], T.float32]
        B: T.Tensor[[m * 2 + 1], T.float32]
        C: T.Tensor[[m + 1], T.float32]
        with T.Kernel(1, threads=1):
            for i in T.serial(m * 2 + 1):
                B[i] = A[i % m] + 1.0
            for i in T.serial(m + 1):
                C[i] = T.cast(i, T.float32)

    assert kernel.attrs is None or "tilelang_out_idx" not in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[1, 2], execution_backend="tvm_ffi")
    assert list(compiled.adapter.prim_func.attrs["tilelang_out_idx"]) == [1, 2]
    assert compiled.adapter.dynamic_symbolic_map is None

    a = torch.arange(19, dtype=torch.float32, device="cuda")
    outputs = compiled(a)

    assert isinstance(outputs, list)
    assert [tuple(value.shape) for value in outputs] == [(39,), (20,)]
    torch.testing.assert_close(outputs[0], a.repeat(3)[:39] + 1.0)
    torch.testing.assert_close(outputs[1], torch.arange(20, dtype=torch.float32, device="cuda"))


def test_pass_configs_only_lazy():
    """annotate_pass_configs should work without T.empty or annotate_compile_flags."""

    @T.prim_func
    def kernel(A, B):
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j] + 1.0

    assert "tilelang_pass_configs" in kernel.attrs
    assert kernel.attrs is None or "tilelang_out_idx" not in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[-1])
    a = torch.randn(32, 32, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a + 1.0)


def test_compile_flags_only_lazy():
    """annotate_compile_flags should work standalone."""

    @T.prim_func
    def kernel(A, B):
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        T.annotate_compile_flags(["--use_fast_math"])
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j] + 1.0

    assert "tilelang_compile_flags" in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[-1])
    a = torch.randn(32, 32, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a + 1.0)


def test_annotations_before_tensor_type():
    """Annotations placed before tensor type annotations should work."""

    @T.prim_func
    def kernel(A, B):
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        T.annotate_compile_flags(["--use_fast_math"])
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j] + 1.0

    assert "tilelang_pass_configs" in kernel.attrs
    assert "tilelang_compile_flags" in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[-1])
    a = torch.randn(32, 32, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a + 1.0)


def test_annotations_after_tensor_type():
    """Annotations placed after tensor type annotations should work."""

    @T.prim_func
    def kernel(A, B):
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        T.annotate_compile_flags(["--use_fast_math"])
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j] + 1.0

    assert "tilelang_pass_configs" in kernel.attrs
    assert "tilelang_compile_flags" in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[-1])
    a = torch.randn(32, 32, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a + 1.0)


if __name__ == "__main__":
    test_out_idx_via_attr_lazy()
    test_all_attrs_together_lazy()
    test_eager_mode_attrs()
    test_out_idx_conflict_detection()
    test_pass_configs_only_lazy()
    test_compile_flags_only_lazy()
    test_annotations_before_tensor_type()
    test_annotations_after_tensor_type()
    print("All tests passed!")
