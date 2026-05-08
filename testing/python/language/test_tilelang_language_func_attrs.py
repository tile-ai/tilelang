"""Test T.annotate_compile_flags, T.annotate_pass_configs, and out_idx via PrimFunc attrs."""

import pytest
import torch
import tilelang
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


def test_no_out_idx_when_not_using_empty():
    """When T.empty is not used, tilelang_out_idx attr should not be present."""

    @T.prim_func
    def kernel(A, B):
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j]

    assert kernel.attrs is None or "tilelang_out_idx" not in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[-1])
    a = torch.randn(32, 32, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a)


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
    test_no_out_idx_when_not_using_empty()
    test_pass_configs_only_lazy()
    test_compile_flags_only_lazy()
    test_annotations_before_tensor_type()
    test_annotations_after_tensor_type()
    print("All tests passed!")
