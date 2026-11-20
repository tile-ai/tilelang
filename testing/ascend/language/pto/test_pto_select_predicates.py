"""PTO selects accept IR i1 predicates even when Python folds an expression."""

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


pytestmark = pytest.mark.pto


def nan_safe_relu(dtype, constant_true):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), B: T.Tensor((128,), dtype)):
        with T.Kernel(1):  # noqa: SIM117
            with T.SimtVF(threads=128):
                for i in T.Parallel(128):
                    x = T.Cast("float32", A[i])
                    z = T.Cast("float32", T.float16(0))
                    value = T.Select(x < z, z, x)
                    if constant_true:
                        value = T.Select(T.isnan(z) == False, value, z)  # noqa: E712
                    else:
                        value = T.Select(T.isnan(z), z, value)
                    B[i] = T.Cast(dtype, T.Select(T.isnan(x), x, value))

    return main


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("constant_true", [False, True])
def test_nan_safe_relu_literal_and_runtime_predicates(dtype, constant_true):
    kernel = tilelang.compile(nan_safe_relu(dtype, constant_true), target="pto", execution_backend="pto", out_idx=-1)
    source = kernel.get_kernel_source()
    assert "pto.select(tl.as_ir_bool(" in source
    assert "pto.select(tl.as_logical_bool(" not in source
    values = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 0.0, -3.0, 2.0, 0.5], dtype=getattr(torch, dtype))
    previous = None
    for scale in (1, 2):
        x = (values * scale).repeat(16)
        # Include both signed zeros and check numerical equality; this test
        # does not require bitwise zero-sign preservation by PTOAS.
        expected = torch.where(x < 0, torch.zeros_like(x), x)
        actual = kernel(x.npu())
        torch.npu.synchronize()
        got = actual.cpu()
        torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)
        if previous is not None:
            torch.testing.assert_close(previous[0].cpu(), previous[1], rtol=0, atol=0, equal_nan=True)
        previous = (actual, got.clone())


def boolean_result(output_dtype, select_branches):
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), output_dtype)):
        with T.Kernel(1):  # noqa: SIM117
            with T.SimtVF(threads=128):
                for i in T.Parallel(128):
                    z = T.Cast("float32", T.float16(0))
                    if select_branches:
                        value = T.Select(A[i] < z, T.isnan(z), T.isnan(z) == False)  # noqa: E712
                    else:
                        value = T.isnan(z)
                    B[i] = T.Cast(output_dtype, value)

    return main


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("output_dtype", ["float32", "int32"])
@pytest.mark.parametrize("select_branches", [False, True])
def test_literal_boolean_cast_and_select_branches(output_dtype, select_branches):
    kernel = tilelang.compile(boolean_result(output_dtype, select_branches), target="pto", execution_backend="pto", out_idx=-1)
    x = torch.arange(128, dtype=torch.float32) - 64
    expected = (x >= 0).to(getattr(torch, output_dtype)) if select_branches else torch.zeros(128, dtype=getattr(torch, output_dtype))
    actual = kernel(x.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_byte_backed_predicates_preserve_nonzero_truthiness():
    @T.prim_func
    def main(A: T.Tensor((128,), "int8"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1):  # noqa: SIM117
            with T.SimtVF(threads=128):
                for i in T.Parallel(128):
                    B[i] = T.Cast("float32", T.Cast("bool", A[i]))

    kernel = tilelang.compile(main, target="pto", execution_backend="pto", out_idx=-1)
    x = torch.tensor([0, 1, 2, -2, 127, -128, -1, 4], dtype=torch.int8).repeat(16)
    actual = kernel(x.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), (x != 0).float(), rtol=0, atol=0)
