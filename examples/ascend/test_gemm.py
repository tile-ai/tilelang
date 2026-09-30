"""Persistent GEMM, epilogue and precision modes."""

import pytest
import torch
import tilelang
import tilelang.testing
from example_gemm import gemm


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
@pytest.mark.parametrize(
    "dtype,out_dtype,mixed,hf32,unit_flag",
    [
        ("float8_e4m3fn", "float32", False, None, True),
        ("float8_e4m3fn", "float32", True, None, True),
        ("bfloat16", "float32", False, None, False),
        ("bfloat16", "bfloat16", False, None, True),
        ("bfloat16", "float32", None, None, True),
        ("float32", "float32", None, None, True),
        ("float32", "float32", None, "nearest_zero", True),
        ("float32", "float32", None, "nearest_even", True),
    ],
    ids=["fp8-cube", "fp8-mixed", "bf16-no-unit-flag", "bf16-output", "bf16-persistent", "fp32", "hf32-truncate", "hf32-round"],
)
def test_gemm(dtype, out_dtype, mixed, hf32, unit_flag, target):
    # More tiles than cores exercise persistent traversal and alternating swizzle windows.
    m, n = (2048, 1280) if dtype == "bfloat16" and mixed is None else (256, 256)
    k = 512
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(m, k, generator=generator).to(getattr(torch, dtype))
    b = torch.randn(n, k, generator=generator).to(getattr(torch, dtype))
    kernel = tilelang.compile(gemm(m, k, n, dtype, out_dtype, mixed, hf32, unit_flag), target=target, out_idx=-1)
    tolerance = 0.1 if hf32 else 1e-2 if out_dtype == "bfloat16" else 1e-3
    expected = (a.float() @ b.float().T).to(getattr(torch, out_dtype))
    torch.testing.assert_close(kernel(a.npu(), b.npu()).cpu(), expected, rtol=tolerance, atol=tolerance)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
@pytest.mark.parametrize("out_dtype", ["float32", "bfloat16"])
def test_gemm_accumulate(out_dtype, target):
    a = (torch.randint(-4, 5, (256, 512)).float() / 8).to(torch.bfloat16)
    b = (torch.randint(-4, 5, (256, 512)).float() / 8).to(torch.bfloat16)
    initial = torch.ones(256, 256, dtype=getattr(torch, out_dtype))
    output = initial.npu()
    kernel = tilelang.compile(gemm(256, 512, 256, out_dtype=out_dtype, acc=True), target=target)
    kernel(a.npu(), b.npu(), output)
    expected = (a.float() @ b.float().T).to(initial.dtype) + initial
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
