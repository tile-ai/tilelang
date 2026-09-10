"""pytest test for example_gemm.py — bf16 + fp32 auto GEMM."""

import torch
import tilelang
import pytest
from example_gemm import gemm, ref_program

TARGETS = ["ascend"]


def _test(dtype, thresh, out_dtype="float32", target="ascend", mixed=None, hf32=None):
    M, K, N = 8192, 8192, 8192
    kernel = tilelang.compile(
        gemm(M, K, N, dtype=dtype, out_dtype=out_dtype, MIXED=mixed, hf32=hf32),
        target=target,
        out_idx=-1,
    )
    device = torch.device("npu")
    x = torch.randn(M, K, device=device).to(dtype=getattr(torch, dtype))
    w = torch.randn(N, K, device=device).to(dtype=getattr(torch, dtype))
    c = kernel(x, w)
    torch.npu.synchronize()
    expected = ref_program(x, w, out_dtype=out_dtype)
    max_diff = (c - expected).abs().max().item()
    assert max_diff < thresh, f"{dtype} target={target} max_diff={max_diff:.2e}"


@pytest.mark.parametrize("target", TARGETS)
def test_gemm_auto_fp8(target):
    _test("float8_e4m3fn", 1e-1, target=target, mixed=False)


@pytest.mark.parametrize("target", TARGETS)
def test_gemm_auto_fp8_mixed(target):
    _test("float8_e4m3fn", 1e-1, target=target, mixed=True)


@pytest.mark.parametrize("target", TARGETS)
def test_gemm_auto_bf16(target):
    _test("bfloat16", 1e-2, target=target, mixed=False)
    _test("bfloat16", 1e-2, "bfloat16", target=target, mixed=False)


@pytest.mark.parametrize("target", TARGETS)
def test_gemm_auto_bf16_mixed(target):
    _test("bfloat16", 1e-2, target=target, mixed=True)


@pytest.mark.parametrize("target", TARGETS)
def test_gemm_auto_fp32(target):
    _test("float32", 5e-3, target=target, mixed=False)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("hf32", ["nearest_zero", "nearest_even"])
def test_gemm_auto_fp32_hf32(target, hf32):
    _test("float32", 2e-1, target=target, mixed=False, hf32=hf32)


def _test_acc(out_dtype, thresh, target="ascend"):
    """C += A @ B^T via store-mode atomic; C is an in/out buffer."""
    M, K, N = 8192, 8192, 8192
    device = torch.device("npu")
    kernel = tilelang.compile(
        gemm(M, K, N, dtype="bfloat16", out_dtype=out_dtype, acc=True),
        target=target,
    )
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    c0 = torch.randn(M, N, device=device, dtype=getattr(torch, out_dtype))
    expected = ref_program(x, w, out_dtype=out_dtype, c=c0)
    c = c0.clone()
    kernel(x, w, c)
    torch.npu.synchronize()
    # Relative error: bf16 output rounds each element, so a max-abs bound is
    # dominated by rounding of large K=8192 accumulations.
    rel = (c.float() - expected.float()).abs().mean().item() / expected.float().abs().mean().clamp_min(1e-6).item()
    assert rel < thresh, f"acc target={target} out={out_dtype} rel={rel:.2e}"


@pytest.mark.parametrize("target", TARGETS)
def test_gemm_auto_acc(target):
    _test_acc("float32", 1e-2, target=target)
    _test_acc("bfloat16", 1e-2, target=target)


if __name__ == "__main__":
    test_gemm_auto_fp8("ascend")
    print("PASS: test_gemm_auto_fp8")
    test_gemm_auto_fp8_mixed("ascend")
    print("PASS: test_gemm_auto_fp8_mixed")
    test_gemm_auto_bf16("ascend")
    print("PASS: test_gemm_auto_bf16")
    test_gemm_auto_bf16_mixed("ascend")
    print("PASS: test_gemm_auto_bf16_mixed")
    test_gemm_auto_fp32("ascend")
    print("PASS: test_gemm_auto_fp32")
    for hf32 in ["nearest_zero", "nearest_even"]:
        test_gemm_auto_fp32_hf32("ascend", hf32)
    print("PASS: test_gemm_auto_fp32_hf32")
    test_gemm_auto_acc("ascend")
    print("PASS: test_gemm_auto_acc")
