"""pytest tests for example_mhc_norm_fn.py through the AscendC backend."""

import pytest
import torch
import tilelang

from example_mhc_norm_fn import (
    mhc_fn_normw_merge_bwd,
    mhc_fn_normw_merge_fwd,
    mhc_pre_norm_fn_fwd_norm,
    ref_program_fn_normw_merge_bwd,
    ref_program_fn_normw_merge_fwd,
    ref_program_pre_norm_fn_fwd_norm,
)


@pytest.mark.parametrize("target", ["ascend"])
def test_mhc_fn_normw_merge_fwd(target):
    m, n = 24, 7168
    kernel = tilelang.compile(mhc_fn_normw_merge_fwd(m, n), target=target, out_idx=-1)

    device = torch.device("npu")
    fn = torch.randn((m, n), device=device, dtype=torch.float32)
    normw = torch.randn((n,), device=device, dtype=torch.float32)

    actual = kernel(fn, normw)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, ref_program_fn_normw_merge_fwd(fn, normw), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("target", ["ascend"])
def test_mhc_fn_normw_merge_bwd(target):
    m, n = 128, 512
    kernel = tilelang.compile(mhc_fn_normw_merge_bwd(m, n), target=target, out_idx=[-2, -1])

    device = torch.device("npu")
    fn = torch.randn((m, n), device=device, dtype=torch.float32)
    normw = torch.randn((n,), device=device, dtype=torch.float32)
    out_fn_grad = torch.randn((m, n), device=device, dtype=torch.float32)

    fn_grad, normw_grad = kernel(fn, normw, out_fn_grad)
    torch.npu.synchronize()
    expected_fn_grad, expected_normw_grad = ref_program_fn_normw_merge_bwd(fn, normw, out_fn_grad)
    torch.testing.assert_close(fn_grad, expected_fn_grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(normw_grad, expected_normw_grad, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("target", ["ascend"])
def test_mhc_pre_norm_fn_fwd_norm(target):
    num_tokens, mhc_mult3, n_rms_group = 128, 4, 8
    rms_group_size, rms_eps, n_splits = 896, 1e-6, 4

    kernel = tilelang.compile(
        mhc_pre_norm_fn_fwd_norm(mhc_mult3, n_rms_group, rms_group_size, rms_eps, n_splits),
        target=target,
        out_idx=[2, 3, 4],
    )

    device = torch.device("npu")
    out_mul_splitted = torch.randn((n_splits, num_tokens, n_rms_group, mhc_mult3), device=device, dtype=torch.float32)
    sqrsum_splitted = torch.rand((n_splits, num_tokens, n_rms_group), device=device, dtype=torch.float32)

    actual_out_mul, actual_sqrsum, actual_out = kernel(out_mul_splitted, sqrsum_splitted)
    torch.npu.synchronize()
    expected_out_mul, expected_sqrsum, expected_out = ref_program_pre_norm_fn_fwd_norm(
        out_mul_splitted, sqrsum_splitted, rms_group_size, rms_eps
    )
    torch.testing.assert_close(actual_out_mul, expected_out_mul, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(actual_sqrsum, expected_sqrsum, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-5, atol=2e-5)


if __name__ == "__main__":
    test_mhc_fn_normw_merge_fwd("ascend")
    print("PASS: test_mhc_fn_normw_merge_fwd")
    test_mhc_fn_normw_merge_bwd("ascend")
    print("PASS: test_mhc_fn_normw_merge_bwd")
    test_mhc_pre_norm_fn_fwd_norm("ascend")
    print("PASS: test_mhc_pre_norm_fn_fwd_norm")
