"""pytest tests for example_mhc_head_compute_mix.py through the AscendC backend."""

import pytest
import torch
import tilelang

from example_mhc_head_compute_mix import mhc_head_compute_mix_fwd, ref_program


@pytest.mark.parametrize("target", ["ascend"])
def test_mhc_head_compute_mix(target):
    num_tokens, mhc_mult = 128, 4
    mhc_pre_eps = 1e-6
    token_block_size = 32

    kernel = tilelang.compile(
        mhc_head_compute_mix_fwd(mhc_mult, mhc_pre_eps, token_block_size),
        target=target,
        out_idx=-1,
    )

    device = torch.device("npu")
    input_mix = torch.randn((num_tokens, mhc_mult), device=device, dtype=torch.float32)
    mhc_scale = torch.randn((1,), device=device, dtype=torch.float32)
    mhc_base = torch.randn((mhc_mult,), device=device, dtype=torch.float32)

    actual = kernel(input_mix, mhc_scale, mhc_base)
    torch.npu.synchronize()

    expected = ref_program(input_mix, mhc_scale, mhc_base, mhc_pre_eps)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)


if __name__ == "__main__":
    test_mhc_head_compute_mix("ascend")
    print("PASS: test_mhc_head_compute_mix")
