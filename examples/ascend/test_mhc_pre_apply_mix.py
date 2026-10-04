"""pytest tests for example_mhc_pre_apply_mix.py through the AscendC backend."""

import pytest
import torch
import tilelang

from example_mhc_pre_apply_mix import mhc_pre_apply_mix_fwd, ref_program


@pytest.mark.parametrize("target", ["ascend"])
def test_mhc_pre_apply_mix_fwd(target):
    num_tokens, mhc_mult, hidden = 128, 4, 1280

    # `tl.disable_shared_memory_reuse` avoids the auto-schedule buffer-alias
    # inference aliasing the two multi-versioned buffers `xs` and `os`.
    kernel = tilelang.compile(
        mhc_pre_apply_mix_fwd(mhc_mult, hidden),
        target=target,
        out_idx=-1,
        pass_configs={"tl.disable_shared_memory_reuse": True},
    )

    device = torch.device("npu")
    x = torch.randn((num_tokens, mhc_mult, hidden), device=device, dtype=torch.bfloat16)
    mix = torch.randn((num_tokens, mhc_mult), device=device, dtype=torch.float32)

    actual = kernel(x, mix)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, ref_program(x, mix), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_mhc_pre_apply_mix_fwd("ascend")
    print("PASS: test_mhc_pre_apply_mix_fwd")
