"""pytest tests for example_rmsnorm.py through the AscendC backend."""

import pytest
import torch
import tilelang

from example_rmsnorm import ref_program, rms_norm_fwd


DS = [4096, 5120, 7168]


def _run_rmsnorm(d, target):
    batch = 4096
    kernel = tilelang.compile(rms_norm_fwd(batch, d, "float32"), target=target, out_idx=[1, 3])
    device = torch.device("npu")
    x = torch.randn(batch, d, dtype=torch.float32, device=device)
    weight = torch.randn(d, dtype=torch.float32, device=device)
    eps = 1e-6

    y_flat, _rstd = kernel(x.view(-1).contiguous(), weight, eps)
    torch.npu.synchronize()

    y = y_flat.view(batch, d)
    expected = ref_program(x, weight, eps)
    max_diff = (y.float() - expected).abs().max().item()
    assert max_diff < 1e-3, f"max_diff={max_diff:.2e}"


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
@pytest.mark.parametrize("d", DS)
def test_rmsnorm(d, target):
    _run_rmsnorm(d, target)


if __name__ == "__main__":
    for target in ("ascend", "pto"):
        for d in DS:
            test_rmsnorm(d, target)
            print(f"PASS: test_rmsnorm d={d} target={target}")
