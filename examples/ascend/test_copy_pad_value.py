"""pytest for example_copy_pad_value.py — Ascend GM->UB copy padding modes."""

import pytest
import torch
import tilelang

from example_copy_pad_value import copy_pad_value, copy_data_select


def _run(program, M, N, N_pad, fill, target):
    device = torch.device("npu")
    kernel = tilelang.compile(program, target=target, out_idx=-1)
    torch.manual_seed(0)
    a = torch.randn(M, N, dtype=torch.float32, device=device)
    out = kernel(a)
    torch.npu.synchronize()
    # Valid region matches the source; the padded tail holds the fill value.
    torch.testing.assert_close(out[:, :N], a)
    expected_tail = torch.full((M, N_pad - N), fill, dtype=torch.float32, device=device)
    torch.testing.assert_close(out[:, N:], expected_tail)


@pytest.mark.parametrize("target", ["ascend"])
def test_copy_pad_value_mode(target):
    M, N, N_pad, fill = 4, 30, 32, -1.0
    _run(copy_pad_value(M, N, N_pad, fill), M, N, N_pad, fill, target)


@pytest.mark.parametrize("target", ["ascend"])
def test_copy_data_select_mode(target):
    M, N, N_pad, fill = 4, 30, 32, -1.0
    _run(copy_data_select(M, N, N_pad, fill), M, N, N_pad, fill, target)


if __name__ == "__main__":
    for target in ("ascend",):
        test_copy_pad_value_mode(target)
        test_copy_data_select_mode(target)
        print(f"PASS: test_copy_pad_value ({target})")
