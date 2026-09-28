"""pytest test for example_while_pipelined.py — while-loop auto-schedule."""

import torch
import tilelang

import pytest

from example_while_pipelined import while_pipelined, ref_program, N


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_while_pipelined(target):
    _run_while_pipelined(target)


def _run_while_pipelined(target):
    device = torch.device("npu")
    torch.manual_seed(0)
    compile_kwargs = {"out_idx": -1, "target": target}
    kernel = tilelang.compile(while_pipelined(), **compile_kwargs)
    a = torch.randn(N, dtype=torch.float32, device=device)
    out = kernel(a)
    torch.npu.synchronize()
    expected = ref_program(a)
    max_diff = (out - expected).abs().max().item()
    assert max_diff < 1e-2, f"max_diff={max_diff:.2e}"


if __name__ == "__main__":
    for target in ("ascend", "pto"):
        test_while_pipelined(target)
        print(f"PASS: test_while_pipelined[{target}]")
