import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.utils.tensor import torch_assert_close

from example_simdvf_per_token_cast_to_fp8 import test as _run_simdvf_correctness
from example_simtvf_per_token_cast_to_fp8 import (
    per_token_cast_to_fp8 as _simtvf_per_token_cast_to_fp8,
    ref_program as _simtvf_ref_program,
)


TEST_M = 8192
TEST_N = 8192


@pytest.mark.parametrize("backend", ["asc"])
def test_simdvf_per_token_cast_to_fp8(backend):
    _run_simdvf_correctness(TEST_M, TEST_N, backend=backend, print_source=False)


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_simtvf_per_token_cast_to_fp8(target):
    kernel = tilelang.compile(
        _simtvf_per_token_cast_to_fp8.get_tir(TEST_M, TEST_N),
        target=target,
        out_idx=[1, 2],
    )
    device = torch.device("npu")
    x = torch.randn(TEST_M, TEST_N, device=device, dtype=torch.float32)

    x_fp8, x_amax = kernel(x)
    torch.npu.synchronize()
    x_fp8_ref, x_amax_ref = _simtvf_ref_program(x)

    torch_assert_close(x_fp8.to(torch.float32), x_fp8_ref.to(torch.float32), rtol=0.01, atol=0.01)
    torch_assert_close(x_amax, x_amax_ref, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    tilelang.testing.main()
