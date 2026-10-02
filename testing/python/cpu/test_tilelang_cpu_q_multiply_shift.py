"""Compile and execute per-axis fixed-point multiplication with integer flags."""

import pytest
import torch

import tilelang
import tilelang.cpu.language as T
from tilelang import tvm


@pytest.mark.parametrize("flag", [0, 1, False, True], ids=["int-false", "int-true", "bool-false", "bool-true"])
def test_cpu_per_axis_shift_flags(flag):
    size = 7

    @T.prim_func
    def main(X: T.Tensor((size,), "int32"), Y: T.Tensor((size,), "int32"), O: T.Tensor((size,), "int32")):
        with T.Kernel(1):
            for i in T.serial(size):
                O[i] = T.q_multiply_shift_per_axis(X[i], Y[i], 2, 7, 15, flag, 1)

    with tvm.target.Target("c"):
        kernel = tilelang.compile(main, out_idx=[2], target="c", target_host="c", execution_backend="cython")

    x = torch.tensor([-1000, -31, -1, 0, 1, 31, 1000], dtype=torch.int32)
    y = torch.full((size,), 1 << 20, dtype=torch.int32)
    result = kernel(x, y)
    shifted = x.to(torch.int64) << 2 if flag else x.to(torch.int64)
    expected = ((shifted * y.to(torch.int64) + (1 << 21)) >> 22).to(torch.int32)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
