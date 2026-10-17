"""NaN-propagating clamp lowers for scalar and vectorized SimtVF values."""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("threads", [32, 128])
def test_clamp_nan_and_inverted_bounds(dtype, threads):
    @T.prim_func
    def main(
        source: T.Tensor((128,), dtype),
        lower: T.Tensor((128,), dtype),
        upper: T.Tensor((128,), dtype),
        output: T.Tensor((128,), dtype),
    ):
        with T.Kernel(1), T.SimtVF(threads=threads):
            for i in T.Parallel(128):
                output[i] = T.clamp(source[i], lower[i], upper[i])

    source = torch.tensor([float("nan"), -2, -0.5, 0.5, 2, float("inf"), -float("inf"), 0], dtype=getattr(torch, dtype)).repeat(16)
    lower, upper = torch.full_like(source, -1), torch.full_like(source, 1)
    lower[9], upper[10] = float("nan"), float("nan")
    lower[16:24], upper[16:24] = 5, 2
    expected = torch.clamp(source, lower, upper)

    kernel = tilelang.compile(main, target="ascend", out_idx=3)
    actual = kernel(source.npu(), lower.npu(), upper.npu()).cpu()
    torch.testing.assert_close(actual, expected, equal_nan=True, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
