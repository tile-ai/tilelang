import math

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang import tvm


@pytest.mark.parametrize(
    "src_shape",
    [
        (2, 3, 4),
        (2, 3, 4, 5),
        (2, 3, 1),
        (2, 1, 3),
    ],
)
def test_cpu_transpose_swaps_only_the_final_two_axes(src_shape):
    dst_shape = (*src_shape[:-2], src_shape[-1], src_shape[-2])

    @T.prim_func
    def main(
        src: T.Tensor(src_shape, T.float32),
        dst: T.Tensor(dst_shape, T.float32),
    ):
        with T.Kernel(1):
            T.transpose(src, dst)

    with tvm.target.Target("c"):
        kernel = tilelang.compile(
            main,
            out_idx=[1],
            target="c",
            target_host="c",
            execution_backend="cython",
        )

    src = torch.arange(math.prod(src_shape), dtype=torch.float32).reshape(src_shape)
    actual = kernel(src)
    expected = src.transpose(-2, -1)

    torch.testing.assert_close(actual, expected)
