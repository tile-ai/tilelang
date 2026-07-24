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


def test_cpu_transpose_maps_destination_boundary_predicate():
    src_shape = (2, 3, 4)
    dst_shape = (2, 4, 3)

    @T.prim_func
    def main(
        src: T.Tensor(src_shape, T.float32),
        dst: T.Tensor(dst_shape, T.float32),
    ):
        with T.Kernel(1):
            T.fill(dst, -1.0)
            T.transpose(
                src[0:2, 0:3, 0:4],
                dst[0:2, 0:4, 1:4],
            )

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
    expected = torch.full(dst_shape, -1.0)
    expected[:, :, 1:] = src[:, :2, :].transpose(-2, -1)

    torch.testing.assert_close(actual, expected)


def test_cpu_transpose_guards_out_of_bounds_singleton_destination():
    src_shape = (2, 3, 1)
    dst_shape = (2, 1, 3)

    @T.prim_func
    def main(
        src: T.Tensor(src_shape, T.float32),
        dst: T.Tensor(dst_shape, T.float32),
    ):
        with T.Kernel(1):
            T.fill(dst, -1.0)
            T.transpose(
                src[0:2, 0:3, 0:1],
                dst[0:2, 1:2, 0:3],
            )

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
    expected = torch.full(dst_shape, -1.0)

    torch.testing.assert_close(actual, expected)


def test_cpu_transpose_guards_out_of_bounds_singleton_source():
    src_shape = (2, 3, 1)
    dst_shape = (2, 1, 3)

    @T.prim_func
    def main(
        src: T.Tensor(src_shape, T.float32),
        dst: T.Tensor(dst_shape, T.float32),
    ):
        with T.Kernel(1):
            T.transpose(
                src[0:2, 0:3, 1:2],
                dst[0:2, 0:1, 0:3],
            )

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
    expected = torch.zeros(dst_shape, dtype=torch.float32)

    torch.testing.assert_close(actual, expected)
