import math

import pytest
import torch

import tilelang
import tilelang.cpu.language as T
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


# ---------------------------------------------------------------------------
# Shape contract: dst must be src with its final two axes swapped.
#
# The check lives in the C++ transpose operator (after region normalization), so
# it also covers a `tl.tileop.transpose` call built without the Python wrapper.
# Extent equality is decided by `arith::Analyzer`, not by comparing Python lists:
# two independently constructed symbolic expressions such as `n + 1` and `n + 1`
# must still be accepted.
# ---------------------------------------------------------------------------


def _transpose_kernel(src_shape, dst_shape):
    @T.prim_func
    def main(
        src: T.Tensor(src_shape, T.float32),
        dst: T.Tensor(dst_shape, T.float32),
    ):
        with T.Kernel(1):
            T.transpose(src, dst)

    return main


@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ((4, 8), (9, 4)),
        ((4, 8), (4, 8)),
        ((2, 3, 4), (3, 4, 2)),
    ],
)
def test_cpu_transpose_rejects_a_destination_that_is_not_the_axis_swap(src_shape, dst_shape):
    with (
        tvm.target.Target("c"),
        pytest.raises(
            tvm.error.InternalError,
            match="Transpose destination shape must be the source shape with the final two axes swapped",
        ),
    ):
        tilelang.lower(_transpose_kernel(src_shape, dst_shape), target="c")


def test_cpu_transpose_rejects_a_destination_with_a_different_rank():
    with (
        tvm.target.Target("c"),
        pytest.raises(tvm.error.InternalError, match="Transpose requires src and dst to have the same rank"),
    ):
        tilelang.lower(_transpose_kernel((2, 3, 4), (4, 3)), target="c")


def test_cpu_transpose_accepts_symbolic_extents_that_the_analyzer_proves_equal():
    n = T.symbolic("n")

    @T.prim_func
    def main(
        src: T.Tensor((3, n + 1), T.float32),
        dst: T.Tensor((n + 1, 3), T.float32),
    ):
        with T.Kernel(1):
            T.transpose(src, dst)

    with tvm.target.Target("c"):
        tilelang.lower(main, target="c")


def test_cpu_transpose_rejects_a_directly_built_tileop_with_a_mismatched_destination():
    from tilelang.utils.language import to_tile_region

    @T.prim_func
    def main(
        src: T.Tensor((4, 8), T.float32),
        dst: T.Tensor((9, 4), T.float32),
    ):
        with T.Kernel(1):
            T.evaluate(
                T.call_intrin(
                    "handle",
                    tvm.ir.Op.get("tl.tileop.transpose"),
                    to_tile_region(src, "r"),
                    to_tile_region(dst, "w"),
                )
            )

    with (
        tvm.target.Target("c"),
        pytest.raises(
            tvm.error.InternalError,
            match="Transpose destination shape must be the source shape with the final two axes swapped",
        ),
    ):
        tilelang.lower(main, target="c")
