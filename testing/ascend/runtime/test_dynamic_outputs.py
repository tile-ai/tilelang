"""Dynamic output allocation preserves Ascend argument and stream handling."""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


pytest.importorskip("torch_npu")

ROWS = T.dynamic("rows")
WIDTH = 128


@T.prim_func
def dynamic_outputs(
    positive: T.Tensor((ROWS, WIDTH), T.float32),
    source: T.Tensor((ROWS, WIDTH), T.float32),
    negative: T.Tensor((ROWS, WIDTH), T.float32),
    bias: T.int32,
):
    with T.Kernel(ROWS) as row, T.SimtVF(threads=WIDTH):
        for i in T.Parallel(WIDTH):
            positive[row, i] = source[row, i] + T.float32(bias)
            negative[row, i] = source[row, i] - T.float32(bias)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("execution_backend", ["cython", "tvm_ffi"])
@pytest.mark.parametrize("rows", [1, 3])
def test_dynamic_outputs_before_sizing_input(execution_backend, rows):
    kernel = tilelang.compile(dynamic_outputs, out_idx=[0, 2], target="ascend", execution_backend=execution_backend)
    reference = torch.arange(rows * WIDTH, dtype=torch.float32).reshape(rows, WIDTH)
    source = reference.npu()
    torch.npu.synchronize()

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        prepared = source + 17
        positive, negative = kernel(prepared, 9)
    stream.synchronize()

    assert positive.device == negative.device == source.device
    torch.testing.assert_close(positive.cpu(), reference + 26, atol=0, rtol=0)
    torch.testing.assert_close(negative.cpu(), reference + 8, atol=0, rtol=0)


@T.prim_func
def expression_outputs(source: T.Tensor((ROWS, WIDTH), T.float32), count: T.int32):
    doubled = T.empty((ROWS * 2, WIDTH), T.float32)
    rounded = T.empty((T.ceildiv(ROWS, 2), WIDTH), T.float32)
    mixed = T.empty((count, count * 2), T.float32)
    with T.Kernel(1), T.SimtVF(threads=WIDTH):
        for row in T.serial(ROWS):
            for i in T.Parallel(WIDTH):
                doubled[row * 2, i] = source[row, i] + 1.0
                doubled[row * 2 + 1, i] = source[row, i] - 1.0
        for row in T.serial(T.ceildiv(ROWS, 2)):
            for i in T.Parallel(WIDTH):
                rounded[row, i] = source[row * 2, i] + 2.0
        for row in T.serial(count):
            for i in T.Parallel(WIDTH):
                if i < count * 2:
                    mixed[row, i] = source[0, i] + 3.0
    return doubled, rounded, mixed


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "target, execution_backend",
    [
        ("ascend", "cython"),
        ("ascend", "tvm_ffi"),
        pytest.param("pto", "pto", marks=pytest.mark.pto),
    ],
)
def test_expression_outputs(target, execution_backend):
    kernel = tilelang.compile(expression_outputs, target=target, execution_backend=execution_backend)
    for rows, count in [(1, 1), (3, 2), (0, 0)]:
        reference = torch.arange(rows * WIDTH, dtype=torch.float32).reshape(rows, WIDTH)
        doubled, rounded, mixed = kernel(reference.npu(), count)
        torch.npu.synchronize()
        expected = torch.stack((reference + 1, reference - 1), dim=1).reshape(rows * 2, WIDTH)
        torch.testing.assert_close(doubled.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(rounded.cpu(), reference[::2] + 2, atol=0, rtol=0)
        torch.testing.assert_close(mixed.cpu(), (reference[:1, : count * 2] + 3).repeat(count, 1), atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
