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


if __name__ == "__main__":
    tilelang.testing.main()
