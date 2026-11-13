"""Regression tests for invalid Ascend launch/pipeline parameters (#3920, #3922)."""

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm

tirx = tvm.tirx


def _kernel_with_blocks(blocks):
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(blocks):
            pass

    return main


def _kernel_with_stages(num_stages):
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            for i in T.Pipelined(8, num_stages=num_stages):
                A[i] = A[i] + 1

    return main


@pytest.mark.parametrize("blocks", [-8, 0])
def test_kernel_rejects_non_positive_blocks(blocks):
    with pytest.raises(ValueError, match="block counts must be positive"):
        tilelang.compile(_kernel_with_blocks(blocks), target="ascend")


def test_pipelined_rejects_negative_num_stages():
    with pytest.raises(ValueError, match="num_stages must be >= 0"):
        tilelang.compile(_kernel_with_stages(-2), target="ascend")


def test_kernel_rejects_negative_constant_prim_expr():
    with pytest.raises(ValueError, match="block counts must be positive"):
        tilelang.compile(_kernel_with_blocks(tirx.IntImm("int32", -1)), target="ascend")
