"""KernelParam.storage_shape must hand PyTorch the logical shape.

This backend has no packed storage ABI, so a sub-byte dtype such as
float4_e2m1fn is exposed to PyTorch element for element rather than being
packed two-per-byte.
"""

import pytest

import tilelang.ascend.language as T
from tilelang.engine.param import KernelParam


def test_bool_storage_shape_is_the_logical_shape():
    boolean = KernelParam(T.bool, [4, 4])

    assert boolean.storage_shape() == [4, 4]


@pytest.mark.parametrize("dtype", [T.float4_e2m1fn, T.int4, T.dtype("uint4")])
def test_sub_byte_storage_shape_is_the_logical_shape(dtype):
    packed = KernelParam(dtype, [4, 256])

    assert packed.storage_shape() == [4, 256]
