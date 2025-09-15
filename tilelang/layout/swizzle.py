"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation

import tvm
from tilelang import _ffi_api


# Use a stable swizzled layout to ensure consistent memory access patterns.
# Swizzling should be enabled or disabled based on whether TMA (Tensor Memory Access) is applied.
def make_swizzled_layout(buffer: tvm.tir.Buffer):
    assert len(buffer.shape) == 2
    return _ffi_api.make_swizzled_layout(
        int(buffer.shape[0]),
        int(buffer.shape[1]),
        int(tvm.DataType(buffer.dtype).bits),
    )
