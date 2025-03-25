# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation

import tvm
from tilelang import _ffi_api


def make_swizzled_layout(buffer: tvm.tir.Buffer):
    assert len(buffer.shape) == 2
    return _ffi_api.make_swizzled_layout(
        int(buffer.shape[0]),
        int(buffer.shape[1]),
        int(tvm.DataType(buffer.dtype).bits),
    )
