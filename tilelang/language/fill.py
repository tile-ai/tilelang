# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm import tir


def fill(buffer: tir.Buffer, value: tir.PrimExpr):
    buffer = buffer.access_ptr("w")
    return tir.call_intrin("handle", tir.op.Op.get("tl.fill"), buffer, value)


def clear(buffer: tir.Buffer):
    return fill(buffer, 0)
