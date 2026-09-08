"""Ascend on-chip buffer allocation: L1 (CBuf) and the L0A/L0B/L0C Cube scopes."""

from __future__ import annotations

from tilelang._typing import DType, ShapeType
from tilelang.language.allocate import _with_span
from tvm.script import tirx as T
from tvm.tirx.buffer import Buffer
from tvm.tirx.script.builder.ir import sblock_attr


def alloc_l1(shape: ShapeType, dtype: DType, scope="shared.l1") -> Buffer:
    """Allocate an L1 buffer (__cbuf__) on Ascend NPU.

    Args:
        shape: Buffer shape
        dtype: Data type
        scope: Memory scope. Defaults to "shared.l1"
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_l0a(shape: ShapeType, dtype: DType, scope="shared.l0a") -> Buffer:
    """Allocate an L0A buffer (__ca__) on Ascend NPU.

    Layout is deferred to the consuming gemm operation, which infers K-major
    or MN-major from its transpose flags.
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_l0b(shape: ShapeType, dtype: DType, scope="shared.l0b") -> Buffer:
    """Allocate an L0B buffer (__cb__) on Ascend NPU.

    Layout is deferred to the consuming gemm operation, which infers K-major
    or MN-major from its transpose flags.
    """
    return _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))


def alloc_l0c(shape: ShapeType, dtype: DType, scope="shared.l0c", layout: bool = True) -> Buffer:
    """Allocate an L0C buffer (__cc__) on Ascend NPU.

    Args:
        shape: Buffer shape
        dtype: Data type
        scope: Memory scope. Defaults to "shared.l0c"
        layout: Whether to annotate the fixed Ascend L0C accumulator layout.
    """
    buf = _with_span(T.sblock_alloc_buffer(shape, dtype, scope=scope))
    if layout:
        from tilelang.layout import make_ascend_l0c_layout

        if len(buf.shape) >= 2:
            sblock_attr({"layout_map": {buf.data: make_ascend_l0c_layout(buf)}})
    return buf


__all__ = [
    "alloc_l1",
    "alloc_l0a",
    "alloc_l0b",
    "alloc_l0c",
]
