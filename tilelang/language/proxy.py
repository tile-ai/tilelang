# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from __future__ import annotations
from typing import Optional

from tvm import tir
from tvm.tir import Var, PrimExpr
from tvm.script.ir_builder.tir import buffer, handle, match_buffer


class BufferProxy:
    """Buffer proxy class for constructing tir buffer."""

    # Index via T.Buffer(...)
    def __call__(
        self,
        shape,
        dtype="float32",
        data=None,
        strides=None,
        elem_offset=None,
        scope="global",
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> tir.Buffer:
        return buffer(
            shape,
            dtype=dtype,
            data=data,
            strides=strides,
            elem_offset=elem_offset,
            scope=scope,
            align=align,
            offset_factor=offset_factor,
            buffer_type=buffer_type,
            axis_separators=axis_separators,
        )

    # Index via T.Buffer[...]
    def __getitem__(self, keys) -> tir.Buffer:
        if not isinstance(keys, tuple):
            return self(keys)
        if len(keys) >= 2 and not isinstance(keys[1], str):
            return self(keys)
        return self(*keys)  # type: ignore[attr-defined] # pylint: disable=no-member

    def from_ptr(self, ptr: Var, shape: tuple[PrimExpr, ...], dtype: str = "float32") -> Buffer:
        return match_buffer(ptr, shape, dtype=dtype)


class TensorProxy:
    """Buffer proxy class for constructing tir buffer."""

    # Index via T.Tensor(...)
    def __call__(
        self,
        shape,
        dtype="float32",
        data=None,
        strides=None,
        elem_offset=None,
        scope="global",
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> tir.Buffer:
        return buffer(
            shape,
            dtype=dtype,
            data=data,
            strides=strides,
            elem_offset=elem_offset,
            scope=scope,
            align=align,
            offset_factor=offset_factor,
            buffer_type=buffer_type,
            axis_separators=axis_separators,
        )

    # Index via T.Tensor[...]
    def __getitem__(self, keys) -> tir.Buffer:
        if not isinstance(keys, tuple):
            return self(keys)
        if len(keys) >= 2 and not isinstance(keys[1], str):
            return self(keys)
        return self(*keys)  # type: ignore[attr-defined] # pylint: disable=no-member

    def from_ptr(self, ptr: Var, shape: tuple[PrimExpr, ...], dtype: str = "float32") -> tir.Buffer:
        return match_buffer(ptr, shape, dtype=dtype)


Buffer = BufferProxy()  # pylint: disable=invalid-name
Tensor = TensorProxy()  # pylint: disable=invalid-name


def ptr(dtype: Optional[str] = None,
        storage_scope: str = "global",
        *,
        is_size_var: bool = False) -> Var:
    """Create a TIR var that represents a pointer.

    Parameters
    ----------
    dtype: str
        The data type of the pointer.

    storage_scope: str
        The storage scope of the pointer.

    is_size_var: bool
        Whether or not to return a SizeVar instead of Var.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type handle or casted expression with type handle.
    """
    return handle(dtype=dtype, storage_scope=storage_scope, is_size_var=is_size_var)
