# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from __future__ import annotations
from typing import Optional

from tvm import tir
from tvm.tir import Var, PrimExpr
from tvm.script.ir_builder.tir import buffer, handle, match_buffer
from tilelang.utils import deprecated


class BufferProxy:
    """Buffer proxy class for constructing tir buffer."""

    # Index via T.Buffer(...)
    @deprecated("T.Buffer(...)", "T.Tensor(...)")
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
    @deprecated("T.Buffer[...]", "T.Tensor(...)")
    def __getitem__(self, keys) -> tir.Buffer:
        if not isinstance(keys, tuple):
            return self(keys)
        if len(keys) >= 2 and not isinstance(keys[1], str):
            return self(keys)
        return self(*keys)  # type: ignore[attr-defined] # pylint: disable=no-member

    def from_ptr(self,
                 pointer_var: Var,
                 shape: tuple[PrimExpr, ...],
                 dtype: str = "float32") -> Buffer:
        """Create a buffer from a pointer, shape, and data type.

        Args:
            pointer_var: The pointer variable
            shape: The shape of the buffer
            dtype: The data type of the buffer (default: float32)

        Returns:
            A buffer created from the given parameters
        """
        return match_buffer(pointer_var, shape, dtype=dtype)


class BaseTensorProxy:
    """Base proxy class for tensor types with configurable defaults.
    
    This class serves as a foundation for different tensor proxy types, providing
    customizable default values for scope, alignment, and offset factors. It implements
    the core functionality for creating TIR buffers with specific memory configurations.
    """
    default_scope = "global"
    default_align = 0
    default_offset_factor = 0

    def __call__(
        self,
        shape,
        dtype="float32",
        data=None,
        strides=None,
        elem_offset=None,
        scope=None,  # Changed to None to use class default
        align=None,
        offset_factor=None,
        buffer_type="",
        axis_separators=None,
    ) -> tir.Buffer:
        # Use class defaults if not specified
        scope = scope or self.default_scope
        align = align or self.default_align
        offset_factor = offset_factor or self.default_offset_factor

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

    def __getitem__(self, keys) -> tir.Buffer:
        if not isinstance(keys, tuple):
            return self(keys)
        if len(keys) >= 2 and not isinstance(keys[1], str):
            return self(keys)
        return self(*keys)

    def from_ptr(self,
                 pointer_var: Var,
                 shape: tuple[PrimExpr, ...],
                 dtype: str = "float32") -> tir.Buffer:
        """Create a buffer from a pointer, shape, and data type.

        Args:
            pointer_var: The pointer variable
            shape: The shape of the buffer
            dtype: The data type of the buffer (default: float32)

        Returns:
            A buffer created from the given parameters
        """
        return match_buffer(pointer_var, shape, dtype=dtype)


class TensorProxy(BaseTensorProxy):
    """Main tensor proxy class for global scope buffers.
    
    This class implements the default tensor proxy with global memory scope,
    inheriting all functionality from BaseTensorProxy without modifications.
    """


class FragmentBufferProxy(BaseTensorProxy):
    """Proxy class for fragment memory buffers.
    
    This class represents tensor proxies specifically for local fragment memory,
    typically used in GPU tensor core operations.
    """
    default_scope = "local.fragment"


class SharedBufferProxy(BaseTensorProxy):
    """Proxy class for shared memory buffers.
    
    This class represents tensor proxies for dynamic shared memory,
    commonly used in GPU shared memory operations.
    """
    default_scope = "shared.dyn"


class LocalBufferProxy(BaseTensorProxy):
    """Proxy class for local memory buffers.
    
    This class represents tensor proxies for local memory scope,
    typically used for temporary computations in GPU kernels.
    """
    default_scope = "local"


Buffer = BufferProxy()  # pylint: disable=invalid-name
# Tensor is an alias for Buffer
# Because when user do jit compile, the input and output will
# be mapped with torch.Tensor.
Tensor = TensorProxy()  # pylint: disable=invalid-name
FragmentBuffer = FragmentBufferProxy()  # pylint: disable=invalid-name
SharedBuffer = SharedBufferProxy()  # pylint: disable=invalid-name
LocalBuffer = LocalBufferProxy()  # pylint: disable=invalid-name


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


def make_tensor(ptr: Var, shape: tuple[PrimExpr, ...], dtype: str = "float32") -> tir.Buffer:
    return Tensor.from_ptr(ptr, shape, dtype)
