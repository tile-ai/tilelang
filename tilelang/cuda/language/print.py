"""
This module provides macros and utilities for debugging TileLang (tl) programs.
It includes functionality to print variables, print values in buffers, conditionally execute debug prints and assert.
"""

from tilelang.language import _construction
from tilelang.language import _intrinsics
from tilelang.language import _tensor_spec
from tvm_tirx import ir as _ir
from tvm_tirx.ir import prim as _prim


from typing import Any
from tilelang.language.kernel import get_thread_bindings
from tilelang.language.common import alloc_shared, copy, macro, serial
from tilelang.language.utils import index_to_coordinates

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tilelang.cuda.debug import device_assert

__all__ = ["device_assert", "print"]


@macro
def print_var(var: _ir.Expr, msg: str = "") -> None:
    """
    Prints the value of a TIR primitive expression (PrimExpr) for debugging purposes.

    Parameters:
        var (tirx.PrimExpr): The variable or expression to be printed.

    Returns:
        tirx.PrimExpr: The TIR expression for the debug print operation.
    """
    _intrinsics.call_extern("handle", "debug_print_var", msg, var)


@macro
def print_var_with_condition(condition: _ir.Expr, var: _ir.Expr, msg: str = "") -> None:
    """
    Conditionally prints a TIR primitive expression (PrimExpr) if a given condition is True.

    Parameters:
        condition (tirx.PrimExpr): A TIR expression representing the condition to check.
        var (tirx.PrimExpr): The variable or expression to be printed.

    Returns:
        tirx.PrimExpr: The TIR expression for the debug print operation, if the condition is True.
    """
    if condition:
        _intrinsics.call_extern("handle", "debug_print_var", msg, var)


@macro
def print_global_buffer_with_condition(condition: _ir.Expr, buffer: _tensor_spec.TensorValue, elems: int, msg: str = "") -> _ir.Expr:
    """
    Conditionally prints the values of a flattened TIR buffer if the condition is True.
    """
    if condition:
        # Iterate through the buffer elements and print each one.
        for i in serial(elems):
            coords = index_to_coordinates(i, buffer.ty.shape)
            _intrinsics.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i, buffer[coords])


@macro
def print_shared_buffer_with_condition(condition: _ir.Expr, buffer: _tensor_spec.TensorValue, elems: int, msg: str = "") -> None:
    """
    Conditionally prints the values of a flattened TIR buffer if the condition is True.

    Parameters:
        condition (tirx.PrimExpr): A TIR expression representing the condition to check.
        buffer (tirx.Buffer): The buffer whose values need to be printed.
        elems (int): The number of elements in the buffer to print.

    Returns:
        tirx.PrimExpr: The TIR expression for the debug print operation.
    """
    if condition:
        # Iterate through the buffer elements and print each one.
        for i in serial(elems):
            coords = index_to_coordinates(i, buffer.ty.shape)
            _intrinsics.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i, buffer[coords])


@macro
def print_fragment_buffer_with_condition(condition: _ir.Expr, buffer: _tensor_spec.TensorValue, elems: int, msg: str = "") -> None:
    """
    Conditionally prints the values of a flattened TIR buffer if the condition is True.

    Parameters:
        condition (tirx.PrimExpr): A TIR expression representing the condition to check.
        buffer (tirx.Buffer): The buffer whose values need to be printed.
        elems (int): The number of elements in the buffer to print.

    Returns:
        tirx.PrimExpr: The TIR expression for the debug print operation.
    """
    smem = alloc_shared(buffer.ty.shape, _construction.dtype_of(buffer), "shared")
    copy(buffer, smem)
    if condition:
        # Iterate through the buffer elements and print each one.
        for i in serial(elems):
            coords = index_to_coordinates(i, buffer.ty.shape)
            _intrinsics.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i, smem[coords])


@macro
def print_msg(msg: str) -> None:
    """
    Prints a message string.
    """
    assert isinstance(msg, str), "msg must be a string"
    assert msg != "", "msg must not be empty"
    _intrinsics.call_extern("handle", "debug_print_msg", msg)


@macro
def print_local_buffer_with_condition(condition: _ir.Expr, buffer: _tensor_spec.TensorValue, elems: int, msg: str = "") -> None:
    """
    Conditionally prints the values of a flattened TIR buffer if the condition is True.

    Parameters:
        condition (tirx.PrimExpr): A TIR expression representing the condition to check.
        buffer (tirx.Buffer): The buffer whose values need to be printed.
        elems (int): The number of elements in the buffer to print.
    """
    if condition:
        # Iterate through the buffer elements and print each one.
        for i in serial(elems):
            coords = index_to_coordinates(i, buffer.ty.shape)
            _intrinsics.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i, buffer[coords])


# NOTE(chaofan): T.print is implemented as a macro, so no return
def print(obj: Any = None, msg: str = "", warp_group_id: int = 0, warp_id: int = 0) -> None:
    """
    A generic print function that handles both TIR buffers and primitive expressions.

    - If the input is a TIR buffer, it prints its values, but only on the first thread (tx=0, ty=0, tz=0).
    - If the input is a TIR primitive expression, it prints its value directly.

    Parameters:
        obj (Any): The object to print. It can be either a tirx.Buffer, tirx.PrimExpr, or None (for msg-only print).
        msg (str): An optional message to include in the print statement.
        warp_group_id (int): The warp group id to print.
        warp_id (int): The warp id to print.
        print thread will be warp_group_id * warp_group_size + warp_id

    Raises:
        ValueError: If the input object type is unsupported.
    """
    if _tensor_spec.is_buffer_like(obj):
        # Buffers must be printed in just one thread to avoid duplicate outputs.
        # Retrieve the thread bindings for thread x, y, and z.
        tx, ty, tz = get_thread_bindings()
        warp_group_size = 128
        warp_size = 32
        main_lane = warp_group_id * warp_group_size + warp_id * warp_size

        # Flatten the buffer for consistent printing. This assumes a 1D flattened buffer.
        buffer = obj
        if str(buffer.ty.scope) == "local":
            # Get the number of elements in the buffer.
            elems = 1
            for dim in buffer.ty.shape:
                elems *= dim
            condition = True
            if not msg:
                msg = f"buffer<{buffer.name}, {_construction.dtype_of(buffer)}>"
            print_local_buffer_with_condition(condition, buffer, elems, msg)
        elif str(buffer.ty.scope) == "local.fragment":
            # Get the number of elements in the buffer.
            elems = 1
            for dim in buffer.ty.shape:
                elems *= dim

            # Ensure only the first thread (tx=0, ty=0, tz=0) executes the print.
            condition = _prim.all(tx == main_lane, ty == 0, tz == 0)
            if not msg:
                msg = f"buffer<{buffer.name}, {_construction.dtype_of(buffer)}>"
            print_fragment_buffer_with_condition(condition, buffer, elems, msg)
        elif str(buffer.ty.scope) in {"shared", "shared.dyn"}:
            # Get the number of elements in the buffer.
            elems = 1
            for dim in buffer.ty.shape:
                elems *= dim

            # Ensure only the first thread (tx=0, ty=0, tz=0) executes the print.
            condition = _prim.all(tx == main_lane, ty == 0, tz == 0)
            if not msg:
                msg = f"buffer<{buffer.name}, {_construction.dtype_of(buffer)}>"
            print_shared_buffer_with_condition(condition, buffer, elems, msg)
        elif str(buffer.ty.scope) == "global":
            # Get the number of elements in the buffer.
            elems = 1
            for dim in buffer.ty.shape:
                elems *= dim
            condition = True
            print_global_buffer_with_condition(condition, buffer, elems, msg)
        else:
            raise ValueError(f"Unsupported buffer scope: {str(buffer.ty.scope)}")

    elif _construction.is_scalar_expr(obj):
        if not msg:
            msg = f"expr<{obj}>"
        # Directly print primitive expressions.
        print_var(obj, msg)

    elif obj is None:
        print_msg(msg)

    else:
        # Unsupported object type.
        raise ValueError(f"Unexpected type: {type(obj)}. Supported types are tirx.Buffer, tirx.PrimExpr, and None.")


def __getattr__(name):
    if name != "device_assert":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from tilelang.cuda.debug import device_assert

    return device_assert
