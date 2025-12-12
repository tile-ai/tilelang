"""
This module provides macros and utilities for debugging TileLang (tl) programs.
It includes functionality to print variables, print values in buffers, conditionally execute debug prints and assert.
"""

from tvm import tir
from typing import Any
import tilelang.language as T
from tilelang.language.kernel import get_thread_bindings
from tilelang.language import copy, macro, serial, alloc_shared
from tilelang.language.utils import index_to_coordinates


@macro
def print_var(var: tir.PrimExpr, msg: str = "") -> tir.PrimExpr:
    """
    Prints the value of a TIR primitive expression (PrimExpr) for debugging purposes.

    Parameters:
        var (tir.PrimExpr): The variable or expression to be printed.

    Returns:
        tir.PrimExpr: The TIR expression for the debug print operation.
    """
    tir.call_extern("handle", "debug_print_var", msg, var)


@macro
def print_var_with_condition(condition: tir.PrimExpr, var: tir.PrimExpr, msg: str = "") -> tir.PrimExpr:
    """
    Conditionally prints a TIR primitive expression (PrimExpr) if a given condition is True.

    Parameters:
        condition (tir.PrimExpr): A TIR expression representing the condition to check.
        var (tir.PrimExpr): The variable or expression to be printed.

    Returns:
        tir.PrimExpr: The TIR expression for the debug print operation, if the condition is True.
    """
    if condition:
        tir.call_extern("handle", "debug_print_var", msg, var)


@macro
def print_global_buffer_with_condition(condition: tir.PrimExpr, buffer: tir.Buffer, elems: int, msg: str = "") -> tir.PrimExpr:
    """
    Conditionally prints the values of a flattened TIR buffer if the condition is True.
    """
    if condition:
        # Iterate through the buffer elements and print each one.
        for i in serial(elems):
            coords = index_to_coordinates(i, buffer.shape)
            tir.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i, buffer[coords])
    else:
        tir.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i, buffer[coords])


@macro
def print_shared_buffer_with_condition(condition: tir.PrimExpr, buffer: tir.Buffer, elems: int, msg: str = "") -> tir.PrimExpr:
    """
    Conditionally prints the values of a flattened TIR buffer if the condition is True.

    Parameters:
        condition (tir.PrimExpr): A TIR expression representing the condition to check.
        buffer (tir.Buffer): The buffer whose values need to be printed.
        elems (int): The number of elements in the buffer to print.

    Returns:
        tir.PrimExpr: The TIR expression for the debug print operation.
    """
    if condition:
        # Iterate through the buffer elements and print each one.
        for i in serial(elems):
            coords = index_to_coordinates(i, buffer.shape)
            tir.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i, buffer[coords])


@macro
def print_fragment_buffer_with_condition(condition: tir.PrimExpr, buffer: tir.Buffer, elems: int, msg: str = "") -> tir.PrimExpr:
    """
    Conditionally prints the values of a flattened TIR buffer if the condition is True.

    Parameters:
        condition (tir.PrimExpr): A TIR expression representing the condition to check.
        buffer (tir.Buffer): The buffer whose values need to be printed.
        elems (int): The number of elements in the buffer to print.

    Returns:
        tir.PrimExpr: The TIR expression for the debug print operation.
    """
    smem = alloc_shared(buffer.shape, buffer.dtype, "shared")
    copy(buffer, smem)
    if condition:
        # Iterate through the buffer elements and print each one.
        for i in serial(elems):
            coords = index_to_coordinates(i, buffer.shape)
            tir.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i, smem[coords])


@macro
def print_local_buffer_with_condition(condition: tir.PrimExpr, buffer: tir.Buffer, elems: int, msg: str = "") -> tir.PrimExpr:
    """
    Conditionally prints the values of a flattened TIR buffer if the condition is True.

    Parameters:
        condition (tir.PrimExpr): A TIR expression representing the condition to check.
        buffer (tir.Buffer): The buffer whose values need to be printed.
        elems (int): The number of elements in the buffer to print.

    Returns:
        tir.PrimExpr: The TIR expression for the debug print operation.
    """
    if condition:
        # Iterate through the buffer elements and print each one.
        for i in serial(elems):
            coords = index_to_coordinates(i, buffer.shape)
            tir.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i, buffer[coords])


from tilelang.utils.target import check_cuda_availability
import warnings

_IS_CUDA_AVAILABLE = check_cuda_availability()


@macro
def device_assert(condition: tir.PrimExpr, msg: str = ""):
    """
    Device-side assert emulation.
    Emits a device-side assert call on CUDA targets when CUDA is available.
    The assert is always enabled and cannot be disabled at runtime.
    """
    if _IS_CUDA_AVAILABLE:
        if msg == "":
            T.call_intrin("void", tir.op.Op.get("tl.device_assert"), condition)
        else:
            warnings.warn("Non-empty msg may slightly slow down the kernel", stacklevel=2)
            T.call_intrin("void", tir.op.Op.get("tl.device_assert_with_msg"), condition, msg)


def print(obj: Any, msg: str = "", warp_group_id: int = 0, warp_id: int = 0) -> tir.PrimExpr:
    """
    A generic print function that handles both TIR buffers and primitive expressions.

    - If the input is a TIR buffer, it prints its values, but only on the first thread (tx=0, ty=0, tz=0).
    - If the input is a TIR primitive expression, it prints its value directly.

    Parameters:
        obj (Any): The object to print. It can be either a tir.Buffer or tir.PrimExpr.
        msg (str): An optional message to include in the print statement.
        warp_group_id (int): The warp group id to print.
        warp_id (int): The warp id to print.
        print thread will be warp_group_id * warp_group_size + warp_id.

    Returns:
        tir.PrimExpr: The TIR expression for the debug print operation.

    Raises:
        ValueError: If the input object type is unsupported.
    """
    if isinstance(obj, tir.Buffer):
        # Buffers must be printed in just one thread to avoid duplicate outputs.
        # Retrieve the thread bindings for thread x, y, and z.
        tx, ty, tz = get_thread_bindings()
        warp_group_size = 128
        warp_size = 32
        main_lane = warp_group_id * warp_group_size + warp_id * warp_size

        # Flatten the buffer for consistent printing. This assumes a 1D flattened buffer.
        buffer = obj
        if buffer.scope() == "local":
            # Get the number of elements in the buffer.
            elems = 1
            for dim in buffer.shape:
                elems *= dim
            condition = True
            if not msg:
                msg = f"buffer<{buffer.name}, {buffer.dtype}>"
            return print_local_buffer_with_condition(condition, buffer, elems, msg)
        elif buffer.scope() == "local.fragment":
            # Get the number of elements in the buffer.
            elems = 1
            for dim in buffer.shape:
                elems *= dim

            # Ensure only the first thread (tx=0, ty=0, tz=0) executes the print.
            condition = tx == main_lane and ty == 0 and tz == 0
            if not msg:
                msg = f"buffer<{buffer.name}, {buffer.dtype}>"
            return print_fragment_buffer_with_condition(condition, buffer, elems, msg)
        elif buffer.scope() in {"shared", "shared.dyn"}:
            # Get the number of elements in the buffer.
            elems = 1
            for dim in buffer.shape:
                elems *= dim

            # Ensure only the first thread (tx=0, ty=0, tz=0) executes the print.
            condition = tx == main_lane and ty == 0 and tz == 0
            if not msg:
                msg = f"buffer<{buffer.name}, {buffer.dtype}>"
            return print_shared_buffer_with_condition(condition, buffer, elems, msg)
        elif buffer.scope() == "global":
            # Get the number of elements in the buffer.
            elems = 1
            for dim in buffer.shape:
                elems *= dim
            condition = True
            return print_global_buffer_with_condition(condition, buffer, elems, msg)
        else:
            raise ValueError(f"Unsupported buffer scope: {buffer.scope()}")

    elif isinstance(obj, tir.PrimExpr):
        if not msg:
            msg = f"expr<{obj}>"
        # Directly print primitive expressions.
        return print_var(obj, msg)

    else:
        # Unsupported object type.
        raise ValueError(f"Unexpected type: {type(obj)}. Supported types are tir.Buffer and tir.PrimExpr.")
