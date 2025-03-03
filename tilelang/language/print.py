# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This module provides macros and utilities for debugging TileLang (tl) programs.
It includes functionality to print variables, print values in buffers, and conditionally execute debug prints.
"""

from tvm import tir
from typing import Any
from tilelang.language.kernel import get_thread_bindings
from tilelang.language import copy, macro, serial, alloc_shared
from tilelang.intrinsics.utils import index_to_coordinates


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
def print_var_with_condition(condition: tir.PrimExpr,
                             var: tir.PrimExpr,
                             msg: str = "") -> tir.PrimExpr:
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
def print_shared_buffer_with_condition(condition: tir.PrimExpr,
                                       buffer: tir.Buffer,
                                       elems: int,
                                       msg: str = "") -> tir.PrimExpr:
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
            tir.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i,
                            buffer[coords])


@macro
def print_fragment_buffer_with_condition(condition: tir.PrimExpr,
                                         buffer: tir.Buffer,
                                         elems: int,
                                         msg: str = "") -> tir.PrimExpr:
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
def print_local_buffer_with_condition(condition: tir.PrimExpr,
                                      buffer: tir.Buffer,
                                      elems: int,
                                      msg: str = "") -> tir.PrimExpr:
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
            tir.call_extern("handle", "debug_print_buffer_value", msg, buffer.name, i,
                            buffer[coords])


def print(obj: Any, msg: str = "") -> tir.PrimExpr:
    """
    A generic print function that handles both TIR buffers and primitive expressions.
    
    - If the input is a TIR buffer, it prints its values, but only on the first thread (tx=0, ty=0, tz=0).
    - If the input is a TIR primitive expression, it prints its value directly.
    
    Parameters:
        obj (Any): The object to print. It can be either a tir.Buffer or tir.PrimExpr.
        msg (str): An optional message to include in the print statement.
        
    Returns:
        tir.PrimExpr: The TIR expression for the debug print operation.
        
    Raises:
        ValueError: If the input object type is unsupported.
    """
    if isinstance(obj, tir.Buffer):
        # Buffers must be printed in just one thread to avoid duplicate outputs.
        # Retrieve the thread bindings for thread x, y, and z.
        tx, ty, tz = get_thread_bindings()

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
            condition = (tx == 0 and ty == 0 and tz == 0)
            if not msg:
                msg = f"buffer<{buffer.name}, {buffer.dtype}>"
            return print_fragment_buffer_with_condition(condition, buffer, elems, msg)
        elif buffer.scope() in {"shared", "shared.dyn"}:
            # Get the number of elements in the buffer.
            elems = 1
            for dim in buffer.shape:
                elems *= dim

            # Ensure only the first thread (tx=0, ty=0, tz=0) executes the print.
            condition = (tx == 0 and ty == 0 and tz == 0)
            if not msg:
                msg = f"buffer<{buffer.name}, {buffer.dtype}>"
            return print_shared_buffer_with_condition(condition, buffer, elems, msg)

    elif isinstance(obj, tir.PrimExpr):
        if not msg:
            msg = f"expr<{obj}>"
        # Directly print primitive expressions.
        return print_var(obj, msg)

    else:
        # Unsupported object type.
        raise ValueError(
            f"Unexpected type: {type(obj)}. Supported types are tir.Buffer and tir.PrimExpr.")
