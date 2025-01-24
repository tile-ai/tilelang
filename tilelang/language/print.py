# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm import tir
import builtins
from typing import Any
from tilelang.language.kernel import get_thread_bindings
from tilelang.language import macro, serial


@macro
def print_var(var: tir.PrimExpr) -> tir.PrimExpr:
    tir.call_extern("handle", "debug_print_var", var)


@macro
def print_var_with_condition(condition: tir.PrimExpr, var: tir.PrimExpr) -> tir.PrimExpr:
    if condition:
        tir.call_extern("handle", "debug_print_var", var)


@macro
def print_flat_buffer_with_condition(condition: tir.PrimExpr, buffer: tir.Buffer,
                                     elems: int) -> tir.PrimExpr:
    if condition:
        for i in serial(elems):
            tir.call_extern("handle", "debug_print_buffer_value", buffer.name, i, buffer[i])


def print(obj: Any) -> tir.PrimExpr:
    builtins.print(obj)
    builtins.print(type(obj))
    builtins.print("isinstance(expr, tir.PrimExpr) ", isinstance(obj, tir.PrimExpr))
    builtins.print("isinstance(expr, tir.Buffer) ", isinstance(obj, tir.Buffer))
    builtins.print("isinstance(expr, tir.Stmt) ", isinstance(obj, tir.Stmt))
    if isinstance(obj, tir.Buffer):
        # print a buffer must be in just one thread
        tx, ty, tz = get_thread_bindings()
        buffer = obj.get_flattened_buffer()
        assert len(buffer.shape) == 1, "buffer must be flattened"
        elems = buffer.shape[-1]
        # only allow print in the first thread
        condition = (tx == 0 and ty == 0 and tz == 0)
        return print_flat_buffer_with_condition(condition, buffer, elems)
        return tir.call_extern("handle", "debug_print_var", buffer.data, elems)
    elif isinstance(obj, tir.PrimExpr):
        return print_var(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
