"""Ascend dialect T.print.

Same debug externs as the common implementation (`debug_print_var`,
`debug_print_buffer_value`, emitted by the Ascend codegen via
src/tl_templates/ascend/debug.h), but with the NPU's execution model: a
kernel body runs once per AI core and UB/L1 buffers are core-local, so
buffer prints need no CUDA-style single-thread gating. Keeping the policy
here (rather than probing hardware inside the CUDA dialect module) keeps
each dialect self-contained.
"""

from __future__ import annotations

from tilelang.language import _construction
from tilelang.language import _tensor_spec


from typing import Any


# The @macro helpers are target-neutral mechanism: they only emit the
# debug_print_* externs that every backend's codegen implements.
from tilelang.cuda.language.print import (
    print_global_buffer_with_condition,
    print_local_buffer_with_condition,
    print_msg,
    print_shared_buffer_with_condition,
    print_var,
)

__all__ = ["print"]


def print(obj: Any = None, msg: str = "") -> None:  # noqa: A001
    """Print a buffer, a primitive expression, or a plain message.

    Args:
        obj: A ``tirx.Buffer`` (``local``, ``shared``, ``shared.dyn`` or
            ``global`` scope), a ``tirx.PrimExpr``, or ``None`` for a
            message-only print.
        msg: Optional message prefix.
    """
    if _tensor_spec.is_buffer_like(obj):
        buffer = obj

        elems = 1
        for dim in buffer.ty.shape:
            elems *= dim

        if not msg:
            msg = f"buffer<{buffer.name}, {_construction.dtype_of(buffer)}>"

        if str(buffer.ty.scope) == "local":
            print_local_buffer_with_condition(True, buffer, elems, msg)
        elif str(buffer.ty.scope) in {"shared", "shared.dyn"}:
            print_shared_buffer_with_condition(True, buffer, elems, msg)
        elif str(buffer.ty.scope) == "global":
            print_global_buffer_with_condition(True, buffer, elems, msg)
        else:
            raise ValueError(
                f"T.print: unsupported buffer scope on Ascend: {str(buffer.ty.scope)}. Copy the buffer to UB (shared) or GM (global) first."
            )

    elif _construction.is_scalar_expr(obj):
        if not msg:
            msg = f"expr<{obj}>"
        print_var(obj, msg)

    elif obj is None:
        print_msg(msg)

    else:
        raise ValueError(f"Unexpected type: {type(obj)}. Supported types are tirx.Buffer, tirx.PrimExpr, and None.")
