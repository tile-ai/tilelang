from __future__ import annotations
from tilelang import tvm as tvm
from tvm import IRModule
from tvm.tir import PrimFunc
from typing import Callable
from . import _ffi_api


def LetInline():
    """LetInline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LetInline()  # type: ignore


def Simplify(simplify_arguments: bool = False):
    """Simplify

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.Simplify(simplify_arguments)  # type: ignore


def _Simplify(stmt: PrimFunc | IRModule, inline_let: bool = False) -> PrimFunc | IRModule:
    if isinstance(stmt, PrimFunc):
        if inline_let:
            mod = LetInline()(IRModule.from_expr(stmt))
            mod = Simplify(simplify_arguments=True)(mod)
        else:
            mod = Simplify(simplify_arguments=True)(IRModule.from_expr(stmt))
        assert len(mod.functions) == 1, "Simplify should return a single function"
        return list(mod.functions.values()).pop()
    elif isinstance(stmt, IRModule):
        if inline_let:
            mod = LetInline()(stmt)
            mod = Simplify(simplify_arguments=True)(mod)
        else:
            mod = Simplify(simplify_arguments=True)(stmt)
        assert len(mod.functions) == 1, "Simplify should return a single function"
        return list(mod.functions.values()).pop()
    else:
        raise ValueError(f"Unsupported type: {type(stmt)}")


# Decorator to simplify the output of a function
def simplify_prim_func(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        stmt: PrimFunc | IRModule = (func)(*args, **kwargs)
        return _Simplify(stmt)

    return wrapper


def apply_simplify(stmt: PrimFunc | IRModule, inline_let: bool = False) -> PrimFunc | IRModule:
    """Apply Simplify pass to a PrimFunc or IRModule."""
    return _Simplify(stmt, inline_let)
