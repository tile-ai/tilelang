# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tilelang import tvm as tvm
from tvm import IRModule
from tvm.tir import PrimFunc
from typing import Union, Callable
from . import _ffi_api


def Simplify():
    """Simplify

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.Simplify()  # type: ignore


def _Simplify(stmt: Union[PrimFunc, IRModule]) -> Union[PrimFunc, IRModule]:
    if isinstance(stmt, PrimFunc):
        mod = Simplify()(IRModule.from_expr(stmt))
        assert len(mod.functions) == 1, "Simplify should return a single function"
        return list(mod.functions.values()).pop()
    elif isinstance(stmt, IRModule):
        return Simplify()(stmt)
    else:
        raise ValueError(f"Unsupported type: {type(stmt)}")


# Decorator to simplify the output of a function
def simplify_prim_func(func: Callable) -> Callable:

    def wrapper(*args, **kwargs):
        stmt: Union[PrimFunc, IRModule] = (func)(*args, **kwargs)
        return _Simplify(stmt)

    return wrapper


def apply_simplify(stmt: Union[PrimFunc, IRModule]) -> Union[PrimFunc, IRModule]:
    """Apply Simplify pass to a PrimFunc or IRModule."""
    return _Simplify(stmt)
