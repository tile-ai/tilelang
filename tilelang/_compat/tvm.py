"""TileLang-owned namespace adapter; this never installs a global ``tvm`` package."""

import tvm_ffi as ffi
import tvm_tirx as _native
from tvm_ffi import dtype as DataType
from tvm_ffi import get_global_func, register_global_func
from tvm_tirx import IRModule, ir, runtime, target
from tvm_tirx import sym as arith
from tvm_tirx.ir import transform

from . import tirx

register_func = register_global_func

__all__ = [
    "DataType",
    "IRModule",
    "arith",
    "ffi",
    "get_global_func",
    "ir",
    "register_func",
    "register_global_func",
    "runtime",
    "target",
    "tirx",
    "transform",
]


def __getattr__(name):
    return getattr(_native, name)
