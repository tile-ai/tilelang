from tilelang import tvm as tvm
from tvm import tir
from tilelang.utils.target import (
    target_is_cuda,)
from tvm.target import Target
from tvm.ir.base import Node
from tvm.runtime import Scriptable
import tvm.ffi
from tilelang.ir import GemmWarpPolicy
from .gemm_mma import GemmMMA


@tvm.ffi.register_func("tl.gemm_py.infer_layout")
def gemm_py_infer_layout(gemm_py, target, thread_bounds):
    thread_nums = thread_bounds.extent
    return gemm_py.infer_layout(target, thread_nums)


@tvm.ffi.register_func("tl.gemm_py.lower")
def gemm_py_lower(gemm_py, target, thread_bounds, thread_var):
    thread_nums = thread_bounds.extent
    stmt = gemm_py.lower(target, thread_nums, thread_var)
    return stmt


@tvm.ffi.register_object("tl.GemmPy")
class GemmPy(Node, Scriptable):
    A: tir.Buffer
    B: tir.Buffer
    C: tir.Buffer

    APtr: tir.PrimExpr
    BPtr: tir.PrimExpr
    CPtr: tir.PrimExpr

    M: int
    N: int
    K: int

    trans_A: bool
    trans_B: bool

    stride_A: int
    stride_B: int
    offset_A: int
    offset_B: int
    clear_accum: bool
    k_pack: int
    wg_wait: int
    policy: GemmWarpPolicy

    def infer_layout(self, target: Target, thread_nums: int):
        if target_is_cuda(target):
            # TODO(lei): Support more cuda architectures, now mma only
            return GemmMMA(self).infer_layout(target, thread_nums)
        else:
            raise ValueError(f"Unsupported target: {target}")

    def lower(self, target: Target, thread_nums: int, thread_var: tir.Var):
        if target_is_cuda(target):
            # TODO(lei): Support more cuda architectures, now mma only
            # Now only implement ssr layout
            return GemmMMA(self).lower(target, thread_nums, thread_var)
        else:
            raise ValueError(f"Unsupported target: {target}")
