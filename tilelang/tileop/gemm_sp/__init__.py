from tilelang import tvm as tvm
from tvm import tir
from tilelang.utils.target import (
    target_is_cuda,
)
from tvm.target import Target
from tvm.ir.base import Node
from tvm.ir import Range
from tvm.runtime import Scriptable
import tvm_ffi
from tilelang.tileop.base import GemmWarpPolicy
from .gemm_sp_mma import GemmSPMMA


@tvm_ffi.register_global_func("tl.gemm_sp_py.infer_layout")
def gemm_sp_py_infer_layout(gemm_sp_py: GemmSPMMA, target: Target, thread_bounds: Range):
    thread_nums = thread_bounds.extent
    return gemm_sp_py.infer_layout(target, thread_nums)


@tvm_ffi.register_global_func("tl.gemm_sp_py.lower")
def gemm_sp_py_lower(gemm_sp_py: GemmSPMMA, target: Target, thread_bounds: Range, thread_var: tir.Var):
    thread_nums = thread_bounds.extent
    stmt = gemm_sp_py.lower(target, thread_nums, thread_var)
    return stmt


@tvm_ffi.register_object("tl.GemmSPPy")
class GemmSPPy(Node, Scriptable):
    A: tir.Buffer
    E: tir.Buffer
    B: tir.Buffer
    C: tir.Buffer

    APtr: tir.PrimExpr
    EPtr: tir.PrimExpr
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
            return GemmSPMMA(self).infer_layout(target, thread_nums)
        else:
            raise ValueError(f"Unsupported target: {target}")

    def lower(self, target: Target, thread_nums: int, thread_var: tir.Var):
        if target_is_cuda(target):
            # TODO(lei): Support more cuda architectures, now mma only
            # Now only implement ssr layout
            return GemmSPMMA(self).lower(target, thread_nums, thread_var)
        else:
            raise ValueError(f"Unsupported target: {target}")
