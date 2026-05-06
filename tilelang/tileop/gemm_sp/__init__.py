from tilelang import tvm as tvm
from tvm import tir
from tvm.target import Target
from tvm.ir.base import Node
from tvm.ir import Range
from tvm.runtime import Scriptable
import tvm_ffi
from tilelang.backend.gemm_sp import resolve_gemm_sp_impl
from tilelang.tileop.base import GemmWarpPolicy


@tvm_ffi.register_global_func("tl.gemm_sp_py.infer_layout")
def gemm_sp_py_infer_layout(gemm_sp_py, target: Target, thread_bounds: Range):
    thread_nums = thread_bounds.extent
    return gemm_sp_py.infer_layout(target, thread_nums)


@tvm_ffi.register_global_func("tl.gemm_sp_py.lower")
def gemm_sp_py_lower(gemm_sp_py, target: Target, thread_bounds: Range, thread_var: tir.Var):
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
        impl_class = resolve_gemm_sp_impl(target)
        return impl_class(self).infer_layout(target, thread_nums)

    def lower(self, target: Target, thread_nums: int, thread_var: tir.Var):
        impl_class = resolve_gemm_sp_impl(target)
        return impl_class(self).lower(target, thread_nums, thread_var)
