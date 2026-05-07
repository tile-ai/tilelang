from tilelang import tvm as tvm
from tvm import tir
from tvm.ir import Range
from tvm.ir.base import Node
from tvm.runtime import Scriptable
from tvm.target import Target
import tvm_ffi

from tilelang.backend.gemm_sp import resolve_gemm_sp_impl
from tilelang.tileop.base import GemmWarpPolicy


@tvm_ffi.register_global_func("tl.gemm_sp.infer_layout")
def gemm_sp_infer_layout(gemm_sp, target: Target, thread_bounds: Range):
    thread_nums = thread_bounds.extent
    return gemm_sp.infer_layout(target, thread_nums)


@tvm_ffi.register_global_func("tl.gemm_sp.lower")
def gemm_sp_lower(gemm_sp, target: Target, thread_bounds: Range, thread_var: tir.Var):
    thread_nums = thread_bounds.extent
    stmt = gemm_sp.lower(target, thread_nums, thread_var)
    return stmt


@tvm_ffi.register_object("tl.GemmSP")
class GemmSP(Node, Scriptable):
    A: tir.Buffer
    E: tir.Buffer
    B: tir.Buffer
    C: tir.Buffer

    M: int
    N: int
    K: int

    trans_A: bool
    trans_B: bool
    trans_E: bool

    stride_A: int
    stride_B: int
    offset_A: int
    offset_B: int
    clear_accum: bool
    kPack: int
    wg_wait: int
    policy: GemmWarpPolicy

    @property
    def k_pack(self):
        return self.kPack

    def infer_layout(self, target: Target, thread_nums: int):
        impl_class = resolve_gemm_sp_impl(target)
        return impl_class(self).infer_layout(target, thread_nums)

    def lower(self, target: Target, thread_nums: int, thread_var: tir.Var):
        impl_class = resolve_gemm_sp_impl(target)
        return impl_class(self).lower(target, thread_nums, thread_var)


# Compatibility for imports that referenced the transitional Python-backed name.
GemmSPPy = GemmSP
