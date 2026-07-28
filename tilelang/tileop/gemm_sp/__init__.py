from tilelang import tvm as tvm
from tilelang.tileop.gemm_sp.registry import resolve_gemm_sp_impl
from tvm import tirx
from tvm.target import Target
from tvm.ir.base import Node
from tvm.ir import Range
from tvm.runtime import Scriptable
import tvm_ffi
from tilelang import _ffi_api
from tilelang.ir import GemmSPWarpPolicy


@tvm_ffi.register_object("tl.GemmSP")
class GemmSP(Node, Scriptable):
    A: tirx.Buffer
    E: tirx.Buffer
    B: tirx.Buffer
    C: tirx.Buffer

    aRegion: tirx.BufferRegion
    eRegion: tirx.BufferRegion
    bRegion: tirx.BufferRegion
    cRegion: tirx.BufferRegion

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
    policy: GemmSPWarpPolicy

    @property
    def k_pack(self):
        return self.kPack

    @tvm_ffi.register_global_func("tl.gemm_sp.infer_layout")
    def gemm_sp_infer_layout(self, target: Target, thread_bounds: Range):
        thread_nums = thread_bounds.extent
        return self.infer_layout(target, thread_nums)

    @tvm_ffi.register_global_func("tl.gemm_sp.lower")
    def gemm_sp_lower(self, target: Target, layout_map: dict, thread_bounds: Range, thread_index: tirx.PrimExpr):
        stmt = self.lower(target, layout_map, thread_bounds, thread_index)
        return stmt

    def infer_layout(self, target: Target, thread_nums: int):
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).infer_layout(target, thread_nums)

    def lower(self, target: Target, layout_map: dict, thread_bounds: Range, thread_index: tirx.PrimExpr):
        thread_nums = thread_bounds.extent
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).lower(layout_map, target, thread_bounds, thread_index)

    def _select_gemm_instruction(self, thread_nums: int, target: Target) -> str:
        return str(_ffi_api.GemmSPGetGemmInstructionKey(self, int(thread_nums), target))

    def _get_implementation_class(self, gemm_inst: str, target: Target):
        return resolve_gemm_sp_impl(gemm_inst, target)
