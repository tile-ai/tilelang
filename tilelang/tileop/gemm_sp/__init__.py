from tilelang.language import _tensor_spec
from tvm_tirx import ir as _ir
from tilelang.tileop.gemm_sp.registry import resolve_gemm_sp_impl

from tvm_tirx.target import Target
from tvm_tirx.ir import Node
from tvm_tirx.ir import Range
from tvm_tirx.ir import Scriptable
import tvm_ffi
from tilelang import _ffi_api
from tilelang.ir import GemmSPWarpPolicy
from tilelang.tileop._lowering import close_lowered_function
from tilelang.tileop import _register_default_callback


@tvm_ffi.register_object("tl.GemmSP")
class GemmSP(Node, Scriptable):
    A: _tensor_spec.TensorValue
    E: _tensor_spec.TensorValue
    B: _tensor_spec.TensorValue
    C: _tensor_spec.TensorValue

    aRegion: _ir.TensorRegion
    eRegion: _ir.TensorRegion
    bRegion: _ir.TensorRegion
    cRegion: _ir.TensorRegion

    M: int
    N: int
    K: int

    trans_A: bool
    trans_B: bool
    trans_E: bool

    clear_accum: bool
    wg_wait: int
    policy: GemmSPWarpPolicy

    @_register_default_callback("tl.gemm_sp.infer_layout")
    def gemm_sp_infer_layout(self, target: Target, thread_bounds: Range):
        thread_nums = thread_bounds.extent
        return self.infer_layout(target, thread_nums)

    @_register_default_callback("tl.gemm_sp.lower")
    def gemm_sp_lower(self, target: Target, layout_map: dict, thread_bounds: Range, thread_index: _ir.Expr):
        function = self.lower(target, layout_map, thread_bounds, thread_index)
        return close_lowered_function(function)

    def infer_layout(self, target: Target, thread_nums: int):
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).infer_layout(target, thread_nums)

    def lower(self, target: Target, layout_map: dict, thread_bounds: Range, thread_index: _ir.Expr):
        thread_nums = thread_bounds.extent
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).lower(layout_map, target, thread_bounds, thread_index)

    def _select_gemm_instruction(self, thread_nums: int, target: Target) -> str:
        return str(_ffi_api.GemmSPGetGemmInstructionKey(self, int(thread_nums), target))

    def _get_implementation_class(self, gemm_inst: str, target: Target):
        return resolve_gemm_sp_impl(gemm_inst, target)
