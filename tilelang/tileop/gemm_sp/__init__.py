from tilelang import tvm as tvm
from tvm import tir
from tvm.target import Target
from tvm.ir.base import Node
from tvm.ir import Range
from tvm.runtime import Scriptable
import tvm_ffi
from tilelang import _ffi_api
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.tileop.gemm_sp.gemm_sp_mma import GemmSPMMA
from tilelang.tileop.gemm_sp.gemm_sp_wgmma import GemmSPWGMMA
from tilelang.tileop.gemm.inst import GemmInst


@tvm_ffi.register_object("tl.GemmSP")
class GemmSP(Node, Scriptable):
    A: tir.Buffer
    E: tir.Buffer
    B: tir.Buffer
    C: tir.Buffer

    aRegion: tir.BufferRegion
    eRegion: tir.BufferRegion
    bRegion: tir.BufferRegion
    cRegion: tir.BufferRegion

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

    @tvm_ffi.register_global_func("tl.gemm_sp.infer_layout")
    def gemm_sp_infer_layout(self, target: Target, thread_bounds: Range):
        print(f"{type(self.A)=}")
        thread_nums = thread_bounds.extent
        return self.infer_layout(target, thread_nums)

    @tvm_ffi.register_global_func("tl.gemm_sp.lower")
    def gemm_sp_lower(self, target: Target, layout_map: dict, thread_bounds: Range, thread_var: tir.Var):
        thread_nums = thread_bounds.extent
        stmt = self.lower(target, layout_map, thread_nums, thread_var)
        return stmt

    def infer_layout(self, target: Target, thread_nums: int):
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).infer_layout(target, thread_nums)

    def lower(self, target: Target, layout_map: dict, thread_nums: int, thread_var: tir.Var):
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        print(f"{gemm_inst=}, {impl_class=}, {target=}, {layout_map=}, {thread_nums=}, {thread_var=}")
        return impl_class(self).lower(layout_map, target, thread_nums, thread_var)

    def _select_gemm_instruction(self, thread_nums: int, target: Target) -> GemmInst:
        # NOTE: use dense counterpart to select instruction
        return GemmInst(_ffi_api.GemmSPGetGemmSPInst(self, int(thread_nums), target))

    def _get_implementation_class(self, gemm_inst: GemmInst, target: Target):
        if gemm_inst == GemmInst.MMA:
            return GemmSPMMA
        elif gemm_inst == GemmInst.WGMMA:
            return GemmSPWGMMA
        else:
            raise ValueError(f"Unsupported gemm instruction: {gemm_inst} for target: {target}")
