"""Common block-scaled GEMM tile op and FFI entry points."""

import tvm_ffi
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target

from ..gemm import Gemm


@tvm_ffi.register_global_func("tl.gemm_blockscaled.infer_layout")
def gemm_blockscaled_infer_layout(gemm, target: Target, thread_bounds: Range):
    thread_nums = thread_bounds.extent
    return gemm.infer_layout(target, thread_nums)


@tvm_ffi.register_global_func("tl.gemm_blockscaled.lower")
def gemm_blockscaled_lower(
    gemm,
    layout_map,
    target: Target,
    thread_bounds: Range,
    thread_index: tirx.PrimExpr,
    mbar_phase_expr: tirx.PrimExpr,
):
    return gemm.lower(layout_map, target, thread_bounds, thread_index, mbar_phase_expr)


@tvm_ffi.register_object("tl.GemmBlockScaled")
class GemmBlockScaled(Gemm):
    """Block-scaled GEMM tile op: ``C (+)= (A * SFA) @ (B * SFB)``.

    A ``GemmNode`` subclass on the C++ side, so it shares the dense GEMM's
    operand layouts, warp partition and scheduling; the extra FFI fields are
    ``sfaRegion``, ``sfbRegion`` and ``sfKStart``. Instruction selection and
    lowering go through the same backend registry as ``Gemm``; the backend
    implementations branch on ``is_blockscaled``.
    """

    # FFI fields added on top of Gemm: sfaRegion, sfbRegion, sfKStart

    @property
    def SFARegion(self):
        return self.sfaRegion

    @property
    def SFBRegion(self):
        return self.sfbRegion

    @property
    def sf_k_start(self):
        return self.sfKStart

    @property
    def is_blockscaled(self) -> bool:
        return True
