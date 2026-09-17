"""Common block-scaled GEMM tile op."""

import tvm_ffi

from ..gemm import Gemm


@tvm_ffi.register_object("tl.GemmBlockScaled")
class GemmBlockScaled(Gemm):
    """Block-scaled GEMM tile op: ``C (+)= (A * SFA) @ (B * SFB)``.

    A ``GemmNode`` subclass on the C++ side, so it shares the dense GEMM's
    operand layouts, warp partition and scheduling, and is lowered through the
    same ``tl.gemm.infer_layout`` / ``tl.gemm.lower`` entry points, which
    dispatch on this Python class. The extra FFI fields are ``sfaRegion``,
    ``sfbRegion`` and ``sfKStart``. Instruction selection goes through the
    same backend registry as ``Gemm``; the backend implementations branch on
    ``is_blockscaled``.
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
