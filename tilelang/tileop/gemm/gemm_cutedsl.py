"""GEMM implementation for CuTeDSL backend - directly calls tl::gemm intrinsic."""

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang import language as T
from tvm import tir
from tvm.target import Target


class GemmCuTeDSL(GemmBase):
    """GEMM implementation for CuTeDSL that directly calls tl::gemm intrinsic.

    This implementation bypasses the complex lowering logic of MMA/WGMMA
    and directly emits a call to tl::gemm, similar to gemm_v1 behavior.
    This is necessary for CuTeDSL backend which requires simpler IR.
    """

    def infer_layout(self, target: Target, thread_nums: int):
        """For CuTeDSL, we still need proper layout inference for A, B, C buffers.

        CuTeDSL uses the same underlying hardware instructions (WGMMA/MMA),
        so it needs the same layout information. We delegate to the appropriate
        implementation based on the instruction type.
        """
        from tilelang.tileop.gemm import GemmInst
        from tilelang.tileop.gemm.gemm_wgmma import GemmWGMMA
        from tilelang.tileop.gemm.gemm_mma import GemmMMA
        from tilelang import _ffi_api

        # Determine which GEMM instruction will be used
        gemm_inst = GemmInst(_ffi_api.GemmPyGemmInst(self.gemm_node, int(thread_nums), target))

        # Use WGMMA or MMA layout inference based on instruction type
        if gemm_inst.is_wgmma():
            return GemmWGMMA(self.gemm_node).infer_layout(target, thread_nums)
        else:
            return GemmMMA(self.gemm_node).infer_layout(target, thread_nums)

    def lower(self, layout_map: dict, target: Target, thread_nums: int, thread_var: tir.Var):
        """Lower to a direct gemm_v1 call without complex MMA/WGMMA lowering."""
        from tilelang.language.gemm_op import gemm_v1
        from tilelang.transform.simplify import _Simplify
        from tilelang.tileop.base import GemmWarpPolicy as PyGemmWarpPolicy

        # Convert C++ GemmWarpPolicy to Python enum value (int)
        policy_int = self.policy.policy_type

        @T.prim_func
        def _gemm_cutedsl() -> None:
            gemm_v1(
                self.A,
                self.B,
                self.C,
                self.trans_A,
                self.trans_B,
                PyGemmWarpPolicy(policy_int),
                self.clear_accum,
                self.k_pack,
                self.wg_wait,
                self.mbar,
            )

        # Simplify and return
        return _Simplify(_gemm_cutedsl, inline_let=True)
