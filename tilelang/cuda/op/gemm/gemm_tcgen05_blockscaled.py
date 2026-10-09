"""Blackwell TCGEN5MMA block-scaled GEMM lowering."""

from __future__ import annotations

from tilelang import language as T
from tilelang.tileop.gemm_blockscaled.gemm_blockscaled_base import GemmBlockScaledMixin
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import retrieve_ptr
from tvm import tirx
from tvm.arith import Analyzer
from tvm.ir import Range
from tvm.target import Target

from .gemm_tcgen05 import GemmTCGEN5

GEMM_INST_TCGEN05_BLOCK_SCALED = "cuda.tcgen05.blockscaled"


class GemmTCGEN5BlockScaled(GemmBlockScaledMixin, GemmTCGEN5):
    """``kind::mxf8f6f4.block_scale`` TCGEN5MMA with A/B in shared memory,
    the accumulator in tensor memory and SFA/SFB already resident in tensor
    memory.

    Shares the shared-operand swizzle inference and emitter plumbing with the
    dense ``GemmTCGEN5``; differs in the fixed 1x1 warp partition (kept even
    under ``cta_group::2``), the dense-only (no ``.ws``) instruction shapes and
    the block-scaled MMA issue.
    """

    tcgen05_allow_ws = False

    def _warp_partition(self, target: Target, thread_nums: int) -> tuple[int, int]:
        return 1, 1

    def _validate_operands(self) -> None:
        if not self.is_gemm_ss():
            raise ValueError(
                f"Block-scaled TCGEN5MMA supports shared-memory A/B operands only, got A scope {self.A.scope()}, B scope {self.B.scope()}"
            )

    def infer_layout(self, target: Target, thread_nums: int):
        self._validate_operands()
        return super().infer_layout(target, thread_nums)

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        """Lower to TIR containing block-scaled TCGEN5MMA calls.

        Follows the same completion protocol as the dense TCGEN05 lowering:
        the synchronous `T.gemm_blockscaled` posts completion to `mbar` and
        waits on it right after issue, while the explicit `is_tcgen05` op
        never waits and may omit `mbar` when the caller or WS schedule emits
        a later completion arrival.
        """
        self._validate_operands()
        mma_emitter = self._make_mma_emitter(target, thread_bounds.extent)
        self._assign_layouts(mma_emitter, layout_map)

        mbar = self.mbar
        if mbar is None and not self.is_tcgen05:
            raise ValueError("Synchronous block-scaled TCGEN5MMA requires a valid mbarrier")
        mbarptr = retrieve_ptr(mbar, "rw") if mbar is not None else None

        A_shared = self.ARegion
        B_shared = self.BRegion
        C_local = self.CRegion
        clear_accum = self.clear_accum
        SFA_tmem = self.SFARegion.buffer
        SFB_tmem = self.SFBRegion.buffer
        sf_k_start = self.sf_k_start
        sf_a_granularity_k = self.sf_a_granularity_k
        sf_b_granularity_k = self.sf_b_granularity_k
        mbar_phase = mbar_phase_expr if mbar_phase_expr is not None else 0

        k = int(self.chunk)
        mma_emitter.get_tcgen5_mma_meta(int(self.M), int(self.N), k, disable_2cta=not self.use_2cta, disable_ws=True)
        enable_2cta = mma_emitter.tcgen05_meta.enable_2cta

        analyzer = Analyzer()
        warp_size = 32
        self._check_warp_aligned_thread_bounds(thread_bounds, analyzer)
        cluster_cond = not enable_2cta or T.block_rank_in_cluster() == 0

        @T.prim_func
        def _gemm_blockscaled_cond() -> None:
            if cluster_cond and thread_index // 32 == thread_bounds.min // warp_size:
                mma_emitter.tcgen05mma_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_tmem,
                    SFB_tmem,
                    mbarptr,
                    sf_k_start=sf_k_start,
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
                    clear_accum=clear_accum,
                )
            if not self.is_tcgen05:
                T.mbarrier_wait_parity(mbar, mbar_phase)

        @T.prim_func
        def _gemm_blockscaled() -> None:
            if cluster_cond:
                mma_emitter.tcgen05mma_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_tmem,
                    SFB_tmem,
                    mbarptr,
                    sf_k_start=sf_k_start,
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
                    clear_accum=clear_accum,
                )
            if not self.is_tcgen05:
                T.mbarrier_wait_parity(mbar, mbar_phase)

        return (
            _Simplify(_gemm_blockscaled, inline_let=True)
            if analyzer.can_prove(thread_bounds.extent == warp_size)
            else _Simplify(_gemm_blockscaled_cond, inline_let=True)
        )
