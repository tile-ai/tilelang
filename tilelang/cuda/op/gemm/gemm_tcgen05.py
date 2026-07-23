from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.layout import (
    Layout,
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
    make_linear_layout,
)
from tilelang.cuda.intrinsics.macro.tcgen05_macro_generator import (
    TensorCoreIntrinEmitter,
)
from tilelang import language as T
from tilelang.utils.language import retrieve_ptr
from tilelang.transform.simplify import _Simplify
from tvm import tirx
from tvm.target import Target
from tvm.ir import Range
from tvm.arith import Analyzer
from collections.abc import Callable


_FLOAT8_DTYPES = {
    "float8_e4m3",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fn",
    "float8_e5m2fnuz",
}


def _shared_layout_continuity(buffer, is_k_major: bool, k_extent: int, mn_extent: int) -> int:
    dtype_bits = buffer.dtype.bits
    if dtype_bits < 8:
        return int(buffer.shape[-1]) if is_k_major else mn_extent
    return k_extent if is_k_major else mn_extent


GEMM_INST_TCGEN05 = "cuda.tcgen05"


class GemmTCGEN5(GemmBase):
    """GEMM operator for Blackwell (SM100) TCGEN5MMA instructions.

    Supports the SS (Shared-Shared) and TS (TensorMemory-Shared) variants,
    as well as block-scaled MXFP8 GEMM when SFA/SFB scale factors are present.
    Layout inference and lowering are dispatched based on the memory scopes
    of operands A and B.
    """

    @property
    def allow_f8f6f4_mixed_dtypes(self) -> bool:
        return True

    def infer_shared_layout(self, buffer: tirx.Buffer, continuity: int) -> Callable[[tirx.Buffer], Layout]:
        """Infer a standard shared-memory swizzle layout for TCGEN05 operands."""
        elem_bits = buffer.dtype.bits
        vectorized_size = 128 // elem_bits
        if continuity % (vectorized_size * 8) == 0:
            return make_full_bank_swizzled_layout
        elif continuity % (vectorized_size * 4) == 0:
            return make_half_bank_swizzled_layout
        elif continuity % (vectorized_size * 2) == 0:
            return make_quarter_bank_swizzled_layout
        else:
            return make_linear_layout

    def infer_layout(self, target: Target, thread_nums: int):
        """Infer swizzled layouts for operands and accumulator.

        For SS: both A and B get swizzled shared-memory layouts.
        For TS: A and C get TMEM store layouts, B gets a swizzled shared-memory layout.
        For block-scaled: same as SS (A and B get swizzle, C gets TMEM store layout).
        """
        # Block-scaled GEMM keeps a 1x1 warp partition even when using cta_group::2.
        if self.is_blockscaled:
            m_warp, n_warp = 1, 1
        else:
            m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_TCGEN05)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mma_emitter = TensorCoreIntrinEmitter(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
        )
        a_is_k_major = not self.trans_A
        b_is_k_major = self.trans_B

        annotations = getattr(self.gemm_node, "annotations", {})
        use_2cta = bool(annotations.get("use_2cta", 0))
        k = int(self.chunk)
        mma_emitter.get_tcgen5_mma_meta(int(self.M), int(self.N), k, disable_2cta=not use_2cta, disable_ws=self.is_blockscaled)

        if self.is_blockscaled or self.is_gemm_ss():
            a_continuity = _shared_layout_continuity(self.A, a_is_k_major, self.K, self.M)
            b_continuity = _shared_layout_continuity(self.B, b_is_k_major, self.K, int(self.B.shape[-1]))

            return {
                self.A: self.infer_shared_layout(self.A, a_continuity)(self.A),
                self.B: self.infer_shared_layout(self.B, b_continuity)(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        if self.is_gemm_ts():
            b_continuity = _shared_layout_continuity(self.B, b_is_k_major, self.K, int(self.B.shape[-1]))
            layouts = {
                self.A: mma_emitter.make_mma_store_layout(self.A),
                self.B: self.infer_shared_layout(self.B, b_continuity)(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
            return layouts
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        """Lower the GEMM tile-op into a TIR prim_func containing TCGEN5MMA calls."""
        thread_nums = thread_bounds.extent
        # Block-scaled GEMM keeps a 1x1 warp partition even when using cta_group::2.
        if self.is_blockscaled:
            m_warp, n_warp = 1, 1
        else:
            m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_TCGEN05)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mma_emitter = TensorCoreIntrinEmitter(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
        )

        if self.A in layout_map:
            mma_emitter._assign_a_shared_layout(layout_map[self.A])
        if self.B in layout_map:
            mma_emitter._assign_b_shared_layout(layout_map[self.B])

        if self.is_blockscaled:
            return self._lower_blockscaled(mma_emitter, thread_bounds, thread_index, mbar_phase_expr)

        if not (self.is_gemm_ss() or self.is_gemm_ts()):
            raise ValueError(f"TCGEN5MMA supports gemm_ss and gemm_ts, got A scope {self.A.scope()}, B scope {self.B.scope()}")

        annotations = getattr(self.gemm_node, "annotations", {})
        use_2cta = bool(annotations.get("use_2cta", 0))
        k = int(self.chunk)
        mma_emitter.get_tcgen5_mma_meta(int(self.M), int(self.N), k, disable_2cta=not use_2cta)
        atom_m, atom_n, atom_k, enable_ws, enable_2cta = mma_emitter.meta

        if self.A.scope() not in {"shared", "shared.dyn", "shared.tmem"}:
            raise ValueError(f"Unsupported A scope for TCGEN5MMA: {self.A.scope()}")
        if self.B.scope() not in {"shared", "shared.dyn"}:
            raise ValueError(f"Unsupported B scope for TCGEN5MMA: {self.B.scope()}")
        if self.C.scope() != "shared.tmem":
            raise ValueError(f"TCGEN5MMA expects C in shared.tmem, got {self.C.scope()}")
        if self.wg_wait not in (0, -1):
            raise ValueError("TCGEN5MMA only accepts wg_wait in {0, -1}")

        mbar = self.mbar
        if mbar is None:
            raise ValueError("TCGEN5MMA requires a valid mbarrier")

        mbarptr = retrieve_ptr(mbar, "rw")

        C_coords = self.C_coords
        if len(C_coords) != 2:
            raise ValueError("TCGEN5MMA expects 2D coordinates for C buffer access")

        accum_dtype = str(self.C.dtype)
        if accum_dtype not in [str(T.float32), str(T.float16), str(T.int32)]:
            raise ValueError(f"Unsupported accumulator dtype for TCGEN5MMA: {accum_dtype}")

        A_shared = self.ARegion
        B_shared = self.BRegion
        C_local = self.C
        clear_accum = self.clear_accum
        mbar_phase = mbar_phase_expr if mbar_phase_expr is not None else 0

        # Since TCGEN5MMA atoms provided by CUTLASS always have an internal
        # `elect_one_sync()`, we check if we are calling it using full warps
        analyzer = Analyzer()
        warp_size = 32
        assert analyzer.can_prove(thread_bounds.min % warp_size == 0 and thread_bounds.extent % warp_size == 0), (
            "TCGEN5MMA requires thread bounds to be multiples of warp size (32) and aligned to warps."
        )

        cluster_cond = not enable_2cta or T.block_rank_in_cluster() == 0

        @T.prim_func
        def _gemm_ss_cond() -> None:
            if cluster_cond and thread_index // 32 == thread_bounds.min // warp_size:
                mma_emitter.tcgen05mma(A_shared, B_shared, C_local, mbarptr, clear_accum)
            if not self.is_tcgen05:
                T.mbarrier_wait_parity(mbar, mbar_phase)

        @T.prim_func
        def _gemm_ss() -> None:
            if cluster_cond:
                mma_emitter.tcgen05mma(A_shared, B_shared, C_local, mbarptr, clear_accum)
            if not self.is_tcgen05:
                T.mbarrier_wait_parity(mbar, mbar_phase)

        return (
            _Simplify(_gemm_ss, inline_let=True)
            if analyzer.can_prove(thread_bounds.extent == warp_size)
            else _Simplify(_gemm_ss_cond, inline_let=True)
        )

    def _lower_blockscaled(self, mma_emitter, thread_bounds, thread_index, mbar_phase_expr: tirx.PrimExpr | None = None):
        """Lower block-scaled MXFP8 GEMM to TIR.

        Block-scaled GEMM follows explicit-async TCGEN5MMA semantics: the MMA
        issue posts completion to `mbar`, and the user (or pipeline pass) is
        responsible for waiting on that barrier at the consumption point. We
        therefore never auto-emit `mbarrier_wait_parity` here. This mirrors the
        `is_tcgen05=True` branch of `_gemm_ss`. `mbar_phase_expr` is accepted
        for API consistency with the rest of the `GemmPyNode.Lower` chain and
        so that a future synchronous block-scaled path can use it without
        needing another signature change.
        """
        mbar = self.mbar
        if mbar is None:
            raise ValueError("Block-scaled GEMM requires a valid mbarrier")
        mbarptr = retrieve_ptr(mbar, "rw")

        A_shared = self.ARegion
        B_shared = self.BRegion
        C_local = self.C
        clear_accum = self.clear_accum
        SFA_tmem = self.SFARegion.buffer
        SFB_tmem = self.SFBRegion.buffer
        sf_k_start = self.sf_k_start
        # NOTE: mbar_phase_expr is intentionally unused in the current
        # frontend, which always requests explicit-async semantics. Keep the
        # parameter so the signature matches `_gemm_ss` and the call site in
        # `lower()` does not need a special case.
        del mbar_phase_expr

        annotations = getattr(self.gemm_node, "annotations", {})
        use_2cta = bool(annotations.get("use_2cta", 0))
        sf_a_granularity_k = annotations.get("sf_a_granularity_k")
        sf_b_granularity_k = annotations.get("sf_b_granularity_k")
        if sf_a_granularity_k is None or sf_b_granularity_k is None:
            raise ValueError("Block-scaled GEMM requires sf_a_granularity_k and sf_b_granularity_k")
        k = int(self.chunk)
        mma_emitter.get_tcgen5_mma_meta(int(self.M), int(self.N), k, disable_2cta=not use_2cta, disable_ws=True)
        _atom_m, _atom_n, _atom_k, _enable_ws, enable_2cta = (int(x) for x in mma_emitter.meta)

        analyzer = Analyzer()
        warp_size = 32
        assert analyzer.can_prove(thread_bounds.min % warp_size == 0 and thread_bounds.extent % warp_size == 0), (
            "Block-scaled GEMM requires thread bounds aligned to warps."
        )
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
                    sf_a_granularity_k=int(sf_a_granularity_k),
                    sf_b_granularity_k=int(sf_b_granularity_k),
                    clear_accum=clear_accum,
                )

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
                    sf_a_granularity_k=int(sf_a_granularity_k),
                    sf_b_granularity_k=int(sf_b_granularity_k),
                    clear_accum=clear_accum,
                )

        return (
            _Simplify(_gemm_blockscaled, inline_let=True)
            if analyzer.can_prove(thread_bounds.extent == warp_size)
            else _Simplify(_gemm_blockscaled_cond, inline_let=True)
        )
