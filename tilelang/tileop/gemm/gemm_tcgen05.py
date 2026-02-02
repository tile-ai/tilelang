from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.layout import make_tcgen05mma_swizzled_layout
from tilelang.intrinsics.tcgen05_macro_generator import (
    TensorCoreIntrinEmitter,
)
from tilelang import language as T
from tilelang.utils.language import retrieve_ptr
from tilelang.transform.simplify import _Simplify
from tvm import tir
from tvm.target import Target
from tvm.ir import Range
from tvm.arith import Analyzer


_FLOAT8_DTYPES = {
    "float8_e4m3",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fn",
    "float8_e5m2fnuz",
}


class GemmTCGEN5(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.TCGEN5MMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mma_emitter = TensorCoreIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
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

        if self.is_gemm_ss():
            a_continuity = self.M if a_is_k_major else 4 * self.K // m_warp
            b_continuity = self.K if b_is_k_major else self.N // n_warp

            return {
                # WGMMA does not support padding
                self.A: make_tcgen05mma_swizzled_layout(self.A, continuity=a_continuity, k_major=a_is_k_major),
                self.B: make_tcgen05mma_swizzled_layout(self.B, continuity=b_continuity, k_major=b_is_k_major),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        # No special swizzle requirement; rely on existing layout.
        return {}

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
        thread_nums = thread_bounds.extent
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.TCGEN5MMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mma_emitter = TensorCoreIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
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

        if not self.is_gemm_ss():
            raise ValueError(f"TCGEN5MMA currently only supports gemm_ss, got A scope {self.A.scope()}, B scope {self.B.scope()}")

        atom_m, atom_n, atom_k, enable_ws, enable_2cta = mma_emitter.get_tcgen5_mma_meta(self.M, self.N, self.K)

        if self.A.scope() not in {"shared", "shared.dyn", "shared.tmem"}:
            raise ValueError(f"Unsupported A scope for TCGEN5MMA: {self.A.scope()}")
        if self.B.scope() not in {"shared", "shared.dyn"}:
            raise ValueError(f"Unsupported B scope for TCGEN5MMA: {self.B.scope()}")
        if self.C.scope() != "shared.tmem":
            raise ValueError(f"TCGEN5MMA expects C in shared.tmem, got {self.C.scope()}")
        if self.wg_wait != -1:
            raise ValueError("TCGEN5MMA currently requires wg_wait == -1")

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

        # Since TCGEN5MMA atoms provided by CUTLASS always have an internal
        # `elect_one_sync()`, we check if we are calling it using full warps
        analyzer = Analyzer()
        warp_size = 32
        assert analyzer.can_prove(thread_bounds.min % warp_size == 0 and thread_bounds.extent % warp_size == 0), (
            "TCGEN5MMA requires thread bounds to be multiples of warp size (32) and aligned to warps."
        )

        @T.prim_func
        def _gemm_ss_cond() -> None:
            if thread_var // 32 == thread_bounds.min // warp_size:
                mma_emitter.tcgen05mma(A_shared, B_shared, C_local, mbarptr, clear_accum)

        @T.prim_func
        def _gemm_ss() -> None:
            mma_emitter.tcgen05mma(A_shared, B_shared, C_local, mbarptr, clear_accum)

        return (
            _Simplify(_gemm_ss, inline_let=True)
            if analyzer.can_prove(thread_bounds.extent == warp_size)
            else _Simplify(_gemm_ss_cond, inline_let=True)
        )
