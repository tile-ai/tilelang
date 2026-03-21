from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang import language as T
from tvm.target import Target
from tvm.ir import Range
from tvm import tir


class GemmScalar(GemmBase):
    """CPU scalar fallback: triple nested loop gemm."""

    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        M, N, K = self.M, self.N, self.K
        A_buf = self.ARegion.buffer
        B_buf = self.BRegion.buffer
        C_buf = self.CRegion.buffer
        trans_A = self.trans_A
        trans_B = self.trans_B
        clear_accum = self.clear_accum
        accum_dtype = self.accum_dtype

        # Region offsets for strided gemm (e.g. T.gemm(A[0:64, :], B, C))
        a0 = self.ARegion.region[0].min
        a1 = self.ARegion.region[1].min
        b0 = self.BRegion.region[0].min
        b1 = self.BRegion.region[1].min
        c0 = self.CRegion.region[0].min
        c1 = self.CRegion.region[1].min

        @T.prim_func
        def _gemm_scalar() -> None:
            if clear_accum:
                T.clear(C_buf)
            for i, j, k in T.grid(M, N, K):
                C_buf[c0 + i, c1 + j] += T.cast(
                    A_buf[a0 + (k if trans_A else i), a1 + (i if trans_A else k)]
                    * B_buf[b0 + (j if trans_B else k), b1 + (k if trans_B else j)],
                    accum_dtype,
                )

        return _gemm_scalar
