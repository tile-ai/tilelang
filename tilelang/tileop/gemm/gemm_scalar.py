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

        # Region offsets: use the last two dimensions so rank>2 buffers
        # (e.g. leading singleton dims) are handled correctly.
        a_ndim = len(self.ARegion.region)
        b_ndim = len(self.BRegion.region)
        c_ndim = len(self.CRegion.region)
        a0 = self.ARegion.region[a_ndim - 2].min
        a1 = self.ARegion.region[a_ndim - 1].min
        b0 = self.BRegion.region[b_ndim - 2].min
        b1 = self.BRegion.region[b_ndim - 1].min
        c0 = self.CRegion.region[c_ndim - 2].min
        c1 = self.CRegion.region[c_ndim - 1].min
        # Build prefix indices for leading dimensions (all at their region min).
        a_prefix = [self.ARegion.region[d].min for d in range(a_ndim - 2)]
        b_prefix = [self.BRegion.region[d].min for d in range(b_ndim - 2)]
        c_prefix = [self.CRegion.region[d].min for d in range(c_ndim - 2)]

        @T.prim_func
        def _gemm_scalar() -> None:
            if clear_accum:
                # Only clear the output tile, not the entire backing buffer.
                for ci, cj in T.grid(M, N):
                    C_buf[*c_prefix, c0 + ci, c1 + cj] = T.cast(0, accum_dtype)
            for i, j, k in T.grid(M, N, K):
                C_buf[*c_prefix, c0 + i, c1 + j] += T.cast(
                    A_buf[*a_prefix, a0 + (k if trans_A else i), a1 + (i if trans_A else k)]
                    * B_buf[*b_prefix, b0 + (j if trans_B else k), b1 + (k if trans_B else j)],
                    accum_dtype,
                )

        return _gemm_scalar
