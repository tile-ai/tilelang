from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.utils.language import is_fragment, is_full_region
from tilelang import language as T
from tilelang.transform.simplify import _Simplify
from tvm.target import Target
from tvm.ir import Range
from tvm import tirx


GEMM_INST_FMA = "cuda.fma"


def _linear_fragment(local_buf, thread_nums: int) -> T.Fragment:
    shape = list(local_buf.shape)
    strides = [1 for _ in shape]
    for idx in range(len(shape) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * shape[idx + 1]

    def forward(*indices):
        flat = 0
        for index, stride in zip(indices, strides):
            flat += index * stride
        return flat % thread_nums, flat // thread_nums

    return T.Fragment(shape, forward_fn=forward)


class GemmFMA(GemmBase):
    """CUDA scalar FMA fallback for GEMM combinations without tensor-core MMA."""

    def infer_layout(self, target: Target, thread_nums: int):
        layouts = {}
        if is_fragment(self.A):
            layouts[self.A] = _linear_fragment(self.A, thread_nums)
        if is_fragment(self.B):
            layouts[self.B] = _linear_fragment(self.B, thread_nums)
        if is_fragment(self.C):
            layouts[self.C] = _linear_fragment(self.C, thread_nums)
        return layouts

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        M, N, K = self.M, self.N, self.K
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        assert len(A_region.region) >= 2, "FMA GEMM requires at least 2D A"
        assert len(B_region.region) >= 2, "FMA GEMM requires at least 2D B"
        assert len(C_region.region) >= 2, "FMA GEMM requires at least 2D C"
        if is_fragment(self.C):
            assert is_full_region(C_region), "Fragment output C must be a full region"

        a_rows = A_region.region[-2].extent
        a_cols = A_region.region[-1].extent
        b_rows = B_region.region[-2].extent
        b_cols = B_region.region[-1].extent

        A_other = [r.min for r in A_region.region[:-2]]
        B_other = [r.min for r in B_region.region[:-2]]
        C_other = [r.min for r in C_region.region[:-2]]
        a0 = A_region.region[-2].min
        a1 = A_region.region[-1].min
        b0 = B_region.region[-2].min
        b1 = B_region.region[-1].min
        c0 = C_region.region[-2].min
        c1 = C_region.region[-1].min

        trans_A = self.trans_A
        trans_B = self.trans_B
        clear_accum = self.clear_accum
        in_dtype = self.in_dtype
        accum_dtype = self.accum_dtype
        thread_nums = thread_bounds.extent

        @T.prim_func
        def _gemm_fma() -> None:
            A_stage = T.alloc_shared((a_rows, a_cols), in_dtype)
            B_stage = T.alloc_shared((b_rows, b_cols), in_dtype)

            T.sync_threads()
            for a_flat in T.serial(thread_var, a_rows * a_cols, thread_nums):
                ai = a_flat // a_cols
                aj = a_flat % a_cols
                A_stage[ai, aj] = A_buf[tuple(A_other) + (a0 + ai, a1 + aj)]
            for b_flat in T.serial(thread_var, b_rows * b_cols, thread_nums):
                bi = b_flat // b_cols
                bj = b_flat % b_cols
                B_stage[bi, bj] = B_buf[tuple(B_other) + (b0 + bi, b1 + bj)]
            T.sync_threads()

            for c_flat in T.serial(thread_var, M * N, thread_nums):
                i = c_flat // N
                j = c_flat % N
                if clear_accum:
                    C_buf[tuple(C_other) + (c0 + i, c1 + j)] = T.cast(0, accum_dtype)
                for k in T.serial(K):
                    C_buf[tuple(C_other) + (c0 + i, c1 + j)] += T.cast(
                        T.cast(A_stage[k if trans_A else i, i if trans_A else k], accum_dtype)
                        * T.cast(B_stage[j if trans_B else k, k if trans_B else j], accum_dtype),
                        accum_dtype,
                    )

        return _Simplify(_gemm_fma, inline_let=True)
