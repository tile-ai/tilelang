from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang import language as T
from tilelang.layout import Fragment
from tilelang.utils.language import is_fragment
from tvm.target import Target
from tvm.ir import Array, Range
from tvm import tir


def _as_static_int(value):
    try:
        return int(value)
    except TypeError:
        if hasattr(value, "value"):
            return int(value.value)
    return None


def _make_completed_replicated_layout_fragment(buffer: tir.Buffer, thread_nums: int):
    def forward_index_fn(*indices):
        return Array(list(indices))

    def forward_thread_fn(*indices_and_rep):
        return indices_and_rep[-1]

    return Fragment(
        list(buffer.shape),
        forward_thread_fn=forward_thread_fn,
        replicate=thread_nums,
        forward_index_fn=forward_index_fn,
        force_replicate_var=True,
    )


class GemmMetalScalar(GemmBase):
    """Metal scalar fallback with conservative replicated fragment output."""

    def infer_layout(self, target: Target, thread_nums: int):
        thread_nums = _as_static_int(thread_nums)
        if thread_nums is None:
            return {}

        layouts = {self.C: _make_completed_replicated_layout_fragment(self.C, thread_nums)}
        if is_fragment(self.A):
            layouts[self.A] = _make_completed_replicated_layout_fragment(self.A, thread_nums)
        if is_fragment(self.B):
            layouts[self.B] = _make_completed_replicated_layout_fragment(self.B, thread_nums)
        return layouts

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

        a0 = self.ARegion.region[-2].min
        a1 = self.ARegion.region[-1].min
        b0 = self.BRegion.region[-2].min
        b1 = self.BRegion.region[-1].min
        c0 = self.CRegion.region[-2].min
        c1 = self.CRegion.region[-1].min
        a_prefix = [r.min for r in self.ARegion.region[:-2]]
        b_prefix = [r.min for r in self.BRegion.region[:-2]]
        c_prefix = [r.min for r in self.CRegion.region[:-2]]

        @T.prim_func
        def _gemm_metal_scalar() -> None:
            if clear_accum:
                for i, j in T.grid(M, N):
                    C_buf[tuple(c_prefix) + (c0 + i, c1 + j)] = T.cast(0, accum_dtype)
            for i, j in T.grid(M, N):
                for k in T.Serial(K):
                    a_val = T.cast(
                        A_buf[tuple(a_prefix) + (a0 + (k if trans_A else i), a1 + (i if trans_A else k))],
                        accum_dtype,
                    )
                    b_val = T.cast(
                        B_buf[tuple(b_prefix) + (b0 + (j if trans_B else k), b1 + (k if trans_B else j))],
                        accum_dtype,
                    )
                    C_buf[tuple(c_prefix) + (c0 + i, c1 + j)] = C_buf[tuple(c_prefix) + (c0 + i, c1 + j)] + a_val * b_val

        return _gemm_metal_scalar
