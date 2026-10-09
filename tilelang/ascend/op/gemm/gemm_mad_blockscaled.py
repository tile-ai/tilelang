"""Ascend block-scaled MAD GEMM lowering."""

from __future__ import annotations

from tilelang.layout import make_ascend_sf_layout, try_extract_fractal_layout
from tilelang.tileop.gemm_blockscaled.gemm_blockscaled_base import GemmBlockScaledMixin
from tilelang.transform.simplify import _Simplify
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target

from tilelang.ascend import language as T

from .gemm_mad import (
    _ASCEND_DTYPE_MAP,
    _compute_extent,
    _compute_flat_offset_excluding_last,
    _find_layout,
    _make_access_ptr,
    _to_int32,
    GemmMAD,
)

GEMM_INST_MAD_BLOCK_SCALED = "ascend.mad.blockscaled"


class GemmMADBlockScaled(GemmBlockScaledMixin, GemmMAD):
    """MXFP8 MAD with explicit SFA/SFB scale-factor operands.

    L1 A/B inputs lower to ``tl.ascend_blockscaled_gemm_l1`` with the scale
    pointers. L0 A/B inputs lower to ``tl.ascend_mad_mx``: SFA/SFB are the MX
    slot handles of the data tiles (``alloc_l0a_sf``/``alloc_l0b_sf``), loaded
    by a preceding ``T.copy(sf_l1, view)``; the MAD reads the slots implied
    by its A/B data addresses, so the SF operands only contribute their read
    regions to scheduling.
    """

    @property
    def is_blockscaled(self) -> bool:
        return True

    def infer_layout(self, target: Target, thread_nums: int):
        layouts = super().infer_layout(target, thread_nums)
        # L1-scoped scale operands carry the SF_K fractal layout, consumed by
        # the fused L1 lowering. L0-flavor operands are the MX slot handles
        # (shared.l0a.sf/.l0b.sf, alloc_l0a_sf/alloc_l0b_sf): their
        # addressing is fixed by the ISA to the bound data tile's address,
        # so they carry no layout.
        for sf_region in (self.SFARegion, self.SFBRegion):
            if sf_region.buffer.scope() == "shared.l1":
                layouts[sf_region.buffer] = make_ascend_sf_layout(sf_region.buffer)
        return layouts

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        self._prepare_lower(layout_map)

        if self._is_l1_input():
            assert not self.trans_A and self.trans_B, "Ascend L1 GEMM currently only supports trans_A=False, trans_B=True (NT)."
            return self._lower_l1_blockscaled()
        return self._lower_l0_blockscaled()

    def _lower_l0_blockscaled(self):
        # L0 block-scaled MAD. The per-block scale factors must already have
        # been loaded into the L0A/L0B MX scale slots by a T.copy into the SF
        # views this op carries as SFA/SFB; asc_mmad_mx then applies them via
        # its A/B data addresses. The call signature matches tl.ascend_mad —
        # the scaling is implicit in the hardware MX slots, so no SF pointer
        # is passed here.
        m, n, k = self._l0_operation_extents()
        c_ptr = _make_access_ptr(self.CRegion.buffer, self.CRegion, 2)
        a_ptr = _make_access_ptr(self.ARegion.buffer, self.ARegion, 1)
        b_ptr = _make_access_ptr(self.BRegion.buffer, self.BRegion, 1)
        call = tirx.call_intrin(
            "void",
            tirx.op.Op.get("tl.ascend_mad_mx"),
            c_ptr,
            a_ptr,
            b_ptr,
            _to_int32(m),
            _to_int32(k),
            _to_int32(n),
            _to_int32(self.unit_flag_ctrl),
            _to_int32(1),  # gemv_ctrl: 1 disables the specialized GEMV mode
            _to_int32(0),  # BTbuf_ctrl
            self.clear_accum,
        )

        @T.prim_func
        def _gemm_mad_mx() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_mad_mx, inline_let=True)

    def _lower_l1_blockscaled(self):
        tile_k_sub = self._compute_tile_k_sub()
        c_ptr = _make_access_ptr(self.CRegion.buffer, self.CRegion, 2)

        a_buf = self.ARegion.buffer
        b_buf = self.BRegion.buffer
        a_ptr = tirx.op.tvm_access_ptr(
            tirx.op.type_annotation(a_buf.dtype),
            a_buf.data,
            _compute_flat_offset_excluding_last(a_buf, self.ARegion),
            _compute_extent(self.ARegion),
            tirx.IntImm("int32", 1),
        )
        b_ptr = tirx.op.tvm_access_ptr(
            tirx.op.type_annotation(b_buf.dtype),
            b_buf.data,
            _compute_flat_offset_excluding_last(b_buf, self.BRegion),
            _compute_extent(self.BRegion),
            tirx.IntImm("int32", 1),
        )

        sfa_region = self.SFARegion
        sfb_region = self.SFBRegion
        sfa_buf = sfa_region.buffer
        sfb_buf = sfb_region.buffer
        sfa_ptr = tirx.op.tvm_access_ptr(
            tirx.op.type_annotation(sfa_buf.dtype),
            sfa_buf.data,
            _compute_flat_offset_excluding_last(sfa_buf, sfa_region),
            _compute_extent(sfa_region),
            tirx.IntImm("int32", 1),
        )
        sfb_ptr = tirx.op.tvm_access_ptr(
            tirx.op.type_annotation(sfb_buf.dtype),
            sfb_buf.data,
            _compute_flat_offset_excluding_last(sfb_buf, sfb_region),
            _compute_extent(sfb_region),
            tirx.IntImm("int32", 1),
        )

        input_dtype = self._input_dtype()
        in_dtype_str = _ASCEND_DTYPE_MAP.get(input_dtype)
        assert in_dtype_str is not None, f"Unsupported dtype for Ascend blockscaled GEMM: {input_dtype}"

        def _sf_dtype_str(dtype) -> str:
            dtype_str = str(dtype)
            if "uint8" in dtype_str:
                return "uint8_t"
            if "int8" in dtype_str:
                return "int8_t"
            if "uint16" in dtype_str:
                return "uint16_t"
            if "int16" in dtype_str:
                return "int16_t"
            if "float8_e4m3" in dtype_str:
                return "float8_e4m3_t"
            return dtype_str

        sf_dtype_str = _sf_dtype_str(sfa_buf.dtype)
        sfa_layout = _find_layout(getattr(self, "_layout_map", {}), sfa_buf)
        sfa_info = try_extract_fractal_layout(sfa_layout, sfa_buf) if sfa_layout is not None else None
        assert sfa_info is not None and sfa_info.kind == 2, f"gemm_blockscaled SFA buffer {sfa_buf.name} must carry an Ascend SF_K layout"
        assert sfa_info.c0_axis == 1, f"gemm_blockscaled SFA buffer {sfa_buf.name}: SF K axis must be col, got c0_axis={sfa_info.c0_axis}"
        sf_nz_stride = sfa_info.outer1
        sf_k_offset = _to_int32(sfa_region.region[-1].min // sfa_info.c0)

        call = tirx.call_intrin(
            "void",
            tirx.op.Op.get("tl.ascend_blockscaled_gemm_l1"),
            c_ptr,
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            _to_int32(self.M),
            _to_int32(self.K),
            _to_int32(self.N),
            _to_int32(tile_k_sub),
            _to_int32(1 if self.trans_B else 0),
            self.clear_accum,
            tirx.StringImm(in_dtype_str),
            tirx.StringImm(sf_dtype_str),
            tirx.StringImm("float"),
            _to_int32(0),  # buf_offset
            sf_k_offset,  # sf_k_offset (auto from region slice)
            _to_int32(sf_nz_stride),
            _to_int32(self.unit_flag_ctrl),
        )

        @T.prim_func
        def _gemm_mad_l1_blockscaled() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_mad_l1_blockscaled, inline_let=True)
