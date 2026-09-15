from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.ascend import language as T
from tilelang.layout import (
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
    make_ascend_sf_layout,
    try_extract_fractal_layout,
)
from tilelang.transform.simplify import _Simplify
from tvm import DataType, arith, tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_MAD = "ascend.mad"


def _to_int32(value):
    return tirx.IntImm("int32", value) if isinstance(value, int) else value


def _compute_flat_offset(buf, region):
    offset = tirx.const(0, "int32")
    ndim = len(region.region)
    for i, rng in enumerate(region.region):
        stride = tirx.const(1, "int32")
        for j in range(i + 1, ndim):
            stride = stride * buf.shape[j]
        offset = offset + tirx.Cast("int32", rng.min) * stride
    return offset


def _compute_flat_offset_excluding_last(buf, region):
    offset = tirx.const(0, "int32")
    ndim = len(region.region)
    for i in range(ndim - 1):
        rng = region.region[i]
        stride = tirx.const(1, "int32")
        for j in range(i + 1, len(buf.shape)):
            stride = stride * buf.shape[j]
        offset = offset + tirx.Cast("int32", rng.min) * stride
    return offset


def _compute_extent(region):
    extent = tirx.const(1, "int32")
    for rng in region.region:
        extent = extent * rng.extent
    return extent


def _make_access_ptr(buf, region, rw_mask):
    return tirx.op.tvm_access_ptr(
        tirx.op.type_annotation(buf.dtype),
        buf.data,
        _compute_flat_offset(buf, region),
        _compute_extent(region),
        tirx.IntImm("int32", rw_mask),
    )


def _find_layout(layout_map, buf):
    if layout_map is None:
        return None
    for key in (buf.data if hasattr(buf, "data") else None, buf):
        if key is not None:
            layout = layout_map.get(key)
            if layout is not None:
                return layout
    for key, layout in layout_map.items():
        if hasattr(key, "data") and hasattr(buf, "data") and key.data.same_as(buf.data):
            return layout
        if hasattr(key, "name") and getattr(key, "name", None) == getattr(buf, "name", None):
            return layout
    return None


_ASCEND_DTYPE_MAP = {
    "bfloat16": "bfloat16_t",
    "float16": "half",
    "float32": "float",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "float8_e4m3": "float8_e4m3_t",
    "float8_e4m3fn": "float8_e4m3_t",
    "float4_e2m1fn": "float4_e2m1x2_t",
}


class GemmMAD(GemmBase):
    @property
    def is_blockscaled(self) -> bool:
        # An Ascend L0-input block-scaled gemm carries no SF regions on the
        # node (the scales were pre-loaded into the MX registers by
        # T.copy(scale=...)); the dialect marks it with the "blockscaled"
        # annotation instead, so widen the base's structural predicate.
        ann = getattr(self.gemm_node, "annotations", {})
        return bool(ann.get("blockscaled", False)) or super().is_blockscaled

    @property
    def unit_flag_ctrl(self) -> tirx.PrimExpr:
        ann = getattr(self.gemm_node, "annotations", {})
        return ann.get("unit_flag_ctrl", tirx.const(0, "int32"))

    def _check_blockscaled_k_alignment(self):
        if not self.is_blockscaled:
            return
        remainder = self.K % 64
        if isinstance(remainder, int):
            aligned = remainder == 0
        else:
            analyzer = arith.Analyzer()
            aligned = analyzer.can_prove_equal(remainder, tirx.const(0, remainder.dtype))
        assert aligned, (
            f"Ascend blockscaled GEMM requires operation K to be divisible by 64, got K={self.K}. "
            "Pad the participating L1/L0 buffers and invoke GEMM with the padded K extent."
        )

    def infer_layout(self, target: Target, thread_nums: int):
        self._check_blockscaled_k_alignment()
        layouts = {}
        # Choose K-major vs MN-major so that each operand's reduce-K axis lands
        # on the C0 axis. Both regular and blockscaled GEMM use the transpose
        # flags to pick the major; blockscaled layouts additionally require
        # K-alignment for the MX scale groups:
        #   A: trans_A=False -> K-major; trans_A=True -> MN-major
        #   B: trans_B=True  -> K-major; trans_B=False -> MN-major
        k_align = 64 if self.is_blockscaled else 1

        def _a_layout(buf):
            return make_ascend_major_mn_layout(buf, k_align=k_align) if self.trans_A else make_ascend_major_k_layout(buf, k_align=k_align)

        def _b_layout(buf):
            return make_ascend_major_k_layout(buf, k_align=k_align) if self.trans_B else make_ascend_major_mn_layout(buf, k_align=k_align)

        if self._is_l1_input() or self._is_l0_input() or self.is_blockscaled:
            layouts[self.A] = _a_layout(self.A)
            layouts[self.B] = _b_layout(self.B)
        if self.SFARegion is not None:
            layouts[self.SFARegion.buffer] = make_ascend_sf_layout(self.SFARegion.buffer)
        if self.SFBRegion is not None:
            layouts[self.SFBRegion.buffer] = make_ascend_sf_layout(self.SFBRegion.buffer)
        return layouts

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        self._check_blockscaled_k_alignment()
        self._layout_map = layout_map

        if self.C is not None:
            layout = _find_layout(layout_map, self.C)
            if layout is not None:
                info = try_extract_fractal_layout(layout, self.C)
                assert info is not None, f"GEMM C buffer {self.C.name} must carry an Ascend fractal layout"

        if self._is_l1_input():
            assert not self.trans_A and self.trans_B, "Ascend L1 GEMM currently only supports trans_A=False, trans_B=True (NT)."
            if self.is_blockscaled:
                assert self.SFARegion is not None and self.SFBRegion is not None, (
                    "blockscaled_gemm with L1 A/B inputs requires sfa and sfb buffers"
                )
                return self._lower_l1_blockscaled()
            return self._lower_l1()
        if self.is_blockscaled:
            return self._lower_l0_blockscaled()
        return self._lower_l0()

    def _is_l1_input(self):
        return self.A.scope() == "shared.l1" and self.B.scope() == "shared.l1"

    def _is_l0_input(self):
        return self.A.scope() == "shared.l0a" and self.B.scope() == "shared.l0b"

    def _l0_operation_extents(self):
        """Return the logical MAD M/N/K carried by the three L0 regions."""
        analyzer = arith.Analyzer()

        def prove_equal(lhs, rhs) -> bool:
            return analyzer.can_prove_equal(lhs, rhs)

        def require_zero_origin(name, region) -> None:
            for axis, rng in enumerate(region.region[-2:], start=-2):
                assert prove_equal(rng.min, tirx.const(0, rng.min.dtype)), (
                    f"Ascend compact L0 region {name} requires zero origin on "
                    f"trailing axis {axis}, got min={rng.min}. Split/non-zero-"
                    "origin L0 tiles are not supported."
                )

        require_zero_origin("A", self.ARegion)
        require_zero_origin("B", self.BRegion)
        require_zero_origin("C", self.CRegion)

        m = self.CRegion.region[-2].extent
        n = self.CRegion.region[-1].extent
        a_m = self.ARegion.region[-1].extent if self.trans_A else self.ARegion.region[-2].extent
        k = self.ARegion.region[-2].extent if self.trans_A else self.ARegion.region[-1].extent
        b_n = self.BRegion.region[-2].extent if self.trans_B else self.BRegion.region[-1].extent
        b_k = self.BRegion.region[-1].extent if self.trans_B else self.BRegion.region[-2].extent

        assert prove_equal(a_m, m), f"Ascend L0 GEMM M-region mismatch: A={a_m}, C={m}"
        assert prove_equal(b_n, n), f"Ascend L0 GEMM N-region mismatch: B={b_n}, C={n}"
        assert prove_equal(b_k, k), f"Ascend L0 GEMM K-region mismatch: A={k}, B={b_k}"
        return m, n, k

    def _input_dtype(self):
        return str(self.a_dtype)

    def _compute_tile_k_sub(self):
        input_dtype = self._input_dtype()
        elem_bits = DataType(input_dtype).bits
        assert 256 % elem_bits == 0, f"Unsupported dtype for Ascend GEMM: {input_dtype}"
        c0 = 256 // elem_bits
        l0_capacity = 65536
        num_l0_stages = 2
        l0_per_stage = l0_capacity // num_l0_stages
        max_k_a = l0_per_stage * 8 // (self.M * elem_bits)
        max_k_b = l0_per_stage * 8 // (self.N * elem_bits)
        max_k = min(max_k_a, max_k_b, self.K)
        tile_k_sub = (max_k // c0) * c0
        assert tile_k_sub > 0, f"Cannot fit any sub-K tile in L0: M={self.M}, N={self.N}, dtype={input_dtype}, L0 capacity={l0_capacity}"
        while self.K % tile_k_sub != 0 and tile_k_sub > c0:
            tile_k_sub -= c0
        assert tile_k_sub > 0 and self.K % tile_k_sub == 0, f"K ({self.K}) is not divisible by any valid TILE_K_SUB <= {max_k}"
        return tile_k_sub

    def _lower_l0(self):
        m, n, k = self._l0_operation_extents()
        c_ptr = _make_access_ptr(self.CRegion.buffer, self.CRegion, 2)
        a_ptr = _make_access_ptr(self.ARegion.buffer, self.ARegion, 1)
        b_ptr = _make_access_ptr(self.BRegion.buffer, self.BRegion, 1)
        call = tirx.call_intrin(
            "void",
            tirx.op.Op.get("tl.ascend_mad"),
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
        def _gemm_mad() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_mad, inline_let=True)

    def _lower_l0_blockscaled(self):
        # L0 block-scaled MAD. The per-block scale factors must already have been
        # loaded into the L0A/L0B MX scale registers (e.g. via
        # T.copy(l1_data, l0, sf=l1_sf)); asc_mmad_mx then applies them. The call
        # signature matches tl.ascend_mad — the scaling is implicit in the
        # hardware MX registers, so no SF pointer is passed here.
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

    def _lower_l1(self):
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

        input_dtype = self._input_dtype()
        dtype_str = _ASCEND_DTYPE_MAP.get(input_dtype)
        assert dtype_str is not None, f"Unsupported dtype for Ascend GEMM: {input_dtype}"

        call = tirx.call_intrin(
            "void",
            tirx.op.Op.get("tl.ascend_gemm_l1"),
            c_ptr,
            a_ptr,
            b_ptr,
            _to_int32(self.M),
            _to_int32(self.K),
            _to_int32(self.N),
            _to_int32(tile_k_sub),
            _to_int32(1 if self.trans_B else 0),
            self.clear_accum,
            tirx.StringImm(dtype_str),
            _to_int32(0),  # buf_offset
            _to_int32(self.unit_flag_ctrl),
        )

        @T.prim_func
        def _gemm_mad_l1() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_mad_l1, inline_let=True)

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
        assert sfa_region is not None and sfb_region is not None, "blockscaled_gemm requires sfa and sfb buffers"
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
        assert sfa_info is not None and sfa_info.kind == 2, f"blockscaled_gemm sfa buffer {sfa_buf.name} must carry an Ascend SF_K layout"
        assert sfa_info.c0_axis == 1, f"blockscaled_gemm sfa buffer {sfa_buf.name}: SF K axis must be col, got c0_axis={sfa_info.c0_axis}"
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
