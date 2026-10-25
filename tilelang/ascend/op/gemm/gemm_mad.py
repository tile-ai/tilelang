from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.ascend import language as T
from tilelang.layout import (
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
    try_extract_fractal_layout,
)
from tilelang.transform.simplify import _Simplify
from tilelang._compat.tvm import DataType, arith, tirx
from tvm_tirx.ir import Range
from tvm_tirx import ir
from tvm_tirx.target import Target


GEMM_INST_MAD = "ascend.mad"


def _to_int32(value):
    return tirx.IntImm("int32", value) if isinstance(value, int) else value


def _compute_extent(region):
    extent = tirx.const(1, "int32")
    for rng in region.region:
        extent = extent * rng.extent
    return extent


def _make_access_ptr(buf, region, rw_mask, *, exclude_last=False):
    indices = [axis.begin for axis in region.region]
    if exclude_last and indices:
        indices[-1] = ir.const(0, str(indices[-1].ty))
    return ir.Call("tl.access_ptr", [ir.TensorLoad(buf, indices), _compute_extent(region), ir.const(rw_mask)])


def _find_layout(layout_map, buf):
    if layout_map is None:
        return None
    return layout_map.get(buf)


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
        # Structural, matching the C++ node hierarchy: explicit SFA/SFB
        # operands build a tl.tileop.gemm_blockscaled op handled by
        # GemmMADBlockScaled, which overrides this to True.
        return False

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
            aligned = analyzer.can_prove_equal(remainder, tirx.const(0, remainder.ty))
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
        return layouts

    def _prepare_lower(self, layout_map: dict) -> None:
        self._check_blockscaled_k_alignment()
        self._layout_map = layout_map

        if self.C is not None:
            layout = _find_layout(layout_map, self.C)
            if layout is not None:
                info = try_extract_fractal_layout(layout, self.C)
                assert info is not None, f"GEMM C buffer {self.C.name} must carry an Ascend fractal layout"

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
            return self._lower_l1()
        return self._lower_l0()

    def _is_l1_input(self):
        return self.A.scope().name == "shared.l1" and self.B.scope().name == "shared.l1"

    def _is_l0_input(self):
        return self.A.scope().name == "shared.l0a" and self.B.scope().name == "shared.l0b"

    def _l0_operation_extents(self):
        """Return the logical MAD M/N/K carried by the three L0 regions."""
        analyzer = arith.Analyzer()

        def prove_equal(lhs, rhs) -> bool:
            return analyzer.can_prove_equal(lhs, rhs)

        def require_zero_origin(name, region) -> None:
            for axis, rng in enumerate(region.region[-2:], start=-2):
                assert prove_equal(rng.begin, tirx.const(0, rng.begin.ty)), (
                    f"Ascend compact L0 region {name} requires zero origin on "
                    f"trailing axis {axis}, got min={rng.begin}. Split/non-zero-"
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
        c_ptr = _make_access_ptr(self.CRegion.source, self.CRegion, 2)
        a_ptr = _make_access_ptr(self.ARegion.source, self.ARegion, 1)
        b_ptr = _make_access_ptr(self.BRegion.source, self.BRegion, 1)
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

    def _lower_l1(self):
        tile_k_sub = self._compute_tile_k_sub()
        c_ptr = _make_access_ptr(self.CRegion.source, self.CRegion, 2)

        a_buf = self.ARegion.source
        b_buf = self.BRegion.source
        a_ptr = _make_access_ptr(a_buf, self.ARegion, 1, exclude_last=True)
        b_ptr = _make_access_ptr(b_buf, self.BRegion, 1, exclude_last=True)

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
