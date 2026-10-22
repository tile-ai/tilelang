"""L0 MAD uses logical operand regions while retaining physical allocation sizes."""

import math

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_major_k_layout, make_ascend_l0c_layout
from tvm import tirx
from testing.ascend._ir import calls, allocated_buffer, kernel


def _lower(m, n, k, *, blockscaled=False):
    dtype = "float8_e4m3fn" if blockscaled else "float32"
    a = tirx.decl_buffer((64, 128), dtype, name="a", scope="shared.l0a")
    b = tirx.decl_buffer((64, 128), dtype, name="b", scope="shared.l0b")
    c = tirx.decl_buffer((64, 64), "float32", name="c", scope="shared.l0c")

    def region(buffer, rows, cols):
        return tirx.BufferRegion(buffer, [tvm.ir.Range(0, rows), tvm.ir.Range(0, cols)])

    bindings = {}
    alloc_buffers = [a, b, c]
    if blockscaled:
        sfa = tirx.decl_buffer((64, 2), "int16", name="sfa", scope="shared.l0a.sf")
        sfb = tirx.decl_buffer((64, 2), "int16", name="sfb", scope="shared.l0b.sf")
        alloc_buffers += [sfa, sfb]
        bindings = {sfa.data: a.data, sfb.data: b.data}
        call = T.gemm_blockscaled(
            region(a, m, k), region(b, n, k), region(c, m, n), region(sfa, m, 2), region(sfb, n, 2), transpose_B=True, clear_accum=True
        )
    else:
        call = T.gemm(region(a, m, k), region(b, n, k), region(c, m, n), transpose_B=True, clear_accum=True)
    body = tirx.Evaluate(call)
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=alloc_buffers,
        annotations={
            "tl.l0_sf_bindings": bindings,
            "layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)},
        },
    )
    target = tvm.target.Target("ascend")
    body = tirx.SBlockRealize([], True, root)
    before = tirx.PrimFunc(tirx.analysis.undefined_vars(body), body).with_attr("target", target)
    with target:
        return transform.AscendLowerTileOp()(tvm.IRModule({"main": before}))


@pytest.mark.parametrize("blockscaled", [False, True], ids=["mad", "blockscaled-mad"])
@pytest.mark.parametrize("symbolic", [False, True], ids=["static-region", "dynamic-region"])
def test_mad_uses_region_geometry_without_shrinking_storage(blockscaled, symbolic):
    m, n = (tirx.Var("m", "int32"), tirx.Var("n", "int32")) if symbolic else (17, 19)
    k = 128 if blockscaled else 24
    after = _lower(m, n, k, blockscaled=blockscaled)
    (mad,) = calls(after, "tl.ascend_mad_mx" if blockscaled else "tl.ascend_mad")
    analyzer = tvm.arith.Analyzer()
    for actual, expected in zip(mad.args[3:6], [m, k, n]):
        assert analyzer.can_prove_equal(actual, expected)
    for name in ("a", "b"):
        assert math.prod(int(x) for x in allocated_buffer(after, name).shape) == 64 * 128


def _sf_gemm(case):
    a = tirx.decl_buffer((2, 3, 32, 128), "float8_e4m3fn", name="a", scope="shared.l0a")
    a2 = tirx.decl_buffer(a.shape, a.dtype, name="a2", scope="shared.l0a")
    b = tirx.decl_buffer((32, 128), a.dtype, name="b", scope="shared.l0b")
    c = tirx.decl_buffer((32, 32), "float32", name="c", scope="shared.l0c")
    sf = tirx.decl_buffer((2, 3, 32, 2), "uint16", name="sf", scope="shared.l0b.sf" if case == "scope" else "shared.l0a.sf")
    sb = tirx.decl_buffer((32, 2), "uint16", name="sb", scope="shared.l0b.sf")
    bindings = {sf.data: a2.data if case == "wrong-data" else a.data, sb.data: b.data}
    if case == "missing":
        del bindings[sf.data]
    if case == "duplicate":
        bindings[tirx.decl_buffer(sf.shape, sf.dtype, scope="shared.l0a.sf").data] = a.data

    def region(buf, mins, extents):
        return tirx.BufferRegion(buf, [tvm.ir.Range.from_min_extent(m, e) for m, e in zip(mins, extents)])

    ar = region(a, [1, 2, 0, 0], [1, 1, 17, 64 if case == "compact" else 128])
    sr = region(sf, [0 if case == "leading" else 1, 2, 1 if case == "origin" else 0, 0], [1, 1, 17, 1 if case in ("k", "compact") else 2])
    br = region(b, [0, 0], [32, 64 if case == "compact" else 128])
    cr = region(c, [0, 0], [17, 32])
    sbr = region(sb, [0, 0], [32, 1 if case == "compact" else 2])
    call = T.gemm_blockscaled(ar, br, cr, sr, sbr, transpose_B=True, clear_accum=True)
    return kernel(
        tirx.Evaluate(call),
        buffers=[a, a2, b, c, sf, sb],
        annotations={
            "tl.l0_sf_bindings": bindings,
            "layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)},
        },
    )


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("wrong-data", "is bound to"),
        ("missing", "no allocation binding"),
        ("scope", "matching L0A/L0B SF scope"),
        ("leading", "same leading index"),
        ("origin", "zero-origin"),
        ("k", "compact data K extent"),
        ("duplicate", "multiple SF handles"),
    ],
)
def test_invalid_binding_and_slice_are_rejected(case, diagnostic):
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        transform.AscendLowerTileOp()(_sf_gemm(case))


@pytest.mark.parametrize("case", ["valid", "compact"])
def test_matching_leading_indices_and_compact_regions(case):
    after = transform.AscendLowerTileOp()(_sf_gemm(case))
    (mad,) = calls(after, "tl.ascend_mad_mx")
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(mad.args[1].args[2], 5 * 32 * 128)
    assert analyzer.can_prove_equal(mad.args[4], 64 if case == "compact" else 128)


def test_mismatched_sf_binding_fails_before_layout_inference():
    with pytest.raises(tvm.error.InternalError, match="is bound to"):
        transform.AscendLayoutInference()(_sf_gemm("wrong-data"))
