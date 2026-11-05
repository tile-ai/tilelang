"""L0 MAD uses logical operand regions while retaining physical allocation sizes."""

from tvm_tirx import ir
from testing.ascend._ir import sblock, tensor

import math

import pytest
import tvm_tirx
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_major_k_layout, make_ascend_l0c_layout
from tvm_tirx import tirx
from testing.ascend._ir import calls, allocated_buffer, kernel
from testing.ascend._memory_ir import element_offset


def _lower(m, n, k, *, blockscaled=False):
    dtype = "float8_e4m3fn" if blockscaled else "float32"
    a = tensor((64, 128), dtype, name="a", scope="shared.l0a")
    b = tensor((64, 128), dtype, name="b", scope="shared.l0b")
    c = tensor((64, 64), "float32", name="c", scope="shared.l0c")

    def region(buffer, rows, cols):
        return ir.TensorRegion(buffer, [tvm_tirx.ir.Range(0, rows), tvm_tirx.ir.Range(0, cols)], ir.TensorRegionType())

    bindings = {}
    alloc_buffers = [a, b, c]
    if blockscaled:
        sfa = tensor((64, 2), "int16", name="sfa", scope="shared.l0a.sf")
        sfb = tensor((64, 2), "int16", name="sfb", scope="shared.l0b.sf")
        alloc_buffers += [sfa, sfb]
        bindings = {sfa: a, sfb: b}
        call = T.gemm_blockscaled(
            region(a, m, k), region(b, n, k), region(c, m, n), region(sfa, m, 2), region(sfb, n, 2), transpose_B=True, clear_accum=True
        )
    else:
        call = T.gemm(region(a, m, k), region(b, n, k), region(c, m, n), transpose_B=True, clear_accum=True)
    body = ir.Evaluate(call)
    root = sblock(
        "root",
        body,
        buffers=alloc_buffers,
        annotations={
            "tl.l0_sf_bindings": bindings,
            "layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)},
        },
    )
    target = tvm_tirx.target.Target("ascend")
    body = root
    before = tirx.Function(tirx.analysis.undefined_vars(body), body).with_attr("target", target)
    with target:
        return transform.AscendLowerTileOp()(tvm_tirx.IRModule({"main": before}))


@pytest.mark.parametrize("blockscaled", [False, True], ids=["mad", "blockscaled-mad"])
@pytest.mark.parametrize("symbolic", [False, True], ids=["static-region", "dynamic-region"])
def test_mad_uses_region_geometry_without_shrinking_storage(blockscaled, symbolic):
    m, n = (ir.Var("m", "int32"), ir.Var("n", "int32")) if symbolic else (17, 19)
    k = 128 if blockscaled else 24
    after = _lower(m, n, k, blockscaled=blockscaled)
    (mad,) = calls(after, "tl.ascend_mad_mx" if blockscaled else "tl.ascend_mad")
    analyzer = tvm_tirx.sym.Analyzer()
    for actual, expected in zip(mad.args[3:6], [m, k, n]):
        assert analyzer.can_prove_equal(actual, expected)
    for name in ("a", "b"):
        assert math.prod(int(x) for x in allocated_buffer(after, name).ty.shape) == 64 * 128


def _sf_gemm(case):
    a = tensor((2, 3, 32, 128), "float8_e4m3fn", name="a", scope="shared.l0a")
    a2 = tensor(a.ty.shape, a.ty.dtype, name="a2", scope="shared.l0a")
    b = tensor((32, 128), a.ty.dtype, name="b", scope="shared.l0b")
    c = tensor((32, 32), "float32", name="c", scope="shared.l0c")
    sf = tensor((2, 3, 32, 2), "uint16", name="sf", scope="shared.l0b.sf" if case == "scope" else "shared.l0a.sf")
    sb = tensor((32, 2), "uint16", name="sb", scope="shared.l0b.sf")
    bindings = {sf: a2 if case == "wrong-data" else a, sb: b}
    if case == "missing":
        del bindings[sf]
    if case == "duplicate":
        bindings[tensor(sf.ty.shape, sf.ty.dtype, scope="shared.l0a.sf")] = a

    def region(buf, mins, extents):
        return ir.TensorRegion(buf, [tvm_tirx.ir.Range.from_min_extent(m, e) for m, e in zip(mins, extents)], ir.TensorRegionType())

    ar = region(a, [1, 2, 0, 0], [1, 1, 17, 64 if case == "compact" else 128])
    sr = region(sf, [0 if case == "leading" else 1, 2, 1 if case == "origin" else 0, 0], [1, 1, 17, 1 if case in ("k", "compact") else 2])
    br = region(b, [0, 0], [32, 64 if case == "compact" else 128])
    cr = region(c, [0, 0], [17, 32])
    sbr = region(sb, [0, 0], [32, 1 if case == "compact" else 2])
    call = T.gemm_blockscaled(ar, br, cr, sr, sbr, transpose_B=True, clear_accum=True)
    return kernel(
        ir.Evaluate(call),
        buffers=[a, a2, b, c, sf, sb],
        annotations={
            "tl.l0_sf_bindings": bindings,
            "layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)},
        },
    )


@pytest.mark.parametrize(
    "case,error_type,diagnostic",
    [
        ("wrong-data", RuntimeError, "is bound to"),
        # Native ownership validation rejects malformed raw-IR metadata before
        # operator lowering. Its ValueError is distinct from the later checks.
        ("missing", ValueError, "no allocation binding"),
        ("scope", ValueError, "matching L0A/L0B data scope"),
        ("leading", RuntimeError, "same leading index"),
        ("origin", RuntimeError, "zero-origin"),
        ("k", RuntimeError, "compact data K extent"),
        ("duplicate", ValueError, "multiple SF handles"),
    ],
)
def test_invalid_binding_and_slice_are_rejected(case, error_type, diagnostic):
    with pytest.raises(error_type, match=diagnostic):
        transform.AscendLowerTileOp()(_sf_gemm(case))


@pytest.mark.parametrize("case", ["valid", "compact"])
def test_matching_leading_indices_and_compact_regions(case):
    after = transform.AscendLowerTileOp()(_sf_gemm(case))
    (mad,) = calls(after, "tl.ascend_mad_mx")
    analyzer = tvm_tirx.sym.Analyzer()
    assert analyzer.can_prove_equal(element_offset(after, mad.args[1]), 5 * 32 * 128)
    assert analyzer.can_prove_equal(mad.args[4], 64 if case == "compact" else 128)


def test_mismatched_sf_binding_fails_before_layout_inference():
    with pytest.raises(RuntimeError, match="is bound to"):
        transform.AscendLayoutInference()(_sf_gemm("wrong-data"))
