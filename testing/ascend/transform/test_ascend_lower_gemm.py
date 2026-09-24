"""L0 MAD uses logical operand regions while retaining physical allocation sizes."""

import math

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_major_k_layout, make_ascend_l0c_layout
from tvm import tirx
from testing.ascend._ir import calls, allocated_buffer


def _lower(m, n, k, *, blockscaled=False):
    dtype = "float8_e4m3fn" if blockscaled else "float32"
    a = tirx.decl_buffer((64, 128), dtype, name="a", scope="shared.l0a")
    b = tirx.decl_buffer((64, 128), dtype, name="b", scope="shared.l0b")
    c = tirx.decl_buffer((64, 64), "float32", name="c", scope="shared.l0c")

    def region(buffer, rows, cols):
        return tirx.BufferRegion(buffer, [tvm.ir.Range(0, rows), tvm.ir.Range(0, cols)])

    alloc_buffers = [a, b, c]
    if blockscaled:
        sfa = tirx.decl_buffer((64, 2), "int16", name="sfa", scope="shared.l0a.sf")
        sfb = tirx.decl_buffer((64, 2), "int16", name="sfb", scope="shared.l0b.sf")
        alloc_buffers += [sfa, sfb]
        call = T.gemm_blockscaled(region(a, m, k), region(b, n, k), region(c, m, n), sfa, sfb, transpose_B=True, clear_accum=True)
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
        annotations={"layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)}},
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
