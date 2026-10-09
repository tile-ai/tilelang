"""InsertNd2Nz recognizes physical layouts before scheduling."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_compact_nz_layout
from tvm import tirx
from testing.ascend._ir import calls


def _module(*, rows=16, cols=128, padding=1, strides=None, staged=False, reshape=False, vf=None):
    src = tirx.decl_buffer((2, rows, cols) if staged else (rows, cols), "bfloat16", name="src", scope="shared.dyn", strides=strides)
    owner = tirx.decl_buffer((rows // 2, cols * 2), "bfloat16", name="storage", scope="shared.dyn", data=src.data) if reshape else src
    dst = tirx.decl_buffer((2, rows + padding, cols) if staged else (rows + padding, cols), "float32", name="dst", scope="shared.dyn")
    i = tirx.Var("stage", "int32")

    def region(buffer):
        ranges = [tvm.ir.Range.from_min_extent(i, 1)] if staged else []
        return tirx.BufferRegion(buffer, ranges + [tvm.ir.Range(0, rows), tvm.ir.Range(0, cols)])

    body = tirx.Evaluate(T.copy(region(src), region(dst)))
    if vf:
        body = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], vf, body))
    if staged:
        body = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, body)
    root = tirx.SBlock(
        [], [], [], "root", body, alloc_buffers=[owner, dst], annotations={"layout_map": {dst: make_ascend_compact_nz_layout(dst)}}
    )
    return tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, root))})


@pytest.mark.parametrize("case", ["plain", "staged", "reshape", "simd", "simt"])
def test_scatter_regions_and_vf_boundary(case):
    before = _module(staged=case == "staged", reshape=case == "reshape", vf={"simd": "SIMD_VF", "simt": "SIMT_VF"}.get(case))
    after = transform.InsertNd2Nz()(before)
    if case == "simt":
        tvm.ir.assert_structural_equal(after, before)
        return
    (scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
    assert [int(x) for x in scatter.args[2:4]] == [16, 128]
    assert [str(x.value) for x in scatter.args[4:]] == ["float", "bfloat16_t"]
    assert not calls(after, "tl.tileop.ascend_copy")
    # The destination footprint includes the bank-conflict padding row.
    analyzer = tvm.arith.Analyzer()
    assert int(analyzer.simplify(scatter.args[0].args[1])) == 16 * 128
    assert int(analyzer.simplify(scatter.args[1].args[1])) == 17 * 128
    if case == "staged":
        src_index = scatter.args[0].args[0].indices[0]
        dst_index = scatter.args[1].args[0].indices[0]
        assert src_index.same_as(dst_index)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"padding": 0}, "reserve exactly one padding row"),
        ({"strides": (256, 1)}, "compact trailing source matrix"),
        ({"staged": True, "strides": (2047, 128, 1)}, "32-byte-aligned source address"),
        ({"staged": True, "strides": (2048, 128, 1)}, None),
    ],
)
def test_scatter_storage_requirements(kwargs, error):
    before = _module(**kwargs)
    if error:
        with pytest.raises(ValueError, match=error):
            transform.InsertNd2Nz()(before)
    else:
        assert len(calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_scatter")) == 1
