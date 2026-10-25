"""InsertNd2Nz recognizes native physical layouts before scheduling."""

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_nz_layout


def _nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _calls(root, name):
    return [call for call in _nodes(root, ir.Call) if call.op.name == name]


def _apply(module):
    assert tirx.analysis.check_well_formed(module)
    result = tvm_ffi.get_global_func("tl.transform.InsertNd2Nz")()(module)
    assert tirx.analysis.check_well_formed(result)
    restored = ir.load_json(ir.save_json(result))
    ir.assert_structural_equal(result, restored)
    assert tirx.analysis.check_well_formed(restored)
    return result


def _module(*, rows=16, cols=128, padding=1, strides=None, staged=False, reshape=False, vf=None):
    source = TensorView.create("source", (2, rows, cols) if staged else (rows, cols), "bfloat16", scope="shared.dyn", strides=strides)
    owner = TensorView.create("storage", (rows // 2, cols * 2), "bfloat16", scope="shared.dyn") if reshape else source
    destination = TensorView.create(
        "destination", (2, rows + padding, cols) if staged else (rows + padding, cols), "float32", scope="shared.dyn"
    )
    stage = ir.Var("stage", "int32")

    def region(tensor):
        ranges = [ir.Range.from_min_extent(stage, 1)] if staged else []
        return ir.TensorRegion(tensor, ranges + [ir.Range(0, rows), ir.Range(0, cols)], ir.TensorRegionType())

    body = ir.Evaluate(ir.Call("tl.tileop.ascend_copy", [region(source.tensor), region(destination.tensor)], ty=ir.TupleType([])))
    if vf:
        body = ir.RegionStmt("tl.scope", [], [], {"name_hint": vf}, body)
    if staged:
        body = ir.For(stage, 0, 2, 1, ir.ForKind.DEFAULT, body)
    allocations = [owner.allocate(), destination.allocate()]
    if reshape:
        allocations.append(source.declare(owner.data))
    root = ir.RegionStmt(
        "tirx.scope", [], [], {"layout_map": {destination.tensor: make_ascend_compact_nz_layout(destination.tensor)}}, [*allocations, body]
    )
    return ir.IRModule({"main": tirx.Function([], root, ret_type=ir.TupleType([]))})


@pytest.mark.parametrize("case", ["plain", "staged", "reshape", "simd", "simt"])
def test_scatter_regions_and_vf_boundary(case):
    before = _module(staged=case == "staged", reshape=case == "reshape", vf={"simd": "SIMD_VF", "simt": "SIMT_VF"}.get(case))
    after = _apply(before)
    if case == "simt":
        ir.assert_structural_equal(after, before)
        return
    (scatter,) = _calls(after, "tl.ascend_nd2nz_scatter")
    assert [int(value) for value in scatter.args[2:4]] == [16, 128]
    assert [str(value.value) for value in scatter.args[4:]] == ["float", "bfloat16_t"]
    assert not _calls(after, "tl.tileop.ascend_copy")
    analyzer = sym.Analyzer()
    assert int(analyzer.simplify(scatter.args[0].args[1])) == 16 * 128
    assert int(analyzer.simplify(scatter.args[1].args[1])) == 17 * 128
    if case == "staged":
        source_index = scatter.args[0].args[0].indices[0]
        destination_index = scatter.args[1].args[0].indices[0]
        assert source_index.same_as(destination_index)


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
            _apply(before)
    else:
        assert len(_calls(_apply(before), "tl.ascend_nd2nz_scatter")) == 1


def test_ub_to_l1_scratch_is_defined_at_the_copy_not_hoisted_out_of_its_loop():
    source = TensorView.create("source", [64, 64], "bfloat16", scope="shared.dyn")
    destination = TensorView.create("destination", [16, 64], "bfloat16", scope="shared.l1")
    index = ir.Var("index", "int32")
    source_region = ir.TensorRegion(source.tensor, [ir.Range.from_min_extent(index * 16, 16), ir.Range(0, 64)], ir.TensorRegionType())
    destination_region = ir.TensorRegion(destination.tensor, [ir.Range(0, 16), ir.Range(0, 64)], ir.TensorRegionType())
    copy = ir.Evaluate(ir.Call("tl.tileop.ascend_copy", [source_region, destination_region], ty=ir.TupleType([])))
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, copy)
    body = ir.RegionStmt(
        "tirx.scope",
        [],
        [],
        {"layout_map": {destination.tensor: make_ascend_nz_layout(destination.tensor)}},
        [source.allocate(), destination.allocate(), loop],
    )
    result = _apply(ir.IRModule({"main": tirx.Function([], body, ret_type=ir.TupleType([]))}))
    (rewritten_loop,) = _nodes(result, ir.For)
    scratch = [binding for binding in _nodes(rewritten_loop, ir.Bind) if binding.var.name == "nz_scratch"]
    assert len(scratch) == 1
    assert int(scratch[0].var.ty.shape[0]) == 17 * 64
    assert len(_calls(rewritten_loop, "tl.ascend_nd2nz_scatter")) == 1
    assert len(_calls(rewritten_loop, "tl.ascend_nd2nz_post_copy")) == 1
