import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


def _nodes(root, kind):
    nodes = []
    stmt_functor.post_order_visit(root, lambda node: nodes.append(node) if isinstance(node, kind) else None)
    return nodes


def _apply(tensors, body, scope="SIMD_VF"):
    region = ir.RegionStmt("tl.scope", [], [], {"name_hint": scope}, body)
    function = tirx.Function([], [*[tensor.allocate() for tensor in tensors], region], ret_type=ir.TupleType([]))
    assert tirx.analysis.check_well_formed(function)
    module = ir.IRModule({"main": function})
    result = tvm_ffi.get_global_func("tl.transform.AscendSimdVFLowerParallel")()(module)
    assert tirx.analysis.check_well_formed(result)
    restored = ir.load_json(ir.save_json(result))
    ir.assert_structural_equal(result, restored)
    return result


def _calls(module, name):
    return [call for call in _nodes(module, ir.Call) if call.op.name == name]


def _region(tensor, ranges):
    return ir.TensorRegion(tensor.tensor, ranges, ir.TensorRegionType())


@pytest.mark.parametrize("begin", [0, 8])
@pytest.mark.parametrize("dtype,extent", [("float32", 64), ("float16", 128)])
def test_parallel_preserves_nonzero_begin_and_typed_pointers(begin, dtype, extent):
    source = TensorView.create("source", [begin + extent], dtype, scope="shared.dyn")
    destination = TensorView.create("destination", [begin + extent], dtype, scope="shared.dyn")
    index = ir.Var("index", "int32")
    loop = ir.For(index, begin, begin + extent, 1, ir.ForKind.PARALLEL, destination.store(index, source.load(index) + ir.const(2, dtype)))
    result = _apply([source, destination], loop)
    assert len(_calls(result, "tl.simd.vadd")) == 1
    assert len(_calls(result, "tl.simd.vld")) == 1
    assert all(loop.kind == ir.ForKind.DEFAULT for loop in _nodes(result, ir.For))
    pointer = _calls(result, "tl.access_ptr")[0]
    assert isinstance(pointer.ty, ir.PtrType)
    chunk = _nodes(result, ir.For)[0].var
    offset = stmt_functor.substitute(pointer.args[0].indices[0], {chunk: ir.const(0)})
    assert int(sym.Analyzer().simplify(offset)) == begin


@pytest.mark.parametrize("transposed", [False, True])
def test_nested_parallel_selects_physical_contiguous_axis(transposed):
    strides = [1, 64] if transposed else None
    shape = [64, 2] if transposed else [2, 64]
    source = TensorView.create("source", shape, "float32", scope="shared.dyn", strides=strides)
    destination = TensorView.create("destination", shape, "float32", scope="shared.dyn", strides=strides)
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    store = destination.store([row, column], source.load([row, column]) * ir.const(2.0))
    loop = ir.For(row, 0, shape[0], 1, ir.ForKind.PARALLEL, ir.For(column, 0, shape[1], 1, ir.ForKind.PARALLEL, store))
    result = _apply([source, destination], loop)
    assert len(_calls(result, "tl.simd.vmul")) == 1
    assert len(_calls(result, "tl.simd.vsts")) == 1
    assert len(_nodes(result, ir.For)) == 2


@pytest.mark.parametrize("begin", [0, 8, 64])
def test_fill_keeps_region_offset_and_length(begin):
    destination = TensorView.create("destination", [128], "float32", scope="shared.dyn")
    fill = ir.Call("tl.tileop.fill", [_region(destination, [ir.Range.from_min_extent(begin, 64)]), ir.const(3.0)])
    result = _apply([destination], ir.Evaluate(fill))
    assert len(_calls(result, "tl.simd.vdup")) == 1
    assert len(_calls(result, "tl.simd.vsts")) == 1
    assert int(_nodes(result, ir.For)[0].end) == 1


@pytest.mark.parametrize("begin", [1, 4])
def test_fill_rejects_misaligned_region(begin):
    destination = TensorView.create("destination", [128], "float32", scope="shared.dyn")
    fill = ir.Call("tl.tileop.fill", [_region(destination, [ir.Range.from_min_extent(begin, 64)]), ir.const(3.0)])
    with pytest.raises(ValueError, match="32-byte-aligned"):
        _apply([destination], ir.Evaluate(fill))


def test_parallel_rejects_noncontiguous_vector_load():
    source = TensorView.create("source", [128], "float32", scope="shared.dyn")
    destination = TensorView.create("destination", [64], "float32", scope="shared.dyn")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 64, 1, ir.ForKind.PARALLEL, destination.store(index, source.load(index * 2)))
    with pytest.raises(ValueError, match="contiguous or broadcast"):
        _apply([source, destination], loop)


def test_simt_is_not_lowered_as_simd():
    destination = TensorView.create("destination", [64], "float32", scope="shared.dyn")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 64, 1, ir.ForKind.PARALLEL, destination.store(index, ir.const(1.0)))
    result = _apply([destination], loop, scope="SIMT_VF")
    assert _nodes(result, ir.For)[0].kind == ir.ForKind.PARALLEL
    assert not _calls(result, "tl.simd.vsts")
