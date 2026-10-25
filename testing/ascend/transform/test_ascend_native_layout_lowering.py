import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.target import Target

import tilelang.ascend.language as T
from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView
from tilelang.layout import make_ascend_l0c_layout, make_ascend_major_k_layout, make_ascend_sf_layout


def _nodes(root, kind):
    nodes = []
    stmt_functor.post_order_visit(root, lambda node: nodes.append(node) if isinstance(node, kind) else None)
    return nodes


def _calls(root, name):
    return [call for call in _nodes(root, ir.Call) if call.op.name == name]


def _scope(name, body, **attrs):
    return ir.RegionStmt("tl.scope", [], [], {"name_hint": name, **attrs}, body)


def _module(body, params=(), attrs=None):
    function = tirx.Function(list(params), body, ret_type=ir.TupleType([]))
    function = function.with_attr("target", Target("ascend")).with_attr("global_symbol", "main")
    for key, value in (attrs or {}).items():
        function = function.with_attr(key, value)
    assert tirx.analysis.check_well_formed(function)
    return ir.IRModule({"main": function})


def _pass(module, name):
    with Target("ascend"):
        result = tvm_ffi.get_global_func("tl.transform." + name)()(module)
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def _hint(tensor, layout):
    return ir.Evaluate(ir.Call("tl.layout_hint", [tensor.tensor], attrs={"layout": layout}))


def test_simt_layouts_use_each_lexical_thread_scope():
    regions = []
    for threads in [32, 64]:
        fragment = TensorView.create(f"fragment_{threads}", [128], "float32", scope="local.fragment")
        index = ir.Var(f"index_{threads}", "int32")
        thread = ir.Var(f"thread_{threads}", "int32")
        loop = ir.For(index, 0, 128, 1, ir.ForKind.PARALLEL, fragment.store(index, ir.const(1.0)))
        launch = ir.RegionStmt("tirx.launch_thread", ["threadIdx.x", threads], [thread], {}, [fragment.allocate(), loop])
        regions.append(_scope("SIMT_VF", launch))
    result = _pass(_module(regions), "AscendLayoutInference")
    layouts = [loop.annotations["parallel_loop_layout"] for loop in _nodes(result, ir.For)]
    assert [int(layout.get_thread_size()) for layout in layouts] == [32, 64]
    lowered = _pass(result, "AscendLowerTileOp")
    assert not any(call.op.name.startswith("tl.tileop.") for call in _nodes(lowered, ir.Call))
    assert not any(loop.kind == ir.ForKind.PARALLEL for loop in _nodes(lowered, ir.For))


def test_simd_layout_inference_keeps_parallel_nest_opaque():
    tensor = TensorView.create("tensor", [2, 64], "float32", scope="shared.dyn")
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    inner = ir.For(column, 0, 64, 1, ir.ForKind.PARALLEL, tensor.store([row, column], ir.const(1.0)))
    outer = ir.For(row, 0, 2, 1, ir.ForKind.PARALLEL, inner)
    before = _module([tensor.allocate(), _scope("SIMD_VF", outer)])
    inferred = _pass(before, "AscendLayoutInference")
    loops = _nodes(inferred, ir.For)
    assert len(loops) == 2
    assert all("parallel_loop_layout" not in loop.annotations for loop in loops)
    lowered = _pass(_pass(inferred, "AscendSimdVFLowerParallel"), "AscendLowerTileOp")
    assert len(_calls(lowered, "tl.simd.vsts")) == 1


@pytest.mark.parametrize("size", [15, 16])
def test_copy_gemm_chain_carries_inferred_fractal_layouts(size):
    source_a = TensorView.create("source_a", [size, 16], "bfloat16")
    source_b = TensorView.create("source_b", [size, 16], "bfloat16")
    output = TensorView.create("output", [size, size], "float32")
    left = TensorView.create("left", [size, 16], "bfloat16", scope="shared.l1")
    right = TensorView.create("right", [size, 16], "bfloat16", scope="shared.l1")
    accum = TensorView.create("accum", [size, size], "float32", scope="shared.l0c")
    body = [tensor.allocate() for tensor in [left, right, accum]]
    body.append(_hint(accum, make_ascend_l0c_layout(accum.tensor)))
    body += [
        ir.Evaluate(T.copy(source_a.tensor, left.tensor)),
        ir.Evaluate(T.copy(source_b.tensor, right.tensor)),
        ir.Evaluate(T.gemm(left.tensor, right.tensor, accum.tensor, transpose_B=True, clear_accum=True)),
        ir.Evaluate(T.copy(accum.tensor, output.tensor)),
    ]
    module = _module(_scope("tilelang_root", body), [source_a.tensor, source_b.tensor, output.tensor])
    inferred = _pass(module, "AscendLayoutInference")
    assert len(_calls(inferred, "tl.layout_hint")) >= 3
    normalized = _pass(inferred, "NormalizeAscendFractalStorage")
    padded = [binding.var for binding in _nodes(normalized, ir.Bind) if tirx.is_tensor_var(binding.var)]
    assert all([int(extent) for extent in tensor.ty.shape] == [16, 16] for tensor in padded)
    for name in ["InsertNd2Nz", "AscendInsertOOBPadding", "AscendLowerTileOp", "LowerAccessPtr"]:
        normalized = _pass(normalized, name)
    assert not any(call.op.name.startswith("tl.tileop.") or call.op.name == "tl.layout_hint" for call in _nodes(normalized, ir.Call))
    assert _calls(normalized, "tl.ascend_gemm_l1")


@pytest.mark.parametrize("leading", [False, True])
def test_l0_pitch_keeps_leading_plane_stride(leading):
    shape = [2, 3, 4, 16, 16] if leading else [3, 4, 16, 16]
    indices = [1, 2, 1, 0, 0] if leading else [2, 1, 0, 0]
    tensor = TensorView.create("tensor", shape, "bfloat16", scope="shared.l0a")
    pointer = ir.Call("tl.access_ptr", [tensor.load(indices), 256, 2], attrs={"tl.ascend_l0_access_pitch": ir.const(6)})
    module = _module([tensor.allocate(), ir.Evaluate(pointer)])
    result = _pass(module, "AscendLowerTileOp")
    (offset,) = _calls(result, "tirx.ptr_byte_offset")
    assert int(sym.Analyzer().simplify(offset.args[1])) == 2048
    assert "tl.ascend_l0_access_pitch" not in offset.args[0].attrs
    _pass(result, "LowerAccessPtr")


@pytest.mark.parametrize("bound", [False, True])
def test_mx_scale_load_uses_data_allocation_without_scale_storage(bound):
    source = TensorView.create("source", [16, 2], "uint16", scope="shared.l1")
    data = TensorView.create("data", [16, 128], "float8_e4m3fn", scope="shared.l0a")
    scale = TensorView.create("scale", [16, 2], "uint16", scope="shared.l0a.sf")
    body = [tensor.allocate() for tensor in [source, data, scale]]
    body += [_hint(source, make_ascend_sf_layout(source.tensor)), _hint(data, make_ascend_major_k_layout(data.tensor))]
    body.append(ir.Evaluate(T.copy(source.tensor, scale.tensor)))
    attrs = {"tl.l0_sf_bindings": {scale.tensor: data.tensor}} if bound else {}
    module = _module(_scope("tilelang_root", body, **attrs))
    if not bound:
        with pytest.raises(ValueError, match="no allocation binding"):
            _pass(module, "AscendLowerTileOp")
        return
    lowered = _pass(module, "AscendLowerTileOp")
    assert not any(binding.var.name == "scale" for binding in _nodes(lowered, ir.Bind))
    assert all("tl.l0_sf_bindings" not in region.attrs for region in _nodes(lowered, ir.RegionStmt))
    assert _calls(lowered, "tl.ascend_load_ca_sf")
