import pytest
import tvm_ffi
from tvm_tirx import ir, tirx, transform
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def launch(body, threads=32):
    thread = ir.Var("thread", "int32")
    return ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(threads)],
        [thread],
        None,
        ir.SeqStmt(body),
    )


def infer(body, parameters=(), cost="register-count"):
    target = Target({"kind": "cuda", "arch": "sm_90"})
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([])).with_attr("target", target)
    assert tirx.analysis.check_well_formed(function)
    module = ir.IRModule({"main": function})
    with target, transform.PassContext(config={"tl.layout_cost_model": cost}):
        result = tvm_ffi.get_global_func("tl.transform.LayoutInference")()(module)
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result["main"]


def nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def hint(tensor, layout):
    return ir.Evaluate(ir.Call("tl.layout_hint", [tensor.tensor], attrs={"layout": layout}))


def layouts(root):
    return {call.args[0]: call.attrs["layout"] for call in nodes(root, ir.Call) if call.op.name == "tl.layout_hint"}


@pytest.mark.parametrize("cost", ["register-count", "io-aware"])
def test_explicit_fragment_hint_preserves_ownership_and_binding(native_layouts, cost):
    tensor = TensorView.create("fragment", [128], "float32", scope="local.fragment")
    layout = native_layouts.Fragment(
        [128],
        forward_thread_fn=lambda coordinate: coordinate % 32,
        forward_index_fn=lambda coordinate: coordinate // 32,
    )
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 128, 1, ir.ForKind.PARALLEL, tensor.store(index, 1.0))
    result = infer(launch([tensor.allocate(), hint(tensor, layout), loop]), cost=cost)
    ir.assert_structural_equal(layouts(result)[tensor.tensor], layout)
    assert len(layouts(result)) == 1
    inferred = nodes(result, ir.For)[0].annotations["parallel_loop_layout"]
    for coordinate in range(128):
        assert int(inferred.map_forward_thread([coordinate])[0]) == coordinate % 32
    restored = ir.load_json(ir.save_json(result))
    allocated = nodes(restored, ir.Bind)[0].var
    assert allocated in layouts(restored)


@pytest.mark.parametrize("cost", ["register-count", "io-aware"])
@pytest.mark.parametrize("threads", [32, 64, 128])
def test_free_inference_copies_parallel_fragment(cost, threads):
    source = TensorView.create("source", [256], "float32")
    fragment = TensorView.create("fragment", [256], "float32", scope="local.fragment")
    output = TensorView.create("output", [256], "float32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 256, 1, ir.ForKind.PARALLEL, fragment.store(index, source.load(index)))
    copy = ir.Evaluate(ir.Call("tl.tileop.copy", [fragment.tensor, output.tensor]))
    result = infer(launch([fragment.allocate(), loop, copy], threads), [source.tensor, output.tensor], cost)
    layout = layouts(result)[fragment.tensor]
    assert list(map(int, layout.get_input_shape())) == [256]
    owners = {int(layout.map_forward_thread([coordinate])[0]) for coordinate in range(256)}
    assert owners == set(range(threads))
    assert "parallel_loop_layout" in nodes(result, ir.For)[0].annotations


def test_annotation_propagates_to_zero_offset_tensor_alias(native_layouts):
    tensor = TensorView.create("shared", [4, 8], "float32", scope="shared")
    alias = TensorView.create("alias", [32], "float32", scope="shared")
    layout = native_layouts.Layout([4, 8], lambda row, column: (column, row))
    result = infer(launch([tensor.allocate(), alias.declare(tensor.data), hint(tensor, layout)]))
    assert tensor.tensor in layouts(result)
    assert alias.tensor in layouts(result)
    inferred = layouts(result)[alias.tensor]
    for coordinate in range(32):
        assert list(map(int, inferred(coordinate))) == [coordinate % 8, coordinate // 8]


def test_inference_restores_thread_scope_between_kernels():
    fragments = [TensorView.create(f"fragment_{threads}", [256], "float32", scope="local.fragment") for threads in (32, 64)]
    kernels = []
    for fragment, threads in zip(fragments, (32, 64)):
        index = ir.Var("index", "int32")
        loop = ir.For(index, 0, 256, 1, ir.ForKind.PARALLEL, fragment.store(index, 1.0))
        kernels.append(launch([fragment.allocate(), loop], threads))
    result = infer(ir.SeqStmt(kernels))
    for fragment, threads in zip(fragments, (32, 64)):
        layout = layouts(result)[fragment.tensor]
        assert {int(layout.map_forward_thread([coordinate])[0]) for coordinate in range(256)} == set(range(threads))


def test_conflicting_hints_are_rejected(native_layouts):
    tensor = TensorView.create("shared", [4, 8], "float32", scope="shared")
    first = native_layouts.Layout([4, 8], lambda row, column: (row, column))
    second = native_layouts.Layout([4, 8], lambda row, column: (column, row))
    with pytest.raises(ValueError, match="Conflicting layout annotations"):
        infer(launch([tensor.allocate(), hint(tensor, first), hint(tensor, second)]))


def test_native_geometry_cannot_be_silently_overwritten(native_layouts):
    tensor = TensorView.create("shared", [4, 8], "float32", scope="shared", strides=[16, 1])
    layout = native_layouts.Layout([4, 8], lambda row, column: (row, column))
    with pytest.raises(ValueError, match="both native and TileLang layouts"):
        infer(launch([tensor.allocate(), hint(tensor, layout)]))
