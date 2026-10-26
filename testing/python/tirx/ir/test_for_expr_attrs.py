from tvm_tirx import tirx
from tilelang._compat import ir
from tilelang._compat import structural


def make_annotated_loop(captured):
    index = ir.Var("index", "int32")
    return ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        ir.Evaluate(index),
        annotations={"metadata": {"nested": [captured, index]}},
    )


def test_for_annotations_participate_in_structural_walk_and_map():
    captured = ir.Var("captured", "int32")
    replacement = ir.Var("replacement", "int32")
    loop = make_annotated_loop(captured)
    visited = []
    structural.structural_walk(loop, (ir.Var, lambda variable: visited.append(variable)))
    assert any(variable.same_as(captured) for variable in visited)
    rewritten = structural.structural_map(
        loop,
        (ir.Var, lambda variable: replacement if variable.same_as(captured) else variable),
    )
    assert loop.annotations["metadata"]["nested"][0].same_as(captured)
    assert rewritten.annotations["metadata"]["nested"][0].same_as(replacement)


def test_for_annotations_follow_renewed_definition_scopes():
    captured = ir.Var("captured", "int32")
    function = tirx.Function([], [ir.Bind(captured, ir.const(7)), make_annotated_loop(captured)])
    renewed = ir.renew_def(function)
    loop = renewed.body.seq[1]
    values = loop.annotations["metadata"]["nested"]
    assert values[0].same_as(renewed.body.seq[0].var)
    assert values[1].same_as(loop.var)
    assert not values[0].same_as(captured)
    assert not tirx.analysis.undefined_vars(renewed.body, [])
    assert ir.check_well_formed(renewed)
    ir.assert_structural_equal(function, renewed)
    ir.assert_structural_equal(renewed, ir.load_json(ir.save_json(renewed)))


def test_function_attributes_preserve_local_definition_identity():
    captured = ir.Var("captured", "int32")
    function = tirx.Function([], [ir.Bind(captured, ir.const(7))]).with_attr("metadata", {"nested": [captured]})
    visited = []
    structural.structural_walk(function, (ir.Var, lambda variable: visited.append(variable)))
    assert sum(variable.same_as(captured) for variable in visited) == 2
    renewed = ir.renew_def(function)
    assert renewed.attrs["metadata"]["nested"][0].same_as(renewed.body.seq[0].var)
    assert not renewed.body.seq[0].var.same_as(captured)
    ir.assert_structural_equal(function, renewed)
    assert ir.structural_hash(function) == ir.structural_hash(renewed)
    ir.assert_structural_equal(renewed, ir.load_json(ir.save_json(renewed)))


def test_renewed_pointer_binding_uses_the_final_inferred_variable():
    source = ir.Var("source", ir.PtrType("void"))
    pointer = ir.Var("pointer", ir.PtrType("void"))
    function = tirx.Function([source], [ir.Bind(pointer, source), ir.Evaluate(ir.Call("tirx.isnullptr", [pointer]))])
    renewed = ir.renew_def(function)
    assert renewed.body.seq[0].var.same_as(renewed.body.seq[1].value.args[0])
    assert ir.check_well_formed(renewed)
    ir.assert_structural_equal(function, renewed)
