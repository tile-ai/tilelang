import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts", "tilelang_builders")


def simplify(body, params, config=None, simplify_arguments=False):
    function = tirx.Function(params, body, ret_type=ir.TupleType([])).with_attr("global_symbol", "simplified")
    with ir.transform.PassContext(config={"tl.Simplify": config or {}}):
        result = tvm_ffi.get_global_func("tl.transform.Simplify")(simplify_arguments)(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def collect(function, node_type):
    nodes = []
    stmt_functor.post_order_visit(function, lambda node: nodes.append(node) if isinstance(node, node_type) else None)
    return nodes


def test_dead_tensor_views_and_pointer_projections_do_not_keep_inputs_live():
    source = TensorView.create("source", [8], "int32")
    pointer = ir.Var("pointer", source.tensor.data.ty)
    alias = TensorView.create("alias", [8], "int32")
    function = simplify(
        [ir.Bind(pointer, source.tensor.data), alias.declare(pointer), ir.Evaluate(ir.const(0))],
        [source.tensor],
    )
    assert not collect(function, ir.Bind)
    assert len(function.params) == 1


def test_live_tensor_views_preserve_storage_uses():
    source = TensorView.create("source", [8], "int32")
    alias = TensorView.create("alias", [8], "int32")
    function = simplify([alias.declare(source.tensor.data), alias.store(0, 7)], [source.tensor])
    assert len(collect(function, ir.Bind)) == 1
    value = runtime.tensor(np.zeros(8, dtype="int32"))
    tirx.compile(function, target="c")(value)
    np.testing.assert_array_equal(value.numpy(), [7, 0, 0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("begin,end,step", [(0, 7, 2), (7, 0, 2), (2, 9, 3), (7, 0, -2), (0, 7, -2), (3, 3, -1)])
def test_nonunit_loop_steps(begin, end, step):
    output = TensorView.create("output", [1], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(
        index,
        begin,
        end,
        step,
        ir.ForKind.DEFAULT,
        output.store(0, output.load(0) + index),
    )
    function = simplify([output.store(0, 0), loop], [output.tensor])
    actual = runtime.empty((1,), "int32")
    tirx.compile(function, target="c")(actual)
    np.testing.assert_array_equal(actual.numpy(), [sum(range(begin, end, step))])


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("inline", [False, True])
def test_mutable_load_binding_is_a_snapshot(target, inline):
    value = TensorView.create("value", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    captured = ir.Var("captured", "int32")
    function = simplify(
        [ir.Bind(captured, value.load(0)), value.store(0, 17), output.store(0, captured)],
        [value.tensor, output.tensor],
        {"enable_simplify_let_inline": inline},
    )
    assert len(collect(function, ir.Bind)) == 1
    executable = tirx.compile(function, target=target)
    actual = runtime.empty((1,), "int32")
    executable(runtime.tensor(np.array([5], dtype="int32")), actual)
    np.testing.assert_array_equal(actual.numpy(), [5])


@pytest.mark.parametrize("enabled", [False, True])
def test_known_value_conditional_propagation(enabled):
    output = TensorView.create("output", [1], "int32")
    function = simplify(
        [output.store(0, 0), ir.If(output.load(0) == 0, output.store(0, 42), None)],
        [output.tensor],
        {"propagate_knowns_to_prove_conditional": enabled},
    )
    assert bool(collect(function, ir.If)) != enabled
    executable = tirx.compile(function, target="llvm")
    actual = runtime.empty((1,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [42])


@pytest.mark.parametrize("enabled", [False, True])
def test_known_value_expression_propagation(enabled):
    output = TensorView.create("output", [1], "int32")
    function = simplify(
        [output.store(0, 5), output.store(0, output.load(0) + 7)], [output.tensor], {"propagate_knowns_to_simplify_expressions": enabled}
    )
    assert bool(collect(function, ir.TensorLoad)) != enabled
    executable = tirx.compile(function, target="llvm")
    actual = runtime.empty((1,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [12])


def test_known_values_are_invalidated_for_potential_parameter_aliases():
    first, second = TensorView.create("first", [1], "int32"), TensorView.create("second", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    function = simplify(
        [first.store(0, 3), second.store(0, 7), output.store(0, first.load(0))],
        [first.tensor, second.tensor, output.tensor],
        {"propagate_knowns_to_simplify_expressions": True},
    )
    assert len(collect(function, ir.TensorLoad)) == 1
    executable = tirx.compile(function, target="llvm")
    shared, actual = runtime.empty((1,), "int32"), runtime.empty((1,), "int32")
    executable(shared, shared, actual)
    np.testing.assert_array_equal(actual.numpy(), [7])


def test_known_values_are_invalidated_by_tensor_alias_stores():
    local = TensorView.create("local", [2], "int32", scope="local")
    alias = TensorView.create("alias", [1], "int32", scope="local", elem_offset=1)
    output = TensorView.create("output", [1], "int32")
    function = simplify(
        [local.allocate(), alias.declare(local.data), local.store(1, 3), alias.store(0, 11), output.store(0, local.load(1))],
        [output.tensor],
        {"propagate_knowns_to_simplify_expressions": True},
    )
    executable = tirx.compile(function, target="llvm")
    actual = runtime.empty((1,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [11])


def test_loop_carried_and_branch_values_are_not_stale():
    output = TensorView.create("output", [1], "int32")
    index, flag = ir.Var("index", "int32"), ir.Var("flag", "bool")
    body = [
        output.store(0, 1),
        ir.For(index, 0, 2, 1, ir.ForKind.DEFAULT, output.store(0, output.load(0) + 1)),
        ir.If(flag, output.store(0, 9), None),
        output.store(0, output.load(0) * 2),
    ]
    function = simplify(
        body, [output.tensor, flag], {"propagate_knowns_to_simplify_expressions": True, "propagate_knowns_to_prove_conditional": True}
    )
    executable = tirx.compile(function, target="llvm")
    for condition, expected in [(False, 6), (True, 18)]:
        actual = runtime.empty((1,), "int32")
        executable(actual, condition)
        np.testing.assert_array_equal(actual.numpy(), [expected])


def test_known_values_do_not_capture_out_of_scope_variables():
    output = TensorView.create("output", [1], "int32")
    scalar, value = ir.Var("scalar", "int32"), ir.Var("value", "int32")
    scope = ir.RegionStmt("tl.scope", [], [], None, [ir.Bind(value, scalar + 1), output.store(0, value)])
    function = simplify(
        [scope, output.store(0, output.load(0) * 2)],
        [output.tensor, scalar],
        {"enable_simplify_let_inline": False, "propagate_knowns_to_simplify_expressions": True},
    )
    executable = tirx.compile(function, target="llvm")
    actual = runtime.empty((1,), "int32")
    executable(actual, 10)
    np.testing.assert_array_equal(actual.numpy(), [22])


@pytest.mark.parametrize("region", ["tl.scope", "tl.scope"])
def test_local_scalar_known_values_survive_transparent_regions(region):
    output = TensorView.create("output", [1], "int32")
    scalar = TensorView.create("scalar", [1], "int32", scope="local")
    value = ir.Var("value", "int32")
    scope = ir.RegionStmt(region, [], [], None, scalar.store(0, value + 1))
    function = simplify([scalar.allocate(), scope, output.store(0, scalar.load(0))], [output.tensor, value])
    assert not collect(function, ir.TensorLoad)
    ir.assert_structural_equal(collect(function, ir.TensorStore)[-1].value, value + 1)


@pytest.mark.parametrize(
    "operator,reverse,expected", [(prim.And, False, 5), (prim.And, True, 5), (prim.Or, False, 10), (prim.Or, True, 10)]
)
def test_boolean_branch_constraints_apply_to_either_operand(operator, reverse, expected):
    output = TensorView.create("output", [1], "bool")
    value = ir.Var("value", "int32")
    operands = [value < 5, value < 10]
    if reverse:
        operands.reverse()
    function = simplify(output.store(0, operator(*operands)), [output.tensor, value], {"apply_constraints_to_boolean_branches": True})
    store = collect(function, ir.TensorStore)[0]
    ir.assert_structural_equal(store.value, value < expected)


def test_scoped_singleton_floordiv_is_folded():
    output = TensorView.create("output", [64], "int32")
    index = ir.Var("index", "int32")
    body = ir.For(index, 64, 128, 1, ir.ForKind.DEFAULT, output.store(index - 64, index // 64))
    function = simplify(body, [output.tensor])
    assert not collect(function, prim.FloorDiv)
    executable = tirx.compile(function, target="llvm")
    actual = runtime.empty((64,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.ones(64))


def test_simplify_preserves_allocation_and_layout_hint_bindings(native_layouts):
    extent = ir.Var("extent", "int32")
    local = TensorView.create("local", [extent], "int32", scope="local")
    output = TensorView.create("output", [1], "int32")
    layout = native_layouts.Layout([extent], lambda element: element)
    body = [
        ir.Bind(extent, ir.const(4)),
        local.allocate(),
        ir.Evaluate(ir.Call("tl.layout_hint", [local.tensor], attrs={"layout": layout})),
        local.store(0, 7),
        output.store(0, local.load(0)),
    ]
    function = simplify(body, [output.tensor])
    hints = [call for call in collect(function, ir.Call) if call.op.name == "tl.layout_hint"]
    assert len(hints) == 1
    allocations = [bind for bind in collect(function, ir.Bind) if tirx.is_tensor_var(bind.var)]
    assert hints[0].args[0].same_as(allocations[0].var)


def test_simplify_arguments_only_removes_unused_tensor_parameters():
    output, unused = TensorView.create("output", [1], "int32"), TensorView.create("unused", [1], "int32")
    scalar = ir.Var("scalar", "int32")
    function = simplify(output.store(0, 2), [output.tensor, unused.tensor, scalar], simplify_arguments=True)
    assert len(function.params) == 2
    assert function.params[0].same_as(output.tensor)
    assert function.params[1].same_as(scalar)


def test_injected_assumptions_remain_lexical_and_prove_constraints():
    output = TensorView.create("output", [1], "int32")
    value = ir.Var("value", "int32")
    body = [ir.Evaluate(ir.Call("prim.assume", [value >= 0])), ir.If(value >= 0, output.store(0, 13), output.store(0, -1))]
    function = tirx.Function([output.tensor, value], body, ret_type=ir.TupleType([]))
    injected = tvm_ffi.get_global_func("tl.transform.InjectAssumes")()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(injected)
    scopes = collect(injected, ir.RegionStmt)
    assert len(scopes) == 1
    ir.assert_structural_equal(scopes[0].attrs["tl.assume_requires_runtime_check"], value >= 0)
    result = simplify(injected.body, list(injected.params))
    assert not collect(result, ir.If)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))


def test_shape_facts_follow_local_tensor_definitions():
    extent = ir.Var("extent", "int32")
    local = TensorView.create("local", [extent], "int32", scope="local")
    function = tirx.Function([], [ir.Bind(extent, ir.const(4)), local.allocate()])
    result = tvm_ffi.get_global_func("tl.transform.InjectAssumes")()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    assert not collect(result, ir.RegionStmt)


@pytest.mark.parametrize("propagate", [False, True])
def test_mutable_branch_predicate_is_not_a_persistent_fact(propagate):
    output = TensorView.create("output", [1], "int32")
    body = ir.If(output.load(0) == 0, [output.store(0, 5), ir.If(output.load(0) == 0, output.store(0, -1), None)], None)
    function = simplify(body, [output.tensor], {"propagate_knowns_to_prove_conditional": propagate})
    executable = tirx.compile(function, target="llvm")
    actual = runtime.tensor(np.zeros((1,), dtype="int32"))
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [5])
