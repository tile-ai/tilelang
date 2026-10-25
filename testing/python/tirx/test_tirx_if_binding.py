import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def bind_guards(function, inline=True):
    with ir.transform.PassContext(config={"tl.if_stmt_binding_inline_replayable_binds": inline}):
        result = tvm_ffi.get_global_func("tl.transform.IfStmtBinding")()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    ir.assert_structural_equal(result, bind_guards_once(result, inline))
    return result


def bind_guards_once(function, inline):
    with ir.transform.PassContext(config={"tl.if_stmt_binding_inline_replayable_binds": inline}):
        return tvm_ffi.get_global_func("tl.transform.IfStmtBinding")()(ir.IRModule({"main": function}))["main"]


def nodes(function, kind):
    result = []
    stmt_functor.post_order_visit(function, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


@pytest.mark.parametrize("inline", [False, True])
def test_pure_scalar_binding_inlining_keeps_lexical_scope(inline):
    output = TensorView.create("output", [2], "int32")
    enabled = ir.Var("enabled", "bool")
    value = ir.Var("value", "int32")
    function = tirx.Function(
        [output.tensor, enabled], ir.If(enabled, [ir.Bind(value, ir.const(7)), output.store(0, value), output.store(1, value + 2)], None)
    ).with_attr("global_symbol", "main")
    result = bind_guards(function, inline)
    assert len(nodes(result, ir.Bind)) == (0 if inline else 1)
    assert len(nodes(result, ir.If)) == (2 if inline else 1)
    executable = tirx.compile(result, target="c")
    for condition in [False, True]:
        actual = runtime.tensor(np.zeros(2, dtype="int32"))
        executable(actual, condition)
        np.testing.assert_array_equal(actual.numpy(), [7, 9] if condition else [0, 0])


@pytest.mark.parametrize("alias_kind", ["same_tensor", "declared_view", "parameters"])
def test_mutated_condition_is_snapshotted_once(alias_kind):
    condition_tensor = TensorView.create("condition", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    alias = condition_tensor if alias_kind == "same_tensor" else TensorView.create("alias", [1], "int32")
    parameters = [condition_tensor.tensor, output.tensor]
    prefix = []
    if alias_kind == "declared_view":
        prefix = [alias.declare(condition_tensor.data)]
    elif alias_kind == "parameters":
        parameters.append(alias.tensor)
    function = tirx.Function(
        parameters, prefix + [ir.If(condition_tensor.load(0) > 0, [alias.store(0, 0), output.store(0, 19)], None)]
    ).with_attr("global_symbol", "main")
    result = bind_guards(function)
    assert any(binding.var.name == "if_condition" for binding in nodes(result, ir.Bind))
    executable = tirx.compile(result, target="llvm")
    value = runtime.tensor(np.array([1], dtype="int32"))
    actual = runtime.tensor(np.zeros(1, dtype="int32"))
    executable(*([value, actual, value] if alias_kind == "parameters" else [value, actual]))
    np.testing.assert_array_equal(actual.numpy(), [19])
    np.testing.assert_array_equal(value.numpy(), [0])


@pytest.mark.parametrize("noalias", [False, True])
def test_read_bind_replays_only_when_writes_cannot_alias(noalias):
    source = TensorView.create("source", [1], "int32")
    output = TensorView.create("output", [2], "int32")
    enabled = ir.Var("enabled", "bool")
    value = ir.Var("value", "int32")
    function = tirx.Function(
        [source.tensor, output.tensor, enabled],
        ir.If(enabled, [ir.Bind(value, source.load(0)), output.store(0, value), output.store(1, value)], None),
    ).with_attr("tirx.noalias", noalias)
    result = bind_guards(function)
    assert len(nodes(result, ir.Bind)) == (0 if noalias else 1)


def test_nonrestrict_parameters_override_noalias_for_guard_replay():
    source = TensorView.create("source", [1], "int32")
    output = TensorView.create("output", [2], "int32")
    value = ir.Var("value", "int32")
    function = (
        tirx.Function(
            [source.tensor, output.tensor],
            ir.If(ir.const(True), [ir.Bind(value, source.load(0)), output.store(0, 0), output.store(1, value)], None),
        )
        .with_attr("tirx.noalias", True)
        .with_attr("tl.non_restrict_param_indices", [0, 1])
    )
    assert len(nodes(bind_guards(function), ir.Bind)) == 1


def test_mutable_read_bind_is_not_replayed_after_store():
    source = TensorView.create("source", [1], "int32")
    output = TensorView.create("output", [2], "int32")
    enabled = ir.Var("enabled", "bool")
    value = ir.Var("value", "int32")
    function = tirx.Function(
        [source.tensor, output.tensor, enabled],
        ir.If(enabled, [ir.Bind(value, source.load(0)), output.store(0, value), source.store(0, 3), output.store(1, value)], None),
    ).with_attr("global_symbol", "main")
    result = bind_guards(function)
    assert len(nodes(result, ir.Bind)) == 1
    executable = tirx.compile(result, target="c")
    original = runtime.tensor(np.array([5], dtype="int32"))
    actual = runtime.tensor(np.zeros(2, dtype="int32"))
    executable(original, actual, True)
    np.testing.assert_array_equal(actual.numpy(), [5, 5])


def test_tensor_allocation_and_users_stay_in_one_guard():
    temporary = TensorView.create("temporary", [2], "int32", scope="local")
    output = TensorView.create("output", [2], "int32")
    enabled = ir.Var("enabled", "bool")
    function = tirx.Function(
        [output.tensor, enabled],
        ir.If(enabled, [output.store(0, 5), temporary.allocate(), temporary.store(0, 9), output.store(1, temporary.load(0))], None),
    ).with_attr("global_symbol", "main")
    result = bind_guards(function)
    assert len(nodes(result, ir.If)) == 2
    executable = tirx.compile(result, target="c")
    actual = runtime.tensor(np.zeros(2, dtype="int32"))
    executable(actual, True)
    np.testing.assert_array_equal(actual.numpy(), [5, 9])


def test_tile_operator_write_is_a_guard_replay_barrier():
    condition = TensorView.create("condition", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    fill = ir.Evaluate(ir.Call("tl.tileop.fill", [condition.tensor, ir.const(0)]))
    function = tirx.Function([condition.tensor, output.tensor], ir.If(condition.load(0) > 0, [fill, output.store(0, 11)], None))
    result = bind_guards(function)
    assert len(nodes(result, ir.If)) == 2
    assert len(nodes(result, ir.Bind)) == 1


def test_atomic_condition_is_evaluated_once():
    counter = TensorView.create("counter", [1], "int32")
    output = TensorView.create("output", [2], "int32")
    atomic = ir.Call("tl.atomic_add_ret_elem_op", [counter.pointer_at(0), ir.const(1), ir.const(0)], ty="int32")
    function = tirx.Function([counter.tensor, output.tensor], ir.If(atomic == 0, [output.store(0, 5), output.store(1, 7)], None)).with_attr(
        "global_symbol", "main"
    )
    result = bind_guards(function)
    module = tvm_ffi.get_global_func("tl.cpu.transform.LowerCPUAtomics")()(ir.IRModule({"main": result}))
    executable = tirx.compile(module, target="c")
    count = runtime.tensor(np.zeros(1, dtype="int32"))
    actual = runtime.tensor(np.zeros(2, dtype="int32"))
    executable(count, actual)
    np.testing.assert_array_equal(count.numpy(), [1])
    np.testing.assert_array_equal(actual.numpy(), [5, 7])
