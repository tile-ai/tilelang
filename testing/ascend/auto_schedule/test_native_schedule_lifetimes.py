import pytest
import tvm_ffi
from tvm_tirx import ir
from tvm_tirx.target import Target

import tilelang
from tilelang._compat.tensor import TensorView
from testing.ascend.auto_schedule._scheduled_ir import _marker
from testing.ascend.auto_schedule.test_native_schedule_pipeline import _lower, _module, _nodes, _prefix


def _execute(module, *arguments, externs=None):
    module = tilelang.transform.LowerOpaqueBlock()(module)
    build = tvm_ffi.get_global_func("target.build.tilelang_python_without_compile")
    source = build(module, Target("c")).inspect_source("py")
    namespace = dict(externs or {})
    exec(source, namespace)
    namespace["main"](*arguments)


@pytest.mark.parametrize("explicit_task", [False, True])
@pytest.mark.parametrize("predicate", [False, True])
def test_dynamic_allocation_never_evaluates_a_disabled_shape(explicit_task, predicate):
    enabled = ir.Var("enabled", "bool")
    sizes = TensorView.create("sizes", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    size = ir.Var("size", "int32")
    storage = TensorView.create("dynamic", [size], "int32", scope="local.var")
    body = ir.SeqStmt([ir.Bind(size, sizes.load(0)), storage.allocate(), storage.store(0, 17), output.store(0, storage.load(0))])
    if explicit_task:
        body = ir.RegionStmt("tl.scope", [], [], {"tl.ascend_task": {}}, body)
    result = _lower(_module(ir.If(enabled, body, None), [enabled, sizes.tensor, output.tensor]))
    destination = [-1]
    _execute(result, predicate, [3] if predicate else [], destination)
    assert destination == [17 if predicate else -1]


@pytest.mark.parametrize("begin,end,step", [(0, 4, 1), (6, -2, -2)])
def test_staged_dynamic_allocations_execute_the_right_logical_iterations(begin, end, step):
    index = ir.Var("index", "int32")
    output = TensorView.create("output", [8], "int32")
    delayed = TensorView.create("delayed", [8], "int32")
    storage = TensorView.create("dynamic", [index + 1], "int32", scope="local.var")
    first = ir.RegionStmt(
        "tl.scope", [], [], {"tl.ascend_stage": 0}, [storage.allocate(), storage.store(0, index), output.store(index, storage.load(0))]
    )
    second = ir.RegionStmt("tl.scope", [], [], {"tl.ascend_stage": 1}, delayed.store(index, index + 10))
    loop = ir.For(index, begin, end, step, ir.ForKind.DEFAULT, [first, second], annotations={"num_stages": 2, "enable_offset": True})
    result = _lower(_module(loop, [output.tensor, delayed.tensor]))
    immediate_result, delayed_result = [-1] * 8, [-1] * 8
    _execute(result, immediate_result, delayed_result)
    expected_immediate, expected_delayed = [-1] * 8, [-1] * 8
    for iteration in range(begin, end, step):
        expected_immediate[iteration] = iteration
        expected_delayed[iteration] = iteration + 10
    assert immediate_result == expected_immediate
    assert delayed_result == expected_delayed


@pytest.mark.parametrize("begin,end,step", [(0, 4, 1), (6, -2, -2)])
def test_impure_scalar_results_are_evaluated_once_in_their_logical_iteration(begin, end, step):
    index = ir.Var("index", "int32")
    value = ir.Var("value", "int32")
    derived = ir.Var("derived", "int32")
    output = TensorView.create("output", [8], "int32")
    second = TensorView.create("second", [8], "int32")
    binding = ir.Bind(value, ir.Call("tirx.call_extern", ["next_value", index], ty="int32"))
    loop = ir.For(
        index,
        begin,
        end,
        step,
        ir.ForKind.DEFAULT,
        [binding, ir.Bind(derived, value * 2), output.store(index, derived), second.store(index, value + 3)],
        annotations={"num_stages": 3, "enable_offset": True},
    )
    module = _module(loop, [output.tensor, second.tensor])
    result = _lower(module)
    calls = [call for call in _nodes(result, ir.Call) if call.op.name == "tirx.call_extern"]
    assert len(calls) == 1
    visited = []

    def next_value(iteration):
        visited.append(iteration)
        return len(visited) * 10 + iteration

    first_result, second_result = [-1] * 8, [-1] * 8
    _execute(result, first_result, second_result, externs={"next_value": next_value})
    assert visited == list(range(begin, end, step))
    expected_first, expected_second = [-1] * 8, [-1] * 8
    for ordinal, iteration in enumerate(visited, 1):
        expected_first[iteration] = (ordinal * 10 + iteration) * 2
        expected_second[iteration] = ordinal * 10 + iteration + 3
    assert first_result == expected_first
    assert second_result == expected_second


@pytest.mark.parametrize("transitive", [False, True])
def test_manual_stages_cannot_reuse_another_iterations_impure_scalar(transitive):
    index = ir.Var("index", "int32")
    value = ir.Var("value", "int32")
    derived = ir.Var("derived", "int32")
    output = TensorView.create("output", [4], "int32")
    definitions = [ir.Bind(value, ir.Call("tirx.call_extern", ["next_value", index], ty="int32"))]
    if transitive:
        definitions.append(ir.Bind(derived, value * 2))
    producer = _marker("tl.ascend_stage", 0, ir.SeqStmt(definitions))
    consumer = ir.RegionStmt("tl.scope", [], [], {"tl.ascend_stage": 1}, output.store(index, derived if transitive else value))
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, [producer, consumer], annotations={"num_stages": 2, "enable_offset": True})
    with pytest.raises(Exception, match="non-recomputable SSA value cannot cross pipeline stages"):
        _prefix(_module(loop, [output.tensor]))


@pytest.mark.parametrize("begin,end,step", [(0, 4, 1), (6, -2, -2)])
def test_scalar_snapshot_clones_precede_pipeline_state_updates(begin, end, step):
    index = ir.Var("index", "int32")
    snapshot = ir.Var("snapshot", "int32")
    derived = ir.Var("derived", "int32")
    state = TensorView.create("state", [8], "int32")
    output = TensorView.create("output", [8], "int32")
    producer = _marker("tl.ascend_stage", 0, ir.SeqStmt([ir.Bind(snapshot, state.load(index)), ir.Bind(derived, snapshot * 2)]))
    consumer = ir.RegionStmt("tl.scope", [], [], {"tl.ascend_stage": 1}, [output.store(index, derived), state.store(index, snapshot + 1)])
    loop = ir.For(index, begin, end, step, ir.ForKind.DEFAULT, [producer, consumer], annotations={"num_stages": 2, "enable_offset": True})
    result = _lower(_module(loop, [state.tensor, output.tensor]))
    state_result, output_result = list(range(10, 18)), [-1] * 8
    _execute(result, state_result, output_result)
    expected = [-1] * 8
    expected_state = list(range(10, 18))
    for iteration in range(begin, end, step):
        expected[iteration] = (10 + iteration) * 2
        expected_state[iteration] += 1
    assert output_result == expected
    assert state_result == expected_state


@pytest.mark.parametrize("begin,end,step", [(0, 4, 1), (6, -2, -2)])
def test_scalar_snapshot_precedes_nested_pipeline_updates(begin, end, step):
    index = ir.Var("index", "int32")
    inner = ir.Var("inner", "int32")
    snapshot = ir.Var("snapshot", "int32")
    state = TensorView.create("state", [1], "int32")
    source = TensorView.create("source", [8], "int32")
    output = TensorView.create("output", [8], "int32")
    update = ir.For(inner, 0, 2, 1, ir.ForKind.DEFAULT, state.store(0, state.load(0) + source.load(index)))
    loop = ir.For(
        index,
        begin,
        end,
        step,
        ir.ForKind.DEFAULT,
        [ir.Bind(snapshot, state.load(0)), update, output.store(index, snapshot)],
        annotations={"num_stages": 2, "enable_offset": True},
    )
    result = _lower(_module(loop, [state.tensor, source.tensor, output.tensor]))
    state_result, output_result = [0], [-1] * 8
    _execute(result, state_result, list(range(8)), output_result)
    expected, accumulated = [-1] * 8, 0
    for iteration in range(begin, end, step):
        expected[iteration] = accumulated
        accumulated += 2 * iteration
    assert output_result == expected
    assert state_result == [accumulated]
