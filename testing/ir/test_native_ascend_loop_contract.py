"""Scheduler loop facts agree with native var < end execution for every step."""

import numpy as np
import pytest
from tvm_tirx import ir, tirx

from test_native_ascend_core_resolution import _kernel, _loop, _unit
from test_native_ascend_dependencies import _api, _ids, _inspect, _pairs
from test_native_ascend_lifetimes import _aliases, _can_alias, _read, _write
from test_native_ascend_schedule_analysis import _int, _memory, _points, _schedule
from test_native_reducer_contracts import allocate, execute, function, integer, tensor


def _for(var, begin, end, step, body):
    dtype = str(var.ty.dtype)
    values = [integer(value, dtype) if isinstance(value, int) else value for value in (begin, end, step)]
    return ir.For(var, *values, ir.ForKind.DEFAULT, body)


def _check(value):
    assert tirx.analysis.check_well_formed(value)
    assert not tirx.analysis.undefined_vars(value)
    ir.assert_structural_equal(value, ir.load_json(ir.save_json(value)))
    return value


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize(
    "begin,end,step,stop_after,points",
    [
        (6, 1, -1, 4, []),
        (3, 5, -1, 4, [3, 2, 1, 0]),
        (5, 8, -2, 3, [5, 3, 1]),
        (3, 5, 0, 3, [3, 3, 3]),
        (0, 7, 2, 8, [0, 2, 4, 6]),
    ],
)
def test_native_execution_and_loop_facts_keep_the_same_condition(dtype, begin, end, step, stop_after, points):
    index, slot = ir.Var("i", dtype), ir.Var("slot", "int32")
    trace, count = tensor("trace", [8], dtype, "global"), tensor("count", [1], "int32", "global")
    body = [
        ir.Bind(slot, ir.TensorLoad(count, [integer(0)])),
        ir.If(
            index < integer(begin, dtype),
            [ir.TensorStore(trace, [slot], index + integer(100, dtype))],
            [ir.TensorStore(trace, [slot], index)],
        ),
        ir.TensorStore(count, [integer(0)], slot + 1),
        ir.If(slot + 1 >= stop_after, [ir.Break()], None),
    ]
    if step < 0 and points:
        # The same final point supplies explicit machine no-wrap evidence for
        # the local ordinal, independently of the counter's execution bound.
        body.append(ir.If(index <= integer(points[-1], dtype), [ir.Break()], None))
    loop = _for(index, begin, end, step, body)
    value = _check(function([trace, count], [loop]))
    original = ir.save_json(value)
    actual, trips = execute(value, [np.full(8, -99, dtype=dtype), np.zeros(1, dtype="int32")], f"raw-for-{dtype}-{begin}-{end}-{step}")
    assert ir.save_json(value) == original
    expected = [point + (100 if point < begin else 0) for point in points]
    np.testing.assert_array_equal(actual, np.array([*expected, *([-99] * (8 - len(points)))], dtype=dtype))
    assert int(trips[0]) == len(points)

    _, root = _schedule([trace, count], [loop])
    result = _api("AnalyzeAscendSchedule")(root, [index < integer(end, dtype), index >= integer(begin, dtype)])
    row = result["loops"][0]
    assert int(row["trip_count"]) == (0 if not points else len(points) if step > 0 else 100)
    assert row["no_wrap"]
    assert not row["known_progress"] and not row["single_trip"]
    if points:
        assert list(row["proofs"]) == [True, step > 0]
    if step == 0 and points:
        with pytest.raises(ValueError, match="nonzero step"):
            _api("AscendLoopOrdinal")(root)
    else:
        (ordinal,) = _api("AscendLoopOrdinal")(root)
        mapping = tirx.IndexMap([index], [ordinal], None)
        assert [_int(mapping.map_indices([integer(point, dtype)])[0]) for point in points] == list(range(len(points)))


@pytest.mark.parametrize(
    "begin,end,step,cost,progress,single",
    [
        (6, 1, -1, 0, False, False),
        (3, 5, -1, 100, False, False),
        (3, 5, 0, 100, False, False),
        (0, 7, 2, 4, True, False),
        (3, 5, 4, 1, True, True),
        (5, 5, 1, 0, False, False),
        (6, 1, 2, 0, False, False),
    ],
)
def test_cost_fallback_is_separate_from_a_finite_progress_proof(begin, end, step, cost, progress, single):
    index = ir.Var("i", "int32")
    _, root = _schedule([], [_for(index, begin, end, step, [ir.Evaluate(index)])])
    row = _api("AnalyzeAscendSchedule")(root, [])["loops"][0]
    assert int(row["trip_count"]) == cost
    assert bool(row["known_progress"]) == progress
    assert bool(row["single_trip"]) == single


@pytest.mark.parametrize("step", [-2, -1, 0, 1, 2])
def test_zero_entry_body_has_no_memory_effects_for_any_step(step):
    source, output = tensor("source", [4], "int32", "global"), tensor("output", [4], "int32", "global")
    index = ir.Var("i", "int32")
    loop = _for(index, 6, 1, step, [ir.TensorStore(output, [integer(0)], ir.TensorLoad(source, [integer(0)]))])
    memory = _memory([loop])
    assert not memory["reads"] and not memory["writes"] and not memory["must_writes"]


def test_zero_entry_keeps_memory_effects_of_the_unconditional_loop_header():
    state, output = tensor("state", [1], "int32", "global"), tensor("output", [1], "int32", "global")
    index = ir.Var("i", "int32")
    step = ir.TensorLoad(state, [integer(0)])
    store = ir.TensorStore(output, [integer(0)], integer(2))
    loop = _for(index, 6, 1, step, [store])
    memory = _memory([loop])
    assert len(memory["reads"]) == 1 and memory["reads"][0].source.same_as(state)
    assert not memory["writes"]
    writer = ir.TensorStore(state, [integer(0)], integer(-1))
    scheduled_loop = _loop(index, [_unit(store)], begin=6, end=1, step=step)
    result = _inspect(_kernel([state, output], [_unit(writer), scheduled_loop]))
    pair = (_ids(result, writer)[0], _ids(result, step, header=True)[0], 0)
    assert any(row[:3] == pair for row in _pairs(result, root_only=True))


@pytest.mark.parametrize("step", [-2, -1, 0])
def test_entering_nonpositive_loops_use_conservative_may_write_domains(step):
    storage = tensor("storage", [8], "int32", "global")
    index = ir.Var("i", "int32")
    loop = _for(index, 3, 5, step, [ir.TensorStore(storage, [index], integer(1)), ir.If(index <= 0, [ir.Break()], None)])
    memory = _memory([loop])
    assert len(memory["writes"]) == 1
    assert {(i,) for i in range(4)} <= _points(memory["writes"][0])
    assert not memory["must_writes"]


@pytest.mark.parametrize("local_bind", [False, True])
@pytest.mark.parametrize("begin,end,expected", [(6, 1, False), (3, 5, True)])
def test_dependencies_do_not_erase_executing_decreasing_iterations(begin, end, expected, local_bind):
    storage = tensor("storage", [1], "int32", "global")
    index = ir.Var("i", "int32")
    value = ir.Var("value", "int32") if local_bind else index
    prefix = [_unit(ir.Bind(value, index + 1))] if local_bind else []
    writer, reader = ir.TensorStore(storage, [integer(0)], value), ir.TensorLoad(storage, [integer(0)])
    body = [*prefix, _unit(writer), _unit(ir.Evaluate(reader)), ir.If(index <= 0, [_unit(ir.Break())], None)]
    result = _inspect(_kernel([], [_unit(allocate(storage)), _loop(index, body, begin=begin, end=end, step=-1)]))
    pair = (_ids(result, writer)[0], _ids(result, reader)[0])
    pairs = _pairs(result, clock=index)
    assert any(row[:3] == (*pair, 0) for row in pairs) == expected
    # The forward RAW edge is already ordered at distance zero. The return
    # WAR edge and the writer's recurrence carry the next-iteration ordering.
    assert any(row[:3] == (pair[1], pair[0], -1) for row in pairs) == expected
    assert any(row[:3] == (pair[0], pair[0], -1) for row in pairs) == expected


@pytest.mark.parametrize("dynamic_step", [False, True])
@pytest.mark.parametrize("local_bind", [False, True])
def test_repeated_iterations_may_have_the_same_coordinate(dynamic_step, local_bind):
    storage = tensor("storage", [8], "int32", "global")
    counter = tensor("counter", [1], "int32", "global")
    index = ir.Var("i", "int32")
    slot = ir.Var("slot", "int32")
    step = ir.Var("step", "int32") if dynamic_step else integer(0)
    value = ir.Var("value", "int32") if local_bind else index
    prefix = [_unit(ir.Bind(value, index))] if local_bind else []
    writer = ir.TensorStore(storage, [value], integer(1))
    reader = ir.TensorLoad(storage, [value])
    body = [
        *prefix,
        _unit(ir.Bind(slot, ir.TensorLoad(counter, [integer(0)]))),
        _unit(writer),
        _unit(ir.Evaluate(reader)),
        _unit(ir.TensorStore(counter, [integer(0)], slot + 1)),
        ir.If(slot + 1 >= 3, [_unit(ir.Break())], None),
    ]
    before = [_unit(allocate(storage)), _unit(allocate(counter)), _unit(ir.TensorStore(counter, [integer(0)], integer(0)))]
    result = _inspect(_kernel([step] if dynamic_step else [], [*before, _loop(index, body, begin=3, end=5, step=step)]))
    pair = (_ids(result, writer)[0], _ids(result, reader)[0])
    pairs = _pairs(result, clock=index)
    assert any(row[:3] == (*pair, 0) for row in pairs)
    assert any(row[:3] == (pair[1], pair[0], -1) for row in pairs)
    assert any(row[:3] == (pair[0], pair[0], -1) for row in pairs)


@pytest.mark.parametrize("begin,end", [(6, 1), (3, 5)])
def test_nonpositive_loops_cannot_prove_a_complete_initializing_write(begin, end):
    a, b = tensor("a", [1], "int32", "shared.dyn"), tensor("b", [1], "int32", "shared.dyn")
    index = ir.Var("i", "int32")
    loop = _for(index, begin, end, -1, [_write(a), ir.If(index <= 0, [ir.Break()], None)])
    ra, wb, rb = _read(a), _write(b), _read(b)
    value = _kernel([], [_unit(allocate(a)), _unit(allocate(b)), _unit(ir.SeqStmt([loop])), _unit(ra), _unit(wb), _unit(rb)])
    edges = [(loop, ra, 0, True), (ra, wb, 0, True), (wb, rb, 0, True)]
    assert not _can_alias(_aliases(value, edges), a, b)


@pytest.mark.parametrize("step", [-1, 0])
def test_flattened_parent_clock_does_not_invent_a_nonpositive_inner_span(step):
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    nested = _for(inner, 3, 5, step, [ir.Evaluate(inner), ir.If(inner <= 3, [ir.Break()], None)])
    _, root = _schedule([], [_for(outer, 0, 2, 1, [nested])])
    with pytest.raises(ValueError, match="positive inner nominal span|nonzero step"):
        _api("AscendLoopOrdinal")(root)


def test_positive_inner_break_leaves_holes_in_the_nominal_clock():
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    output = tensor("output", [2], "int64", "global")

    def program(coordinate):
        child = _for(inner, 3, 5, 1, [ir.TensorStore(output, [outer], coordinate), ir.Break()])
        return _for(outer, 0, 2, 1, [child])

    original = program(ir.PrimType("int64")(outer * 2 + inner - 3))
    _, root = _schedule([output], [original])
    ordinal = _api("AscendLoopOrdinal")(root)[1]
    rewritten = program(ordinal)
    for label, loop in [("direct", original), ("derived", rewritten)]:
        value = _check(function([output], [loop]))
        (actual,) = execute(value, [np.full(2, -99, dtype="int64")], "raw-for-nominal-clock-" + label)
        np.testing.assert_array_equal(actual, [0, 2])


def test_zero_entry_inner_clock_does_not_construct_a_division_by_zero():
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    _, root = _schedule([], [_for(outer, 0, 2, 1, [_for(inner, 6, 1, 0, [ir.Evaluate(inner)])])])
    ordinals = _api("AscendLoopOrdinal")(root)
    assert _int(ordinals[1]) == 0


def test_zero_entry_outer_clock_does_not_evaluate_an_unknown_inner_header():
    state = tensor("state", [1], "int32", "global")
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    step = ir.TensorLoad(state, [integer(0)])
    _, root = _schedule([state], [_for(outer, 0, 0, 1, [_for(inner, 3, 5, step, [ir.Evaluate(inner)])])])
    assert [_int(value) for value in _api("AscendLoopOrdinal")(root)] == [0, 0]


def test_decreasing_outer_clock_composes_with_a_finite_positive_inner_loop():
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    output = tensor("output", [12], "int32", "global")
    local_ordinal = (3 - outer) * 3 + inner
    child = _for(inner, 0, 3, 1, [ir.TensorStore(output, [local_ordinal], outer * 10 + inner)])
    loop = _for(outer, 3, 5, -1, [child, ir.If(outer <= 0, [ir.Break()], None)])
    value = _check(function([output], [loop]))
    expected = np.array([30, 31, 32, 20, 21, 22, 10, 11, 12, 0, 1, 2], dtype="int32")
    (actual,) = execute(value, [np.full(12, -99, dtype="int32")], "raw-for-nested-decreasing-outer")
    np.testing.assert_array_equal(actual, expected)
    _, root = _schedule([output], [loop])
    ordinals = _api("AscendLoopOrdinal")(root)
    mapping = tirx.IndexMap([outer, inner], [ordinals[1]], None)
    assert [_int(mapping.map_indices([integer(i), integer(j)])[0]) for i in [3, 2, 1, 0] for j in [0, 1, 2]] == list(range(12))


@pytest.mark.parametrize(
    "dtype,begin,end,step,points",
    [
        ("int8", 120, 121, 127, [120, -9, 118, -11]),
        ("int16", 32760, 32761, 32767, [32760, -9, 32758, -11]),
        ("uint8", 250, 251, 255, [250, 249, 248, 247]),
        ("uint16", 65530, 65531, 65535, [65530, 65529, 65528, 65527]),
    ],
)
def test_wrapping_positive_induction_cannot_supply_a_finite_range(dtype, begin, end, step, points):
    index = ir.Var("i", dtype)
    output = tensor("output", [3], "int32", "global")

    def load(position):
        return ir.TensorLoad(output, [integer(position)])

    body = [
        ir.TensorStore(output, [integer(0)], load(0) + 1),
        ir.If(index < integer(begin, dtype), [ir.TensorStore(output, [integer(1)], load(1) + 1)], None),
        ir.TensorStore(output, [integer(2)], load(2) + ir.prim.Cast("int32", index)),
        ir.If(load(0) >= 4, [ir.Break()], None),
    ]
    loop = _for(index, begin, end, step, body)
    value = _check(function([output], [loop]))
    (actual,) = execute(value, [np.zeros(3, dtype="int32")], "raw-for-wrap-" + dtype)
    np.testing.assert_array_equal(actual, [4, sum(point < begin for point in points), sum(points)])

    _, root = _schedule([], [_for(index, begin, end, step, [ir.Evaluate(index)])])
    row = _api("AnalyzeAscendSchedule")(root, [index < integer(end, dtype), index >= integer(begin, dtype)])["loops"][0]
    assert int(row["trip_count"]) == 100
    assert not row["no_wrap"]
    assert not row["known_progress"] and not row["single_trip"]
    assert list(row["proofs"]) == [True, False]
    with pytest.raises(ValueError, match="non.wrapping|finite.*span"):
        _api("AscendLoopOrdinal")(root)


def test_wrapping_positive_loop_keeps_the_complete_may_access_domain():
    storage = tensor("storage", [256], "int32", "global")
    index = ir.Var("i", "int8")
    body = [
        ir.TensorStore(storage, [ir.prim.Cast("int32", index) + 128], integer(1)),
        ir.If(index == integer(-11, "int8"), [ir.Break()], None),
    ]
    loop = _for(index, 120, 121, 127, body)
    memory = _memory([loop])
    assert {(point + 128,) for point in [120, -9, 118, -11]} <= _points(memory["writes"][0])
    assert not memory["must_writes"]


@pytest.mark.parametrize("dtype", ["int8", "int16", "uint8", "uint16"])
def test_safe_unit_step_keeps_finite_range_cost_and_ordinal(dtype):
    index = ir.Var("i", dtype)
    _, root = _schedule([], [_for(index, 0, 7, 1, [ir.Evaluate(index)])])
    row = _api("AnalyzeAscendSchedule")(root, [index >= integer(0, dtype), index < integer(7, dtype)])["loops"][0]
    assert int(row["trip_count"]) == 7
    assert row["known_progress"] and row["no_wrap"] and not row["single_trip"]
    assert all(row["proofs"])
    (ordinal,) = _api("AscendLoopOrdinal")(root)
    mapping = tirx.IndexMap([index], [ordinal], None)
    assert [_int(mapping.map_indices([integer(point, dtype)])[0]) for point in range(7)] == list(range(7))


def test_negative_ordinal_requires_a_machine_no_wrap_proof():
    index, stop = ir.Var("i", "int64"), ir.Var("stop", "bool")
    loop = _for(index, 3, 5, -1, [ir.Evaluate(index), ir.If(stop, [ir.Break()], None)])
    _, root = _schedule([stop], [loop])
    row = _api("AnalyzeAscendSchedule")(root, [])["loops"][0]
    assert not row["no_wrap"]
    with pytest.raises(ValueError, match="non.wrapping"):
        _api("AscendLoopOrdinal")(root)


def test_full_width_signed_distance_is_not_a_single_trip_or_int64_clock():
    index = ir.Var("i", "int64")
    _, root = _schedule([], [_for(index, -(2**63), 2**63 - 1, 1, [ir.Evaluate(index)])])
    row = _api("AnalyzeAscendSchedule")(root, [])["loops"][0]
    assert row["known_progress"] and row["no_wrap"] and not row["single_trip"]
    with pytest.raises(ValueError, match="int64"):
        _api("AscendLoopOrdinal")(root)


@pytest.mark.parametrize("overflow", [False, True])
def test_nominal_clock_products_must_fit_int64(overflow):
    variables = [ir.Var(name, "int64") for name in ("i", "j", "k")]
    extent = 2**22 if overflow else 2**21
    body = [ir.Evaluate(variables[-1])]
    for index in reversed(variables):
        body = [_for(index, 0, extent, 1, body)]
    _, root = _schedule([], body)
    if overflow:
        with pytest.raises(ValueError, match="int64"):
            _api("AscendLoopOrdinal")(root)
    else:
        ordinal = _api("AscendLoopOrdinal")(root)[-1]
        mapping = tirx.IndexMap(variables, [ordinal], None)
        assert _int(mapping.map_indices([integer(extent - 1, "int64")] * 3)[0]) == 2**63 - 1


def test_nominal_span_avoids_an_overflowing_ceiling_division_numerator():
    outer, inner = ir.Var("outer", "int64"), ir.Var("inner", "int64")
    begin, end = -(2**62), 2**62 - 1
    child = _for(inner, begin, end, 2, [ir.Evaluate(inner)])
    _, root = _schedule([], [_for(outer, 0, 2, 1, [child])])
    ordinal = _api("AscendLoopOrdinal")(root)[-1]
    mapping = tirx.IndexMap([outer, inner], [ordinal], None)
    # The span is 2**62, but distance + step - 1 is 2**63. Both
    # the full mixed-radix clock and every actual induction update fit.
    assert _int(mapping.map_indices([integer(1, "int64"), integer(end - 1, "int64")])[0]) == 2**63 - 1


def test_wrapping_machine_coordinates_do_not_erase_writer_recurrence():
    storage = tensor("storage", [256], "int32", "global")
    counter = tensor("counter", [1], "int32", "global")
    index, slot = ir.Var("i", "int8"), ir.Var("slot", "int32")
    writer = ir.TensorStore(storage, [ir.prim.Cast("int32", index) + 128], integer(1))
    body = [
        _unit(ir.Bind(slot, ir.TensorLoad(counter, [integer(0)]))),
        _unit(writer),
        _unit(ir.TensorStore(counter, [integer(0)], slot + 1)),
        ir.If(slot + 1 >= 130, [_unit(ir.Break())], None),
    ]
    loop = _unit(_for(index, -128, 127, 2, body))
    before = [_unit(allocate(storage)), _unit(allocate(counter)), _unit(ir.TensorStore(counter, [integer(0)], integer(0)))]
    result = _inspect(_kernel([], [*before, loop]))
    task = _ids(result, writer)[0]
    # After 128 updates the machine coordinate is -128 again. Distinct
    # executions can therefore write the identical physical element.
    assert any(row[:3] == (task, task, -1) for row in _pairs(result, clock=index))


@pytest.mark.parametrize("case", ["continue", "inner_break", "partial_core", "same_name"])
def test_negative_induction_needs_the_correct_mandatory_break(case):
    index = ir.Var("i", "int32")
    params = []
    cutoff = index
    body = [_unit(ir.Evaluate(index))]
    if case == "same_name":
        cutoff = ir.Var("i", "int32")
        params.append(cutoff)
    if case == "continue":
        body.append(ir.If(index <= 0, [_unit(ir.Continue())], None))
    stop = ir.If(cutoff <= 0, [_unit(ir.Break(), core=1 if case == "partial_core" else 7)], None)
    if case == "inner_break":
        child = ir.Var("inner", "int32")
        body.append(_loop(child, [stop], begin=0, end=1, step=1))
    else:
        body.append(stop)
    value = _kernel(params, [_loop(index, body, begin=3, end=5, step=-1)])
    root = value.body
    _check(value)
    row = _api("AnalyzeAscendSchedule")(root, [])["loops"][0]
    assert not row["no_wrap"]
    with pytest.raises(ValueError, match="non.wrapping"):
        _api("AscendLoopOrdinal")(root)


def test_induction_uses_current_children_after_the_original_tail_is_removed():
    index = ir.Var("i", "int32")
    _, root = _schedule([], [_for(index, 3, 5, -1, [ir.Evaluate(index), ir.If(index <= 0, [ir.Break()], None)])])
    before = ir.save_json(root)
    (row,) = _api("AscendInductionAfterDroppingTail")(root)
    assert row["before"] and not row["after"]
    assert ir.save_json(root) == before
