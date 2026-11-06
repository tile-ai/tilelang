"""Storage counters retain predicate identity, memory epochs and native scopes."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from test_native_ascend_core_passes import _apply, _assert_schedule_consumed, _core, _metadata, _run
from test_native_ascend_core_resolution import _kernel, _unit
from test_native_ascend_schedule_units import _attributes, _bind_target, _fields
from test_native_reducer_contracts import allocate, annotation, attribute, execute, function, integer, objects, tensor


def _fixture(relation, wrapper, *, reverse_allocations=False):
    left = tensor("left", [1], "int32", "shared.dyn")
    right = tensor("right", [1], "int32", "shared.dyn")
    pred = tensor("pred", [1], "int32", "global")
    output = tensor("output", [10], "int32", "global")
    i, p, q = (ir.Var(name, "int32") for name in ("i", "p", "q"))
    first = p > 0
    second = p > 0 if relation == "same" else p >= 1 if relation == "equivalent" else q > 0
    prefix, middle = [], []
    params = [output, p, q]
    if relation in {"mutated-memory", "unchanged-memory", "equivalent-snapshots"}:
        first, second = ir.Var("before_write", "bool"), ir.Var("after_write", "bool")
        first_value = p > 0 if relation == "equivalent-snapshots" else ir.TensorLoad(pred, [integer(0)]) > 0
        second_value = p >= 1 if relation == "equivalent-snapshots" else ir.TensorLoad(pred, [integer(0)]) > 0
        prefix = [_unit(ir.Bind(first, first_value), 1)]
        if relation == "mutated-memory":
            middle.append(_unit(ir.TensorStore(pred, [integer(0)], integer(1)), 1))
        middle.append(_unit(ir.Bind(second, second_value), 1))
        if relation != "equivalent-snapshots":
            params = [pred, output]
    elif relation == "live-memory":
        first = second = ir.TensorLoad(pred, [integer(0)]) > 0
        middle = [_unit(ir.TensorStore(pred, [integer(0)], integer(1)), 1)]
        params = [pred, output]

    def accesses(storage, guard, offset, value):
        return [
            _unit(ir.TensorStore(storage, [integer(0)], i + value), 1, guard=guard),
            _unit(ir.TensorStore(output, [i + offset], ir.TensorLoad(storage, [integer(0)])), 1, guard=guard),
        ]

    first_part = [*prefix, *accesses(left, first, 0, 10)]
    second_part = [*middle, *accesses(right, second, 4, 20)]
    body = [*first_part, *second_part]
    fields = {"multi_buffer_eligible": [left, right], "num_stages": integer(1)}
    if wrapper == "siblings":
        body = [
            attribute(None, "tl.loop.annotations", {"enable_offset": False}, first_part),
            attribute(None, "tl.loop.annotations", {"num_stages": integer(1)}, second_part),
        ]
    elif wrapper in {"sole", "nested"}:
        if wrapper == "nested":
            body = [attribute(None, "tl.loop.annotations", {"enable_offset": False}, body)]
        body = [attribute(None, "tl.loop.annotations", fields, body)]
    owner = ir.For(i, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, body, fields if wrapper in {"flat", "siblings"} else {})
    allocations = [right, left] if reverse_allocations else [left, right]
    value = _bind_target(
        _kernel(
            params,
            [
                *[_unit(allocate(storage), 1) for storage in allocations],
                annotation({"tl.buffer_version_mode": {left: "counter", right: "counter"}}),
                _unit(owner),
            ],
            {left: 2, right: 3},
            count=1,
        )
    )
    return value, (left, right), (first, second), output


def _epoch_fields(value):
    (owner,) = objects(value, ir.For)
    fields = dict(owner.annotations)
    for metadata in _metadata(owner):
        fields.update(metadata)
    return fields


def _statements(value, parents=()):
    yield value, parents
    if isinstance(value, ir.SeqStmt):
        children = value.seq
    elif isinstance(value, ir.If):
        children = [value.then_case] + ([value.else_case] if value.else_case is not None else [])
    elif isinstance(value, (ir.RegionStmt, ir.For)):
        children = [value.body]
    else:
        children = []
    for child in children:
        yield from _statements(child, (*parents, value))


def _assert_protocol(value, storages, expected_guards, shared):
    fields = _epoch_fields(value)
    clocks, guards = fields["tl.multi_buffer_counter_map"], fields["tl.storage_epoch_guard_map"]
    assert clocks[storages[0]].same_as(clocks[storages[1]]) == shared
    for storage, expected in zip(storages, expected_guards):
        ir.assert_structural_equal(guards[storage], expected)
        (parents,) = [
            parents
            for statement, parents in _statements(value.body)
            if isinstance(statement, ir.TensorStore)
            and statement.dest.same_as(clocks[storage])
            and not isinstance(statement.value, ir.IntImm)
        ]
        conditions = [parent.condition for parent in parents if isinstance(parent, ir.If)]
        analyzer = tvm_tirx.sym.Analyzer()
        if analyzer.can_prove(guards[storage]):
            assert all(analyzer.can_prove(condition) for condition in conditions)
        else:
            assert any(analyzer.can_prove_equal(condition, guards[storage]) for condition in conditions)
    assert int(fields["num_stages"]) == 1
    return clocks, guards


@pytest.mark.parametrize("wrapper", ["flat", "sole", "nested"])
@pytest.mark.parametrize("relation", ["same", "equivalent", "different", "mutated-memory", "live-memory"])
def test_prepare_counter_groups_require_proven_guard_equivalence(relation, wrapper):
    value, storages, predicates, _ = _fixture(relation, wrapper)
    prepared = _run("PrepareMultiBuffer", value)
    expected = (
        (ir.const(True, "bool"),) * 2 if relation == "live-memory" else (predicates[0],) * 2 if relation == "equivalent" else predicates
    )
    _assert_protocol(prepared, storages, expected, relation in {"same", "equivalent"})
    if wrapper == "nested":
        assert not bool(_epoch_fields(prepared)["enable_offset"])
    if relation == "mutated-memory":
        bindings = [binding for binding in objects(prepared, ir.Bind) if binding.var.name in {"before_write", "after_write"}]
        assert len(bindings) == 2
        assert all(objects(binding.value, ir.TensorLoad) for binding in bindings)


@pytest.mark.parametrize("wrapper", ["flat", "sole", "nested"])
@pytest.mark.parametrize("reverse_allocations", [False, True])
def test_snapshot_representatives_preserve_proof_and_definition_scope(wrapper, reverse_allocations):
    value, storages, predicates, _ = _fixture("equivalent-snapshots", wrapper, reverse_allocations=reverse_allocations)
    prepared = _run("PrepareMultiBuffer", value)
    # The loop proof context includes flat SSA definitions. Region-local
    # definitions are deliberately not imported into that enclosing context;
    # their separate counters must retain the original in-scope predicates.
    shared = wrapper == "flat"
    expected = (predicates[0],) * 2 if shared else predicates
    _, guards = _assert_protocol(prepared, storages, expected, shared)
    assert all(guards[storage].same_as(predicate) for storage, predicate in zip(storages, expected))


@pytest.mark.parametrize("relation", ["mutated-memory", "unchanged-memory"])
def test_sibling_metadata_scopes_do_not_prove_guard_equivalence(relation):
    value, storages, _, _ = _fixture(relation, "siblings")
    prepared = _run("PrepareMultiBuffer", value)
    # Sibling-local snapshots cannot escape into an owner-level protocol. The
    # lexical fallback preserves separate clocks without claiming equality of
    # the actual predicates, even when no intervening store is present.
    _assert_protocol(prepared, storages, (ir.const(True, "bool"),) * 2, False)
    markers = _attributes(prepared, "tl.loop.annotations")
    for marker_name, expected in (("enable_offset", 0), ("num_stages", 1)):
        (marker,) = [marker for marker in markers if marker_name in _fields(marker)["value"]]
        assert int(_fields(marker)["value"][marker_name]) == expected
        assert any(binding.var.name in {"before_write", "after_write"} for binding in objects(marker.body, ir.Bind))


def _observe_generated_counters(prepared, storages, output):
    clocks = _epoch_fields(prepared)["tl.multi_buffer_counter_map"]
    root = prepared.body

    def append_observers(node):
        if not node.same_as(root):
            return node
        observers = [
            _unit(ir.TensorStore(output, [integer(8 + index)], ir.TensorLoad(clocks[storage], [integer(0)])), 1)
            for index, storage in enumerate(storages)
        ]
        return ir.SeqStmt([*node.seq, *observers], node.loc)

    # Observe the actual generated state after its owner loop; retain every
    # snapshot binding, protocol update and metadata consumer in the pipeline.
    return tvm_ffi.structural_map(prepared, (ir.SeqStmt, append_observers), order="pre")


@pytest.mark.parametrize("wrapper", ["flat", "sole", "nested"])
@pytest.mark.parametrize("initial", [-1, 0, 1])
def test_memory_snapshots_execute_with_distinct_active_epoch_counts(wrapper, initial):
    value, storages, predicates, output = _fixture("mutated-memory", wrapper)
    value = _run("PrepareMultiBuffer", value)
    _assert_protocol(value, storages, predicates, False)
    value = _observe_generated_counters(value, storages, output)
    for name in ("ResolveCore", "InsertSync", "MaterializeMultiBuffer", "LowerScheduledTIR"):
        value = _run(name, value)
    _assert_schedule_consumed(value)
    value = _bind_target(function(list(value.params), _core(value, "VECTOR").body))
    for name in ("NormalizeBufferVersion", "RewriteAscendBufferVersionLayout"):
        value = _run(name, value)
    value = _run("AscendThreadSync", value, "shared")
    value = _apply(tirx.transform.LowerTensorLayouts(), value)
    value = _run("MergeUBAllocations", value, 32, False)
    observed_pred, observed_output = execute(
        function(list(value.params), value.body),
        [np.array([initial], dtype="int32"), np.full(10, -99, dtype="int32")],
        f"storage-epoch-snapshots-{wrapper}-{initial}",
    )
    np.testing.assert_array_equal(observed_pred, [1])
    np.testing.assert_array_equal(observed_output, [10 if initial > 0 else -99, 11, 12, 13, 20, 21, 22, 23, 4 if initial > 0 else 3, 4])
