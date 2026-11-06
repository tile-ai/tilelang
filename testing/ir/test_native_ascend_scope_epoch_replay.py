"""Replaying total SSA predicates preserves memory snapshots and epoch counts."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from test_native_ascend_core_passes import _apply, _assert_schedule_consumed, _core, _metadata, _run
from test_native_ascend_core_resolution import _kernel, _unit
from test_native_ascend_schedule_units import _bind_target
from test_native_reducer_contracts import allocate, annotation, execute, function, integer, objects, tensor


def _snapshot_program(kind):
    storage = tensor("storage", [1], "int32", "shared.dyn")
    output = tensor("output", [13], "int32", "global")
    predicate = tensor("predicate", [1], "int32", "global")
    index = ir.Var("i", "int32")
    outer, inner = ir.Var("outer", "bool"), ir.Var("inner", "bool")
    denominator = ir.Var("denominator", "int32")
    inner_value = index % 3 == 0
    params = [output]
    after_capture = []
    if kind == "mutable":
        inner_value = ir.TensorLoad(predicate, [integer(0)]) > 0
        params.append(predicate)
        after_capture.append(_unit(ir.TensorStore(predicate, [integer(0)], integer(1)), 1))
    elif kind == "division":
        inner_value = index // denominator > 0
        params.append(denominator)
    elif kind == "arithmetic":
        inner_value = (index + 1) % 3 == 0
    body = [
        _unit(ir.Bind(outer, index % 2 == 0), 1),
        ir.If(
            outer,
            [
                _unit(ir.Bind(inner, inner_value), 1),
                *after_capture,
                ir.If(
                    inner,
                    [
                        _unit(ir.TensorStore(storage, [integer(0)], index + 1), 1),
                        _unit(ir.TensorStore(output, [index], ir.TensorLoad(storage, [integer(0)])), 1),
                    ],
                    None,
                ),
            ],
            None,
        ),
    ]
    loop = ir.For(
        index, integer(0), integer(12), integer(1), ir.ForKind.DEFAULT, body, {"multi_buffer_eligible": [storage], "num_stages": integer(1)}
    )
    value = _bind_target(
        _kernel(
            params,
            [
                _unit(allocate(storage), 1),
                annotation({"tl.buffer_version_mode": {storage: "counter"}}),
                _unit(loop, 1),
            ],
            {storage: 3},
            count=1,
        )
    )
    return value, storage, output, inner, inner_value, index


def _prepared_fields(value):
    (owner,) = objects(value, ir.For)
    result = dict(owner.annotations)
    for metadata in _metadata(owner):
        result.update(metadata)
    return result


@pytest.mark.parametrize("kind", ["pure", "mutable", "division", "arithmetic"])
def test_epoch_replay_requires_available_total_integer_predicates(kind):
    value, storage, _, inner, original_value, index = _snapshot_program(kind)
    before = ir.save_json(value)
    prepared = _run("PrepareMultiBuffer", value)
    assert ir.save_json(value) == before
    guard = _prepared_fields(prepared)["tl.storage_epoch_guard_map"][storage]
    expected = ir.prim.all(index % 2 == 0, index % 3 == 0) if kind == "pure" else ir.const(True, "bool")
    assert tvm_tirx.sym.Analyzer().can_prove_equal(guard, expected)
    (capture,) = [binding for binding in objects(prepared, ir.Bind) if binding.var.same_as(inner)]
    # Neither the original capture nor its evaluation position is rewritten.
    ir.assert_structural_equal(capture.value.args[0], original_value)


@pytest.mark.parametrize("kind", ["pure", "mutable"])
def test_replayed_epochs_execute_with_the_original_snapshot_values(kind):
    value, storage, output, _, _, _ = _snapshot_program(kind)
    value = _run("PrepareMultiBuffer", value)
    clock = _prepared_fields(value)["tl.multi_buffer_counter_map"][storage]
    root = value.body

    def observe(node):
        if not node.same_as(root):
            return node
        observation = _unit(ir.TensorStore(output, [integer(12)], ir.TensorLoad(clock, [integer(0)])), 1)
        return ir.SeqStmt([*node.seq, observation], node.loc)

    value = tvm_ffi.structural_map(value, (ir.SeqStmt, observe), order="pre")
    for name in ("ResolveCore", "InsertSync", "MaterializeMultiBuffer", "LowerScheduledTIR"):
        value = _run(name, value)
    _assert_schedule_consumed(value)
    value = _bind_target(function(list(value.params), _core(value, "VECTOR").body))
    for name in ("NormalizeBufferVersion", "RewriteAscendBufferVersionLayout"):
        value = _run(name, value)
    value = _run("AscendThreadSync", value, "shared")
    value = _apply(tirx.transform.LowerTensorLayouts(), value)
    value = _run("MergeUBAllocations", value, 32, False)
    arguments = [np.full(13, -99, dtype="int32")]
    if kind == "mutable":
        arguments.append(np.array([0], dtype="int32"))
    observed = execute(function(list(value.params), value.body), arguments, "scope-epoch-replay-" + kind)[0]
    expected = np.full(13, -99, dtype="int32")
    active = [0, 6] if kind == "pure" else [2, 4, 6, 8, 10]
    for index in active:
        expected[index] = index + 1
    expected[12] = 2 if kind == "pure" else 12
    np.testing.assert_array_equal(observed, expected)
