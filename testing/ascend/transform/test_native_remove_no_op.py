"""Ascend no-op removal keeps memory, metadata, effects and loop termination."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.testing.packed import compile_packed_function


def tensor(name, shape=(8,), scope="global"):
    return ir.Var(name, tirx.TensorType(shape, "int32", scope=scope))


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def attribute(key, value, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def annotation(data):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(data)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def allocation(var, attrs=None):
    return ir.Bind(var, ir.Call("tirx.alloc_tensor", [], attrs=attrs, ty_args=[var.ty]))


def apply(params, body, dataflow=False):
    before = tirx.Function(params, body)
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    transform = tvm_ffi.get_global_func("tl.transform.AscendRemoveNoOp")()
    with tvm_tirx.transform.PassContext(config={"tl.AscendRemoveNoOp": {"use_dataflow_analysis": dataflow}}):
        after = transform(ir.IRModule({"main": before}))["main"]
        assert tirx.analysis.check_well_formed(after)
        for call in objects(after, ir.Call):
            call.validate()
        ir.assert_structural_equal(transform(ir.IRModule({"main": after}))["main"], after)
    assert ir.save_json(before) == saved
    return after


def test_dead_tensor_allocation_and_self_copy_are_removed():
    a = tensor("a", scope="local")
    result = apply([], [allocation(a), ir.TensorStore(a, [0], ir.TensorLoad(a, [0]))])
    assert len(result.body.seq) == 0


def test_unused_alias_chain_disappears_without_dropping_referenced_storage():
    a, b = tensor("a", scope="local"), tensor("b", scope="local")
    alias = ir.Bind(b, ir.Call("tirx.decl_tensor", [ir.Call("tirx.tensor_data_ptr", [a])], ty_args=[b.ty]))
    result = apply([], [allocation(a), alias])
    assert len(result.body.seq) == 0
    result = apply([], [allocation(a), alias, ir.TensorStore(b, [0], ir.IntImm("int32", 2))])
    assert len(objects(result, ir.Bind)) == 2


def test_same_data_pointer_different_views_are_not_assumed_same_element():
    a, b = tensor("a"), tensor("b")
    result = apply([a, b], [ir.TensorStore(a, [0], ir.TensorLoad(b, [0]))])
    assert len(objects(result, ir.TensorStore)) == 1


def test_metadata_only_allocation_use_is_retained():
    a = tensor("a", scope="local")
    result = apply([], [allocation(a), annotation({"layout": a})])
    assert len(objects(result, ir.Bind)) == 1
    assert any(call.op.name == "tl.annotation" for call in objects(result, ir.Call))


def test_volatile_self_copy_and_unused_load_are_retained():
    a = tensor("a")
    result = apply(
        [a],
        [
            attribute(
                "volatile_scope", ir.IntImm("int32", 1), [ir.TensorStore(a, [0], ir.TensorLoad(a, [0])), ir.Evaluate(ir.TensorLoad(a, [1]))]
            )
        ],
    )
    assert len(objects(result, ir.TensorStore)) == 1
    assert len(objects(result, ir.TensorLoad)) == 2


@pytest.mark.parametrize("dataflow", [False, True])
def test_side_effectful_call_and_condition_are_evaluated_once(dataflow):
    effect = tirx.call_extern("touch", ty="int32")
    result = apply([], [ir.If(effect > 0, [], None)], dataflow)
    assert len(objects(result, ir.If)) == 0
    assert sum(call.op.name == "tirx.call_extern" for call in objects(result, ir.Call)) == 1


def test_pure_empty_branch_removes_condition():
    x = ir.Var("x", "int32")
    assert len(apply([x], [ir.If(x > 0, [], None)]).body.seq) == 0


def test_empty_then_inverts_guard():
    a, x = tensor("a"), ir.Var("x", "int32")
    result = apply([a, x], [ir.If(x > 0, [], [ir.TensorStore(a, [0], x)])])
    branch = objects(result, ir.If)[0]
    assert branch.else_case is None
    ir.assert_structural_equal(branch.condition, ir.prim.Not(x > 0))


@pytest.mark.parametrize("step,expected", [(1, 0), (0, 1), (-1, 1)])
def test_empty_loop_removal_preserves_possible_nontermination(step, expected):
    i = ir.Var("i", "int32")
    loop = ir.For(i, ir.IntImm("int32", 0), ir.IntImm("int32", 4), ir.IntImm("int32", step), ir.ForKind.DEFAULT, [])
    assert len(objects(apply([], [loop]), ir.For)) == expected


def test_empty_range_negative_step_is_empty_under_native_less_than_condition():
    i = ir.Var("i", "int32")
    loop = ir.For(i, ir.IntImm("int32", 4), ir.IntImm("int32", 0), ir.IntImm("int32", -1), ir.ForKind.DEFAULT, [ir.Break()])
    assert len(objects(apply([], [loop]), ir.For)) == 0


def test_negative_async_wait_is_removed_but_body_preserved():
    a = tensor("a")
    result = apply(
        [a],
        [
            attribute(
                "async_wait_queue_scope",
                ir.IntImm("int32", 0),
                [attribute("async_wait_inflight_count", ir.IntImm("int32", -1), [ir.TensorStore(a, [0], ir.IntImm("int32", 3))])],
            )
        ],
    )
    assert not objects(result, ir.RegionStmt)
    assert len(objects(result, ir.TensorStore)) == 1


def test_debug_skip_removes_effectful_region_by_explicit_contract():
    a = tensor("a")
    result = apply([a], [attribute("pragma_debug_skip_region", ir.IntImm("int32", 1), [ir.TensorStore(a, [0], ir.IntImm("int32", 3))])])
    assert len(result.body.seq) == 0


def test_dataflow_option_removes_duplicate_and_overwritten_pure_stores():
    a = tensor("a")
    body = [ir.TensorStore(a, [0], ir.IntImm("int32", v)) for v in (2, 2, 3)]
    assert len(objects(apply([a], body), ir.TensorStore)) == 3
    result = apply([a], body, True)
    assert [int(store.value.value) for store in objects(result, ir.TensorStore)] == [3]


def test_dataflow_never_crosses_memory_read_or_potential_alias_write():
    a, b, output = tensor("a"), tensor("b"), tensor("output")
    x = ir.Var("x", "int32")
    body = [
        ir.TensorStore(a, [0], ir.IntImm("int32", 2)),
        ir.Bind(x, ir.TensorLoad(b, [0])),
        ir.TensorStore(a, [0], ir.IntImm("int32", 3)),
        ir.TensorStore(output, [0], x),
    ]
    assert len(objects(apply([a, b, output], body, True), ir.TensorStore)) == 3
    body = [
        ir.TensorStore(a, [0], ir.IntImm("int32", 2)),
        ir.TensorStore(b, [0], ir.IntImm("int32", 8)),
        ir.TensorStore(a, [0], ir.IntImm("int32", 2)),
    ]
    assert len(objects(apply([a, b], body, True), ir.TensorStore)) == 3


def test_native_numeric_effects_and_alias_dse(tmp_path):
    a, b, output = tensor("a"), tensor("b"), tensor("output")
    x = ir.Var("x", "int32")
    body = [
        ir.TensorStore(a, [0], ir.IntImm("int32", 2)),
        ir.Bind(x, ir.TensorLoad(b, [0])),
        ir.TensorStore(a, [0], ir.IntImm("int32", 3)),
        ir.TensorStore(output, [0], x),
    ]
    result = apply([a, b, output], body, True).with_attr("global_symbol", "main")
    executable = compile_packed_function(result)
    backing = tvm_tirx.runtime.tensor(np.arange(8, dtype="int32"))
    destination = tvm_tirx.runtime.tensor(np.full(8, -1, dtype="int32"))
    executable(backing, backing, destination)
    assert backing.numpy()[0] == 3
    assert destination.numpy()[0] == 2
