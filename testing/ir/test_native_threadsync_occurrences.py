"""Barrier participation belongs to a lexical occurrence of a shared node."""

import pytest
import tilelang
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target


@pytest.fixture(params=["shared", "shared.dyn"])
def scope(request):
    return request.param


def _sync(scope):
    return ir.Evaluate(tirx.op.gpu_storage_sync(scope))


def _launch(variable, extent, body):
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(extent)], [variable], {}, body)


def _function(variable, body, *, params=(), threads=128):
    return tirx.Function(list(params), [_launch(variable, threads, body)]).with_attr("target", Target("cuda"))


def _nodes(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="post")
    return found


def _barriers(value):
    return [
        call
        for call in _nodes(value, ir.Call)
        if isinstance(call.op, ir.Op) and call.op.name in {"tirx.gpu_storage_sync", "tl.storage_sync"}
    ]


def _apply(function, scope):
    assert tirx.analysis.check_well_formed(function)
    original = ir.save_json(function)
    try:
        result = tilelang.transform.ThreadSync(scope)(ir.IRModule({"main": function}))["main"]
    finally:
        assert ir.save_json(function) == original
    assert tirx.analysis.check_well_formed(result)
    return result


def _ids_and_counts(calls):
    assert all(call.op.name == "tl.storage_sync" and len(call.args) == 3 for call in calls)
    return [int(call.args[1]) for call in calls], [int(call.args[2]) for call in calls]


def test_shared_sync_has_distinct_then_and_else_participants(scope):
    tx = ir.Var("tx", "int32")
    shared = _sync(scope)
    branch = ir.If(tx < 64, shared, shared)
    assert branch.then_case.seq[0].same_as(branch.else_case.seq[0])
    result = _apply(_function(tx, branch), scope)
    ids, counts = _ids_and_counts(_barriers(result))
    assert len(ids) == 2 and ids[0] != ids[1]
    assert counts == [64, 64]


def test_shared_sequence_retains_order_and_equivalent_rectangle_ids(scope):
    tx = ir.Var("tx", "int32")
    shared = ir.SeqStmt([ir.Evaluate(ir.const(10)), _sync(scope), ir.Evaluate(ir.const(20))])
    branches = [ir.If(condition, shared, None) for condition in (tx < 64, tx >= 64, tx // 64 == 0)]
    assert all(branch.then_case.same_as(shared) for branch in branches)
    result = _apply(_function(tx, branches), scope)
    ids, counts = _ids_and_counts(_barriers(result))
    assert len(ids) == 3 and ids[0] == ids[2] and ids[0] != ids[1]
    assert counts == [64, 64, 64]
    for branch in _nodes(result, ir.If):
        statements = _nodes(branch.then_case, ir.Evaluate)
        assert len(statements) == 3
        assert int(statements[0].value) == 10 and int(statements[2].value) == 20
        assert statements[1].value.op.name == "tl.storage_sync"


def test_partial_occurrence_keeps_full_domain_siblings_unmodified(scope):
    tx = ir.Var("tx", "int32")
    shared = _sync(scope)
    result = _apply(_function(tx, [shared, ir.If(tx < 64, shared, None), shared]), scope)
    calls = _barriers(result)
    assert [call.op.name for call in calls] == ["tirx.gpu_storage_sync", "tl.storage_sync", "tirx.gpu_storage_sync"]
    assert int(calls[1].args[2]) == 64
    assert calls[0].same_as(shared.value) and calls[2].same_as(shared.value)


def test_empty_occurrence_does_not_remove_live_siblings(scope):
    tx = ir.Var("tx", "int32")
    shared = _sync(scope)
    result = _apply(_function(tx, [ir.If(tx < 0, shared, None), ir.If(tx < 64, shared, None), shared]), scope)
    branches = _nodes(result, ir.If)
    assert len(branches) == 2 and not _barriers(branches[0].then_case)
    calls = _barriers(result)
    assert [call.op.name for call in calls] == ["tl.storage_sync", "tirx.gpu_storage_sync"]
    assert int(calls[0].args[2]) == 64 and calls[1].same_as(shared.value)


def test_loop_occurrences_retain_distinct_thread_domains(scope):
    tx, iteration = ir.Var("tx", "int32"), ir.Var("iteration", "int32")
    shared = _sync(scope)
    loop = ir.For(
        iteration,
        ir.const(0),
        ir.const(2),
        ir.const(1),
        ir.ForKind.DEFAULT,
        [ir.If(tx < 64, shared, None), ir.If(tx >= 64, shared, None)],
    )
    result = _apply(_function(tx, loop), scope)
    (retained,) = _nodes(result, ir.For)
    assert retained.var.same_as(iteration) and int(retained.end) == 2
    ids, counts = _ids_and_counts(_barriers(retained.body))
    assert len(ids) == 2 and ids[0] != ids[1]
    assert counts == [64, 64]


def test_named_and_unrelated_evaluates_preserve_their_positions(scope):
    tx = ir.Var("tx", "int32")
    shared = _sync(scope)
    named = ir.Evaluate(ir.Call("tl.storage_sync", [ir.StringImm(scope), 11, 64]))
    body = [ir.Evaluate(ir.const(10)), ir.If(tx < 64, shared, None), named, ir.If(tx >= 64, shared, None), ir.Evaluate(ir.const(20))]
    assert len(_nodes(_function(tx, body), ir.Evaluate)) == 5
    result = _apply(_function(tx, body), scope)
    calls = _barriers(result)
    ids, counts = _ids_and_counts(calls)
    assert len(ids) == 3 and ids[0] != ids[2] and ids[1] == 11
    assert counts == [64, 64, 64] and calls[1].same_as(named.value)
    statements = _nodes(result, ir.Evaluate)
    assert len(statements) == 5 and int(statements[0].value) == 10 and int(statements[-1].value) == 20


def test_sibling_launch_regions_use_their_own_domains(scope):
    first, second = ir.Var("first", "int32"), ir.Var("second", "int32")
    shared = _sync(scope)
    function = tirx.Function([], [_launch(first, 64, shared), _launch(second, 128, ir.If(second < 64, shared, None))])
    result = _apply(function.with_attr("target", Target("cuda")), scope)
    calls = _barriers(result)
    assert [call.op.name for call in calls] == ["tirx.gpu_storage_sync", "tl.storage_sync"]
    assert calls[0].same_as(shared.value) and int(calls[1].args[2]) == 64


def test_block_uniform_occurrence_remains_a_full_domain_barrier(scope):
    tx, condition = ir.Var("tx", "int32"), ir.Var("condition", "bool")
    shared = _sync(scope)
    result = _apply(_function(tx, [ir.If(condition, shared, None), ir.If(tx < 64, shared, None)], params=[condition]), scope)
    calls = _barriers(result)
    assert [call.op.name for call in calls] == ["tirx.gpu_storage_sync", "tl.storage_sync"]
    assert calls[0].same_as(shared.value) and int(calls[1].args[2]) == 64


@pytest.mark.parametrize("kind", ["partial_warp", "holes", "runtime"])
def test_valid_shared_occurrence_does_not_hide_an_invalid_sibling(scope, kind):
    tx = ir.Var("tx", "int32")
    flags = ir.Var("flags", tirx.TensorType([128], "bool", scope="global", layout="default"))
    condition = {"partial_warp": tx < 48, "holes": tx % 64 < 32, "runtime": ir.TensorLoad(flags, [tx])}[kind]
    expected = {"partial_warp": "warp-multiple|warp-uniform", "holes": "thread rectangle", "runtime": "runtime-dependent participation"}[
        kind
    ]
    shared = _sync(scope)
    function = _function(tx, [ir.If(tx < 64, shared, None), ir.If(condition, shared, None)], params=[flags])
    with pytest.raises((RuntimeError, ValueError), match=expected):
        _apply(function, scope)
