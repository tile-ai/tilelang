"""Inferred synchronization is attached to a lexical statement occurrence."""

import pytest
import tilelang
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target


@pytest.fixture(params=["shared", "shared.dyn"])
def scope(request):
    return request.param


def _nodes(value, kind):
    nodes = []
    tvm_ffi.structural_walk(value, (kind, nodes.append))
    return nodes


def _barriers(value):
    return [
        call
        for call in _nodes(value, ir.Call)
        if isinstance(call.op, ir.Op) and call.op.name in {"tirx.gpu_storage_sync", "tl.storage_sync"}
    ]


def _tensor(name, scope):
    return ir.Var(name, tirx.TensorType((64,), "int32", scope=scope, layout="default"))


def _function(shared, output, tx, body):
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(64)], [tx], {}, body)
    return tirx.Function([output], [ir.Bind(shared, ir.Call("tirx.alloc_tensor", [], ty_args=[shared.ty])), launch]).with_attr(
        "target", Target("cuda")
    )


def _apply(function, scope):
    assert tirx.analysis.check_well_formed(function)
    serialized = ir.save_json(function)
    try:
        result = tilelang.transform.ThreadSync(scope)(ir.IRModule({"main": function}))["main"]
    finally:
        assert ir.save_json(function) == serialized
    assert tirx.analysis.check_well_formed(result)
    assert len(result.params) == len(function.params)
    assert all(left.same_as(right) for left, right in zip(result.params, function.params))
    return result


def _accesses(kind, shared, output, tx):
    if kind == "reader":
        return ir.TensorStore(output, [tx], ir.TensorLoad(shared, [tx])), ir.TensorStore(shared, [(tx + 1) % 64], ir.const(1))
    return ir.TensorStore(shared, [tx], ir.const(1)), ir.TensorStore(output, [tx], ir.TensorLoad(shared, [(tx + 1) % 64]))


def _initialized(shared, tx, scope):
    return [ir.TensorStore(shared, [tx], ir.const(0)), ir.Evaluate(tirx.op.gpu_storage_sync(scope))]


@pytest.mark.parametrize("kind", ["reader", "writer"])
@pytest.mark.parametrize("wrapper", ["statement", "sequence", "for", "while"])
@pytest.mark.parametrize("first_count", [48, 64])
def test_later_conflict_does_not_prefence_an_earlier_shared_occurrence(scope, kind, wrapper, first_count):
    shared, output, tx = _tensor("shared", scope), _tensor("output", "global"), ir.Var("tx", "int32")
    reused, between = _accesses(kind, shared, output, tx)
    if wrapper == "sequence":
        statement = ir.SeqStmt([ir.Evaluate(ir.const(10)), reused, ir.Evaluate(ir.const(20))])
    elif wrapper == "for":
        row = ir.Var("row", "int32")
        statement = ir.For(row, ir.const(0), ir.const(2), ir.const(1), ir.ForKind.DEFAULT, [reused])
    elif wrapper == "while":
        statement = ir.While(ir.const(True, "bool"), [reused, ir.Break()])
    else:
        statement = reused
    later = statement
    if wrapper == "for":
        # Native TIR requires unique binders. Share the statement body across
        # two distinct loops; reusing the For itself would redefine its Var.
        later = ir.For(ir.Var("later_row", "int32"), statement.begin, statement.end, statement.step, statement.kind, statement.body)
    first = ir.If(tx < first_count, statement, None)
    function = _function(shared, output, tx, [*_initialized(shared, tx, scope), first, between, later])
    assert sum(node.same_as(reused) for node in _nodes(function, ir.TensorStore)) == 2
    if wrapper == "for":
        nodes = _nodes(function, ir.For)
        assert len(nodes) == 2 and not nodes[0].var.same_as(nodes[1].var)
        assert nodes[0].body.same_as(nodes[1].body)
    elif wrapper == "while":
        nodes = _nodes(function, ir.While)
        assert len(nodes) == 2 and nodes[0].same_as(nodes[1])
    # Both synchronization edges are real: each intervening access to index
    # (thread+1)%64 overlaps a different thread's ordinary index. The first
    # occurrence is already ordered after initialization by the explicit sync.
    assert all((thread + 1) % 64 != thread for thread in range(64))
    result = _apply(function, scope)
    assert len(_barriers(result)) == 3
    branches = _nodes(result, ir.If)
    assert len(branches) == 1 and not _barriers(branches[0])
    assert len(_nodes(result, ir.TensorStore)) == 4
    assert all(call.op.name == "tirx.gpu_storage_sync" and len(call.args) == 1 for call in _barriers(result))
    if wrapper == "sequence":
        constants = [int(node.value) for node in _nodes(result, ir.Evaluate) if isinstance(node.value, ir.IntImm)]
        assert constants == [10, 20, 10, 20]
    if wrapper in ("for", "while"):
        loops = _nodes(result, ir.For if wrapper == "for" else ir.While)
        assert len(loops) == 2 and all(not _barriers(loop) for loop in loops)


def test_unentered_loop_body_does_not_shift_later_shared_occurrences(scope):
    shared, output, tx = _tensor("shared", scope), _tensor("output", "global"), ir.Var("tx", "int32")
    reused, writer = _accesses("reader", shared, output, tx)
    row = ir.Var("row", "int32")
    dead = ir.For(row, ir.const(0), ir.const(0), ir.const(1), ir.ForKind.DEFAULT, [reused])
    function = _function(shared, output, tx, [*_initialized(shared, tx, scope), dead, writer, reused])
    assert sum(node.same_as(reused) for node in _nodes(function, ir.TensorStore)) == 2
    result = _apply(function, scope)
    assert len(_barriers(result)) == 2
    (retained,) = _nodes(result, ir.For)
    assert int(retained.begin) == int(retained.end) == 0
    assert not _barriers(retained)
    assert len(_nodes(result, ir.TensorStore)) == 4


@pytest.mark.parametrize("kind", ["reader", "writer"])
def test_required_intra_branch_conflict_still_rejects_partial_warp(scope, kind):
    shared, output, tx = _tensor("shared", scope), _tensor("output", "global"), ir.Var("tx", "int32")
    reused, between = _accesses(kind, shared, output, tx)
    first = ir.If(tx < 48, [between, reused], None)
    function = _function(shared, output, tx, [*_initialized(shared, tx, scope), reused, first])
    assert sum(node.same_as(reused) for node in _nodes(function, ir.TensorStore)) == 2
    with pytest.raises(RuntimeError, match="not warp-uniform"):
        _apply(function, scope)
