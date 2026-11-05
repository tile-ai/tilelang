"""Shared storage stays live until its own committed TMA reads have finished."""

import pytest
from tvm_tirx import ir

from testing.python.transform.test_tilelang_native_shared_arena import (
    allocate,
    function,
    merge,
    offset,
    store,
    tensor,
)


def _call(name, args=()):
    return ir.Evaluate(ir.Call(name, list(args), ty_args=[ir.PrimType("void")]))


def _issue(source, destination):
    return _call("tl.tma_store", [destination.data, source.data, ir.const(64)])


def _commit():
    return _call("tl.tma_store_arrive")


def _wait(count=0, read_only=True):
    return _call("tl.tma_store_wait", [ir.const(count), ir.const(read_only, "bool")])


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("scope", ["sequence", "branch", "loop", "lexical", "explicit_lexical"])
def test_wait_zero_finishes_reads_before_reuse(read_only, scope):
    a, b = tensor("a"), tensor("b")
    output = tensor("output", scope="global")
    predicate, index = ir.Var("predicate", "bool"), ir.Var("index", "int32")
    body = ir.SeqStmt([_issue(a, output), _commit(), _wait(read_only=read_only)])
    if scope == "branch":
        body = ir.If(predicate, body, None)
    elif scope == "loop":
        body = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, body)
    statements = [allocate(a), body, allocate(b), store(b)]
    if scope in ("lexical", "explicit_lexical"):
        attrs = {"explicit_alloc_scope": True} if scope == "explicit_lexical" else {}
        statements = [ir.RegionStmt("tl.lexical_scope", [], [], attrs, statements)]
    original = function(statements, [output, predicate])
    before = ir.save_json(original)
    result = merge(original)
    assert offset(result, a) == offset(result, b)
    assert ir.save_json(original) == before


@pytest.mark.parametrize(
    "protocol",
    ["no_commit", "no_wait", "partial_wait", "later_uncommitted", "conditional_wait", "loop_wait", "sibling_branch"],
)
def test_incomplete_or_different_scope_wait_does_not_release_storage(protocol):
    a, b = tensor("a"), tensor("b")
    output = tensor("output", scope="global")
    predicate, index = ir.Var("predicate", "bool"), ir.Var("index", "int32")
    if protocol == "no_commit":
        operations = [_issue(a, output), _wait()]
    elif protocol == "no_wait":
        operations = [_issue(a, output), _commit()]
    elif protocol == "partial_wait":
        operations = [_issue(a, output), _commit(), _wait(1)]
    elif protocol == "later_uncommitted":
        operations = [_issue(a, output), _commit(), _issue(a, output), _wait()]
    elif protocol == "conditional_wait":
        operations = [_issue(a, output), _commit(), ir.If(predicate, [_wait()], None)]
    elif protocol == "loop_wait":
        operations = [_issue(a, output), _commit(), ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, [_wait()])]
    else:
        operations = [ir.If(predicate, [_issue(a, output), _commit()], [_wait()])]
    result = merge(function([allocate(a), *operations, allocate(b), store(b)], [output, predicate]))
    assert abs(offset(result, a) - offset(result, b)) >= 64


def test_storage_cannot_reuse_before_the_wait():
    a, b = tensor("a"), tensor("b")
    output = tensor("output", scope="global")
    result = merge(function([allocate(a), _issue(a, output), _commit(), allocate(b), store(b), _wait()], [output]))
    assert abs(offset(result, a) - offset(result, b)) >= 64


def test_wait_zero_does_not_complete_a_later_uncommitted_group():
    a, pending, b = tensor("a"), tensor("pending"), tensor("b")
    output = tensor("output", scope="global")
    result = merge(
        function(
            [allocate(a), allocate(pending), _issue(a, output), _commit(), _issue(pending, output), _wait(), allocate(b), store(b)],
            [output],
        )
    )
    assert offset(result, a) == offset(result, b)
    assert abs(offset(result, pending) - offset(result, b)) >= 64


def test_completed_tma_reads_do_not_release_an_unmatched_cp_async():
    a, b = tensor("a"), tensor("b")
    output = tensor("output", scope="global")
    result = merge(
        function(
            [
                allocate(a),
                _call("tl.ptx_cp_async_bytes", [a.data, output.data, ir.const(4)]),
                _issue(a, output),
                _commit(),
                _wait(),
                allocate(b),
                store(b),
            ],
            [output],
        )
    )
    assert abs(offset(result, a) - offset(result, b)) >= 64


@pytest.mark.parametrize("reuse", [False, True])
def test_aliased_tma_source_and_explicit_reuse_policy(reuse):
    a, alias, b = tensor("a"), tensor("alias"), tensor("b")
    output = tensor("output", scope="global")
    declaration = ir.Call("tirx.decl_tensor", [a.data], ty_args=[alias.ty])
    result = merge(
        function(
            [allocate(a), ir.Bind(alias, declaration), _issue(alias, output), _commit(), _wait(), allocate(b), store(b)],
            [output],
        ),
        reuse=reuse,
    )
    assert (offset(result, a) == offset(result, b)) == reuse


def test_lexical_scope_exception_does_not_relax_worker_contexts():
    a, b = tensor("a"), tensor("b")
    output = tensor("output", scope="global")
    body = [allocate(a), _issue(a, output), _commit(), _wait(), allocate(b), store(b)]
    workers = ir.RegionStmt("tirx.parallel_launch", [], [], {}, body)
    result = merge(function([workers], [output]))
    assert abs(offset(result, a) - offset(result, b)) >= 64


def test_symbolic_wait_count_does_not_prove_completion():
    a, b = tensor("a"), tensor("b")
    output = tensor("output", scope="global")
    count = ir.Var("pending_groups", "int32")
    wait = _call("tl.tma_store_wait", [count, ir.const(True, "bool")])
    body = [allocate(a), _issue(a, output), _commit(), wait, allocate(b), store(b)]
    result = merge(function(body, [output, count]))
    assert abs(offset(result, a) - offset(result, b)) >= 64
