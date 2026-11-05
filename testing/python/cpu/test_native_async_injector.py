"""Native cp.async injection preserves copy widths and synchronous boundaries.

These are compiler IR tests; GPU execution is covered by the CUDA workstream.
"""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx


def tensor(name, scope, dtype="float32"):
    return ir.Var(name, tirx.TensorType([16], dtype, scope=scope))


def calls(statement):
    result = []
    tvm_ffi.structural_walk(statement, (ir.Call, lambda call: result.append(call)), order="post")
    return [call for call in result if isinstance(call.op, ir.Op)]


def inject(statements, explicit=False):
    return tvm_ffi.get_global_func("tl.testing.InjectPTXAsyncCopy")(ir.SeqStmt(statements), explicit)


def copy(source, destination, value=None):
    return ir.TensorStore(destination, [ir.IntImm("int32", 0)], ir.TensorLoad(source, [ir.IntImm("int32", 0)]) if value is None else value)


def test_sync_copy_commits_and_waits_before_shared_consumer():
    source, shared, output = tensor("source", "global"), tensor("shared", "shared"), tensor("output", "global")
    result = inject([copy(source, shared), copy(shared, output)])
    assert result["injected"]
    names = [call.op.name for call in calls(result["statement"])]
    assert names[-3:] == ["tl.ptx_cp_async", "tl.ptx_commit_group", "tl.ptx_wait_group"]
    statements = list(result["statement"])
    assert isinstance(statements[-1], ir.TensorStore)
    cp = next(call for call in calls(result["statement"]) if call.op.name == "tl.ptx_cp_async")
    assert int(cp.args[2]) == 1
    cp.validate()


def test_explicit_async_has_no_implicit_commit_or_wait():
    source, shared = tensor("source", "global"), tensor("shared", "shared")
    result = inject([copy(source, shared)], explicit=True)
    names = [call.op.name for call in calls(result["statement"])]
    assert "tl.ptx_cp_async" in names
    assert "tl.ptx_commit_group" not in names and "tl.ptx_wait_group" not in names


@pytest.mark.parametrize("zero,expected", [(0.0, True), (-0.0, False), (7.0, False)])
def test_predicated_copy_only_zero_fills_positive_zero_bits(zero, expected):
    source, shared = tensor("source", "global"), tensor("shared", "shared")
    predicate = ir.Var("predicate", "bool")
    value = ir.Call("prim.if_then_else", [predicate, ir.TensorLoad(source, [0]), ir.FloatImm("float32", zero)])
    result = inject([copy(source, shared, value)])
    assert bool(result["injected"]) is expected
    if expected:
        cp = next(call for call in calls(result["statement"]) if call.op.name == "tl.ptx_cp_async")
        assert len(cp.args) == 4 and cp.args[3].same_as(predicate)
        cp.validate()


def test_noncontiguous_vectorized_copy_keeps_synchronous_load_store():
    source, shared = tensor("source", "global"), tensor("shared", "shared")
    v = ir.Var("v", "int32")
    body = ir.For(
        v,
        ir.IntImm("int32", 0),
        ir.IntImm("int32", 2),
        ir.IntImm("int32", 1),
        ir.ForKind.VECTORIZED,
        [ir.TensorStore(shared, [v], ir.TensorLoad(source, [v * 2]))],
    )
    result = inject([body])
    assert not result["injected"]
    assert not any(call.op.name.startswith("tl.ptx_") for call in calls(result["statement"]))
