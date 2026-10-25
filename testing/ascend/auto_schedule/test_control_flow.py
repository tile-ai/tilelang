"""Control normalization snapshots mutable memory before scheduling splits tasks."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tilelang._compat import tirx
from testing.ascend._ir import kernel, nodes, root_scope, unseq


@pytest.mark.parametrize("control", ["condition", "loop-bound"])
def test_control_read_is_snapshotted_before_body_mutates_it(control):
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope="shared.dyn")
    write = tirx.BufferStore(pred, 0, [0])
    if control == "condition":
        body = tirx.IfThenElse(pred[0] > 0, write, tirx.BufferStore(pred, 1, [0]))
    else:
        body = tirx.For(tirx.Var("i", "int32"), 0, pred[0], 1, tirx.ForKind.DEFAULT, write)
    before = kernel(body, buffers=[pred])
    after = transform.NormalizeControlFlowForSchedule()(before)
    (snapshot,) = [binding for binding in nodes(after, tirx.Bind) if nodes(binding.value, tirx.BufferLoad)]
    (load,) = nodes(snapshot.value, tirx.BufferLoad)
    assert load.source.same_as(pred)
    if control == "condition":
        (branch,) = nodes(after, tirx.IfThenElse)
        assert branch.condition.same_as(snapshot.var)
        assert isinstance(unseq(branch.then_case), tirx.BufferStore) and isinstance(unseq(branch.else_case), tirx.BufferStore)
    else:
        (loop,) = nodes(after, tirx.For)
        assert loop.end.same_as(snapshot.var)
    _, before_snapshot, after_snapshot = root_scope(after).body.seq
    assert before_snapshot.same_as(snapshot)
    tvm.ir.assert_structural_equal(nodes(after_snapshot, tirx.BufferStore)[0], write)
