"""Control normalization snapshots mutable memory before scheduling splits tasks."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import kernel, nodes


@pytest.mark.parametrize("control", ["condition", "loop-bound"])
def test_control_read_is_snapshotted_before_body_mutates_it(control):
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope="shared.dyn")
    write = tirx.BufferStore(pred, 0, [0])
    if control == "condition":
        body = tirx.IfThenElse(pred[0] > 0, write, tirx.BufferStore(pred, 1, [0]))
    else:
        body = tirx.For(tirx.Var("i", "int32"), 0, pred[0], tirx.ForKind.SERIAL, write)
    before = kernel(body, buffers=[pred])
    after = transform.NormalizeControlFlowForSchedule()(before)
    (snapshot,) = nodes(after, tirx.Bind)
    (load,) = nodes(snapshot.value, tirx.BufferLoad)
    assert load.buffer.same_as(pred)
    if control == "condition":
        (branch,) = nodes(after, tirx.IfThenElse)
        assert branch.condition.same_as(snapshot.var)
        assert isinstance(branch.then_case, tirx.BufferStore) and isinstance(branch.else_case, tirx.BufferStore)
    else:
        (loop,) = nodes(after, tirx.For)
        assert loop.extent.same_as(snapshot.var)
    before_snapshot, after_snapshot = after["main"].body.block.body.seq
    assert before_snapshot.same_as(snapshot)
    tvm.ir.assert_structural_equal(nodes(after_snapshot, tirx.BufferStore)[0], write)
