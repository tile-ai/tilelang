"""Control normalization snapshots mutable memory before scheduling splits tasks."""

import pytest
from tilelang.ascend import transform
from tvm_tirx import ir, tirx
from testing.ascend._ir import allocated_buffer, kernel, kernel_body, nodes, tensor


@pytest.mark.parametrize("control", ["condition", "loop-bound"])
def test_control_read_is_snapshotted_before_body_mutates_it(control):
    pred = tensor((1,), "int32", name="pred", scope="shared.dyn")
    write = ir.TensorStore(pred, [0], ir.const(0))
    if control == "condition":
        body = ir.If(ir.TensorLoad(pred, [0]) > 0, [write], [ir.TensorStore(pred, [0], ir.const(1))])
    else:
        body = ir.For(ir.Var("i", "int32"), ir.const(0), ir.TensorLoad(pred, [0]), ir.const(1), ir.ForKind.DEFAULT, [write])
    before = kernel(body, buffers=[pred])
    after = transform.NormalizeControlFlowForSchedule()(before)
    (snapshot,) = [binding for binding in nodes(after, ir.Bind) if not tirx.is_tensor_var(binding.var)]
    (load,) = nodes(snapshot.value, ir.TensorLoad)
    assert load.source.same_as(pred)
    if control == "condition":
        (branch,) = nodes(after, ir.If)
        assert branch.condition.same_as(snapshot.var)
        assert isinstance(branch.then_case.seq[0], ir.TensorStore) and isinstance(branch.else_case.seq[0], ir.TensorStore)
    else:
        (loop,) = nodes(after, ir.For)
        assert loop.end.same_as(snapshot.var)
    allocation, before_snapshot, after_snapshot = kernel_body(after).seq
    assert allocation.var.same_as(allocated_buffer(after, "pred"))
    assert before_snapshot.same_as(snapshot)
    ir.assert_structural_equal(nodes(after_snapshot, ir.TensorStore)[0], write)
