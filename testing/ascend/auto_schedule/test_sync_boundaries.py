"""Synchronization across optional loops and unequal guard/epoch domains."""

from tvm_tirx import ir
from testing.ascend._ir import sblock, tensor

import pytest
import tvm_tirx as tvm
from tilelang.ascend import transform
from testing.ascend._ir import copy, kernel, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule._finite_protocol import assert_finite_protocol


def _flag_sites(mod, operation, pipe):
    for stmt, parents in statements(mod):
        if isinstance(stmt, ir.Evaluate) and isinstance(stmt.value, ir.Call):
            call = stmt.value
            if call.op.name == "tl.ascend_" + operation + "_flag" and call.args[0].value == pipe:
                yield call, parents


@pytest.mark.parametrize("counter_mode", [False, True], ids=["iteration-clock", "counter-clock"])
def test_acquire_dominates_possibly_empty_consumer_loop(counter_mode):
    a = tensor((4, 64), "float32", name="A")
    c = tensor((4, 64), "float32", name="C")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    epoch = tensor((1,), "int32", name="epoch", scope="local.var")
    i, j, count = (ir.Var(name, "int32") for name in ("i", "j", "count"))
    child = ir.For(
        j, ir.const(0), ir.const(0) + count, ir.const(1), ir.ForKind.DEFAULT, unit(copy(ub, c, dst_indices=[i, 0], dst_shape=[1, 64]))
    )
    tasks = [unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64])), unit(child, core=None)]
    annotations = {"multi_buffer_eligible": [ub], "tl.storage_epoch_guard_map": {ub: ir.const(True, "bool")}}
    if counter_mode:
        annotations["tl.multi_buffer_counter_map"] = {ub: epoch}
        tasks.append(unit(ir.TensorStore(epoch, [0], epoch[0] + 1)))
    owner = unit(
        ir.For(i, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, seq(*tasks), annotations=annotations), core=None
    )
    body = seq(unit(ir.TensorStore(epoch, [0], 0)), owner) if counter_mode else owner
    before = kernel(
        body,
        buffers=[ub] + ([epoch] if counter_mode else []),
        params=[a, c, count],
        annotations={"tl.buffer_versions_map": {ub: 2}},
        allocation_stage=0,
    )
    after = transform.InsertSync()(before)
    waits = list(_flag_sites(after, "wait", "MTE2_MTE3"))
    assert waits
    # Even count=0 must consume the producer's token. An acquire inside j
    # would leave an unmatched notification and corrupt the next ring reuse.
    assert all(not any(isinstance(parent, ir.For) and parent.var.same_as(j) for parent in parents) for _, parents in waits)
    assert all(any(isinstance(parent, ir.For) and parent.var.same_as(i) for parent in parents) for _, parents in waits)


@pytest.mark.parametrize(
    "nested, second_stage",
    [(False, 0), (True, 0), (True, 1), (True, 2)],
    ids=["sibling-owners", "repeated-siblings", "uncovered-stage-distance", "wider-stage-distance"],
)
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_missing_counter_channel_keeps_lexical_handshake(nested, second_stage, versions):
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    epoch = tensor((1,), "int32", name="epoch", scope="local.var")
    owners = []
    for index in range(2):
        i = ir.Var(f"owner{index}", "int32")
        write = ir.TensorStore(ub, [0], ir.const(index + 1, "float32"))
        # SIMD work and scalar work have no common counter channel. A VF
        # marker is sufficient for pipe analysis; no SIMD lowering is needed.
        if index == 0:
            write = sblock("SIMD_VF", write)
        body = seq(unit(write), unit(ir.TensorStore(epoch, [0], epoch[0] + 1)))
        annotations = {
            "multi_buffer_eligible": [ub],
            "tl.multi_buffer_counter_map": {ub: epoch},
            "tl.storage_epoch_guard_map": {ub: ir.const(True, "bool")},
        }
        owners.append(
            unit(
                ir.For(i, ir.const(0), ir.const(0) + ir.const(1), ir.const(1), ir.ForKind.DEFAULT, body, annotations=annotations),
                core=None,
                stage=second_stage if index else 0,
            )
        )
    body = seq(*owners)
    if nested:
        outer = ir.Var("outer", "int32")
        body = unit(ir.For(outer, ir.const(0), ir.const(0) + ir.const(3), ir.const(1), ir.ForKind.DEFAULT, body), core=None)
    before = kernel(
        seq(unit(ir.TensorStore(epoch, [0], 0)), body),
        buffers=[ub, epoch],
        annotations={"tl.buffer_versions_map": {ub: versions}},
        allocation_stage=0,
    )
    if second_stage:
        with pytest.raises((ValueError, RuntimeError), match="owner exclusion at logical distance") as error:
            transform.InsertSync()(before)
        message = str(error.value)
        assert f"logical distance {-second_stage} (paired distance {1 + second_stage})" in message
        return
    after = transform.InsertSync()(before)
    forward_ids = []
    analyzer = tvm.sym.Analyzer()
    for operation in ("set", "wait"):
        forward = list(_flag_sites(after, operation, "V_S"))
        identifiers = [analyzer.simplify(call.args[1]) for call, _ in forward]
        forward_ids.append([int(identifier) for identifier in identifiers])
        assert forward and all(isinstance(identifier, ir.IntImm) for identifier in identifiers)
        if nested:
            reverse = list(_flag_sites(after, operation, "S_V"))
            assert reverse
            assert any(not any(isinstance(parent, ir.For) for parent in parents) for _, parents in reverse)
    assert forward_ids[0] == forward_ids[1]
    if nested:
        assert sorted(int(analyzer.simplify(call.args[1])) for call, _ in _flag_sites(after, "set", "S_V")) == sorted(
            int(analyzer.simplify(call.args[1])) for call, _ in _flag_sites(after, "wait", "S_V")
        )


def test_narrow_child_handshake_cannot_replace_wider_parent_dependency():
    a = tensor((4, 128), "float32", name="A")
    c = tensor((4, 128), "float32", name="C")
    ub = tensor((128,), "float32", name="ub", scope="shared.dyn")
    p, q = ir.Var("p", "bool"), ir.Var("q", "bool")
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")

    def load(offset, guard):
        return unit(copy(a, ub, src_indices=[i, offset], src_shape=[1, 64], dst_indices=[offset], dst_shape=[64]), guard=guard)

    def store(offset, guard):
        return unit(copy(ub, c, src_indices=[offset], src_shape=[64], dst_indices=[i, offset], dst_shape=[1, 64]), guard=guard)

    child = ir.For(
        j,
        ir.const(0),
        ir.const(0) + ir.const(1),
        ir.const(1),
        ir.ForKind.DEFAULT,
        seq(load(64, p), store(64, p)),
        annotations={"tl.storage_epoch_guard_map": {ub: p}},
    )
    body = seq(load(0, q), unit(child, core=None), store(0, q))
    loop = ir.For(
        i,
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        body,
        annotations={"tl.storage_epoch_guard_map": {ub: ir.prim.Or(p, q)}},
    )
    before = kernel(unit(loop, core=None), buffers=[ub], params=[a, c, p, q], allocation_stage=0)
    after = transform.InsertSync()(before)
    # A one-trip child handshake can execute even when its guarded work does
    # not. Check all p/q paths and actual completion, including the parent.
    assert_finite_protocol(before, after, "parent_child")


def test_branch_switch_acquires_previous_iteration_before_nested_overwrite():
    a = tensor((4, 64), "float32", name="A")
    c = tensor((4, 64), "float32", name="C")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    out = tensor((64,), "float32", name="out", scope="shared.dyn")
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    phase = ir.Var("phase", "bool")

    def protocol(row, guard=None):
        compute = sblock("SIMD_VF", ir.TensorStore(out, [0], ub[0]))
        return seq(
            unit(copy(a, ub, src_indices=[row, 0], src_shape=[1, 64]), guard=guard),
            unit(compute, guard=guard),
            unit(copy(out, c, dst_indices=[row, 0], dst_shape=[1, 64]), guard=guard),
        )

    child = ir.For(j, ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, protocol(j))
    # The later execution phase occurs first in source order. A wait placed
    # only in the else branch cannot protect the transition into the child.
    body = seq(unit(ir.Bind(phase, i >= 2)), unit(child, core=None, guard=phase), protocol(i, ir.prim.Not(phase)))
    before = kernel(
        unit(ir.For(i, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, body), core=None),
        buffers=[ub, out],
        params=[a, c],
        allocation_stage=0,
    )
    after = transform.InsertSync()(before)
    # The prior vector reader may complete through a V->MTE3->MTE2 chain.
    # Its completion must still precede every nested overwrite and token reuse.
    assert_finite_protocol(before, after, "branch_switch")
