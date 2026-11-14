"""Synchronization across optional loops and unequal guard/epoch domains."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import copy, kernel, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


def _flag_sites(mod, operation, pipe):
    for stmt, parents in statements(mod):
        if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            call = stmt.value
            if call.op.name == "tl.ascend_" + operation + "_flag" and call.args[0].value == pipe:
                yield call, parents


@pytest.mark.parametrize("counter_mode", [False, True], ids=["iteration-clock", "counter-clock"])
def test_acquire_dominates_possibly_empty_consumer_loop(counter_mode):
    a = tirx.decl_buffer((4, 64), "float32", name="A")
    c = tirx.decl_buffer((4, 64), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    epoch = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var")
    i, j, count = (tirx.Var(name, "int32") for name in ("i", "j", "count"))
    child = tirx.For(j, 0, count, tirx.ForKind.SERIAL, unit(copy(ub, c, dst_indices=[i, 0], dst_shape=[1, 64])))
    tasks = [unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64])), unit(child, core=None)]
    annotations = {"multi_buffer_eligible": [ub.data], "tl.storage_epoch_guard_map": {ub.data: tirx.const(True, "bool")}}
    if counter_mode:
        annotations["tl.multi_buffer_counter_map"] = {ub.data: epoch}
        tasks.append(unit(tirx.BufferStore(epoch, epoch[0] + 1, [0])))
    owner = unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, seq(*tasks), annotations=annotations), core=None)
    body = seq(unit(tirx.BufferStore(epoch, 0, [0])), owner) if counter_mode else owner
    before = kernel(
        body, buffers=[ub] + ([epoch] if counter_mode else []), params=[a, c, count], annotations={"tl.buffer_versions_map": {ub.data: 2}}
    )
    after = transform.InsertSync()(before)
    waits = list(_flag_sites(after, "wait", "MTE2_MTE3"))
    assert waits
    # Even count=0 must consume the producer's token. An acquire inside j
    # would leave an unmatched notification and corrupt the next ring reuse.
    assert all(not any(isinstance(parent, tirx.For) and parent.loop_var.same_as(j) for parent in parents) for _, parents in waits)
    assert all(any(isinstance(parent, tirx.For) and parent.loop_var.same_as(i) for parent in parents) for _, parents in waits)


@pytest.mark.parametrize(
    "nested, second_stage",
    [(False, 0), (True, 0), (True, 1), (True, 2)],
    ids=["sibling-owners", "repeated-siblings", "uncovered-stage-distance", "wider-stage-distance"],
)
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_missing_counter_channel_keeps_lexical_handshake(nested, second_stage, versions):
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    epoch = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var")
    owners = []
    for index in range(2):
        i = tirx.Var(f"owner{index}", "int32")
        write = tirx.BufferStore(ub, tirx.const(index + 1, "float32"), [0])
        # SIMD work and scalar work have no common counter channel. A VF
        # marker is sufficient for pipe analysis; no SIMD lowering is needed.
        if index == 0:
            write = tirx.SBlock([], [], [], "SIMD_VF", write)
        body = seq(unit(write), unit(tirx.BufferStore(epoch, epoch[0] + 1, [0])))
        annotations = {
            "multi_buffer_eligible": [ub.data],
            "tl.multi_buffer_counter_map": {ub.data: epoch},
            "tl.storage_epoch_guard_map": {ub.data: tirx.const(True, "bool")},
        }
        owners.append(
            unit(tirx.For(i, 0, 1, tirx.ForKind.SERIAL, body, annotations=annotations), core=None, stage=second_stage if index else 0)
        )
    body = seq(*owners)
    if nested:
        outer = tirx.Var("outer", "int32")
        body = unit(tirx.For(outer, 0, 3, tirx.ForKind.SERIAL, body), core=None)
    before = kernel(
        seq(unit(tirx.BufferStore(epoch, 0, [0])), body), buffers=[ub, epoch], annotations={"tl.buffer_versions_map": {ub.data: versions}}
    )
    if second_stage:
        with pytest.raises(tvm.error.InternalError, match="owner exclusion at logical distance") as error:
            transform.InsertSync()(before)
        message = str(error.value)
        assert f"logical distance {-second_stage} (paired distance {1 + second_stage})" in message
        return
    after = transform.InsertSync()(before)
    forward_ids = []
    for operation in ("set", "wait"):
        forward = list(_flag_sites(after, operation, "V_S"))
        forward_ids.append([int(call.args[1]) for call, _ in forward])
        assert forward and all(isinstance(call.args[1], tirx.IntImm) for call, _ in forward)
        if nested:
            reverse = list(_flag_sites(after, operation, "S_V"))
            assert reverse
            assert any(not any(isinstance(parent, tirx.For) for parent in parents) for _, parents in reverse)
    assert forward_ids[0] == forward_ids[1]
    if nested:
        assert sorted(int(call.args[1]) for call, _ in _flag_sites(after, "set", "S_V")) == sorted(
            int(call.args[1]) for call, _ in _flag_sites(after, "wait", "S_V")
        )


def test_narrow_child_handshake_cannot_replace_wider_parent_dependency():
    a = tirx.decl_buffer((4, 128), "float32", name="A")
    c = tirx.decl_buffer((4, 128), "float32", name="C")
    ub = tirx.decl_buffer((128,), "float32", name="ub", scope="shared.dyn")
    p, q = tirx.Var("p", "bool"), tirx.Var("q", "bool")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")

    def load(offset, guard):
        return unit(copy(a, ub, src_indices=[i, offset], src_shape=[1, 64], dst_indices=[offset], dst_shape=[64]), guard=guard)

    def store(offset, guard):
        return unit(copy(ub, c, src_indices=[offset], src_shape=[64], dst_indices=[i, offset], dst_shape=[1, 64]), guard=guard)

    child = tirx.For(j, 0, 1, tirx.ForKind.SERIAL, seq(load(64, p), store(64, p)), annotations={"tl.storage_epoch_guard_map": {ub.data: p}})
    body = seq(load(0, q), unit(child, core=None), store(0, q))
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"tl.storage_epoch_guard_map": {ub.data: tirx.Or(p, q)}})
    before = kernel(unit(loop, core=None), buffers=[ub], params=[a, c, p, q])
    after = transform.InsertSync()(before)
    # Evaluate the structural guards for p=False,q=True: a parent handshake
    # must still execute even though all inner work (and its handshake) skips.
    for operation in ("set", "wait"):
        covering = []
        for call, parents in _flag_sites(after, operation, "MTE2_MTE3"):
            if any(isinstance(parent, tirx.For) and parent.loop_var.same_as(j) for parent in parents):
                continue
            guards = [parent.condition for parent in parents if isinstance(parent, tirx.IfThenElse)]
            active = tirx.all(*guards) if guards else tirx.const(True, "bool")
            active = tirx.stmt_functor.substitute(active, {p: tirx.const(False, "bool"), q: tirx.const(True, "bool")})
            if tvm.arith.Analyzer().can_prove(active):
                covering.append(call)
        assert covering, "The child-only ordering does not cover the parent access"


def test_branch_switch_acquires_previous_iteration_before_nested_overwrite():
    a = tirx.decl_buffer((4, 64), "float32", name="A")
    c = tirx.decl_buffer((4, 64), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((64,), "float32", name="out", scope="shared.dyn")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    phase = tirx.Var("phase", "bool")

    def protocol(row, guard=None):
        compute = tirx.SBlock([], [], [], "SIMD_VF", tirx.BufferStore(out, ub[0], [0]))
        return seq(
            unit(copy(a, ub, src_indices=[row, 0], src_shape=[1, 64]), guard=guard),
            unit(compute, guard=guard),
            unit(copy(out, c, dst_indices=[row, 0], dst_shape=[1, 64]), guard=guard),
        )

    child = tirx.For(j, 0, 2, tirx.ForKind.SERIAL, protocol(j))
    # The later execution phase occurs first in source order. A wait placed
    # only in the else branch cannot protect the transition into the child.
    body = seq(unit(tirx.Bind(phase, i >= 2)), unit(child, core=None, guard=phase), protocol(i, tirx.Not(phase)))
    before = kernel(unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body), core=None), buffers=[ub, out], params=[a, c])
    after = transform.InsertSync()(before)
    ordered = list(statements(after))
    child_position = next(index for index, (stmt, _) in enumerate(ordered) if isinstance(stmt, tirx.For) and stmt.loop_var.same_as(j))
    waits = []
    for call, parents in _flag_sites(after, "wait", "V_MTE2"):
        if any(isinstance(parent, tirx.For) and parent.loop_var.same_as(j) for parent in parents):
            continue
        position = next(index for index, (stmt, _) in enumerate(ordered) if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(call))
        if position < child_position and any(isinstance(parent, tirx.For) and parent.loop_var.same_as(i) for parent in parents):
            waits.append(call)
    assert waits, "The cross-iteration acquire must dominate the nested overwrite"
    compute_position = max(
        index for index, (stmt, _) in enumerate(ordered) if isinstance(stmt, tirx.SBlock) and stmt.name_hint == "SIMD_VF"
    )
    releases = []
    for call, parents in _flag_sites(after, "set", "V_MTE2"):
        position = next(index for index, (stmt, _) in enumerate(ordered) if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(call))
        if position > compute_position and any(isinstance(parent, tirx.For) and parent.loop_var.same_as(i) for parent in parents):
            releases.append(call)
    assert any(tvm.ir.structural_equal(wait.args[1], release.args[1]) for wait in waits for release in releases)
