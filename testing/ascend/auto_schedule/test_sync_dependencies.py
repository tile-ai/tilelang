"""InsertSync protects RAW and storage reuse independently of scheduling."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import allocated_buffer, calls, copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit


def _events(mod, operation, pipe):
    return [call for call in calls(mod, "tl.ascend_" + operation + "_flag") if call.args[0].value == pipe]


@pytest.mark.parametrize("mode, owners, guarded", [("iteration", 1, False), ("counter", 1, True), ("counter", 2, True)])
def test_ring_flags_use_the_data_clock(mode, owners, guarded):
    before = copy_ring(mode=mode, owners=owners, guarded=guarded, prepared=True)
    after = transform.InsertSync()(before)
    clock = allocated_buffer(before, "epoch")[0] if mode == "counter" else nodes(before, tirx.For)[0].loop_var
    analyzer = tvm.arith.Analyzer()
    for pipe in ("MTE2_MTE3", "MTE3_MTE2"):
        for operation in ("set", "wait"):
            events = _events(after, operation, pipe)
            dynamic = [event.args[1] for event in events if not isinstance(event.args[1], tirx.IntImm)]
            assert len(dynamic) == owners
            for flag in dynamic:
                base = analyzer.simplify(flag - clock % 2)
                assert isinstance(base, tirx.IntImm), "The ring must advance with the storage clock"
            if operation == "set":
                released = dynamic
            else:
                assert all(analyzer.can_prove_equal(left, right) for left, right in zip(released, dynamic))
    # Ring reuse needs an initial token and a final drain for every slot. IDs
    # need only match each other, not a particular allocator numbering.
    initial = {int(event.args[1]) for event in _events(after, "set", "MTE3_MTE2") if isinstance(event.args[1], tirx.IntImm)}
    drained = {int(event.args[1]) for event in _events(after, "wait", "MTE3_MTE2") if isinstance(event.args[1], tirx.IntImm)}
    assert initial == drained and len(initial) == 2
    if guarded:
        guarded_events = []
        for stmt, parents in statements(after):
            if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
                continue
            if stmt.value.op.name not in ("tl.ascend_set_flag", "tl.ascend_wait_flag") or isinstance(stmt.value.args[1], tirx.IntImm):
                continue
            owner = next(parent for parent in reversed(parents) if isinstance(parent, tirx.For))
            assert any(
                isinstance(parent, tirx.IfThenElse) and analyzer.can_prove_equal(parent.condition, owner.loop_var % 2 == 0)
                for parent in parents
            )
            guarded_events.append(stmt)
        assert guarded_events


@pytest.mark.parametrize("guard_inside_task", [False, True], ids=["guarded-task", "atomic-internal-guard"])
def test_sync_respects_atomic_task_boundary(guard_inside_task):
    a = tirx.decl_buffer((64,), "float32", name="A")
    c = tirx.decl_buffer((64,), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    enabled = tirx.Var("enabled", "bool")
    tasks = [copy(a, ub), copy(ub, c)]
    if guard_inside_task:
        tasks = [unit(tirx.IfThenElse(enabled, task, None)) for task in tasks]
    else:
        tasks = [unit(task, guard=enabled) for task in tasks]
    before = kernel(seq(*tasks), buffers=[ub], params=[a, c, enabled])
    after = transform.InsertSync()(before)
    events = _events(after, "set", "MTE2_MTE3") + _events(after, "wait", "MTE2_MTE3")
    assert len(events) == 2
    for stmt, parents in statements(after):
        if isinstance(stmt, tirx.Evaluate) and any(stmt.value.same_as(event) for event in events):
            guards = [parent.condition for parent in parents if isinstance(parent, tirx.IfThenElse)]
            if guard_inside_task:
                assert not guards
            if guards:
                tvm.ir.assert_structural_equal(guards[-1], enabled)


def test_dma_issue_order_is_not_completion_order():
    a = tirx.decl_buffer((64,), "float32", name="A")
    c = tirx.decl_buffer((64,), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    # A DMA store must finish before MTE2 may overwrite its UB source.
    before = kernel(seq(unit(copy(ub, c)), unit(copy(a, ub))), buffers=[ub], params=[a, c])
    after = transform.InsertSync()(before)
    assert len(_events(after, "set", "MTE3_MTE2")) == 1
    assert len(_events(after, "wait", "MTE3_MTE2")) == 1


def test_manually_indexed_storage_keeps_the_lexical_clock_under_a_guard():
    a = tirx.decl_buffer((4, 64), "float32", name="A")
    c = tirx.decl_buffer((4, 64), "float32", name="C")
    ub = tirx.decl_buffer((2, 64), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    body = seq(
        unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64], dst_indices=[i % 2, 0], dst_shape=[1, 64]), guard=i % 2 == 0),
        unit(copy(ub, c, src_indices=[i % 2, 0], src_shape=[1, 64], dst_indices=[i, 0], dst_shape=[1, 64]), guard=i % 2 == 0),
    )
    before = kernel(
        unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body), core=None),
        buffers=[ub],
        params=[a, c],
        annotations={"tl.manual_multi_buffer": {ub.data: 2}},
    )
    after = transform.InsertSync()(before)
    dynamic = []
    for stmt, parents in statements(after):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if call.op.name not in ("tl.ascend_set_flag", "tl.ascend_wait_flag") or call.args[0].value != "MTE2_MTE3":
            continue
        assert not any(isinstance(parent, tirx.IfThenElse) for parent in parents)
        assert isinstance(tvm.arith.Analyzer().simplify(call.args[1] - i % 2), tirx.IntImm)
        dynamic.append(call)
    assert len(dynamic) == 2
    tvm.ir.assert_structural_equal(dynamic[0].args[1], dynamic[1].args[1])


@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint-rows", "same-row"])
def test_dma_row_writes_only_serialize_overlapping_regions(overlap):
    a = tirx.decl_buffer((2, 64), "float32", name="A")
    out = tirx.decl_buffer((2, 64), "float32", name="out")
    ub = tirx.decl_buffer((2, 64), "float32", name="ub", scope="shared.dyn")
    row = tirx.Var("row", "int32")
    load = copy(a, ub, src_indices=[row, 0], dst_indices=[0 if overlap else row, 0], src_shape=[1, 64], dst_shape=[1, 64])
    rows = tirx.For(row, 0, 2, tirx.ForKind.SERIAL, unit(load))
    before = kernel(seq(unit(rows, core=None), unit(copy(ub, out))), buffers=[ub], params=[a, out])
    after = transform.InsertSync()(before)
    barriers = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE2"]
    assert bool(barriers) == overlap
    # The whole-buffer consumer must still wait for the row loads to finish.
    releases = _events(after, "set", "MTE2_MTE3")
    waits = _events(after, "wait", "MTE2_MTE3")
    assert len(releases) == len(waits) == 1
    tvm.ir.assert_structural_equal(releases[0].args[1], waits[0].args[1])
