"""Atomic task boundaries and per-core candidate synchronization."""

from tvm_tirx import ir
import tvm_ffi
from testing.ascend._ir import (
    access_tensor,
    attribute,
    attribute_fields,
    attribute_key,
    call_annotations,
    launch_thread,
    substitute,
    tensor,
)

import pytest
import tilelang.ascend.language as T
import tvm_tirx
from tilelang.ascend import transform
from tvm_tirx import tirx
from testing.ascend._ir import calls, copy, kernel, nodes, region, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


@pytest.mark.parametrize(
    "case, message",
    [
        ("cross-pipe", "exactly one Ascend hardware pipe"),
        ("cross-core", "both AIC and AIV HBM paths"),
        ("scalar-fill", "exactly one Ascend hardware pipe"),
        ("vector-copy", "exactly one Ascend hardware pipe"),
        ("two-syncs", "must be the task's only statement"),
        ("nested-per-core", "cannot be nested inside"),
    ],
)
def test_invalid_atomic_task(case, message):
    @T.prim_func
    def before(A: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 16), "float32")
            l0 = T.alloc_l0a((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            other = T.alloc_shared((16, 16), "float32")
            with T.Task():
                if case == "cross-pipe":
                    T.copy(A, l1)
                    T.copy(l1, l0)
                elif case == "cross-core":
                    T.copy(A, l1)
                    T.copy(A, ub)
                elif case == "scalar-fill":
                    T.copy(A, ub)
                    T.fill(ub, 0)
                elif case == "vector-copy":
                    T.copy(A, ub)
                    T.copy(ub, other)
                elif case == "two-syncs":
                    T.ascend_sync_inter_arrive("PIPE_V", 3)
                    T.ascend_sync_inter_wait("PIPE_V", 3)
                else:
                    with T.PerCoreTask():
                        T.copy(A, ub)

    with pytest.raises((ValueError, RuntimeError), match=message):
        transform.MaterializeScheduleUnits()(tvm_tirx.IRModule({"main": before}))


@pytest.mark.parametrize("explicit, mixed", [(True, False), (False, False), (True, True)], ids=["explicit", "inferred", "mixed"])
def test_per_core_candidates_preserve_conditions(explicit, mixed):
    @T.prim_func
    def before(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32"), enabled: T.bool):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 16), "float32")
            with T.PerCoreTask():
                if enabled:
                    if explicit:
                        with T.Task():
                            T.copy(A, l1)
                    else:
                        T.copy(A, l1)
                if not enabled:
                    if explicit and not mixed:
                        with T.Task():
                            T.copy(B, l1)
                    else:
                        T.copy(B, l1)

    after = transform.MaterializeScheduleUnits()(tvm_tirx.IRModule({"main": before}))
    groups = [node for node in nodes(after, ir.RegionStmt) if attribute_key(node) == "tl.ascend_per_core_task"]
    assert len(groups) == 1
    candidates = [node for node in nodes(groups[0].body, ir.RegionStmt) if attribute_key(node) == "tl.ascend_task"]
    assert len(candidates) == 2
    assert len(nodes(groups[0].body, ir.If)) == 2


@pytest.mark.parametrize("routing", ["constant", "parameter", "local"])
@pytest.mark.parametrize("bound_guard", [False, True])
@pytest.mark.parametrize("enclosing_loop", [False, True])
@pytest.mark.parametrize("side_effect", [False, True])
def test_per_core_fixpipe_candidates_keep_separate_routes(routing, bound_guard, enclosing_loop, side_effect):
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tensor((16, 16), "float32", name="ub", scope="shared.dyn")
    counter = tensor((4,), "int32", name="counter")
    sid, block = ir.Var("sid", "int32"), ir.Var("block", "int32")
    selected = block % 2 == 0
    prefix, candidates = [], []
    if bound_guard:
        condition = ir.Var("condition", "bool")
        # Pure candidate control stays in the PerCoreTask's lexical SSA scope.
        prefix.append(ir.Bind(condition, selected))
        selected = condition
    for index in range(2):
        target = ir.IntImm("int32", index) if routing == "constant" else (block // 2 + index) % 2
        body = []
        if routing == "local":
            route = ir.Var(f"route{index}", "int32")
            body.append(ir.Bind(route, target))
            target = route
        if side_effect:
            body.append(ir.TensorStore(counter, [block], counter[block] + 1))
        body.append(copy(accum, ub, annotations={"sub_blockid": target}))
        candidates.append(ir.If(selected if index == 0 else ir.prim.Not(selected), unit(seq(*body), core=0).body, None))
    body = unit(attribute({}, "tl.ascend_per_core_task", 1, seq(*prefix, *candidates)), core=None)
    if enclosing_loop:
        body = unit(ir.For(ir.Var("i", "int32"), ir.const(0), ir.const(0) + (sid + 2), ir.const(1), ir.ForKind.DEFAULT, body), core=None)
    before = kernel(body, buffers=[accum, ub], params=[block, counter], allocation_stage=0, allocation_core=0)
    thread = sid
    before = tvm_tirx.IRModule({"main": before["main"].with_body(launch_thread(thread, "cthread", 2, before["main"].body))})
    if side_effect and routing == "local":
        with pytest.raises((ValueError, RuntimeError), match="Fixpipe routing.*external side effects"):
            transform.AssignCore()(before)
        return
    after = transform.AssignCore()(before)
    assert not tirx.analysis.undefined_vars(after["main"].body, after["main"].params)
    groups = [
        (stmt, parents)
        for stmt, parents in statements(after)
        if isinstance(stmt, ir.RegionStmt) and attribute_key(stmt) == "tl.ascend_per_core_task"
    ]
    assert len(groups) == 2
    analyzer = tvm_tirx.sym.Analyzer()
    for group, parents in groups:
        projected_sid = int(attribute_fields(group)["node"]["subcore"])
        assert int(attribute_fields(group)["node"]["core_mask"]) == 2
        copies = calls(group, "tl.tileop.ascend_copy")
        assert len(copies) == (1 if routing == "constant" else 2)
        assert {int(call_annotations(call)["sub_blockid"]) for call in copies} == {projected_sid}
        scalar_controls = []
        for statement, ancestors in statements(group.body):
            if isinstance(statement, (ir.Evaluate, ir.Bind, ir.TensorStore)):
                in_candidate = any(isinstance(parent, ir.RegionStmt) and attribute_key(parent) == "tl.ascend_task" for parent in ancestors)
                if isinstance(statement, ir.Bind) and not in_candidate:
                    assert bound_guard
                    tvm_tirx.ir.assert_structural_equal(statement.value, block % 2 == 0)
                    scalar_controls.append(statement)
                else:
                    assert in_candidate
        assert len(scalar_controls) == int(bound_guard)
        active = ir.const(True, "bool")
        for parent in parents:
            if isinstance(parent, ir.If):
                active = ir.prim.And(active, parent.condition)
        for block_id in range(4):
            route = block_id % 2 if routing == "constant" else (block_id // 2 + block_id % 2) % 2
            expected = routing == "local" or route == projected_sid
            actual = analyzer.simplify(substitute(active, {block: ir.IntImm("int32", block_id)}))
            assert isinstance(actual, ir.IntImm) and bool(actual.value) == expected


@pytest.mark.parametrize("direction", ["load", "store"])
def test_per_core_vector_candidates_project_to_distinct_aivs(direction):
    gm = tensor((256,), "float32", name="gm")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    sid, block = ir.Var("sid", "int32"), ir.Var("block", "int32")
    selected = (block + sid) % 2 == 0
    candidates = []
    for index in range(2):
        offset = index * 128 + sid * 64
        operation = (
            copy(gm, ub, src_indices=[offset], src_shape=[64])
            if direction == "load"
            else copy(ub, gm, dst_indices=[offset], dst_shape=[64])
        )
        candidates.append(ir.If(selected if index == 0 else ir.prim.Not(selected), unit(operation, core=0).body, None))
    group = attribute({}, "tl.ascend_per_core_task", 1, seq(*candidates))
    before = kernel(unit(group, core=None), buffers=[ub], params=[gm, block], allocation_stage=0, allocation_core=0)
    thread = sid
    before = tvm_tirx.IRModule({"main": before["main"].with_body(launch_thread(thread, "cthread", 2, before["main"].body))})
    after = transform.AssignCore()(before)
    groups = [node for node in nodes(after, ir.RegionStmt) if attribute_key(node) == "tl.ascend_per_core_task"]
    assert len(groups) == 2
    storages = []
    for group in groups:
        projected_sid = int(attribute_fields(group)["node"]["subcore"])
        assert int(attribute_fields(group)["node"]["core_mask"]) == (1 if projected_sid == 0 else 4)
        operations = calls(group, "tl.tileop.ascend_copy")
        assert len(operations) == 2
        for index, operation in enumerate(operations):
            source, destination = operation.args[:2]
            global_access, local_access = (source, destination) if direction == "load" else (destination, source)
            assert access_tensor(global_access).same_as(gm)
            assert int(tvm_tirx.sym.Analyzer().simplify(global_access.region[0].begin)) == index * 128 + projected_sid * 64
            if index == 0:
                storages.append(access_tensor(local_access))
            else:
                assert access_tensor(local_access).same_as(storages[-1])
    assert not storages[0].same_as(storages[1])


@pytest.mark.parametrize(
    "routing, sync_inside, sids, enclosing_loop",
    [
        ("mixed", True, 2, False),
        ("implicit", True, 2, False),
        ("local", True, 2, False),
        ("sid", True, 2, False),
        ("sid-expression", True, 2, False),
        ("mixed", True, 2, True),
        ("explicit", True, 2, False),
        ("parameter", True, 2, False),
        ("mixed", False, 2, False),
        ("sid", False, 2, False),
        ("implicit", True, 1, False),
    ],
)
def test_per_core_fixpipe_sync_requires_single_execution(routing, sync_inside, sids, enclosing_loop):
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tensor((16, 16), "float32", name="ub", scope="shared.dyn")
    sid, block = ir.Var("sid", "int32"), ir.Var("block", "int32")
    sync = [
        unit(ir.Evaluate(ir.Call(tvm_tirx.ir.Op.get(op), [0, "PIPE_FIX", 15], ty="void")), core=0)
        for op in ("tl.ascend_cross_core_set_flag", "tl.ascend_cross_core_wait_flag")
    ]
    candidates = []
    for index in range(2):
        target = (block // 2 + index) % 2 if routing == "parameter" else ir.IntImm("int32", index)
        annotations, prefix = {}, []
        if routing == "sid":
            target = sid
        elif routing == "sid-expression":
            target = ir.prim.Select(block // 2 % 2 == 0, sid, 1 - sid)
        elif routing == "local":
            route = ir.Var(f"route{index}", "int32")
            prefix.append(ir.Bind(route, (block // 2 + index) % 2))
            target = route
        if routing != "implicit" and (routing != "mixed" or index == 1):
            annotations["sub_blockid"] = target
        candidate = unit(seq(*prefix, copy(accum, ub, annotations=annotations)), core=0).body
        candidates.append(ir.If(block % 2 == index, candidate, None))
    body = seq(candidates[0], *(event.body for event in sync), candidates[1]) if sync_inside else seq(*candidates)
    group = unit(attribute({}, "tl.ascend_per_core_task", 1, body), core=None)
    body = group if sync_inside else seq(group, *sync)
    if enclosing_loop:
        body = unit(ir.For(ir.Var("i", "int32"), ir.const(0), ir.const(0) + (sid + 2), ir.const(1), ir.ForKind.DEFAULT, body), core=None)
    before = kernel(
        body,
        buffers=[accum, ub],
        params=[block],
        annotations={"vector_count": ir.IntImm("int32", sids)},
        allocation_stage=0,
        allocation_core=0,
    )
    thread = sid
    before = tvm_tirx.IRModule({"main": before["main"].with_body(launch_thread(thread, "cthread", sids, before["main"].body))})
    if sync_inside and sids == 2 and routing in ("mixed", "implicit", "local", "sid", "sid-expression"):
        with pytest.raises((ValueError, RuntimeError), match="PerCoreTask.*explicit synchronization.*Fixpipe.*outside"):
            transform.AssignCore()(before)
        return
    after = transform.AssignCore()(before)
    events = {
        operation: [
            (statement.value, parents)
            for statement, parents in statements(after)
            if isinstance(statement, ir.Evaluate) and calls(statement, "tl.ascend_cross_core_" + operation + "_flag")
        ]
        for operation in ("set", "wait")
    }
    assert events["set"] and events["wait"]
    analyzer = tvm_tirx.sym.Analyzer()
    for block_id in range(4):
        selected = {}
        for operation, candidates in events.items():
            active_events = []
            for event, parents in candidates:
                active = ir.const(True, "bool")
                for parent in parents:
                    if isinstance(parent, ir.If):
                        active = ir.prim.And(active, parent.condition)
                active = analyzer.simplify(substitute(active, {block: ir.IntImm("int32", block_id)}))
                assert isinstance(active, ir.IntImm)
                if int(active):
                    active_events.append(event)
            (selected[operation],) = active_events
        tvm_tirx.ir.assert_structural_equal(selected["set"].args, selected["wait"].args)


def test_sync_set_is_migrated_into_each_per_core_candidate():
    a = tensor((16, 16), "float32", name="A")
    b = tensor((16, 16), "float32", name="B")
    l1 = tensor((16, 16), "float32", name="l1", scope="shared.l1")
    l0 = tensor((16, 16), "float32", name="l0", scope="shared.l0a")
    enabled = ir.Var("enabled", "bool")
    candidates = seq(
        ir.If(enabled, unit(copy(a, l1), core=2).body, None),
        ir.If(ir.prim.Not(enabled), unit(copy(b, l1), core=2).body, None),
    )
    group = attribute({"core_mask": ir.IntImm("int64", 2)}, "tl.ascend_per_core_task", 1, candidates)
    before = kernel(seq(unit(group, core=None), unit(copy(l1, l0), core=2)), buffers=[l1, l0], params=[a, b, enabled], allocation_stage=0)
    after = transform.InsertSync()(before)
    sets = [
        stmt.value
        for stmt, _ in statements(after)
        if isinstance(stmt, ir.Evaluate)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "tl.ascend_set_flag"
        and stmt.value.args[0].value == "MTE2_MTE1"
    ]
    waits = [call for call in calls(after, "tl.ascend_wait_flag") if call.args[0].value == "MTE2_MTE1"]
    assert len(sets) == 2 and len(waits) == 1
    guards = []
    for stmt, parents in statements(after):
        if isinstance(stmt, ir.Evaluate) and any(stmt.value.same_as(event) for event in sets):
            guards.append(next(parent.condition for parent in reversed(parents) if isinstance(parent, ir.If)))
    assert len(guards) == 2
    assert tvm_tirx.sym.Analyzer().can_prove(ir.prim.Or(*guards))
    assert all(tvm_tirx.sym.Analyzer().can_prove_equal(event.args[1], waits[0].args[1]) for event in sets)


def test_guarded_per_core_task_keeps_wider_sync_outside_candidates():
    a = tensor((16, 16), "float32", name="A")
    l1 = tensor((16, 16), "float32", name="l1", scope="shared.l1")
    l0 = tensor((16, 16), "float32", name="l0", scope="shared.l0a")
    enabled = ir.Var("enabled", "bool")
    group = attribute({"core_mask": 2}, "tl.ascend_per_core_task", 1, unit(copy(a, l1), core=2).body)
    before = kernel(
        seq(unit(copy(a, l1), core=2), unit(group, core=None, guard=enabled), unit(copy(l1, l0), core=2)),
        buffers=[l1, l0],
        params=[a, enabled],
        allocation_stage=0,
    )
    after = transform.InsertSync()(before)
    sets = []
    for stmt, parents in statements(after):
        if isinstance(stmt, ir.Evaluate) and calls(stmt, "tl.ascend_set_flag"):
            assert not any(isinstance(parent, ir.RegionStmt) and attribute_key(parent) == "tl.ascend_per_core_task" for parent in parents)
            assert not any(isinstance(parent, ir.If) for parent in parents)
            sets.append(stmt.value)
    assert sets and calls(after, "tl.ascend_wait_flag")


@pytest.mark.parametrize("dynamic", [False, True], ids=["static-fill", "dynamic-fill"])
def test_fill_outside_vf_synchronizes_on_scalar_pipe(dynamic):
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    out = tensor((64,), "float32", name="out")
    extent = ir.Var("extent", "int32")
    fill = ir.Evaluate(
        tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
            ir.PrimType("void"),
            tvm_tirx.ir.Op.get("tl.tileop.fill"),
            [region(ub, 2, shape=[extent if dynamic else 64]), ir.const(0, "float32")],
            {},
            ir.UnknownLoc(),
        )
    )
    before = kernel(seq(unit(fill), unit(copy(ub, out))), buffers=[ub], params=[out, extent], allocation_stage=0)
    after = transform.InsertSync()(before)
    for op in ("set", "wait"):
        assert len([call for call in calls(after, "tl.ascend_" + op + "_flag") if call.args[0].value == "S_MTE3"]) == 1
