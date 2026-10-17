"""Atomic task boundaries and per-core candidate synchronization."""

import pytest
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
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

    with pytest.raises(tvm.error.InternalError, match=message):
        transform.MaterializeScheduleUnits()(tvm.IRModule({"main": before}))


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

    after = transform.MaterializeScheduleUnits()(tvm.IRModule({"main": before}))
    groups = [node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.ascend_per_core_task"]
    assert len(groups) == 1
    candidates = [node for node in nodes(groups[0].body, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]
    assert len(candidates) == 2
    assert len(nodes(groups[0].body, tirx.IfThenElse)) == 2


@pytest.mark.parametrize("routing", ["constant", "parameter", "local"])
@pytest.mark.parametrize("bound_guard", [False, True])
@pytest.mark.parametrize("enclosing_loop", [False, True])
@pytest.mark.parametrize("side_effect", [False, True])
def test_per_core_fixpipe_candidates_keep_separate_routes(routing, bound_guard, enclosing_loop, side_effect):
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    counter = tirx.decl_buffer((4,), "int32", name="counter")
    sid, block = tirx.Var("sid", "int32"), tirx.Var("block", "int32")
    selected = block % 2 == 0
    prefix, candidates = [], []
    if bound_guard:
        condition = tirx.Var("condition", "bool")
        prefix.append(unit(tirx.Bind(condition, selected), core=0).body)
        selected = condition
    for index in range(2):
        target = tirx.IntImm("int32", index) if routing == "constant" else (block // 2 + index) % 2
        body = []
        if routing == "local":
            route = tirx.Var(f"route{index}", "int32")
            body.append(tirx.Bind(route, target))
            target = route
        if side_effect:
            body.append(tirx.BufferStore(counter, counter[block] + 1, [block]))
        body.append(copy(accum, ub, annotations={"sub_blockid": target}))
        candidates.append(tirx.IfThenElse(selected if index == 0 else tirx.Not(selected), unit(seq(*body), core=0).body, None))
    body = unit(tirx.AttrStmt({}, "tl.ascend_per_core_task", 1, seq(*prefix, *candidates)), core=None)
    if enclosing_loop:
        body = unit(tirx.For(tirx.Var("i", "int32"), 0, sid + 2, tirx.ForKind.SERIAL, body), core=None)
    before = kernel(body, buffers=[accum, ub], params=[block, counter])
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    if side_effect and routing == "local":
        with pytest.raises(tvm.error.InternalError, match="Fixpipe routing.*external side effects"):
            transform.AssignCore()(before)
        return
    after = transform.AssignCore()(before)
    assert not tirx.analysis.undefined_vars(after["main"].body, after["main"].params)
    groups = [
        (stmt, parents)
        for stmt, parents in statements(after)
        if isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "tl.ascend_per_core_task"
    ]
    assert len(groups) == 2
    analyzer = tvm.arith.Analyzer()
    for group, parents in groups:
        projected_sid = int(group.node["subcore"])
        assert int(group.node["core_mask"]) == 2
        copies = calls(group, "tl.tileop.ascend_copy")
        assert len(copies) == (1 if routing == "constant" else 2)
        assert {int(call.annotations["sub_blockid"]) for call in copies} == {projected_sid}
        for statement, ancestors in statements(group.body):
            if isinstance(statement, (tirx.Evaluate, tirx.Bind, tirx.BufferStore)):
                assert any(isinstance(parent, tirx.AttrStmt) and parent.attr_key == "tl.ascend_task" for parent in ancestors)
        active = tirx.const(True, "bool")
        for parent in parents:
            if isinstance(parent, tirx.IfThenElse):
                active = tirx.And(active, parent.condition)
        for block_id in range(4):
            route = block_id % 2 if routing == "constant" else (block_id // 2 + block_id % 2) % 2
            expected = routing == "local" or route == projected_sid
            actual = analyzer.simplify(tirx.stmt_functor.substitute(active, {block: tirx.IntImm("int32", block_id)}))
            assert isinstance(actual, tirx.IntImm) and bool(actual.value) == expected


@pytest.mark.parametrize("direction", ["load", "store"])
def test_per_core_vector_candidates_project_to_distinct_aivs(direction):
    gm = tirx.decl_buffer((256,), "float32", name="gm")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    sid, block = tirx.Var("sid", "int32"), tirx.Var("block", "int32")
    selected = (block + sid) % 2 == 0
    candidates = []
    for index in range(2):
        offset = index * 128 + sid * 64
        operation = (
            copy(gm, ub, src_indices=[offset], src_shape=[64])
            if direction == "load"
            else copy(ub, gm, dst_indices=[offset], dst_shape=[64])
        )
        candidates.append(tirx.IfThenElse(selected if index == 0 else tirx.Not(selected), unit(operation, core=0).body, None))
    group = tirx.AttrStmt({}, "tl.ascend_per_core_task", 1, seq(*candidates))
    before = kernel(unit(group, core=None), buffers=[ub], params=[gm, block])
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    after = transform.AssignCore()(before)
    groups = [node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.ascend_per_core_task"]
    assert len(groups) == 2
    storages = []
    for group in groups:
        projected_sid = int(group.node["subcore"])
        assert int(group.node["core_mask"]) == (1 if projected_sid == 0 else 4)
        operations = calls(group, "tl.tileop.ascend_copy")
        assert len(operations) == 2
        for index, operation in enumerate(operations):
            source, destination = [operand.args[0] for operand in operation.args]
            global_access, local_access = (source, destination) if direction == "load" else (destination, source)
            assert global_access.buffer.same_as(gm)
            assert int(tvm.arith.Analyzer().simplify(global_access.indices[0])) == index * 128 + projected_sid * 64
            if index == 0:
                storages.append(local_access.buffer.data)
            else:
                assert local_access.buffer.data.same_as(storages[-1])
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
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    sid, block = tirx.Var("sid", "int32"), tirx.Var("block", "int32")
    sync = [
        unit(tirx.Evaluate(tirx.call_intrin("handle", tvm.ir.Op.get(op), 0, "PIPE_FIX", 15)), core=0)
        for op in ("tl.ascend_cross_core_set_flag", "tl.ascend_cross_core_wait_flag")
    ]
    candidates = []
    for index in range(2):
        target = (block // 2 + index) % 2 if routing == "parameter" else tirx.IntImm("int32", index)
        annotations, prefix = {}, []
        if routing == "sid":
            target = sid
        elif routing == "sid-expression":
            target = tirx.if_then_else(block // 2 % 2 == 0, sid, 1 - sid)
        elif routing == "local":
            route = tirx.Var(f"route{index}", "int32")
            prefix.append(tirx.Bind(route, (block // 2 + index) % 2))
            target = route
        if routing != "implicit" and (routing != "mixed" or index == 1):
            annotations["sub_blockid"] = target
        candidate = unit(seq(*prefix, copy(accum, ub, annotations=annotations)), core=0).body
        candidates.append(tirx.IfThenElse(block % 2 == index, candidate, None))
    body = seq(candidates[0], *(event.body for event in sync), candidates[1]) if sync_inside else seq(*candidates)
    group = unit(tirx.AttrStmt({}, "tl.ascend_per_core_task", 1, body), core=None)
    body = group if sync_inside else seq(group, *sync)
    if enclosing_loop:
        body = unit(tirx.For(tirx.Var("i", "int32"), 0, sid + 2, tirx.ForKind.SERIAL, body), core=None)
    before = kernel(body, buffers=[accum, ub], params=[block], annotations={"vector_count": tirx.IntImm("int32", sids)})
    thread = tirx.IterVar(tvm.ir.Range(0, sids), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", sids, before["main"].body))})
    if sync_inside and sids == 2 and routing in ("mixed", "implicit", "local", "sid", "sid-expression"):
        with pytest.raises(tvm.error.InternalError, match="PerCoreTask.*explicit synchronization.*Fixpipe.*outside"):
            transform.AssignCore()(before)
        return
    after = transform.AssignCore()(before)
    events = {
        operation: [
            (statement.value, parents)
            for statement, parents in statements(after)
            if isinstance(statement, tirx.Evaluate) and calls(statement, "tl.ascend_cross_core_" + operation + "_flag")
        ]
        for operation in ("set", "wait")
    }
    assert events["set"] and events["wait"]
    analyzer = tvm.arith.Analyzer()
    for block_id in range(4):
        selected = {}
        for operation, candidates in events.items():
            active_events = []
            for event, parents in candidates:
                active = tirx.const(True, "bool")
                for parent in parents:
                    if isinstance(parent, tirx.IfThenElse):
                        active = tirx.And(active, parent.condition)
                active = analyzer.simplify(tirx.stmt_functor.substitute(active, {block: tirx.IntImm("int32", block_id)}))
                assert isinstance(active, tirx.IntImm)
                if int(active):
                    active_events.append(event)
            (selected[operation],) = active_events
        tvm.ir.assert_structural_equal(selected["set"].args, selected["wait"].args)


def test_sync_set_is_migrated_into_each_per_core_candidate():
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    b = tirx.decl_buffer((16, 16), "float32", name="B")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    l0 = tirx.decl_buffer((16, 16), "float32", name="l0", scope="shared.l0a")
    enabled = tirx.Var("enabled", "bool")
    candidates = seq(
        tirx.IfThenElse(enabled, unit(copy(a, l1), core=2).body, None),
        tirx.IfThenElse(tirx.Not(enabled), unit(copy(b, l1), core=2).body, None),
    )
    group = tirx.AttrStmt({"core_mask": tirx.IntImm("int64", 2)}, "tl.ascend_per_core_task", 1, candidates)
    before = kernel(seq(unit(group, core=None), unit(copy(l1, l0), core=2)), buffers=[l1, l0], params=[a, b, enabled])
    after = transform.InsertSync()(before)
    sets = [
        stmt.value
        for stmt, _ in statements(after)
        if isinstance(stmt, tirx.Evaluate)
        and isinstance(stmt.value, tirx.Call)
        and stmt.value.op.name == "tl.ascend_set_flag"
        and stmt.value.args[0].value == "MTE2_MTE1"
    ]
    waits = [call for call in calls(after, "tl.ascend_wait_flag") if call.args[0].value == "MTE2_MTE1"]
    assert len(sets) == 2 and len(waits) == 1
    guards = []
    for stmt, parents in statements(after):
        if isinstance(stmt, tirx.Evaluate) and any(stmt.value.same_as(event) for event in sets):
            guards.append(next(parent.condition for parent in reversed(parents) if isinstance(parent, tirx.IfThenElse)))
    assert len(guards) == 2
    assert tvm.arith.Analyzer().can_prove(tirx.Or(*guards))
    assert all(tvm.arith.Analyzer().can_prove_equal(event.args[1], waits[0].args[1]) for event in sets)


def test_guarded_per_core_task_keeps_wider_sync_outside_candidates():
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    l0 = tirx.decl_buffer((16, 16), "float32", name="l0", scope="shared.l0a")
    enabled = tirx.Var("enabled", "bool")
    group = tirx.AttrStmt({"core_mask": 2}, "tl.ascend_per_core_task", 1, unit(copy(a, l1), core=2).body)
    before = kernel(
        seq(unit(copy(a, l1), core=2), unit(group, core=None, guard=enabled), unit(copy(l1, l0), core=2)),
        buffers=[l1, l0],
        params=[a, enabled],
    )
    after = transform.InsertSync()(before)
    sets = []
    for stmt, parents in statements(after):
        if isinstance(stmt, tirx.Evaluate) and calls(stmt, "tl.ascend_set_flag"):
            assert not any(isinstance(parent, tirx.AttrStmt) and parent.attr_key == "tl.ascend_per_core_task" for parent in parents)
            assert not any(isinstance(parent, tirx.IfThenElse) for parent in parents)
            sets.append(stmt.value)
    assert sets and calls(after, "tl.ascend_wait_flag")


@pytest.mark.parametrize("dynamic", [False, True], ids=["static-fill", "dynamic-fill"])
def test_fill_outside_vf_synchronizes_on_scalar_pipe(dynamic):
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((64,), "float32", name="out")
    extent = tirx.Var("extent", "int32")
    fill = tirx.Evaluate(
        tirx.Call("handle", tvm.ir.Op.get("tl.tileop.fill"), [region(ub, 2, shape=[extent if dynamic else 64]), tirx.const(0, "float32")])
    )
    before = kernel(seq(unit(fill), unit(copy(ub, out))), buffers=[ub], params=[out, extent])
    after = transform.InsertSync()(before)
    for op in ("set", "wait"):
        assert len([call for call in calls(after, "tl.ascend_" + op + "_flag") if call.args[0].value == "S_MTE3"]) == 1
