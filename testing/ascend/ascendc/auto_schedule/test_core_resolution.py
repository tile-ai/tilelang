"""Core assignment follows storage access and scalar availability."""

import pytest
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import allocated_buffer, calls, copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit


def _tasks(mod):
    return [node for node in nodes(mod, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]


@pytest.mark.parametrize("scope, expected", [("shared.dyn", 5), ("shared.l1", 2)])
@pytest.mark.parametrize("sids", [1, 2])
def test_copy_destination_selects_core(scope, expected, sids):
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    local = tirx.decl_buffer((16, 16), "float32", name="local", scope=scope)
    before = kernel(unit(copy(a, local), core=0), buffers=[local], params=[a], annotations={"vector_count": tirx.IntImm("int32", sids)})
    after = transform.AssignCore()(before)
    (task,) = _tasks(after)
    assert int(task.node["core_mask"]) == (expected & (3 if sids == 1 else 7))


def test_scalar_cannot_mix_cube_and_vector_local_memory():
    ub = tirx.decl_buffer((1,), "int32", name="ub", scope="shared.dyn")
    l1 = tirx.decl_buffer((1,), "int32", name="l1", scope="shared.l1")
    value = tirx.Var("value", "int32")
    before = kernel(unit(tirx.Bind(value, ub[0] + l1[0]), core=0), buffers=[ub, l1])
    with pytest.raises(tvm.error.InternalError, match="incompatible core-local memories"):
        transform.AssignCore()(before)


@pytest.mark.parametrize(
    "producer_scope, consumer_scope", [("shared.dyn", "shared.l1"), ("shared.l1", "shared.dyn")], ids=["vector-to-cube", "cube-to-vector"]
)
def test_core_local_condition_cannot_control_other_core(producer_scope, consumer_scope):
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope=producer_scope)
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    dst = tirx.decl_buffer((16, 16), "float32", name="dst", scope=consumer_scope)
    condition = tirx.Var("condition", "bool")
    before = kernel(
        seq(unit(tirx.Bind(condition, pred[0] > 0), core=0), unit(copy(a, dst), core=0, guard=condition)), buffers=[pred, dst], params=[a]
    )
    with pytest.raises(tvm.error.InternalError, match="unavailable on every legal execution core"):
        transform.AssignCore()(before)


@pytest.mark.parametrize("mixed", [False, True], ids=["vector-only", "shared-with-cube"])
def test_counter_updates_cover_all_protocol_cores(mixed):
    before = copy_ring(prepared=True, owners=2, core=7)
    counter = allocated_buffer(before, "epoch")
    if mixed:
        # The first owner consumes L0C on Cube and shares the UB epoch with
        # Vector. Only this copy changes; the counter producer remains flexible.
        accum = tirx.decl_buffer((1, 64), "float32", name="accum", scope="shared.l0c")
        ub = allocated_buffer(before, "ub")
        changed = False

        def add_cube_reader(node):
            nonlocal changed
            if (
                not changed
                and isinstance(node, tirx.AttrStmt)
                and node.attr_key == "tl.ascend_task"
                and any(call.op.name == "tl.tileop.ascend_copy" for call in nodes(node.body, tirx.Call))
            ):
                changed = True
                return unit(copy(accum, ub), core=2).body

        func = before["main"]
        body = tirx.stmt_functor.ir_transform(func.body, None, add_cube_reader)
        # The pass reasons about the L0C source scope; the allocation belongs
        # to the kernel even though no producer is needed in this pass input.
        root = body.block
        body = tirx.SBlockRealize(
            [],
            True,
            tirx.SBlock([], [], [], "tilelang_root", root.body, alloc_buffers=[*root.alloc_buffers, accum], annotations=root.annotations),
        )
        before = tvm.IRModule({"main": func.with_body(body)})
    after = transform.ResolveCore()(before)
    writes = [task for task in _tasks(after) if any(store.buffer.same_as(counter) for store in nodes(task.body, tirx.BufferStore))]
    assert len(writes) == 3
    assert {int(task.node["core_mask"]) for task in writes} == ({7} if mixed else {5})


def test_control_extent_restricts_flexible_scalar_task():
    ub = tirx.decl_buffer((1,), "int32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((4,), "int32", name="out")
    i = tirx.Var("i", "int32")
    loop = tirx.For(i, 0, ub[0], tirx.ForKind.SERIAL, unit(tirx.BufferStore(out, i, [i]), core=7))
    before = kernel(unit(loop, core=None), buffers=[ub], params=[out])
    after = transform.ResolveCore()(before)
    assert {int(task.node["core_mask"]) for task in _tasks(after)} == {5}


@pytest.mark.parametrize("control", ["bound", "assume"])
def test_control_reads_must_be_available_on_child_core(control):
    ub = tirx.decl_buffer((1,), "int32", name="ub", scope="shared.dyn")
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    i = tirx.Var("i", "int32")
    loop = tirx.For(i, 0, ub[0] if control == "bound" else 1, tirx.ForKind.SERIAL, unit(copy(a, l1), core=2))
    if control == "assume":
        loop = tirx.AttrStmt(ub[0] >= 0, "tl.assume", 1, loop)
    before = kernel(unit(loop, core=None), buffers=[ub, l1], params=[a])
    with pytest.raises(tvm.error.InternalError, match="Cannot evaluate a loop bound or control guard"):
        transform.ResolveCore()(before)


@pytest.mark.parametrize("source", ["global", "shared.dyn"])
def test_epoch_guard_producer_must_cover_all_protocol_cores(source):
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope=source)
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    out = tirx.decl_buffer((16, 16), "float32", name="out")
    epoch = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var")
    guard, i = tirx.Var("guard", "bool"), tirx.Var("i", "int32")
    loop = tirx.For(
        i,
        0,
        4,
        tirx.ForKind.SERIAL,
        seq(
            unit(copy(accum, ub), core=2),
            unit(copy(ub, out), guard=guard),
            unit(tirx.BufferStore(epoch, epoch[0] + 1, [0]), core=7, guard=guard),
        ),
        annotations={
            "multi_buffer_eligible": [ub.data],
            "tl.multi_buffer_counter_map": {ub.data: epoch},
            "tl.storage_epoch_guard_map": {ub.data: guard},
        },
    )
    before = kernel(
        seq(
            unit(tirx.Bind(guard, pred[0] > 0), core=7 if source == "global" else 5),
            unit(tirx.BufferStore(epoch, 0, [0]), core=7),
            unit(loop, core=None),
        ),
        buffers=[ub, accum, epoch] + ([pred] if source != "global" else []),
        params=[out] + ([pred] if source == "global" else []),
        annotations={"tl.buffer_versions_map": {ub.data: 2}},
    )
    if source != "global":
        with pytest.raises(tvm.error.InternalError, match="storage-epoch guard .* is unavailable"):
            transform.ResolveCore()(before)
    else:
        after = transform.ResolveCore()(before)
        (producer,) = [task for task in _tasks(after) if isinstance(task.body, tirx.Bind) and task.body.var.same_as(guard)]
        assert int(producer.node["core_mask"]) == 7


@pytest.mark.parametrize("context", ["task-guard", "ancestor", "pure-kernel"])
def test_ambiguous_cross_core_pipe_uses_available_context(context):
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope="shared.dyn")
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    flag = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_cross_core_set_flag"), [4, "PIPE_S", 3]))
    flag_unit = unit(flag, core=0, guard=pred[0] > 0 if context == "task-guard" else None)
    if context == "ancestor":
        i = tirx.Var("i", "int32")
        flag_unit = unit(tirx.For(i, 0, pred[0], tirx.ForKind.SERIAL, flag_unit), core=None)
    # A Cube copy supplies an opposing kernel context. Local scalar inputs
    # must take precedence; without those inputs the pure kernel selects Cube.
    before = kernel(seq(unit(copy(a, l1), core=0), flag_unit), buffers=[pred, l1], params=[a])
    after = transform.AssignCore()(before)
    (placed,) = [task for task in _tasks(after) if isinstance(task.body, tirx.Evaluate) and task.body.value.same_as(flag.value)]
    assert int(placed.node["core_mask"]) == (2 if context == "pure-kernel" else 5)


@pytest.mark.parametrize("scope, core", [("shared.dyn", 1), ("shared.l1", 2)])
@pytest.mark.parametrize("assumption", [False, True])
def test_subblock_guard_is_specialized_before_core_assignment(scope, core, assumption):
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    dst = tirx.decl_buffer((16, 16), "float32", name="dst", scope=scope)
    sid = tirx.Var("sid", "int32")
    scheduled = unit(copy(a, dst), core=0, guard=sid == 0)
    if assumption:
        scheduled = tirx.AttrStmt(
            scheduled.node,
            scheduled.attr_key,
            scheduled.value,
            tirx.AttrStmt(sid >= 0, "tl.assume", 1, scheduled.body),
        )
    before = kernel(scheduled, buffers=[dst], params=[a])
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    after = transform.AssignCore()(before)
    assert {int(task.node["core_mask"]) for task in _tasks(after)} == {core}
    assert len(_tasks(after)) == 1
    assert all(not any(var.same_as(sid) for var in tirx.analysis.undefined_vars(task.body)) for task in _tasks(after))


def test_scalar_global_store_uses_the_existing_vector_core():
    src = tirx.decl_buffer((16,), "float32", name="src")
    out = tirx.decl_buffer((16,), "float32", name="out")
    ub = tirx.decl_buffer((16,), "float32", name="ub", scope="shared.dyn")
    before = kernel(seq(unit(copy(src, ub), core=5), unit(tirx.BufferStore(out, src[0] + 1, [0]), core=7)), buffers=[ub], params=[src, out])
    after = transform.ResolveCore()(before)
    assert {int(task.node["core_mask"]) for task in _tasks(after)} == {5}


@pytest.mark.parametrize("field", ["shape", "strides", "elem_offset"])
@pytest.mark.parametrize("view", [False, True], ids=["allocation", "view"])
def test_subcore_buffer_expressions_preserve_definitions_and_views(field, view):
    sid = tirx.Var("sid", "int32")
    storage = tirx.decl_buffer((16, 64), "float32", name="storage", scope="shared.dyn")
    buffer = tirx.decl_buffer(
        (16, (sid + 1) * 16 if field == "shape" else 16),
        "float32",
        name="view" if view else "ub",
        data=storage.data if view else None,
        scope="shared.dyn",
        strides=[(sid + 1) * 16 if field == "strides" else 32, 1],
        elem_offset=sid * 16 if field == "elem_offset" else 0,
    )
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    # The L0C producer is outside AssignCore's contract. Both accesses must
    # reuse the projected view, while allocation ownership follows its data.
    task = unit(seq(tirx.DeclBuffer(buffer), copy(accum, buffer, dst_shape=[16, 16])), core=0)
    before = kernel(
        seq(task, unit(copy(accum, buffer, dst_shape=[16, 16]), core=0)),
        buffers=[accum, storage if view else buffer],
        annotations={"test.buffer_metadata": {buffer: buffer}},
    )
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    after = transform.AssignCore()(before)
    root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
    metadata = root.annotations["test.buffer_metadata"]
    tasks = _tasks(after)
    assert sorted(int(task.node["subcore"]) for task in tasks) == [0, 0, 1, 1]
    projected, declared = {}, []
    for task in tasks:
        owner = int(task.node["subcore"])
        (operation,) = calls(task, "tl.tileop.ascend_copy")
        result = operation.args[1].args[0].buffer
        assert [int(value) for value in result.shape] == [16, (owner + 1) * 16 if field == "shape" else 16]
        assert [int(value) for value in result.strides] == [(owner + 1) * 16 if field == "strides" else 32, 1]
        assert int(result.elem_offset) == (owner * 16 if field == "elem_offset" else 0)
        if owner in projected:
            assert result.same_as(projected[owner])
        projected[owner] = result
        for definition in nodes(task, tirx.DeclBuffer):
            assert definition.buffer.same_as(result)
            declared.append(owner)
        assert metadata[result].same_as(result)
        (allocation,) = [allocated for allocated in root.alloc_buffers if allocated.data.same_as(result.data)]
        assert view or allocation.same_as(result)
    assert set(projected) == {0, 1}
    assert sorted(declared) == [0, 1]
    assert not projected[0].data.same_as(projected[1].data)


@pytest.mark.parametrize("field", ["shape", "strides", "elem_offset"])
def test_nested_subcore_projection_reuses_binders_and_views(field):
    sid, outer, middle = (tirx.Var(name, "int32") for name in ("sid", "outer", "middle"))
    # A source name may already have this suffix; projection must use identity.
    i, value = tirx.Var("i_aiv0", "int32"), tirx.Var("value", "int32")
    storage = tirx.decl_buffer((64,), "int32", name="storage", scope="shared.dyn")
    view = tirx.decl_buffer(
        (i + 1 if field == "shape" else 1,),
        "int32",
        name="view",
        data=storage.data,
        scope="shared.dyn",
        strides=[i + 1 if field == "strides" else 1],
        elem_offset=i if field == "elem_offset" else 0,
    )
    body = seq(tirx.Bind(value, i + outer + middle + sid), tirx.DeclBuffer(view), tirx.BufferStore(view, value, [0]))
    inner = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"test.buffer_metadata": {view: view}})
    nested = unit(tirx.For(middle, 0, 2, tirx.ForKind.SERIAL, unit(inner, core=0)), core=None)
    before = kernel(unit(tirx.For(outer, 0, sid + 1, tirx.ForKind.SERIAL, nested), core=None), buffers=[storage])
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})

    after = transform.AssignCore()(before)
    assert not tirx.analysis.undefined_vars(after["main"].body, after["main"].params)
    loops = nodes(after, tirx.For)
    assert len(loops) == 6
    assert {loop.loop_var.name for loop in loops} == {f"{var.name}_aiv{p}" for var in (outer, middle, i) for p in range(2)}
    tasks = _tasks(after)
    assert sorted(int(task.node["subcore"]) for task in tasks) == [0, 1]
    for task in tasks:
        owner = int(task.node["subcore"])
        (loop,) = nodes(task.body, tirx.For)
        (binding,) = nodes(task.body, tirx.Bind)
        (definition,) = nodes(task.body, tirx.DeclBuffer)
        (store,) = nodes(task.body, tirx.BufferStore)
        result = definition.buffer
        assert binding.var.name == f"value_aiv{owner}"
        assert store.value.same_as(binding.var)
        assert result.name == f"view_aiv{owner}"
        assert store.buffer.same_as(result)
        assert loop.annotations["test.buffer_metadata"][result].same_as(result)
        assert result.data.same_as(allocated_buffer(after, f"storage_aiv{owner}").data)
        expression = result.elem_offset if field == "elem_offset" else getattr(result, field)[0]
        tvm.ir.assert_structural_equal(expression, loop.loop_var if field == "elem_offset" else loop.loop_var + 1)


@pytest.mark.parametrize("field", ["shape", "strides", "elem_offset"])
@pytest.mark.parametrize("scope", ["global", "shared.l1"])
def test_subcore_view_metadata_projects_without_cloning_common_storage(field, scope):
    from tilelang.layout import Layout

    sid = tirx.Var("sid", "int32")
    storage = tirx.decl_buffer((16, 64), "float32", name="storage", scope=scope)
    view = tirx.decl_buffer(
        (16, (sid + 1) * 16 if field == "shape" else 16),
        "float32",
        name="view",
        data=storage.data,
        scope=scope,
        strides=[(sid + 1) * 16 if field == "strides" else 32, 1],
        elem_offset=sid * 16 if field == "elem_offset" else 0,
    )
    destination = tirx.decl_buffer((16, 16), "float32", name="destination", scope="shared.l1" if scope == "global" else "shared.l0a")
    annotations = {"test.buffer_metadata": {view: view}}
    layout = Layout((16, 16), lambda i, j: (i, j))
    if field != "shape":
        annotations["layout_map"] = {view: layout}
    # No address or control expression mentions sid: the view's metadata is
    # the only reason these Cube copies require subcore projection.
    before = kernel(
        unit(copy(view, destination, src_shape=[16, 16]), core=0),
        buffers=[destination] + ([storage] if scope != "global" else []),
        params=[storage] if scope == "global" else [],
        annotations=annotations,
    )
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    after = transform.AssignCore()(before)
    root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
    tasks = _tasks(after)
    assert sorted(int(task.node["subcore"]) for task in tasks) == [0, 1]
    metadata = root.annotations["test.buffer_metadata"]
    for task in tasks:
        owner = int(task.node["subcore"])
        (operation,) = calls(task, "tl.tileop.ascend_copy")
        result = operation.args[0].args[0].buffer
        assert result.data.same_as(storage.data)
        assert operation.args[1].args[0].buffer.same_as(destination)
        assert [int(value) for value in result.shape] == [16, (owner + 1) * 16 if field == "shape" else 16]
        assert [int(value) for value in result.strides] == [(owner + 1) * 16 if field == "strides" else 32, 1]
        assert int(result.elem_offset) == (owner * 16 if field == "elem_offset" else 0)
        assert metadata[result].same_as(result)
        if field != "shape":
            assert root.annotations["layout_map"][result].same_as(layout)
    if scope == "global":
        assert after["main"].buffer_map[storage.data].same_as(storage)
    else:
        assert any(allocated.same_as(storage) for allocated in root.alloc_buffers)


@pytest.mark.parametrize(
    "targets", [(0, 0), (1, 1), (0, 1), (None, 0)], ids=["aiv0", "aiv1", "different-subcores", "implicit-and-explicit"]
)
@pytest.mark.parametrize("per_core", [False, True])
def test_grouped_fixpipe_copies_require_matching_subcore_targets(targets, per_core):
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((16, 16), "float32", name="out")
    sid = tirx.Var("sid", "int32")
    copies = [
        copy(
            accum,
            ub,
            annotations={"unit_flag_ctrl": tirx.IntImm("int32", flag)}
            | ({} if target is None else {"sub_blockid": tirx.IntImm("int32", target)}),
        )
        for target, flag in zip(targets, (2, 3))
    ]
    # AssignCore only needs the copy destinations; the L0C producer is outside
    # this pass's contract.
    producer = unit(seq(*copies), core=0)
    if per_core:
        producer = unit(tirx.AttrStmt({}, "tl.ascend_per_core_task", 1, producer.body), core=None)
    before = kernel(seq(producer, unit(copy(ub, out), core=0)), buffers=[accum, ub], params=[out])
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    if targets[0] != targets[1]:
        with pytest.raises(tvm.error.InternalError, match="same sub_blockid.*separate T.Task"):
            transform.AssignCore()(before)
    else:
        after = transform.AssignCore()(before)
        (task,) = [
            task
            for task in nodes(after, tirx.AttrStmt)
            if task.attr_key == ("tl.ascend_per_core_task" if per_core else "tl.ascend_task")
            and any(call.args[0].args[0].buffer.same_as(accum) for call in calls(task, "tl.tileop.ascend_copy"))
        ]
        projected = calls(task, "tl.tileop.ascend_copy")
        assert len(projected) == len(copies)
        assert int(task.node["core_mask"]) == 2
        assert {int(call.annotations["sub_blockid"]) for call in projected} == {targets[0]}
        assert [int(call.annotations["unit_flag_ctrl"]) for call in projected] == [2, 3]


@pytest.mark.parametrize("route_kind", ["bind", "load", "loop-var"])
def test_fixpipe_route_keeps_its_task_local_scope(route_kind):
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    routes = tirx.decl_buffer((1,), "int32", name="routes")
    sid, route = tirx.Var("sid", "int32"), tirx.Var("route", "int32")
    state = tirx.decl_buffer((1,), "int32", name="state", scope="local.var")
    target = state[0] if route_kind == "load" else route
    body = copy(accum, ub, annotations={"sub_blockid": target})
    if route_kind == "bind":
        body = seq(tirx.Bind(route, routes[0] % 2), body)
    elif route_kind == "load":
        body = seq(tirx.BufferStore(state, routes[0] % 2, [0]), body)
    elif route_kind == "loop-var":
        body = tirx.For(route, 0, 2, tirx.ForKind.SERIAL, body)
    before = kernel(unit(body, core=0), buffers=[accum, ub] + ([state] if route_kind == "load" else []), params=[routes])
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    after = transform.AssignCore()(before)
    assert not tirx.analysis.undefined_vars(after["main"].body, after["main"].params)
    tasks = _tasks(after)
    assert len(tasks) == 2
    for task in tasks:
        (branch,) = nodes(task.body, tirx.IfThenElse)
        (operation,) = calls(branch.then_case, "tl.tileop.ascend_copy")
        if route_kind == "bind":
            expected = nodes(task.body, tirx.Bind)[0].var
        elif route_kind == "loop-var":
            expected = nodes(task.body, tirx.For)[0].loop_var
        else:
            expected = nodes(task.body, tirx.BufferStore)[0].buffer[0]
        tvm.ir.assert_structural_equal(branch.condition, expected == int(operation.annotations["sub_blockid"]))


@pytest.mark.parametrize(
    "effect, routing, context",
    [
        (effect, routing, context)
        for effect in ("store", "extern", "route-call", "gm-copy", "pure-call")
        for routing, context in (
            ("dynamic", "task"),
            ("dynamic", "loop"),
            ("dynamic", "nested-loop"),
            ("constant", "task"),
            ("constant", "loop"),
            ("constant", "nested-loop"),
            ("single-projection", "task"),
            ("single-projection", "loop"),
            ("single-projection", "nested-loop"),
            # Only single-projection routing adds an enclosing guard; otherwise
            # guarded-loop produces exactly the same input as loop.
            ("single-projection", "guarded-loop"),
        )
    ]
    + [
        ("gm-copy", "implicit", "task"),
        ("gm-copy", "implicit", "nested-loop"),
        ("gm-copy", "implicit-single-projection", "task"),
        ("gm-copy", "implicit-single-projection", "guarded-loop"),
        ("pure-call", "implicit", "task"),
        ("gm-copy", "implicit", "separate"),
        ("gm-copy", "sid", "task"),
        ("gm-copy", "sid", "nested-loop"),
        ("gm-copy", "sid-expression", "task"),
        ("gm-copy", "sid-alias", "task"),
        ("gm-copy", "sid-single-projection", "guarded-loop"),
        ("gm-copy", "sid", "separate"),
        ("pure-call", "sid", "task"),
    ],
)
def test_projected_fixpipe_route_rejects_repeated_external_effects(effect, routing, context):
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((16, 16), "float32", name="out")
    routes = tirx.decl_buffer((1,), "int32", name="routes")
    sid, route = tirx.Var("sid", "int32"), tirx.Var("route", "int32")
    selector = tirx.Var("selector", "int32")
    sid_alias = tirx.Var("sid_alias", "int32")
    if effect == "store":
        side_effect = tirx.BufferStore(routes, routes[0] + 1, [0])
    elif effect == "gm-copy":
        side_effect = copy(accum, out)
    elif effect == "pure-call":
        side_effect = tirx.Evaluate(tirx.call_pure_extern("int32", "read_route", route))
    elif effect == "route-call":
        side_effect = tirx.Evaluate(0)
    else:
        side_effect = tirx.Evaluate(tirx.call_extern("int32", "record_route", route))
    implicit = routing.startswith("implicit")
    sid_routing = routing.startswith("sid")
    single_projection = "single-projection" in routing
    target = tirx.IntImm("int32", 0) if routing == "constant" else route
    if sid_routing:
        target = sid
        if routing == "sid-expression":
            target = tirx.if_then_else(selector % 2 == 0, sid, 1 - sid)
        elif routing == "sid-alias":
            target = sid_alias
    elif effect == "route-call" and routing != "constant":
        target = tirx.call_extern("int32", "next_route")
    elif effect == "pure-call" and routing != "constant":
        target = tirx.call_pure_extern("int32", "read_route", routes[0])
    fix = copy(accum, ub, annotations={} if implicit else {"sub_blockid": target})
    if context == "separate":
        body = seq(unit(fix, core=0), unit(side_effect, core=0))
    else:
        effects = (fix, side_effect) if implicit or sid_routing else (side_effect, fix)
        body = seq(tirx.Bind(route, routes[0] % 2), *effects)
        body = unit(body, core=0, guard=sid == 0 if single_projection and context != "guarded-loop" else None)
    if routing == "sid-alias":
        body = seq(unit(tirx.Bind(sid_alias, sid), core=0), body)
    if context not in ("task", "separate"):
        i = tirx.Var("i", "int32")
        if context == "nested-loop":
            body = unit(tirx.For(tirx.Var("j", "int32"), 0, 2, tirx.ForKind.SERIAL, body), core=None)
        body = unit(
            tirx.For(i, 0, sid + 2, tirx.ForKind.SERIAL, body),
            core=None,
            guard=sid == 0 if single_projection and context == "guarded-loop" else None,
        )
    before = kernel(
        body,
        buffers=[accum, ub],
        params=[routes, out] + ([selector] if routing == "sid-expression" else []),
    )
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    repeated = (routing in ("dynamic", "implicit") or sid_routing) and not single_projection
    if repeated and context != "separate" and effect != "pure-call":
        with pytest.raises(tvm.error.InternalError, match="Fixpipe routing.*external side effects.*separate T.Task"):
            transform.AssignCore()(before)
    else:
        after = transform.AssignCore()(before)
        expected_tasks = 3 if context == "separate" else 2 if repeated else 1
        assert len(_tasks(after)) == expected_tasks
        if effect == "gm-copy":
            gm_copies = [call for call in calls(after, "tl.tileop.ascend_copy") if call.args[1].args[0].buffer.same_as(out)]
            assert len(gm_copies) == 1


@pytest.mark.parametrize(
    "cube_owners, nested, via_scalar",
    [
        # With no Cube consumer, nesting and scalar demand do not change IR.
        pytest.param((), False, False, id="vector-only"),
        pytest.param((0,), False, False, id="aiv0-direct-copy"),
        pytest.param((0,), True, False, id="aiv0-nested-copy"),
        pytest.param((0,), False, True, id="aiv0-direct-scalar"),
        pytest.param((0,), True, True, id="aiv0-nested-scalar"),
        pytest.param((1,), False, False, id="aiv1-direct-copy"),
        pytest.param((1,), True, False, id="aiv1-nested-copy"),
        pytest.param((1,), False, True, id="aiv1-direct-scalar"),
        pytest.param((1,), True, True, id="aiv1-nested-scalar"),
        pytest.param((None,), False, False, id="shared-direct-copy"),
        pytest.param((None,), True, False, id="shared-nested-copy"),
        pytest.param((None,), False, True, id="shared-direct-scalar"),
        pytest.param((None,), True, True, id="shared-nested-scalar"),
        pytest.param((0, 1), False, False, id="both-direct-copy"),
        pytest.param((0, 1), True, False, id="both-nested-copy"),
        pytest.param((0, 1), False, True, id="both-direct-scalar"),
        pytest.param((0, 1), True, True, id="both-nested-scalar"),
    ],
)
def test_projected_loop_break_follows_its_cube_branch(cube_owners, nested, via_scalar):
    src = tirx.decl_buffer((16,), "float32", name="src")
    buffers = [tirx.decl_buffer((16,), "float32", name=f"ub{sid}", scope="shared.dyn") for sid in range(2)]
    i = tirx.Var("i", "int32")
    body = []
    for sid, core in enumerate((1, 4)):
        marker = tirx.AttrStmt({"core_mask": core | 2, "subcore": sid}, "tl.ascend_task", 1, tirx.Evaluate(T.loop_break()))
        body.extend((unit(marker, core=None, guard=i >= sid + 1), unit(copy(src, buffers[sid]), core=core)))
    l1 = tirx.decl_buffer((16,), "float32", name="l1", scope="shared.l1")
    prefix, suffix = [], []
    for sid in cube_owners:
        metadata = {"core_mask": 2}
        if sid is not None:
            metadata["subcore"] = sid
        cube = unit(tirx.AttrStmt(metadata, "tl.ascend_task", 1, copy(src, l1)), core=None)
        if via_scalar:
            value = tirx.decl_buffer((1,), "int32", name=f"value_{sid}", scope="local.var")
            buffers.append(value)
            prefix.append(unit(tirx.BufferStore(value, 0, [0]), core=7))
            suffix.append(unit(cube.body, core=None, guard=value[0] > 0))
            cube = unit(tirx.AttrStmt({**metadata, "core_mask": 7}, "tl.ascend_task", 1, tirx.BufferStore(value, i, [0])), core=None)
        if nested:
            cube = unit(tirx.For(tirx.Var("j", "int32"), 0, 2, tirx.ForKind.SERIAL, cube), core=None)
        body.append(cube)
    # A common scalar producer may be recomputed on Cube without owning its
    # loop control. Its fallback placement must not broaden either break.
    body.append(unit(tirx.Bind(tirx.Var("value", "int32"), i), core=7))
    buffers.append(l1)
    before = kernel(
        seq(*prefix, unit(tirx.For(i, 0, 3, tirx.ForKind.SERIAL, seq(*body)), core=None), *suffix), buffers=buffers, params=[src]
    )
    if cube_owners in ((None,), (0, 1)):
        with pytest.raises(tvm.error.InternalError, match="Independent subcore loop_break.*separate loop"):
            transform.ResolveCore()(before)
        return
    after = transform.ResolveCore()(before)
    exits = [task for task in _tasks(after) if calls(task, "tl.loop_break")]
    assert len(exits) == 2
    assert {int(task.node["subcore"]): int(task.node["core_mask"]) for task in exits} == {
        sid: core | (2 if sid in cube_owners else 0) for sid, core in enumerate((1, 4))
    }


@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("consumer", ["copy", "loop-bound", "loop-break"])
@pytest.mark.parametrize("shared_consumer", [False, True])
def test_projected_loop_exit_follows_scalar_demand_from_later_consumers(owner, consumer, shared_consumer):
    src = tirx.decl_buffer((4, 16), "float32", name="src")
    l1 = tirx.decl_buffer((16,), "float32", name="l1", scope="shared.l1")
    ub = tirx.decl_buffer((16,), "float32", name="ub", scope="shared.dyn")
    count = tirx.decl_buffer((1,), "int32", name="count", scope="local.var")
    i, j, offset = (tirx.Var(name, "int32") for name in ("i", "j", "offset"))
    stop = tirx.Var("stop", "bool")

    def task(body, core, sid=None, guard=None):
        metadata = {"core_mask": core}
        if sid is not None:
            metadata["subcore"] = sid
        return unit(tirx.AttrStmt(metadata, "tl.ascend_task", 1, body), core=None, guard=guard)

    loop = unit(
        tirx.For(
            i,
            0,
            4,
            tirx.ForKind.SERIAL,
            seq(
                # A common predicate is required by both exits. Its Cube
                # execution must inherit the actual consumer's domain.
                task(tirx.Bind(stop, i == 1), 7),
                task(tirx.Evaluate(T.loop_break()), 3, 0, stop),
                task(tirx.Evaluate(T.loop_break()), 6, 1, stop),
                task(tirx.BufferStore(count, count[0] + 1, [0]), 7, owner),
                task(copy(src, ub, src_indices=[i, 0], src_shape=[1, 16]), 4 if owner == 0 else 1, 1 - owner),
            ),
        ),
        core=None,
    )
    consumer_sid = None if shared_consumer else owner
    read = task(copy(src, l1, src_indices=[offset if consumer == "copy" else 0, 0], src_shape=[1, 16]), 2, consumer_sid)
    if consumer == "copy":
        read = seq(task(tirx.Bind(offset, count[0] - 1), 7), read)
    else:
        if consumer == "loop-break":
            read = seq(task(tirx.Evaluate(T.loop_break()), 7, guard=j >= count[0]), read)
        read = unit(tirx.For(j, 0, count[0] if consumer == "loop-bound" else 4, tirx.ForKind.SERIAL, read), core=None)
    before = kernel(seq(task(tirx.BufferStore(count, 0, [0]), 7, owner), loop, read), buffers=[l1, ub, count], params=[src])
    after = transform.ResolveCore()(before)
    exits = [task for task in _tasks(after) if calls(task, "tl.loop_break") and "subcore" in task.node]
    assert {int(task.node["subcore"]): int(task.node["core_mask"]) for task in exits} == {
        sid: 2 if sid == owner else core for sid, core in enumerate((1, 4))
    }
    count_writes = [task for task in _tasks(after) if any(store.buffer.same_as(count) for store in nodes(task, tirx.BufferStore))]
    assert len(count_writes) == 2
    assert all(int(task.node["core_mask"]) & 2 for task in count_writes)


def test_unit_flag_pairing_between_source_copies_remains_caller_managed():
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    sid = tirx.Var("sid", "int32")
    fix = copy(accum, ub, annotations={"unit_flag_ctrl": tirx.IntImm("int32", 3)})
    # Reusing an IR node still describes two source operations. This pass does
    # not validate their caller-managed protocol; no such kernel is executed.
    before = kernel(seq(unit(fix, core=0, guard=sid == 0), unit(fix, core=0, guard=sid == 1)), buffers=[accum, ub])
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    after = transform.AssignCore()(before)
    copies = calls(after, "tl.tileop.ascend_copy")
    assert len(copies) == 2
    assert {int(call.annotations["sub_blockid"]) for call in copies} == {0, 1}
    assert all(int(call.annotations["unit_flag_ctrl"]) == 3 for call in copies)
