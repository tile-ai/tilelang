"""Core assignment follows storage access and scalar availability."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import allocated_buffer, copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit


def _tasks(mod):
    return [node for node in nodes(mod, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]


@pytest.mark.parametrize("scope, expected", [("shared.dyn", 1), ("shared.l1", 2)])
def test_copy_destination_selects_core(scope, expected):
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    local = tirx.decl_buffer((16, 16), "float32", name="local", scope=scope)
    before = kernel(unit(copy(a, local), core=0), buffers=[local], params=[a])
    after = transform.AssignCore()(before)
    (task,) = _tasks(after)
    assert int(task.node["core_mask"]) == expected


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
    before = copy_ring(prepared=True, owners=2, core=3)
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
    assert {int(task.node["core_mask"]) for task in writes} == ({3} if mixed else {1})


def test_control_extent_restricts_flexible_scalar_task():
    ub = tirx.decl_buffer((1,), "int32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((4,), "int32", name="out")
    i = tirx.Var("i", "int32")
    loop = tirx.For(i, 0, ub[0], tirx.ForKind.SERIAL, unit(tirx.BufferStore(out, i, [i]), core=3))
    before = kernel(unit(loop, core=None), buffers=[ub], params=[out])
    after = transform.ResolveCore()(before)
    assert {int(task.node["core_mask"]) for task in _tasks(after)} == {1}


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
            unit(tirx.BufferStore(epoch, epoch[0] + 1, [0]), core=3, guard=guard),
        ),
        annotations={
            "multi_buffer_eligible": [ub.data],
            "tl.multi_buffer_counter_map": {ub.data: epoch},
            "tl.storage_epoch_guard_map": {ub.data: guard},
        },
    )
    before = kernel(
        seq(
            unit(tirx.Bind(guard, pred[0] > 0), core=3 if source == "global" else 1),
            unit(tirx.BufferStore(epoch, 0, [0]), core=3),
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
        assert int(producer.node["core_mask"]) == 3


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
    assert int(placed.node["core_mask"]) == (2 if context == "pure-kernel" else 1)


@pytest.mark.parametrize("scope, legal", [("shared.dyn", True), ("shared.l1", False)])
def test_subblock_id_is_only_available_on_vector(scope, legal):
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    dst = tirx.decl_buffer((16, 16), "float32", name="dst", scope=scope)
    sid = tirx.Var("sid", "int32")
    before = kernel(unit(copy(a, dst), core=0, guard=sid == 0), buffers=[dst], params=[a])
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, before["main"].body))})
    if legal:
        assert {int(task.node["core_mask"]) for task in _tasks(transform.AssignCore()(before))} == {1}
    else:
        with pytest.raises(tvm.error.InternalError, match="unavailable on every legal execution core"):
            transform.AssignCore()(before)


def test_scalar_global_store_uses_the_existing_vector_core():
    src = tirx.decl_buffer((16,), "float32", name="src")
    out = tirx.decl_buffer((16,), "float32", name="out")
    ub = tirx.decl_buffer((16,), "float32", name="ub", scope="shared.dyn")
    before = kernel(seq(unit(copy(src, ub), core=1), unit(tirx.BufferStore(out, src[0] + 1, [0]), core=3)), buffers=[ub], params=[src, out])
    after = transform.ResolveCore()(before)
    assert {int(task.node["core_mask"]) for task in _tasks(after)} == {1}
