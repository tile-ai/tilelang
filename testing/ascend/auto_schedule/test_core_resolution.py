"""Core assignment follows storage access and scalar availability."""

from testing.ascend._ir import kernel_body

from tvm_tirx import ir
from testing.ascend._ir import (
    access_tensor,
    allocate,
    allocated_tensors,
    annotation,
    attribute,
    attribute_fields,
    attribute_key,
    call_annotations,
    launch_thread,
    loop_annotations,
    scope_annotations,
    scheduled_bind,
    scheduled_fields,
    single_statement,
    tensor,
)

import pytest
import tvm_tirx
import tvm_ffi
from tilelang.ascend import transform
from tvm_tirx import tirx
from tvm_tirx.tirx.layout import Axis, Iter
from testing.ascend._ir import allocated_buffer, calls, copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit


def _tasks(mod):
    return [node for node in nodes(mod, ir.RegionStmt) if attribute_key(node) == "tl.ascend_task"]


def _tensor_declarations(value):
    declarations = []
    for binding in nodes(value, ir.Bind):
        expression = binding.value.args[0] if scheduled_fields(binding) is not None else binding.value
        if isinstance(expression, ir.Call) and expression.op.name == "tirx.decl_tensor":
            declarations.append(binding)
    return declarations


@pytest.mark.parametrize("scope, expected", [("shared.dyn", 5), ("shared.l1", 2)])
@pytest.mark.parametrize("sids", [1, 2])
def test_copy_destination_selects_core(scope, expected, sids):
    a = tensor((16, 16), "float32", name="A")
    local = tensor((16, 16), "float32", name="local", scope=scope)
    before = kernel(
        unit(copy(a, local), core=0),
        buffers=[local],
        params=[a],
        annotations={"vector_count": ir.IntImm("int32", sids)},
        allocation_stage=0,
        allocation_core=0,
    )
    after = transform.AssignCore()(before)
    (task,) = _tasks(after)
    assert int(attribute_fields(task)["node"]["core_mask"]) == (expected & (3 if sids == 1 else 7))


def test_scalar_cannot_mix_cube_and_vector_local_memory():
    ub = tensor((1,), "int32", name="ub", scope="shared.dyn")
    l1 = tensor((1,), "int32", name="l1", scope="shared.l1")
    value = ir.Var("value", "int32")
    before = kernel(unit(ir.Bind(value, ub[0] + l1[0]), core=0), buffers=[ub, l1], allocation_stage=0, allocation_core=0)
    with pytest.raises((ValueError, RuntimeError), match="incompatible core-local memories"):
        transform.AssignCore()(before)


@pytest.mark.parametrize(
    "producer_scope, consumer_scope", [("shared.dyn", "shared.l1"), ("shared.l1", "shared.dyn")], ids=["vector-to-cube", "cube-to-vector"]
)
def test_core_local_condition_cannot_control_other_core(producer_scope, consumer_scope):
    pred = tensor((1,), "int32", name="pred", scope=producer_scope)
    a = tensor((16, 16), "float32", name="A")
    dst = tensor((16, 16), "float32", name="dst", scope=consumer_scope)
    condition = ir.Var("condition", "bool")
    before = kernel(
        seq(unit(ir.Bind(condition, pred[0] > 0), core=0), unit(copy(a, dst), core=0, guard=condition)),
        buffers=[pred, dst],
        params=[a],
        allocation_stage=0,
        allocation_core=0,
    )
    with pytest.raises((ValueError, RuntimeError), match="unavailable on every legal execution core"):
        transform.AssignCore()(before)


@pytest.mark.parametrize("mixed", [False, True], ids=["vector-only", "shared-with-cube"])
def test_counter_updates_cover_all_protocol_cores(mixed):
    before = copy_ring(prepared=True, owners=2, core=7)
    counter = allocated_buffer(before, "epoch")
    if mixed:
        # The first owner consumes L0C on Cube and shares the UB epoch with
        # Vector. Only this copy changes; the counter producer remains flexible.
        accum = tensor((1, 64), "float32", name="accum", scope="shared.l0c")
        ub = allocated_buffer(before, "ub")
        changed = False

        def add_cube_reader(node):
            nonlocal changed
            if (
                not changed
                and isinstance(node, ir.RegionStmt)
                and attribute_key(node) == "tl.ascend_task"
                and any(call.op.name == "tl.tileop.ascend_copy" for call in nodes(node.body, ir.Call))
            ):
                changed = True
                fields = attribute_fields(node)
                return attribute({**dict(fields["node"]), "core_mask": 2}, fields["key"], fields["value"], copy(accum, ub), loc=node.loc)
            return node

        func = before["main"]
        body = tvm_ffi.structural_map(func.body, (ir.RegionStmt, add_cube_reader))
        assert changed
        # The pass reasons about the L0C source scope; the allocation belongs
        # to the kernel even though no producer is needed in this pass input.
        body = ir.SeqStmt([unit(allocate(accum), core=7), *body.seq], loc=body.loc)
        before = tvm_tirx.IRModule({"main": func.with_body(body)})
    after = transform.ResolveCore()(before)
    writes = [task for task in _tasks(after) if any(access_tensor(store).same_as(counter) for store in nodes(task.body, ir.TensorStore))]
    assert len(writes) == 3
    assert {int(attribute_fields(task)["node"]["core_mask"]) for task in writes} == ({7} if mixed else {5})


def test_control_extent_restricts_flexible_scalar_task():
    ub = tensor((1,), "int32", name="ub", scope="shared.dyn")
    out = tensor((4,), "int32", name="out")
    i = ir.Var("i", "int32")
    loop = ir.For(i, ir.const(0), ir.const(0) + ub[0], ir.const(1), ir.ForKind.DEFAULT, unit(ir.TensorStore(out, [i], i), core=7))
    before = kernel(unit(loop, core=None), buffers=[ub], params=[out], allocation_stage=0)
    after = transform.ResolveCore()(before)
    assert {int(attribute_fields(task)["node"]["core_mask"]) for task in _tasks(after)} == {5}


@pytest.mark.parametrize("control", ["bound", "assume"])
def test_control_reads_must_be_available_on_child_core(control):
    ub = tensor((1,), "int32", name="ub", scope="shared.dyn")
    a = tensor((16, 16), "float32", name="A")
    l1 = tensor((16, 16), "float32", name="l1", scope="shared.l1")
    i = ir.Var("i", "int32")
    loop = ir.For(
        i, ir.const(0), ir.const(0) + (ub[0] if control == "bound" else 1), ir.const(1), ir.ForKind.DEFAULT, unit(copy(a, l1), core=2)
    )
    body = unit(loop, core=None)
    if control == "assume":
        body = attribute(ub[0] >= 0, "tilelang_assume", 1, body)
    before = kernel(body, buffers=[ub, l1], params=[a], allocation_stage=0)
    with pytest.raises((ValueError, RuntimeError), match="Cannot evaluate a loop bound or control guard"):
        transform.ResolveCore()(before)


@pytest.mark.parametrize("source", ["global", "shared.dyn"])
def test_epoch_guard_producer_must_cover_all_protocol_cores(source):
    pred = tensor((1,), "int32", name="pred", scope=source)
    ub = tensor((16, 16), "float32", name="ub", scope="shared.dyn")
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    out = tensor((16, 16), "float32", name="out")
    epoch = tensor((1,), "int32", name="epoch", scope="local.var")
    guard, i = ir.Var("guard", "bool"), ir.Var("i", "int32")
    loop = ir.For(
        i,
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        seq(
            unit(copy(accum, ub), core=2),
            unit(copy(ub, out), guard=guard),
            unit(ir.TensorStore(epoch, [0], epoch[0] + 1), core=7, guard=guard),
        ),
        annotations={"multi_buffer_eligible": [ub], "tl.multi_buffer_counter_map": {ub: epoch}, "tl.storage_epoch_guard_map": {ub: guard}},
    )
    before = kernel(
        seq(
            unit(ir.Bind(guard, pred[0] > 0), core=7 if source == "global" else 5),
            unit(ir.TensorStore(epoch, [0], 0), core=7),
            unit(loop, core=None),
        ),
        buffers=[ub, accum, epoch] + ([pred] if source != "global" else []),
        params=[out] + ([pred] if source == "global" else []),
        annotations={"tl.buffer_versions_map": {ub: 2}},
        allocation_stage=0,
    )
    if source != "global":
        with pytest.raises((ValueError, RuntimeError), match="storage-epoch guard .* is unavailable"):
            transform.ResolveCore()(before)
    else:
        after = transform.ResolveCore()(before)
        (producer,) = [binding for binding in nodes(after, ir.Bind) if binding.var.same_as(guard)]
        assert int(scheduled_fields(producer)["task"]["core_mask"]) == 7


@pytest.mark.parametrize("context", ["task-guard", "ancestor", "pure-kernel"])
def test_ambiguous_cross_core_pipe_uses_available_context(context):
    pred = tensor((1,), "int32", name="pred", scope="shared.dyn")
    a = tensor((16, 16), "float32", name="A")
    l1 = tensor((16, 16), "float32", name="l1", scope="shared.l1")
    flag = ir.Evaluate(ir.Call(tvm_tirx.ir.Op.get("tl.ascend_cross_core_set_flag"), [4, "PIPE_S", 3], ty="void"))
    flag_unit = unit(flag, core=0, guard=pred[0] > 0 if context == "task-guard" else None)
    if context == "ancestor":
        i = ir.Var("i", "int32")
        flag_unit = unit(ir.For(i, ir.const(0), ir.const(0) + pred[0], ir.const(1), ir.ForKind.DEFAULT, flag_unit), core=None)
    # A Cube copy supplies an opposing kernel context. Local scalar inputs
    # must take precedence; without those inputs the pure kernel selects Cube.
    before = kernel(seq(unit(copy(a, l1), core=0), flag_unit), buffers=[pred, l1], params=[a], allocation_stage=0, allocation_core=0)
    after = transform.AssignCore()(before)
    (placed,) = [
        task
        for task in _tasks(after)
        if isinstance(single_statement(task.body), ir.Evaluate) and single_statement(task.body).value.same_as(flag.value)
    ]
    assert int(attribute_fields(placed)["node"]["core_mask"]) == (2 if context == "pure-kernel" else 5)


@pytest.mark.parametrize("scope, core", [("shared.dyn", 1), ("shared.l1", 2)])
@pytest.mark.parametrize("assumption", [False, True])
def test_subblock_guard_is_specialized_before_core_assignment(scope, core, assumption):
    a = tensor((16, 16), "float32", name="A")
    dst = tensor((16, 16), "float32", name="dst", scope=scope)
    sid = ir.Var("sid", "int32")
    scheduled = unit(copy(a, dst), core=0, guard=sid == 0)
    if assumption:
        scheduled = attribute(sid >= 0, "tilelang_assume", 1, scheduled)
    before = kernel(scheduled, buffers=[dst], params=[a], allocation_stage=0, allocation_core=0)
    thread = sid
    before = tvm_tirx.IRModule(
        {
            "main": before["main"].with_body(
                launch_thread(ir.Var("block", "int32"), "blockIdx.x", 1, launch_thread(thread, "cthread", 2, before["main"].body))
            )
        }
    )
    after = transform.AssignCore()(before)
    assert {int(attribute_fields(task)["node"]["core_mask"]) for task in _tasks(after)} == {core}
    assert len(_tasks(after)) == 1
    assert all(not any(var.same_as(sid) for var in tirx.analysis.undefined_vars(task.body)) for task in _tasks(after))


def test_scalar_global_store_uses_the_existing_vector_core():
    src = tensor((16,), "float32", name="src")
    out = tensor((16,), "float32", name="out")
    ub = tensor((16,), "float32", name="ub", scope="shared.dyn")
    before = kernel(
        seq(unit(copy(src, ub), core=5), unit(ir.TensorStore(out, [0], src[0] + 1), core=7)),
        buffers=[ub],
        params=[src, out],
        allocation_stage=0,
    )
    after = transform.ResolveCore()(before)
    assert {int(attribute_fields(task)["node"]["core_mask"]) for task in _tasks(after)} == {5}


@pytest.mark.parametrize("field", ["shape", "strides", "elem_offset"])
@pytest.mark.parametrize("view", [False, True], ids=["allocation", "view"])
def test_subcore_buffer_expressions_preserve_definitions_and_views(field, view):
    sid = ir.Var("sid", "int32")
    storage = tensor((16, 64), "float32", name="storage", scope="shared.dyn")
    shape = (16, (sid + 1) * 16 if field == "shape" else 16)
    strides = ((sid + 1) * 16 if field == "strides" else 32, 1)
    layout = tirx.TileLayout.from_iters(
        [Iter(extent, stride, "m") for extent, stride in zip(shape, strides)],
        offset={"m": sid * 16 if field == "elem_offset" else 0},
    )
    buffer = tensor(shape, "float32", name="view" if view else "ub", scope="shared.dyn", layout=layout)
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    # The L0C producer is outside AssignCore's contract. Both accesses must
    # reuse the projected view, while allocation ownership follows its data.
    # Native SSA declarations dominate both tasks and operand-bearing metadata.
    declarations = [unit(ir.Bind(buffer, ir.Call("tirx.decl_tensor", [storage.data], ty_args=[buffer.ty])), core=0)] if view else []
    task = unit(copy(accum, buffer, dst_shape=[16, 16]), core=0)
    before = kernel(
        seq(
            *declarations,
            annotation({"test.buffer_metadata": {buffer: buffer}}),
            task,
            unit(copy(accum, buffer, dst_shape=[16, 16]), core=0),
        ),
        buffers=[accum, storage if view else buffer],
        allocation_stage=0,
        allocation_core=0,
    )
    thread = sid
    before = tvm_tirx.IRModule(
        {
            "main": before["main"].with_body(
                launch_thread(ir.Var("block", "int32"), "blockIdx.x", 1, launch_thread(thread, "cthread", 2, before["main"].body))
            )
        }
    )
    after = transform.AssignCore()(before)
    root = kernel_body(after)
    metadata = scope_annotations(root)["test.buffer_metadata"]
    tasks = _tasks(after)
    assert sorted(int(attribute_fields(task)["node"]["subcore"]) for task in tasks) == [0, 0, 1, 1]
    projected, projected_storage = {}, {}
    for task in tasks:
        owner = int(attribute_fields(task)["node"]["subcore"])
        (operation,) = calls(task, "tl.tileop.ascend_copy")
        result = access_tensor(operation.args[1])
        analyzer = tvm_tirx.sym.Analyzer()
        assert [int(analyzer.simplify(value)) for value in result.ty.shape] == [16, (owner + 1) * 16 if field == "shape" else 16]
        assert [int(analyzer.simplify(item.stride)) for item in result.ty.layout.shard] == [
            (owner + 1) * 16 if field == "strides" else 32,
            1,
        ]
        assert int(analyzer.simplify(result.ty.layout.offset.get(Axis.get("m"), ir.const(0)))) == (
            owner * 16 if field == "elem_offset" else 0
        )
        if owner in projected:
            assert result.same_as(projected[owner])
        projected[owner] = result
        assert metadata[result].same_as(result)
        if view:
            (definition,) = [binding for binding in _tensor_declarations(root) if binding.var.same_as(result)]
            (pointer,) = calls(definition, "tirx.tensor_data_ptr")
            storage_result = pointer.args[0]
        else:
            storage_result = result
        (allocation,) = [allocated for allocated in allocated_tensors(root, include_scheduled=True) if allocated.same_as(storage_result)]
        assert view or allocation.same_as(result)
        projected_storage[owner] = allocation
    assert set(projected) == {0, 1}
    definitions = _tensor_declarations(root) if view else nodes(root, ir.Bind)
    declared = [owner for binding in definitions for owner, result in projected.items() if binding.var.same_as(result)]
    assert sorted(declared) == [0, 1]
    assert not projected_storage[0].same_as(projected_storage[1])


@pytest.mark.parametrize("field", ["shape", "strides", "elem_offset"])
def test_nested_subcore_projection_reuses_binders_and_views(field):
    sid, outer, middle = (ir.Var(name, "int32") for name in ("sid", "outer", "middle"))
    # A source name may already have this suffix; projection must use identity.
    i, value = ir.Var("i_aiv0", "int32"), ir.Var("value", "int32")
    storage = tensor((64,), "int32", name="storage", scope="shared.dyn")
    shape = (i + 1 if field == "shape" else 1,)
    layout = tirx.TileLayout.from_iters(
        [Iter(shape[0], i + 1 if field == "strides" else 1, "m")],
        offset={"m": i if field == "elem_offset" else 0},
    )
    view = tensor(shape, "int32", name="view", scope="shared.dyn", layout=layout)
    definition = ir.Bind(view, ir.Call("tirx.decl_tensor", [storage.data], ty_args=[view.ty]))
    body = seq(
        ir.Bind(value, i + outer + middle + sid),
        definition,
        annotation({"test.buffer_metadata": {view: view}}),
        ir.TensorStore(view, [0], value),
    )
    inner = ir.For(i, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, body)
    nested = unit(ir.For(middle, ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, unit(inner, core=0)), core=None)
    before = kernel(
        unit(ir.For(outer, ir.const(0), ir.const(0) + (sid + 1), ir.const(1), ir.ForKind.DEFAULT, nested), core=None),
        buffers=[storage],
        allocation_stage=0,
        allocation_core=0,
    )
    thread = sid
    before = tvm_tirx.IRModule(
        {
            "main": before["main"].with_body(
                launch_thread(ir.Var("block", "int32"), "blockIdx.x", 1, launch_thread(thread, "cthread", 2, before["main"].body))
            )
        }
    )

    after = transform.AssignCore()(before)
    assert not tirx.analysis.undefined_vars(after["main"].body, after["main"].params)
    loops = nodes(after, ir.For)
    assert len(loops) == 6
    assert {loop.var.name for loop in loops} == {f"{var.name}_aiv{p}" for var in (outer, middle, i) for p in range(2)}
    tasks = _tasks(after)
    assert sorted(int(attribute_fields(task)["node"]["subcore"]) for task in tasks) == [0, 1]
    for task in tasks:
        owner = int(attribute_fields(task)["node"]["subcore"])
        (loop,) = nodes(task.body, ir.For)
        (binding,) = [binding for binding in nodes(task.body, ir.Bind) if not tirx.is_tensor_var(binding.var)]
        (definition,) = _tensor_declarations(task.body)
        (store,) = nodes(task.body, ir.TensorStore)
        result = definition.var
        assert binding.var.name == f"value_aiv{owner}"
        assert store.value.same_as(binding.var)
        assert result.name == f"view_aiv{owner}"
        assert access_tensor(store).same_as(result)
        assert loop_annotations(loop)["test.buffer_metadata"][result].same_as(result)
        (pointer,) = calls(definition, "tirx.tensor_data_ptr")
        assert pointer.args[0].same_as(allocated_buffer(after, f"storage_aiv{owner}"))
        expression = (
            result.ty.layout.offset.get(Axis.get("m"), ir.const(0))
            if field == "elem_offset"
            else result.ty.shape[0]
            if field == "shape"
            else result.ty.layout.shard[0].stride
        )
        tvm_tirx.ir.assert_structural_equal(expression, loop.var if field == "elem_offset" else loop.var + 1)


@pytest.mark.parametrize("field", ["shape", "strides", "elem_offset"])
@pytest.mark.parametrize("scope", ["global", "shared.l1"])
def test_subcore_view_metadata_projects_without_cloning_common_storage(field, scope):
    from tilelang.layout import Layout

    sid = ir.Var("sid", "int32")
    storage = tensor((16, 64), "float32", name="storage", scope=scope)
    shape = (16, (sid + 1) * 16 if field == "shape" else 16)
    strides = ((sid + 1) * 16 if field == "strides" else 32, 1)
    physical_layout = tirx.TileLayout.from_iters(
        [Iter(extent, stride, "m") for extent, stride in zip(shape, strides)],
        offset={"m": sid * 16 if field == "elem_offset" else 0},
    )
    view = tensor(shape, "float32", name="view", scope=scope, layout=physical_layout)
    destination = tensor((16, 16), "float32", name="destination", scope="shared.l1" if scope == "global" else "shared.l0a")
    annotations = {"test.buffer_metadata": {view: view}}
    layout = Layout((16, 16), lambda i, j: (i, j))
    if field != "shape":
        annotations["layout_map"] = {view: layout}
    # No address or control expression mentions sid: the view's metadata is
    # the only reason these Cube copies require subcore projection.
    before = kernel(
        seq(
            unit(ir.Bind(view, ir.Call("tirx.decl_tensor", [storage.data], ty_args=[view.ty])), core=0),
            annotation(annotations),
            unit(copy(view, destination, src_shape=[16, 16]), core=0),
        ),
        buffers=[destination] + ([storage] if scope != "global" else []),
        params=[storage] if scope == "global" else [],
        allocation_stage=0,
        allocation_core=0,
    )
    thread = sid
    before = tvm_tirx.IRModule(
        {
            "main": before["main"].with_body(
                launch_thread(ir.Var("block", "int32"), "blockIdx.x", 1, launch_thread(thread, "cthread", 2, before["main"].body))
            )
        }
    )
    after = transform.AssignCore()(before)
    root = kernel_body(after)
    tasks = _tasks(after)
    assert sorted(int(attribute_fields(task)["node"]["subcore"]) for task in tasks) == [0, 1]
    metadata = scope_annotations(root)["test.buffer_metadata"]
    for task in tasks:
        owner = int(attribute_fields(task)["node"]["subcore"])
        (operation,) = calls(task, "tl.tileop.ascend_copy")
        result = access_tensor(operation.args[0])
        (definition,) = [binding for binding in _tensor_declarations(root) if binding.var.same_as(result)]
        (pointer,) = calls(definition, "tirx.tensor_data_ptr")
        assert pointer.args[0].same_as(storage)
        assert access_tensor(operation.args[1]).same_as(destination)
        analyzer = tvm_tirx.sym.Analyzer()
        assert [int(analyzer.simplify(value)) for value in result.ty.shape] == [16, (owner + 1) * 16 if field == "shape" else 16]
        assert [int(analyzer.simplify(item.stride)) for item in result.ty.layout.shard] == [
            (owner + 1) * 16 if field == "strides" else 32,
            1,
        ]
        assert int(analyzer.simplify(result.ty.layout.offset.get(Axis.get("m"), ir.const(0)))) == (
            owner * 16 if field == "elem_offset" else 0
        )
        assert metadata[result].same_as(result)
        if field != "shape":
            # Operand metadata decodes a fresh Layout with the same complete
            # input domain and mapping; tensor/storage identities stay exact.
            tvm_tirx.ir.assert_structural_equal(scope_annotations(root)["layout_map"][result], layout)
    if scope == "global":
        assert any(parameter.same_as(storage) for parameter in after["main"].params)
    else:
        assert any(allocated.same_as(storage) for allocated in allocated_tensors(root, include_scheduled=True))


@pytest.mark.parametrize(
    "targets", [(0, 0), (1, 1), (0, 1), (None, 0)], ids=["aiv0", "aiv1", "different-subcores", "implicit-and-explicit"]
)
@pytest.mark.parametrize("per_core", [False, True])
def test_grouped_fixpipe_copies_require_matching_subcore_targets(targets, per_core):
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tensor((16, 16), "float32", name="ub", scope="shared.dyn")
    out = tensor((16, 16), "float32", name="out")
    sid = ir.Var("sid", "int32")
    copies = [
        copy(
            accum,
            ub,
            annotations={"unit_flag_ctrl": ir.IntImm("int32", flag)}
            | ({} if target is None else {"sub_blockid": ir.IntImm("int32", target)}),
        )
        for target, flag in zip(targets, (2, 3))
    ]
    # AssignCore only needs the copy destinations; the L0C producer is outside
    # this pass's contract.
    producer = unit(seq(*copies), core=0)
    if per_core:
        producer = unit(attribute({}, "tl.ascend_per_core_task", 1, producer.body), core=None)
    before = kernel(seq(producer, unit(copy(ub, out), core=0)), buffers=[accum, ub], params=[out], allocation_stage=0, allocation_core=0)
    thread = sid
    before = tvm_tirx.IRModule(
        {
            "main": before["main"].with_body(
                launch_thread(ir.Var("block", "int32"), "blockIdx.x", 1, launch_thread(thread, "cthread", 2, before["main"].body))
            )
        }
    )
    if targets[0] != targets[1]:
        with pytest.raises((ValueError, RuntimeError), match="same sub_blockid.*separate T.Task"):
            transform.AssignCore()(before)
    else:
        after = transform.AssignCore()(before)
        (task,) = [
            task
            for task in nodes(after, ir.RegionStmt)
            if attribute_key(task) == ("tl.ascend_per_core_task" if per_core else "tl.ascend_task")
            and any(access_tensor(call.args[0]).same_as(accum) for call in calls(task, "tl.tileop.ascend_copy"))
        ]
        projected = calls(task, "tl.tileop.ascend_copy")
        assert len(projected) == len(copies)
        assert int(attribute_fields(task)["node"]["core_mask"]) == 2
        assert {int(call_annotations(call)["sub_blockid"]) for call in projected} == {targets[0]}
        assert [int(call_annotations(call)["unit_flag_ctrl"]) for call in projected] == [2, 3]


@pytest.mark.parametrize("route_kind", ["bind", "load", "loop-var"])
def test_fixpipe_route_keeps_its_task_local_scope(route_kind):
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tensor((16, 16), "float32", name="ub", scope="shared.dyn")
    routes = tensor((1,), "int32", name="routes")
    sid, route = ir.Var("sid", "int32"), ir.Var("route", "int32")
    state = tensor((1,), "int32", name="state", scope="local.var")
    target = state[0] if route_kind == "load" else route
    body = copy(accum, ub, annotations={"sub_blockid": target})
    if route_kind == "bind":
        body = seq(ir.Bind(route, routes[0] % 2), body)
    elif route_kind == "load":
        body = seq(ir.TensorStore(state, [0], routes[0] % 2), body)
    elif route_kind == "loop-var":
        body = ir.For(route, ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, body)
    before = kernel(
        unit(body, core=0),
        buffers=[accum, ub] + ([state] if route_kind == "load" else []),
        params=[routes],
        allocation_stage=0,
        allocation_core=0,
    )
    thread = sid
    before = tvm_tirx.IRModule(
        {
            "main": before["main"].with_body(
                launch_thread(ir.Var("block", "int32"), "blockIdx.x", 1, launch_thread(thread, "cthread", 2, before["main"].body))
            )
        }
    )
    after = transform.AssignCore()(before)
    assert not tirx.analysis.undefined_vars(after["main"].body, after["main"].params)
    tasks = _tasks(after)
    assert len(tasks) == 2
    for task in tasks:
        (branch,) = nodes(task.body, ir.If)
        (operation,) = calls(branch.then_case, "tl.tileop.ascend_copy")
        if route_kind == "bind":
            expected = nodes(task.body, ir.Bind)[0].var
        elif route_kind == "loop-var":
            expected = nodes(task.body, ir.For)[0].var
        else:
            expected = access_tensor(nodes(task.body, ir.TensorStore)[0])[0]
        tvm_tirx.ir.assert_structural_equal(branch.condition, expected == int(call_annotations(operation)["sub_blockid"]))


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
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tensor((16, 16), "float32", name="ub", scope="shared.dyn")
    out = tensor((16, 16), "float32", name="out")
    routes = tensor((1,), "int32", name="routes")
    sid, route = ir.Var("sid", "int32"), ir.Var("route", "int32")
    selector = ir.Var("selector", "int32")
    sid_alias = ir.Var("sid_alias", "int32")
    if effect == "store":
        side_effect = ir.TensorStore(routes, [0], routes[0] + 1)
    elif effect == "gm-copy":
        side_effect = copy(accum, out)
    elif effect == "pure-call":
        side_effect = ir.Evaluate(ir.Call("tirx.call_pure_extern", [ir.StringImm("read_route"), route], ty="int32"))
    elif effect == "route-call":
        side_effect = ir.Evaluate(0)
    else:
        side_effect = ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("record_route"), route], ty="int32"))
    implicit = routing.startswith("implicit")
    sid_routing = routing.startswith("sid")
    single_projection = "single-projection" in routing
    target = ir.IntImm("int32", 0) if routing == "constant" else route
    if sid_routing:
        target = sid
        if routing == "sid-expression":
            target = ir.prim.Select(selector % 2 == 0, sid, 1 - sid)
        elif routing == "sid-alias":
            target = sid_alias
    elif effect == "route-call" and routing != "constant":
        target = ir.Call("tirx.call_extern", [ir.StringImm("next_route")], ty="int32")
    elif effect == "pure-call" and routing != "constant":
        target = ir.Call("tirx.call_pure_extern", [ir.StringImm("read_route"), routes[0]], ty="int32")
    fix = copy(accum, ub, annotations={} if implicit else {"sub_blockid": target})
    if context == "separate":
        body = seq(unit(fix, core=0), unit(side_effect, core=0))
    else:
        effects = (fix, side_effect) if implicit or sid_routing else (side_effect, fix)
        body = seq(ir.Bind(route, routes[0] % 2), *effects)
        body = unit(body, core=0, guard=sid == 0 if single_projection and context != "guarded-loop" else None)
    if routing == "sid-alias":
        body = seq(unit(ir.Bind(sid_alias, sid), core=0), body)
    if context not in ("task", "separate"):
        i = ir.Var("i", "int32")
        if context == "nested-loop":
            body = unit(
                ir.For(ir.Var("j", "int32"), ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, body), core=None
            )
        body = unit(
            ir.For(i, ir.const(0), ir.const(0) + (sid + 2), ir.const(1), ir.ForKind.DEFAULT, body),
            core=None,
            guard=sid == 0 if single_projection and context == "guarded-loop" else None,
        )
    before = kernel(
        body,
        buffers=[accum, ub],
        params=[routes, out] + ([selector] if routing == "sid-expression" else []),
        allocation_stage=0,
        allocation_core=0,
    )
    thread = sid
    before = tvm_tirx.IRModule(
        {
            "main": before["main"].with_body(
                launch_thread(ir.Var("block", "int32"), "blockIdx.x", 1, launch_thread(thread, "cthread", 2, before["main"].body))
            )
        }
    )
    repeated = (routing in ("dynamic", "implicit") or sid_routing) and not single_projection
    if repeated and context != "separate" and effect != "pure-call":
        with pytest.raises((ValueError, RuntimeError), match="Fixpipe routing.*external side effects.*separate T.Task"):
            transform.AssignCore()(before)
    else:
        after = transform.AssignCore()(before)
        expected_tasks = 3 if context == "separate" else 2 if repeated else 1
        assert len(_tasks(after)) == expected_tasks
        if effect == "gm-copy":
            gm_copies = [call for call in calls(after, "tl.tileop.ascend_copy") if access_tensor(call.args[1]).same_as(out)]
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
    src = tensor((16,), "float32", name="src")
    buffers = [tensor((16,), "float32", name=f"ub{sid}", scope="shared.dyn") for sid in range(2)]
    i = ir.Var("i", "int32")
    body = []
    for sid, core in enumerate((1, 4)):
        marker = attribute({"core_mask": core | 2, "subcore": sid}, "tl.ascend_task", 1, ir.Break())
        body.extend((unit(marker, core=None, guard=i >= sid + 1), unit(copy(src, buffers[sid]), core=core)))
    l1 = tensor((16,), "float32", name="l1", scope="shared.l1")
    prefix, suffix = [], []
    for sid in cube_owners:
        metadata = {"core_mask": 2}
        if sid is not None:
            metadata["subcore"] = sid
        cube = unit(attribute(metadata, "tl.ascend_task", 1, copy(src, l1)), core=None)
        if via_scalar:
            value = tensor((1,), "int32", name=f"value_{sid}", scope="local.var")
            buffers.append(value)
            prefix.append(unit(ir.TensorStore(value, [0], 0), core=7))
            suffix.append(unit(cube.body, core=None, guard=value[0] > 0))
            cube = unit(attribute({**metadata, "core_mask": 7}, "tl.ascend_task", 1, ir.TensorStore(value, [0], i)), core=None)
        if nested:
            cube = unit(
                ir.For(ir.Var("j", "int32"), ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, cube), core=None
            )
        body.append(cube)
    # A common scalar producer may be recomputed on Cube without owning its
    # loop control. Its fallback placement must not broaden either break.
    body.append(unit(ir.Bind(ir.Var("value", "int32"), i), core=7))
    buffers.append(l1)
    before = kernel(
        seq(
            *prefix,
            unit(ir.For(i, ir.const(0), ir.const(0) + ir.const(3), ir.const(1), ir.ForKind.DEFAULT, seq(*body)), core=None),
            *suffix,
        ),
        buffers=buffers,
        params=[src],
        allocation_stage=0,
    )
    if cube_owners in ((None,), (0, 1)):
        with pytest.raises((ValueError, RuntimeError), match="Independent subcore loop_break.*separate loop"):
            transform.ResolveCore()(before)
        return
    after = transform.ResolveCore()(before)
    exits = [task for task in _tasks(after) if nodes(task, ir.Break)]
    assert len(exits) == 2
    assert {int(attribute_fields(task)["node"]["subcore"]): int(attribute_fields(task)["node"]["core_mask"]) for task in exits} == {
        sid: core | (2 if sid in cube_owners else 0) for sid, core in enumerate((1, 4))
    }


@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("consumer", ["copy", "loop-bound", "loop-break"])
@pytest.mark.parametrize("shared_consumer", [False, True])
def test_projected_loop_exit_follows_scalar_demand_from_later_consumers(owner, consumer, shared_consumer):
    src = tensor((4, 16), "float32", name="src")
    l1 = tensor((16,), "float32", name="l1", scope="shared.l1")
    ub = tensor((16,), "float32", name="ub", scope="shared.dyn")
    count = tensor((1,), "int32", name="count", scope="local.var")
    i, j, offset = (ir.Var(name, "int32") for name in ("i", "j", "offset"))
    stop = ir.Var("stop", "bool")

    def task(body, core, sid=None, guard=None):
        metadata = {"core_mask": core}
        if sid is not None:
            metadata["subcore"] = sid
        if isinstance(body, ir.Bind):
            binding = scheduled_bind(body, 0, metadata)
            return ir.If(guard, [binding], None) if guard is not None else binding
        return unit(attribute(metadata, "tl.ascend_task", 1, body), core=None, guard=guard)

    loop = unit(
        ir.For(
            i,
            ir.const(0),
            ir.const(0) + ir.const(4),
            ir.const(1),
            ir.ForKind.DEFAULT,
            seq(
                task(ir.Bind(stop, i == 1), 7),
                task(ir.Break(), 3, 0, stop),
                task(ir.Break(), 6, 1, stop),
                task(ir.TensorStore(count, [0], count[0] + 1), 7, owner),
                task(copy(src, ub, src_indices=[i, 0], src_shape=[1, 16]), 4 if owner == 0 else 1, 1 - owner),
            ),
        ),
        core=None,
    )
    consumer_sid = None if shared_consumer else owner
    read = task(copy(src, l1, src_indices=[offset if consumer == "copy" else 0, 0], src_shape=[1, 16]), 2, consumer_sid)
    if consumer == "copy":
        read = seq(task(ir.Bind(offset, count[0] - 1), 7), read)
    else:
        if consumer == "loop-break":
            read = seq(task(ir.Break(), 7, guard=j >= count[0]), read)
        read = unit(
            ir.For(j, ir.const(0), ir.const(0) + (count[0] if consumer == "loop-bound" else 4), ir.const(1), ir.ForKind.DEFAULT, read),
            core=None,
        )
    before = kernel(
        seq(task(ir.TensorStore(count, [0], 0), 7, owner), loop, read), buffers=[l1, ub, count], params=[src], allocation_stage=0
    )
    after = transform.ResolveCore()(before)
    exits = [task for task in _tasks(after) if nodes(task, ir.Break) and "subcore" in attribute_fields(task)["node"]]
    assert {int(attribute_fields(task)["node"]["subcore"]): int(attribute_fields(task)["node"]["core_mask"]) for task in exits} == {
        sid: 2 if sid == owner else core for sid, core in enumerate((1, 4))
    }
    count_writes = [task for task in _tasks(after) if any(access_tensor(store).same_as(count) for store in nodes(task, ir.TensorStore))]
    assert len(count_writes) == 2
    assert all(int(attribute_fields(task)["node"]["core_mask"]) & 2 for task in count_writes)


def test_unit_flag_pairing_between_source_copies_remains_caller_managed():
    accum = tensor((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tensor((16, 16), "float32", name="ub", scope="shared.dyn")
    sid = ir.Var("sid", "int32")
    fix = copy(accum, ub, annotations={"unit_flag_ctrl": ir.IntImm("int32", 3)})
    # Reusing an IR node still describes two source operations. This pass does
    # not validate their caller-managed protocol; no such kernel is executed.
    before = kernel(
        seq(unit(fix, core=0, guard=sid == 0), unit(fix, core=0, guard=sid == 1)),
        buffers=[accum, ub],
        allocation_stage=0,
        allocation_core=0,
    )
    thread = sid
    before = tvm_tirx.IRModule(
        {
            "main": before["main"].with_body(
                launch_thread(ir.Var("block", "int32"), "blockIdx.x", 1, launch_thread(thread, "cthread", 2, before["main"].body))
            )
        }
    )
    after = transform.AssignCore()(before)
    copies = calls(after, "tl.tileop.ascend_copy")
    assert len(copies) == 2
    assert {int(call_annotations(call)["sub_blockid"]) for call in copies} == {0, 1}
    assert all(int(call_annotations(call)["unit_flag_ctrl"]) == 3 for call in copies)
