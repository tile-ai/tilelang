"""MaterializeMultiBuffer expands storage and all its views with one clock."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import allocated_buffer, calls, kernel, nodes, region, seq
from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit


@pytest.mark.parametrize("mode, owners, guarded", [("iteration", 1, False), ("counter", 1, True), ("counter", 2, True)])
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_physical_versions_use_the_prepared_clock(mode, owners, guarded, versions):
    before = copy_ring(mode=mode, versions=versions, owners=owners, guarded=guarded, prepared=True)
    storage = allocated_buffer(before, "ub").data
    after = transform.MaterializeMultiBuffer()(before)
    assert tuple(int(dim) for dim in allocated_buffer(after, "ub").shape) == ((versions, 64) if versions > 1 else (64,))
    accesses = [load for load in nodes(after, tirx.BufferLoad) if load.buffer.data.same_as(storage)]
    assert accesses and all(len(load.indices) == (2 if versions > 1 else 1) for load in accesses)
    if mode == "counter":
        clock = allocated_buffer(before, "epoch")[0]
    else:
        (loop,) = nodes(before, tirx.For)
        clock = loop.loop_var
    analyzer = tvm.arith.Analyzer()
    if versions > 1:
        assert all(analyzer.can_prove_equal(load.indices[0], clock % versions) for load in accesses)
    else:
        assert all(analyzer.can_prove_equal(load.indices[0], 0) for load in accesses)
        if mode == "counter":
            counter = allocated_buffer(before, "epoch")
            assert not any(buffer.data.same_as(counter.data) for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers)
            assert not any(
                node.buffer.data.same_as(counter.data) for kind in (tirx.BufferLoad, tirx.BufferStore) for node in nodes(after, kind)
            )
    assert all(
        "tl.multi_buffer_counter_map" not in loop.annotations and "tl.storage_epoch_guard_map" not in loop.annotations
        for loop in nodes(after, tirx.For)
    )
    root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
    assert "tl.buffer_versions_map" not in root.annotations
    assert int(root.annotations.get("tl.manual_multi_buffer", {}).get(storage, 1)) == versions


@pytest.mark.parametrize(
    "shape, dtype, pitch",
    [((4, 4), "float32", 16), ((4,), "float32", 16), ((32,), "float16", 32)],
    ids=["reshape", "subview", "reinterpret"],
)
def test_aliases_use_allocation_pitch(shape, dtype, pitch):
    base = tirx.decl_buffer((16,), "float32", name="base", scope="shared.dyn")
    alias = tirx.decl_buffer(shape, dtype, name="alias", data=base.data)
    counter = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var")
    i = tirx.Var("i", "int32")
    alias_index = [0] * len(shape)
    body = seq(
        unit(tirx.BufferStore(base, tirx.const(1, "float32"), [0])), unit(tirx.BufferStore(alias, tirx.const(2, dtype), alias_index))
    )
    loop = tirx.For(
        i,
        0,
        4,
        tirx.ForKind.SERIAL,
        body,
        annotations={
            "multi_buffer_eligible": [base.data],
            "tl.multi_buffer_counter_map": {base.data: counter},
            "tl.storage_epoch_guard_map": {base.data: tirx.const(True, "bool")},
        },
    )
    before = kernel(unit(loop, core=None), buffers=[base, counter], annotations={"tl.buffer_versions_map": {base.data: 2}})
    after = transform.MaterializeMultiBuffer()(before)
    stores = nodes(after, tirx.BufferStore)
    base_store, alias_store = stores
    assert tuple(int(dim) for dim in alias_store.buffer.shape) == (2, *shape)
    assert int(base_store.buffer.strides[0]) == 16
    assert int(alias_store.buffer.strides[0]) == pitch
    assert alias_store.buffer.data.same_as(base_store.buffer.data)
    tvm.ir.assert_structural_equal(alias_store.indices[0], base_store.indices[0])


@pytest.mark.parametrize("partial", [False, True], ids=["whole-tile", "partial-tile"])
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_broadcast_fill_initializes_every_version(partial, versions):
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    fill = tirx.Evaluate(
        tirx.Call("handle", tvm.ir.Op.get("tl.tileop.fill"), [region(ub, 2, shape=[32 if partial else 64]), tirx.const(0, "float32")])
    )
    fill = tirx.AttrStmt({"core_mask": tirx.IntImm("int64", 1), "tl.multi_buffer_broadcast_fill": [ub.data]}, "tl.ascend_task", 1, fill)
    owner = tirx.For(
        i,
        0,
        4,
        tirx.ForKind.SERIAL,
        unit(tirx.BufferStore(ub, tirx.const(1, "float32"), [0])),
        annotations={"multi_buffer_eligible": [ub.data]},
    )
    before = kernel(
        seq(unit(fill, core=None), unit(owner, core=None)), buffers=[ub], annotations={"tl.buffer_versions_map": {ub.data: versions}}
    )
    after = transform.MaterializeMultiBuffer()(before)
    (fill,) = calls(after, "tl.tileop.fill")
    destination = fill.args[0]
    expected_shape = (32 if partial else 64,)
    if versions > 1:
        expected_shape = (versions, *expected_shape)
    assert tuple(int(dim) for dim in destination.args[2:]) == expected_shape
    assert all(int(index) == 0 for index in destination.args[0].indices)
    tasks = [node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]
    assert tasks and all("tl.multi_buffer_broadcast_fill" not in task.node for task in tasks)


@pytest.mark.parametrize("kind", ["condition", "assume"])
def test_storage_reads_in_guards_follow_the_owner_version(kind):
    ub = tirx.decl_buffer((1,), "int32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((4,), "int32", name="out")
    i = tirx.Var("i", "int32")
    read = unit(tirx.BufferStore(out, ub[0], [i])).body
    guard = ub[0] >= 0
    guarded_read = tirx.IfThenElse(guard, read, None) if kind == "condition" else tirx.AttrStmt(guard, "tl.assume", "", read)
    body = seq(unit(tirx.BufferStore(ub, i, [0])), unit(guarded_read, core=None))
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [ub.data]})
    before = kernel(unit(loop, core=None), buffers=[ub], params=[out], annotations={"tl.buffer_versions_map": {ub.data: 2}})
    after = transform.MaterializeMultiBuffer()(before)
    if kind == "condition":
        expressions = [node.condition for node in nodes(after, tirx.IfThenElse)]
    else:
        expressions = [node.node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.assume"]
    assert len(expressions) == 1
    (load,) = nodes(expressions[0], tirx.BufferLoad)
    assert len(load.indices) == 2
    assert tvm.arith.Analyzer().can_prove_equal(load.indices[0], i % 2)


@pytest.mark.parametrize("start, step", [(0, 1), (3, 2)])
def test_iteration_clock_counts_logical_trips_across_nested_loops(start, step):
    ub = tirx.decl_buffer((1,), "int32", name="ub", scope="shared.dyn")
    outer, inner = tirx.Var("outer", "int32"), tirx.Var("inner", "int32")
    owner = tirx.For(
        inner,
        start,
        4,
        tirx.ForKind.SERIAL,
        unit(tirx.BufferStore(ub, inner, [0])),
        annotations={"multi_buffer_eligible": [ub.data]},
        step=step,
    )
    before = kernel(
        unit(tirx.For(outer, 0, 2, tirx.ForKind.SERIAL, unit(owner, core=None)), core=None),
        buffers=[ub],
        annotations={"tl.buffer_versions_map": {ub.data: 3}},
    )
    after = transform.MaterializeMultiBuffer()(before)
    (store,) = nodes(after, tirx.BufferStore)
    expected = (outer * (4 // step) + (inner - start) // step) % 3
    assert tvm.arith.Analyzer().can_prove_equal(store.indices[0], expected)


@pytest.mark.parametrize("versions", [(1, 1), (1, 2)], ids=["unit-only-group", "mixed-group"])
def test_counter_group_survives_only_for_physical_versions(versions):
    x = tirx.decl_buffer((1,), "int32", name="x", scope="shared.dyn")
    y = tirx.decl_buffer((1,), "int32", name="y", scope="shared.dyn")
    epoch = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var")
    i = tirx.Var("i", "int32")
    body = seq(unit(tirx.BufferStore(x, i, [0])), unit(tirx.BufferStore(y, i, [0])), unit(tirx.BufferStore(epoch, epoch[0] + 1, [0])))
    owner = tirx.For(
        i,
        0,
        4,
        tirx.ForKind.SERIAL,
        body,
        annotations={
            "multi_buffer_eligible": [x.data, y.data],
            "tl.multi_buffer_counter_map": {x.data: epoch, y.data: epoch},
            "tl.storage_epoch_guard_map": {x.data: tirx.const(True, "bool"), y.data: tirx.const(True, "bool")},
        },
    )
    before = kernel(
        seq(unit(tirx.BufferStore(epoch, 0, [0])), unit(owner, core=None)),
        buffers=[x, y, epoch],
        annotations={"tl.buffer_versions_map": {x.data: versions[0], y.data: versions[1]}},
    )
    after = transform.MaterializeMultiBuffer()(before)
    stores = nodes(after, tirx.BufferStore)
    (x_store,) = [store for store in stores if store.buffer.data.same_as(x.data)]
    (y_store,) = [store for store in stores if store.buffer.data.same_as(y.data)]
    assert len(x_store.indices) == 1
    counters = [buffer for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers if buffer.data.same_as(epoch.data)]
    updates = [store for store in stores if store.buffer.data.same_as(epoch.data)]
    if versions[1] > 1:
        assert len(counters) == 1 and len(updates) == 2
        assert len(y_store.indices) == 2
        assert tvm.arith.Analyzer().can_prove_equal(y_store.indices[0], epoch[0] % versions[1])
    else:
        assert not counters and not updates
        assert len(y_store.indices) == 1


@pytest.mark.parametrize("use", ["read", "guard", "compound-task", "loop-bound"])
def test_single_version_counter_elision_rejects_live_uses(use):
    before = copy_ring(versions=1, prepared=True)
    root = next(block for block in nodes(before, tirx.SBlock) if block.name_hint == "tilelang_root")
    epoch = allocated_buffer(before, "epoch")
    ub = allocated_buffer(before, "ub")
    write = tirx.BufferStore(ub, tirx.const(1, "float32"), [0])
    if use == "read":
        unexpected = unit(tirx.BufferStore(ub, tirx.Cast("float32", epoch[0]), [0]))
    elif use == "guard":
        unexpected = unit(write, guard=epoch[0] > 0)
    elif use == "compound-task":
        unexpected = unit(seq(tirx.BufferStore(epoch, epoch[0] + 1, [0]), write))
    else:
        unexpected = unit(tirx.For(tirx.Var("i", "int32"), 0, epoch[0], tirx.ForKind.SERIAL, unit(write)), core=None)
    before = kernel(
        seq(root.body, unexpected),
        buffers=root.alloc_buffers,
        params=list(before["main"].buffer_map.values()),
        annotations=root.annotations,
    )
    with pytest.raises(tvm.error.InternalError, match="Cannot elide single-version counter epoch"):
        transform.MaterializeMultiBuffer()(before)
