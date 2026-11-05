"""MaterializeMultiBuffer expands storage and all its views with one clock."""

from tvm_tirx import ir
import tvm_ffi
from testing.ascend._ir import (
    access_tensor,
    allocated_tensors,
    attribute,
    attribute_fields,
    attribute_key,
    loop_annotations,
    sblock_annotations,
    sblock_metadata,
    sblocks,
    tensor,
)

import pytest
import tvm_tirx as tvm
from tilelang.ascend import transform
from testing.ascend._ir import allocated_buffer, calls, kernel, nodes, region, seq
from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit


@pytest.mark.parametrize("mode, owners, guarded", [("iteration", 1, False), ("counter", 1, True), ("counter", 2, True)])
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_physical_versions_use_the_prepared_clock(mode, owners, guarded, versions):
    before = copy_ring(mode=mode, versions=versions, owners=owners, guarded=guarded, prepared=True)
    after = transform.MaterializeMultiBuffer()(before)
    storage = allocated_buffer(after, "ub")
    assert tuple(int(dim) for dim in allocated_buffer(after, "ub").ty.shape) == ((versions, 64) if versions > 1 else (64,))
    accesses = [
        operand
        for call in calls(after, "tl.tileop.ascend_copy")
        for operand in call.args
        if isinstance(operand, ir.TensorRegion) and operand.source.same_as(storage)
    ]
    assert accesses and all(len(operand.region) == (2 if versions > 1 else 1) for operand in accesses)
    if mode == "counter":
        clock = allocated_buffer(before, "epoch")[0]
    else:
        (loop,) = nodes(before, ir.For)
        clock = loop.var
    analyzer = tvm.sym.Analyzer()
    if versions > 1:
        assert all(analyzer.can_prove_equal(operand.region[0].begin, clock % versions) for operand in accesses)
    else:
        assert all(analyzer.can_prove_equal(operand.region[0].begin, 0) for operand in accesses)
        if mode == "counter":
            counter = allocated_buffer(before, "epoch")
            assert not any(buffer.same_as(counter) for buffer in allocated_tensors(after, include_scheduled=True))
            assert not any(access_tensor(node).same_as(counter) for kind in (ir.TensorLoad, ir.TensorStore) for node in nodes(after, kind))
    assert all(
        "tl.multi_buffer_counter_map" not in loop_annotations(loop) and "tl.storage_epoch_guard_map" not in loop_annotations(loop)
        for loop in nodes(after, ir.For)
    )
    root = next(block for block in sblocks(after) if sblock_metadata(block)["name"] == "tilelang_root")
    assert "tl.buffer_versions_map" not in sblock_annotations(root)
    assert int(sblock_annotations(root).get("tl.manual_multi_buffer", {}).get(storage, 1)) == versions


@pytest.mark.parametrize(
    "scope, base_dtype, shape, dtype, pitch",
    [
        ("shared.dyn", "float32", (4, 4), "float32", 16),
        ("shared.dyn", "float32", (4,), "float32", 16),
        ("shared.dyn", "float32", (32,), "float16", 32),
        ("shared.ss", "uint32", (8,), "uint64", 8),
    ],
    ids=["reshape", "subview", "reinterpret", "ss-packed"],
)
def test_aliases_use_allocation_pitch(scope, base_dtype, shape, dtype, pitch):
    base = tensor((16,), base_dtype, name="base", scope=scope)
    alias = tensor(shape, dtype, name="alias", scope=scope)
    pointer = base.data
    if dtype != base_dtype:
        pointer = ir.Call("tirx.reinterpret", [pointer], ty=ir.PtrType(dtype, base.ty.scope))
    declaration = unit(ir.Bind(alias, ir.Call("tirx.decl_tensor", [pointer], ty_args=[alias.ty])), core=7)
    counter = tensor((1,), "int32", name="epoch", scope="local.var")
    i = ir.Var("i", "int32")
    alias_index = [0] * len(shape)
    body = seq(unit(ir.TensorStore(base, [0], ir.const(1, base_dtype))), unit(ir.TensorStore(alias, alias_index, ir.const(2, dtype))))
    loop = ir.For(
        i,
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        body,
        annotations={
            "multi_buffer_eligible": [base],
            "tl.multi_buffer_counter_map": {base: counter},
            "tl.storage_epoch_guard_map": {base: ir.const(True, "bool")},
        },
    )
    before = kernel(
        seq(declaration, unit(loop, core=None)),
        buffers=[base, counter],
        annotations={"tl.buffer_versions_map": {base: 2}},
        allocation_stage=0,
    )
    after = transform.MaterializeMultiBuffer()(before)
    stores = nodes(after, ir.TensorStore)
    base_store, alias_store = stores
    assert tuple(int(dim) for dim in access_tensor(alias_store).ty.shape) == (2, *shape)
    assert int(access_tensor(base_store).ty.layout.shard[0].stride) == 16
    assert int(access_tensor(alias_store).ty.layout.shard[0].stride) == pitch
    assert tuple(int(dim) for dim in allocated_buffer(after, "base").ty.shape) == (2, 16)
    (view_definition,) = [binding for binding in nodes(after, ir.Bind) if binding.var.same_as(access_tensor(alias_store))]
    (storage_pointer,) = calls(view_definition, "tirx.tensor_data_ptr")
    assert storage_pointer.args[0].same_as(access_tensor(base_store))
    tvm.ir.assert_structural_equal(alias_store.indices[0], base_store.indices[0])


@pytest.mark.parametrize("partial", [False, True], ids=["whole-tile", "partial-tile"])
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_broadcast_fill_initializes_every_version(partial, versions):
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    i = ir.Var("i", "int32")
    fill = ir.Evaluate(
        tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
            ir.PrimType("void"),
            tvm.ir.Op.get("tl.tileop.fill"),
            [region(ub, 2, shape=[32 if partial else 64]), ir.const(0, "float32")],
            {},
            ir.UnknownLoc(),
        )
    )
    fill = attribute({"core_mask": ir.IntImm("int64", 5), "tl.multi_buffer_broadcast_fill": [ub]}, "tl.ascend_task", 1, fill)
    owner = ir.For(
        i,
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        unit(ir.TensorStore(ub, [0], ir.const(1, "float32"))),
        annotations={"multi_buffer_eligible": [ub]},
    )
    before = kernel(
        seq(unit(fill, core=None), unit(owner, core=None)),
        buffers=[ub],
        annotations={"tl.buffer_versions_map": {ub: versions}},
        allocation_stage=0,
    )
    after = transform.MaterializeMultiBuffer()(before)
    (fill,) = calls(after, "tl.tileop.fill")
    destination = fill.args[0]
    expected_shape = (32 if partial else 64,)
    if versions > 1:
        expected_shape = (versions, *expected_shape)
    assert isinstance(destination, ir.TensorRegion)
    assert tuple(int(dimension.extent) for dimension in destination.region) == expected_shape
    assert all(int(dimension.begin) == 0 for dimension in destination.region)
    tasks = [node for node in nodes(after, ir.RegionStmt) if attribute_key(node) == "tl.ascend_task"]
    assert tasks and all("tl.multi_buffer_broadcast_fill" not in attribute_fields(task)["node"] for task in tasks)


@pytest.mark.parametrize("kind", ["condition", "assume"])
def test_storage_reads_in_guards_follow_the_owner_version(kind):
    ub = tensor((1,), "int32", name="ub", scope="shared.dyn")
    out = tensor((4,), "int32", name="out")
    i = ir.Var("i", "int32")
    read = unit(ir.TensorStore(out, [i], ub[0]))
    guard = ub[0] >= 0
    guarded_read = ir.If(guard, [read], None) if kind == "condition" else attribute(guard, "tilelang_assume", "", read)
    body = seq(unit(ir.TensorStore(ub, [0], i)), guarded_read)
    loop = ir.For(
        i, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, body, annotations={"multi_buffer_eligible": [ub]}
    )
    before = kernel(unit(loop, core=None), buffers=[ub], params=[out], annotations={"tl.buffer_versions_map": {ub: 2}}, allocation_stage=0)
    after = transform.MaterializeMultiBuffer()(before)
    if kind == "condition":
        expressions = [node.condition for node in nodes(after, ir.If)]
    else:
        expressions = [attribute_fields(node)["node"] for node in nodes(after, ir.RegionStmt) if attribute_key(node) == "tilelang_assume"]
    assert len(expressions) == 1
    (load,) = nodes(expressions[0], ir.TensorLoad)
    assert len(load.indices) == 2
    assert tvm.sym.Analyzer().can_prove_equal(load.indices[0], i % 2)


@pytest.mark.parametrize("start, step", [(0, 1), (3, 2)])
def test_iteration_clock_counts_logical_trips_across_nested_loops(start, step):
    ub = tensor((1,), "int32", name="ub", scope="shared.dyn")
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    owner = ir.For(
        inner,
        start,
        start + ir.const(4),
        step,
        ir.ForKind.DEFAULT,
        unit(ir.TensorStore(ub, [0], inner)),
        annotations={"multi_buffer_eligible": [ub]},
    )
    before = kernel(
        unit(ir.For(outer, ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, unit(owner, core=None)), core=None),
        buffers=[ub],
        annotations={"tl.buffer_versions_map": {ub: 3}},
        allocation_stage=0,
    )
    after = transform.MaterializeMultiBuffer()(before)
    (store,) = nodes(after, ir.TensorStore)
    expected = (outer * (4 // step) + (inner - start) // step) % 3
    assert tvm.sym.Analyzer().can_prove_equal(store.indices[0], expected)


@pytest.mark.parametrize("versions", [(1, 1), (1, 2)], ids=["unit-only-group", "mixed-group"])
def test_counter_group_survives_only_for_physical_versions(versions):
    x = tensor((1,), "int32", name="x", scope="shared.dyn")
    y = tensor((1,), "int32", name="y", scope="shared.dyn")
    epoch = tensor((1,), "int32", name="epoch", scope="local.var")
    i = ir.Var("i", "int32")
    body = seq(unit(ir.TensorStore(x, [0], i)), unit(ir.TensorStore(y, [0], i)), unit(ir.TensorStore(epoch, [0], epoch[0] + 1)))
    owner = ir.For(
        i,
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        body,
        annotations={
            "multi_buffer_eligible": [x, y],
            "tl.multi_buffer_counter_map": {x: epoch, y: epoch},
            "tl.storage_epoch_guard_map": {x: ir.const(True, "bool"), y: ir.const(True, "bool")},
        },
    )
    before = kernel(
        seq(unit(ir.TensorStore(epoch, [0], 0)), unit(owner, core=None)),
        buffers=[x, y, epoch],
        annotations={"tl.buffer_versions_map": {x: versions[0], y: versions[1]}},
        allocation_stage=0,
    )
    after = transform.MaterializeMultiBuffer()(before)
    stores = nodes(after, ir.TensorStore)
    (x_store,) = [store for store in stores if access_tensor(store).same_as(allocated_buffer(after, "x"))]
    (y_store,) = [store for store in stores if access_tensor(store).same_as(allocated_buffer(after, "y"))]
    assert len(x_store.indices) == 1
    counters = [buffer for buffer in allocated_tensors(after, include_scheduled=True) if buffer.same_as(epoch)]
    updates = [store for store in stores if access_tensor(store).same_as(epoch)]
    if versions[1] > 1:
        assert len(counters) == 1 and len(updates) == 2
        assert len(y_store.indices) == 2
        assert tvm.sym.Analyzer().can_prove_equal(y_store.indices[0], epoch[0] % versions[1])
    else:
        assert not counters and not updates
        assert len(y_store.indices) == 1


@pytest.mark.parametrize("use", ["read", "guard", "compound-task", "loop-bound"])
def test_single_version_counter_elision_rejects_live_uses(use):
    before = copy_ring(versions=1, prepared=True)
    root = next(block for block in sblocks(before) if sblock_metadata(block)["name"] == "tilelang_root")
    epoch = allocated_buffer(before, "epoch")
    ub = allocated_buffer(before, "ub")
    write = ir.TensorStore(ub, [0], ir.const(1, "float32"))
    if use == "read":
        unexpected = unit(ir.TensorStore(ub, [0], ir.prim.Cast("float32", epoch[0])))
    elif use == "guard":
        unexpected = unit(write, guard=epoch[0] > 0)
    elif use == "compound-task":
        unexpected = unit(seq(ir.TensorStore(epoch, [0], epoch[0] + 1), write))
    else:
        unexpected = unit(
            ir.For(ir.Var("i", "int32"), ir.const(0), ir.const(0) + epoch[0], ir.const(1), ir.ForKind.DEFAULT, unit(write)), core=None
        )
    fields = attribute_fields(root)
    root = attribute(fields["node"], fields["key"], fields["value"], [*root.body.seq, unexpected], loc=root.loc)
    before = ir.IRModule({"main": before["main"].with_body([root])})
    message = (
        "Single-version counter is used by a control header"
        if use in ("guard", "loop-bound")
        else "Single-version counter is read by a task that must survive"
    )
    with pytest.raises((ValueError, RuntimeError), match=message):
        transform.MaterializeMultiBuffer()(before)


def test_bound_data_and_sf_preserve_user_leading_dimensions_under_one_ring():
    data = tensor((2, 3, 32, 128), "float8_e4m3fn", name="data", scope="shared.l0a")
    sf = tensor((2, 3, 32, 2), "uint16", name="sf", scope="shared.l0a.sf")
    i = ir.Var("i", "int32")
    body = seq(*[unit(ir.TensorStore(buf, [1, 2, 0, 0], ir.const(1, buf.ty.dtype)), core=2) for buf in (data, sf)])
    owner = ir.For(
        i, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, body, annotations={"multi_buffer_eligible": [data, sf]}
    )
    before = kernel(
        unit(owner, core=None),
        buffers=[data, sf],
        annotations={
            "tl.l0_sf_bindings": {sf: data},
            "tl.buffer_versions_map": {data: 2, sf: 2},
        },
        allocation_stage=0,
    )
    after = transform.MaterializeMultiBuffer()(before)
    for name, tail in [("data", (32, 128)), ("sf", (32, 2))]:
        assert tuple(int(x) for x in allocated_buffer(after, name).ty.shape) == (2, 2, 3, *tail)
    stores = nodes(after, ir.TensorStore)
    assert len(stores) == 2
    for store in stores:
        assert tvm.sym.Analyzer().can_prove_equal(store.indices[0], i % 2)
        assert [int(x) for x in store.indices[1:]] == [1, 2, 0, 0]
