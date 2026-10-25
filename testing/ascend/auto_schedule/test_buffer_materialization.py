"""MaterializeMultiBuffer expands native storage and all views with one clock."""

import pytest
from tvm_tirx import ir, sym, tirx
from tilelang.ascend import transform
from tilelang._compat.tensor import TensorView, tensor_strides
from testing.ascend._ir import allocated_buffer, calls, kernel, nodes, region, seq
from testing.ascend.auto_schedule._scheduled_ir import _marker, copy_ring, unit


def _tensor(name, shape, dtype, scope="global"):
    return TensorView.create(name, shape, dtype, scope=scope).tensor


def _root(module):
    return next(scope for scope in nodes(module, ir.RegionStmt) if scope.attrs.get("name_hint") == "tilelang_root")


def _materialize(module):
    assert tirx.analysis.check_well_formed(module)
    result = transform.MaterializeMultiBuffer()(module)
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def _stores(module, name):
    return [store for store in nodes(module, ir.TensorStore) if store.dest.name == name]


@pytest.mark.parametrize("mode, owners, guarded", [("iteration", 1, False), ("counter", 1, True), ("counter", 2, True)])
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_physical_versions_use_the_prepared_clock(mode, owners, guarded, versions):
    before = copy_ring(mode=mode, versions=versions, owners=owners, guarded=guarded, prepared=True)
    after = _materialize(before)
    storage = allocated_buffer(after, "ub")
    assert tuple(int(dim) for dim in storage.shape) == ((versions, 64) if versions > 1 else (64,))
    accesses = [access for access in nodes(after, ir.TensorRegion) if access.source.same_as(storage)]
    assert accesses and all(len(access.region) == (2 if versions > 1 else 1) for access in accesses)
    if mode == "counter":
        clock = allocated_buffer(before, "epoch")[0]
    else:
        (loop,) = nodes(before, ir.For)
        clock = loop.var
    analyzer = sym.Analyzer()
    assert all(analyzer.can_prove_equal(access.region[0].begin, clock % versions if versions > 1 else 0) for access in accesses)
    if versions == 1 and mode == "counter":
        assert not any(variable.name == "epoch" for variable in nodes(after, ir.Var))
    assert all(
        "tl.multi_buffer_counter_map" not in loop.annotations and "tl.storage_epoch_guard_map" not in loop.annotations
        for loop in nodes(after, ir.For)
    )
    root = _root(after)
    assert "tl.buffer_versions_map" not in root.attrs
    assert int(root.attrs.get("tl.manual_multi_buffer", {}).get(storage, 1)) == versions


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
    base = _tensor("base", (16,), base_dtype, scope)
    alias = TensorView.create("alias", shape, dtype, scope=scope)
    counter = _tensor("epoch", (1,), "int32", "local.var")
    index = ir.Var("index", "int32")
    body = seq(
        unit(ir.TensorStore(base, [0], ir.const(1, base_dtype))),
        unit(alias.store([0] * len(shape), ir.const(2, dtype))),
    )
    loop = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        body,
        annotations={
            "multi_buffer_eligible": [base],
            "tl.multi_buffer_counter_map": {base: counter},
            "tl.storage_epoch_guard_map": {base: ir.const(True, "bool")},
        },
    )
    pointer = base.data if base_dtype == dtype else ir.Call("tirx.reinterpret", [base.data], ty=alias.data.ty)
    before = kernel(
        seq(unit(alias.declare(pointer)), unit(loop, core=None)),
        buffers=[base, counter],
        annotations={"tl.buffer_versions_map": {base: 2}},
    )
    after = _materialize(before)
    (base_store,) = _stores(after, "base")
    (alias_store,) = _stores(after, "alias")
    assert tuple(int(dim) for dim in alias_store.dest.shape) == (2, *shape)
    assert int(tensor_strides(base_store.dest)[0]) == 16
    assert int(tensor_strides(alias_store.dest)[0]) == pitch
    assert tuple(int(dim) for dim in allocated_buffer(after, "base").shape) == (2, 16)
    declaration = next(binding for binding in nodes(after, ir.Bind) if binding.var.name.startswith("alias"))
    assert declaration.value.op.name == "tirx.decl_tensor"
    assert any(variable.same_as(base_store.dest) for variable in nodes(declaration.value, ir.Var))
    ir.assert_structural_equal(alias_store.indices[0], base_store.indices[0])


@pytest.mark.parametrize("partial", [False, True], ids=["whole-tile", "partial-tile"])
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_broadcast_fill_initializes_every_version(partial, versions):
    shared = _tensor("ub", (64,), "float32", "shared.dyn")
    index = ir.Var("index", "int32")
    fill = ir.Evaluate(ir.Call("tl.tileop.fill", [region(shared, 2, shape=[32 if partial else 64]), ir.const(0, "float32")], ty="handle"))
    fill = _marker("tl.ascend_task", {"core_mask": 5, "tl.multi_buffer_broadcast_fill": [shared]}, fill)
    owner = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        unit(ir.TensorStore(shared, [0], ir.const(1, "float32"))),
        annotations={"multi_buffer_eligible": [shared]},
    )
    before = kernel(
        seq(unit(fill, core=None), unit(owner, core=None)), buffers=[shared], annotations={"tl.buffer_versions_map": {shared: versions}}
    )
    after = _materialize(before)
    (fill,) = calls(after, "tl.tileop.fill")
    destination = fill.args[0]
    expected_shape = (32 if partial else 64,)
    if versions > 1:
        expected_shape = (versions, *expected_shape)
    assert tuple(int(dimension.extent) for dimension in destination.region) == expected_shape
    assert all(int(dimension.begin) == 0 for dimension in destination.region)
    tasks = [scope for scope in nodes(after, ir.RegionStmt) if "tl.ascend_task" in scope.attrs]
    assert tasks and all("tl.multi_buffer_broadcast_fill" not in task.attrs["tl.ascend_task"] for task in tasks)


@pytest.mark.parametrize("kind", ["condition", "assume"])
def test_storage_reads_in_guards_follow_the_owner_version(kind):
    shared = _tensor("ub", (1,), "int32", "shared.dyn")
    output = _tensor("out", (4,), "int32")
    index = ir.Var("index", "int32")
    read = _marker("tl.ascend_task", {"core_mask": 5}, ir.TensorStore(output, [index], shared[0]))
    guard_key = "tl.schedule_condition" if kind == "condition" else "tilelang_assume"
    guarded_read = _marker(guard_key, shared[0] >= 0, read)
    body = seq(unit(ir.TensorStore(shared, [0], index)), unit(guarded_read, core=None))
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, body, annotations={"multi_buffer_eligible": [shared]})
    before = kernel(unit(loop, core=None), buffers=[shared], params=[output], annotations={"tl.buffer_versions_map": {shared: 2}})
    after = _materialize(before)
    expressions = [scope.attrs[guard_key] for scope in nodes(after, ir.RegionStmt) if guard_key in scope.attrs]
    assert len(expressions) == 1
    (load,) = nodes(expressions[0], ir.TensorLoad)
    assert len(load.indices) == 2
    assert sym.Analyzer().can_prove_equal(load.indices[0], index % 2)


@pytest.mark.parametrize("start, step", [(0, 1), (3, 2), (7, -2)])
def test_iteration_clock_counts_logical_trips_across_nested_loops(start, step):
    shared = _tensor("ub", (1,), "int32", "shared.dyn")
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    trip_count = 4 // abs(step)
    owner = ir.For(
        inner,
        start,
        start + trip_count * step,
        step,
        ir.ForKind.DEFAULT,
        unit(ir.TensorStore(shared, [0], inner)),
        annotations={"multi_buffer_eligible": [shared]},
    )
    before = kernel(
        unit(ir.For(outer, 0, 2, 1, ir.ForKind.DEFAULT, unit(owner, core=None)), core=None),
        buffers=[shared],
        annotations={"tl.buffer_versions_map": {shared: 3}},
    )
    after = _materialize(before)
    (store,) = nodes(after, ir.TensorStore)
    expected = (outer * trip_count + (inner - start) // step) % 3
    assert sym.Analyzer().can_prove_equal(store.indices[0], expected)


@pytest.mark.parametrize("versions", [(1, 1), (1, 2)], ids=["unit-only-group", "mixed-group"])
def test_counter_group_survives_only_for_physical_versions(versions):
    first = _tensor("first", (1,), "int32", "shared.dyn")
    second = _tensor("second", (1,), "int32", "shared.dyn")
    epoch = _tensor("epoch", (1,), "int32", "local.var")
    index = ir.Var("index", "int32")
    body = seq(
        unit(ir.TensorStore(first, [0], index)), unit(ir.TensorStore(second, [0], index)), unit(ir.TensorStore(epoch, [0], epoch[0] + 1))
    )
    owner = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        body,
        annotations={
            "multi_buffer_eligible": [first, second],
            "tl.multi_buffer_counter_map": {first: epoch, second: epoch},
            "tl.storage_epoch_guard_map": {first: ir.const(True, "bool"), second: ir.const(True, "bool")},
        },
    )
    before = kernel(
        seq(unit(ir.TensorStore(epoch, [0], 0)), unit(owner, core=None)),
        buffers=[first, second, epoch],
        annotations={"tl.buffer_versions_map": {first: versions[0], second: versions[1]}},
    )
    after = _materialize(before)
    (first_store,) = _stores(after, "first")
    (second_store,) = _stores(after, "second")
    assert len(first_store.indices) == 1
    updates = _stores(after, "epoch")
    if versions[1] > 1:
        assert allocated_buffer(after, "epoch") is not None and len(updates) == 2
        assert len(second_store.indices) == 2
        assert sym.Analyzer().can_prove_equal(second_store.indices[0], epoch[0] % versions[1])
    else:
        assert not any(variable.name == "epoch" for variable in nodes(after, ir.Var)) and not updates
        assert len(second_store.indices) == 1


@pytest.mark.parametrize("use", ["read", "guard", "compound-task", "loop-bound"])
def test_single_version_counter_elision_rejects_live_uses(use):
    before = copy_ring(versions=1, prepared=True)
    root = _root(before)
    epoch = allocated_buffer(before, "epoch")
    shared = allocated_buffer(before, "ub")
    write = ir.TensorStore(shared, [0], ir.const(1, "float32"))
    if use == "read":
        unexpected = unit(ir.TensorStore(shared, [0], ir.prim.Cast("float32", epoch[0])))
    elif use == "guard":
        unexpected = unit(write, guard=epoch[0] > 0)
    elif use == "compound-task":
        unexpected = unit(seq(ir.TensorStore(epoch, [0], epoch[0] + 1), write))
    else:
        unexpected = unit(ir.For(ir.Var("index", "int32"), 0, epoch[0], 1, ir.ForKind.DEFAULT, unit(write)), core=None)
    extended = ir.RegionStmt(root.op, root.args, root.body_params, root.attrs, [root.body, unexpected], root.result_vars)
    before = ir.IRModule({"main": before["main"].with_body(extended)})
    with pytest.raises(Exception, match="Cannot elide single-version counter epoch"):
        transform.MaterializeMultiBuffer()(before)


def test_bound_data_and_sf_preserve_user_leading_dimensions_under_one_ring():
    data = _tensor("data", (2, 3, 32, 128), "float8_e4m3fn", "shared.l0a")
    scale = _tensor("sf", (2, 3, 32, 2), "uint16", "shared.l0a.sf")
    index = ir.Var("index", "int32")
    body = seq(*[unit(ir.TensorStore(tensor, [1, 2, 0, 0], ir.const(1, tensor.dtype)), core=2) for tensor in (data, scale)])
    owner = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, body, annotations={"multi_buffer_eligible": [data, scale]})
    before = kernel(
        unit(owner, core=None),
        buffers=[data, scale],
        annotations={
            "tl.l0_sf_bindings": {scale: data},
            "tl.buffer_versions_map": {data: 2, scale: 2},
        },
    )
    after = _materialize(before)
    for name, tail in [("data", (32, 128)), ("sf", (32, 2))]:
        assert tuple(int(extent) for extent in allocated_buffer(after, name).shape) == (2, 2, 3, *tail)
    stores = nodes(after, ir.TensorStore)
    assert len(stores) == 2
    for store in stores:
        assert sym.Analyzer().can_prove_equal(store.indices[0], index % 2)
        assert [int(value) for value in store.indices[1:]] == [1, 2, 0, 0]
