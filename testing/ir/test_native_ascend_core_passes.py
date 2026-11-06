"""Native core passes preserve definitions, clocks and synchronization domains."""

from testing.ascend._ir import kernel_body, scheduled_fields

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import Iter

from tilelang.testing.ir import collect_calls

from test_native_ascend_core_resolution import _kernel, _loop, _unit
from test_native_ascend_schedule_units import _attributes, _bind_target, _fields
from test_native_reducer_contracts import allocate, annotation, attribute, decoded, integer, objects, region, tensor, tileop


def _apply(transform, value):
    assert tirx.analysis.check_well_formed(value)
    assert not tirx.analysis.undefined_vars(value)
    before = ir.save_json(value)
    result = transform(ir.IRModule({"main": value}))["main"]
    assert ir.save_json(value) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def _run(name, value, *arguments):
    return _apply(tvm_ffi.get_global_func("tl.transform." + name)(*arguments), value)


def _thread_body(shared, output, *, write=True, read=True, shift=1):
    thread = ir.Var("thread", "int32")
    body = []
    if write:
        body.append(ir.TensorStore(shared, [thread], ir.const(1.0)))
    if read:
        body.append(ir.TensorStore(output, [thread], ir.TensorLoad(shared, [(thread + shift) % 32])))
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(32)], [thread], {}, body)


def _vf(body):
    return attribute(None, "tl.sblock", {"name": "SIMT_VF", "no_realize": True}, body)


def _sync_function(body_builder, scope="shared", ascend=True):
    shared = tensor("shared", [32], "float32", scope)
    output = tensor("output", [32], "float32", "global")
    result = tirx.Function([output], [allocate(shared), *body_builder(shared, output)]).with_attr("global_symbol", "main")
    return _bind_target(result) if ascend else result


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_places_barrier_inside_vf(scope):
    value = _sync_function(lambda shared, output: [_vf([_thread_body(shared, output)])], scope)
    result = _run("AscendThreadSync", value, scope)
    barriers = collect_calls(result, op="tirx.gpu_storage_sync")
    assert len(barriers) == 1
    (vf,) = [node for node in _attributes(result, "tl.sblock") if _fields(node)["value"]["name"] == "SIMT_VF"]
    assert len(collect_calls(vf, op="tirx.gpu_storage_sync")) == 1
    assert barriers[0].args[0].value == scope
    assert not collect_calls(result, op="tl.storage_sync")


def test_ascend_thread_sync_keeps_vf_domains_separate():
    value = _sync_function(
        lambda shared, output: [
            _vf([_thread_body(shared, output, read=False)]),
            _vf([_thread_body(shared, output, write=False)]),
        ]
    )
    result = _run("AscendThreadSync", value, "shared")
    assert not collect_calls(result, op="tirx.gpu_storage_sync")


@pytest.mark.parametrize("ascend,expected", [(True, 0), (False, 1)])
def test_ascend_thread_sync_selects_target_execution_domain(ascend, expected):
    value = _sync_function(lambda shared, output: [_thread_body(shared, output)], ascend=ascend)
    result = _run("AscendThreadSync", value, "shared")
    assert len(collect_calls(result, op="tirx.gpu_storage_sync")) == expected


def test_ascend_thread_sync_skips_thread_private_accesses():
    value = _sync_function(lambda shared, output: [_vf([_thread_body(shared, output, shift=0)])])
    result = _run("AscendThreadSync", value, "shared")
    assert not collect_calls(result, op="tirx.gpu_storage_sync")


@pytest.mark.parametrize("disabled", [True, False])
def test_ascend_thread_sync_obeys_disable_option(disabled):
    value = _sync_function(lambda shared, output: [_vf([_thread_body(shared, output)])])
    with tvm_tirx.transform.PassContext(config={"tl.disable_thread_storage_sync": disabled}):
        result = _run("AscendThreadSync", value, "shared")
    assert len(collect_calls(result, op="tirx.gpu_storage_sync")) == (0 if disabled else 1)


def test_ascend_thread_sync_global_is_unchanged():
    value = _sync_function(lambda shared, output: [_vf([_thread_body(shared, output)])])
    result = _run("AscendThreadSync", value, "global")
    ir.assert_structural_equal(value, result)


def _prepared(mode, *, versions=2, guard=False, begin=0, end=4, step=1):
    storage = tensor("storage", [32], "int32", "shared")
    output = tensor("output", [32], "int32", "global")
    i = ir.Var("i", "int32")
    flag = ir.Var("flag", "bool")
    write = _unit(ir.TensorStore(storage, [integer(0)], i), 1)
    read = _unit(ir.TensorStore(output, [integer(0)], ir.TensorLoad(storage, [integer(0)])), 1)
    body = [ir.If(flag, [write, read], None)] if guard else [write, read]
    loop = _loop(i, body, {"multi_buffer_eligible": [storage]}, begin=begin, end=end, step=step)
    value = _kernel(
        [output, flag],
        [_unit(allocate(storage), 1), annotation({"tl.buffer_version_mode": {storage: mode}}), loop],
        {storage: versions},
    )
    return _run("PrepareMultiBuffer", value), storage


@pytest.mark.parametrize("mode", ["auto", "iteration", "counter"])
def test_prepare_multi_buffer_uses_real_native_counters(mode):
    result, _ = _prepared(mode)
    counters = [bind for bind in objects(result, ir.Bind) if "version_counter" in bind.var.name]
    if mode != "counter":
        assert not counters
        return
    assert len(counters) == 1
    counter = counters[0].var
    assert isinstance(counter.ty, tirx.TensorType)
    assert counter.ty.scope.name == "local.var"
    allocation = counters[0].value.args[0]
    assert allocation.op.name == "tirx.alloc_tensor"
    stores = [store for store in objects(result, ir.TensorStore) if store.dest.same_as(counter)]
    assert len(stores) == 2
    assert any(isinstance(store.value, ir.TensorLoad) or objects(store.value, ir.TensorLoad) for store in stores)


@pytest.mark.parametrize("kind", ["conditional", "negative", "zero"])
def test_prepare_auto_uses_counter_for_nonordinal_domains(kind):
    result, _ = _prepared("auto", guard=kind == "conditional", step={"conditional": 1, "negative": -1, "zero": 0}[kind])
    assert any("version_counter" in bind.var.name for bind in objects(result, ir.Bind))


def _costed(body, core=1, *, stage=-1):
    return _unit(body, core, stage=stage, metadata={"latency": integer(3), "ii": integer(1)})


def test_auto_schedule_preserves_scalar_definition_order():
    value = ir.Var("value", "int32")
    output = tensor("output", [1], "int32", "global")
    function = _kernel([output], [_costed(ir.Bind(value, integer(7))), _costed(ir.TensorStore(output, [integer(0)], value))])
    result = _run("AutoSchedule", _bind_target(function))
    assert [bind.var.name for bind in objects(result, ir.Bind)] == ["value"]
    assert len(objects(result, ir.TensorStore)) == 1
    for unit in _attributes(result, "tl.schedule_unit"):
        assert int(_fields(unit)["node"]["stage"]) == 0


@pytest.mark.parametrize("scope", ["if", "assume"])
def test_auto_schedule_preserves_lexical_header(scope):
    output = tensor("output", [1], "int32", "global")
    flag = ir.Var("flag", "bool")
    store = _costed(ir.TensorStore(output, [integer(0)], integer(9)))
    body = ir.If(flag, [store], None) if scope == "if" else attribute(flag, "tilelang_assume", integer(1), [store])
    result = _run("AutoSchedule", _bind_target(_kernel([output, flag], [body])))
    if scope == "if":
        assert len(objects(result, ir.If)) == 1
    else:
        assert len(_attributes(result, "tilelang_assume")) == 1
    assert len(objects(result, ir.TensorStore)) == 1


def _metadata(value):
    result = []
    for call in objects(value, ir.Call):
        if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation":
            result.append(tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"]))
    for marker in _attributes(value, "tl.loop.annotations"):
        result.append(_fields(marker)["value"])
    return result


@pytest.mark.parametrize("mode", ["iteration", "counter"])
def test_materialize_multi_buffer_rewrites_real_storage_and_accesses(mode):
    prepared, old_storage = _prepared(mode)
    result = _run("MaterializeMultiBuffer", prepared)
    (binding,) = [bind for bind in objects(result, ir.Bind) if bind.var.name == old_storage.name]
    storage = binding.var
    assert not storage.same_as(old_storage)
    assert [int(extent) for extent in storage.ty.shape] == [2, 32]
    assert binding.value.args[0].ty.same_as(storage.ty)
    stores = [store for store in objects(result, ir.TensorStore) if store.dest.same_as(storage)]
    loads = [load for load in objects(result, ir.TensorLoad) if load.source.same_as(storage)]
    assert len(stores) == len(loads) == 1
    assert len(stores[0].indices) == len(loads[0].indices) == 2
    ir.assert_structural_equal(stores[0].indices[0], loads[0].indices[0])
    if mode == "counter":
        assert objects(stores[0].indices[0], ir.TensorLoad)
    else:
        assert not objects(stores[0].indices[0], ir.TensorLoad)
    for fields in _metadata(result):
        assert "tl.multi_buffer_counter_map" not in fields
        assert "tl.storage_epoch_guard_map" not in fields
        assert "tl.buffer_versions_map" not in fields


def test_materialize_unit_version_counter_elides_generated_state():
    prepared, storage = _prepared("counter", versions=1)
    assert any("version_counter" in bind.var.name for bind in objects(prepared, ir.Bind))
    result = _run("MaterializeMultiBuffer", prepared)
    assert not any("version_counter" in var.name for var in objects(result, ir.Var))
    assert any(bind.var.same_as(storage) for bind in objects(result, ir.Bind))
    assert len(objects(result, ir.TensorStore)) == 2


def _physical_offset(tensor_var, coordinates):
    values = [integer(value, "int64") for value in coordinates]
    mapped = tensor_var.ty.layout.apply(*values, shape=tensor_var.ty.shape)
    return int(tvm_tirx.sym.Analyzer().simplify(mapped["m"]))


def test_materialize_multibuffer_preserves_alias_bytes_and_native_strides():
    from test_native_ascend_ops import strided_tensor

    storage = strided_tensor("storage", [4], [3], 2, "int32", "shared")
    alias = strided_tensor("alias", [2], [2], 0, "int16", "shared")
    pointer = ir.Call("tirx.ptr_byte_offset", [storage.data, integer(4, "int64")], ty=storage.data.ty)
    output = tensor("output", [1], "int16", "global")
    i = ir.Var("i", "int32")
    body = [
        _unit(ir.TensorStore(alias, [integer(1)], ir.const(7, "int16")), 1),
        _unit(ir.TensorStore(output, [integer(0)], ir.TensorLoad(alias, [integer(1)])), 1),
    ]
    value = _kernel(
        [output],
        [
            _unit(allocate(storage), 1),
            _unit(ir.Bind(alias, ir.Call("tirx.decl_tensor", [pointer], ty_args=[alias.ty])), 1),
            _loop(i, body, {"multi_buffer_eligible": [storage]}),
        ],
        {storage: 2},
    )
    result = _run("MaterializeMultiBuffer", _run("PrepareMultiBuffer", value))
    bindings = {bind.var.name: bind for bind in objects(result, ir.Bind)}
    new_storage, new_alias = bindings["storage"].var, bindings["alias"].var
    assert _physical_offset(new_storage, [1, 0]) == 12 + 2
    assert _physical_offset(new_alias, [1, 1]) == 24 + 2
    alias_call = bindings["alias"].value.args[0]
    assert alias_call.op.name == "tirx.decl_tensor"
    assert alias_call.args[0].op.name == "tirx.ptr_byte_offset"
    assert int(alias_call.args[0].args[1]) == 4
    assert alias_call.args[0].args[0].args[0].same_as(new_storage)


def test_materialize_multibuffer_rounds_fp4_storage_to_physical_bytes():
    storage = tensor("storage", [3], "float4_e2m1fn", "shared")
    output = tensor("output", [1], "float4_e2m1fn", "global")
    i = ir.Var("i", "int32")
    body = [
        _unit(ir.TensorStore(storage, [integer(1)], ir.const(1, "float4_e2m1fn")), 1),
        _unit(ir.TensorStore(output, [integer(0)], ir.TensorLoad(storage, [integer(1)])), 1),
    ]
    value = _kernel([output], [_unit(allocate(storage), 1), _loop(i, body, {"multi_buffer_eligible": [storage]})], {storage: 2})
    result = _run("MaterializeMultiBuffer", _run("PrepareMultiBuffer", value))
    (new_storage,) = [bind.var for bind in objects(result, ir.Bind) if bind.var.name == storage.name]
    assert _physical_offset(new_storage, [0, 1]) == 1
    assert _physical_offset(new_storage, [1, 1]) == 5


def test_materialize_multibuffer_preserves_composed_swizzle_per_version():
    tile = tirx.TileLayout.from_iters([Iter(integer(16), integer(1), "m")])
    layout = tirx.ComposeLayout(0, 1, 2, tile)
    storage = ir.Var("storage", tirx.TensorType([16], "int32", scope="shared", layout=layout))
    output = tensor("output", [1], "int32", "global")
    i = ir.Var("i", "int32")
    body = [
        _unit(ir.TensorStore(storage, [integer(3)], integer(7)), 1),
        _unit(ir.TensorStore(output, [integer(0)], ir.TensorLoad(storage, [integer(3)])), 1),
    ]
    value = _kernel([output], [_unit(allocate(storage), 1), _loop(i, body, {"multi_buffer_eligible": [storage]})], {storage: 2})
    result = _run("MaterializeMultiBuffer", _run("PrepareMultiBuffer", value))
    (new_storage,) = [bind.var for bind in objects(result, ir.Bind) if bind.var.name == storage.name]
    assert isinstance(new_storage.ty.layout, tirx.ComposeLayout)
    for index in range(16):
        for version in range(2):
            assert _physical_offset(new_storage, [version, index]) == _physical_offset(storage, [index]) + 16 * version


def test_materialize_multibuffer_rewrites_regions_and_broadcast_fill():
    storage = tensor("storage", [4], "int32", "shared")
    output = tensor("output", [4], "int32", "global")
    i = ir.Var("i", "int32")
    fill = _unit(tileop("fill", region(storage), integer(0)), 1, metadata={"tl.multi_buffer_broadcast_fill": [storage]})
    body = [_unit(tileop("copy", region(storage), region(output)), 1)]
    value = _kernel([output], [_unit(allocate(storage), 1), fill, _loop(i, body, {"multi_buffer_eligible": [storage]})], {storage: 3})
    result = _run("MaterializeMultiBuffer", _run("PrepareMultiBuffer", value))
    (filled,) = collect_calls(result, op="tl.tileop.fill")
    (copied,) = collect_calls(result, op="tl.tileop.copy")
    filled_region = decoded(filled)[0][0]
    copied_region = decoded(copied)[0][0]
    assert len(filled_region.region) == len(copied_region.region) == 2
    assert int(filled_region.region[0].begin) == 0
    assert int(filled_region.region[0].extent) == 3
    assert int(copied_region.region[0].extent) == 1
    assert filled_region.source.same_as(copied_region.source)
    assert all("tl.multi_buffer_broadcast_fill" not in _fields(marker)["node"] for marker in _attributes(result, "tl.ascend_task"))


def test_materialize_multibuffer_preserves_masked_access_operands():
    storage = tensor("storage", [4], "int32", "shared")
    output = tensor("output", [1], "int32", "global")
    mask = ir.Var("mask", "bool")
    i = ir.Var("i", "int32")
    write = ir.Evaluate(ir.Call("tirx.masked_store", [storage, integer(7), integer(0), mask], ty="void"))
    read = ir.TensorStore(output, [integer(0)], ir.Call("tirx.masked_load", [storage, integer(0), mask], ty="int32"))
    value = _kernel(
        [output, mask],
        [_unit(allocate(storage), 1), _loop(i, [_unit(write, 1), _unit(read, 1)], {"multi_buffer_eligible": [storage]})],
        {storage: 2},
    )
    result = _run("MaterializeMultiBuffer", _run("PrepareMultiBuffer", value))
    (write,) = collect_calls(result, op="tirx.masked_store")
    (read,) = collect_calls(result, op="tirx.masked_load")
    assert len(write.args) == 5 and len(read.args) == 4
    assert write.args[0].same_as(read.args[0])
    assert write.args[-1].same_as(mask) and read.args[-1].same_as(mask)
    assert int(write.args[1]) == 7 and int(write.args[-2]) == int(read.args[-2]) == 0
    ir.assert_structural_equal(write.args[2], read.args[1])


def test_materialize_multibuffer_offsets_typed_pointer_by_physical_bytes():
    storage = tensor("storage", [4], "int32", "shared")
    i = ir.Var("i", "int32")
    access = ir.Evaluate(ir.Call("tl.ptr_access", [storage.data, integer(3), integer(1), integer(3)]))
    value = _kernel([], [_unit(allocate(storage), 1), _loop(i, [_unit(access, 1)], {"multi_buffer_eligible": [storage]})], {storage: 2})
    result = _run("MaterializeMultiBuffer", _run("PrepareMultiBuffer", value))
    (access,) = collect_calls(result, op="tl.ptr_access")
    pointer = access.args[0]
    assert pointer.op.name == "tirx.ptr_byte_offset"
    assert int(tvm_tirx.sym.Analyzer().simplify(pointer.args[1] - (ir.prim.Cast("int64", i) % 2) * 16)) == 0
    assert [int(arg) for arg in access.args[1:]] == [3, 1, 3]
    assert pointer.args[0].args[0].ty.shape[0] == 2
    ir.assert_structural_equal(access.ty, storage.data.ty)


def test_materialize_multibuffer_retains_dynamic_type_definition_dominance():
    n = ir.Var("n", "int32")
    size = ir.Var("size", "int32")
    storage = tensor("storage", [size], "int32", "shared")
    output = tensor("output", [1], "int32", "global")
    i = ir.Var("i", "int32")
    body = [
        _unit(ir.TensorStore(storage, [integer(0)], i), 1),
        _unit(ir.TensorStore(output, [integer(0)], ir.TensorLoad(storage, [integer(0)])), 1),
    ]
    value = _kernel(
        [n, output],
        [_unit(ir.Bind(size, n + 1), 1), _unit(allocate(storage), 1), _loop(i, body, {"multi_buffer_eligible": [storage]})],
        {storage: 2},
    )
    result = _run("MaterializeMultiBuffer", _run("PrepareMultiBuffer", value))
    bindings = objects(result, ir.Bind)
    assert bindings[0].var.same_as(size)
    assert bindings[1].var.ty.shape[1].same_as(size)
    assert int(bindings[1].var.ty.shape[0]) == 2


def test_materialize_multibuffer_rewrites_runtime_scope_header():
    storage = tensor("storage", [4], "int32", "shared")
    output = tensor("output", [1], "int32", "global")
    i = ir.Var("i", "int32")
    condition = ir.TensorLoad(storage, [integer(0)]) != 0
    body = [
        _unit(ir.TensorStore(storage, [integer(0)], i), 1),
        ir.If(condition, [_unit(ir.TensorStore(output, [integer(0)], integer(1)), 1)], None),
    ]
    value = _kernel([output], [_unit(allocate(storage), 1), _loop(i, body, {"multi_buffer_eligible": [storage]})], {storage: 2})
    result = _run("MaterializeMultiBuffer", _run("PrepareMultiBuffer", value))
    (branch,) = objects(result, ir.If)
    (loaded,) = objects(branch.condition, ir.TensorLoad)
    assert len(loaded.indices) == 2
    assert int(loaded.source.ty.shape[0]) == 2


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("write", [False, True])
def test_native_masked_memory_records_may_access_without_full_initialization(physical, write):
    from test_native_ascend_schedule_analysis import _memory, _points

    storage = tensor("storage", [4], "int32", "shared")
    mask = ir.Var("mask", "bool")
    name = "tirx.masked_store" if write else "tirx.masked_load"
    args = [storage, integer(7), integer(1), mask] if write else [storage, integer(1), mask]
    call = ir.Call(name, args, ty="void" if write else "int32")
    access = _memory([ir.Evaluate(call)], [allocate(storage)], physical=physical)
    assert len(access["writes" if write else "reads"]) == 1
    assert not access["reads" if write else "writes"]
    assert _points(access["writes" if write else "reads"][0]) == {(1,)}
    assert not access["must_writes"]
    assert not access["opaque"]


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("byte_offset", [0, 4])
@pytest.mark.parametrize("mask", [0, 1, 2, 3])
def test_native_raw_pointer_memory_retains_alias_root_direction_and_opacity(physical, byte_offset, mask):
    from test_native_ascend_ops import strided_tensor
    from test_native_ascend_schedule_analysis import _has, _memory

    storage = strided_tensor("storage", [4], [3], 2, "int32", "shared")
    alias = tensor("alias", [2], "int16", "shared")
    pointer = ir.Call("tirx.ptr_byte_offset", [storage.data, integer(byte_offset, "int64")])
    pointer = ir.Call("tirx.reinterpret", [pointer], ty=alias.data.ty)
    definitions = [allocate(storage), ir.Bind(alias, ir.Call("tirx.decl_tensor", [pointer], ty_args=[alias.ty]))]
    access = ir.Evaluate(ir.Call("tl.ptr_access", [alias.data, integer(1), integer(1), integer(mask)]))
    result = _memory([access], definitions, physical=physical)
    assert len(result["reads"]) == bool(mask & 1)
    assert len(result["writes"]) == bool(mask & 2)
    for access_region in [*result["reads"], *result["writes"]]:
        assert access_region.source.same_as(storage)
    assert _has(result["opaque"], storage) == bool(mask)
    assert not result["must_writes"]


def test_native_raw_pointer_memory_keeps_unknown_address_opaque():
    from test_native_ascend_schedule_analysis import _has, _memory

    address = ir.Var("address", "uint64")
    pointer = ir.Call("tirx.reinterpret", [address], ty=ir.PtrType("int32", "shared"))
    access = ir.Evaluate(ir.Call("tl.ptr_access", [pointer, integer(0), integer(1), integer(3)]))
    result = _memory([access])
    assert not result["reads"] and not result["writes"]
    assert _has(result["opaque"], address)


def _copy_task(source, destination, core=1):
    return _unit(tileop("copy", region(source), region(destination)), core)


def test_insert_sync_orders_real_copy_pipes():
    source = tensor("source", [16], "float32", "global")
    storage = tensor("storage", [16], "float32", "shared")
    output = tensor("output", [16], "float32", "global")
    value = _kernel([source, output], [_unit(allocate(storage), 1), _copy_task(source, storage), _copy_task(storage, output)])
    result = _run("InsertSync", value)
    (set_flag,) = collect_calls(result, op="tl.ascend_set_flag")
    (wait_flag,) = collect_calls(result, op="tl.ascend_wait_flag")
    assert set_flag.args[0].value == wait_flag.args[0].value == "MTE2_MTE3"
    ir.assert_structural_equal(set_flag.args[1], wait_flag.args[1])
    assert str(set_flag.ty) == str(ir.PrimType("void"))


@pytest.mark.parametrize("same_storage", [False, True])
@pytest.mark.parametrize("external_source", [False, True])
def test_insert_sync_same_pipe_completion_depends_on_overlap(same_storage, external_source):
    source = tensor("source", [16], "float32", "global")
    first = tensor("first", [16], "float32", "shared")
    second = first if same_storage else tensor("second", [16], "float32", "shared")
    definitions = ([] if external_source else [_unit(allocate(source), 1)]) + [_unit(allocate(first), 1)]
    definitions += [] if same_storage else [_unit(allocate(second), 1)]
    result = _run(
        "InsertSync", _kernel([source] if external_source else [], [*definitions, _copy_task(source, first), _copy_task(source, second)])
    )
    barriers = collect_calls(result, op="tl.ascend_pipe_barrier")
    # A trusted global source cannot overlap fresh on-chip storage. Repeated
    # reads of that source are safe; only overlapping destinations need a barrier.
    assert len(barriers) == int(same_storage)
    if barriers:
        assert barriers[0].args[0].value == "PIPE_MTE2"


def test_insert_sync_keeps_branch_local_flags_inside_the_branch():
    source = tensor("source", [16], "float32", "global")
    storage = tensor("storage", [16], "float32", "shared")
    output = tensor("output", [16], "float32", "global")
    enabled = ir.Var("enabled", "bool")
    branch = ir.If(enabled, [_copy_task(source, storage), _copy_task(storage, output)], None)
    result = _run("InsertSync", _kernel([source, output, enabled], [_unit(allocate(storage), 1), branch]))
    (branch,) = objects(result, ir.If)
    assert len(collect_calls(branch, op="tl.ascend_set_flag")) == len(collect_calls(result, op="tl.ascend_set_flag")) == 1
    assert len(collect_calls(branch, op="tl.ascend_wait_flag")) == 1


def test_insert_sync_waits_before_a_native_memory_condition_header():
    source = tensor("source", [16], "float32", "global")
    storage = tensor("storage", [16], "float32", "shared")
    output = tensor("output", [16], "float32", "global")
    branch = ir.If(ir.TensorLoad(storage, [integer(0)]) > ir.const(0.0), [_copy_task(storage, output)], None)
    result = _run("InsertSync", _kernel([source, output], [_unit(allocate(storage), 1), _copy_task(source, storage), branch]))
    root = kernel_body(result)
    before_branch = []
    for statement in root.seq:
        if isinstance(statement, ir.If):
            break
        before_branch.extend(collect_calls(statement, op="tl.ascend_wait_flag"))
    assert any(call.args[0].value == "MTE2_S" for call in before_branch)


@pytest.mark.parametrize("same_guard", [False, True])
@pytest.mark.parametrize("producer", [False, True])
def test_insert_sync_consumes_the_qualified_current_tree_hint_index(same_guard, producer):
    from test_native_ascend_scoped_conflicts import _marker, _record

    first = tensor("first", [1], "int32", "global")
    second = tensor("second", [1], "int32", "global")
    enabled = ir.Var("enabled", "bool")
    hint = _marker(first, second) if producer else _record([first, second])
    work = [_unit(ir.TensorStore(first, [integer(0)], integer(1)), 2), _unit(ir.Evaluate(ir.TensorLoad(second, [integer(0)])), 5)]
    if same_guard:
        work = [ir.If(enabled, work, None)]
    value = _kernel([first, second, enabled], [ir.If(enabled, [hint], None), *work])
    if producer:
        value = _run("NormalizeConflictHints", value)
    result = _run("InsertSync", value)
    assert bool(collect_calls(result, op="tl.ascend_cross_core_set_flag")) == (not same_guard)
    assert bool(collect_calls(result, op="tl.ascend_cross_core_wait_flag")) == (not same_guard)


@pytest.mark.parametrize("mode", ["iteration", "counter"])
@pytest.mark.parametrize("branch", [False, True])
def test_insert_sync_materializes_prepared_ring_protocols(mode, branch):
    source = tensor("source", [16], "float32", "global")
    storage = tensor("storage", [16], "float32", "shared")
    output = tensor("output", [16], "float32", "global")
    enabled, i = ir.Var("enabled", "bool"), ir.Var("i", "int32")
    body = [_copy_task(source, storage), _copy_task(storage, output)]
    if branch:
        body = [ir.If(enabled, body, None)]
    value = _kernel(
        [source, output, enabled],
        [
            _unit(allocate(storage), 1),
            annotation({"tl.buffer_version_mode": {storage: mode}}),
            _loop(i, body, {"multi_buffer_eligible": [storage]}),
        ],
        {storage: 2},
    )
    prepared = _run("ResolveCore", _run("PrepareMultiBuffer", value))
    synchronized = _run("InsertSync", prepared)
    assert collect_calls(synchronized, op="tl.ascend_set_flag")
    assert collect_calls(synchronized, op="tl.ascend_wait_flag")
    result = _run("MaterializeMultiBuffer", synchronized)
    assert all(len(access.region) == 2 for access in objects(result, ir.TensorRegion) if access.source.name == "storage")


def _core(value, name):
    (result,) = [marker for marker in _attributes(value, "tl.sblock") if _fields(marker)["value"]["name"] == name]
    return result


def _assert_schedule_consumed(value):
    for key in ("tl.schedule_unit", "tl.ascend_task", "tl.ascend_per_core_task"):
        assert not _attributes(value, key)
    assert not collect_calls(value, op="tl.ascend.scheduled_bind")
    consumed = {
        "num_stages",
        "enable_offset",
        "tl_pipeline_order",
        "tl_pipeline_stage",
        "multi_buffer_eligible",
        "tl.buffer_versions_map",
        "tl.buffer_version_mode",
        "conflict_hint",
        "tl.root_conflict_hints",
        "tl.scoped_conflict_hints",
        "tl.conflict_scope_id",
        "tl.split_subcores",
    }
    assert all(not (set(fields) & consumed) for fields in _metadata(value))
    assert all(not (set(loop.annotations) & consumed) for loop in objects(value, ir.For))


def test_lower_scheduled_tir_extracts_cores_with_fresh_shared_definitions():
    shared = ir.Var("scalar", "int32")
    cube, vector = tensor("cube", [1], "int32", "global"), tensor("vector", [1], "int32", "global")
    value = _kernel(
        [cube, vector],
        [
            _unit(ir.Bind(shared, integer(7)), 7),
            _unit(ir.TensorStore(cube, [integer(0)], shared), 2),
            _unit(ir.TensorStore(vector, [integer(0)], shared + 1), 5),
        ],
    )
    result = _run("LowerScheduledTIR", value)
    _assert_schedule_consumed(result)
    bindings = []
    for name, output in (("CUBE", cube), ("VECTOR", vector)):
        stream = _core(result, name)
        (binding,) = objects(stream, ir.Bind)
        (store,) = objects(stream, ir.TensorStore)
        assert store.dest.same_as(output)
        assert not binding.var.same_as(shared)
        assert any(var.same_as(binding.var) for var in objects(store.value, ir.Var))
        bindings.append(binding.var)
    assert not bindings[0].same_as(bindings[1])
    assert not [region for region in objects(_core(result, "CUBE"), ir.RegionStmt) if region.op.name == "tirx.launch_thread"]
    (launch,) = [region for region in objects(_core(result, "VECTOR"), ir.RegionStmt) if region.op.name == "tirx.launch_thread"]
    assert launch.args[0].value == "cthread" and int(launch.args[1]) == 2


@pytest.mark.parametrize("core,sid", [(1, 0), (4, 1)])
def test_lower_scheduled_tir_guards_the_selected_aiv(core, sid):
    output = tensor("output", [1], "int32", "global")
    result = _run("LowerScheduledTIR", _kernel([output], [_unit(ir.TensorStore(output, [integer(0)], integer(3)), core)]))
    (launch,) = [region for region in objects(result, ir.RegionStmt) if region.op.name == "tirx.launch_thread"]
    (branch,) = objects(launch, ir.If)
    predicate = tirx.IndexMap([launch.body_params[0]], [branch.condition], None)
    assert [bool(predicate.map_indices([integer(value)])[0]) for value in (0, 1)] == [sid == 0, sid == 1]
    _assert_schedule_consumed(result)


def test_lower_scheduled_tir_moves_external_cthread_into_vector_only():
    sid = ir.Var("outer_sid", "int32")
    cube, vector = tensor("cube", [1], "int32", "global"), tensor("vector", [2], "int32", "global")
    kernel = _kernel(
        [cube, vector], [_unit(ir.TensorStore(cube, [integer(0)], integer(9)), 2), _unit(ir.TensorStore(vector, [sid], sid), 5)]
    )
    value = tirx.Function(
        [cube, vector], [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("cthread"), integer(2)], [sid], {}, kernel.body.seq)]
    )
    result = _run("LowerScheduledTIR", value)
    assert not [region for region in objects(_core(result, "CUBE"), ir.RegionStmt) if region.op.name == "tirx.launch_thread"]
    (launch,) = [region for region in objects(result, ir.RegionStmt) if region.op.name == "tirx.launch_thread"]
    assert not launch.body_params[0].same_as(sid)
    (store,) = objects(launch, ir.TensorStore)
    assert store.value.same_as(launch.body_params[0])


def test_lower_scheduled_tir_retains_branch_local_definition_scope():
    flag = ir.Var("flag", "bool")
    output = tensor("output", [2], "int32", "global")
    then_value, else_value = ir.Var("then_value", "int32"), ir.Var("else_value", "int32")
    branch = ir.If(
        flag,
        [_unit(ir.Bind(then_value, integer(3)), 5), _unit(ir.TensorStore(output, [integer(0)], then_value), 5)],
        [_unit(ir.Bind(else_value, integer(7)), 5), _unit(ir.TensorStore(output, [integer(1)], else_value), 5)],
    )
    result = _run("LowerScheduledTIR", _kernel([flag, output], [branch]))
    (branch,) = objects(result, ir.If)
    assert len(objects(branch.then_case, ir.Bind)) == len(objects(branch.else_case, ir.Bind)) == 1
    assert len(objects(branch.then_case, ir.TensorStore)) == len(objects(branch.else_case, ir.TensorStore)) == 1
    _assert_schedule_consumed(result)


def test_lower_scheduled_tir_filters_and_renews_local_storage_maps_per_core():
    left = tensor("cube_storage", [4], "int32", "shared.dyn")
    right = tensor("vector_storage", [4], "int32", "shared")
    fields = {
        "tl.buffer_alias_map": {left: [], right: []},
        "tl.aiv_storage_map": {right: 0},
        "tl.manual_multi_buffer": {left: 2, right: 3},
        "tl.split_subcores": integer(1),
    }
    value = _kernel(
        [],
        [
            _unit(allocate(left), 2),
            _unit(allocate(right), 5),
            _unit(ir.TensorStore(left, [integer(0)], integer(1)), 2),
            _unit(ir.TensorStore(right, [integer(0)], integer(2)), 5),
            annotation(fields),
        ],
    )
    result = _run("LowerScheduledTIR", value)
    for name, expected in (("CUBE", 2), ("VECTOR", 3)):
        stream = _core(result, name)
        (binding,) = objects(stream, ir.Bind)
        (fields,) = _metadata(stream)
        (key,) = fields["tl.buffer_alias_map"]
        assert key.same_as(binding.var)
        assert int(fields["tl.manual_multi_buffer"][key]) == expected
        assert not key.same_as(left) and not key.same_as(right)
        assert len(fields["tl.aiv_storage_map"]) == int(name == "VECTOR")
    _assert_schedule_consumed(result)


@pytest.mark.parametrize("stages", [(0, 2), (2, 0)])
@pytest.mark.parametrize("begin,end,step", [(3, 10, 2), (-5, 3, 3), (5, 5, 1)])
def test_lower_scheduled_tir_executes_pipeline_fill_and_drain(stages, begin, end, step):
    import numpy as np
    from test_native_reducer_contracts import execute, function

    i = ir.Var("i", "int32")
    left, right = tensor("left", [8], "int32", "global"), tensor("right", [8], "int32", "global")
    position = (i - begin) // step
    body = [
        _unit(ir.TensorStore(left, [position], i * 3), 1, stage=stages[0]),
        _unit(ir.TensorStore(right, [position], i * 7), 1, stage=stages[1]),
    ]
    loop = _loop(i, body, {"num_stages": integer(3)}, begin=begin, end=end, step=step)
    result = _run("LowerScheduledTIR", _kernel([left, right], [loop], count=1))
    _assert_schedule_consumed(result)
    (physical,) = objects(result, ir.For)
    assert str(physical.var.ty.dtype) == "int64"
    actual = execute(
        function([left, right], _core(result, "VECTOR").body),
        [np.full(8, -99, dtype="int32"), np.full(8, -99, dtype="int32")],
        f"lower-pipeline-{stages[0]}-{stages[1]}-{begin}-{end}-{step}",
    )
    points = list(range(begin, end, step))
    for observed, factor in zip(actual, (3, 7)):
        np.testing.assert_array_equal(observed, np.array([*[i * factor for i in points], *([-99] * (8 - len(points)))], dtype="int32"))


def test_lower_scheduled_tir_guards_scalar_reads_without_hiding_bindings():
    import numpy as np
    from test_native_reducer_contracts import execute, function

    i, captured = ir.Var("i", "int32"), ir.Var("captured", "int32")
    state, output = tensor("state", [1], "int32", "global"), tensor("output", [4], "int32", "global")
    # The scalar read follows a write in its stage. A stage-2 consumer creates
    # fill/drain, but does not consume captured, so no scheduler clone is needed.
    body = [
        _unit(ir.TensorStore(state, [integer(0)], i + 10), 1),
        _unit(ir.Bind(captured, ir.TensorLoad(state, [integer(0)])), 1),
        _unit(ir.TensorStore(output, [i], captured), 1),
        _unit(ir.Evaluate(i), 1, stage=2),
    ]
    result = _run("LowerScheduledTIR", _kernel([state, output], [_loop(i, body)], count=1))
    (binding,) = objects(result, ir.Bind)
    assert any("if_then_else" in call.op.name for call in objects(binding, ir.Call))
    actual = execute(
        function([state, output], _core(result, "VECTOR").body),
        [np.zeros(1, dtype="int32"), np.full(4, -99, dtype="int32")],
        "lower-pipeline-scalar-read",
    )
    np.testing.assert_array_equal(actual[-1], [10, 11, 12, 13])


@pytest.mark.parametrize("begin,end,step,dtype", [(0, 4, 0, "int32"), (0, 4, -1, "int32"), (120, 121, 127, "int8")])
def test_lower_scheduled_tir_rejects_unproved_pipeline_progress(begin, end, step, dtype):
    i = ir.Var("i", dtype)
    output = tensor("output", [2], dtype, "global")
    loop = _loop(
        i,
        [_unit(ir.TensorStore(output, [integer(0)], i), 1), _unit(ir.TensorStore(output, [integer(1)], i), 1, stage=1)],
        begin=integer(begin, dtype),
        end=integer(end, dtype),
        step=integer(step, dtype),
    )
    with pytest.raises(ValueError, match="positive, non-wrapping For"):
        _run("LowerScheduledTIR", _kernel([output], [loop], count=1))


@pytest.mark.parametrize("step", [0, -1])
def test_lower_scheduled_tir_keeps_single_stage_nonpositive_loop_semantics(step):
    import numpy as np
    from test_native_reducer_contracts import execute, function

    i, slot = ir.Var("i", "int32"), ir.Var("slot", "int32")
    trace, count = tensor("trace", [3], "int32", "global"), tensor("count", [1], "int32", "global")
    body = [
        _unit(ir.Bind(slot, ir.TensorLoad(count, [integer(0)])), 1),
        _unit(ir.TensorStore(trace, [slot], i), 1),
        _unit(ir.TensorStore(count, [integer(0)], slot + 1), 1),
        _unit(ir.If(slot == 2, [ir.Break()], None), 1),
    ]
    result = _run("LowerScheduledTIR", _kernel([trace, count], [_loop(i, body, begin=3, end=5, step=step)], count=1))
    (loop,) = objects(result, ir.For)
    assert int(loop.step) == step
    actual = execute(
        function([trace, count], _core(result, "VECTOR").body),
        [np.full(3, -99, dtype="int32"), np.zeros(1, dtype="int32")],
        f"lower-single-stage-{step}",
    )
    np.testing.assert_array_equal(actual[0], [3 + step * index for index in range(3)])
    np.testing.assert_array_equal(actual[1], [3])


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("carrier", [False, True])
def test_auto_schedule_uses_the_real_loop_solver_with_native_metadata(manual, carrier):
    import importlib

    importlib.import_module("tilelang.ascend.transform.z3_scheduler")
    i = ir.Var("i", "int32")
    left, right = tensor("left", [4], "int32", "shared"), tensor("right", [4], "int32", "shared")
    fields = {"enable_offset": True, "num_stages": integer(2)}
    body = [
        _costed(ir.TensorStore(left, [i], i), stage=0 if manual else -1),
        _costed(ir.TensorStore(right, [i], i + 1), stage=1 if manual else -1),
    ]
    if carrier:
        body = [attribute(None, "tl.loop.annotations", fields, body)]
    loop = ir.For(i, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, body, {} if carrier else fields)
    value = _bind_target(_kernel([], [_costed(allocate(left), stage=0), _costed(allocate(right), stage=0), _unit(loop)], count=1))
    result = _run("AutoSchedule", value)
    stages = [
        int(_fields(unit)["node"]["stage"])
        for unit in _attributes(result, "tl.schedule_unit")
        if objects(unit.body, ir.TensorStore) and not objects(unit.body, ir.For)
    ]
    assert len(stages) == 2 and min(stages) >= 0
    if manual:
        assert sorted(stages) == [0, 1]
    lowered = _run("LowerScheduledTIR", result)
    _assert_schedule_consumed(lowered)


@pytest.mark.parametrize("carrier", [False, True])
def test_auto_schedule_executes_chained_scalar_clones_in_each_consumer_stage(carrier):
    import importlib
    import numpy as np
    from test_native_reducer_contracts import execute, function

    importlib.import_module("tilelang.ascend.transform.z3_scheduler")
    i, first, second = ir.Var("i", "int32"), ir.Var("first", "int32"), ir.Var("second", "int32")
    output = tensor("output", [8], "int32", "global")
    fields = {"enable_offset": True, "num_stages": integer(3)}
    body = [
        _costed(ir.Bind(first, i * 3 + 2), stage=0),
        _costed(ir.Bind(second, first * 5 + 7), stage=0),
        _costed(ir.TensorStore(output, [i], second), stage=0),
        _costed(ir.TensorStore(output, [i + 4], second + first), stage=2),
    ]
    if carrier:
        body = [attribute(None, "tl.loop.annotations", fields, body)]
    loop = ir.For(i, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, body, {} if carrier else fields)
    scheduled = _run("AutoSchedule", _bind_target(_kernel([output], [_unit(loop)], count=1)))
    for name in ("first", "second"):
        bindings = [bind for bind in objects(scheduled, ir.Bind) if bind.var.name == name]
        assert len(bindings) == 2 and not bindings[0].var.same_as(bindings[1].var)
        assert sorted(int(scheduled_fields(bind)["stage"]) for bind in bindings) == [0, 2]
    result = _run("LowerScheduledTIR", scheduled)
    _assert_schedule_consumed(result)
    (observed,) = execute(
        function([output], _core(result, "VECTOR").body),
        [np.full(8, -99, dtype="int32")],
        f"auto-chained-scalar-clones-{carrier}",
    )
    first_values = np.arange(4, dtype="int32") * 3 + 2
    second_values = first_values * 5 + 7
    np.testing.assert_array_equal(observed, np.concatenate((second_values, second_values + first_values)))


@pytest.mark.parametrize("autoschedule", [False, True])
@pytest.mark.parametrize("mode", ["iteration", "counter"])
@pytest.mark.parametrize("guarded", [False, True])
def test_native_core_pipeline_executes_with_autoschedule_on_and_off(autoschedule, mode, guarded):
    import importlib
    import numpy as np
    from test_native_reducer_contracts import execute, function

    importlib.import_module("tilelang.ascend.transform.z3_scheduler")
    i = ir.Var("i", "int32")
    left, right = tensor("left", [1], "int32", "shared.dyn"), tensor("right", [1], "int32", "shared.dyn")
    output = tensor("output", [6], "int32", "global")

    def task(statement):
        return _costed(statement) if autoschedule else statement

    body = [
        task(ir.TensorStore(left, [integer(0)], i + 1)),
        task(ir.TensorStore(right, [integer(0)], i + 3)),
        task(ir.TensorStore(output, [(i - 2) // 2], ir.TensorLoad(left, [integer(0)]) * ir.TensorLoad(right, [integer(0)]))),
    ]
    if guarded:
        body = [ir.If((i // 2) % 2 == 0, body, None)]
    if autoschedule:
        fields = {"multi_buffer_eligible": [left, right], "num_stages": integer(1), "enable_offset": False}
        body = [attribute(None, "tl.loop.annotations", fields, body)]
    loop = ir.For(i, integer(2), integer(14), integer(2), ir.ForKind.DEFAULT, body)
    body = [task(allocate(left)), task(allocate(right))]
    if autoschedule:
        body += [annotation({"tl.buffer_version_mode": {left: mode, right: mode}}), _unit(loop, stage=-1)]
    else:
        body += [loop]
    value = _bind_target(_kernel([output], body, {left: 2, right: 2}, count=1) if autoschedule else function([output], body))
    with tvm_tirx.transform.PassContext(config={"tl.enable_auto_schedule": autoschedule}) as context:
        assert bool(context.config["tl.enable_auto_schedule"]) == autoschedule
        if context.config["tl.enable_auto_schedule"]:
            for name in (
                "NormalizeConflictHints",
                "AutoSchedule",
                "PrepareMultiBuffer",
                "ResolveCore",
                "InsertSync",
                "MaterializeMultiBuffer",
                "LowerScheduledTIR",
            ):
                value = _run(name, value)
                if name == "PrepareMultiBuffer":
                    counters = [bind for bind in objects(value, ir.Bind) if "version_counter" in bind.var.name]
                    assert len(counters) == int(mode == "counter")
            _assert_schedule_consumed(value)
            value = _bind_target(function(list(value.params), _core(value, "VECTOR").body))
            value = _run("NormalizeBufferVersion", value)
            value = _run("RewriteAscendBufferVersionLayout", value)
        else:
            root = kernel_body(value)
            value = _bind_target(function(list(value.params), root))
        value = _run("AscendThreadSync", value, "shared")
        value = _apply(tirx.transform.LowerTensorLayouts(), value)
        if not autoschedule:
            value = _run("InferBufferAliases", value)
        value = _run("MergeUBAllocations", value, 32, False)
    (observed,) = execute(
        function(list(value.params), value.body),
        [np.full(6, -99, dtype="int32")],
        f"native-core-pipeline-{autoschedule}-{mode}-{guarded}",
    )
    expected = [(index + 1) * (index + 3) if not guarded or (index // 2) % 2 == 0 else -99 for index in range(2, 14, 2)]
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("carrier", [False, True])
def test_lower_scheduled_tir_renews_nested_loop_metadata_after_stage_expansion(carrier):
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    output = tensor("output", [4], "int32", "global")
    fields = {"outer_coordinate": i}
    inner_body = [_unit(ir.TensorStore(output, [i], i + j), 1)]
    if carrier:
        inner_body = [attribute(None, "tl.loop.annotations", fields, inner_body)]
    inner = ir.For(j, integer(0), integer(1), integer(1), ir.ForKind.DEFAULT, inner_body, {} if carrier else fields)
    outer = _loop(i, [_unit(inner, stage=1), _unit(ir.Evaluate(i), 1, stage=0)])
    result = _run("LowerScheduledTIR", _kernel([output], [outer], count=1))
    _assert_schedule_consumed(result)
    (physical,) = [loop for loop in objects(result, ir.For) if "pipeline" in loop.var.name]
    retained = _metadata(result) + [loop.annotations for loop in objects(result, ir.For)]
    (coordinate,) = [fields["outer_coordinate"] for fields in retained if "outer_coordinate" in fields]
    assert any(var.same_as(physical.var) for var in objects(coordinate, ir.Var))
    assert not any(var.same_as(i) for var in objects(result, ir.Var))


@pytest.mark.parametrize("mode", ["iteration", "counter"])
@pytest.mark.parametrize("carrier", [False, True])
def test_insert_sync_keeps_pipeline_event_stages_through_loop_metadata(mode, carrier):
    source, output = tensor("source", [16], "float32", "global"), tensor("output", [16], "float32", "global")
    storage = tensor("storage", [16], "float32", "shared")
    i = ir.Var("i", "int32")
    body = [
        _unit(tileop("copy", region(source), region(storage)), 1, stage=0),
        _unit(tileop("copy", region(storage), region(output)), 1, stage=1),
    ]
    fields = {"multi_buffer_eligible": [storage], "enable_offset": True, "num_stages": integer(2)}
    if carrier:
        body = [attribute(None, "tl.loop.annotations", fields, body)]
    loop = ir.For(i, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, body, {} if carrier else fields)
    value = _kernel(
        [source, output],
        [_unit(allocate(storage), 1), annotation({"tl.buffer_version_mode": {storage: mode}}), _unit(loop)],
        {storage: 2},
        count=1,
    )
    if mode == "counter":
        with pytest.raises(RuntimeError, match="multiple schedule stages"):
            _run("PrepareMultiBuffer", value)
        return
    prepared = _run("ResolveCore", _run("PrepareMultiBuffer", value))
    synchronized = _run("InsertSync", prepared)
    (loop,) = objects(synchronized, ir.For)
    events = []
    for unit in _attributes(loop.body, "tl.schedule_unit"):
        stage = int(_fields(unit)["node"]["stage"])
        for name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            for call in collect_calls(unit, op=name):
                events.append((name, call.args[0].value, stage))
    assert events
    for name, event, stage in events:
        if event == "MTE2_MTE3":
            assert stage == (0 if name.endswith("set_flag") else 1)
        elif event == "MTE3_MTE2":
            assert stage == (1 if name.endswith("set_flag") else 0)
    lowered = _run("LowerScheduledTIR", _run("MaterializeMultiBuffer", synchronized))
    _assert_schedule_consumed(lowered)


@pytest.mark.parametrize("scope", ["metadata", "branch", "memory_header", "nested_loop"])
def test_prepare_counter_advance_retains_the_owner_relative_access_stage(scope):
    storage = tensor("storage", [1], "int32", "shared")
    output = tensor("output", [4], "int32", "global")
    i, j, flag = ir.Var("i", "int32"), ir.Var("j", "int32"), ir.Var("flag", "bool")
    write = _unit(ir.TensorStore(storage, [integer(0)], i + 1), 1, stage=2)
    read = _unit(ir.TensorStore(output, [i], ir.TensorLoad(storage, [integer(0)])), 1, stage=2)
    if scope == "metadata":
        body = [attribute(None, "tl.loop.annotations", {"num_stages": integer(3)}, [write, read])]
    elif scope == "branch":
        body = [ir.If(flag, [write, read], None)]
    elif scope == "memory_header":
        body = [write, ir.If(ir.TensorLoad(storage, [integer(0)]) > 0, [read], None)]
    else:
        inner = ir.For(
            j,
            integer(0),
            integer(2),
            integer(1),
            ir.ForKind.DEFAULT,
            [
                _unit(ir.TensorStore(storage, [integer(0)], j), 1, stage=0),
                _unit(ir.TensorStore(output, [i], ir.TensorLoad(storage, [integer(0)])), 1, stage=1),
            ],
        )
        body = [_unit(inner, stage=2)]
    value = _kernel(
        [output, flag],
        [
            _unit(allocate(storage), 1),
            annotation({"tl.buffer_version_mode": {storage: "counter"}}),
            _loop(i, body, {"multi_buffer_eligible": [storage]}),
        ],
        {storage: 2},
        count=1,
    )
    prepared = _run("PrepareMultiBuffer", value)
    (owner,) = [loop for loop in objects(prepared, ir.For) if loop.var.same_as(i)]
    advances = []
    for unit in _attributes(owner.body, "tl.schedule_unit"):
        if objects(unit.body, ir.For):
            continue
        for store in objects(unit.body, ir.TensorStore):
            if "version_counter" in store.dest.name and objects(store.value, ir.TensorLoad):
                advances.append(int(_fields(unit)["node"]["stage"]))
    assert advances == [2]


@pytest.mark.parametrize("explicit_id", [0, -1, 17])
def test_insert_sync_reserves_explicit_cross_core_flag_banks(explicit_id):
    source, other = tensor("source", [1], "int32", "global"), tensor("other", [1], "int32", "global")
    explicit = ir.Evaluate(
        ir.Call("tl.ascend_cross_core_set_flag", [integer(2), ir.StringImm("PIPE_S"), integer(explicit_id)], ty=ir.PrimType("void"))
    )
    value = _kernel(
        [source, other],
        [
            _unit(explicit, 2),
            _unit(ir.TensorStore(source, [integer(0)], integer(1)), 2),
            _unit(ir.Evaluate(ir.TensorLoad(other, [integer(0)])), 5),
        ],
    )
    result = _run("InsertSync", value)
    generated = [call for call in collect_calls(result, op="tl.ascend_cross_core_set_flag") if int(call.args[0]) == 4]
    assert generated
    analyzer = tvm_tirx.sym.Analyzer()
    assert all(int(analyzer.simplify(call.args[2])) % 16 != explicit_id % 16 for call in generated)


def test_insert_sync_rejects_unbounded_explicit_cross_core_flag_ids():
    source, other = tensor("source", [1], "int32", "global"), tensor("other", [1], "int32", "global")
    flag = ir.Var("flag", "int32")
    explicit = ir.Evaluate(ir.Call("tl.ascend_cross_core_set_flag", [integer(2), ir.StringImm("PIPE_S"), flag], ty=ir.PrimType("void")))
    value = _kernel(
        [source, other, flag],
        [
            _unit(explicit, 2),
            _unit(ir.TensorStore(source, [integer(0)], integer(1)), 2),
            _unit(ir.Evaluate(ir.TensorLoad(other, [integer(0)])), 5),
        ],
    )
    with pytest.raises(RuntimeError, match="integer range cannot be bounded"):
        _run("InsertSync", value)


def test_insert_sync_keeps_then_and_else_completion_domains_separate():
    source, output = tensor("source", [16], "float32", "global"), tensor("output", [16], "float32", "global")
    storage = tensor("storage", [16], "float32", "shared")
    flag = ir.Var("flag", "bool")

    def body():
        return [_copy_task(source, storage), _copy_task(storage, output)]

    value = _kernel([source, output, flag], [_unit(allocate(storage), 1), ir.If(flag, body(), body())])
    result = _run("InsertSync", value)
    (branch,) = objects(result, ir.If)
    for arm in (branch.then_case, branch.else_case):
        assert collect_calls(arm, op="tl.ascend_set_flag")
        assert collect_calls(arm, op="tl.ascend_wait_flag")
    assert len(collect_calls(result, op="tl.ascend_set_flag")) == len(collect_calls(branch, op="tl.ascend_set_flag"))
