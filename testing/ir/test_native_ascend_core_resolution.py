"""Native core demand follows lexical control, storage epochs and scalar SSA."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.ascend import language as T

from test_native_ascend_core_placement import _assign, _copy, _kernel as _raw_kernel
from test_native_ascend_dual_copy import _materialize
from test_native_ascend_ops import function, strided_tensor, subregion
from test_native_ascend_schedule_units import _attributes, _fields, _scheduled_bind, _task
from test_native_cuda_copy import calls
from test_native_reducer_contracts import allocate, annotation, attribute, decoded, integer, objects, region, run_pass, tensor, tileop


def _unit(body, core=7, *, stage=0, sid=None, guard=None, metadata=None):
    fields = {"core_mask": integer(core, "int64")} if core else {}
    fields.update(metadata or {})
    if sid is not None:
        fields["subcore"] = integer(sid)
    if isinstance(body, ir.Bind):
        result = ir.Bind(body.var, _scheduled_bind(body.value, stage=stage, task=fields))
    else:
        task = body if isinstance(body, ir.For) else _task([body], fields)
        result = attribute({"stage": integer(stage, "int64")}, "tl.schedule_unit", integer(1), [task])
    return result if guard is None else ir.If(guard, [result], None)


def _loop(var, body, metadata=None, *, begin=0, end=4, step=1):
    if metadata:
        body = [attribute(None, "tl.loop.annotations", metadata, body)]
    values = [integer(value) if isinstance(value, int) else value for value in (begin, end, step)]
    return _unit(ir.For(var, *values, ir.ForKind.DEFAULT, body))


def _kernel(params, body, versions=None, count=2):
    if versions:
        # Metadata operands follow the definitions they reference.
        index = next((i for i, stmt in enumerate(body) if not isinstance(stmt, ir.Bind)), len(body))
        body = [*body[:index], annotation({"tl.buffer_versions_map": versions}), *body[index:]]
    root = attribute(
        None, "tl.sblock", {"name": "tilelang_root", "no_realize": True, "annotations": {"vector_count": integer(count)}}, body
    )
    return function(params, [root])


def _resolve(value):
    assert tirx.analysis.check_well_formed(value)
    assert not tirx.analysis.undefined_vars(value)
    before = ir.save_json(value)
    result = run_pass("ResolveCore", value)
    assert ir.save_json(value) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def _binding_mask(value, var):
    (bind,) = [bind for bind in objects(value, ir.Bind) if bind.var.same_as(var)]
    return int(decoded(bind.value)[1]["task"]["core_mask"])


def _tasks(value):
    return _attributes(value, "tl.ascend_task")


def _mask(task):
    return int(_fields(task)["node"]["core_mask"])


def _store_masks(value, storage):
    return [_mask(task) for task in _tasks(value) if any(store.dest.same_as(storage) for store in objects(task, ir.TensorStore))]


def _view(storage, name, shape, dtype="int32", *, offset=0, origin=0):
    view = strided_tensor(name, shape, [1] * len(shape), origin, dtype, storage.ty.scope)
    pointer = ir.Call("tirx.ptr_byte_offset", [storage.data, integer(offset, "int64")], ty=storage.data.ty)
    return view, _unit(ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])))


def _inspect(value, storages):
    return tvm_ffi.get_global_func("tl.testing.InspectAscendMultiBufferPlan")(_attributes(value, "tl.sblock")[0], storages)


@pytest.mark.parametrize("consumer", ["load", "store", "both"])
def test_tensor_definitions_and_pointer_views_follow_scalar_access_cores(consumer):
    state = tensor("state", [4], "int32", "local.var")
    alias, definition = _view(state, "alias", [1], offset=4)
    output = tensor("output", [1], "int32", "global")
    accesses = []
    if consumer in {"load", "both"}:
        accesses.append(_unit(ir.TensorStore(output, [integer(0)], ir.TensorLoad(alias, [integer(0)])), 5))
    if consumer == "store":
        accesses.append(_unit(ir.TensorStore(alias, [integer(0)], integer(7)), 2))
    elif consumer == "both":
        accesses.append(_unit(ir.TensorStore(output, [integer(0)], ir.TensorLoad(alias, [integer(0)])), 2))
    result = _resolve(_kernel([output], [_unit(allocate(state)), definition, *accesses]))
    expected = {"load": 5, "store": 2, "both": 7}[consumer]
    assert _binding_mask(result, state) == _binding_mask(result, alias) == expected


def test_register_write_demand_uses_alias_storage_identity():
    source = tensor("source", [16], "int32", "global")
    state = tensor("state", [4], "int32", "local.var")
    alias, definition = _view(state, "alias", [1], offset=4)
    output = tensor("output", [1], "int32", "global")
    writer = _unit(ir.TensorStore(state, [integer(1)], integer(3)))
    reader = _unit(ir.TensorStore(output, [integer(0)], ir.TensorLoad(alias, [integer(0)])), 2)
    result = _resolve(_kernel([source, output], [_unit(allocate(state)), definition, writer, reader]))
    assert _store_masks(result, state) == [2]
    assert _binding_mask(result, alias) == _binding_mask(result, state) == 2


@pytest.mark.parametrize("wrapper", ["if", "else", "assume", "begin", "end", "step"])
def test_header_reads_demand_scalar_producers_on_descendant_cores(wrapper):
    value = ir.Var("value", "int32")
    source = tensor("source", [1], "int32", "global")
    output = tensor("output", [1], "int32", "global")
    store = _unit(ir.TensorStore(output, [integer(0)], integer(9)), 2)
    child = ir.If(value > 0, [store], None)
    if wrapper == "else":
        child = ir.If(value > 0, [], [store])
    elif wrapper == "assume":
        child = attribute(value > 0, "tilelang_assume", integer(1), [store])
    elif wrapper in {"begin", "end", "step"}:
        child = _loop(ir.Var("i", "int32"), [store], **{wrapper: value})
    result = _resolve(_kernel([source, output], [_unit(ir.Bind(value, ir.TensorLoad(source, [integer(0)]))), child]))
    assert _binding_mask(result, value) == 2


@pytest.mark.parametrize("wrapper", ["if", "else", "assume", "loop"])
def test_core_local_header_cannot_control_the_other_core(wrapper):
    local = tensor("local", [1], "int32", "shared.dyn")
    output = tensor("output", [1], "int32", "global")
    bound = ir.TensorLoad(local, [integer(0)])
    store = _unit(ir.TensorStore(output, [integer(0)], integer(9)), 2)
    child = ir.If(bound > 0, [store], None)
    if wrapper == "else":
        child = ir.If(bound > 0, [], [store])
    elif wrapper == "assume":
        child = attribute(bound > 0, "tilelang_assume", integer(1), [store])
    elif wrapper == "loop":
        child = _loop(ir.Var("i", "int32"), [store], end=bound)
    with pytest.raises((ValueError, RuntimeError), match="Cannot evaluate a loop bound or control guard"):
        _resolve(_kernel([output], [_unit(allocate(local)), child]))


def _counter_ring(*, versions=2, owners=2, mixed=False, wrapper="plain", alias=False):
    source = tensor("source", [owners * 4, 16], scope="global")
    output = tensor("output", [owners * 4, 16], scope="global")
    ub = tensor("ub", [32], scope="shared.dyn")
    epoch = tensor("epoch", [1], "int32", "local.var")
    accum = tensor("accum", [1, 16], scope="shared.l0c")
    prefix = [_unit(allocate(ub)), _unit(allocate(epoch)), _unit(allocate(accum))]
    data, counter = ub, epoch
    if alias:
        data, define_data = _view(ub, "view", [16], "float32", offset=32)
        counter, define_counter = _view(epoch, "counter_view", [1])
        prefix.extend([define_data, define_counter])
    prefix.append(annotation({"tl.buffer_versions_map": {ub: versions}}))
    prefix.append(_unit(ir.TensorStore(epoch, [integer(0)], integer(0))))
    loops = []
    for owner in range(owners):
        var = ir.Var("i" + str(owner), "int32")
        guard = var % 2 == 0
        data_region = subregion(data, [0], [16])
        incoming = region(accum) if mixed and owner == 0 else subregion(source, [owner * 4 + var, 0], [1, 16])
        body = [
            _unit(tileop("ascend_copy", incoming, data_region), 2 if mixed and owner == 0 else 5, guard=guard),
            _unit(tileop("ascend_copy", data_region, subregion(output, [owner * 4 + var, 0], [1, 16])), 5, stage=1, guard=guard),
            _unit(ir.TensorStore(epoch, [integer(0)], ir.TensorLoad(epoch, [integer(0)]) + 1), guard=guard),
        ]
        metadata = {"multi_buffer_eligible": [ub], "tl.multi_buffer_counter_map": {ub: counter}, "tl.storage_epoch_guard_map": {ub: guard}}
        loops.append(_loop(var, body, metadata))
    enabled = ir.Var("enabled", "bool")
    if wrapper == "if_else":
        loops = [ir.If(enabled, [loops[0]], loops[1:] or [])]
    elif wrapper == "assume":
        loops = [attribute(enabled, "tilelang_assume", integer(1), loops)]
    return _kernel([source, output, enabled], [*prefix, *loops]), ub, epoch, counter


@pytest.mark.parametrize("versions", [1, 3])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("wrapper,alias", [("plain", False), ("if_else", True), ("assume", True)])
def test_counter_updates_cover_all_owners_and_native_alias_users(versions, mixed, wrapper, alias):
    before, storage, epoch, counter = _counter_ring(versions=versions, mixed=mixed, wrapper=wrapper, alias=alias)
    plan = _inspect(before, [storage])
    (info,) = plan["infos"]
    assert int(info["versions"]) == versions and info["uses_counter"]
    assert len(info["owners"]) == 2
    assert info["counter"].same_as(counter) and info["counter_storage"].same_as(epoch)
    result = _resolve(before)
    assert _store_masks(result, epoch) == [7 if mixed else 5] * 3
    assert _binding_mask(result, counter) == (7 if mixed else 5)
    after_plan = _inspect(result, [storage])
    ir.assert_structural_equal(plan["infos"], after_plan["infos"])


@pytest.mark.parametrize("source_scope", ["global", "shared.dyn", "local.var"])
def test_prepared_epoch_guard_reads_keep_pointer_provenance(source_scope):
    pred = tensor("pred", [1], "int32", source_scope)
    guard, define_guard = _view(pred, "guard_view", [1])
    ub = tensor("ub", [16, 16], scope="shared.dyn")
    accum = tensor("accum", [16, 16], scope="shared.l0c")
    out = tensor("out", [16, 16], scope="global")
    epoch = tensor("epoch", [1], "int32", "local.var")
    prefix = ([] if source_scope == "global" else [_unit(allocate(pred))]) + [
        define_guard,
        _unit(allocate(ub)),
        _unit(allocate(accum)),
        _unit(allocate(epoch)),
    ]
    if source_scope != "global":
        prefix.append(_unit(ir.TensorStore(pred, [integer(0)], integer(1)), 5 if source_scope == "shared.dyn" else 7))
    prefix.extend([annotation({"tl.buffer_versions_map": {ub: 2}}), _unit(ir.TensorStore(epoch, [integer(0)], integer(0)))])
    active = ir.TensorLoad(guard, [integer(0)]) > 0
    body = [
        _unit(_copy(accum, ub), 2),
        _unit(_copy(ub, out), 5),
        _unit(ir.TensorStore(epoch, [integer(0)], ir.TensorLoad(epoch, [integer(0)]) + 1)),
    ]
    owner = _loop(
        ir.Var("i", "int32"),
        body,
        {"multi_buffer_eligible": [ub], "tl.multi_buffer_counter_map": {ub: epoch}, "tl.storage_epoch_guard_map": {ub: active}},
    )
    before = _kernel([out, *([pred] if source_scope == "global" else [])], [*prefix, owner])
    if source_scope == "shared.dyn":
        with pytest.raises((ValueError, RuntimeError), match="storage-epoch guard.*is unavailable"):
            _resolve(before)
    else:
        result = _resolve(before)
        assert _binding_mask(result, guard) == 7
        if source_scope == "local.var":
            assert _store_masks(result, pred) == [7]


@pytest.mark.parametrize("header", ["if", "else", "assume", "loop"])
def test_counter_demand_includes_header_only_storage_accesses(header):
    ub = tensor("ub", [1], "int32", "shared.dyn")
    epoch = tensor("epoch", [1], "int32", "local.var")
    output = tensor("output", [1], "int32", "global")
    access = ir.TensorLoad(ub, [integer(0)])
    task = _unit(ir.TensorStore(output, [integer(0)], integer(1)), 5)
    child = ir.If(access > 0, [task], None)
    if header == "else":
        child = ir.If(access > 0, [], [task])
    elif header == "assume":
        child = attribute(access > 0, "tilelang_assume", integer(1), [task])
    elif header == "loop":
        child = _loop(ir.Var("j", "int32"), [task], end=access)
    metadata = {
        "multi_buffer_eligible": [ub],
        "tl.multi_buffer_counter_map": {ub: epoch},
        "tl.storage_epoch_guard_map": {ub: integer(True, "bool")},
    }
    owner = _loop(
        ir.Var("i", "int32"), [child, _unit(ir.TensorStore(epoch, [integer(0)], ir.TensorLoad(epoch, [integer(0)]) + 1))], metadata
    )
    # An unrelated Cube task would attract an incorrectly undemanded counter.
    before = _kernel(
        [output],
        [
            _unit(allocate(ub)),
            _unit(allocate(epoch)),
            annotation({"tl.buffer_versions_map": {ub: 2}}),
            _unit(ir.TensorStore(epoch, [integer(0)], integer(0))),
            owner,
            _unit(ir.TensorStore(output, [integer(0)], integer(2)), 2),
        ],
    )
    result = _resolve(before)
    assert _store_masks(result, epoch) == [5, 5]
    assert _binding_mask(result, epoch) == 5


@pytest.mark.parametrize("kind", ["iteration", "counter"])
def test_broadcast_fill_outside_owner_preserves_marker_and_has_no_counter_demand(kind):
    before, storage, epoch, _ = _counter_ring(owners=1)
    root = _attributes(before, "tl.sblock")[0]
    body = list(root.body.seq)
    fill = _unit(
        tileop("fill", subregion(storage, [0], [16]), ir.FloatImm("float32", 0)), 5, metadata={"tl.multi_buffer_broadcast_fill": [storage]}
    )
    body.insert(len(body) - 1, fill)
    if kind == "iteration":
        owner = objects(before, ir.For)[0]
        owner_fields = _fields(owner.body.seq[0])["value"]
        tasks = [stmt for stmt in owner.body.seq[0].body.seq if not objects(stmt, ir.TensorStore)]
        body[-1] = _loop(owner.var, tasks, {"multi_buffer_eligible": [storage]})
        assert "tl.multi_buffer_counter_map" in owner_fields
    result = _resolve(_kernel(list(before.params), body))
    (filled,) = [task for task in _tasks(result) if calls(task, "tl.tileop.fill")]
    assert _fields(filled)["node"]["tl.multi_buffer_broadcast_fill"][0].same_as(storage)
    assert _mask(filled) == 5


@pytest.mark.parametrize(
    "malformed,pattern",
    [
        ("no_owner", "has no owner"),
        ("missing_guard", "missing its active guard"),
        ("missing_counter", "Every disjoint owner"),
        ("different_counter", "share one counter"),
        ("nested", "nested owner"),
        ("duplicate", "Duplicate multi-buffer owner"),
    ],
)
def test_prepared_multi_buffer_protocol_rejects_inconsistent_metadata(malformed, pattern):
    storage = tensor("storage", [1], "int32", "shared.dyn")
    epoch = tensor("epoch", [1], "int32", "local.var")
    other = tensor("other", [1], "int32", "local.var")
    prefix = [_unit(allocate(var)) for var in [storage, epoch, other]]
    loops = []
    for index in range(2 if malformed in {"missing_counter", "different_counter", "nested"} else 1):
        fields = {
            "multi_buffer_eligible": [storage],
            "tl.multi_buffer_counter_map": {storage: epoch},
            "tl.storage_epoch_guard_map": {storage: integer(True, "bool")},
        }
        if malformed == "no_owner":
            fields.pop("multi_buffer_eligible")
        elif malformed == "missing_guard":
            fields.pop("tl.storage_epoch_guard_map")
        elif malformed == "missing_counter" and index == 1:
            fields.pop("tl.multi_buffer_counter_map")
        elif malformed == "different_counter" and index == 1:
            fields["tl.multi_buffer_counter_map"] = {storage: other}
        elif malformed == "duplicate":
            fields["multi_buffer_eligible"] = [storage, storage]
        body = [_unit(ir.TensorStore(storage, [integer(0)], integer(1)), 5)]
        if malformed == "nested" and loops:
            body.extend(loops)
            loops.clear()
        loops.append(_loop(ir.Var("i" + str(index), "int32"), body, fields))
    before = _kernel([], [*prefix, *loops], {storage: 2})
    with pytest.raises((ValueError, RuntimeError), match=pattern):
        _resolve(before)


def test_loop_header_is_outside_its_own_storage_epoch():
    storage = tensor("storage", [1], "int32", "shared.dyn")
    epoch = tensor("epoch", [1], "int32", "local.var")
    metadata = {
        "multi_buffer_eligible": [storage],
        "tl.multi_buffer_counter_map": {storage: epoch},
        "tl.storage_epoch_guard_map": {storage: integer(True, "bool")},
    }
    owner = _loop(
        ir.Var("i", "int32"),
        [_unit(ir.TensorStore(storage, [integer(0)], integer(1)), 5)],
        metadata,
        end=ir.TensorLoad(storage, [integer(0)]),
    )
    before = _kernel([], [_unit(allocate(storage)), _unit(allocate(epoch)), owner], {storage: 2})
    with pytest.raises((ValueError, RuntimeError), match="outside every owner"):
        _resolve(before)


@pytest.mark.parametrize("wrapper", ["if", "else", "assume", "loop"])
def test_special_register_writers_nested_in_lexical_scopes_follow_outer_readers(wrapper):
    source = tensor("source", [16], scope="global")
    ub = tensor("ub", [16], scope="shared.dyn")
    condition = ir.Var("enabled", "bool")
    writer = _unit(ir.Evaluate(T.ascend_set_copy_pad_value(-1.0, dtype="float32")))
    nested = ir.If(condition, [writer], None)
    if wrapper == "else":
        nested = ir.If(condition, [], [writer])
    elif wrapper == "assume":
        nested = attribute(condition, "tilelang_assume", integer(1), [writer])
    elif wrapper == "loop":
        nested = _loop(ir.Var("i", "int32"), [writer])
    reader = _unit(_copy(source, ub, {"data_select": integer(1)}), 5)
    result = _resolve(_kernel([source, condition], [_unit(allocate(ub)), nested, reader]))
    (setter,) = [task for task in _tasks(result) if calls(task, "tl.ascend_set_copy_pad_value")]
    assert _mask(setter) == 5


@pytest.mark.parametrize("cube_owners", [(), (0,), (1,), (None,), (0, 1)])
@pytest.mark.parametrize("wrapper", ["plain", "if", "loop"])
def test_projected_break_retains_its_cube_domain_through_native_scopes(cube_owners, wrapper):
    source = tensor("source", [16], scope="global")
    buffers = [tensor("ub" + str(i), [16], scope="shared.dyn") for i in range(2)]
    l1 = tensor("l1", [16], scope="shared.l1")
    index = ir.Var("i", "int32")
    body = []
    for sid, mask in enumerate((1, 4)):
        body.extend([_unit(ir.Break(), mask | 2, sid=sid, guard=index >= sid + 1), _unit(_copy(source, buffers[sid]), mask)])
    for sid in cube_owners:
        cube = _unit(_copy(source, l1), 2, sid=sid)
        if wrapper == "if":
            cube = ir.If(index > 0, [cube], None)
        elif wrapper == "loop":
            cube = _loop(ir.Var("j", "int32"), [cube])
        body.append(cube)
    # Opaque scheduling markers for pure Bind values must not seed shared Cube.
    body.append(_unit(ir.Bind(ir.Var("unused", "int32"), index)))
    before = _kernel([source], [*[_unit(allocate(var)) for var in [*buffers, l1]], _loop(index, body)])
    if cube_owners in {(None,), (0, 1)}:
        with pytest.raises((ValueError, RuntimeError), match="Independent subcore loop_break.*separate loop"):
            _resolve(before)
    else:
        result = _resolve(before)
        exits = [task for task in _tasks(result) if objects(task, ir.Break)]
        assert {int(_fields(task)["node"]["subcore"]): _mask(task) for task in exits} == {
            sid: mask | (2 if sid in cube_owners else 0) for sid, mask in enumerate((1, 4))
        }


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("materialized", [False, True])
def test_resolve_uses_kernel_vector_count_for_raw_and_materialized_launches(count, materialized):
    source = tensor("source", [16], scope="global")
    ub = tensor("ub", [16], scope="shared.dyn")
    scalar = ir.Var("scalar", "float32")
    sid = ir.Var("sid", "int32")
    output = tensor("output", [1], scope="global")
    before = _raw_kernel(
        [source, output],
        [
            allocate(ub),
            _copy(source, ub),
            ir.Bind(scalar, ir.TensorLoad(source, [integer(0)])),
            ir.TensorStore(output, [integer(0)], scalar),
        ],
        sid,
        count,
    )
    if materialized:
        before = _materialize(before)
    result = _resolve(_assign(before))
    assert _binding_mask(result, scalar) == (1 if count == 1 else 5)
    assert _store_masks(result, output) == [1 if count == 1 else 5]


@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("consumer", ["copy", "loop_bound", "loop_break"])
@pytest.mark.parametrize("shared_consumer", [False, True])
def test_later_scalar_consumers_propagate_projected_loop_exit_demand(owner, consumer, shared_consumer):
    source = tensor("source", [4, 16], scope="global")
    l1 = tensor("l1", [16], scope="shared.l1")
    ub = tensor("ub", [16], scope="shared.dyn")
    count = tensor("count", [1], "int32", "local.var")
    index, later, offset = [ir.Var(name, "int32") for name in ["i", "j", "offset"]]
    stop = ir.Var("stop", "bool")
    body = [
        _unit(ir.Bind(stop, index == 1)),
        _unit(ir.Break(), 3, sid=0, guard=stop),
        _unit(ir.Break(), 6, sid=1, guard=stop),
        _unit(ir.TensorStore(count, [integer(0)], ir.TensorLoad(count, [integer(0)]) + 1), sid=owner),
        _unit(tileop("ascend_copy", subregion(source, [index, 0], [1, 16]), region(ub)), 4 if owner == 0 else 1, sid=1 - owner),
    ]
    copy = _unit(
        tileop("ascend_copy", subregion(source, [offset if consumer == "copy" else integer(0), 0], [1, 16]), region(l1)),
        2,
        sid=None if shared_consumer else owner,
    )
    if consumer == "copy":
        suffix = [_unit(ir.Bind(offset, ir.TensorLoad(count, [integer(0)]) - 1)), copy]
    else:
        reads = [copy]
        if consumer == "loop_break":
            reads.insert(0, _unit(ir.Break(), guard=later >= ir.TensorLoad(count, [integer(0)])))
        suffix = [_loop(later, reads, end=ir.TensorLoad(count, [integer(0)]) if consumer == "loop_bound" else 4)]
    before = _kernel(
        [source],
        [
            *[_unit(allocate(var)) for var in [l1, ub, count]],
            _unit(ir.TensorStore(count, [integer(0)], integer(0)), sid=owner),
            _loop(index, body),
            *suffix,
        ],
    )
    result = _resolve(before)
    exits = [task for task in _tasks(result) if objects(task, ir.Break) and "subcore" in _fields(task)["node"]]
    assert {int(_fields(task)["node"]["subcore"]): _mask(task) for task in exits} == {
        sid: 2 if sid == owner else mask for sid, mask in enumerate((1, 4))
    }
    assert _store_masks(result, count) == [2, 2]


def test_epoch_projection_uses_each_branch_context_and_distinct_sibling_lists():
    n = ir.Var("n", "int32")
    x, y = [tensor(name, [1], "int32", "shared.dyn") for name in ["x", "y"]]
    loops = []
    for name in ["then_loop", "else_loop"]:
        body = [_unit(ir.TensorStore(var, [integer(0)], integer(1)), 5) for var in [x, y]]
        loops.append(_loop(ir.Var(name, "int32"), body, {"tl.storage_epoch_guard_map": {x: n >= 0, y: n >= 1}}))
    branch = ir.If(n >= 1, [loops[0]], [loops[1]])
    before = _kernel([n], [_unit(allocate(x)), _unit(allocate(y)), branch])
    plan = _inspect(before, [x, y])
    branch_scopes = [scope for scope in plan["scopes"] if isinstance(scope.get("owner"), ir.If)]
    assert len(branch_scopes) == 2
    assert {int(scope["list"]) for scope in branch_scopes} == {0, 1}
    assert len({int(scope["unconditional"]) for scope in branch_scopes}) == 2
    loop_scopes = {scope["owner"].var.name: scope for scope in plan["scopes"] if isinstance(scope.get("owner"), ir.For)}
    assert all(all(bool(item) for item in row) for row in loop_scopes["then_loop"]["projection"])
    assert bool(loop_scopes["else_loop"]["projection"][0][1])
    assert not bool(loop_scopes["else_loop"]["projection"][1][0])


@pytest.mark.parametrize("different_guards", [False, True])
def test_shared_counter_group_requires_matching_owner_guards(different_guards):
    n = ir.Var("n", "int32")
    x, y = [tensor(name, [1], "int32", "shared.dyn") for name in ["x", "y"]]
    epoch = tensor("epoch", [1], "int32", "local.var")
    metadata = {
        "multi_buffer_eligible": [x, y],
        "tl.multi_buffer_counter_map": {x: epoch, y: epoch},
        "tl.storage_epoch_guard_map": {x: n > 0, y: n > (1 if different_guards else 0)},
    }
    owner = _loop(ir.Var("i", "int32"), [_unit(ir.TensorStore(var, [integer(0)], integer(1)), 5) for var in [x, y]], metadata)
    before = _kernel([n], [_unit(allocate(var)) for var in [x, y, epoch]] + [owner], {x: 2, y: 3})
    if different_guards:
        with pytest.raises((ValueError, RuntimeError), match="inconsistent active guards"):
            _inspect(before, [x, y])
    else:
        plan = _inspect(before, [x, y])
        assert len({int(info["counter_group"]) for info in plan["infos"]}) == 1
        (scope,) = [scope for scope in plan["scopes"] if isinstance(scope.get("owner"), ir.For)]
        assert scope["domains"][0]["id"] == scope["domains"][1]["id"]


@pytest.mark.parametrize(
    "case,expected",
    [
        ("pure", True),
        ("alias_fill", True),
        ("read_alias", False),
        ("store_alias", False),
        ("bounds", False),
        ("metadata", False),
        ("assume", False),
        ("fresh_read", True),
        ("external_read", False),
    ],
)
def test_broadcast_fill_rejects_target_reads_and_unproven_aliases(case, expected):
    storage = tensor("storage", [16], "int32", "shared.dyn")
    alias, define_alias = _view(storage, "alias", [8], offset=16)
    other = tensor("other", [1], "int32", "shared.dyn")
    load = ir.TensorLoad(alias, [integer(0)])
    destination = region(alias if case == "alias_fill" else storage)
    value = load if case == "read_alias" else ir.TensorLoad(other, [integer(0)]) if case in {"fresh_read", "external_read"} else integer(3)
    if case == "bounds":
        destination = subregion(storage, [load], [1])
    fill = tileop("fill", destination, value, metadata={"custom": load} if case == "metadata" else {})
    body = [fill]
    if case == "store_alias":
        body.append(ir.TensorStore(alias, [integer(0)], integer(2)))
    if case == "assume":
        body = [attribute(load > 0, "tilelang_assume", integer(1), body)]
    definitions = [allocate(storage), define_alias]
    if case != "external_read":
        definitions.append(allocate(other))
    result = tvm_ffi.get_global_func("tl.testing.CanBroadcastAscendFill")(_task(body), storage, ir.SeqStmt(definitions))
    assert bool(result) is expected


@pytest.mark.parametrize("per_core", [False, True])
def test_broadcast_metadata_roundtrip_preserves_native_operands(per_core):
    storage = tensor("storage", [1], "int32", "shared.dyn")
    scalar = ir.Var("scalar", "int32")
    key = "tl.ascend_per_core_task" if per_core else "tl.ascend_task"
    before = attribute({"core_mask": integer(5), "custom": {storage: scalar}}, key, integer(1), [tileop("fill", region(storage), scalar)])
    update = tvm_ffi.get_global_func("tl.testing.SetAscendBroadcastFill")
    added = update(before, storage, True)
    repeated = update(added, storage, True)
    assert len(_fields(added)["node"]["tl.multi_buffer_broadcast_fill"]) == 1
    ir.assert_structural_equal(added, repeated)
    assert "tl.multi_buffer_broadcast_fill" not in _fields(before)["node"]
    ir.assert_structural_equal(before, update(repeated, storage, False))


def test_native_break_inside_if_exits_all_enclosing_loop_streams():
    ub = tensor("ub", [4], "int32", "shared.dyn")
    l1 = tensor("l1", [4], "int32", "shared.l1")
    enabled = ir.Var("enabled", "bool")
    index = ir.Var("i", "int32")
    loop = ir.For(
        index,
        integer(0),
        integer(4),
        integer(1),
        ir.ForKind.DEFAULT,
        [ir.If(enabled, [ir.Break()], None), ir.TensorStore(ub, [index], index), ir.TensorStore(l1, [index], index)],
    )
    result = _resolve(_assign(_raw_kernel([enabled], [allocate(ub), allocate(l1), loop])))
    (exit_task,) = [task for task in _tasks(result) if objects(task, ir.Break)]
    assert _mask(exit_task) == 7
    assert len(objects(result, ir.Break)) == 1


@pytest.mark.parametrize("kind", ["for", "while"])
def test_task_internal_native_break_does_not_exit_an_enclosing_scheduled_loop(kind):
    source = tensor("source", [16], scope="global")
    ub = tensor("ub", [16], scope="shared.dyn")
    l1 = tensor("l1", [16], scope="shared.l1")
    inner = ir.For(ir.Var("j", "int32"), integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [ir.Break()])
    if kind == "while":
        inner = ir.While(integer(True, "bool"), [ir.Break()])
    grouped = _unit(ir.SeqStmt([inner, _copy(source, ub)]), 1, sid=0)
    owner = _loop(ir.Var("i", "int32"), [grouped, _unit(_copy(source, l1), 2)])
    result = _resolve(_kernel([source], [_unit(allocate(ub)), _unit(allocate(l1)), owner]))
    (group,) = [task for task in _tasks(result) if objects(task, ir.Break)]
    assert _mask(group) == 1


def test_native_break_keeps_may_writes_but_cannot_prove_full_initialization():
    output = tensor("output", [4], "int32", "shared.dyn")
    index = ir.Var("i", "int32")
    statements = [ir.TensorStore(output, [index], integer(1)), ir.If(index == 1, [ir.Break()], None)]
    loop = ir.For(index, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, statements)
    result = tvm_ffi.get_global_func("tl.testing.AnalyzeAscendMemory")(loop, ir.SeqStmt([allocate(output)]), False)
    assert len(result["writes"]) == 1 and not result["must_writes"]


@pytest.mark.parametrize("origin", ["parameter", "integer", "fresh"])
@pytest.mark.parametrize("writer_core", [2, 7])
def test_unknown_register_roots_share_demand_until_allocations_prove_disjoint(origin, writer_core):
    left, right = [tensor(name, [1], "int32", "local.var") for name in ["left", "right"]]
    params, prefix = [], []
    for value in [left, right]:
        if origin == "fresh":
            prefix.append(_unit(allocate(value)))
        else:
            pointer_type = ir.PtrType(ir.PrimType("int32"), "local.var")
            source = ir.Var(value.name + "_address", "uint64" if origin == "integer" else pointer_type)
            params.append(source)
            pointer = ir.Call("tirx.reinterpret", [source], ty=pointer_type) if origin == "integer" else source
            prefix.append(_unit(ir.Bind(value, ir.Call("tirx.decl_tensor", [pointer], ty_args=[value.ty]))))
    output = tensor("output", [1], "int32", "global")
    params.append(output)
    writer = _unit(ir.TensorStore(right, [integer(0)], integer(4)), writer_core)
    reader = _unit(ir.TensorStore(output, [integer(0)], ir.TensorLoad(left, [integer(0)])), 5)
    unrelated_cube = _unit(ir.TensorStore(output, [integer(0)], integer(7)), 2)
    before = _kernel(params, [*prefix, writer, reader, unrelated_cube])
    if origin != "fresh" and writer_core == 2:
        with pytest.raises((ValueError, RuntimeError), match="unavailable on every legal execution core"):
            _resolve(before)
    else:
        result = _resolve(before)
        assert _store_masks(result, right) == [2 if origin == "fresh" else 5]


@pytest.mark.parametrize("materialized", [False, True])
def test_per_core_candidate_selection_demands_its_scalar_predicate(materialized):
    source = tensor("source", [16], scope="global")
    ub = tensor("ub", [16], scope="shared.dyn")
    choice = tensor("choice", [1], "int32", "global")
    predicate = ir.Var("predicate", "bool")
    group = attribute(
        {}, "tl.ascend_per_core_task", integer(1), [ir.If(predicate, [_task([_copy(source, ub)])], [_task([_copy(source, ub)])])]
    )
    before = _raw_kernel(
        [source, choice], [allocate(ub), ir.Bind(predicate, ir.TensorLoad(choice, [integer(0)]) > 0), group], ir.Var("sid", "int32")
    )
    if materialized:
        before = _materialize(before)
    result = _resolve(_assign(before))
    (resolved,) = _attributes(result, "tl.ascend_per_core_task")
    assert _mask(resolved) == _binding_mask(result, predicate) == 5
    assert len(_attributes(resolved.body, "tl.ascend_task")) == 2
