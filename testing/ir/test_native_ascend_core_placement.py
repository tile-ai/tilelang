"""Ascend placement preserves native SSA, lexical controls and routing effects."""

from testing.ascend._ir import kernel_body

import operator

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from test_native_ascend_dual_copy import _launch
from test_native_ascend_ops import function, strided_tensor
from test_native_ascend_schedule_units import SCHEDULED_BIND, _attributes, _check, _fields, _task
from test_native_cuda_copy import calls
from test_native_reducer_contracts import (
    allocate,
    annotation,
    annotations,
    attribute,
    decoded,
    integer,
    objects,
    region,
    run_pass,
    tensor,
    tileop,
)


def _kernel(params, body, sid=None, count=2):
    root = ir.SeqStmt([annotation({"vector_count": integer(count)}), *body])
    return function(params, [_launch([root], var=sid, extent=count) if sid is not None else root])


def _assign(value):
    scheduled = _check(value)
    before = ir.save_json(scheduled)
    result = run_pass("AssignCore", scheduled)
    assert ir.save_json(scheduled) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def _tasks(value):
    return _attributes(value, "tl.ascend_task")


def _metadata(task):
    return _fields(task)["node"]


def _mask(task):
    return int(_metadata(task)["core_mask"])


def _copy(source, destination, metadata=None):
    return tileop("ascend_copy", region(source), region(destination), metadata=metadata)


def _flag(pipe, flag=3):
    return ir.Evaluate(ir.Call("tl.ascend_cross_core_set_flag", [integer(4), ir.StringImm(pipe), integer(flag)], ty=ir.PrimType("void")))


def _int(expr):
    return int(tvm_tirx.sym.Analyzer().simplify(expr))


@pytest.mark.parametrize("scope,expected", [("shared.dyn", 5), ("shared.l1", 2)])
@pytest.mark.parametrize("count", [1, 2])
def test_copy_destination_and_vector_count_select_the_core(scope, expected, count):
    source = tensor("source", [16, 16], scope="global")
    destination = tensor("destination", [16, 16], scope=scope)
    result = _assign(_kernel([source], [allocate(destination), _copy(source, destination)], count=count))
    (task,) = _tasks(result)
    assert _mask(task) == expected & (3 if count == 1 else 7)


def test_uniform_vector_work_preserves_one_body_and_storage():
    source = tensor("source", [16], scope="global")
    scratch = tensor("scratch", [16], scope="shared.dyn")
    sid = ir.Var("sid", "int32")
    result = _assign(_kernel([source], [allocate(scratch), _copy(source, scratch)], sid))
    assert len(_tasks(result)) == 1
    assert objects(result, ir.Bind)[0].var.same_as(scratch)
    assert "subcore" not in _metadata(_tasks(result)[0])
    assert not any("tl.split_subcores" in row for row in annotations(result))


@pytest.mark.parametrize("scope,core", [("shared.dyn", 1), ("shared.l1", 2)])
@pytest.mark.parametrize("selected", [0, 1])
@pytest.mark.parametrize("wrapper", ["if", "assume"])
def test_subcore_condition_is_specialized_before_core_assignment(scope, core, selected, wrapper):
    source = tensor("source", [16, 16], scope="global")
    destination = tensor("destination", [16, 16], scope=scope)
    sid = ir.Var("sid", "int32")
    child = ir.If(sid == selected, [_copy(source, destination)], None)
    if wrapper == "assume":
        child = attribute(sid >= 0, "tilelang_assume", integer(1), [child])
    result = _assign(_kernel([source], [allocate(destination), child], sid))
    (task,) = _tasks(result)
    assert _mask(task) == (4 if core == 1 and selected == 1 else core)
    assert int(_metadata(task)["subcore"]) == selected
    assert not any(var.same_as(sid) for var in tirx.analysis.undefined_vars(task))


@pytest.mark.parametrize("control", ["if", "else", "loop", "assume", "bind"])
def test_enclosing_core_local_inputs_cannot_control_the_other_core(control):
    source = tensor("source", [16, 16], scope="global")
    pred = tensor("pred", [1], "int32", "shared.dyn")
    destination = tensor("destination", [16, 16], scope="shared.l1")
    condition = ir.TensorLoad(pred, [integer(0)]) > 0
    copy = _copy(source, destination)
    prefix = []
    if control == "bind":
        saved = ir.Var("condition", "bool")
        prefix = [ir.Bind(saved, condition)]
        child = ir.If(saved, [copy], None)
    elif control == "if":
        child = ir.If(condition, [copy], None)
    elif control == "else":
        child = ir.If(condition, [], [copy])
    elif control == "loop":
        child = ir.For(ir.Var("i", "int32"), integer(0), ir.TensorLoad(pred, [integer(0)]), integer(1), ir.ForKind.DEFAULT, [copy])
    else:
        child = attribute(condition, "tilelang_assume", integer(1), [copy])
    with pytest.raises((ValueError, RuntimeError), match="unavailable on every legal execution core"):
        _assign(_kernel([source], [allocate(pred), allocate(destination), *prefix, child]))


def test_scalar_memory_task_cannot_mix_cube_and_vector_storage():
    ub = tensor("ub", [1], "int32", "shared.dyn")
    l1 = tensor("l1", [1], "int32", "shared.l1")
    value = ir.Var("value", "int32")
    with pytest.raises((ValueError, RuntimeError), match="incompatible core-local memories"):
        _assign(
            _kernel([], [allocate(ub), allocate(l1), ir.Bind(value, ir.TensorLoad(ub, [integer(0)]) + ir.TensorLoad(l1, [integer(0)]))])
        )


@pytest.mark.parametrize("control", ["if_else", "assume", "loop"])
def test_cross_core_flags_resolve_in_each_lexical_child_list(control):
    enabled = ir.Var("enabled", "bool")
    vector, cube = [_flag("PIPE_V"), _flag("PIPE_S")], [_flag("PIPE_M"), _flag("PIPE_S")]
    if control == "if_else":
        children = [ir.If(enabled, vector, cube)]
    elif control == "assume":
        children = [attribute(enabled, "tilelang_assume", integer(1), vector), attribute(~enabled, "tilelang_assume", integer(1), cube)]
    else:
        children = [
            ir.For(ir.Var("i", "int32"), integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, vector),
            ir.For(ir.Var("i", "int32"), integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, cube),
        ]
    result = _assign(_kernel([enabled], children))
    assert [_mask(task) for task in _tasks(result)] == [5, 5, 2, 2]


@pytest.mark.parametrize("control", ["if", "loop", "assume"])
def test_ambiguous_scalar_pipe_uses_ancestor_inputs(control):
    source = tensor("source", [16], scope="global")
    cube = tensor("cube", [16], scope="shared.l1")
    pred = tensor("pred", [1], "int32", "shared.dyn")
    flag = _flag("PIPE_S")
    condition = ir.TensorLoad(pred, [integer(0)]) > 0
    if control == "if":
        flag = ir.If(condition, [flag], None)
    elif control == "loop":
        flag = ir.For(ir.Var("i", "int32"), integer(0), ir.TensorLoad(pred, [integer(0)]), integer(1), ir.ForKind.DEFAULT, [flag])
    else:
        flag = attribute(condition, "tilelang_assume", integer(1), [flag])
    result = _assign(_kernel([source], [allocate(cube), allocate(pred), _copy(source, cube), flag]))
    assert [_mask(task) for task in _tasks(result)] == [2, 5]


def test_same_scope_cross_core_flag_cannot_resolve_to_opposing_cores():
    with pytest.raises((ValueError, RuntimeError), match="both Vector and Cube"):
        _assign(_kernel([], [_flag("PIPE_V"), _flag("PIPE_M")]))


def test_same_scope_ambiguous_cross_core_flag_requires_a_core():
    with pytest.raises((ValueError, RuntimeError), match="Cannot infer the core"):
        _assign(_kernel([], [_flag("PIPE_S")]))


@pytest.mark.parametrize("field", ["shape", "stride", "origin"])
@pytest.mark.parametrize("dtype,bits", [("int32", 32), ("float4_e2m1fn", 4)])
def test_projected_tensor_types_pointer_aliases_and_metadata_preserve_physical_offsets(field, dtype, bits):
    sid = ir.Var("sid", "int32")
    storage = tensor("storage", [256], "uint8", "shared.dyn")
    shape = [sid + 2 if field == "shape" else integer(2)]
    stride = [sid + 1 if field == "stride" else integer(2)]
    origin = sid + 3 if field == "origin" else integer(3)
    view = strided_tensor("view", shape, stride, origin, dtype, "shared.dyn")
    pointer = ir.Var("pointer", ir.PtrType(ir.PrimType(dtype), "shared.dyn"))
    offset = ir.Call("tirx.ptr_byte_offset", [storage.data, integer(5, "int64")], ty=storage.data.ty)
    cast = ir.Call("tirx.reinterpret", [offset], ty=pointer.ty)
    define = ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty]))
    output = tensor("output", [2], dtype, "global")
    body = [
        allocate(storage),
        ir.Bind(pointer, cast),
        define,
        annotation({"test.views": {view: view}, "tl.buffer_versions_map": {storage: 2}}),
        ir.If(sid < 2, [ir.TensorStore(output, [sid], ir.TensorLoad(view, [integer(1)]))], None),
    ]
    result = _assign(_kernel([output], body, sid))
    views = [bind.var for bind in objects(result, ir.Bind) if bind.var.name.startswith("view_aiv")]
    assert len(views) == 2 and not views[0].same_as(views[1])
    root = kernel_body(result)
    rows = tvm_ffi.get_global_func("tl.testing.InspectAscendStorage")(root, views, [[integer(1)], [integer(1)]])
    assert not rows[0]["root"].same_as(rows[1]["root"])
    for p, (view, row) in enumerate(zip(views, rows)):
        expected_offset = (p + 3 if field == "origin" else 3) + (p + 1 if field == "stride" else 2)
        assert row["exact"] and _int(row["bytes"]) == 5
        assert _int(row["bits"]) == 40 + expected_offset * bits
        assert _int(view.ty.shape[0]) == (p + 2 if field == "shape" else 2)
        mappings = [fields["test.views"] for fields in annotations(result) if "test.views" in fields]
        assert len(mappings) == 1 and mappings[0][view].same_as(view)
    owners = {var: int(owner) for fields in annotations(result) for var, owner in fields.get("tl.aiv_storage_map", {}).items()}
    assert sorted(owners.values()) == [0, 1]
    assert all(row["root"] in owners for row in rows)


def _evaluate(expr, values):
    if isinstance(expr, ir.IntImm):
        return int(expr)
    if isinstance(expr, ir.Var):
        return values[expr]
    operators = {
        "Add": operator.add,
        "Sub": operator.sub,
        "Mul": operator.mul,
        "FloorDiv": operator.floordiv,
        "FloorMod": operator.mod,
        "LT": operator.lt,
        "LE": operator.le,
        "GT": operator.gt,
        "GE": operator.ge,
        "EQ": operator.eq,
        "NE": operator.ne,
    }
    name = type(expr).__name__
    if name in operators:
        return operators[name](_evaluate(expr.a, values), _evaluate(expr.b, values))
    raise AssertionError(type(expr))


def _store_trace(body, values):
    """Finite-domain mathematical execution; independent of core-mask inference."""
    events = []
    values = dict(values)

    class LoopBreak(Exception):
        pass

    def visit(stmt):
        if isinstance(stmt, ir.SeqStmt):
            for child in stmt.seq:
                visit(child)
        elif isinstance(stmt, ir.RegionStmt):
            visit(stmt.body)
        elif isinstance(stmt, ir.Bind):
            value = stmt.value
            if isinstance(value, ir.Call) and value.op.name == SCHEDULED_BIND:
                value = value.args[0]
            if isinstance(stmt.var.ty, ir.PrimType):
                values[stmt.var] = _evaluate(value, values)
        elif isinstance(stmt, ir.If):
            branch = stmt.then_case if _evaluate(stmt.condition, values) else stmt.else_case
            if branch is not None:
                visit(branch)
        elif isinstance(stmt, ir.For):
            index, end, step = (_evaluate(bound, values) for bound in (stmt.begin, stmt.end, stmt.step))
            trips = 0
            while index < end:
                trips += 1
                assert trips <= 4096, "The trace fixture must terminate"
                values[stmt.var] = index
                try:
                    visit(stmt.body)
                except LoopBreak:
                    break
                index += step
        elif isinstance(stmt, ir.Break):
            raise LoopBreak
        elif isinstance(stmt, ir.TensorStore):
            events.append((tuple(_evaluate(index, values) for index in stmt.indices), _evaluate(stmt.value, values)))
        else:
            assert isinstance(stmt, ir.Evaluate)

    visit(body)
    return events


@pytest.mark.parametrize("negative_step", [False, True])
def test_nested_projected_loops_keep_fresh_binders_and_match_independent_execution(negative_step):
    sid, i, j, value = (ir.Var(name, "int32") for name in ("sid", "i_aiv0", "j", "value"))
    output = tensor("output", [2, 5, 3], "int32", "global")
    inner = ir.For(
        j,
        integer(0),
        integer(3),
        integer(1),
        ir.ForKind.DEFAULT,
        [ir.Bind(value, sid * 100 + i * 10 + j), ir.TensorStore(output, [sid, i, j], value)],
    )
    outer = ir.For(
        i,
        sid + 3 if negative_step else integer(0),
        integer(-1) if negative_step else sid + 3,
        integer(-1 if negative_step else 1),
        ir.ForKind.DEFAULT,
        [attribute(i >= 0, "tilelang_assume", integer(1), [inner])],
    )
    before = _kernel([output], [outer], sid)
    result = _assign(before)
    loops = objects(result, ir.For)
    assert len(loops) == 4
    assert len({loop.var for loop in loops}) == 4
    assert {loop.var.name for loop in loops} == {f"{var.name}_aiv{p}" for var in (i, j) for p in (0, 1)}
    expected = [event for p in (0, 1) for event in _store_trace(kernel_body(before), {sid: p})]
    actual = _store_trace(kernel_body(result), {})
    assert actual == expected


@pytest.mark.parametrize("kind", ["immutable", "load", "loop"])
def test_fixpipe_routing_keeps_task_local_definitions_and_copy_hints(kind):
    sid, route = ir.Var("sid", "int32"), ir.Var("route", "int32")
    accum = tensor("accum", [16, 16], scope="shared.l0c")
    ub = tensor("ub", [16, 16], scope="shared.dyn")
    routes = tensor("routes", [1], "int32", "global")
    prefix = []
    if kind == "immutable":
        prefix = [ir.Bind(route, integer(1))]
    elif kind == "load":
        prefix = [ir.Bind(route, ir.TensorLoad(routes, [integer(0)]) % 2)]
    copy = _copy(accum, ub, {"sub_blockid": route, "unit_flag_ctrl": integer(3)})
    payload = prefix + [copy]
    if kind == "loop":
        payload = [ir.For(route, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [copy])]
    result = _assign(_kernel([routes], [allocate(accum), allocate(ub), _task(payload)], sid))
    tasks = _tasks(result)
    assert len(tasks) == (1 if kind == "immutable" else 2)
    for task in tasks:
        assert _mask(task) == 2
        for call in calls(task, "tl.tileop.ascend_copy"):
            _, hints = decoded(call)
            assert int(hints["sub_blockid"]) == int(_metadata(task)["subcore"])
            assert int(hints["unit_flag_ctrl"]) == 3
    if kind in {"load", "loop"}:
        assert all(len(objects(task, ir.If)) == 1 for task in tasks)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("route_kind", ["implicit", "sid", "local"])
def test_fixpipe_projection_rejects_replayed_external_effects_but_accepts_one_survivor(single, route_kind):
    sid, route = ir.Var("sid", "int32"), ir.Var("route", "int32")
    accum = tensor("accum", [16, 16], scope="shared.l0c")
    ub = tensor("ub", [16, 16], scope="shared.dyn")
    output = tensor("output", [1], "int32", "global")
    prefix = [ir.Bind(route, ir.TensorLoad(output, [integer(0)]) % 2)] if route_kind == "local" else []
    metadata = {} if route_kind == "implicit" else {"sub_blockid": sid if route_kind == "sid" else route}
    task = _task([*prefix, ir.TensorStore(output, [integer(0)], integer(7)), _copy(accum, ub, metadata)])
    if single:
        task = ir.If(sid == 0, [task], None)
    value = _kernel([output], [allocate(accum), allocate(ub), task], sid)
    if single:
        result = _assign(value)
        assert len(_tasks(result)) == 1
        assert len(objects(result, ir.TensorStore)) == 1
    else:
        with pytest.raises((ValueError, RuntimeError), match="Fixpipe routing.*external side effects"):
            _assign(value)


@pytest.mark.parametrize("selected", [0, 1])
def test_per_core_candidates_have_independent_explicit_fixpipe_routes(selected):
    sid = ir.Var("sid", "int32")
    selector = ir.Var("selector", "bool")
    accum = tensor("accum", [16, 16], scope="shared.l0c")
    ub = tensor("ub", [16, 16], scope="shared.dyn")
    candidates = [
        ir.If(
            selector,
            [_task([_copy(accum, ub, {"sub_blockid": integer(selected)})])],
            [_task([_copy(accum, ub, {"sub_blockid": integer(1 - selected)})])],
        )
    ]
    per_core = attribute({}, "tl.ascend_per_core_task", integer(1), candidates)
    result = _assign(_kernel([selector], [allocate(accum), allocate(ub), per_core], sid))
    groups = _attributes(result, "tl.ascend_per_core_task")
    assert len(groups) == 2
    for group in groups:
        assert _mask(group) == 2
        (candidate,) = _tasks(group)
        _, hints = decoded(calls(candidate, "tl.tileop.ascend_copy")[0])
        assert int(hints["sub_blockid"]) == int(_metadata(group)["subcore"])


def test_assign_core_rejects_a_second_assignment():
    source, ub = tensor("source", [16], scope="global"), tensor("ub", [16], scope="shared.dyn")
    result = _assign(_kernel([source], [allocate(ub), _copy(source, ub)]))
    with pytest.raises((ValueError, RuntimeError), match="must run once"):
        run_pass("AssignCore", result)


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_public_mixed_kernel_hints_survive_both_launch_forms(materialized, count):
    from tilelang.ascend import language as T
    from test_native_ascend_dual_copy import _materialize

    @T.prim_func
    def before(A: T.Tensor((16,), "float32")):
        with T.MixedKernel(1, sids=count) as (block, sid):
            temp = T.alloc_shared((16,), "float32")
            if sid == count - 1:
                T.copy(A, temp)

    value = before.with_attr("target", tvm_tirx.target.Target("ascend"))
    if materialized:
        value = _materialize(value)
    result = _assign(value)
    (task,) = _tasks(result)
    assert _mask(task) == (1 if count == 1 else 4)
    assert int(_metadata(task)["subcore"]) == count - 1


def test_projected_native_region_body_parameters_receive_fresh_definitions():
    sid, lane = ir.Var("sid", "int32"), ir.Var("lane", "int32")
    output = tensor("output", [2, 4], "int32", "global")
    inner = _launch([ir.TensorStore(output, [sid, lane], sid + lane)], tag="threadIdx.x", extent=4, var=lane)
    task = _task([inner])
    result = _assign(_kernel([output], [ir.If(sid < 2, [task], None)], sid))
    regions = [
        node for node in objects(result, ir.RegionStmt) if node.op.name == "tirx.launch_thread" and node.args[0].value == "threadIdx.x"
    ]
    assert len(regions) == 2
    assert not regions[0].body_params[0].same_as(regions[1].body_params[0])
    for native_region in regions:
        (store,) = objects(native_region.body, ir.TensorStore)
        assert store.indices[1].same_as(native_region.body_params[0])


@pytest.mark.parametrize("wrapper", ["if", "assume"])
def test_uniform_mutable_condition_is_evaluated_once_during_projection(wrapper):
    sid = ir.Var("sid", "int32")
    flag = tensor("flag", [1], "int32", "global")
    accum = tensor("accum", [16, 16], scope="shared.l0c")
    ub = tensor("ub", [16, 16], scope="shared.dyn")
    condition = ir.TensorLoad(flag, [integer(0)]) > 0
    children = [_copy(accum, ub)]
    scope = ir.If(condition, children, None) if wrapper == "if" else attribute(condition, "tilelang_assume", integer(1), children)
    result = _assign(_kernel([flag], [allocate(accum), allocate(ub), scope], sid))
    loads = [node for node in objects(result, ir.TensorLoad) if node.source.same_as(flag)]
    assert len(loads) == 1
    assert len(_tasks(result)) == 2


@pytest.mark.parametrize("varying", [False, True])
def test_hardware_dual_destination_copy_retains_uniform_storage_and_control(varying):
    sid = ir.Var("sid", "int32")
    accum = tensor("accum", [16, 16], scope="shared.l0c")
    ub = tensor("ub", [16, 16], scope="shared.dyn")
    copy = _copy(accum, ub, {"dual_dst_ctl": integer(1), "unit_flag_ctrl": integer(3)})
    if varying:
        copy = ir.If(sid == 0, [copy], None)
    value = _kernel([], [allocate(accum), allocate(ub), copy], sid)
    if varying:
        with pytest.raises((ValueError, RuntimeError), match="Hardware L0C dual_copy.*uniform"):
            _assign(value)
    else:
        result = _assign(value)
        (operation,) = calls(result, "tl.tileop.ascend_copy")
        operands, hints = decoded(operation)
        assert operands[1].source.same_as(ub)
        assert int(hints["dual_dst_ctl"]) == 1 and int(hints["unit_flag_ctrl"]) == 3


@pytest.mark.parametrize("operation", ["padding", "atomic"])
def test_projected_special_register_configuration_retains_cube_eligibility(operation):
    from tilelang.ascend import language as T

    source = tensor("source", [16], scope="global")
    ub = tensor("ub", [16], scope="shared.dyn")
    sid = ir.Var("sid", "int32")
    setters = (
        [T.ascend_set_copy_pad_value(value, dtype="float32") for value in (1.0, 2.0)]
        if operation == "padding"
        else [T.set_atomic("add", "float32"), T.set_atomic_none()]
    )
    body = [ir.If(sid == index, [ir.Evaluate(setter)], None) for index, setter in enumerate(setters)]
    result = _assign(_kernel([source], [allocate(ub), *body, _copy(source, ub)], sid))
    names = {setter.op.name for setter in setters}
    tasks = [task for task in _tasks(result) if any(call.op.name in names for call in objects(task, ir.Call))]
    assert {int(_metadata(task)["subcore"]): _mask(task) for task in tasks} == {0: 3, 1: 6}


@pytest.mark.parametrize("routing,count", [("implicit", 2), ("local", 2), ("sid", 2), ("explicit", 2), ("parameter", 2), ("implicit", 1)])
def test_per_core_explicit_sync_cannot_follow_repeated_fixpipe_routing(routing, count):
    sid, block = ir.Var("sid", "int32"), ir.Var("block", "int32")
    accum = tensor("accum", [16, 16], scope="shared.l0c")
    ub = tensor("ub", [16, 16], scope="shared.dyn")
    candidates = []
    for index in (0, 1):
        route = (block // 2 + index) % 2 if routing == "parameter" else integer(index)
        prefix = []
        if routing == "sid":
            route = sid
        elif routing == "local":
            variable = ir.Var("route", "int32")
            prefix = [ir.Bind(variable, (block // 2 + index) % 2)]
            route = variable
        hints = {} if routing == "implicit" else {"sub_blockid": route}
        candidates.append(ir.If(block % 2 == index, [_task([*prefix, _copy(accum, ub, hints)])], None))
    group = attribute({}, "tl.ascend_per_core_task", integer(1), [candidates[0], _flag("PIPE_FIX"), candidates[1]])
    value = _kernel([block], [allocate(accum), allocate(ub), group], sid, count)
    if count == 2 and routing in {"implicit", "local", "sid"}:
        with pytest.raises((ValueError, RuntimeError), match="PerCoreTask.*explicit synchronization.*Fixpipe"):
            _assign(value)
    else:
        result = _assign(value)
        assert calls(result, "tl.ascend_cross_core_set_flag")
        assert all(_mask(task) == 2 for task in _attributes(result, "tl.ascend_per_core_task"))
