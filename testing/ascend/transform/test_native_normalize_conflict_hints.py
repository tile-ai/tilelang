"""Conflict-hint normalization retains the definitions of each source site.

These are producer/IR contracts. The context-qualified scheduling consumer is
tested separately in its native dependency-analysis suite.
"""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from testing.ascend._ir import launch_thread, scope_annotations


DIRECT = "conflict_hint"
SCOPED = "tl.scoped_conflict_hints"
ROOT = "tl.root_conflict_hints"
SCOPE_ID = "tl.conflict_scope_id"


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, shape=(8,)):
    return ir.Var(name, tirx.TensorType(shape, "int32"))


def region(value, start=0, extent=1):
    return ir.TensorRegion(value, [ir.Range.from_min_extent(start, extent)], ir.TensorRegionType())


def marker(a, b=None, *, level=-2, cross=-1, group="", polarity=False, loc=None):
    return ir.Evaluate(
        ir.Call(
            "tl.conflict_hint",
            [a, a if b is None else b, integer(level), integer(cross), ir.StringImm(group), integer(polarity, "bool")],
            ty="int32",
        ),
        loc=ir.UnknownLoc() if loc is None else loc,
    )


def attribute(key, value, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def kernel(body):
    return ir.SeqStmt(body)


def loop(value, body):
    return ir.For(value, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, body)


def declare(value, storage):
    return ir.Bind(value, ir.Call("tirx.decl_tensor", [ir.Call("tirx.tensor_data_ptr", [storage])], ty_args=[value.ty]))


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def fields(call):
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"])


def carriers(value):
    return [call for call in objects(value, ir.Call) if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation"]


def records(value):
    return [entry for call in carriers(value) for entry in fields(call).get(SCOPED, [])]


def apply(params, body, already_root=False):
    before = tirx.Function(params, body if already_root else [kernel(body)]).with_attr("target", tvm_tirx.target.Target("ascend"))
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    transform = tvm_ffi.get_global_func("tl.transform.NormalizeConflictHints")()
    result = transform(ir.IRModule({"main": before}))["main"]
    assert ir.save_json(before) == saved
    assert tirx.analysis.check_well_formed(result)
    assert not any(call.op.name == "tl.conflict_hint" for call in objects(result, ir.Call) if isinstance(call.op, ir.Op))
    for call in objects(result, ir.Call):
        call.validate()
    ir.assert_structural_equal(transform(ir.IRModule({"main": result}))["main"], result)
    loaded = ir.load_json(ir.save_json(result))
    assert tirx.analysis.check_well_formed(loaded)
    ir.assert_structural_equal(loaded, result)
    renewed = tirx.renew_def(result)
    assert tirx.analysis.check_well_formed(renewed)
    ir.assert_structural_equal(renewed, result)
    return result


@pytest.mark.parametrize("polarity", [False, True])
@pytest.mark.parametrize("cross", [-1, 0, 1])
def test_root_direct_pair_retains_native_pointer_identity(polarity, cross):
    a, b = tensor("A"), tensor("B")
    result = apply([a, b], [marker(ir.Call("tirx.tensor_data_ptr", [a]), b, cross=cross, polarity=polarity)])
    (call,) = carriers(result)
    (entry,) = fields(call)[ROOT]
    assert entry[0].same_as(a) and entry[1].same_as(b)
    assert int(entry[2]) == cross and bool(entry[3]) == polarity
    assert not records(result)


def test_direct_loop_hint_is_after_its_local_scalar_and_typed_view_definitions():
    a, b = tensor("A"), tensor("B")
    i, size = ir.Var("i", "int32"), ir.Var("size", "int32")
    view = tensor("view", [size])
    body = [ir.Bind(size, i + 1), declare(view, a), marker(region(view, i), region(b, i), level=0)]
    result = apply([a, b], [loop(i, body)])
    (control,) = objects(result, ir.For)
    assert isinstance(control.body.seq[0], ir.Bind) and control.body.seq[0].var.same_as(size)
    assert isinstance(control.body.seq[1], ir.Bind) and control.body.seq[1].var.same_as(view)
    (entry,) = fields(carriers(control)[0])[DIRECT]
    assert entry[0].source.same_as(view)
    assert entry[0].source.ty.shape[0].same_as(size)
    assert entry[0].region[0].begin.same_as(i)
    assert DIRECT not in control.annotations and SCOPE_ID not in control.annotations
    assert not records(result)


@pytest.mark.parametrize("level,expected_direct,expected_scoped", [(-2, 1, 2), (-1, 0, 1), (0, 0, 1), (1, 1, 0)])
def test_selected_ancestor_scopes_use_static_ids_without_escaping_inner_var(level, expected_direct, expected_scoped):
    a, b = tensor("A"), tensor("B")
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    result = apply([a, b], [loop(i, [loop(j, [marker(region(a, j), region(b, j), level=level)])])])
    outer, inner = objects(result, ir.For)
    (call,) = carriers(inner)
    assert len(fields(call).get(DIRECT, [])) == expected_direct
    scoped = fields(call).get(SCOPED, [])
    assert sum(len(entry["target_scopes"]) for entry in scoped) == expected_scoped
    for entry in scoped:
        assert entry["role"] == "pair"
        assert entry["operands"][0].region[0].begin.same_as(j)
        targets = list(entry["target_scopes"])
        if level in {-2, 0}:
            assert outer.annotations[SCOPE_ID] in targets
        if level in {-2, -1}:
            assert "root" in targets
    assert DIRECT not in outer.annotations
    assert ROOT not in scope_annotations(result.body)


@pytest.mark.parametrize("branch", ["then", "else"])
def test_conditional_marker_retains_its_branch_source_context(branch):
    a, b, flag, i = tensor("A"), tensor("B"), ir.Var("flag", "bool"), ir.Var("i", "int32")
    operation = marker(region(a, i), region(b, i), level=0)
    condition = ir.If(flag, [operation] if branch == "then" else [], [operation] if branch == "else" else None)
    result = apply([a, b, flag], [loop(i, [condition])])
    (out_if,) = objects(result, ir.If)
    source = out_if.then_case if branch == "then" else out_if.else_case
    (call,) = carriers(source)
    assert DIRECT not in fields(call)
    (record,) = fields(call)[SCOPED]
    assert record["role"] == "pair"
    assert len(record["target_scopes"]) == 1
    assert out_if.condition.same_as(flag)


@pytest.mark.parametrize("common_outer", [False, True])
@pytest.mark.parametrize("polarity", [False, True])
def test_grouped_sibling_views_and_shape_locals_remain_at_their_individual_sites(common_outer, polarity):
    a, b = tensor("A"), tensor("B")
    k, i, j = (ir.Var(name, "int32") for name in ("k", "i", "j"))
    left_n, right_n = ir.Var("size", "int32"), ir.Var("size", "int32")
    left, right = tensor("view", [left_n]), tensor("view", [right_n])
    level = 0 if common_outer else -1
    first = loop(
        i, [ir.Bind(left_n, i + 1), declare(left, a), marker(region(left, i), group="pair", level=level, cross=1, polarity=polarity)]
    )
    second = loop(
        j, [ir.Bind(right_n, j + 1), declare(right, b), marker(region(right, j), group="pair", level=level, cross=1, polarity=polarity)]
    )
    result = apply([a, b], [loop(k, [first, second])] if common_outer else [first, second])
    lhs, rhs = records(result)
    assert lhs["hint_id"] == rhs["hint_id"]
    assert [lhs["role"], rhs["role"]] == ["lhs", "rhs"]
    assert lhs["operands"][0].source.same_as(left) and rhs["operands"][0].source.same_as(right)
    assert lhs["operands"][0].source.ty.shape[0].same_as(left_n)
    assert rhs["operands"][0].source.ty.shape[0].same_as(right_n)
    assert int(lhs["cross"]) == int(rhs["cross"]) == 1
    assert bool(lhs["polarity"]) == bool(rhs["polarity"]) == polarity
    assert list(lhs["target_scopes"]) == list(rhs["target_scopes"])
    controls = objects(result, ir.For)
    if common_outer:
        assert list(lhs["target_scopes"]) == [controls[0].annotations[SCOPE_ID]]
        assert SCOPE_ID not in controls[1].annotations and SCOPE_ID not in controls[2].annotations
    else:
        assert list(lhs["target_scopes"]) == ["root"]
    assert all(DIRECT not in fields(call) and ROOT not in fields(call) for call in carriers(result))


def test_groups_keep_each_conditional_guard_including_disjoint_arms():
    a, b, flag = tensor("A"), tensor("B"), ir.Var("flag", "bool")
    branch = ir.If(flag, [marker(a, group="g", level=-1)], [marker(b, group="g", level=-1)])
    result = apply([a, b, flag], [branch])
    (out_if,) = objects(result, ir.If)
    assert records(out_if.then_case)[0]["role"] == "lhs"
    assert records(out_if.else_case)[0]["role"] == "rhs"
    assert records(out_if.then_case)[0]["hint_id"] == records(out_if.else_case)[0]["hint_id"]
    assert not any(ROOT in fields(call) for call in carriers(result))


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("only-once", "only once"),
        ("three", "more than twice"),
        ("level", "mismatched level"),
        ("cross", "mismatched cross"),
        ("out-of-range", "out of range"),
    ],
)
def test_invalid_group_contracts_reject_without_mutating_input(case, diagnostic):
    a, b, i = tensor("A"), tensor("B"), ir.Var("i", "int32")
    hints = [marker(a, group="g", level=-1)]
    if case != "only-once":
        hints.append(marker(b, group="g", level=0 if case == "level" else -1, cross=0 if case == "cross" else -1))
    if case == "three":
        hints.append(marker(a, group="g", level=-1))
    if case == "out-of-range":
        hints = [loop(i, [marker(a, group="g", level=0)]), marker(b, group="g", level=0)]
    before = tirx.Function([a, b], [kernel(hints)]).with_attr("target", tvm_tirx.target.Target("ascend"))
    saved = ir.save_json(before)
    with pytest.raises(ValueError, match=diagnostic):
        tvm_ffi.get_global_func("tl.transform.NormalizeConflictHints")()(ir.IRModule({"main": before}))
    assert saved == ir.save_json(before)


@pytest.mark.parametrize("level,cross,diagnostic", [(-3, -1, "level must"), (-1, 2, "cross mode"), (0, -1, "out of range")])
def test_invalid_level_and_cross_are_diagnosed(level, cross, diagnostic):
    a = tensor("A")
    with pytest.raises(ValueError, match=diagnostic):
        apply([a], [marker(a, level=level, cross=cross)])


def test_polarity_is_a_separate_group_domain_and_ids_do_not_collide():
    a, b = tensor("A"), tensor("B")
    hints = [marker(value, group="same", polarity=polarity, level=-1) for polarity in (False, True) for value in (a, b)]
    result = apply([a, b], hints)
    output = records(result)
    assert output[0]["hint_id"] == output[1]["hint_id"]
    assert output[2]["hint_id"] == output[3]["hint_id"]
    assert output[0]["hint_id"] != output[2]["hint_id"]


def test_same_group_spelling_is_independent_in_distinct_kernels():
    a, b = tensor("A"), tensor("B")
    roots = [
        kernel([marker(a, group="g", level=-1), marker(b, group="g", level=-1)]),
        kernel([marker(b, group="g", level=-1), marker(a, group="g", level=-1)]),
    ]
    roots = [launch_thread(ir.Var(f"block{index}", "int32"), "blockIdx.x", 1, body) for index, body in enumerate(roots)]
    result = apply([a, b], roots, already_root=True)
    first, second = result.body.seq
    assert len(records(first)) == len(records(second)) == 2
    assert records(first)[0]["hint_id"] == records(first)[1]["hint_id"]
    assert records(second)[0]["hint_id"] == records(second)[1]["hint_id"]


def test_multiple_direct_markers_keep_source_order_and_repeated_call_occurrences():
    a, b, i = tensor("A"), tensor("B"), ir.Var("i", "int32")
    shared = marker(a, b, level=0, polarity=True)
    result = apply([a, b], [loop(i, [marker(a, b, level=0), shared, shared])])
    hints = [fields(call)[DIRECT][0] for call in carriers(result)]
    assert [bool(entry[3]) for entry in hints] == [False, True, True]


def test_no_kernel_is_an_unchanged_function():
    a = tensor("A")
    before = tirx.Function([a], [ir.TensorStore(a, [0], integer(1))])
    transform = tvm_ffi.get_global_func("tl.transform.NormalizeConflictHints")()
    ir.assert_structural_equal(transform(ir.IRModule({"main": before}))["main"], before)


def jump_statement(kind):
    return {"break": ir.Break, "continue": ir.Continue, "return": lambda: ir.Return(ir.Tuple([]))}[kind]()


def jump_task(kind):
    # MaterializeScheduleUnits requires explicit task boundaries for jumps.
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": {}, "key": "tl.ascend_task", "value": integer(1)})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, [jump_statement(kind)])


@pytest.mark.parametrize("jump", ["break", "continue", "return"])
@pytest.mark.parametrize("branch", ["then", "else"])
@pytest.mark.parametrize("polarity", [False, True])
def test_only_the_prefix_before_a_conditional_exit_has_direct_scope(jump, branch, polarity):
    a, b, flag, i = tensor("A"), tensor("B"), ir.Var("flag", "bool"), ir.Var("i", "int32")
    exit_statement = jump_statement(jump)
    condition = ir.If(flag, [exit_statement] if branch == "then" else [], [exit_statement] if branch == "else" else None)
    shared_marker = marker(a, b, level=0, polarity=polarity)
    result = apply([a, b, flag], [loop(i, [shared_marker, condition, shared_marker])])
    first, last = carriers(result)
    assert DIRECT in fields(first) and SCOPED not in fields(first)
    assert DIRECT not in fields(last)
    (record,) = fields(last)[SCOPED]
    assert bool(record["polarity"]) == polarity
    (owner,) = objects(result, ir.For)
    assert list(record["target_scopes"]) == [owner.annotations[SCOPE_ID]]


@pytest.mark.parametrize("polarity", [False, True])
def test_a_conditional_root_return_does_not_create_a_global_direct_hint(polarity):
    a, b, flag = tensor("A"), tensor("B"), ir.Var("flag", "bool")
    result = apply([a, b, flag], [ir.If(flag, [jump_statement("return")], None), marker(a, b, level=-1, polarity=polarity)])
    (call,) = carriers(result)
    assert ROOT not in fields(call)
    (record,) = fields(call)[SCOPED]
    assert list(record["target_scopes"]) == ["root"]


@pytest.mark.parametrize("jump", ["break", "continue", "return"])
@pytest.mark.parametrize("polarity", [False, True])
def test_nested_loop_owned_jumps_are_distinct_from_return(jump, polarity):
    a, b, flag = tensor("A"), tensor("B"), ir.Var("flag", "bool")
    outer, inner = ir.Var("i", "int32"), ir.Var("i", "int32")
    nested = loop(inner, [ir.If(flag, [jump_statement(jump)], None)])
    result = apply([a, b, flag], [loop(outer, [nested, marker(a, b, level=0, polarity=polarity)])])
    (call,) = carriers(result)
    assert (DIRECT in fields(call)) == (jump != "return")
    assert (SCOPED in fields(call)) == (jump == "return")


@pytest.mark.parametrize(
    "dtype,begin,end,step,direct",
    [("int32", 0, 4, 1, True), ("int32", 5, 2, -1, True), ("int8", 126, 127, 2, False), ("int32", 0, 2, -1, False)],
)
@pytest.mark.parametrize("polarity", [False, True])
def test_following_hint_requires_the_nested_loop_to_finish(dtype, begin, end, step, direct, polarity):
    a, b, outer, inner = tensor("A"), tensor("B"), ir.Var("outer", "int32"), ir.Var("inner", dtype)
    nested = ir.For(inner, integer(begin, dtype), integer(end, dtype), integer(step, dtype), ir.ForKind.DEFAULT, [ir.Evaluate(inner)])
    result = apply([a, b], [loop(outer, [nested, marker(a, b, level=0, polarity=polarity)])])
    (call,) = carriers(result)
    assert (DIRECT in fields(call)) == direct
    assert (SCOPED in fields(call)) != direct


@pytest.mark.parametrize("polarity", [False, True])
def test_early_exit_qualifies_every_all_scope_projection(polarity):
    a, b, flag, i = tensor("A"), tensor("B"), ir.Var("flag", "bool"), ir.Var("i", "int32")
    body = [attribute("tl.loop.annotations", {}, [ir.If(flag, [ir.Continue()], None)]), marker(a, b, level=-2, polarity=polarity)]
    result = apply([a, b, flag], [loop(i, body)])
    (call,) = carriers(result)
    assert DIRECT not in fields(call) and ROOT not in fields(call)
    (record,) = fields(call)[SCOPED]
    (owner,) = objects(result, ir.For)
    assert set(record["target_scopes"]) == {"root", str(owner.annotations[SCOPE_ID])}


def has_dependency(normalized, writer, reader):
    materialize = tvm_ffi.get_global_func("tl.transform.MaterializeScheduleUnits")()
    scheduled = materialize(ir.IRModule({"main": normalized}))["main"]
    assert tirx.analysis.check_well_formed(scheduled)
    root = scheduled.body
    result = tvm_ffi.get_global_func("tl.testing.AnalyzeAscendDependencies")(root, {}, {})

    def task_id(anchor):
        return next(
            int(task["id"])
            for task in result["tasks"]
            if not task["header"] and any(obj.same_as(anchor) for obj in objects(task["statement"], type(anchor)))
        )

    source, target = task_id(writer), task_id(reader)
    return any(
        int(pair[0]) == source and int(pair[1]) == target and int(dependency["distance"]) == 0
        for scope in result["scopes"]
        for dependency in scope["dependencies"]
        for pair in dependency["pairs"]
    )


@pytest.mark.parametrize("jump", ["break", "continue", "return"])
@pytest.mark.parametrize("polarity", [False, True])
def test_real_dependency_consumer_does_not_apply_a_suffix_hint_to_prior_work(jump, polarity):
    a, b, flag, i = tensor("A"), tensor("B"), ir.Var("flag", "bool"), ir.Var("i", "int32")
    writer = ir.TensorStore(a, [0], integer(1))
    reader = ir.TensorLoad(b, [0])
    prefix = [ir.Bind(var, ir.Call("tirx.alloc_tensor", [], ty_args=[var.ty])) for var in (a, b)] if polarity else []
    params = [flag] if polarity else [a, b, flag]
    body = [writer, ir.Evaluate(reader), ir.If(flag, [jump_task(jump)], None), marker(a, b, level=0, polarity=polarity)]
    normalized = apply(params, [*prefix, loop(i, body)])
    assert has_dependency(normalized, writer, reader) == (not polarity)


@pytest.mark.parametrize("kind", ["false-branch", "empty-for", "false-while"])
@pytest.mark.parametrize("polarity", [False, True])
def test_unreachable_exits_keep_the_following_direct_domain(kind, polarity):
    a, b, i = tensor("A"), tensor("B"), ir.Var("i", "int32")
    exit_statement = jump_statement("return")
    control = {
        "false-branch": lambda: ir.If(integer(False, "bool"), [exit_statement], None),
        "empty-for": lambda: ir.For(i, integer(5), integer(2), integer(-1), ir.ForKind.DEFAULT, [exit_statement]),
        "false-while": lambda: ir.While(integer(False, "bool"), [exit_statement]),
    }[kind]()
    result = apply([a, b], [control, marker(a, b, level=-1, polarity=polarity)])
    (call,) = carriers(result)
    assert ROOT in fields(call) and SCOPED not in fields(call)


@pytest.mark.parametrize("polarity", [False, True])
def test_an_inner_loop_has_its_own_direct_domain_after_an_outer_exit(polarity):
    a, b, flag = tensor("A"), tensor("B"), ir.Var("flag", "bool")
    outer, inner = ir.Var("i", "int32"), ir.Var("i", "int32")
    body = [ir.If(flag, [ir.Continue()], None), loop(inner, [marker(a, b, level=-2, polarity=polarity)])]
    result = apply([a, b, flag], [loop(outer, body)])
    (call,) = carriers(result)
    (direct,) = fields(call)[DIRECT]
    assert bool(direct[3]) == polarity
    (record,) = fields(call)[SCOPED]
    first, last = objects(result, ir.For)
    assert set(record["target_scopes"]) == {"root", str(first.annotations[SCOPE_ID])}
    assert SCOPE_ID not in last.annotations


@pytest.mark.parametrize("polarity", [False, True])
def test_group_halves_on_opposite_sides_of_an_exit_keep_separate_execution_domains(polarity):
    a, b, flag, i = tensor("A"), tensor("B"), ir.Var("flag", "bool"), ir.Var("i", "int32")
    writer, reader = ir.TensorStore(a, [0], integer(1)), ir.TensorLoad(b, [0])
    prefix = [ir.Bind(var, ir.Call("tirx.alloc_tensor", [], ty_args=[var.ty])) for var in (a, b)] if polarity else []
    params = [flag] if polarity else [a, b, flag]
    body = [
        writer,
        ir.Evaluate(reader),
        marker(a, group="across-exit", level=0, polarity=polarity),
        ir.If(flag, [jump_task("continue")], None),
        marker(b, group="across-exit", level=0, polarity=polarity),
    ]
    normalized = apply(params, [*prefix, loop(i, body)])
    left, right = records(normalized)
    assert left["hint_id"] == right["hint_id"]
    assert [left["role"], right["role"]] == ["lhs", "rhs"]
    assert all(DIRECT not in fields(call) and ROOT not in fields(call) for call in carriers(normalized))
    assert has_dependency(normalized, writer, reader) == (not polarity)
