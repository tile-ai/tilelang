"""Conflict assertions retain each source execution and definition domain."""

import pytest
from tvm_tirx import ir, tirx

from test_native_ascend_core_resolution import _kernel, _loop, _unit
from test_native_ascend_dependencies import _api, _declare, _ids, _inspect, _pairs, _root
from test_native_ascend_schedule_units import _check as _materialize
from test_native_ascend_schedule_units import _fields, _kernel as _raw_kernel
from test_native_reducer_contracts import allocate, annotation, attribute, function, integer, objects, run_pass, tensor


SCOPED = "tl.scoped_conflict_hints"
ROOT = "tl.root_conflict_hints"
SCOPE_ID = "tl.conflict_scope_id"


def _record(operands, *, role="pair", hint="h0", targets=("root",), cross=-1, polarity=False, **changes):
    fields = {
        "schema_version": 1,
        "hint_id": hint,
        "role": role,
        "operands": operands,
        "target_scopes": list(targets),
        "cross": integer(cross) if isinstance(cross, int) else cross,
        "polarity": integer(polarity, "bool") if isinstance(polarity, (int, bool)) else polarity,
    }
    fields.update(changes)
    return annotation({SCOPED: [fields]})


def _direct(a, b, *, polarity=False, cross=-1):
    return annotation({ROOT: [[a, b, integer(cross), integer(polarity, "bool")]]})


def _marker(a, b=None, *, group="", level=-1, cross=-1, polarity=False):
    return ir.Evaluate(
        ir.Call(
            "tl.conflict_hint",
            [a, a if b is None else b, integer(level), integer(cross), ir.StringImm(group), integer(polarity, "bool")],
            ty="int32",
        )
    )


def _region(storage, coordinate):
    return ir.TensorRegion(storage, [ir.Range.from_min_extent(coordinate, integer(1))], ir.TensorRegionType())


def _tagged_loop(index, body, tag="l0"):
    return _unit(ir.For(index, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, body, annotations={SCOPE_ID: tag}))


def _writes(a, b):
    writer = ir.TensorStore(a, [integer(0)], integer(1))
    reader = ir.TensorLoad(b, [integer(0)])
    return writer, reader


def _has_edge(result, writer, reader, *, clock=None, root_only=True, distance=0):
    source = _ids(result, writer, header=False)[0]
    target = _ids(result, reader, header=False)[0]
    return any(row[:3] == (source, target, distance) for row in _pairs(result, clock=clock, root_only=root_only))


def _carriers(value, key):
    return [
        _fields(call)[key]
        for call in objects(value, ir.Call)
        if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation" and key in _fields(call)
    ]


def _well_formed(value):
    assert tirx.analysis.check_well_formed(value)
    assert not tirx.analysis.undefined_vars(value)
    loaded = ir.load_json(ir.save_json(value))
    assert tirx.analysis.check_well_formed(loaded)
    ir.assert_structural_equal(loaded, value)
    renewed = tirx.renew_def(value)
    assert tirx.analysis.check_well_formed(renewed)
    assert not tirx.analysis.undefined_vars(renewed)
    ir.assert_structural_equal(renewed, value)
    return value


@pytest.mark.parametrize("polarity", [False, True])
@pytest.mark.parametrize("domain", ["same", "opposite", "unguarded", "false"])
def test_each_scoped_pair_requires_its_source_guard(polarity, domain):
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    flag = ir.Var("flag", "bool")
    writer, reader = _writes(a, b)
    site_guard = integer(False, "bool") if domain == "false" else flag
    marker = ir.If(site_guard, [_record([a, b], polarity=polarity)], None)
    work = [_unit(writer), _unit(ir.Evaluate(reader))]
    if domain == "same":
        work = [ir.If(flag, work, None)]
    elif domain == "opposite":
        work = [ir.If(ir.prim.Not(flag), work, None)]
    prefix = [_unit(allocate(a)), _unit(allocate(b))] if polarity else []
    value = _kernel([flag] if polarity else [a, b, flag], [*prefix, marker, *work])
    result = _inspect(value)
    assert _has_edge(result, writer, reader, root_only=False) == (polarity if domain == "same" else not polarity)


@pytest.mark.parametrize("polarity", [False, True])
@pytest.mark.parametrize("query_guard", ["none", "left", "right"])
def test_group_source_guards_are_never_merged_into_a_vacuous_proof(polarity, query_guard):
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    flag = ir.Var("flag", "bool")
    writer, reader = _writes(a, b)
    hints = [
        ir.If(flag, [_record([a], role="lhs", polarity=polarity)], None),
        ir.If(ir.prim.Not(flag), [_record([b], role="rhs", polarity=polarity)], None),
    ]
    work = [_unit(writer), _unit(ir.Evaluate(reader))]
    if query_guard != "none":
        work = [ir.If(flag if query_guard == "left" else ir.prim.Not(flag), work, None)]
    prefix = [_unit(allocate(a)), _unit(allocate(b))] if polarity else []
    value = _kernel([flag] if polarity else [a, b, flag], [*prefix, *hints, *work])
    assert _has_edge(_inspect(value), writer, reader, root_only=False) == (not polarity)


@pytest.mark.parametrize("direct_forced", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_forced_conflict_wins_between_direct_and_scoped_records(direct_forced, reverse):
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    writer, reader = _writes(a, b)
    hints = [_direct(a, b, polarity=direct_forced), _record([a, b], polarity=not direct_forced)]
    if reverse:
        hints.reverse()
    value = _kernel([a, b], [*hints, _unit(writer), _unit(ir.Evaluate(reader))])
    assert _has_edge(_inspect(value), writer, reader)


@pytest.mark.parametrize("ancestor", [False, True])
@pytest.mark.parametrize("local_definition", [False, True])
def test_grouped_sibling_regions_keep_their_own_iteration_definitions(ancestor, local_definition):
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    left, right, outer = [ir.Var(name, "int32") for name in ("i", "j", "outer")]
    lvalue = ir.Var("offset", "int32") if local_definition else left
    rvalue = ir.Var("offset", "int32") if local_definition else right
    writer = ir.TensorStore(a, [lvalue], integer(1))
    reader = ir.TensorLoad(b, [rvalue])
    targets = ("l0",) if ancestor else ("root",)
    lprefix = [_unit(ir.Bind(lvalue, left + 1))] if local_definition else []
    rprefix = [_unit(ir.Bind(rvalue, right + 1))] if local_definition else []

    def program(hints):
        lbody = [*lprefix, _unit(writer)]
        rbody = [*rprefix, _unit(ir.Evaluate(reader))]
        if hints:
            lbody.append(_record([_region(a, lvalue)], role="lhs", targets=targets))
            rbody.append(_record([_region(b, rvalue)], role="rhs", targets=targets))
        body = [_loop(left, lbody, end=2), _loop(right, rbody, end=2)]
        return _kernel([a, b], [_tagged_loop(outer, body)] if ancestor else body)

    options = {"clock": outer if ancestor else None, "root_only": not ancestor}
    assert _has_edge(_inspect(program(False)), writer, reader, **options)
    assert not _has_edge(_inspect(_well_formed(program(True))), writer, reader, **options)


@pytest.mark.parametrize("cross", [-1, 0, 1])
def test_scoped_hint_cross_mode_is_applied_at_the_resolved_loop(cross):
    a, b = tensor("A", [1], "int32", "global"), tensor("B", [1], "int32", "global")
    index = ir.Var("i", "int32")
    writer, reader = _writes(a, b)
    body = [_record([a, b], targets=("l0",), cross=cross), _unit(writer), _unit(ir.Evaluate(reader))]
    result = _inspect(_kernel([a, b], [_tagged_loop(index, body)]))
    assert _has_edge(result, writer, reader, clock=index, root_only=False) == (cross == 1)
    source, target = _ids(result, writer)[0], _ids(result, reader)[0]
    returned = any(row[:3] == (target, source, -1) for row in _pairs(result, clock=index))
    assert returned == (cross == 0)


@pytest.mark.parametrize("polarity", [False, True])
@pytest.mark.parametrize("same_core,same_subcore", [(True, True), (False, True), (True, False)])
def test_group_halves_need_compatible_task_core_domains(same_core, same_subcore, polarity):
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    writer, reader = _writes(a, b)
    left = _unit(ir.SeqStmt([_record([a], role="lhs", polarity=polarity), writer]), core=1, sid=0)
    right = _unit(
        ir.SeqStmt([_record([b], role="rhs", polarity=polarity), ir.Evaluate(reader)]),
        core=1 if same_core else 2,
        sid=0 if same_subcore else 1,
    )
    prefix = [_unit(allocate(a)), _unit(allocate(b))] if polarity else []
    value = _kernel([] if polarity else [a, b], [*prefix, left, right])
    assert _has_edge(_inspect(value), writer, reader) == (polarity if same_core and same_subcore else not polarity)


def test_copied_loop_scope_ids_resolve_to_the_actual_ancestor():
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    outer, left, right = [ir.Var(name, "int32") for name in ("outer", "i", "j")]
    writer = ir.TensorStore(a, [left], integer(1))
    reader = ir.TensorLoad(b, [right])
    body = [
        _loop(left, [_record([_region(a, left)], role="lhs", targets=("l0",)), _unit(writer)], end=2),
        _loop(right, [_record([_region(b, right)], role="rhs", targets=("l0",)), _unit(ir.Evaluate(reader))], end=2),
    ]
    original = _tagged_loop(outer, body)
    # Renew only this fragment's definitions; A and B remain free and shared.
    copied = tirx.renew_def(function([], [original])).body.seq[0]
    value = _well_formed(_kernel([a, b], [original, copied]))
    result = _inspect(value)
    controls = [loop for loop in objects(value, ir.For) if SCOPE_ID in loop.annotations]
    assert len(controls) == 2 and controls[0].annotations[SCOPE_ID] == controls[1].annotations[SCOPE_ID]
    for control in controls:
        (store,) = objects(control, ir.TensorStore)
        (load,) = objects(control, ir.TensorLoad)
        assert not _has_edge(result, store, load, clock=control.var, root_only=False)
    copied_read = objects(copied, ir.TensorLoad)[0]
    assert _has_edge(result, writer, copied_read)


def test_ambiguous_copied_root_group_is_not_a_third_user_declaration():
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    writer, reader = _writes(a, b)
    records = [_record([a], role="lhs"), _record([b], role="rhs")]
    copied = tirx.renew_def(function([], records)).body
    # A transformation preserves free storage operands and copies both IDs.
    # Such records are accepted but do not prove a unique root pair.
    assert len(copied.seq) == 2
    value = _kernel([a, b], [*records, *copied.seq, _unit(writer), _unit(ir.Evaluate(reader))])
    assert _has_edge(_inspect(value), writer, reader)


@pytest.mark.parametrize("polarity", [False, True])
def test_a_shared_branch_does_not_pair_halves_from_different_root_clones(polarity):
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    flag, index = ir.Var("flag", "bool"), ir.Var("offset", "int32")
    fragment = function(
        [],
        [
            ir.Bind(index, integer(0)),
            _record([_region(a, index)], role="lhs", polarity=polarity),
            _record([_region(b, index)], role="rhs", polarity=polarity),
        ],
    )
    copied = tirx.renew_def(fragment)
    definition, left, right = fragment.body.seq
    clone_definition, clone_left, clone_right = copied.body.seq
    clone_index = clone_definition.var
    # After copying and scalar specialization, a branch may contain one half
    # from each clone. No retained loop identity links this mixed pair.
    writer = ir.TensorStore(a, [index], integer(1))
    reader = ir.TensorLoad(b, [clone_index])
    body = [
        _unit(definition),
        _unit(ir.Bind(clone_index, integer(1))),
        ir.If(flag, [left, clone_right, _unit(writer), _unit(ir.Evaluate(reader))], None),
        ir.If(ir.prim.Not(flag), [clone_left, right], None),
    ]
    prefix = [_unit(allocate(a)), _unit(allocate(b))] if polarity else []
    value = _well_formed(_kernel([flag] if polarity else [a, b, flag], [*prefix, *body]))
    assert _has_edge(_inspect(value), writer, reader, root_only=False) == (not polarity)


@pytest.mark.parametrize("jump", ["break", "continue"])
def test_a_marker_after_an_unmodeled_exit_cannot_remove_prior_access_dependencies(jump):
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    flag, index = ir.Var("flag", "bool"), ir.Var("i", "int32")
    writer, reader = _writes(a, b)
    exit_statement = ir.Break() if jump == "break" else ir.Continue()
    body = [
        _unit(writer),
        _unit(ir.Evaluate(reader)),
        ir.If(flag, [_unit(exit_statement)], None),
        _record([a, b], targets=("l0",)),
    ]
    value = _kernel([a, b, flag], [_tagged_loop(index, body)])
    assert _has_edge(_inspect(value), writer, reader, clock=index, root_only=False)


@pytest.mark.parametrize("shared_capture", [False, True])
def test_mutable_conditions_require_the_same_captured_execution_domain(shared_capture):
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    state = tensor("state", [1], "int32", "global")
    condition = ir.TensorLoad(state, [integer(0)]) > 0
    writer, reader = _writes(a, b)
    work = [_unit(writer), _unit(ir.Evaluate(reader))]
    if shared_capture:
        body = [ir.If(condition, [_record([a, b]), *work], None)]
    else:
        body = [
            ir.If(condition, [_record([a, b])], None),
            _unit(ir.TensorStore(state, [integer(0)], integer(0))),
            ir.If(condition, work, None),
        ]
    assert _has_edge(_inspect(_kernel([a, b, state], body)), writer, reader, root_only=False) == (not shared_capture)


def test_same_name_condition_variables_do_not_supply_a_guard_proof():
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    left, right = ir.Var("flag", "bool"), ir.Var("flag", "bool")
    writer, reader = _writes(a, b)
    body = [ir.If(left, [_record([a, b])], None), ir.If(right, [_unit(writer), _unit(ir.Evaluate(reader))], None)]
    assert _has_edge(_inspect(_kernel([a, b, left, right], body)), writer, reader, root_only=False)


def test_a_local_tensor_alias_can_match_its_proven_storage_root():
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    alias = tensor("alias", [8], "int32", "global")
    writer, reader = _writes(a, b)
    body = [_unit(_declare(alias, a)), _record([alias, b]), _unit(writer), _unit(ir.Evaluate(reader))]
    assert not _has_edge(_inspect(_well_formed(_kernel([a, b], body))), writer, reader)


def test_root_direct_lists_survive_two_codec_rounds_at_their_definition_sites():
    extent, size = ir.Var("extent", "int32"), ir.Var("size", "int32")
    a, b = tensor("A", [size], "int32", "global"), tensor("B", [size], "int32", "global")
    body = [
        _unit(ir.Bind(size, extent + 1)),
        _unit(allocate(a)),
        _unit(allocate(b)),
        _direct(a, b),
        _unit(ir.Evaluate(size)),
        _direct(b, a, polarity=True),
    ]
    value = _well_formed(_kernel([extent], body))
    root = _root(value)
    for _ in range(2):
        row = _api("AscendRootConflictHintCodec")(root, False)
        assert len(row["hints"]) == 2
        assert [bool(hint[3]) for hint in row["hints"]] == [False, True]
        root = row["root"]
        carriers = _carriers(root, ROOT)
        assert len(carriers) == 2 and all(len(entries) == 1 for entries in carriers)
        assert ROOT not in _fields(root)["value"].get("annotations", {})
        _well_formed(function([extent], [root]))
    consumed = _api("AscendRootConflictHintCodec")(root, True)["root"]
    assert not _carriers(consumed, ROOT)
    _well_formed(function([extent], [consumed]))


@pytest.mark.parametrize("mode", ["clone", "foreign", "statement", "core", "reparent", "remove_child", "header"])
def test_an_old_index_is_rejected_before_reusing_dependency_cache(mode):
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    writer, reader = _writes(a, b)
    body = [_unit(writer), _unit(ir.Evaluate(reader)), attribute(None, "tl.loop.annotations", {}, [_record([a, b])])]
    root = _root(_kernel([a, b], body))
    with pytest.raises(ValueError, match="stale|another kernel"):
        _api("AscendScopedIndexAfterMutation")(root, mode, False)
    row = _api("AscendScopedIndexAfterMutation")(root, mode, True)
    assert int(row["before"]) == 0
    if mode in ("statement", "remove_child"):
        assert int(row["after"]) > 0


@pytest.mark.parametrize("case", ["version", "cross_type", "polarity_type", "role", "arity", "foreign_scope"])
def test_invalid_scoped_record_contracts_are_diagnosed(case):
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    changes = {
        "version": {"schema_version": 2},
        "cross_type": {"cross": integer(0, "int64")},
        "polarity_type": {"polarity": integer(0)},
        "role": {"role": "unknown"},
        "arity": {"role": "lhs"},
        "foreign_scope": {"targets": ("l0",)},
    }[case]
    value = _kernel([a, b], [_record([a, b], **changes)])
    with pytest.raises(ValueError, match="Scoped conflict"):
        _inspect(value)


def test_real_producer_groups_feed_the_native_dependency_consumer():
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    left, right = ir.Var("i", "int32"), ir.Var("j", "int32")
    writer, reader = ir.TensorStore(a, [left], integer(1)), ir.TensorLoad(b, [right])
    body = [
        ir.For(left, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [_marker(_region(a, left), group="g"), writer]),
        ir.For(right, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [_marker(_region(b, right), group="g"), ir.Evaluate(reader)]),
    ]
    source = _well_formed(_raw_kernel([a, b], body))
    saved = ir.save_json(source)
    normalized = run_pass("NormalizeConflictHints", source)
    assert ir.save_json(source) == saved
    value = _well_formed(_materialize(normalized))
    assert len(_carriers(value, SCOPED)) == 2
    assert not _has_edge(_inspect(value), writer, reader)


def test_real_third_user_group_declaration_is_still_rejected_by_the_producer():
    a, b = tensor("A", [8], "int32", "global"), tensor("B", [8], "int32", "global")
    source = _well_formed(_raw_kernel([a, b], [_marker(a, group="g"), _marker(b, group="g"), _marker(a, group="g")]))
    saved = ir.save_json(source)
    with pytest.raises(ValueError, match="more than twice"):
        run_pass("NormalizeConflictHints", source)
    assert ir.save_json(source) == saved
