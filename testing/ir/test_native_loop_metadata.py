"""Loop metadata keeps native operands at their dominating definition sites."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from test_native_ascend_core_resolution import _kernel, _loop, _unit
from test_native_ascend_dependencies import _declare, _ids, _inspect, _pairs
from test_native_ascend_ops import strided_tensor, subregion
from test_native_reducer_contracts import allocate, annotation, attribute, function, integer, tensor


def _api(name):
    return tvm_ffi.get_global_func("tl.testing." + name)


def _for(body, var=None, *, parallel=False, metadata=None, loc=None):
    return ir.For(
        var if var is not None else ir.Var("i", "int32"),
        integer(0),
        integer(4),
        integer(1),
        ir.ForKind.PARALLEL if parallel else ir.ForKind.DEFAULT,
        body,
        annotations=metadata or {},
        loc=loc or ir.UnknownLoc(),
    )


def _check(loop, params=()):
    value = function(list(params), [loop])
    assert tirx.analysis.check_well_formed(value)
    assert not tirx.analysis.undefined_vars(value)
    ir.assert_structural_equal(value, ir.load_json(ir.save_json(value)))
    return loop


def _invoke(name, loop, *args):
    before = ir.save_json(loop)
    result = _api(name)(loop, *args)
    assert ir.save_json(loop) == before
    return result


def _hint(left, right, cross=-1, conflict=False):
    return [left, right, integer(cross), integer(conflict, "bool")]


def _decode(value):
    call = value.value if isinstance(value, ir.Evaluate) else value
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"])


def _located_carrier(fields, line):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(fields)
    call_loc = ir.SourceLoc(ir.SourceName("loop_metadata.py"), line, 2, line, 8)
    stmt_loc = ir.SourceLoc(ir.SourceName("loop_metadata.py"), line, 1, line, 9)
    return ir.Evaluate(
        ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema, "keep": line}, loc=call_loc),
        loc=stmt_loc,
    )


def test_reader_concatenates_local_lists_after_legacy_wrapper_in_source_order():
    a, b, alias = (tensor(name, [8], "int32", "global") for name in ("A", "B", "alias"))
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    hints = [_hint(a, b, 0, True), _hint(alias, subregion(b, [j], [1]), -1), _hint(a, b, 1)]
    body = [
        _declare(alias, a),
        ir.Bind(j, i + 1),
        annotation({"conflict_hint": [hints[1]], "override": 2}),
        ir.TensorStore(alias, [j], integer(3)),
        annotation({"conflict_hint": [hints[2]], "override": 3}),
    ]
    loop = _check(
        _for([attribute(None, "tl.loop.annotations", {"conflict_hint": [hints[0]], "override": 1}, body)], i, metadata={"static": 7}),
        [a, b],
    )
    fields = _invoke("ReadLoopAnnotations", loop)
    assert int(fields["static"]) == 7 and int(fields["override"]) == 3
    assert len(fields["conflict_hint"]) == 3
    for actual, expected in zip(fields["conflict_hint"], hints):
        ir.assert_structural_equal(actual, tvm_ffi.convert(expected))


def test_canonical_sequence_flattening_keeps_its_definitions_in_the_enclosing_body():
    local = ir.Var("nested", "int32")
    loop = _check(_for([ir.SeqStmt([ir.Bind(local, integer(1)), annotation({"nested": local})]), annotation({"tail": 2})]))
    fields = _invoke("ReadLoopAnnotations", loop)
    assert fields["nested"].same_as(local) and int(fields["tail"]) == 2
    assert isinstance(loop.body.seq[0], ir.Bind)
    result = _check(_invoke("AppendLoopAnnotation", loop, {"tail_value": local}))
    assert _decode(result.body.seq[-1])["tail_value"].same_as(local)


@pytest.mark.parametrize("domain", ["serial", "parallel", "if", "else", "while", "assume", "task", "other_attribute"])
def test_reader_does_not_import_directives_from_another_execution_domain(domain):
    flag = ir.Var("flag", "bool")
    inner = [annotation({"child_only": 1})]
    if domain in {"serial", "parallel"}:
        child = _for(inner, parallel=domain == "parallel")
    elif domain in {"if", "else"}:
        child = ir.If(flag, inner if domain == "if" else [], inner if domain == "else" else None)
    elif domain == "while":
        child = ir.While(flag, inner)
    elif domain == "assume":
        child = attribute(flag, "tilelang_assume", integer(1), inner)
    elif domain == "task":
        child = _unit(annotation({"child_only": 1}))
    else:
        child = attribute(None, "unrelated_scope", integer(1), inner)
    loop = _check(_for([annotation({"parent_only": 2}), child]), [flag])
    fields = _invoke("ReadLoopAnnotations", loop)
    assert set(fields) == {"parent_only"}
    seen = []

    def rewrite(value):
        seen.append(set(value))
        return value

    assert _invoke("RewriteLoopAnnotationCarriers", loop, rewrite).same_as(loop)
    assert seen == [{"parent_only"}]


@pytest.mark.parametrize("wrapped", [False, True])
def test_sole_child_parallel_nest_remains_one_logical_loop(wrapped):
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    body = [annotation({"inner": i + j})]
    if wrapped:
        body = [attribute(None, "tl.loop.annotations", {"legacy": i}, body)]
    loop = _check(_for([_for(body, j, parallel=True)], i, parallel=True, metadata={"static": 4}))
    fields = _invoke("ReadLoopAnnotations", loop)
    assert "inner" in fields and int(fields["static"]) == 4
    result = _check(_invoke("AppendLoopAnnotation", loop, {"appended": i + j}))
    assert "appended" in _invoke("ReadLoopAnnotations", result)
    assert result.body.seq[0].var.same_as(j)


@pytest.mark.parametrize("outer_parallel,inner_parallel", [(False, False), (False, True), (True, False)])
def test_sole_child_convention_does_not_cross_a_serial_loop(outer_parallel, inner_parallel):
    loop = _check(_for([_for([annotation({"child": 1})], parallel=inner_parallel)], parallel=outer_parallel))
    assert not _invoke("ReadLoopAnnotations", loop)


def test_rewriter_preserves_each_carriers_position_attributes_and_locations():
    i, local, inner_local = (ir.Var(name, "int32") for name in ("i", "local", "inner_local"))
    first = _located_carrier({"tag": "first"}, 10)
    second = _located_carrier({"tag": "second", "value": local}, 20)
    third = _located_carrier({"tag": "third", "value": inner_local}, 30)
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(
        {"node": None, "key": "tl.loop.annotations", "value": {"tag": "wrapper", "value": local}}
    )
    wrapper_loc = ir.SourceLoc(ir.SourceName("loop_metadata.py"), 25, 1, 35, 1)
    wrapper = ir.RegionStmt(
        "tl.attr",
        [payload],
        [],
        {"schema_version": 1, "annotation_schema": schema, "keep": 25},
        [ir.Bind(inner_local, local + 1), third],
        loc=wrapper_loc,
    )
    child = _for([annotation({"tag": "child"})])
    body = ir.SeqStmt([first, ir.Bind(local, i + 1), second, wrapper, child], loc=wrapper_loc)
    loop = _check(_for(body, i, metadata={"static": 23}, loc=wrapper_loc))
    seen = []

    def rewrite(fields):
        seen.append(str(fields["tag"]))
        return {**fields, "visit": len(seen)}

    result = _check(_invoke("RewriteLoopAnnotationCarriers", loop, rewrite))
    assert seen == ["first", "second", "wrapper", "third"]
    assert result.loc.same_as(loop.loc) and result.body.loc.same_as(body.loc)
    assert result.annotations.same_as(loop.annotations)
    assert result.body.seq[1].same_as(body.seq[1]) and result.body.seq[-1].same_as(child)
    for before, after in [(first, result.body.seq[0]), (second, result.body.seq[2]), (third, result.body.seq[3].body.seq[1])]:
        assert after.loc.same_as(before.loc) and after.value.loc.same_as(before.value.loc)
        assert int(after.value.attrs["keep"]) == int(before.value.attrs["keep"])
        assert "visit" in _decode(after) and "visit" not in _decode(before)
    replacement = result.body.seq[3]
    assert replacement.loc.same_as(wrapper.loc) and int(replacement.attrs["keep"]) == 25
    assert replacement.body.seq[0].same_as(wrapper.body.seq[0])


def test_noop_rewrite_and_empty_append_retain_ir_identity():
    loop = _check(_for([annotation({"value": 1})]))
    assert _invoke("RewriteLoopAnnotationCarriers", loop, lambda fields: dict(fields)).same_as(loop)
    assert _invoke("AppendLoopAnnotation", loop, {}).same_as(loop)


@pytest.mark.parametrize("wrapper_depth", [0, 1, 2])
def test_append_uses_the_last_dominating_position_inside_sole_metadata_wrappers(wrapper_depth):
    a, alias = tensor("A", [8], "int32", "global"), tensor("alias", [8], "int32", "global")
    i, local = ir.Var("i", "int32"), ir.Var("local", "int32")
    body = [annotation({"head": 1}), _declare(alias, a), ir.Bind(local, i + 1), ir.TensorStore(alias, [local], integer(0))]
    for depth in range(wrapper_depth):
        body = [attribute(None, "tl.loop.annotations", {"depth": depth}, body)]
    loop = _check(_for(body, i, metadata={"static": 2}), [a])
    result = _check(_invoke("AppendLoopAnnotation", loop, {"conflict_hint": [_hint(alias, subregion(a, [local], [1]))]}), [a])
    assert result.annotations.same_as(loop.annotations)
    tail = result.body
    for _ in range(wrapper_depth):
        assert len(tail.seq) == 1 and isinstance(tail.seq[0], ir.RegionStmt)
        tail = tail.seq[0].body
    assert isinstance(tail.seq[-2], ir.TensorStore)
    assert "conflict_hint" in _decode(tail.seq[-1])
    assert len(_invoke("ReadLoopAnnotations", result)["conflict_hint"]) == 1


@pytest.mark.parametrize("operand", ["scalar", "tensor", "region"])
def test_rewrite_rejects_a_reference_to_a_later_local_definition(operand):
    a, local = tensor("A", [4], "int32", "global"), ir.Var("local", "int32")
    alias = tensor("alias", [4], "int32", "global")
    loop = _check(_for([annotation({"head": 1}), ir.Bind(local, integer(1)), _declare(alias, a)]), [a])
    value = local if operand == "scalar" else alias if operand == "tensor" else subregion(a, [local], [1])
    before = ir.save_json(loop)
    with pytest.raises(ValueError, match="dominating definition scope"):
        _api("RewriteLoopAnnotationCarriers")(loop, lambda fields: {**fields, "bad": value})
    assert ir.save_json(loop) == before


@pytest.mark.parametrize("domain", ["serial", "parallel", "if", "else", "while", "metadata", "assume", "task"])
def test_append_rejects_local_operands_that_belong_to_nested_scopes(domain):
    flag, local = ir.Var("flag", "bool"), ir.Var("local", "int32")
    definition = ir.Bind(local, integer(1))
    if domain in {"serial", "parallel"}:
        scope = _for([definition], parallel=domain == "parallel")
    elif domain in {"if", "else"}:
        scope = ir.If(flag, [definition] if domain == "if" else [], [definition] if domain == "else" else None)
    elif domain == "while":
        scope = ir.While(flag, [definition])
    else:
        key = {"metadata": "tl.loop.annotations", "assume": "tilelang_assume", "task": "tl.ascend_task"}[domain]
        scope = attribute(flag if domain == "assume" else None, key, {} if domain == "metadata" else integer(1), [definition])
    # The trailing statement makes a metadata wrapper a nested lexical scope,
    # rather than the sole wrapper into which Append is allowed to descend.
    loop = _check(_for([scope, annotation({"tail": 1})]), [flag])
    before = ir.save_json(loop)
    with pytest.raises(ValueError, match="dominating definition scope"):
        _api("AppendLoopAnnotation")(loop, {"bad": local})
    assert ir.save_json(loop) == before


@pytest.mark.parametrize("dependency", ["shape", "layout"])
@pytest.mark.parametrize("site", ["forward", "nested"])
def test_new_outer_tensor_operands_cannot_capture_unavailable_locals_in_their_types(dependency, site):
    i, size = ir.Var("i", "int32"), ir.Var("size", "int32")
    view = (
        tensor("outer_view", [size], "int32", "global")
        if dependency == "shape"
        else strided_tensor("outer_view", [4], [size], dtype="int32")
    )
    definition = ir.Bind(size, i + 1)
    if site == "nested":
        definition = _for([definition])
    loop = _check(_for([annotation({"head": 1}), definition], i))
    before = ir.save_json(loop)
    with pytest.raises(ValueError, match="size.*dominating definition scope"):
        _api("RewriteLoopAnnotationCarriers")(loop, lambda fields: {**fields, "view": view})
    if site == "nested":
        with pytest.raises(ValueError, match="size.*dominating definition scope"):
            _api("AppendLoopAnnotation")(loop, {"view": view})
    assert ir.save_json(loop) == before


@pytest.fixture(scope="module")
def scoped_region_op():
    name = "test.loop_metadata_scoped_results"
    ir._ffi_api.RegisterOp(name, "Test the shared native region parameter/result scope contract")
    op = ir.Op.get(name)
    op.set_signature(args=[])
    op.set_attrs_type_key("ir.DictAttrs")
    ir.register_op_attr(name, "FRegionGetBodyParams", lambda call: [ir.Var("body_parameter", "int32")])
    return op


def test_nonmetadata_region_results_dominate_following_carriers_but_body_variables_do_not(scoped_region_op):
    parameter, local, output = (ir.Var(name, "int32") for name in ("parameter", "body_local", "result"))
    scope = ir.RegionStmt(
        scoped_region_op,
        [],
        [parameter],
        {},
        [ir.Bind(local, parameter + 1), annotation({"body_only": local})],
        result_vars=[output],
    )
    loop = _check(_for([scope, annotation({"result": output})]))
    assert set(_invoke("ReadLoopAnnotations", loop)) == {"result"}
    result = _check(_invoke("AppendLoopAnnotation", loop, {"appended_result": output}))
    assert result.body.seq[0].same_as(scope)
    result = _check(_invoke("RewriteLoopAnnotationCarriers", loop, lambda fields: {**fields, "copied_result": output}))
    assert result.body.seq[0].same_as(scope)
    before = ir.save_json(loop)
    for variable in [parameter, local]:
        with pytest.raises(ValueError, match="dominating definition scope"):
            _api("AppendLoopAnnotation")(loop, {"bad": variable})
        with pytest.raises(ValueError, match="dominating definition scope"):
            _api("RewriteLoopAnnotationCarriers")(loop, lambda fields, var=variable: {**fields, "bad": var})
    assert ir.save_json(loop) == before


@pytest.mark.parametrize("operand", ["tensor", "pointer", "region"])
@pytest.mark.parametrize("mode", ["no_conflict", "force", "false_then_true", "true_then_false"])
def test_dependency_consumer_reads_local_carriers_and_force_conflict_wins(operand, mode):
    a, b, a_alias, b_alias = (tensor(name, [4], "int32", "global") for name in ("A", "B", "a_alias", "b_alias"))
    i, local = ir.Var("i", "int32"), ir.Var("local", "int32")
    a_ptr, b_ptr = ir.Var("a_ptr", a.data.ty), ir.Var("b_ptr", b.data.ty)
    definitions = [_unit(_declare(a_alias, a)), _unit(_declare(b_alias, b)), _unit(ir.Bind(local, i))]
    if operand == "pointer":
        definitions.extend([_unit(ir.Bind(a_ptr, a_alias.data)), _unit(ir.Bind(b_ptr, b_alias.data))])
        left, right = a_ptr, b_ptr
    elif operand == "region":
        left, right = subregion(a_alias, [local], [1]), subregion(b_alias, [local], [1])
    else:
        left, right = a_alias, b_alias
    writer = ir.TensorStore(a_alias if operand == "region" else a, [local], integer(1))
    reader = ir.TensorLoad(b_alias if operand == "region" else b, [local])
    polarities = {"no_conflict": [False], "force": [True], "false_then_true": [False, True], "true_then_false": [True, False]}[mode]
    carriers = [annotation({"conflict_hint": [_hint(left, right, 0, polarity)]}) for polarity in polarities]
    params = [a, b] if mode == "no_conflict" else []
    prefix = [] if params else [_unit(allocate(a)), _unit(allocate(b))]
    result = _inspect(
        _kernel(params, [*prefix, _loop(i, [*definitions, carriers[0], _unit(writer), *carriers[1:], _unit(ir.Evaluate(reader))])])
    )
    pair = (_ids(result, writer)[0], _ids(result, reader)[0], 0)
    assert any(row[:3] == pair for row in _pairs(result, clock=i)) == (mode != "no_conflict")


@pytest.mark.parametrize("pointer_hint", [False, True])
def test_bare_alias_hints_also_match_proven_storage_of_opaque_accesses(pointer_hint):
    a, b, alias = (tensor(name, [4], "int32", "global") for name in ("A", "B", "alias"))
    i, pointer = ir.Var("i", "int32"), ir.Var("pointer", b.data.ty)
    writer = ir.TensorStore(a, [i], integer(1))
    opaque = ir.Call("tirx.call_extern", [ir.StringImm("opaque_consumer"), b.data], ty="int32")
    result = _inspect(
        _kernel(
            [a, b],
            [
                _loop(
                    i,
                    [
                        _unit(_declare(alias, b)),
                        _unit(ir.Bind(pointer, alias.data)),
                        annotation({"conflict_hint": [_hint(a, pointer if pointer_hint else alias, 0)]}),
                        _unit(writer),
                        _unit(ir.Evaluate(opaque)),
                    ],
                )
            ],
        )
    )
    pair = (_ids(result, writer)[0], _ids(result, opaque)[0], 0)
    assert not any(row[:3] == pair for row in _pairs(result, clock=i))


def test_inexact_integer_pointer_roots_do_not_prove_hint_alias_equivalence():
    bits, i = ir.Var("bits", "uint64"), ir.Var("i", "int32")
    b, used, hinted = (tensor(name, [4], "int32", "global") for name in ("B", "used", "hinted"))
    address = ir.Call("tirx.reinterpret", [bits], ty=used.data.ty)
    definitions = [_unit(ir.Bind(view, ir.Call("tirx.decl_tensor", [address], ty_args=[view.ty]))) for view in (used, hinted)]
    writer, reader = ir.TensorStore(used, [i], integer(1)), ir.TensorLoad(b, [i])
    result = _inspect(
        _kernel(
            [bits, b],
            [_loop(i, [*definitions, annotation({"conflict_hint": [_hint(hinted, b, 0)]}), _unit(writer), _unit(ir.Evaluate(reader))])],
        )
    )
    pair = (_ids(result, writer)[0], _ids(result, reader)[0], 0)
    assert any(row[:3] == pair for row in _pairs(result, clock=i))
