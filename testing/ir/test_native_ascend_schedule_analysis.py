"""Production Ascend scheduler codec and memory proofs preserve native semantics."""

import itertools

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from test_native_ascend_ops import function, strided_tensor, subregion
from test_native_ascend_schedule_units import _attributes, _check, _fields, _kernel, _scheduled_bind
from test_native_reducer_contracts import allocate, annotation, annotations, attribute, integer, objects, region, tensor, tileop


def _api(name):
    return tvm_ffi.get_global_func("tl.testing." + name)


def _schedule(params, body):
    value = _check(_kernel(params, body))
    return value, _attributes(value, "tl.sblock")[0]


def _has(values, value):
    return any(item.same_as(value) for item in values)


def _int(expression):
    return int(tvm_tirx.sym.Analyzer().simplify(expression))


def _raw_for_values(begin, end, step):
    values = []
    current = begin
    while current < end:
        assert len(values) < 32, "This fixture must have a finite raw For trace"
        values.append(current)
        current += step
    return values


def _points(region):
    bounds = [range(_int(axis.begin), _int(axis.begin + axis.extent)) for axis in region.region]
    return set(itertools.product(*bounds))


def _memory(body, definitions=(), physical=True):
    combined = ir.SeqStmt([*definitions, *body])
    assert tirx.analysis.check_well_formed(function(tirx.analysis.undefined_vars(combined), [combined]))
    return _api("AnalyzeAscendMemory")(ir.SeqStmt(body), ir.SeqStmt(definitions), physical)


def _assert_function(value, root):
    result = tirx.Function(value.params, [root], value.ret_type, value.attrs)
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


@pytest.mark.parametrize("clone", [False, True])
def test_schedule_codec_preserves_scoped_bindings_metadata_and_locations(clone):
    extent = ir.Var("extent", "int32")
    output = tensor("output", [4], "int32", "global")
    size = ir.Var("size", "int32")
    left, right = ir.Var("local", "int32"), ir.Var("local", "int32")
    index = ir.Var("index", "int32")
    scratch = strided_tensor("scratch", [size], [2], 3, "int32", "shared.dyn")
    source_location = ir.SourceLoc(ir.SourceName("schedule_analysis.py"), 11, 2, 11, 20)
    value, root = _schedule(
        [extent, output],
        [
            ir.Bind(size, extent + 1, loc=source_location),
            allocate(scratch),
            annotation({"custom": {scratch: [size]}, "tl.buffer_versions_map": {scratch: 2}}),
            ir.If(
                extent > 2,
                [ir.Bind(left, size + 1), ir.TensorStore(output, [integer(0)], left)],
                [ir.Bind(right, size + 2), ir.TensorStore(output, [integer(0)], right)],
            ),
            attribute(
                extent > 0,
                "tilelang_assume",
                integer(1),
                [ir.For(index, integer(3), integer(-1), integer(-1), ir.ForKind.DEFAULT, [ir.TensorStore(output, [index], size)])],
            ),
        ],
    )
    before = ir.save_json(root)
    encoded = _api("RoundTripAscendSchedule")(root, clone)
    assert ir.save_json(root) == before
    ir.assert_structural_equal(encoded, root)
    assert ir.save_json(encoded) == before
    _assert_function(value, encoded)
    bindings = objects(encoded, ir.Bind)
    assert next(bind for bind in bindings if bind.var.same_as(size)).loc.same_as(source_location)
    assert _has([bind.var for bind in bindings], left)
    assert _has([bind.var for bind in bindings], right)


@pytest.mark.parametrize("with_existing_carrier", [False, True])
def test_generated_storage_metadata_stays_after_lexical_definitions(with_existing_carrier):
    a, b = tensor("a", [8]), tensor("b", [8])
    body = [allocate(a), allocate(b)]
    if with_existing_carrier:
        body.append(annotation({"tl.buffer_alias_map": {}, "tl.buffer_versions_map": {a: 2}, "keep": [a]}))
    value, root = _schedule([], body)
    encoded = _api("SetAscendScheduleMetadata")(root, {a: 3}, {a: [b], b: [a]}, True)
    _assert_function(value, encoded)
    assert not _fields(encoded)["value"].get("annotations")
    metadata = annotations(encoded)
    versions = [item["tl.buffer_versions_map"] for item in metadata if "tl.buffer_versions_map" in item]
    aliases = [item["tl.buffer_alias_map"] for item in metadata if "tl.buffer_alias_map" in item]
    assert len(versions) == len(aliases) == 1
    assert len(versions[0]) == 1
    assert int(versions[0][a]) == 3
    assert len(aliases[0]) == 2
    assert _has(aliases[0][a], b)
    assert _has(aliases[0][b], a)
    # Newly emitted storage entries follow their own definition; an existing
    # carrier may retain the maps together. In either form every referenced
    # storage Var must already be defined at that carrier.
    visible = []
    for statement in encoded.body.seq:
        if isinstance(statement, ir.Bind):
            visible.append(statement.var)
        for item in annotations(statement):
            for storage in item.get("tl.buffer_versions_map", {}):
                assert _has(visible, storage)
            for storage, peers in item.get("tl.buffer_alias_map", {}).items():
                assert _has(visible, storage)
                assert all(_has(visible, peer) for peer in peers)
    if with_existing_carrier:
        assert _has(next(item["keep"] for item in metadata if "keep" in item), a)
    assert isinstance(encoded.body.seq[-1], ir.Evaluate)
    second = _api("RoundTripAscendSchedule")(encoded, True)
    ir.assert_structural_equal(second, encoded)


def test_present_empty_alias_contract_survives_roundtrip():
    value, root = _schedule([], [])
    encoded = _api("SetAscendScheduleMetadata")(root, {}, {}, True)
    _assert_function(value, encoded)
    assert len(annotations(encoded)) == 1
    assert "tl.buffer_alias_map" in annotations(encoded)[0]
    assert len(annotations(encoded)[0]["tl.buffer_alias_map"]) == 0
    ir.assert_structural_equal(_api("RoundTripAscendSchedule")(encoded, False), encoded)


@pytest.mark.parametrize("invalid", ["self", "asymmetric", "missing", "zero_versions", "negative_versions", "scalar_key"])
def test_invalid_storage_metadata_is_rejected(invalid):
    a, b = tensor("a", [8]), tensor("b", [8])
    _, root = _schedule([], [allocate(a), allocate(b)])
    versions, aliases = {}, {}
    if invalid == "self":
        aliases = {a: [a]}
    elif invalid == "asymmetric":
        aliases = {a: [b], b: []}
    elif invalid == "missing":
        aliases = {a: [b]}
    elif invalid == "scalar_key":
        versions = {ir.Var("scalar", "int32"): 2}
    else:
        versions = {a: 0 if invalid == "zero_versions" else -1}
    with pytest.raises((ValueError, RuntimeError, TypeError), match="alias|storage|Storage|count|positive|tensor|pointer|0"):
        _api("SetAscendScheduleMetadata")(root, versions, aliases, True)


def test_branch_constraints_and_ssa_definitions_do_not_leak():
    n, x = ir.Var("n", "int32"), ir.Var("x", "int32")
    lhs, rhs, after = (ir.Var(name, "int32") for name in ("branch", "branch", "after"))
    _, root = _schedule(
        [n],
        [ir.Bind(x, n + 1), ir.If(n > 4, [ir.Bind(lhs, x * 2)], [ir.Bind(rhs, x * 3)]), ir.Bind(after, x * 4)],
    )
    result = _api("AnalyzeAscendSchedule")(root, [x == n + 1, n > 4, n <= 4])
    for variable, expected in [(lhs, [True, True, False]), (rhs, [True, False, True]), (after, [True, False, False])]:
        row = next(item for item in result["tasks"] if _has(item["write_vars"], variable))
        assert list(row["proofs"]) == expected
    order = [[int(item) for item in row] for row in result["order"]]
    for i in range(len(order)):
        assert order[i][i] == 0
        for j in range(i + 1, len(order)):
            assert order[i][j] < 0 and order[j][i] > 0


@pytest.mark.parametrize("wrapper", ["if", "assume"])
def test_mutable_condition_snapshot_does_not_prove_a_later_load(wrapper):
    flag, output = tensor("flag", [1], "int32", "global"), tensor("output", [1], "int32", "global")
    loaded = ir.TensorLoad(flag, [integer(0)])
    body = [ir.TensorStore(flag, [integer(0)], integer(0)), ir.TensorStore(output, [integer(0)], integer(3))]
    scope = ir.If(loaded > 0, body, None) if wrapper == "if" else attribute(loaded > 0, "tilelang_assume", integer(1), body)
    _, root = _schedule([flag, output], [scope])
    result = _api("AnalyzeAscendSchedule")(root, [loaded > 0])
    assert len(result["tasks"]) == 2
    assert all(not row["proofs"][0] for row in result["tasks"])


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("start,stop,step", [(0, 7, 2), (7, 0, -2), (5, 5, 1), (2, 9, 1)])
def test_raw_for_trip_count_and_ordinal_use_the_fixed_less_than_condition(dtype, start, stop, step):
    index, value = ir.Var("i", dtype), ir.Var("value", dtype)
    _, root = _schedule(
        [],
        [ir.For(index, integer(start, dtype), integer(stop, dtype), integer(step, dtype), ir.ForKind.DEFAULT, [ir.Bind(value, index)])],
    )
    points = _raw_for_values(start, stop, step)
    result = _api("AnalyzeAscendSchedule")(root, [index >= min(start, stop + 1), index <= max(start, stop - 1)])
    assert int(result["loops"][0]["trip_count"]) == len(points)
    if points:
        assert all(result["loops"][0]["proofs"])
    (ordinal,) = _api("AscendLoopOrdinal")(root)
    mapping = tirx.IndexMap([index], [ordinal], None)
    assert [_int(mapping.map_indices([integer(point, dtype)])[0]) for point in points] == list(range(len(points)))


def test_unknown_signed_step_cannot_prove_positive_loop_bounds_or_ordinal():
    step, index, value = (ir.Var(name, "int32") for name in ("step", "index", "value"))
    _, root = _schedule([step], [ir.For(index, integer(0), integer(7), step, ir.ForKind.DEFAULT, [ir.Bind(value, index)])])
    result = _api("AnalyzeAscendSchedule")(root, [index < 7, index >= 0])
    assert list(result["tasks"][0]["proofs"]) == [True, False]
    assert int(result["loops"][0]["trip_count"]) == 100
    with pytest.raises((ValueError, RuntimeError), match="nonzero step"):
        _api("AscendLoopOrdinal")(root)


def test_nested_loop_ordinal_crosses_lexical_scope_in_execution_order():
    outer, inner, value = (ir.Var(name, "int32") for name in ("outer", "inner", "value"))
    loop = ir.For(inner, integer(1), integer(6), integer(2), ir.ForKind.DEFAULT, [ir.Bind(value, outer + inner)])
    scope = attribute(outer > 0, "tilelang_assume", integer(1), [loop])
    _, root = _schedule([], [ir.For(outer, integer(1), integer(4), integer(1), ir.ForKind.DEFAULT, [scope])])
    ordinals = _api("AscendLoopOrdinal")(root)
    mapping = tirx.IndexMap([outer, inner], [ordinals[1]], None)
    actual = [_int(mapping.map_indices([integer(i), integer(j)])[0]) for i in _raw_for_values(1, 4, 1) for j in _raw_for_values(1, 6, 2)]
    assert actual == list(range(9))


def test_int32_trip_count_subtraction_does_not_overflow_before_widening():
    index, value = ir.Var("index", "int32"), ir.Var("value", "int32")
    _, root = _schedule([], [ir.For(index, integer(-(2**30)), integer(2**30), integer(1), ir.ForKind.DEFAULT, [ir.Bind(value, index)])])
    result = _api("AnalyzeAscendSchedule")(root, [])
    assert int(result["loops"][0]["trip_count"]) == 2**31


@pytest.mark.parametrize("dtype,bits", [("int32", 32), ("float16", 16), ("float4_e2m1fn", 4)])
def test_storage_alias_offsets_match_independent_bit_address_calculation(dtype, bits):
    storage = tensor("storage", [256], "uint8", "global")
    view = strided_tensor("view", [3, 4], [7, 1], 5, dtype)
    alias = ir.Var("alias", view.ty)
    pointer = ir.Var("pointer", storage.data.ty)
    shifted = ir.Call("tirx.ptr_byte_offset", [storage.data, integer(9, "int64")], ty=pointer.ty)
    typed = ir.Call("tirx.reinterpret", [pointer], ty=ir.PtrType(view.ty.dtype, "global"))
    definitions = [ir.Bind(pointer, shifted), ir.Bind(view, ir.Call("tirx.decl_tensor", [typed], ty_args=[view.ty])), ir.Bind(alias, view)]
    indices = [integer(2), integer(3)]
    before = ir.save_json(ir.SeqStmt(definitions))
    rows = _api("InspectAscendStorage")(ir.SeqStmt(definitions), [view, alias], [indices, indices])
    assert ir.save_json(ir.SeqStmt(definitions)) == before
    for row in rows:
        assert row["exact"] and row["root"].same_as(storage)
        assert _int(row["bytes"]) == 9
        assert _int(row["bits"]) == 9 * 8 + (5 + 2 * 7 + 3) * bits


def test_unknown_pointer_source_cannot_establish_physical_storage_identity():
    pointer = ir.Var("opaque", ir.PtrType(ir.PrimType("int32"), "global"))
    view = tensor("view", [4], "int32", "global")
    definitions = [
        ir.Bind(pointer, ir.Call("tirx.call_extern", [ir.StringImm("opaque_storage")], ty=pointer.ty)),
        ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
    ]
    (row,) = _api("InspectAscendStorage")(ir.SeqStmt(definitions), [view], [[integer(1)]])
    assert not row["exact"] and "bits" not in row
    access = _memory([ir.TensorStore(view, [integer(0)], integer(2))], definitions)
    assert _has(access["opaque"], view)


def test_tensor_declarations_preserve_shape_stride_origin_and_pointer_dependencies():
    size, stride, origin, offset = (ir.Var(name, "int32") for name in ("size", "stride", "origin", "offset"))
    source = tensor("source", [256], "int32", "global")
    view = strided_tensor("view", [size], [stride], origin, "int32")
    address = ir.Call("tirx.ptr_byte_offset", [source.data, offset], ty=source.data.ty)
    declaration = ir.Bind(view, ir.Call("tirx.decl_tensor", [address], ty_args=[view.ty]))
    result = _memory([declaration])
    assert all(_has(result["read_vars"], value) for value in [source, size, stride, origin, offset])
    assert _has(result["write_vars"], view)
    assert not result["reads"] and not result["writes"] and not result["opaque"]


@pytest.mark.parametrize("escape", [False, True])
def test_pointer_alias_formation_and_escape_have_distinct_storage_effects(escape):
    storage = tensor("storage", [16], "int32", "global")
    pointer = ir.Var("pointer", storage.data.ty)
    definitions = [ir.Bind(pointer, ir.Call("tirx.ptr_byte_offset", [storage.data, integer(4, "int64")], ty=pointer.ty))]
    body = list(definitions)
    if escape:
        body.append(ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("consume_pointer"), pointer], ty="int32")))
    result = _memory(body)
    assert _has(result["read_vars"], storage)
    assert _has(result["write_vars"], pointer)
    assert not result["reads"] and not result["writes"]
    assert _has(result["opaque"], storage) == escape


def test_task_specialization_remaps_cached_pointer_origins_without_mutating_input():
    old, new = ir.Var("offset", "int32"), ir.Var("specialized_offset", "int32")
    storage = tensor("storage", [32], "int32", "global")
    output = tensor("output", [1], "int32", "global")
    view = tensor("view", [4], "int32", "global")
    pointer = ir.Var("pointer", storage.data.ty)
    _, root = _schedule(
        [old, storage, output],
        [
            ir.Bind(pointer, ir.Call("tirx.ptr_byte_offset", [storage.data, old], ty=pointer.ty)),
            ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
            ir.TensorStore(output, [integer(0)], ir.TensorLoad(view, [integer(0)])),
        ],
    )
    before = ir.save_json(root)
    result = _api("SubstituteAscendSchedule")(root, old, new)
    assert ir.save_json(root) == before
    rewritten = function([new, storage, output], [result["root"]])
    assert tirx.analysis.check_well_formed(rewritten)
    assert not tirx.analysis.undefined_vars(rewritten)
    assert len(result["read_byte_offsets"]) == 1
    assert tvm_tirx.sym.Analyzer().can_prove_equal(result["read_byte_offsets"][0], ir.prim.Cast("int64", new))


@pytest.mark.parametrize("pattern", ["dense", "strided", "diagonal", "conditional", "stepped", "empty", "break"])
def test_complete_write_proofs_require_actual_dense_unconditional_coverage(pattern):
    shape = [8, 8] if pattern == "diagonal" else [16]
    output = tensor("output", shape, "int32", "global")
    index, enabled = ir.Var("i", "int32"), ir.Var("enabled", "bool")
    start, stop, step = 0, 4, 1
    if pattern == "stepped":
        stop, step = 8, 2
    elif pattern == "empty":
        stop = 0
    indices = [index, index] if pattern == "diagonal" else [index * 2 if pattern == "strided" else index]
    statements = [ir.TensorStore(output, indices, integer(7))]
    if pattern == "break":
        statements.append(ir.Break())
    if pattern == "conditional":
        statements = [ir.If(enabled, statements, None)]
    body = [ir.For(index, integer(start), integer(stop), integer(step), ir.ForKind.DEFAULT, statements)]
    result = _memory(body)
    points = _raw_for_values(start, stop, step)
    if pattern == "break":
        points = points[:1]
    actual = {(i, i) if pattern == "diagonal" else (i * 2 if pattern == "strided" else i,) for i in points}
    if actual:
        may = set().union(*(_points(region) for region in result["writes"]))
        assert actual <= may
    if pattern == "dense":
        assert len(result["must_writes"]) == 1
        assert _points(result["must_writes"][0]) == actual
    else:
        assert not result["must_writes"]


@pytest.mark.parametrize("mask", [1, 2, 3])
def test_pointer_access_mask_and_dense_row_carry_match_physical_elements(mask):
    source = tensor("source", [3, 4], "float32", "global")
    pointer = ir.Call("tl.access_ptr", [ir.TensorLoad(source, [integer(0), integer(3)]), integer(3), integer(mask)])
    result = _memory([ir.Evaluate(pointer)])
    for name, bit in [("reads", 1), ("writes", 2)]:
        if mask & bit:
            assert len(result[name]) == 1
            assert {(0, 3), (1, 0), (1, 1)} <= _points(result[name][0])
        else:
            assert not result[name]
    assert not result["opaque"] and not result["must_writes"]


@pytest.mark.parametrize("padded", [False, True])
def test_copy_access_contract_includes_hardware_padding_only_in_physical_mode(padded):
    source = tensor("source", [2, 5], "float16", "global")
    destination = tensor("destination", [2, 16], "float16", "shared.dyn")
    copy = tileop("ascend_copy", region(source), subregion(destination, [0, 0], [2, 5]), metadata={"data_select": int(padded)})
    for physical in [False, True]:
        result = _memory([copy], physical=physical)
        assert len(result["reads"]) == len(result["writes"]) == 1
        assert _points(result["reads"][0]) == set(itertools.product(range(2), range(5)))
        width = 16 if physical and padded else 5
        expected = set(itertools.product(range(2), range(width)))
        assert _points(result["writes"][0]) == expected
        if physical:
            assert len(result["must_writes"]) == 1
            assert _points(result["must_writes"][0]) == expected
        assert not result["opaque"]


def test_scheduled_scalar_producer_core_is_required_by_a_later_index():
    index = ir.Var("index", "int32")
    output = tensor("output", [8], "int32", "global")
    value = function(
        [output],
        [
            attribute(
                None,
                "tl.sblock",
                {"name": "tilelang_root", "no_realize": True},
                [
                    ir.Bind(index, _scheduled_bind(integer(2), task={"core_mask": integer(2)})),
                    attribute(
                        {"stage": integer(-1, "int64")},
                        "tl.schedule_unit",
                        integer(1),
                        [
                            attribute({}, "tl.ascend_task", integer(1), [ir.TensorStore(output, [index], integer(7))]),
                        ],
                    ),
                ],
            )
        ],
    )
    assert tirx.analysis.check_well_formed(value)
    result = _api("AnalyzeAscendSchedule")(_attributes(value, "tl.sblock")[0], [])
    assert int(result["tasks"][0]["core"]) == 2
    assert int(result["tasks"][1]["input_core"]) == 2
