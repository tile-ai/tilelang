"""Retire only consumed SF version metadata, retaining physical data owners."""

import pytest
from test_native_ascend_ops import constant, function
from test_native_ascend_sf_lowering import lower, sf_variables
from test_native_reducer_contracts import (
    allocate,
    annotation,
    annotations,
    attribute,
    integer,
    loop,
    objects,
    run_pass,
    tensor,
)

from tvm_tirx import ir, tirx

VERSION = "tl.buffer_version"
BINDINGS = "tl.l0_sf_bindings"


def group(count=2, *, scope="shared.l0a", dtype="uint16", leading=None):
    leading = ([count] if count > 1 else []) if leading is None else leading
    data = tensor("data", [*leading, 16, 64], "float8_e4m3fn", scope)
    sf = tensor("sf", [*leading, 16, 2 if dtype == "uint8" else 1], dtype, scope + ".sf")
    return data, sf


def metadata_case(data, sf, statements, params=()):
    return function(
        list(params),
        [allocate(data), allocate(sf), annotation({BINDINGS: {sf: data}}), *statements],
    )


def version_maps(value):
    return [fields[VERSION] for fields in annotations(value) if VERSION in fields]


def assert_data_entries(value, expected):
    maps = version_maps(value)
    assert len(maps) == len(expected)
    for actual, required in zip(maps, expected):
        assert len(actual) == len(required)
        for data, count in required.items():
            assert data in actual
            assert constant(actual[data]) == count
    assert not sf_variables(value)
    assert BINDINGS not in ir.save_json(value)


def rejected_unchanged(value, diagnostic):
    original = ir.save_json(value)
    with pytest.raises(ValueError, match=diagnostic):
        run_pass("AscendLowerTileOp", value)
    assert ir.save_json(value) == original
    assert sf_variables(value)


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_sf_versions_retire_but_exact_data_and_unrelated_entries_survive(count, scope, dtype):
    data, sf = group(count, scope=scope, dtype=dtype)
    ordinary = tensor("ordinary", [3, 32], "float16", "shared.dyn")
    marker = ir.Var("marker", "int32")
    value = function(
        [marker],
        [
            allocate(data),
            allocate(sf),
            allocate(ordinary),
            annotation(
                {
                    BINDINGS: {sf: data},
                    VERSION: {data: count, sf: count, ordinary: 3},
                    "foreign_marker": marker,
                }
            ),
        ],
    )
    result = lower(value, "sf-version-" + scope + dtype + str(count), infer=False)
    assert_data_entries(result, [{data: count, ordinary: 3}])
    assert any(fields.get("foreign_marker") is not None and fields["foreign_marker"].same_as(marker) for fields in annotations(result))
    allocated = [bind.var for bind in objects(result, ir.Bind) if isinstance(bind.var.ty, tirx.TensorType)]
    assert len(allocated) == 2
    assert any(var.same_as(data) for var in allocated)
    assert any(var.same_as(ordinary) for var in allocated)


@pytest.mark.parametrize("placement", ["split", "duplicate", "sf_only", "data_only", "typed_count"])
def test_split_and_partial_version_maps_keep_all_physical_data_entries(placement):
    data, sf = group()
    count = integer(2, "int64") if placement == "typed_count" else 2
    fields = [annotation({VERSION: {sf: count}})]
    expected = []
    if placement != "sf_only":
        fields.append(annotation({VERSION: {data: count}}))
        expected.append({data: 2})
    if placement == "data_only":
        fields.pop(0)
    if placement == "duplicate":
        fields.append(annotation({VERSION: {sf: count, data: count}}))
        expected.append({data: 2})
    value = metadata_case(data, sf, fields)
    result = lower(value, "sf-version-placement-" + placement, infer=False)
    assert_data_entries(result, expected)


def test_distinct_same_named_groups_keep_independent_physical_counts():
    data0, sf0 = group(2)
    data1, sf1 = group(3)
    assert data0.name == data1.name and sf0.name == sf1.name
    value = function(
        [],
        [
            *(allocate(var) for var in (data0, sf0, data1, sf1)),
            annotation({BINDINGS: {sf0: data0, sf1: data1}, VERSION: {data0: 2, sf0: 2, data1: 3, sf1: 3}}),
        ],
    )
    result = lower(value, "sf-version-same-named-groups", infer=False)
    assert_data_entries(result, [{data0: 2, data1: 3}])


@pytest.mark.parametrize("placement", ["branch", "loop", "binding_attribute", "binding_kernel"])
def test_version_consumption_preserves_surrounding_owner_and_side_effects(placement):
    data, sf = group()
    marker = tensor("marker", [1], "int32", "global")
    condition = ir.Var("condition", "bool")
    index = ir.Var("index", "int32")
    version = annotation({VERSION: {sf: 2, data: 2}})
    side_effect = ir.TensorStore(marker, [0], integer(19))
    if placement == "branch":
        statements = [ir.If(condition, [version, side_effect], None)]
    elif placement == "loop":
        statements = [loop(index, 5, [version, side_effect], parallel=False)]
    else:
        key, fields = (
            (BINDINGS, {sf: data}) if placement == "binding_attribute" else ("tl.kernel", {BINDINGS: {sf: data}, "foreign_marker": 7})
        )
        statements = [attribute(None, key, fields, [version, side_effect])]
    value = metadata_case(data, sf, statements, [marker, condition])
    result = lower(value, "sf-version-owner-" + placement, infer=False)
    assert_data_entries(result, [{data: 2}])
    stores = objects(result, ir.TensorStore)
    assert len(stores) == 1 and stores[0].dest.same_as(marker)
    assert constant(stores[0].value) == 19
    if placement == "branch":
        branches = objects(result, ir.If)
        assert len(branches) == 1 and branches[0].condition.same_as(condition)
    elif placement == "loop":
        loops = objects(result, ir.For)
        assert len(loops) == 1 and loops[0].var.same_as(index)
        assert (constant(loops[0].begin), constant(loops[0].end), constant(loops[0].step)) == (
            0,
            5,
            1,
        )
    elif placement == "binding_attribute":
        assert any(stmt.op.name == "tl.lexical_scope" for stmt in objects(result, ir.RegionStmt))
    else:
        assert "foreign_marker" in ir.save_json(result)


@pytest.mark.parametrize("bad", ["zero", "negative", "overflow", "dynamic", "bool", "float", "typed_bool", "vector"])
@pytest.mark.parametrize("member", ["sf", "data"])
def test_invalid_group_version_counts_fail_before_retirement(bad, member):
    data, sf = group()
    dynamic = ir.Var("dynamic", "int32")
    count = {
        "zero": 0,
        "negative": -1,
        "overflow": 2**31,
        "dynamic": dynamic,
        "bool": True,
        "float": 2.0,
        "typed_bool": ir.const(True, "bool"),
        "vector": ir.prim.Broadcast(integer(2), integer(2)),
    }[bad]
    selected = sf if member == "sf" else data
    value = metadata_case(data, sf, [annotation({VERSION: {selected: count}})], [dynamic])
    rejected_unchanged(value, "positive int32 constant")


@pytest.mark.parametrize("member", ["sf", "data"])
def test_duplicate_group_count_conflicts_are_not_hidden(member):
    data, sf = group()
    selected = sf if member == "sf" else data
    value = metadata_case(data, sf, [annotation({VERSION: {selected: 2}}), annotation({VERSION: {selected: 3}})])
    rejected_unchanged(value, "Conflicting L0 SF physical version counts")


@pytest.mark.parametrize("split", [False, True])
def test_bound_data_and_sf_must_agree_on_version_count(split):
    data, sf = group()
    statements = [annotation({VERSION: {sf: 2}}), annotation({VERSION: {data: 3}})] if split else [annotation({VERSION: {sf: 2, data: 3}})]
    rejected_unchanged(metadata_case(data, sf, statements), "Bound L0 data/SF physical version counts disagree")


@pytest.mark.parametrize("leading", [[], [3], [1, 2]])
def test_sf_version_count_requires_the_same_physical_leading_axis(leading):
    data, sf = group(leading=leading)
    rejected_unchanged(
        metadata_case(data, sf, [annotation({VERSION: {sf: 2, data: 2}})]),
        "does not match its leading axis",
    )


@pytest.mark.parametrize("carrier", ["attribute", "kernel", "sblock", "foreign_field"])
def test_version_fields_on_foreign_carriers_still_fail_sf_use_verification(carrier):
    data, sf = group()
    versions = {sf: 2, data: 2}
    if carrier == "attribute":
        statement = attribute(None, VERSION, versions, [ir.Evaluate(integer(0))])
    elif carrier == "kernel":
        statement = attribute(None, "tl.kernel", {VERSION: versions}, [ir.Evaluate(integer(0))])
    elif carrier == "sblock":
        statement = attribute(
            None,
            "tl.sblock",
            {"name": "foreign", "no_realize": True, "annotations": {VERSION: versions}},
            [ir.Evaluate(integer(0))],
        )
    else:
        statement = annotation({"foreign_version_field": versions})
    rejected_unchanged(metadata_case(data, sf, [statement]), "unlowered consumer")


@pytest.mark.parametrize("use", ["load", "pointer", "metadata"])
def test_live_sf_uses_are_not_erased_with_valid_version_metadata(use):
    data, sf = group()
    consumer = {
        "load": ir.Evaluate(ir.TensorLoad(sf, [0, 0, 0])),
        "pointer": ir.Evaluate(sf.data),
        "metadata": annotation({"foreign_metadata": sf}),
    }[use]
    value = metadata_case(data, sf, [annotation({VERSION: {sf: 2, data: 2}}), consumer])
    rejected_unchanged(value, "unlowered consumer")


def test_version_metadata_cannot_supply_a_missing_sf_allocation_binding():
    data, sf = group()
    value = function([], [allocate(data), allocate(sf), annotation({VERSION: {sf: 2, data: 2}})])
    rejected_unchanged(value, "no allocation binding")
