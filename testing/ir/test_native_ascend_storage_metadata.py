"""Repeated storage metadata retains each key at a valid native definition site."""

import pytest
from tilelang.ascend import transform
from tvm_tirx import ir

from test_native_ascend_core_passes import _apply, _costed, _metadata
from test_native_ascend_core_resolution import _kernel, _unit
from test_native_ascend_schedule_units import _bind_target
from test_native_reducer_contracts import allocate, annotation, integer, objects, region, tensor, tileop


KEYS = ["tl.buffer_versions_map", "tl.manual_multi_buffer", "tl.buffer_version_mode"]


def _program(key, *, interleaved=False, empty=False, repeated=False, conflict=False):
    a = tensor("a", [8], "int32", "shared.dyn")
    b = tensor("b", [8], "int32", "shared.dyn")
    first, second = ("iteration", "counter") if key == "tl.buffer_version_mode" else (2, 3)
    definitions = [_unit(allocate(a), core=0)]
    if not interleaved:
        definitions.append(_unit(allocate(b), core=0))
    definitions.append(annotation({key: {a: first}, "test.storage_marker": a}))
    if interleaved:
        definitions.append(_unit(allocate(b), core=0))
    definitions.append(annotation({key: {b: second}}))
    if repeated:
        definitions.append(annotation({key: {a: second if conflict else first}}))
    if empty:
        definitions.append(annotation({key: {}}))
    value = _bind_target(
        _kernel(
            [],
            [
                *definitions,
                _unit(ir.TensorStore(a, [integer(0)], integer(1)), core=0),
                _unit(ir.TensorStore(b, [integer(0)], integer(2)), core=0),
            ],
        )
    )
    return value, {"a": first, "b": second}


def _entries(value, key):
    result = []
    for fields in _metadata(value):
        if key in fields:
            result.append(
                {str(storage.name): str(entry) if key == "tl.buffer_version_mode" else int(entry) for storage, entry in fields[key].items()}
            )
    return result


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_distinct_map_entries_survive_scheduled_roundtrip(key, interleaved, empty):
    value, expected = _program(key, interleaved=interleaved, empty=empty)
    value = _apply(transform.AssignCore(), value)
    pieces = _entries(value, key)
    assert pieces == [{"a": expected["a"]}, {"b": expected["b"]}]
    assert {name: entry for piece in pieces for name, entry in piece.items()} == expected
    (marker,) = [fields["test.storage_marker"] for fields in _metadata(value) if "test.storage_marker" in fields]
    assert str(marker.name) == "a"


@pytest.mark.parametrize("key", KEYS)
def test_repeated_identical_declaration_preserves_other_storage_entries(key):
    value, expected = _program(key, interleaved=True, repeated=True)
    result = _apply(transform.AssignCore(), value)
    assert _entries(result, key) == [{"a": expected["a"]}, {"b": expected["b"]}, {"a": expected["a"]}]


@pytest.mark.parametrize("key", KEYS)
def test_conflicting_repeated_storage_declarations_are_rejected(key):
    value, _ = _program(key, interleaved=True, repeated=True, conflict=True)
    with pytest.raises((ValueError, RuntimeError), match="Conflicting .* for storage"):
        _apply(transform.AssignCore(), value)


def test_new_sf_group_version_entry_follows_its_data_definition():
    sf = tensor("sf", [32, 2], "uint16", "shared.l0a.sf")
    data = tensor("data", [32, 128], "float8_e4m3fn", "shared.l0a")
    source = tensor("source", [32, 128], "float8_e4m3fn", "shared.l1")
    scale = tensor("scale", [32, 2], "uint16", "shared.l1")
    output = tensor("output", [4], "uint16", "global")
    index = ir.Var("i", "int32")
    body = [
        _costed(tileop("ascend_copy", region(source), region(data)), core=2, stage=0),
        _costed(tileop("ascend_copy", region(scale), region(sf)), core=2, stage=0),
        _costed(ir.TensorStore(output, [index], ir.TensorLoad(sf, [integer(0), integer(0)])), core=2, stage=2),
    ]
    loop = ir.For(
        index,
        integer(0),
        integer(4),
        integer(1),
        ir.ForKind.DEFAULT,
        body,
        {"num_stages": integer(3), "enable_offset": True, "multi_buffer_eligible": [data, sf]},
    )
    value = _bind_target(
        _kernel(
            [output],
            [
                _costed(allocate(sf), core=2, stage=0),
                annotation({"tl.buffer_versions_map": {sf: 3}}),
                *[_costed(allocate(storage), core=2, stage=0) for storage in (data, source, scale)],
                annotation({"tl.l0_sf_bindings": {sf: data}}),
                _unit(loop),
            ],
            count=1,
        )
    )
    result = _apply(transform.AutoSchedule(), value)
    entries = _entries(result, "tl.buffer_versions_map")
    assert {name: count for part in entries for name, count in part.items()} == {"sf": 3, "data": 3}
    assert len(objects(result, ir.For)) == 1
