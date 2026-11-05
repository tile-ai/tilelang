"""Observe storage metadata without erasing entries or crossing lexical scopes."""

import pytest
from tvm_tirx import ir

from testing.ascend._ir import allocate, annotation, sblock, sblock_annotations, tensor


STORAGE_MAPS = ["tl.buffer_versions_map", "tl.manual_multi_buffer", "tl.buffer_version_mode"]


def _values(key):
    return ("iteration", "counter") if key == "tl.buffer_version_mode" else (2, 3)


@pytest.mark.parametrize("key", STORAGE_MAPS)
def test_split_storage_entries_follow_distinct_definitions_and_identities(key):
    a, b = [tensor((8,), "int32", name="same_name", scope="shared.dyn") for _ in range(2)]
    first, second = _values(key)
    repeated = first if isinstance(first, str) else ir.const(first, "int64")
    parent = sblock(
        "parent",
        [
            allocate(a),
            annotation({key: {a: first}}),
            allocate(b),
            ir.SeqStmt([annotation({key: {b: second}})]),
            annotation({key: {}}),
            annotation({key: {a: repeated}}),
        ],
    )
    fields = sblock_annotations(parent)[key]
    assert len(fields) == 2
    assert a in fields and b in fields and not a.same_as(b)
    assert fields[a] == first if isinstance(first, str) else int(fields[a]) == first
    assert fields[b] == second if isinstance(second, str) else int(fields[b]) == second


@pytest.mark.parametrize("key", STORAGE_MAPS)
def test_child_loop_and_region_storage_entries_remain_private(key):
    a, b = [tensor((8,), "int32", name=name, scope="shared.dyn") for name in ("a", "b")]
    first, second = _values(key)
    child = sblock("child", [annotation({key: {a: second, b: second}})])
    i = ir.Var("i", "int32")
    loop = ir.For(i, 0, 2, 1, ir.ForKind.DEFAULT, [annotation({key: {a: second}})])
    branch = ir.If(ir.const(True), [annotation({key: {a: second}})], None)
    parent = sblock("parent", [allocate(a), allocate(b), annotation({key: {a: first}}), child, loop, branch])
    fields = sblock_annotations(parent)[key]
    assert len(fields) == 1 and next(iter(fields)).same_as(a)
    assert fields[a] == first if isinstance(first, str) else int(fields[a]) == first
    child_fields = sblock_annotations(child)[key]
    assert len(child_fields) == 2
    assert child_fields[a] == second if isinstance(second, str) else int(child_fields[a]) == second


@pytest.mark.parametrize("key", STORAGE_MAPS)
def test_conflicting_storage_entries_do_not_silently_replace_the_first_value(key):
    a = tensor((8,), "int32", name="a", scope="shared.dyn")
    first, second = _values(key)
    parent = sblock("parent", [allocate(a), annotation({key: {a: first}}), annotation({key: {a: second}})])
    with pytest.raises(ValueError, match=f"Conflicting {key} for storage"):
        sblock_annotations(parent)


def test_other_map_annotations_retain_the_existing_replacement_rule():
    a, b = [tensor((8,), "int32", name=name, scope="shared.dyn") for name in ("a", "b")]
    parent = sblock("parent", [allocate(a), allocate(b), annotation({"custom_map": {a: 2}}), annotation({"custom_map": {b: 3}})])
    fields = sblock_annotations(parent)["custom_map"]
    assert len(fields) == 1 and b in fields and a not in fields
