"""Internal allocation metadata does not widen public source-block scope."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.language as T


def _nodes(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


@pytest.mark.parametrize("roundtrip", ["direct", "json", "renew"])
def test_root_allocations_keep_their_own_typed_metadata(roundtrip):
    @T.prim_func
    def allocations():
        _first = T.alloc_reducer((1,), "float32", op="sum")
        _second = T.alloc_reducer((2,), "int32", op="max")
        _barrier = T.alloc_barrier([32, 64])
        _cluster = T.alloc_cluster_barrier(128)

    if roundtrip == "json":
        allocations = ir.load_json(ir.save_json(allocations))
    elif roundtrip == "renew":
        allocations = tirx.renew_def(allocations)
    assert tirx.analysis.check_well_formed(allocations)
    bindings = {bind.var for bind in _nodes(allocations, ir.Bind)}
    records = []
    for call in _nodes(allocations, ir.Call):
        if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation":
            records.append(tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[-1], call.attrs["annotation_schema"]))
    reducers, barriers = {}, {}
    for record in records:
        reducers.update(record.get("reducer_info_v2", {}))
        barriers.update(record.get("barrier_init", {}))
        assert "tl.source_block_annotation" not in record
    assert len(reducers) == len(barriers) == 2
    assert all(var in bindings for var in [*reducers, *barriers])
    assert sorted((int(var.ty.shape[0]), info["op"]) for var, info in reducers.items()) == [(1, "sum"), (2, "max")]
    assert sorted(tuple(map(int, counts)) for counts in barriers.values()) == [(32, 64), (128,)]


def test_internal_allocation_metadata_keeps_explicit_owner_conflict_checks():
    with pytest.raises(ValueError, match=r'Try to merge two annotations with different values for key `"op"`'):

        @T.prim_func
        def invalid():
            with T.sblock("owner"):
                reducer = T.alloc_reducer((1,), "float32", op="sum")
                T.sblock_attr({"reducer_info_v2": {reducer: {"op": "max"}}})


def test_root_allocation_does_not_create_a_public_annotation_owner():
    with pytest.raises(ValueError, match="outside of any T.sblock"):

        @T.prim_func
        def invalid():
            reducer = T.alloc_reducer((1,), "float32", op="sum")
            T.sblock_attr({"reducer_info_v2": {reducer: {"op": "sum"}}})
