"""Finalization preserves the execution domains consumed by Ascend emitters."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang.ascend  # noqa: F401 -- Register the target and its backend-owned operators.
import tilelang.transform as tr


def attribute(key, value, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": key, "value": value})
    return ir.RegionStmt(
        ir.Op.get("tl.attr"),
        [payload],
        [],
        {
            "schema_version": 1,
            "annotation_schema": schema,
        },
        ir.SeqStmt(body),
        [],
    )


@pytest.mark.parametrize("name,scope", [("SIMT_VF", "tl.simtvf_scope"), ("SIMD_VF", "tl.simdvf_scope")])
def test_vf_block_and_scope_are_kept_for_the_owning_emitter(name, scope):
    region = attribute(
        "tl.sblock",
        {"name": name, "no_realize": True, "annotations": {"tl.vf_source_index": 7}},
        [
            attribute(scope, 1, [ir.Evaluate(ir.IntImm("int32", 3))]),
        ],
    )
    function = tirx.Function([], [region], ir.NoneType()).with_attr("target", Target("ascend"))
    actual = tr.FinalizeAnnotations()(ir.IRModule({"main": function}))["main"]
    assert tvm_ffi.structural_equal(function, actual)
    assert tirx.analysis.check_well_formed(actual)
    foreign = function.with_attr("target", Target("c"))
    with pytest.raises(ValueError, match="requires an Ascend target"):
        tr.FinalizeAnnotations()(ir.IRModule({"main": foreign}))


def test_vf_source_index_rejects_invalid_annotation():
    region = attribute("tl.sblock", {"name": "SIMD_VF", "annotations": {"tl.vf_source_index": -1}}, [])
    function = tirx.Function([], [region], ir.NoneType()).with_attr("target", Target("ascend"))
    with pytest.raises(ValueError, match="nonnegative"):
        tr.FinalizeAnnotations()(ir.IRModule({"main": function}))
