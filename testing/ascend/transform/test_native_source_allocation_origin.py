"""Ascend arena ownership consumes validated source allocation origins."""

import pytest
import tilelang
import tvm_ffi
import tvm_tirx as tvm
from tilelang.ascend import transform
from tvm_tirx import ir, tirx

from testing.ascend._ir import allocations, annotation, tensor


def _module(attrs, count=2):
    variables = [tensor((8,), "int32", name=f"a{i}", scope="shared.dyn", align=128) for i in range(count)]
    body = [ir.Bind(var, ir.Call("tirx.alloc_tensor", [], attrs=attrs, ty_args=[var.ty])) for var in variables]
    # The allocator consumes an explicit alias-analysis result. Without one it
    # intentionally leaves allocations unchanged, even when reuse is disabled.
    body.append(annotation({"tl.buffer_alias_map": {var: [] for var in variables}}))
    body.extend(ir.TensorStore(var, [0], i) for i, var in enumerate(variables))
    function = tirx.Function([], body).with_attr("target", tvm.target.Target("ascend")).with_attr("global_symbol", "main")
    return ir.IRModule({"main": function})


def test_source_origin_does_not_change_the_sequential_arena_plan():
    plain = transform.MergeUBAllocations(disable_reuse=True)(_module({}))
    marked = transform.MergeUBAllocations(disable_reuse=True)(_module({"tl.source_block_alloc": ir.IntImm("int32", 1)}))
    tvm.ir.assert_structural_equal(plain, marked, map_free_vars=True)
    (arena,) = allocations(marked)
    assert int(arena.var.ty.data_alignment) == 128
    assert "tl.source_block_alloc" not in (dict(arena.value.attrs) if arena.value.attrs is not None else {})


def test_valid_source_origin_keeps_the_volatile_attribute():
    marked = transform.MergeUBAllocations(disable_reuse=True)(
        _module({"tl.source_block_alloc": ir.IntImm("int32", 1), "tirx.volatile": True})
    )
    (arena,) = allocations(marked)
    assert bool(arena.value.attrs["tirx.volatile"])
    assert "tl.source_block_alloc" not in arena.value.attrs


@pytest.mark.parametrize("value", [ir.IntImm("int32", 0), ir.IntImm("int32", 2), "invalid"])
def test_invalid_source_origin_is_not_discarded(value):
    with pytest.raises(ValueError, match="must be the source allocation marker"):
        transform.MergeUBAllocations()(_module({"tl.source_block_alloc": value}))


def test_unrelated_allocation_metadata_still_requires_its_consumer():
    with pytest.raises(ValueError, match="must consume allocation annotation unexplained"):
        transform.MergeUBAllocations()(_module({"tl.source_block_alloc": ir.IntImm("int32", 1), "unexplained": 1}))


def test_unmerged_allocation_retains_origin_until_finalization():
    marked = transform.MergeUBAllocations()(_module({"tl.source_block_alloc": ir.IntImm("int32", 1)}, count=1))
    (original,) = allocations(marked)
    assert int(original.value.attrs["tl.source_block_alloc"]) == 1
    finalized = tilelang.transform.FinalizeAnnotations()(marked)
    (result,) = allocations(finalized)
    assert "tl.source_block_alloc" not in result.value.attrs
    assert tvm_ffi.structural_equal(original.var.ty, result.var.ty)
