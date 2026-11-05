"""Undeclared launch dimensions retain their typed region until a backend opts in."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang


def _fixture(tags=("blockIdx.x", "cthread"), *, used_thread=False):
    params = [ir.Var(name, "int64") for name in ("block", "subblock")]
    params += [ir.Var(name, "int32") for name in ("tx", "ty", "tz")]
    hint = ir.Var("hint", "int64")
    args = [
        ir.Tuple([ir.const(2, "int64"), ir.const(1, "int64")]),
        ir.Tuple([ir.const(11, "int64"), ir.const(9, "int64")]),
        ir.Tuple([ir.const(3, "int64"), ir.const(2, "int64")]),
        ir.Tuple([ir.const(32), ir.const(2), ir.const(1)]),
        ir.Tuple([hint]),
    ]
    attrs = {
        "schema_version": 1,
        "grid_tags": list(tags),
        "thread_tags": ["threadIdx.x", "threadIdx.y", "threadIdx.z"],
        "threads_was_specified": True,
        "hint_schema": ["map", [["str", "dynamic_hint"], ["expr", 0]]],
    }
    expression = params[0] * 16 + params[1]
    if used_thread:
        expression += ir.prim.Cast(ir.PrimType("int64"), params[2])
    region = ir.RegionStmt("tl.launch_kernel", args, params, attrs, [ir.Evaluate(expression)])
    return tirx.Function([hint], [region]).with_attr("global_symbol", "main"), region, params


def _lower(function, *, tags=(), grid=True, threads=False):
    transform = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=grid,
        lower_thread_binding=threads,
        default_threads=[128, 1, 1],
        launch_dim_tags=list(tags),
    )
    return transform(ir.IRModule({"main": function}))["main"]


def _regions(value, name):
    result = []
    tvm_ffi.structural_walk(value, (ir.RegionStmt, lambda node: result.append(node) if node.op.name == name else None))
    return result


@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("threads", [False, True])
def test_partial_launch_retains_unrecognized_tail_and_can_finish_later(grid, threads):
    function, original, params = _fixture(used_thread=threads)
    original_json = ir.save_json(function)
    partial = _lower(function, grid=grid, threads=threads)
    (tail,) = _regions(partial, "tl.launch_kernel")
    assert list(tail.attrs["grid_tags"]) == ["cthread"]
    assert all(left.same_as(right) for left, right in zip(tail.body_params, params[1:]))
    assert [int(tail.args[i].fields[0]) for i in range(3)] == [1, 9, 2]
    assert tail.body.same_as(original.body)
    assert tail.args[3].same_as(original.args[3])
    assert tail.args[4].same_as(original.args[4])
    assert tirx.analysis.check_well_formed(partial)
    roundtrip = ir.load_json(ir.save_json(partial))
    completed = _lower(roundtrip, grid=grid, threads=threads, tags=["cthread"])
    direct = _lower(function, grid=grid, threads=threads, tags=["cthread"])
    assert not _regions(completed, "tl.launch_kernel")
    ir.assert_structural_equal(completed, direct)
    assert tirx.analysis.check_well_formed(completed)
    assert ir.save_json(function) == original_json


@pytest.mark.parametrize("grid", [False, True])
def test_unrecognized_outer_dimension_preserves_whole_launch(grid):
    function, region, _ = _fixture(tags=("cthread", "blockIdx.x"))
    partial = _lower(function, grid=grid)
    assert partial.body[0].same_as(region)
    assert not _regions(partial, "tirx.launch_thread")
    assert tirx.analysis.check_well_formed(partial)


def test_thread_use_remains_bound_until_backend_finishes_the_launch():
    function, _, _ = _fixture(used_thread=True)
    partial = _lower(function)
    assert tirx.analysis.check_well_formed(partial)
    with pytest.raises(ValueError, match="no SIMT threads"):
        _lower(partial, tags=["cthread"])
