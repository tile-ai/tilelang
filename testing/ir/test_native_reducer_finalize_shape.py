"""Finalize validates full logical shapes before publishing reducer results."""

from tilelang.testing.packed import compile_packed_function

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from test_native_reducer_contracts import (
    allocate,
    annotation,
    function,
    integer,
    lower,
    objects,
    region,
    run_pass,
    tensor,
    tileop,
)


def finalize_fixture(
    reducer_shape,
    destination_shape,
    *,
    params=(),
    reducer_dtype="float32",
    destination_dtype="float32",
    target="llvm",
    prefix=(),
    epoch_condition=None,
    sliced=False,
    output=None,
):
    acc = tensor("acc", reducer_shape, reducer_dtype, "local.reducer")
    dst = tensor("dst", destination_shape, destination_dtype, "local.fragment")
    source_region, destination_region = region(acc), region(dst)
    if sliced:
        # V2 finalizes the whole tensor handles; matching slice extents do
        # not establish that their full logical shapes agree.
        source_region = ir.TensorRegion(acc, [ir.Range(0, 1)], ir.TensorRegionType())
        destination_region = ir.TensorRegion(dst, [ir.Range(0, 1)], ir.TensorRegionType())
    metadata = {"reducer_info_v2": {acc: {"op": "sum"}}}
    if target == "cuda":
        from tilelang.layout import Fragment

        def local_index(*indices):
            result = 0
            for extent, index in zip(destination_shape, indices):
                result = result * extent + index
            return result

        metadata["layout_map"] = {
            dst: Fragment(destination_shape, forward_thread_fn=lambda *indices: indices[-1], forward_index_fn=local_index, replicate=128)
        }
    body = [allocate(acc), allocate(dst), annotation(metadata), *prefix]
    epoch = [
        tileop("reducer_init", region(acc), ir.FloatImm(reducer_dtype, 5.0)),
        tileop("finalize_reducer_v2", source_region, destination_region),
    ]
    if output is not None:
        epoch.append(tileop("copy", region(dst), region(output)))
    body.extend(epoch if epoch_condition is None else [ir.If(epoch_condition, epoch, None)])
    if target == "cuda":
        thread = ir.Var("threadIdx_x", "int32")
        body = [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(128)], [thread], {}, body)]
    return function(list(params), body, target=target)


def shape_checks(value):
    return [
        statement
        for statement in objects(value, ir.AssertStmt)
        if "finalize_reducer:" in "".join(part.value for part in statement.message_parts)
    ]


def lower_sequential_cpu(value):
    # The CPU materializer explicitly supports sequential dynamic storage
    # without fragment plans. Exercise that boundary directly: the separate
    # layout cost model requires constant fragment dimensions even on CPU.
    stages = {}
    for name in ("CanonicalizeLegacyReducer", "VerifyReducerEpoch", "ReducerPlanAndMaterialize", "LowerTileOp", "VerifyReducerConsumed"):
        value = run_pass(name, value)
        stages[name] = value
        assert tirx.analysis.check_well_formed(value), name
        assert not tirx.analysis.undefined_vars(value), name
    return value, stages


@pytest.mark.parametrize("target", ["llvm", "cuda"])
@pytest.mark.parametrize(
    "source_shape,destination_shape,axis",
    [([2], [3], 0), ([3], [2], 0), ([2, 3], [3, 2], 0), ([2, 2], [2, 3], 1), ([2, 3], [2, 2], 1)],
)
def test_finalize_rejects_each_mismatched_logical_extent(source_shape, destination_shape, axis, target):
    value = finalize_fixture(source_shape, destination_shape, target=target)
    with pytest.raises(ValueError, match=rf"finalize_reducer:.*logical shape \(axis {axis}\)"):
        lower(value)


@pytest.mark.parametrize("source_extent,destination_extent", [(2, 3), (3, 2)])
def test_matching_slices_do_not_hide_mismatched_tensor_shapes(source_extent, destination_extent):
    value = finalize_fixture([source_extent], [destination_extent], sliced=True)
    with pytest.raises(ValueError, match=r"finalize_reducer:.*logical shape \(axis 0\)"):
        lower(value)


@pytest.mark.parametrize("shape", [[], [2], [2, 3]])
def test_equal_static_shapes_need_no_runtime_check(shape):
    result, stages, _ = lower(finalize_fixture(shape, shape))
    assert not shape_checks(stages["ReducerPlanAndMaterialize"])
    assert not shape_checks(result)


@pytest.mark.parametrize("equivalent_expression", [False, True])
def test_equal_dynamic_shapes_need_no_runtime_check(equivalent_expression):
    n = ir.Var("n", "int32")
    source_extent, destination_extent = (n + 1, 1 + n) if equivalent_expression else (n, n)
    value = finalize_fixture([source_extent], [destination_extent], params=[n])
    result, stages = lower_sequential_cpu(value)
    assert not shape_checks(stages["ReducerPlanAndMaterialize"])
    assert not shape_checks(result)


def test_provably_unequal_dynamic_shapes_are_rejected():
    n = ir.Var("n", "int32")
    with pytest.raises(ValueError, match=r"finalize_reducer:.*logical shape \(axis 0\)"):
        lower_sequential_cpu(finalize_fixture([n + 1], [n], params=[n]))


@pytest.mark.parametrize("fact", ["assert", "branch", "expired_branch"])
def test_shape_proof_uses_only_current_analyzer_facts(fact):
    m, n = ir.Var("m", "int32"), ir.Var("n", "int32")
    prefix = []
    condition = None
    if fact == "assert":
        prefix = [ir.AssertStmt(m == n, ir.StringImm("ValueError"), [ir.StringImm("input shapes differ")])]
    elif fact == "branch":
        condition = m == n
    else:
        prefix = [ir.If(m == n, [ir.Evaluate(integer(0))], None)]
    value = finalize_fixture([m], [n], params=[m, n], prefix=prefix, epoch_condition=condition)
    result, stages = lower_sequential_cpu(value)
    expected = 1 if fact == "expired_branch" else 0
    assert len(shape_checks(stages["ReducerPlanAndMaterialize"])) == expected
    assert len(shape_checks(result)) == expected


@pytest.mark.parametrize("comparison", ["not_equal", "smaller"])
def test_branch_proves_dynamic_shapes_unequal(comparison):
    m, n = ir.Var("m", "int32"), ir.Var("n", "int32")
    condition = m != n if comparison == "not_equal" else m < n
    value = finalize_fixture([m], [n], params=[m, n], epoch_condition=condition)
    with pytest.raises(ValueError, match=r"finalize_reducer:.*logical shape \(axis 0\)"):
        lower_sequential_cpu(value)


@pytest.mark.parametrize("problem", ["dtype", "rank"])
def test_materialization_checks_dtype_and_rank_without_layout_inference(problem):
    value = finalize_fixture([2], [2, 1] if problem == "rank" else [2], destination_dtype="int32" if problem == "dtype" else "float32")
    # The shared tile-op constructor already rejects these before planning.
    match = "destination dtype" if problem == "dtype" else "logical shape"
    with pytest.raises(RuntimeError, match=match):
        run_pass("ReducerPlanAndMaterialize", value)


def compile_cpu(value):
    result, stages = lower_sequential_cpu(value.with_attr("target", tvm_tirx.target.Target("c")))
    module = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")("c")(ir.IRModule({"main": result}))
    executable = compile_packed_function(module, tilelang_owned=False)
    return executable, stages


def test_unknown_dynamic_shapes_execute_equal_and_reject_both_mismatch_directions():
    m, n = ir.Var("m", "int32"), ir.Var("n", "int32")
    output = tensor("output", [n], "float32", "global")
    executable, stages = compile_cpu(finalize_fixture([m], [n], params=[m, n, output], output=output))
    assert len(shape_checks(stages["ReducerPlanAndMaterialize"])) == 1
    for source_extent, destination_extent in [(2, 2), (2, 3), (3, 2), (4, 4)]:
        before = np.full(destination_extent, -37.0, dtype="float32")
        actual = tvm_tirx.runtime.tensor(before)
        if source_extent == destination_extent:
            executable(source_extent, destination_extent, actual)
            np.testing.assert_array_equal(actual.numpy(), np.full(destination_extent, 5.0, dtype="float32"))
        else:
            with pytest.raises(ValueError, match=r"finalize_reducer:.*logical shape \(axis 0\)"):
                executable(source_extent, destination_extent, actual)
            # A rejected finalize publishes no partial output, in either
            # the out-of-bounds or the silent-truncation direction.
            np.testing.assert_array_equal(actual.numpy(), before)


def test_equal_symbolic_expression_executes_without_guard():
    n = ir.Var("n", "int32")
    output = tensor("output", [1 + n], "float32", "global")
    executable, stages = compile_cpu(finalize_fixture([n + 1], [1 + n], params=[n, output], output=output))
    assert not shape_checks(stages["ReducerPlanAndMaterialize"])
    actual = tvm_tirx.runtime.tensor(np.zeros(4, dtype="float32"))
    executable(3, actual)
    np.testing.assert_array_equal(actual.numpy(), np.full(4, 5.0, dtype="float32"))
