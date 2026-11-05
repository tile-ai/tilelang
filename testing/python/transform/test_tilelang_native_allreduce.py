"""TileLang allreduce policy reuses native TIRx reduction and definition repair."""

import json

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.script.ir_builder import IRBuilder

import tilelang.language as T
from tilelang.testing.ir import collect_calls, collect_nodes


def make_reduction(
    dtype="float32", threads=32, groups=1, dynamic_allocs=0, valid=None, maximum=False, with_user_barrier=False, reduce_axis="x"
):
    """Construct a real TileLang tensor/launch boundary with native allreduce."""
    source = ir.Var("source", T.Tensor((groups, threads), dtype).as_type())
    output = ir.Var("output", T.Tensor((groups, threads), dtype).as_type())
    combine = ir.LambdaExpr([dtype, dtype], (lambda a, b: (prim.max(a, b),)) if maximum else (lambda a, b: (a + b,)))
    identity = prim.min_value(ty=dtype) if maximum else ir.const(0, dtype)
    group_axis = "y" if reduce_axis == "x" else "x"
    with (
        IRBuilder() as builder,
        T.launch_thread("blockIdx.x", 1),
        T.launch_thread(f"threadIdx.{group_axis}", groups) as group,
        T.launch_thread(f"threadIdx.{reduce_axis}", threads) as thread,
    ):
        for _ in range(dynamic_allocs):
            T.alloc_shared((16,), "float32", scope="shared.dyn")
        reduced = T.alloc_local((1,), dtype)
        T.evaluate(
            tirx.gpu_thread_allreduce(
                combine,
                (identity,),
                (ir.TensorLoad(source, [group, thread]),),
                ir.const(True) if valid is None else thread < valid,
                (ir.TensorLoad(reduced, [ir.const(0)]),),
                (thread,),
            )
        )
        if with_user_barrier:
            T.evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("warp")]))
        T.buffer_store(output, ir.TensorLoad(reduced, [ir.const(0)]), [group, thread])
    function = tirx.Function([source, output], builder.get(), ret_type=ir.PrimType("void"), attrs=ir.DictAttrs({"global_symbol": "main"}))
    function = tvm_ffi.get_global_func("tl.transform.LowerFrontendScopes")()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(function)
    return function


def make_tuple_reduction(threads=32):
    source = ir.Var("source", T.Tensor((threads,), "int8").as_type())
    sum_output = ir.Var("sum_output", T.Tensor((threads,), "int32").as_type())
    max_output = ir.Var("max_output", T.Tensor((threads,), "int8").as_type())
    combine = ir.LambdaExpr(["int32", "int8", "int32", "int8"], lambda a, b, c, d: (a + c, prim.max(b, d)))
    with IRBuilder() as builder, T.launch_thread("blockIdx.x", 1), T.launch_thread("threadIdx.x", threads) as thread:
        sum_value = T.alloc_local((1,), "int32")
        max_value = T.alloc_local((1,), "int8")
        value = ir.TensorLoad(source, [thread])
        T.evaluate(
            tirx.gpu_thread_allreduce(
                combine,
                (ir.const(0), ir.const(-128, "int8")),
                (prim.Cast("int32", value), value),
                ir.const(True),
                (ir.TensorLoad(sum_value, [ir.const(0)]), ir.TensorLoad(max_value, [ir.const(0)])),
                (thread,),
            )
        )
        T.buffer_store(sum_output, ir.TensorLoad(sum_value, [ir.const(0)]), [thread])
        T.buffer_store(max_output, ir.TensorLoad(max_value, [ir.const(0)]), [thread])
    function = tirx.Function(
        [source, sum_output, max_output], builder.get(), ret_type=ir.PrimType("void"), attrs=ir.DictAttrs({"global_symbol": "main"})
    )
    return tvm_ffi.get_global_func("tl.transform.LowerFrontendScopes")()(ir.IRModule({"main": function}))["main"]


def lower(function, target="cuda", *, use_native_default=False):
    backend = tvm_tirx.target.Target(target)
    function = function.with_attr("target", backend)
    original = ir.save_json(function)
    if use_native_default:
        transform = tirx.transform.LowerThreadAllreduce()
    else:
        transform = tvm_ffi.get_global_func("tl.transform.LowerThreadAllreduce")()
    result = transform(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == original
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    assert not collect_calls(result, op="tirx.gpu_thread_allreduce")
    return result


@pytest.mark.parametrize("dtype", ["float32", "int8", "int64", "float16"])
@pytest.mark.parametrize("threads,groups", [(1, 1), (3, 1), (16, 3), (32, 2), (48, 1), (64, 1), (100, 1), (128, 1)])
def test_native_allreduce_definitions_types_and_idempotence(dtype, threads, groups):
    original = make_reduction(dtype, threads, groups)
    result = lower(original)
    assert tvm_ffi.structural_equal(lower(result), result)
    assert all(call.ty == ir.PrimType("void") for call in collect_calls(result, op="tl.sync_warp"))
    assert all(bind.var.ty == bind.value.ty for bind in collect_nodes(result, ir.Bind))


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_scratch_policy_preserves_input_scopes(count):
    original = make_reduction("int8", 32, dynamic_allocs=count)
    result = lower(original)
    input_allocs = [
        bind for bind in collect_nodes(original, ir.Bind) if tirx.is_tensor_var(bind.var) and str(bind.var.ty.scope) == "shared.dyn"
    ]
    assert len(input_allocs) == count
    allocations = [
        bind for bind in collect_nodes(result, ir.Bind) if isinstance(bind.value, ir.Call) and bind.value.op.name == "tirx.alloc_tensor"
    ]
    scratch = [bind.var for bind in allocations if bind.var.name.startswith("red_buf")]
    assert scratch
    expected = "shared.dyn" if count > 1 else "shared"
    assert all(str(var.ty.scope) == expected for var in scratch)
    assert all(any(bind.var.same_as(old.var) for bind in allocations) for old in input_allocs)
    assert all(bind.value.attrs["tirx.volatile"] for bind in allocations if bind.var in scratch)


def test_cuda_warp_mask_is_computed_before_its_participant_guard():
    result = lower(make_reduction("int8", 32, with_user_barrier=True))
    masks = collect_nodes(result, ir.Bind)
    active = [bind for bind in masks if bind.var.name == "allreduce_active_mask"]
    participant = [bind for bind in masks if bind.var.name == "allreduce_participant_mask"]
    assert len(active) == len(participant) == 1
    assert str(active[0].var.ty.dtype) == str(participant[0].var.ty.dtype) == "uint32"
    ballots = collect_calls(result, op="tl.ballot_sync")
    assert len(ballots) == 1 and ballots[0].args[0].same_as(active[0].var)
    barriers = collect_calls(result, op="tl.sync_warp")
    assert barriers and len(barriers) % 2 == 0
    assert all(call.args[0].same_as(participant[0].var) for call in barriers)
    # The callback is restricted to generated warp-memory stages.
    original = [call for call in collect_calls(result, op="tirx.gpu_storage_sync") if str(call.args[0].value) == "warp"]
    assert len(original) == 1
    containing = [seq for seq in collect_nodes(result, ir.SeqStmt) if any(item.same_as(participant[0]) for item in seq.seq)]
    assert len(containing) == 1
    sequence = list(containing[0].seq)
    position = next(i for i, item in enumerate(sequence) if item.same_as(participant[0]))
    assert position > 0 and sequence[position - 1].same_as(active[0])
    assert isinstance(sequence[position + 1], ir.If)
    assert tvm_ffi.structural_equal(sequence[position + 1].condition, ballots[0].args[1])


def test_rocm_wave64_keeps_full_participation_mask():
    target = {"kind": "rocm", "thread_warp_size": 64, "max_num_threads": 1024}
    result = lower(make_reduction("int8", 64), target)
    masks = [bind for bind in collect_nodes(result, ir.Bind) if bind.var.name.startswith("allreduce_")]
    assert len(masks) == 2
    assert all(str(bind.var.ty.dtype) == "uint64" for bind in masks)


def test_ascend_default_target_uses_block_memory_reduction():
    # The actual Ascend Target schema has no warp-size option. Its inherited
    # default is one, so allreduce must not invent CUDA-style warp shuffles.
    result = lower(make_reduction("int8", 64), "ascend")
    assert not collect_calls(result, op="tl.sync_warp")
    assert not collect_calls(result, op="tirx.gpu_warp_shuffle")
    assert collect_calls(result, op="tirx.gpu_storage_sync")


@pytest.mark.parametrize("target", ["metal", "webgpu"])
def test_without_warp_callback_memory_reduction_uses_uniform_block_barriers(target):
    result = lower(make_reduction("int8", 32, dynamic_allocs=2), target)
    assert not collect_calls(result, op="tl.sync_warp")
    for call in collect_calls(result, op="tirx.gpu_storage_sync"):
        assert call.args[0].value == "shared.dyn"
    for branch in collect_nodes(result, ir.If):
        assert not collect_calls(branch, op="tirx.gpu_storage_sync")


def test_native_default_still_uses_static_shared_memory():
    result = lower(make_reduction("int8", 32, dynamic_allocs=2), use_native_default=True)
    scratch = [bind.var for bind in collect_nodes(result, ir.Bind) if tirx.is_tensor_var(bind.var) and bind.var.name.startswith("red_buf")]
    assert scratch and all(str(var.ty.scope) == "shared" for var in scratch)
    assert not collect_calls(result, op="tl.sync_warp")


def test_omitted_allocation_annotations_use_the_same_scratch_contract():
    original = make_reduction("int8", 32, dynamic_allocs=2)

    def with_allocation_annotations(annotations):
        def replace(call):
            if call.op.name == "tirx.alloc_tensor":
                return ir.Call(call.op, call.args, attrs=annotations, ty=call.ty, ty_args=call.ty_args, loc=call.loc)
            return call

        return tvm_ffi.structural_map(original, (ir.Call, replace), order="post")

    # Build two equivalent raw-IR inputs. The facade can add nonempty
    # source-allocation metadata, so only dropping an already-empty dict would
    # not exercise the omitted-attrs contract at all.
    omitted = with_allocation_annotations(None)
    explicit = with_allocation_annotations({})
    assert all(call.attrs is None for call in collect_calls(omitted, op="tirx.alloc_tensor"))
    before = ir.save_json(omitted)
    result = lower(omitted)
    assert ir.save_json(omitted) == before
    ir.assert_structural_equal(result, lower(explicit))


@pytest.mark.parametrize("scope", ["local", "global", "shared.custom"])
def test_tilelang_allreduce_does_not_accept_arbitrary_scratch_scopes(scope):
    # Scratch policy is selected by TileLang from actual allocations. The
    # removed fork-only factory is not part of the direct-upstream contract.
    with pytest.raises(TypeError):
        tvm_ffi.get_global_func("tl.transform.LowerThreadAllreduce")(scope)


def test_allreduce_keeps_global_callees_and_module_context_visible_to_instruments():
    backend = tvm_tirx.target.Target("cuda")
    helper_var = ir.GlobalVar("helper")
    helper = tirx.Function([], [ir.Evaluate(ir.const(0))], ret_type=ir.PrimType("void"), attrs=ir.DictAttrs({"target": backend}))
    main = make_reduction("int8").with_attr("target", backend)
    main = main.with_body([ir.Evaluate(ir.Call(helper_var, [], ty="void")), *main.body.seq])
    module = ir.IRModule({"main": main, helper_var: helper}, attrs={"allreduce_sentinel": "keep"})
    assert tirx.analysis.check_well_formed(module)
    before = ir.save_json(module)
    observed = []

    @ir.transform.pass_instrument
    class CheckModuleContext:
        def run_before_pass(self, current, info):
            observed.append(info.name)
            assert {var.name_hint for var in current.functions} == {"main", "helper"}, "Pass instrumentation must see the complete module"
            assert current.attrs["allreduce_sentinel"] == "keep"

    with ir.transform.PassContext(instruments=[CheckModuleContext()]):
        result = tvm_ffi.get_global_func("tl.transform.LowerThreadAllreduce")()(module)
    assert observed == ["tl.LowerThreadAllreduce"]
    assert ir.save_json(module) == before
    assert tvm_ffi.structural_equal(result[helper_var], helper)
    assert result["main"].body[0].value.op.same_as(helper_var)
    assert tirx.analysis.check_well_formed(result)


@pytest.mark.parametrize("threads", [3, 32, 64])
def test_tuple_reduction_repairs_both_output_definitions(threads):
    result = lower(make_tuple_reduction(threads))
    assert len(result.params) == 3
    assert all(bind.var.ty == bind.value.ty for bind in collect_nodes(result, ir.Bind))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,threads,groups,valid,maximum",
    [
        ("int8", 32, 1, None, False),
        ("int8", 64, 1, 61, True),
        ("int8", 48, 1, None, False),
        ("float32", 32, 2, None, False),
        ("float32", 64, 1, 61, True),
        ("float32", 3, 1, 0, False),
        ("int64", 16, 3, None, False),
        ("int8", 16, 3, None, True),
    ],
)
def test_tilelang_allreduce_cuda_executes(dtype, threads, groups, valid, maximum, tmp_path):
    from testing.python.cuda.test_native_codegen_cuda import build_cuda

    function = lower(make_reduction(dtype, threads, groups, valid=valid, maximum=maximum)).without_attr("target")
    module, source = build_cuda(function, tmp_path)
    rng = np.random.default_rng(373)
    values = rng.integers(-3, 0, size=(groups, threads)).astype(dtype)
    selected = values if valid is None else values[:, :valid]
    expected = selected.max(axis=-1) if maximum else selected.sum(axis=-1).astype(dtype)
    expected = np.repeat(expected[:, None], threads, axis=1)
    device = tvm_tirx.cuda(0)
    src = tvm_tirx.runtime.tensor(values, device)
    output = tvm_tirx.runtime.tensor(np.zeros_like(values), device)
    for _ in range(10):
        module(src, output)
        np.testing.assert_array_equal(output.numpy(), expected)
    if dtype == "int8" and threads in (32, 48, 64):
        assert "__ballot_sync" in source and "__syncwarp" in source
    (tmp_path / "oracle.json").write_text(
        json.dumps(
            {
                "seed": 373,
                "dtype": dtype,
                "input": values.tolist(),
                "expected": expected.tolist(),
                "actual": output.numpy().tolist(),
                "runs": 10,
            }
        )
    )


@pytest.mark.gpu
def test_noncontiguous_reduction_axis_executes(tmp_path):
    from testing.python.cuda.test_native_codegen_cuda import build_cuda

    function = lower(make_reduction("int8", 16, 3, maximum=True, reduce_axis="y")).without_attr("target")
    module, _ = build_cuda(function, tmp_path)
    values = (np.arange(48) % 5 - 8).reshape(3, 16).astype("int8")
    expected = np.repeat(values.max(axis=1)[:, None], 16, axis=1)
    source = tvm_tirx.runtime.tensor(values, tvm_tirx.cuda(0))
    output = tvm_tirx.runtime.tensor(np.zeros_like(values), tvm_tirx.cuda(0))
    for _ in range(10):
        module(source, output)
        np.testing.assert_array_equal(output.numpy(), expected)
    (tmp_path / "oracle.json").write_text(
        json.dumps({"input": values.tolist(), "expected": expected.tolist(), "actual": output.numpy().tolist(), "runs": 10})
    )


@pytest.mark.gpu
def test_tuple_allreduce_executes_with_two_result_types(tmp_path):
    from testing.python.cuda.test_native_codegen_cuda import build_cuda

    function = lower(make_tuple_reduction()).without_attr("target")
    module, _ = build_cuda(function, tmp_path)
    values = (np.arange(32) % 7 - 10).astype("int8")
    source = tvm_tirx.runtime.tensor(values, tvm_tirx.cuda(0))
    sums = tvm_tirx.runtime.tensor(np.zeros(32, dtype="int32"), tvm_tirx.cuda(0))
    maxima = tvm_tirx.runtime.tensor(np.zeros(32, dtype="int8"), tvm_tirx.cuda(0))
    for _ in range(10):
        module(source, sums, maxima)
        np.testing.assert_array_equal(sums.numpy(), np.full(32, values.sum(dtype="int32")))
        np.testing.assert_array_equal(maxima.numpy(), np.full(32, values.max()))
    (tmp_path / "oracle.json").write_text(
        json.dumps({"input": values.tolist(), "sum": sums.numpy().tolist(), "max": maxima.numpy().tolist(), "runs": 10})
    )
