import importlib.util
import re
from pathlib import Path

import pytest
import torch

import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def load_factory(relative_path, name="make_reducer"):
    path = Path(__file__).resolve().parents[3] / relative_path
    spec = importlib.util.spec_from_file_location("reduction_aware", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def make_reducer(**kwargs):
    return load_factory("maint/layout_inference/cases/reduction_aware.py")(**kwargs)


def make_communication_reducer(**kwargs):
    return load_factory("maint/layout_inference/cases/reduction_aware.py", "make_communication_reducer")(**kwargs)


def make_shared_reducer(**kwargs):
    return load_factory("maint/layout_inference/cases/reduction_aware_shared.py")(**kwargs)


def make_weighted_pooling(**kwargs):
    return load_factory("maint/layout_inference/bench_reduction_aware.py", "make_weighted_pooling")(**kwargs)


def reducer_candidate_costs(diagnostics):
    """Exclude unrelated register-count components with zero execution cost."""
    costs = [
        {name: int(value) for name, value in re.findall(r"(\w+)=(-?\d+)", line)}
        for line in diagnostics.splitlines()
        if "[ReducerVectorPlan]" in line
    ]
    return [cost for cost in costs if cost["execution"] > 0]


def infer(factory=make_reducer, model=None, target=None, pass_configs=None, **kwargs):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"} if target is None else target)
    configs = dict(pass_configs or {})
    if model is not None:
        configs["tl.layout_cost_model"] = model
    with target, tl.transform.PassContext(config=configs):
        module = tvm.IRModule({"main": factory(**kwargs)})
        module = tvm.tirx.transform.BindTarget(target)(module)
        module = tl.transform.MaterializeKernelLaunch()(module)
        module = tl.transform.LayoutInference()(module)
    layouts = {}

    def collect(node):
        if isinstance(node, tvm.tirx.SBlock):
            for buffer, layout in node.annotations.get("layout_map", {}).items():
                layouts[buffer.name] = layout

    tvm.tirx.stmt_functor.post_order_visit(module["main"].body, collect)
    return module["main"], layouts


def reducer_cost(function):
    return {buffer.name: dict(cost) for buffer, cost in tvm.get_global_func("tl.analysis.ReducerCost")(function).items()}


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_column_reduction_avoids_communication():
    function, layouts = infer()
    assert int(layouts["acc"].combine_size) == 1
    cost = reducer_cost(function)["acc"]
    assert cost["known"]
    assert list(cost["steps"]) == []
    assert cost["barriers"] == cost["shuffle_issues"] == 0


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_default_intermediate_width_beats_both_endpoints():
    _, automatic = infer(rows=4, columns=256)
    _, intermediate = infer(rows=4, columns=256, width=2)
    _, scalar = infer(rows=4, columns=256, width=1)
    _, native = infer(rows=4, columns=256, width=4)
    assert automatic["values"].is_equal(intermediate["values"])
    assert not automatic["values"].is_equal(scalar["values"])
    assert not automatic["values"].is_equal(native["values"])
    assert int(automatic["acc"].combine_size) == 1


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_scalar_fp32_arithmetic_is_not_discounted_by_layout_width():
    native, _ = infer(width=4)
    scalar, _ = infer(width=1)
    native_cost = reducer_cost(native)["acc"]
    scalar_cost = reducer_cost(scalar)["acc"]
    assert native_cost["local_issues"] == scalar_cost["local_issues"] == 8
    assert [tuple(map(int, step)) for step in native_cost["steps"]] == [(128, 32)]
    assert native_cost["barriers"] == 16
    assert native_cost["issue_cost"] == 544
    assert scalar_cost["issue_cost"] == 8


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_equal_register_scores_account_for_communication(capfd):
    scores, features = {}, {}
    for within_warp in (False, True):
        function, _ = infer(
            factory=make_communication_reducer,
            within_warp=within_warp,
            pass_configs={"tl.enable_reducer_plan_verbose": True},
        )
        candidates = reducer_candidate_costs(capfd.readouterr().err)
        assert candidates and all(cost["known"] and cost["bank_conflict_free"] for cost in candidates)
        scores[within_warp] = min(candidates, key=lambda cost: cost["total"])
        features[within_warp] = reducer_cost(function)["acc"]
    for score in scores.values():
        assert score["regs"] == 16 and score["spill"] == 0
    block, warp = features[False], features[True]
    assert block["barriers"] == 16 and block["shuffle_issues"] == 0
    assert warp["barriers"] == 0 and warp["shuffle_issues"] == 8
    assert block["local_issues"] == warp["local_issues"]
    communication_delta = (block["issue_cost"] - warp["issue_cost"]) * 128 * 16
    assert communication_delta > 0
    assert scores[False]["execution"] - scores[True]["execution"] == communication_delta
    assert scores[False]["total"] - scores[True]["total"] == communication_delta


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("within_warp", [False, True])
def test_equal_register_communication_kernel_correctness(within_warp):
    kernel = tl.compile(make_communication_reducer(within_warp=within_warp), out_idx=-1, target="cuda")
    inputs = torch.randn((8, 128), device="cuda")
    torch.testing.assert_close(kernel(inputs), inputs.sum(dim=0), atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("queries", [1, 32])
def test_weighted_pooling_selects_less_communication_at_equal_register_cost(queries, capfd):
    features, scores, layouts = {}, {}, {}
    for width in (4, 1, None):
        function, layouts[width] = infer(
            factory=make_weighted_pooling,
            groups=1,
            queries=queries,
            width=width,
            pass_configs={"tl.enable_reducer_plan_verbose": True},
        )
        candidates = reducer_candidate_costs(capfd.readouterr().err)
        assert candidates and all(cost["known"] and cost["bank_conflict_free"] for cost in candidates)
        scores[width] = min(candidates, key=lambda cost: cost["total"])
        features[width] = reducer_cost(function)["acc"]
    assert scores[4]["regs"] == scores[1]["regs"] == scores[None]["regs"] == 18
    assert all(score["spill"] == 0 for score in scores.values())
    assert features[4]["local_issues"] == features[1]["local_issues"] == features[None]["local_issues"]
    assert features[4]["barriers"] == 16 * queries
    assert features[1]["barriers"] == features[None]["barriers"] == 0
    assert layouts[None]["values"].is_equal(layouts[1]["values"])
    assert scores[None]["total"] == scores[1]["total"] < scores[4]["total"]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("width", [4, 1, None])
def test_weighted_pooling_kernel_correctness(width):
    kernel = tl.compile(make_weighted_pooling(groups=4, queries=3, width=width), out_idx=-1, target="cuda")
    inputs = torch.randn((4, 8, 128), device="cuda")
    weights = torch.randn((4, 3, 8), device="cuda")
    expected = (weights.unsqueeze(-1) * inputs.unsqueeze(1)).sum(dim=2)
    torch.testing.assert_close(kernel(inputs, weights), expected, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_batch_shares_barriers_but_not_value_work():
    scalar, _ = infer(width=4)
    batched, _ = infer(width=4, batch=4)
    scalar_cost = reducer_cost(scalar)["acc"]
    batched_cost = reducer_cost(batched)["acc"]
    assert scalar_cost["barriers"] == 4 * batched_cost["barriers"]
    assert scalar_cost["combine_issues"] == batched_cost["combine_issues"]
    assert scalar_cost["shared_issues"] == batched_cost["shared_issues"]
    assert batched_cost["issue_cost"] == 160


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("width", [None, 1])
def test_local_complete_ignores_unused_batch(width):
    function, layouts = infer(width=width, batch=4)
    cost = reducer_cost(function)["acc"]
    assert cost["known"]
    assert int(layouts["acc"].combine_size) == 1
    assert list(cost["steps"]) == []


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_multiple_updates_and_serial_repeats_do_not_duplicate_finalize():
    baseline, _ = infer(width=4)
    repeated, _ = infer(width=4, updates=2, repeats=3)
    baseline_cost = reducer_cost(baseline)["acc"]
    repeated_cost = reducer_cost(repeated)["acc"]
    assert repeated_cost["local_issues"] == 6 * baseline_cost["local_issues"]
    assert repeated_cost["barriers"] == baseline_cost["barriers"]


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_reduction_axis_updates_remain_scalar():
    function, layouts = infer(rows=4, columns=256, total=True, width=4)
    cost = reducer_cost(function)["acc"]
    assert int(layouts["acc"].combine_size) == 128
    assert list(map(int, cost["vector_widths"])) == [1]
    assert cost["shuffle_issues"] > 0


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"width": 1}, 1),
        ({"width": 4}, 4),
        ({"rows": 4, "columns": 256}, 2),
        ({"rows": 4, "columns": 256, "total": True, "width": 4}, 1),
    ],
)
def test_predicted_update_width_matches_lowering(kwargs, expected):
    function, _ = infer(**kwargs)
    assert list(map(int, reducer_cost(function)["acc"]["vector_widths"])) == [expected]
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target, tl.transform.PassContext():
        artifact = tl.lower(make_reducer(**kwargs), target=target, enable_device_compile=False)
    widths = []

    def visit(node):
        if not isinstance(node, tvm.tirx.BufferStore) or node.buffer.name != "acc":
            return

        def visit_load(value):
            if isinstance(value, tvm.tirx.BufferLoad) and value.buffer.same_as(node.buffer):
                widths.append(node.value.dtype.lanes)

        tvm.tirx.stmt_functor.post_order_visit(node.value, visit_load)

    for device_function in artifact.device_mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(device_function.body, visit)
    assert max(widths) == expected


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_default_policy_is_reduction_aware_with_or_without_its_existing_name():
    _, register_count = infer(model="register-count", rows=4, columns=256)
    _, io_aware = infer(model="io-aware", rows=4, columns=256)
    _, automatic = infer(rows=4, columns=256)
    _, native = infer(rows=4, columns=256, width=4)
    assert register_count["values"].is_equal(automatic["values"])
    assert io_aware["values"].is_equal(native["values"])


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_unknown_serial_extent_retains_register_count_search():
    @T.prim_func
    def kernel(inputs: T.Tensor((4, 256), "float32"), output: T.Tensor((256,), "float32"), count: T.int32):
        with T.Kernel(1, threads=128):
            values = T.alloc_fragment((4, 256), "float32")
            acc = T.alloc_reducer((256,), "float32")
            result = T.alloc_fragment((256,), "float32")
            T.copy(inputs, values)
            T.reducer_init(acc)
            for _repeat in T.serial(count):
                for row, column in T.Parallel(4, 256):
                    T.reducer_update(acc[column], values[row, column])
            T.finalize_reducer(acc, result)
            T.copy(result, output)

    _, automatic = infer(factory=lambda: kernel)
    _, fallback = infer(rows=4, columns=256, width=1)
    assert automatic["values"].is_equal(fallback["values"])
    assert automatic["acc"].is_equal(fallback["acc"])


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("constraints", [{"width": 4}, {"pinned": True}, {"pin_loop": True}])
def test_explicit_constraints_are_preserved(constraints):
    factory = load_factory("maint/layout_inference/cases/reducer_scalar_candidates.py")
    _, automatic = infer(factory=factory, **constraints)
    _, pinned = infer(factory=factory, width=4)
    assert int(automatic["acc"].combine_size) == 4
    assert automatic["values"].is_equal(pinned["values"])


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_annotated_partial_is_authoritative():
    @T.prim_func
    def kernel(inputs: T.Tensor((32, 32), "float32"), output: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=128):
            values = T.alloc_fragment((32, 32), "float32")
            acc = T.alloc_reducer((32,), "float32")
            result = T.alloc_fragment((32,), "float32")
            T.annotate_layout({acc: T.PartialFragment((32,), forward_thread_fn=lambda row, replica: row * 4 + replica, replicate=4)})
            T.copy(inputs, values)
            T.reducer_init(acc)
            for row, column in T.Parallel(32, 32):
                T.reducer_update(acc[row], values[row, column])
            T.finalize_reducer(acc, result)
            T.copy(result, output)

    function, layouts = infer(factory=lambda: kernel)
    assert int(layouts["acc"].combine_size) == 4
    cost = reducer_cost(function)["acc"]
    assert cost["shared_rounds"] == 0
    assert cost["shuffle_issues"] == 2
    assert cost["issue_cost"] == 18


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("width", [None, 4])
def test_cost_tracks_materialization_wide_fallback(width):
    factory = load_factory(
        "testing/python/transform/test_tilelang_transform_reducer_scalar_candidates.py", "make_reducer_with_pinned_consumer"
    )
    function, _ = infer(factory=factory, width=width)
    cost = reducer_cost(function)["acc"]
    if width is None:
        assert cost["narrow"]
        assert list(cost["steps"]) == []
    else:
        assert not cost["narrow"]
        assert [tuple(map(int, step)) for step in cost["steps"]] == [(128, 1)]
        assert cost["barriers"] > 0


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_full_reduction_keeps_native_input_vectorization():
    factory = load_factory("maint/layout_inference/cases/reducer_scalar_candidates.py")
    _, automatic = infer(factory=factory, total=True)
    _, native = infer(factory=factory, total=True, pinned=True)
    assert int(automatic["acc"].combine_size) == 128
    assert automatic["values"].is_equal(native["values"])


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_bank_conflict_free_precedes_combined_cost(capfd):
    _, automatic = infer(factory=make_shared_reducer, pass_configs={"tl.enable_reducer_plan_verbose": True})
    costs = reducer_candidate_costs(capfd.readouterr().err)
    _, conflict_free = infer(factory=make_shared_reducer, width=4)
    _, conflicting = infer(factory=make_shared_reducer, width=16)
    assert automatic["values"].is_equal(conflict_free["values"])
    assert not automatic["values"].is_equal(conflicting["values"])
    measurable = [cost for cost in costs if cost["known"]]
    assert measurable and all(cost["spill"] == 0 for cost in measurable)
    free_costs = [cost["total"] for cost in measurable if cost["bank_conflict_free"]]
    conflicting_costs = [cost["total"] for cost in measurable if not cost["bank_conflict_free"]]
    assert free_costs and conflicting_costs
    assert min(conflicting_costs) < min(free_costs)


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("kwargs", [{}, {"repeats": 3}, {"width": 4, "batch": 4}])
def test_spill_execution_registers_share_total_cost(capfd, kwargs):
    infer(pass_configs={"tl.enable_reducer_plan_verbose": True}, **kwargs)
    costs = reducer_candidate_costs(capfd.readouterr().err)
    assert costs and all(cost["known"] for cost in costs)
    for cost in costs:
        assert cost["total"] == cost["spill"] + cost["execution"] + cost["regs"] * 128 * 4


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("width", [4, 16])
def test_bank_conflict_priority_preserves_explicit_layouts(capfd, width):
    _, layouts = infer(factory=make_shared_reducer, width=width, pass_configs={"tl.enable_reducer_plan_verbose": True})
    costs = reducer_candidate_costs(capfd.readouterr().err)
    expected = T.Fragment(
        (2048,),
        forward_thread_fn=lambda element: element // width % 128,
        forward_index_fn=lambda element: element // (128 * width) * width + element % width,
    )
    assert layouts["values"].is_equal(expected)
    assert costs and all(cost["known"] for cost in costs)
    assert all(cost["bank_conflict_free"] == (width == 4) for cost in costs)


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_unknown_shared_geometry_is_not_conflict_free(capfd):
    infer(factory=make_shared_reducer, swizzle=True, pass_configs={"tl.enable_reducer_plan_verbose": True})
    costs = reducer_candidate_costs(capfd.readouterr().err)
    assert costs
    assert all(not cost["known"] and not cost["bank_conflict_free"] for cost in costs)
    assert all(cost["total"] == -1 for cost in costs)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("width", [None, 4, 16])
def test_mixed_dtype_shared_reduction(width):
    kernel = tl.compile(
        make_shared_reducer(width=width),
        out_idx=-1,
        target="cuda",
        pass_configs={"tl.disable_vectorize_256": True},
    )
    inputs = torch.randn((2048,), device="cuda")
    masks = torch.randint(0, 2, (8, 2048), dtype=torch.int8, device="cuda")
    expected = (inputs * masks.to(torch.float32)).sum().reshape(1)
    torch.testing.assert_close(kernel(inputs, masks), expected, atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_arithmetic_packing_is_target_specific():
    function, _ = infer(width=4, target={"kind": "cuda", "arch": "sm_100a"})
    assert reducer_cost(function)["acc"]["local_issues"] == 4


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_forced_baseline_is_not_priced_as_local_complete():
    function, _ = infer(pass_configs={"tl.reducer_force_baseline": True})
    cost = reducer_cost(function)["acc"]
    assert not cost["narrow"]
    assert [tuple(map(int, step)) for step in cost["steps"]] == [(128, 1)]


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_vector_width_respects_other_updates_in_the_same_loop():
    @T.prim_func
    def kernel(inputs: T.Tensor((8, 128), "float32"), columns: T.Tensor((128,), "float32"), total: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            values = T.alloc_fragment((8, 128), "float32")
            column_acc = T.alloc_reducer((128,), "float32")
            total_acc = T.alloc_reducer((1,), "float32")
            column_result = T.alloc_fragment((128,), "float32")
            total_result = T.alloc_fragment((1,), "float32")
            T.copy(inputs, values)
            T.reducer_init(column_acc)
            T.reducer_init(total_acc)
            for row, column in T.Parallel(8, 128, coalesced_width=T.int32(4)):
                T.reducer_update(column_acc[column], values[row, column])
                T.reducer_update(total_acc[0], values[row, column])
            T.finalize_reducer(column_acc, column_result)
            T.finalize_reducer(total_acc, total_result)
            T.copy(column_result, columns)
            T.copy(total_result, total)

    function, _ = infer(factory=lambda: kernel)
    for cost in reducer_cost(function).values():
        assert list(map(int, cost["vector_widths"])) == [1]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("total", [False, True])
def test_direct_memory_reducer_updates(staged, total):
    outputs = 1 if total else 256

    @T.prim_func
    def program(inputs: T.Tensor((4, 256), "float32"), output: T.Tensor((outputs,), "float32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((4, 256), "float32")
            acc = T.alloc_reducer((outputs,), "float32")
            result = T.alloc_fragment((outputs,), "float32")
            if staged:
                T.copy(inputs, shared)
            T.reducer_init(acc)
            for row, column in T.Parallel(4, 256):
                T.reducer_update(acc[0 if total else column], shared[row, column] if staged else inputs[row, column])
            T.finalize_reducer(acc, result)
            T.copy(result, output)

    function, _ = infer(factory=lambda: program)
    cost = reducer_cost(function)["acc"]
    assert cost["known"]
    assert cost["local_issues"] == 8
    if total:
        assert list(map(int, cost["vector_widths"])) == [1]
    kernel = tl.compile(program, out_idx=-1, target="cuda")
    inputs = torch.randn((4, 256), device="cuda")
    expected = inputs.sum().reshape(1) if total else inputs.sum(dim=0)
    torch.testing.assert_close(kernel(inputs), expected, atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_rocm(support_required="compile-only")
def test_non_cuda_target_retains_register_count():
    target = {"kind": "hip", "mcpu": "gfx90a"}
    _, automatic = infer(target=target, rows=4, columns=256)
    _, baseline = infer(target=target, rows=4, columns=256, width=1)
    assert automatic["values"].is_equal(baseline["values"])


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 7, "seed": 2.5},
        {"updates": 2, "repeats": 3, "seed": 2.5},
        {"width": 4, "batch": 4},
        {"batch": 4},
        {"batch": 4, "seed": 2.5},
    ],
)
def test_seeded_repeated_and_batched_reductions(kwargs):
    kernel = tl.compile(make_reducer(**kwargs), out_idx=-1, target="cuda")
    inputs = torch.randn((kwargs.get("rows", 8), 128), device="cuda")
    expected = inputs.sum(dim=0) * kwargs.get("updates", 1) * kwargs.get("repeats", 1) + kwargs.get("seed", 0)
    torch.testing.assert_close(kernel(inputs), expected, atol=1e-4, rtol=1e-4)


def check_reduction_aware_kernel(columns, dtype, op):
    kernel = tl.compile(
        make_reducer(columns=columns, dtype=dtype, op=op),
        out_idx=-1,
        target="cuda",
    )
    inputs = torch.randn((8, columns), dtype=getattr(torch, dtype), device="cuda")
    expected = {"sum": torch.sum, "max": torch.amax, "min": torch.amin}[op](inputs, dim=0)
    tolerance = 1e-5 if dtype == "float32" else 2e-2
    torch.testing.assert_close(kernel(inputs), expected, atol=tolerance, rtol=tolerance)
    assert "tl::AllReduce<" not in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("columns", [128, 256])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_reduction_aware_kernel_correctness(columns, dtype, op):
    check_reduction_aware_kernel(columns, dtype, op)


@tilelang.testing.requires_cuda_compute_version_ge(8)
@pytest.mark.parametrize("columns", [128, 256])
@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_reduction_aware_bfloat16_kernel_correctness(columns, op):
    check_reduction_aware_kernel(columns, "bfloat16", op)


@tilelang.testing.requires_cuda_compute_version_ge(8)
def test_full_reduction_kernel_correctness():
    factory = load_factory("maint/layout_inference/cases/reducer_scalar_candidates.py")
    kernel = tl.compile(factory(total=True), out_idx=-1, target="cuda")
    inputs = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(kernel(inputs), inputs.float().sum().reshape(1), atol=1e-4, rtol=1e-4)
    assert "tl::AllReduce<" in kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
