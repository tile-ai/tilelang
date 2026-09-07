import importlib.util
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


def infer(factory=make_reducer, model="reduction-aware", target=None, pass_configs=None, **kwargs):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"} if target is None else target)
    with target, tl.transform.PassContext(config={"tl.layout_cost_model": model, **(pass_configs or {})}):
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
def test_intermediate_width_beats_both_endpoints():
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
    with target, tl.transform.PassContext(config={"tl.layout_cost_model": "reduction-aware"}):
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
def test_existing_policies_keep_their_winners():
    _, register_count = infer(model="register-count", rows=4, columns=256)
    _, io_aware = infer(model="io-aware", rows=4, columns=256)
    _, scalar = infer(rows=4, columns=256, width=1)
    _, native = infer(rows=4, columns=256, width=4)
    assert register_count["values"].is_equal(scalar["values"])
    assert io_aware["values"].is_equal(native["values"])


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_unknown_serial_extent_retains_register_count_search():
    @T.prim_func
    def kernel(inputs: T.Tensor((8, 128), "float32"), output: T.Tensor((128,), "float32"), count: T.int32):
        with T.Kernel(1, threads=128):
            values = T.alloc_fragment((8, 128), "float32")
            acc = T.alloc_reducer((128,), "float32")
            result = T.alloc_fragment((128,), "float32")
            T.copy(inputs, values)
            T.reducer_init(acc)
            for _repeat in T.serial(count):
                for row, column in T.Parallel(8, 128):
                    T.reducer_update(acc[column], values[row, column])
            T.finalize_reducer(acc, result)
            T.copy(result, output)

    _, automatic = infer(factory=lambda: kernel)
    _, fallback = infer(factory=lambda: kernel, model="register-count")
    assert automatic["values"].is_equal(fallback["values"])
    assert automatic["acc"].is_equal(fallback["acc"])


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("constraints", [{"width": 4}, {"pinned": True}, {"pin_loop": True}])
def test_explicit_constraints_are_preserved(constraints):
    factory = load_factory("maint/layout_inference/cases/reducer_scalar_candidates.py")
    _, automatic = infer(factory=factory, **constraints)
    _, pinned = infer(factory=factory, model="register-count", width=4)
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
    kernel = tl.compile(program, out_idx=-1, target="cuda", pass_configs={"tl.layout_cost_model": "reduction-aware"})
    inputs = torch.randn((4, 256), device="cuda")
    expected = inputs.sum().reshape(1) if total else inputs.sum(dim=0)
    torch.testing.assert_close(kernel(inputs), expected, atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_rocm(support_required="compile-only")
def test_non_cuda_target_retains_register_count():
    target = {"kind": "hip", "mcpu": "gfx90a"}
    _, automatic = infer(target=target)
    _, baseline = infer(target=target, model="register-count")
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
    kernel = tl.compile(make_reducer(**kwargs), out_idx=-1, target="cuda", pass_configs={"tl.layout_cost_model": "reduction-aware"})
    inputs = torch.randn((kwargs.get("rows", 8), 128), device="cuda")
    expected = inputs.sum(dim=0) * kwargs.get("updates", 1) * kwargs.get("repeats", 1) + kwargs.get("seed", 0)
    torch.testing.assert_close(kernel(inputs), expected, atol=1e-4, rtol=1e-4)


def check_reduction_aware_kernel(columns, dtype, op):
    kernel = tl.compile(
        make_reducer(columns=columns, dtype=dtype, op=op),
        out_idx=-1,
        target="cuda",
        pass_configs={"tl.layout_cost_model": "reduction-aware"},
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
    kernel = tl.compile(factory(total=True), out_idx=-1, target="cuda", pass_configs={"tl.layout_cost_model": "reduction-aware"})
    inputs = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(kernel(inputs), inputs.float().sum().reshape(1), atol=1e-4, rtol=1e-4)
    assert "tl::AllReduce<" in kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
