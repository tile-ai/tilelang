import tilelang
import tilelang.language as T
import tilelang.testing
import pytest
import torch
from tilelang import tvm
from tvm.tirx.stmt_functor import post_order_visit


_COMPILE_FLAGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

_PROJECTED_ROW_LOOP_LAYOUT = T.Fragment(
    (8, 4),
    forward_fn=lambda i, k, rep: (i + 8 * k + 32 * rep, 0),
    replicate=4,
)

_VECTORIZED_OUTPUT_EXTENT = 256
_VECTORIZED_REDUCTION_EXTENT = 256
_VECTORIZED_LOCAL_EXTENT = 8
_VECTORIZED_THREADS = 32

_VECTORIZED_OUTPUT_LAYOUT = T.Fragment(
    (_VECTORIZED_OUTPUT_EXTENT,),
    forward_fn=lambda i: (i // _VECTORIZED_LOCAL_EXTENT, i % _VECTORIZED_LOCAL_EXTENT),
)

_VECTORIZED_REDUCTION_LAYOUT = T.Fragment(
    (_VECTORIZED_REDUCTION_EXTENT,),
    forward_fn=lambda k: (k // _VECTORIZED_LOCAL_EXTENT, k % _VECTORIZED_LOCAL_EXTENT),
)

_UNRELATED_FRAGMENT_LAYOUT = T.Fragment(
    (32,),
    forward_fn=lambda i: (i, 0),
)

_LOCAL_COMPLETE_COMPACT_LAYOUT = T.Fragment(
    (8,),
    forward_fn=lambda i: (i // 2, i % 2),
)

_INCOMPATIBLE_CONTRIBUTION_LAYOUT = T.Fragment(
    (8,),
    forward_fn=lambda i: (i, 0),
)

_CONFLICT_LOOP_LAYOUT = T.Fragment(
    (128,),
    forward_fn=lambda i: (i % 32, i // 32),
)

_CONFLICT_RESULT_A_LAYOUT = T.Fragment(
    (128,),
    forward_fn=lambda i: (i // 4, i % 4),
)

_CONFLICT_RESULT_B_LAYOUT = T.Fragment(
    (128,),
    forward_fn=lambda i: ((i // 4 + 1) % 32, i % 4),
)


@T.prim_func
def reducer_sum_v2(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
    with T.Kernel(1, threads=128):
        src = T.alloc_fragment((8,), T.float32)
        T.copy(A, src)
        total = T.alloc_reducer((1,), T.float32, op="sum")
        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[0], src[i])
        result = T.alloc_fragment((1,), T.float32)
        T.finalize_reducer(total, result)
        if T.get_thread_binding() == 0:
            B[0] = result[0]


@T.prim_func
def two_reducers_v2(A: T.Tensor((8,), T.float32), B: T.Tensor((2,), T.float32)):
    with T.Kernel(1, threads=128):
        src = T.alloc_fragment((8,), T.float32)
        T.copy(A, src)
        sum_total = T.alloc_reducer((1,), T.float32, op="sum", seed=5.0)
        max_total = T.alloc_reducer((1,), T.float32, op="max")
        T.reducer_init(sum_total)
        T.reducer_init(max_total)
        for i in T.Parallel(8):
            if i < 4:
                T.reducer_update(sum_total[0], src[i])
            T.reducer_update(max_total[0], src[i])
        sum_result = T.alloc_fragment((1,), T.float32)
        max_result = T.alloc_fragment((1,), T.float32)
        T.finalize_reducer(sum_total, sum_result)
        T.finalize_reducer(max_total, max_result)
        if T.get_thread_binding() == 0:
            B[0] = sum_result[0]
            B[1] = max_result[0]


@T.prim_func
def bitwise_reducers_v2(A: T.Tensor((8,), T.int32), B: T.Tensor((3,), T.int32)):
    with T.Kernel(1, threads=128):
        src = T.alloc_fragment((8,), T.int32)
        T.copy(A, src)
        and_total = T.alloc_reducer((1,), T.int32, op="bitand")
        or_total = T.alloc_reducer((1,), T.int32, op="bitor")
        xor_total = T.alloc_reducer((1,), T.int32, op="bitxor")
        T.reducer_init(and_total)
        T.reducer_init(or_total)
        T.reducer_init(xor_total)
        for i in T.Parallel(8):
            T.reducer_update(and_total[0], src[i])
            T.reducer_update(or_total[0], src[i])
            T.reducer_update(xor_total[0], src[i])
        and_result = T.alloc_fragment((1,), T.int32)
        or_result = T.alloc_fragment((1,), T.int32)
        xor_result = T.alloc_fragment((1,), T.int32)
        T.finalize_reducer(and_total, and_result)
        T.finalize_reducer(or_total, or_result)
        T.finalize_reducer(xor_total, xor_result)
        if T.get_thread_binding() == 0:
            B[0] = and_result[0]
            B[1] = or_result[0]
            B[2] = xor_result[0]


@T.prim_func
def unique_owner_reducer_v2(A: T.Tensor((8,), T.float32), B: T.Tensor((8,), T.float32)):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((8,), T.float32, op="sum", seed=2.0)
        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[i], A[i])
        result = T.alloc_fragment((8,), T.float32)
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def unique_owner_serial_reduction_v2(A: T.Tensor((8, 4), T.float32), B: T.Tensor((8,), T.float32)):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((8,), T.float32, op="sum")
        T.reducer_init(total)
        for i in T.Parallel(8):
            for k in T.serial(4):
                T.reducer_update(total[i], A[i, k])
        result = T.alloc_fragment((8,), T.float32)
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def local_complete_layout_propagation_v2(
    A: T.Tensor((128, 8), T.float32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=32):
        # Keep a partial explicit layout map in the source IR. Reducer layout
        # propagation must consume LayoutInference's complete result instead of
        # mistaking this unrelated annotation for the final map.
        scratch = T.alloc_fragment((32,), T.float32)
        T.annotate_layout({scratch: _UNRELATED_FRAGMENT_LAYOUT})
        T.clear(scratch)

        total = T.alloc_reducer((128,), T.float32, op="sum")
        result = T.alloc_fragment((128,), T.float32)
        T.reducer_init(total)
        for i in T.Parallel(128):
            for k in range(8):
                T.reducer_update(total[i], A[i, k])
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def local_complete_constrained_solve_v2(
    A: T.Tensor((8,), T.float32),
    B: T.Tensor((8,), T.float32),
):
    with T.Kernel(1, threads=32):
        total = T.alloc_reducer((8,), T.float32, op="sum")
        result = T.alloc_fragment((8,), T.float32)
        T.annotate_layout({result: _LOCAL_COMPLETE_COMPACT_LAYOUT})

        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[i], A[i])
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def local_complete_fragment_contribution_constrained_solve_v2(
    A: T.Tensor((8,), T.float32),
    B: T.Tensor((8,), T.float32),
):
    with T.Kernel(1, threads=32):
        x_frag = T.alloc_fragment((8,), T.float32)
        T.copy(A, x_frag)

        total = T.alloc_reducer((8,), T.float32, op="sum")
        result = T.alloc_fragment((8,), T.float32)
        T.annotate_layout({result: _LOCAL_COMPLETE_COMPACT_LAYOUT})

        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[i], x_frag[i])
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def incompatible_fragment_contribution_fallback_v2(
    A: T.Tensor((8,), T.float32),
    B: T.Tensor((8,), T.float32),
):
    with T.Kernel(1, threads=32):
        x_frag = T.alloc_fragment((8,), T.float32)
        T.annotate_layout({x_frag: _INCOMPATIBLE_CONTRIBUTION_LAYOUT})
        T.copy(A, x_frag)

        total = T.alloc_reducer((8,), T.float32, op="sum")
        result = T.alloc_fragment((8,), T.float32)
        T.annotate_layout({result: _LOCAL_COMPLETE_COMPACT_LAYOUT})

        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[i], x_frag[i])
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def conflicting_local_complete_layouts_v2(
    A: T.Tensor((128,), T.float32),
    B: T.Tensor((2, 128), T.float32),
):
    with T.Kernel(1, threads=32):
        total_a = T.alloc_reducer((128,), T.float32, op="sum")
        total_b = T.alloc_reducer((128,), T.float32, op="sum")
        result_a = T.alloc_fragment((128,), T.float32)
        result_b = T.alloc_fragment((128,), T.float32)
        T.annotate_layout(
            {
                result_a: _CONFLICT_RESULT_A_LAYOUT,
                result_b: _CONFLICT_RESULT_B_LAYOUT,
            }
        )

        T.reducer_init(total_a)
        T.reducer_init(total_b)
        for i in T.Parallel(128, loop_layout=_CONFLICT_LOOP_LAYOUT):
            T.reducer_update(total_a[i], A[i])
            T.reducer_update(total_b[i], A[i])
        T.finalize_reducer(total_a, result_a)
        T.finalize_reducer(total_b, result_b)
        T.copy(result_a, B[0, :])
        T.copy(result_b, B[1, :])


@T.prim_func
def explicit_loop_layout_conflicts_with_local_complete_v2(
    A: T.Tensor((128,), T.float32),
    B: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=32):
        total = T.alloc_reducer((128,), T.float32, op="sum")
        result = T.alloc_fragment((128,), T.float32)
        T.annotate_layout({result: _CONFLICT_RESULT_A_LAYOUT})

        T.reducer_init(total)
        for i in T.Parallel(128, loop_layout=_CONFLICT_LOOP_LAYOUT):
            T.reducer_update(total[i], A[i])
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def unique_owner_with_global_side_effect_v2(A: T.Tensor((8,), T.float32), B: T.Tensor((8,), T.float32)):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((8,), T.float32, op="sum")
        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[i], A[i])
            B[i] = A[i]
        result = T.alloc_fragment((8,), T.float32)
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def projected_row_reducer_v2(A: T.Tensor((8, 4), T.float32), B: T.Tensor((8,), T.float32)):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((8,), T.float32, op="sum")
        T.reducer_init(total)
        for i, k in T.Parallel(8, 4, loop_layout=_PROJECTED_ROW_LOOP_LAYOUT):
            T.reducer_update(total[i], A[i, k])
        result = T.alloc_fragment((8,), T.float32)
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def split_projection_groups_v2(
    A: T.Tensor((8,), T.float32),
    C: T.Tensor((16,), T.float32),
    B: T.Tensor((1,), T.float32),
):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((1,), T.float32, op="sum", seed=3.0)
        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[0], A[i])
        for j in T.Parallel(16):
            T.reducer_update(total[0], C[j])
        result = T.alloc_fragment((1,), T.float32)
        T.finalize_reducer(total, result)
        if T.get_thread_binding() == 0:
            B[0] = result[0]


def _run_reducer_layout_and_planning(func):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    mod = tvm.IRModule.from_expr(func)
    with target:
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.transform.VerifyReducerEpochs()(mod)
        after_layout = tilelang.transform.LayoutInference()(mod)
        after_planning = tilelang.transform.PlanAndMaterializeReducers()(after_layout)
    return after_layout, after_planning


def _find_reducer_parallel_layout(func):
    layouts = []

    def visit(node):
        if not isinstance(node, tvm.tirx.For) or node.kind != tvm.tirx.ForKind.PARALLEL or "parallel_loop_layout" not in node.annotations:
            return
        contains_update = False

        def find_update(child):
            nonlocal contains_update
            contains_update |= (
                isinstance(child, tvm.tirx.Call) and isinstance(child.op, tvm.ir.Op) and child.op.name == "tl.tileop.reducer_update"
            )

        post_order_visit(node.body, find_update)
        if contains_update:
            layouts.append(node.annotations["parallel_loop_layout"])

    post_order_visit(func.body, visit)
    assert len(layouts) == 1
    return layouts[0]


def _find_reducer_parallel_predicate(func):
    predicates = []

    def _visit(node):
        if isinstance(node, tvm.tirx.For) and "parallel_loop_layout" in node.annotations and "parallel_loop_predicate" in node.annotations:
            predicates.append(node.annotations["parallel_loop_predicate"])

    post_order_visit(func.body, _visit)
    assert len(predicates) == 1
    return predicates[0]


def _find_fragment_layout(func, buffer_name):
    layouts = []

    def visit(node):
        if not isinstance(node, tvm.tirx.SBlock) or "layout_map" not in node.annotations:
            return
        for buffer, layout in node.annotations["layout_map"].items():
            if buffer.name == buffer_name:
                layouts.append(layout)

    post_order_visit(func.body, visit)
    assert layouts
    for layout in layouts[1:]:
        assert tvm.ir.structural_equal(layouts[0], layout)
    return layouts[0]


def _same_fragment_mapping(lhs, rhs):
    return tvm.ir.structural_equal(lhs.forward_thread, rhs.forward_thread) and tvm.ir.structural_equal(lhs.forward_index, rhs.forward_index)


@T.prim_func
def mixed_projection_and_fallback_v2(
    A: T.Tensor((8,), T.float32),
    C: T.Tensor((6,), T.float32),
    B: T.Tensor((1,), T.float32),
):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((1,), T.float32, op="sum")
        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[0], A[i])
        # The first projected implementation only accepts power-of-two
        # collective widths. This site shares one canonical fallback group
        # without degrading the independent 8-way group above.
        for j in T.Parallel(6):
            T.reducer_update(total[0], C[j])
        result = T.alloc_fragment((1,), T.float32)
        T.finalize_reducer(total, result)
        if T.get_thread_binding() == 0:
            B[0] = result[0]


@T.prim_func
def vectorized_independent_outputs_reducer_v2(
    A: T.Tensor((_VECTORIZED_OUTPUT_EXTENT,), T.float16),
    C: T.Tensor((_VECTORIZED_OUTPUT_EXTENT,), T.float16),
    B: T.Tensor((_VECTORIZED_OUTPUT_EXTENT,), T.float16),
):
    with T.Kernel(1, threads=_VECTORIZED_THREADS):
        total = T.alloc_reducer((_VECTORIZED_OUTPUT_EXTENT,), T.float16, op="sum")
        result = T.alloc_fragment((_VECTORIZED_OUTPUT_EXTENT,), T.float16)
        T.annotate_layout({result: _VECTORIZED_OUTPUT_LAYOUT})

        T.reducer_init(total)
        for i in T.Parallel(
            _VECTORIZED_OUTPUT_EXTENT,
            loop_layout=_VECTORIZED_OUTPUT_LAYOUT,
        ):
            T.reducer_update(total[i], A[i])
            T.reducer_update(total[i], C[i])
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def vectorized_local_partial_reducer_v2(
    A: T.Tensor((_VECTORIZED_REDUCTION_EXTENT,), T.float16),
    B: T.Tensor((1,), T.float16),
):
    with T.Kernel(1, threads=_VECTORIZED_THREADS):
        src = T.alloc_fragment((_VECTORIZED_REDUCTION_EXTENT,), T.float16)
        T.annotate_layout({src: _VECTORIZED_REDUCTION_LAYOUT})
        T.copy(A, src)

        total = T.alloc_reducer((1,), T.float16, op="sum")
        result = T.alloc_fragment((1,), T.float16)
        T.reducer_init(total)
        for k in T.Parallel(
            _VECTORIZED_REDUCTION_EXTENT,
            loop_layout=_VECTORIZED_REDUCTION_LAYOUT,
        ):
            T.reducer_update(total[0], src[k])
        T.finalize_reducer(total, result)

        if T.get_thread_binding() == 0:
            B[0] = result[0]


def test_reducer_v2_frontend_ir():
    source = reducer_sum_v2.script()
    assert 'scope="local.reducer"' in source
    assert "T.reducer_init" in source
    assert "T.reducer_update" in source
    assert "T.finalize_reducer" in source


def test_reducer_v2_update_requires_indexed_target():
    with pytest.raises(TypeError, match="indexed local.reducer element"):

        @T.prim_func
        def invalid(A: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=32):
                total = T.alloc_reducer((1,), T.float32)
                T.reducer_init(total)
                # A bare reducer handle is not an indexed update target.
                T.reducer_update(total, A[0])


def test_reducer_v2_update_rejects_non_reducer_target():
    with pytest.raises(ValueError, match="local.reducer target"):

        @T.prim_func
        def invalid(A: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=32):
                total = T.alloc_fragment((1,), T.float32)
                T.reducer_update(total[0], A[0])


def test_reducer_v2_cuda_codegen():
    source = tilelang.compile(
        reducer_sum_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "tl::AllReduce<tl::SumOp, 8, 1" in source
    assert "tl::AllReduce<tl::SumOp, 128" not in source
    assert "local.reducer" not in source


def test_reducer_v2_layout_inference_propagates_local_complete_layout():
    after_layout, after_planning = _run_reducer_layout_and_planning(local_complete_layout_propagation_v2)
    layout_func = after_layout["local_complete_layout_propagation_v2"]
    planning_func = after_planning["local_complete_layout_propagation_v2"]

    result_layout = _find_fragment_layout(layout_func, "result")
    inferred_loop_layout = _find_reducer_parallel_layout(layout_func)
    planned_loop_layout = _find_reducer_parallel_layout(planning_func)

    assert tvm.ir.structural_equal(inferred_loop_layout, result_layout)
    assert tvm.ir.structural_equal(planned_loop_layout, inferred_loop_layout)


def test_reducer_v2_local_complete_constraint_drives_final_layout_solve():
    after_layout, after_planning = _run_reducer_layout_and_planning(local_complete_constrained_solve_v2)
    layout_func = after_layout["local_complete_constrained_solve_v2"]
    planning_func = after_planning["local_complete_constrained_solve_v2"]

    result_layout = _find_fragment_layout(layout_func, "result")
    inferred_loop_layout = _find_reducer_parallel_layout(layout_func)
    planned_loop_layout = _find_reducer_parallel_layout(planning_func)

    assert _same_fragment_mapping(inferred_loop_layout, result_layout)
    assert tvm.ir.structural_equal(planned_loop_layout, inferred_loop_layout)
    # The compact destination uses four of the 32 participants. This predicate
    # is rebuilt by ParallelOp during the constrained solve; replacing for_map
    # after a completed solve would leave it absent.
    predicate = _find_reducer_parallel_predicate(layout_func)
    assert "< 4" in str(predicate)
    assert tvm.ir.structural_equal(_find_reducer_parallel_predicate(planning_func), predicate)

    source = tilelang.compile(
        local_complete_constrained_solve_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "float total_partial_0[2];" in source
    assert "if (((int)threadIdx.x) < 4)" in source
    assert "float2" in source
    assert "tl::AllReduce<" not in source


def test_reducer_v2_local_complete_constraint_propagates_to_fragment():
    after_layout, after_planning = _run_reducer_layout_and_planning(local_complete_fragment_contribution_constrained_solve_v2)
    name = "local_complete_fragment_contribution_constrained_solve_v2"
    layout_func = after_layout[name]
    planning_func = after_planning[name]

    result_layout = _find_fragment_layout(layout_func, "result")
    contribution_layout = _find_fragment_layout(layout_func, "x_frag")
    inferred_loop_layout = _find_reducer_parallel_layout(layout_func)
    planned_loop_layout = _find_reducer_parallel_layout(planning_func)

    assert _same_fragment_mapping(inferred_loop_layout, result_layout)
    assert _same_fragment_mapping(contribution_layout, result_layout)
    assert tvm.ir.structural_equal(planned_loop_layout, inferred_loop_layout)

    source = tilelang.compile(
        local_complete_fragment_contribution_constrained_solve_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "float x_frag[2];" in source
    assert "float total_partial_0[2];" in source
    assert "float2" in source
    assert "tl::AllReduce<" not in source


def test_reducer_v2_incompatible_fragment_constraint_falls_back():
    after_layout, after_planning = _run_reducer_layout_and_planning(incompatible_fragment_contribution_fallback_v2)
    name = "incompatible_fragment_contribution_fallback_v2"
    layout_func = after_layout[name]
    planning_func = after_planning[name]

    result_layout = _find_fragment_layout(layout_func, "result")
    contribution_layout = _find_fragment_layout(layout_func, "x_frag")
    inferred_loop_layout = _find_reducer_parallel_layout(layout_func)
    planned_loop_layout = _find_reducer_parallel_layout(planning_func)

    assert _same_fragment_mapping(inferred_loop_layout, contribution_layout)
    assert not _same_fragment_mapping(inferred_loop_layout, result_layout)
    assert tvm.ir.structural_equal(planned_loop_layout, inferred_loop_layout)

    source = tilelang.compile(
        incompatible_fragment_contribution_fallback_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "float total_partial_0[8];" in source
    assert "tl::AllReduce<tl::SumOp, 32" in source


@pytest.mark.parametrize(
    ("func", "name"),
    [
        (projected_row_reducer_v2, "projected_row_reducer_v2"),
        (reducer_sum_v2, "reducer_sum_v2"),
    ],
)
def test_reducer_v2_planning_preserves_non_local_complete_loop_layout(func, name):
    after_layout, after_planning = _run_reducer_layout_and_planning(func)
    inferred_loop_layout = _find_reducer_parallel_layout(after_layout[name])
    planned_loop_layout = _find_reducer_parallel_layout(after_planning[name])
    assert tvm.ir.structural_equal(planned_loop_layout, inferred_loop_layout)


def test_reducer_v2_conflicting_local_complete_layouts_preserve_loop_layout():
    after_layout, after_planning = _run_reducer_layout_and_planning(conflicting_local_complete_layouts_v2)
    layout_func = after_layout["conflicting_local_complete_layouts_v2"]
    planning_func = after_planning["conflicting_local_complete_layouts_v2"]

    inferred_loop_layout = _find_reducer_parallel_layout(layout_func)
    planned_loop_layout = _find_reducer_parallel_layout(planning_func)
    result_a_layout = _find_fragment_layout(layout_func, "result_a")
    result_b_layout = _find_fragment_layout(layout_func, "result_b")

    assert _same_fragment_mapping(inferred_loop_layout, _CONFLICT_LOOP_LAYOUT)
    assert not _same_fragment_mapping(inferred_loop_layout, result_a_layout)
    assert not _same_fragment_mapping(inferred_loop_layout, result_b_layout)
    assert tvm.ir.structural_equal(planned_loop_layout, inferred_loop_layout)

    source = tilelang.compile(
        conflicting_local_complete_layouts_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "float total_a_partial_0[128];" in source
    assert "float total_b_partial_0[128];" in source
    assert source.count("tl::AllReduce<tl::SumOp, 32") == 2


def test_reducer_v2_explicit_loop_layout_conflict_falls_back():
    after_layout, after_planning = _run_reducer_layout_and_planning(explicit_loop_layout_conflicts_with_local_complete_v2)
    name = "explicit_loop_layout_conflicts_with_local_complete_v2"
    layout_func = after_layout[name]
    planning_func = after_planning[name]

    inferred_loop_layout = _find_reducer_parallel_layout(layout_func)
    planned_loop_layout = _find_reducer_parallel_layout(planning_func)
    result_layout = _find_fragment_layout(layout_func, "result")

    assert _same_fragment_mapping(inferred_loop_layout, _CONFLICT_LOOP_LAYOUT)
    assert not _same_fragment_mapping(inferred_loop_layout, result_layout)
    assert tvm.ir.structural_equal(planned_loop_layout, inferred_loop_layout)

    source = tilelang.compile(
        explicit_loop_layout_conflicts_with_local_complete_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "float total_partial_0[128];" in source
    assert "tl::AllReduce<tl::SumOp, 32" in source


def test_reducer_v2_unique_owner_codegen_uses_local_complete_plan():
    source = tilelang.compile(
        unique_owner_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "tl::AllReduce<" not in source
    assert "NamedBarrier<" not in source
    assert "workspace" not in source
    assert "float total_partial_0[1];" in source


def test_reducer_v2_unique_owner_serial_reduction_codegen():
    source = tilelang.compile(
        unique_owner_serial_reduction_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "tl::AllReduce<" not in source
    assert "NamedBarrier<" not in source
    assert "workspace" not in source
    assert "float total_partial_0[1];" in source


def test_reducer_v2_unique_owner_side_effect_falls_back():
    source = tilelang.compile(
        unique_owner_with_global_side_effect_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "tl::AllReduce<tl::SumOp, 128" in source


def test_reducer_v2_parallel_mk_uses_projected_row_group():
    source = tilelang.compile(
        projected_row_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    # k is carried by thread bits 3-4, so four logical lanes are represented
    # by a 32-thread span with stride 8.
    assert "tl::AllReduce<tl::SumOp, 32, 8" in source
    assert "tl::AllReduce<tl::SumOp, 128" not in source
    assert "float total_partial_0[1];" in source


def test_reducer_v2_incompatible_projections_use_separate_groups():
    source = tilelang.compile(
        split_projection_groups_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "float total_partial_0[1];" in source
    assert "float total_partial_1[1];" in source
    assert "tl::AllReduce<tl::SumOp, 8, 1" in source
    assert "tl::AllReduce<tl::SumOp, 16, 1" in source
    assert "tl::AllReduce<tl::SumOp, 128" not in source


def test_reducer_v2_projected_and_canonical_groups_can_mix():
    source = tilelang.compile(
        mixed_projection_and_fallback_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "float total_partial_0[1];" in source
    assert "float total_partial_1[1];" in source
    assert "tl::AllReduce<tl::SumOp, 8, 1" in source
    assert "tl::AllReduce<tl::SumOp, 128" in source


def test_reducer_v2_vectorizes_independent_output_updates():
    source = tilelang.compile(
        vectorized_independent_outputs_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "tl::add2" in source
    assert "tl::AllReduce<" not in source


def test_reducer_v2_vectorizes_thread_local_partial_updates():
    source = tilelang.compile(
        vectorized_local_partial_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "tl::add2" in source
    assert "tl::AllReduce<tl::SumOp, 32, 1" in source


@tilelang.testing.requires_cuda
def test_reducer_v2_unique_owner_local_complete_correctness():
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    B = tilelang.compile(
        unique_owner_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, A + 2.0, atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_unique_owner_serial_reduction_correctness():
    A = torch.arange(1, 33, dtype=torch.float32, device="cuda").reshape(8, 4)
    B = tilelang.compile(
        unique_owner_serial_reduction_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, A.sum(dim=1), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_local_complete_fragment_contribution_correctness():
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    B = tilelang.compile(
        local_complete_fragment_contribution_constrained_solve_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, A, atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_incompatible_fragment_fallback_correctness():
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    B = tilelang.compile(
        incompatible_fragment_contribution_fallback_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, A, atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_projected_row_group_correctness():
    A = torch.arange(1, 33, dtype=torch.float32, device="cuda").reshape(8, 4)
    B = tilelang.compile(
        projected_row_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, A.sum(dim=1), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_split_projection_groups_correctness():
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    C = torch.arange(1, 17, dtype=torch.float32, device="cuda")
    B = tilelang.compile(
        split_projection_groups_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A, C)
    torch.testing.assert_close(B, (A.sum() + C.sum() + 3.0).reshape(1), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_mixed_projection_and_fallback_correctness():
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    C = torch.arange(1, 7, dtype=torch.float32, device="cuda")
    B = tilelang.compile(
        mixed_projection_and_fallback_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A, C)
    torch.testing.assert_close(B, (A.sum() + C.sum()).reshape(1), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_vectorized_independent_output_correctness():
    A = torch.arange(
        _VECTORIZED_OUTPUT_EXTENT,
        dtype=torch.float16,
        device="cuda",
    )
    C = torch.arange(
        _VECTORIZED_OUTPUT_EXTENT,
        dtype=torch.float16,
        device="cuda",
    ).flip(0)
    B = tilelang.compile(
        vectorized_independent_outputs_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A, C)
    torch.testing.assert_close(B, A + C, atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_vectorized_local_partial_correctness():
    A = torch.arange(
        1,
        _VECTORIZED_REDUCTION_EXTENT + 1,
        dtype=torch.float16,
        device="cuda",
    )
    B = tilelang.compile(
        vectorized_local_partial_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, A.sum().reshape(1), atol=1e-1, rtol=1e-3)


@tilelang.testing.requires_cuda
def test_reducer_v2_multiple_reducers_seed_and_predicate():
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    B = tilelang.compile(
        two_reducers_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, torch.tensor([15.0, 8.0], device="cuda"), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_builtin_bitwise_reductions():
    A = torch.arange(1, 9, dtype=torch.int32, device="cuda")
    B = tilelang.compile(
        bitwise_reducers_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, torch.tensor([0, 15, 8], dtype=torch.int32, device="cuda"))


def test_reducer_v2_rejects_legacy_replication_keyword():
    with pytest.raises(Exception, match="replication"):

        @T.prim_func
        def legacy(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=32):
                total = T.alloc_reducer((1,), T.float32, replication="all")
                T.clear(total)
                B[0] = A[0]


def test_reducer_v2_requires_out_of_place_destination():
    with pytest.raises(Exception, match="destination"):

        @T.prim_func
        def legacy(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=32):
                total = T.alloc_reducer((1,), T.float32)
                T.reducer_init(total)
                T.reducer_update(total[0], A[0])
                T.finalize_reducer(total)
                B[0] = A[0]


def test_reducer_v2_rejects_direct_clear():
    @T.prim_func
    def invalid(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            total = T.alloc_reducer((1,), T.float32)
            T.clear(total)
            T.reducer_init(total)
            T.reducer_update(total[0], A[0])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="forbids ordinary BufferLoad access"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_missing_init():
    @T.prim_func
    def invalid(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_update(total[0], A[0])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="must be dominated by T.reducer_init"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_missing_finalize():
    @T.prim_func
    def invalid(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_init(total)
            T.reducer_update(total[0], A[0])
            B[0] = A[0]

    with pytest.raises(Exception, match="must have exactly one explicit T.reducer_init"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_double_init():
    @T.prim_func
    def invalid(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_init(total)
            T.reducer_init(total)
            T.reducer_update(total[0], A[0])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="must occur exactly once per reducer allocation"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_unprovable_output_index():
    @T.prim_func
    def invalid(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_init(total)
            T.reducer_update(total[1], A[0])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="is not provably within"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_physical_replica_dependent_update():
    @T.prim_func
    def invalid(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((8,), T.float32)
            T.copy(A, src)
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_init(total)
            for i in T.Parallel(8):
                T.reducer_update(total[0], src[i] + T.get_thread_binding())
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="must be invariant across physical replicas"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_private_local_replica_dependent_update():
    @T.prim_func
    def invalid(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((8,), T.float32)
            T.copy(A, src)
            thread_value = T.alloc_local((1,), T.float32)
            thread_value[0] = A[T.get_thread_binding() % 8]
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_init(total)
            for i in T.Parallel(8):
                T.reducer_update(total[0], src[i] + thread_value[0])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="must be invariant across physical replicas"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)
