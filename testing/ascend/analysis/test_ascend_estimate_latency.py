import pytest

import tilelang
import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.backend.target import determine_target
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform, post_order_visit


def _make_program(latency=None, ii=None):
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((16,), "float32")
            with T.Task(latency=latency, ii=ii):
                T.copy(A, temp)
            T.copy(temp, B)

    return main


def _make_guarded_program():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(2) as bx:
            temp = T.alloc_shared((16,), "float32")
            if bx == 0:
                T.copy(A, temp)
            else:
                T.copy(temp, B)

    return main


def _make_partially_staged_program():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((16,), "float32")
            for i in T.Pipelined(4, annotations={"enable_offset": True}):
                # Source order intentionally opposes logical stage order.
                with T.Stage(1), T.Task(latency=1, ii=1):
                    T.copy(temp, B[i * 16 : (i + 1) * 16])
                with T.Task(latency=1, ii=1):
                    T.copy(A[i * 16 : (i + 1) * 16], temp)

    return main


def _make_dual_copy_program(split_n: bool):
    full_shape = (64, 256) if split_n else (128, 128)
    half_shape = (64, 128)

    @T.prim_func
    def main(
        A: T.Tensor(full_shape, "bfloat16"),
        B: T.Tensor(full_shape, "bfloat16"),
    ):
        with T.Kernel(1):
            temp = T.alloc_shared(half_shape, "bfloat16")
            T.dual_copy(A, temp)
            T.dual_copy(temp, B)

    return main


def _make_ordinary_copy_program(split_n: bool, use_sid: bool, bind_sid: bool = False):
    full_shape = (64, 256) if split_n else (128, 128)
    half_shape = (64, 128)

    @T.prim_func
    def main(
        A: T.Tensor(full_shape, "bfloat16"),
        B: T.Tensor(full_shape, "bfloat16"),
    ):
        with T.MixedKernel(1) as (_, sid):
            temp = T.alloc_shared(half_shape, "bfloat16")
            partition = T.bind(sid) if bind_sid else sid
            if not use_sid:
                partition = 0
            if split_n:
                T.copy(A[:, partition * 128 : (partition + 1) * 128], temp)
                T.copy(temp, B[:, partition * 128 : (partition + 1) * 128])
            else:
                T.copy(A[partition * 64 : (partition + 1) * 64, :], temp)
                T.copy(temp, B[partition * 64 : (partition + 1) * 64, :])

    return main


def _make_pure_vector_copy_program(split_n: bool):
    full_shape = (64, 256) if split_n else (128, 128)
    half_shape = (64, 128)

    @T.prim_func
    def main(
        A: T.Tensor(full_shape, "bfloat16"),
        B: T.Tensor(full_shape, "bfloat16"),
    ):
        with T.Kernel(64):
            temp = T.alloc_shared(half_shape, "bfloat16")
            if split_n:
                T.copy(A[:, :128], temp)
                T.copy(temp, B[:, :128])
            else:
                T.copy(A[:64, :], temp)
                T.copy(temp, B[:64, :])

    return main


def _make_mixed_aiv_copy_program(rows_per_aiv: int, columns: int):
    full_shape = (rows_per_aiv * 2, columns)
    half_shape = (rows_per_aiv, columns)

    @T.prim_func
    def main(
        A: T.Tensor(full_shape, "bfloat16"),
        B: T.Tensor(full_shape, "bfloat16"),
    ):
        with T.MixedKernel(1) as (_, sid):
            temp = T.alloc_shared(half_shape, "bfloat16")
            T.copy(A[sid * rows_per_aiv : (sid + 1) * rows_per_aiv, :], temp)
            T.copy(temp, B[sid * rows_per_aiv : (sid + 1) * rows_per_aiv, :])

    return main


def _make_symbolic_stride_copy_program():
    row_stride = T.dynamic("row_stride")

    @T.prim_func
    def main(
        A: T.StridedTensor(
            shape=[128, 128],
            strides=[row_stride, 1],
            dtype="bfloat16",
        ),
        B: T.StridedTensor(
            shape=[128, 128],
            strides=[row_stride, 1],
            dtype="bfloat16",
        ),
    ):
        with T.MixedKernel(1):
            temp = T.alloc_shared((64, 128), "bfloat16")
            T.copy(A[:64, :], temp)
            T.copy(temp, B[:64, :])

    return main


def _make_high_dim_strided_copy_program():
    @T.prim_func
    def main(
        A: T.Tensor((4, 2, 128), "bfloat16"),
        B: T.Tensor((4, 2, 128), "bfloat16"),
    ):
        with T.MixedKernel(1):
            temp = T.alloc_shared((2, 1, 128), "bfloat16")
            T.copy(A[0:2, 0:1, :], temp)
            T.copy(temp, B[0:2, 0:1, :])

    return main


def _make_rank_one_dual_copy_program():
    @T.prim_func
    def main(
        A: T.Tensor((8192,), "bfloat16"),
        B: T.Tensor((8192,), "bfloat16"),
    ):
        with T.Kernel(1):
            temp = T.alloc_shared((4096,), "bfloat16")
            T.dual_copy(A, temp)
            T.dual_copy(temp, B)

    return main


def _make_n512_dual_copy_program():
    @T.prim_func
    def main(
        A: T.Tensor((32, 512), "bfloat16"),
        B: T.Tensor((32, 512), "bfloat16"),
    ):
        with T.Kernel(1):
            temp = T.alloc_shared((32, 256), "bfloat16")
            T.dual_copy(A, temp)
            T.dual_copy(temp, B)

    return main


def _make_rank_one_strided_copy_program(explicit_rows: bool):
    if explicit_rows:

        @T.prim_func
        def main(A: T.StridedTensor([4096, 1], [2, 1], "bfloat16")):
            with T.Kernel(64):
                temp = T.alloc_shared((4096, 1), "bfloat16")
                T.copy(A[:, :], temp)

    else:

        @T.prim_func
        def main(A: T.StridedTensor([4096], [2], "bfloat16")):
            with T.Kernel(64):
                temp = T.alloc_shared((4096,), "bfloat16")
                T.copy(A[:], temp)

    return main


def _make_sid_padded_copy_program():
    @T.prim_func
    def main(A: T.StridedTensor([128, 32], [64, 1], "bfloat16")):
        with T.MixedKernel(1) as (_, sid):
            temp = T.alloc_shared((64, 32), "bfloat16")
            T.copy(A[sid * 64 : (sid + 1) * 64, :], temp)

    return main


def _make_single_sid_copy_program():
    @T.prim_func
    def main(A: T.Tensor((64, 128), "bfloat16")):
        with T.MixedKernel(1, sids=1) as (_, sid):
            temp = T.alloc_shared((64, 128), "bfloat16")
            T.copy(A[sid * 64 : (sid + 1) * 64, :], temp)

    return main


def _make_gm_to_l1_program(size: int):
    @T.prim_func
    def main(A: T.Tensor((size, size), "bfloat16")):
        with T.Kernel(1):
            temp = T.alloc_l1((size, size), "bfloat16")
            T.copy(A, temp)

    return main


def _make_two_small_gm_to_l1_copies_program():
    @T.prim_func
    def main(
        A: T.Tensor((16, 16), "bfloat16"),
        B: T.Tensor((16, 16), "bfloat16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 16), "bfloat16")
            b_l1 = T.alloc_l1((16, 16), "bfloat16")
            with T.Task():
                T.copy(A, a_l1)
                T.copy(B, b_l1)

    return main


def _make_l0c_copy_program(dst_scope: str, dtype: str):
    if dst_scope == "ub":

        @T.prim_func
        def main():
            with T.Kernel(1):
                src = T.alloc_l0c((128, 128), "float32")
                dst = T.alloc_shared((128, 128), dtype)
                T.copy(src, dst)

    else:

        @T.prim_func
        def main(dst: T.Tensor((128, 128), dtype)):
            with T.Kernel(1):
                src = T.alloc_l0c((128, 128), "float32")
                T.copy(src, dst)

    return main


def _make_narrow_fixpipe_dual_program(split_n: bool):
    full_shape = (16, 32) if split_n else (32, 16)

    @T.prim_func
    def main():
        with T.Kernel(1):
            src = T.alloc_l0c(full_shape, "float32")
            dst = T.alloc_shared((16, 16), "float32")
            T.dual_copy(src, dst)

    return main


def _make_unit_axis_fixpipe_dual_program():
    @T.prim_func
    def main():
        with T.Kernel(1):
            src = T.alloc_l0c((1, 32, 16), "float32")
            dst = T.alloc_shared((16, 16, 1), "float32")
            T.dual_copy(src, dst)

    return main


def _bind_target(program):
    mod = tvm.IRModule.from_expr(program.with_attr("global_symbol", "main"))
    return tirx.transform.BindTarget(determine_target("ascend"))(mod)


def _materialize_schedule_units(mod):
    return ascend_transform.MaterializeScheduleUnits()(mod)


def _estimate_copy_metadata(program, rewrite_dual_copy=False):
    mod = _bind_target(program)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    if rewrite_dual_copy:
        mod = ascend_transform.RewriteDualCopy()(mod)
    mod = _materialize_schedule_units(mod)
    mod = ascend_transform.EstimateLatency()(mod)
    metadata = _collect_task_metadata(mod)
    return list(zip(metadata["latency"], metadata["ii"]))


def _collect_task_metadata(mod):
    metadata = {"latency": [], "ii": [], "core_mask": []}

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.ascend_task":
            for key in metadata:
                if key in node.node:
                    metadata[key].append(int(node.node[key]))

    post_order_visit(mod["main"].body, visit)
    return metadata


def _collect_schedule_units(mod):
    units = []

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit":
            units.append(node)

    post_order_visit(mod["main"].body, visit)
    return units


def _direct_schedule_unit_stages(stmt):
    statements = stmt.seq if isinstance(stmt, tirx.SeqStmt) else [stmt]
    return [
        int(statement.node["stage"])
        for statement in statements
        if isinstance(statement, tirx.AttrStmt) and statement.attr_key == "tl.schedule_unit"
    ]


def _duplicate_one_task_schedule_unit(mod):
    duplicated = False

    def rewrite(node):
        nonlocal duplicated
        if duplicated or not isinstance(node, tirx.AttrStmt) or node.attr_key != "tl.schedule_unit":
            return None
        if not isinstance(node.body, tirx.AttrStmt) or node.body.attr_key != "tl.ascend_task":
            return None
        duplicated = True
        return tirx.AttrStmt(
            node.node,
            node.attr_key,
            node.value,
            tirx.SeqStmt([node.body, node.body]),
        )

    func = mod["main"]
    body = ir_transform(func.body, None, rewrite)
    assert duplicated
    return tvm.IRModule({"main": func.with_body(body)})


def test_estimate_latency_annotates_all_materialized_tasks():
    mod = ascend_transform.EstimateLatency()(_materialize_schedule_units(_bind_target(_make_program())))
    metadata = _collect_task_metadata(mod)

    assert len(metadata["latency"]) == 2
    assert len(metadata["ii"]) == 2
    assert all(latency >= 0 for latency in metadata["latency"])
    assert all(ii > 0 for ii in metadata["ii"])
    assert _collect_schedule_units(mod)


def test_frontend_task_hints_feed_shared_task_metadata():
    mod = ascend_transform.EstimateLatency()(_materialize_schedule_units(_bind_target(_make_program(latency=17, ii=5))))
    metadata = _collect_task_metadata(mod)

    assert 17 in metadata["latency"]
    assert 5 in metadata["ii"]


@pytest.mark.parametrize("split_n", [False, True])
def test_rewritten_dual_copy_matches_equivalent_aiv_mte_access(split_n):
    unrewritten = _estimate_copy_metadata(_make_dual_copy_program(split_n))
    rewritten = _estimate_copy_metadata(_make_dual_copy_program(split_n), rewrite_dual_copy=True)
    sid_partitioned = _estimate_copy_metadata(_make_ordinary_copy_program(split_n, use_sid=True))
    bound_sid_partitioned = _estimate_copy_metadata(_make_ordinary_copy_program(split_n, use_sid=True, bind_sid=True))
    fixed_partition = _estimate_copy_metadata(_make_ordinary_copy_program(split_n, use_sid=False))
    pure_vector = _estimate_copy_metadata(_make_pure_vector_copy_program(split_n))

    expected = [(403, 328), (478, 298)] if split_n else [(418, 328), (465, 285)]
    assert unrewritten == rewritten == sid_partitioned == fixed_partition == pure_vector == expected
    assert bound_sid_partitioned == [(1, 1), *expected]


def test_aiv_mte_large_hbm_uses_full_occupancy_ii():
    estimated = _estimate_copy_metadata(_make_mixed_aiv_copy_program(64, 128))

    assert estimated == [(418, 328), (465, 285)]


def test_aiv_mte_small_copy_keeps_descriptor_floor_ii():
    estimated = _estimate_copy_metadata(_make_mixed_aiv_copy_program(1, 64))

    assert estimated == [(93, 13), (183, 10)]


def test_cthread_symbolic_stride_uses_conservative_unknown_split():
    estimated = _estimate_copy_metadata(_make_symbolic_stride_copy_program())

    assert estimated == [(403, 328), (478, 298)]


def test_aiv_mte_high_dim_copy_uses_actual_mte_row_axis():
    estimated = _estimate_copy_metadata(_make_high_dim_strided_copy_program())

    assert estimated == [(86, 13), (190, 10)]


def test_rank_one_dual_copy_keeps_geometry_across_rewrite():
    program = _make_rank_one_dual_copy_program()

    assert (
        _estimate_copy_metadata(program)
        == _estimate_copy_metadata(program, rewrite_dual_copy=True)
        == [
            (254, 164),
            (323, 143),
        ]
    )


def test_n512_copy_uses_measured_full_width_mte3_rate():
    program = _make_n512_dual_copy_program()

    assert (
        _estimate_copy_metadata(program)
        == _estimate_copy_metadata(program, rewrite_dual_copy=True)
        == [
            (403, 328),
            (465, 285),
        ]
    )


def test_equivalent_strided_regions_share_mte_row_geometry():
    rank_one = _estimate_copy_metadata(_make_rank_one_strided_copy_program(explicit_rows=False))
    explicit_rows = _estimate_copy_metadata(_make_rank_one_strided_copy_program(explicit_rows=True))

    assert rank_one == explicit_rows == [(403, 328)]


def test_sid_partition_does_not_override_strided_mte_geometry():
    estimated = _estimate_copy_metadata(_make_sid_padded_copy_program())

    assert estimated == [(239, 164)]


def test_single_sid_copy_lowers_with_auto_schedule():
    estimated = _estimate_copy_metadata(_make_single_sid_copy_program())
    source = tilelang.lower(_make_single_sid_copy_program(), target="ascend").kernel_source

    assert estimated == [(418, 328)]
    assert "__global__ __vector__" in source


def test_updated_non_aiv_copy_costs_match_full_path_remeasurement():
    assert _estimate_copy_metadata(_make_gm_to_l1_program(16)) == [(196, 64)]
    assert _estimate_copy_metadata(_make_gm_to_l1_program(128)) == [(518, 328)]
    assert _estimate_copy_metadata(_make_two_small_gm_to_l1_copies_program()) == [(260, 128)]
    assert _estimate_copy_metadata(_make_l0c_copy_program("ub", "float32")) == [(570, 514)]
    assert _estimate_copy_metadata(_make_l0c_copy_program("gm", "float32")) == [(712, 514)]


def test_fixpipe_quant_cost_uses_destination_payload_bytes():
    assert _estimate_copy_metadata(_make_l0c_copy_program("ub", "bfloat16")) == [(314, 258)]
    assert _estimate_copy_metadata(_make_l0c_copy_program("gm", "bfloat16")) == [(456, 258)]


@pytest.mark.parametrize("split_n", [False, True])
def test_narrow_fixpipe_dual_uses_physical_trailing_row_width(split_n):
    assert _estimate_copy_metadata(_make_narrow_fixpipe_dual_program(split_n)) == [(71, 16)]


def test_fixpipe_dual_ignores_destination_unit_axis_for_row_width():
    assert _estimate_copy_metadata(_make_unit_axis_fixpipe_dual_program()) == [(71, 16)]


def test_materialize_schedule_units_builds_trivial_guarded_schedule():
    mod = _materialize_schedule_units(_bind_target(_make_guarded_program()))
    schedule_units = _collect_schedule_units(mod)
    task_metadata = _collect_task_metadata(mod)

    assert schedule_units
    assert all(int(unit.node["stage"]) == -1 for unit in schedule_units)
    assert all(len(unit.node) == 1 for unit in schedule_units)
    assert sum(isinstance(unit.body, tirx.IfThenElse) for unit in schedule_units) >= 2
    assert task_metadata["latency"] == []
    assert task_metadata["ii"] == []


def test_materialize_schedule_units_normalizes_partial_manual_stages():
    mod = _materialize_schedule_units(_bind_target(_make_partially_staged_program()))
    loop_units = [unit for unit in _collect_schedule_units(mod) if isinstance(unit.body, tirx.For)]

    assert len(loop_units) == 1
    assert _direct_schedule_unit_stages(loop_units[0].body.body) == [1, 0]


def test_auto_schedule_analyzes_dependencies_in_manual_stage_order():
    mod = _bind_target(_make_partially_staged_program())
    for transform in (
        ascend_transform.NormalizeControlFlowForSchedule,
        ascend_transform.NormalizeConflictHints,
        ascend_transform.MaterializeScheduleUnits,
        ascend_transform.AnnotateMultiBufferEligible,
        ascend_transform.EstimateLatency,
        ascend_transform.AutoSchedule,
    ):
        mod = transform()(mod)

    loop_units = [unit for unit in _collect_schedule_units(mod) if isinstance(unit.body, tirx.For)]
    assert len(loop_units) == 1
    # Stage-ordered dependency and eligibility analysis recovers the stage-0
    # write as the producer. AutoSchedule can therefore enable two versions and
    # emit the producer before the stage-1 consumer.
    assert _direct_schedule_unit_stages(loop_units[0].body.body) == [0, 1]


def test_auto_schedule_requires_estimate_latency():
    mod = _materialize_schedule_units(_bind_target(_make_program()))
    with pytest.raises(tvm.error.InternalError, match="EstimateLatency"):
        ascend_transform.AutoSchedule()(mod)


def test_auto_schedule_requires_materialized_schedule_units():
    mod = _bind_target(_make_program())
    with pytest.raises(tvm.error.InternalError, match="MaterializeScheduleUnits"):
        ascend_transform.AutoSchedule()(mod)


def test_auto_schedule_rejects_multiple_tasks_in_one_schedule_unit():
    mod = ascend_transform.EstimateLatency()(_materialize_schedule_units(_bind_target(_make_program())))
    malformed = _duplicate_one_task_schedule_unit(mod)

    with pytest.raises(tvm.error.InternalError, match="exactly one outer T.Task/T.PerCoreTask"):
        ascend_transform.AutoSchedule()(malformed)


def test_task_cost_arguments_are_validated():
    with pytest.raises(ValueError, match="non-negative integer"):
        _make_program(latency=-1)
    with pytest.raises(ValueError, match="positive integer"):
        _make_program(ii=0)
    with pytest.raises(ValueError, match="greater than or equal to ii"):
        _make_program(latency=1, ii=2)
    with pytest.raises(tvm.error.InternalError, match="greater than or equal to ii"):
        ascend_transform.EstimateLatency()(_materialize_schedule_units(_bind_target(_make_program(latency=0))))


if __name__ == "__main__":
    pytest.main([__file__])
