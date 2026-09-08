import pytest
import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.engine.lower import lower
from tvm import tirx


def _pass_snapshots(program, pass_names):
    snapshots = {name: [] for name in pass_names}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in snapshots:
                snapshots[info.name].append(mod.script())

    with tvm.transform.PassContext(opt_level=3, instruments=[Capture()]):
        lower(program, target="ascend")
    return snapshots


def _pass_script(program, pass_name):
    scripts = _pass_snapshots(program, {pass_name})[pass_name]
    assert len(scripts) == 1
    return scripts[0]


def _lower_without_resolve(program):
    mod = tvm.IRModule.from_expr(program.with_attr("global_symbol", "main"))
    mod = tirx.transform.BindTarget(determine_target("ascend"))(mod)
    for transform in (
        ascend_transform.NormalizeControlFlowForSchedule,
        ascend_transform.AnnotateMultiBufferEligible,
        ascend_transform.NormalizeNoConflictHints,
        ascend_transform.MaterializeScheduleUnits,
        ascend_transform.EstimateLatency,
        ascend_transform.AutoSchedule,
        ascend_transform.AssignCore,
        ascend_transform.PrepareMultiBuffer,
        ascend_transform.InsertSync,
        ascend_transform.MaterializeMultiBuffer,
        ascend_transform.LowerScheduledTIR,
    ):
        mod = transform()(mod)
    return mod.script()


def _make_sid_guarded_program(cube_guarded):
    tile = 16

    @T.prim_func
    def main(
        A: T.Buffer((tile, tile), "bfloat16"),
        B: T.Buffer((tile, tile), "bfloat16"),
        C: T.Buffer((tile,), "float32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            b_l1 = T.alloc_l1((tile, tile), "bfloat16")
            l0c = T.alloc_l0c((tile, tile), "float32")
            ub = T.alloc_shared((tile,), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            if cube_guarded:
                if sid == 0:
                    T.gemm(a_l1, b_l1, l0c, transpose_B=True, clear_accum=True)
            else:
                T.gemm(a_l1, b_l1, l0c, transpose_B=True, clear_accum=True)
                if sid == 0:
                    T.copy(C, ub)

    return main


def test_sid_cannot_control_cube_task():
    with pytest.raises(Exception, match="unavailable on every legal execution core"):
        _pass_script(_make_sid_guarded_program(cube_guarded=True), "tl.AssignCore")


def test_sid_remains_available_to_vector_task():
    lower(_make_sid_guarded_program(cube_guarded=False), target="ascend")


def _make_core_local_condition_program(use_ub):
    tile = 16

    @T.prim_func
    def main(
        pred: T.Buffer((1, tile), "int32"),
        A: T.Buffer((tile, tile), "bfloat16"),
        B: T.Buffer((tile, tile), "bfloat16"),
    ):
        with T.Kernel(1):
            pred_ub = T.alloc_shared((tile,), "int32")
            pred_l1 = T.alloc_l1((1, tile), "int32")
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            b_l1 = T.alloc_l1((tile, tile), "bfloat16")
            l0c = T.alloc_l0c((tile, tile), "float32")
            ub = T.alloc_shared((tile,), "int32")
            T.copy(pred[0, :], pred_ub)
            T.copy(pred, pred_l1)
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            if use_ub:
                active = T.bind(pred_ub[0] > 0)
                if active:
                    T.gemm(a_l1, b_l1, l0c, transpose_B=True, clear_accum=True)
            else:
                active = T.bind(pred_l1[0, 0] > 0)
                if active:
                    T.copy(pred_ub, ub)

    return main


@pytest.mark.parametrize("use_ub", [True, False], ids=["ub-to-cube", "l1-to-vector"])
def test_assign_rejects_cross_core_condition_producer(use_ub):
    with pytest.raises(Exception, match="unavailable on every legal execution core"):
        _pass_script(_make_core_local_condition_program(use_ub), "tl.AssignCore")


def _make_sid_guarded_shared_counter_program():
    tile = 16

    @T.prim_func
    def main(
        A: T.Buffer((2 * tile, tile), "bfloat16"),
        C: T.Buffer((2 * tile, tile), "bfloat16"),
    ):
        with T.MixedKernel(1) as (_, sid):
            l0c = T.alloc_l0c((tile, tile), "float32")
            ub = T.alloc_shared((tile, tile), "bfloat16")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for _vector_owner in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                if sid == 0:
                    T.copy(A[0:tile, :], ub)
                    T.copy(ub, C[0:tile, :])
            for _mixed_owner in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                if sid == 0:
                    T.copy(ub, C[tile : 2 * tile, :])

    return main


def test_resolve_rejects_sid_guard_unavailable_on_counter_protocol_core():
    with pytest.raises(
        tvm.error.InternalError,
        match="storage-epoch guard .* is unavailable on every legal execution core",
    ):
        _pass_script(_make_sid_guarded_shared_counter_program(), "tl.ResolveCore")


def _make_versioned_ub_assume_guarding_cube_loop():
    tile = 16

    @T.prim_func
    def main(
        pred: T.Buffer((2,), "int32"),
        A: T.Buffer((tile, tile), "bfloat16"),
        B: T.Buffer((tile, tile), "bfloat16"),
    ):
        with T.Kernel(1):
            pred_ub = T.alloc_shared((1,), "int32")
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            b_l1 = T.alloc_l1((tile, tile), "bfloat16")
            l0c = T.alloc_l0c((tile, tile), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.annotate_buffer_versions({pred_ub: (2, "counter")})
            for i in T.serial(2, annotations={"multi_buffer_eligible": [pred_ub]}):
                pred_ub[0] = pred[i]
                T.assume(pred_ub[0] >= 0)
                for _j in T.serial(1):
                    T.gemm(
                        a_l1,
                        b_l1,
                        l0c,
                        transpose_B=True,
                        clear_accum=i == 0,
                    )

    return main


def test_resolve_rejects_core_local_storage_in_control_assume():
    with pytest.raises(Exception, match="Cannot evaluate a loop bound or control guard"):
        _pass_script(_make_versioned_ub_assume_guarding_cube_loop(), "tl.ResolveCore")


def _make_ub_loop_extent_program():
    tile = 16

    @T.prim_func
    def main(
        pred: T.Buffer((1,), "int32"),
        A: T.Buffer((tile, tile), "bfloat16"),
        B: T.Buffer((tile, tile), "bfloat16"),
    ):
        with T.Kernel(1):
            pred_ub = T.alloc_shared((1,), "int32")
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            b_l1 = T.alloc_l1((tile, tile), "bfloat16")
            l0c = T.alloc_l0c((tile, tile), "float32")
            T.copy(pred, pred_ub)
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            for _i in T.serial(pred_ub[0]):
                T.gemm(a_l1, b_l1, l0c, transpose_B=True, clear_accum=True)

    return main


def test_resolve_rejects_core_local_loop_extent():
    with pytest.raises(Exception, match="Cannot evaluate a loop bound or control guard"):
        _pass_script(_make_ub_loop_extent_program(), "tl.ResolveCore")


def _make_ub_loop_extent_flexible_scalar_program(propagate_to_reader=False):
    tile = 16

    @T.prim_func
    def main(
        pred: T.Buffer((1,), "int32"),
        A: T.Buffer((tile, tile), "bfloat16"),
        B: T.Buffer((tile, tile), "bfloat16"),
        C: T.Buffer((tile,), "int32"),
    ):
        with T.Kernel(1):
            pred_ub = T.alloc_shared((1,), "int32")
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            b_l1 = T.alloc_l1((tile, tile), "bfloat16")
            l0c = T.alloc_l0c((tile, tile), "float32")
            scalar = T.alloc_var("int32", init=0)
            T.copy(pred, pred_ub)
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, l0c, transpose_B=True, clear_accum=True)
            for i in T.serial(pred_ub[0]):
                if propagate_to_reader:
                    scalar = i
                else:
                    C[i] = 0
            if propagate_to_reader:
                C[0] = scalar

    return main


@pytest.mark.parametrize("propagate_to_reader", [False, True], ids=["direct", "reader"])
def test_resolve_fallback_respects_control_expression_core(propagate_to_reader):
    lower(
        _make_ub_loop_extent_flexible_scalar_program(propagate_to_reader),
        target="ascend",
    )


def _make_ub_indexed_l1_copy_program():
    tile = 16

    @T.prim_func
    def main(
        offset: T.Buffer((1,), "int32"),
        A: T.Buffer((4 * tile, tile), "bfloat16"),
    ):
        with T.Kernel(1):
            offset_ub = T.alloc_shared((1,), "int32")
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            T.copy(offset, offset_ub)
            T.copy(A[offset_ub[0] : offset_ub[0] + tile, :], a_l1)

    return main


def test_assign_rejects_core_local_task_index():
    with pytest.raises(Exception, match="unavailable on every legal execution core"):
        _pass_script(_make_ub_indexed_l1_copy_program(), "tl.AssignCore")


def _make_ub_cross_core_flag_id_program():
    @T.prim_func
    def main(flag: T.Buffer((1,), "int32")):
        with T.Kernel(1):
            flag_ub = T.alloc_shared((1,), "int32")
            T.copy(flag, flag_ub)
            T.ascend_cross_core_set_flag(4, "PIPE_M", flag_ub[0])

    return main


def test_assign_rejects_cross_core_flag_id_from_wrong_core():
    with pytest.raises(Exception, match="unavailable on that core"):
        _pass_script(_make_ub_cross_core_flag_id_program(), "tl.AssignCore")


def _make_per_core_context_for_ambiguous_pipe_program():
    tile = 16

    @T.prim_func
    def main(A: T.Buffer((tile, tile), "bfloat16")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            with T.PerCoreTask(), T.Task():
                T.copy(A, a_l1)
            T.ascend_cross_core_set_flag(0, "PIPE_MTE2", 3)

    return main


def test_per_core_task_supplies_pure_kernel_context_for_ambiguous_pipe():
    source = lower(_make_per_core_context_for_ambiguous_pipe_program(), target="ascend").kernel_source
    assert "__global__ __cube__ void main_kernel" in source
    assert "asc_sync_inter_arrive(PIPE_MTE2, 3)" in source


def _make_sid_context_for_ambiguous_pipe_program():
    tile = 16

    @T.prim_func
    def main(
        A: T.Buffer((tile, tile), "bfloat16"),
        B: T.Buffer((tile, tile), "bfloat16"),
    ):
        with T.MixedKernel(1) as (_, sid):
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            b_l1 = T.alloc_l1((tile, tile), "bfloat16")
            l0c = T.alloc_l0c((tile, tile), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, l0c, transpose_B=True, clear_accum=True)
            if sid == 0:
                T.ascend_cross_core_set_flag(4, "PIPE_S", 3)

    return main


def test_task_guard_resolves_ambiguous_cross_core_pipe():
    source = lower(_make_sid_context_for_ambiguous_pipe_program(), target="ascend").kernel_source
    assert "asc_sync_intra_arrive(PIPE_S, 3)" in source


def _make_ancestor_controlled_ambiguous_pipe_program():
    tile = 16

    @T.prim_func
    def main(
        pred: T.Buffer((1,), "int32"),
        A: T.Buffer((tile, tile), "bfloat16"),
        B: T.Buffer((tile, tile), "bfloat16"),
    ):
        with T.MixedKernel(1):
            pred_ub = T.alloc_shared((1,), "int32")
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            b_l1 = T.alloc_l1((tile, tile), "bfloat16")
            l0c = T.alloc_l0c((tile, tile), "float32")
            T.copy(pred, pred_ub)
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, l0c, transpose_B=True, clear_accum=True)
            for _ in T.serial(pred_ub[0]):
                T.ascend_cross_core_set_flag(4, "PIPE_S", 3)

    return main


def test_ancestor_control_resolves_ambiguous_cross_core_pipe():
    source = lower(_make_ancestor_controlled_ambiguous_pipe_program(), target="ascend").kernel_source
    flag = source.index("asc_sync_intra_arrive(PIPE_S, 3)")
    assert source.rfind("if ASC_IS_AIV", 0, flag) > source.rfind("if ASC_IS_AIC", 0, flag)


def _make_broadcast_only_scalar_program():
    @T.prim_func
    def main(A: T.Buffer((1,), "int32"), C: T.Buffer((1,), "int32")):
        with T.Kernel(1):
            C[0] = A[0] + 1

    return main


def test_lower_broadcast_only_kernel_uses_vector_fallback():
    script = _lower_without_resolve(_make_broadcast_only_scalar_program())
    assert 'with T.sblock("VECTOR",' in script
    assert "C[0] = A[0] + 1" in script


if __name__ == "__main__":
    tilelang.testing.main()
