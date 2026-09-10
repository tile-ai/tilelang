import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.backend.target import determine_target
from tvm import tirx


def _annotated_script(program):
    mod = tvm.IRModule.from_expr(program.with_attr("global_symbol", "main"))
    mod = tirx.transform.BindTarget(determine_target("ascend"))(mod)
    for transform in (
        ascend_transform.NormalizeControlFlowForSchedule,
        ascend_transform.NormalizeConflictHints,
        ascend_transform.MaterializeScheduleUnits,
        ascend_transform.AnnotateMultiBufferEligible,
    ):
        mod = transform()(mod)
    return mod.script()


def _owner_line(program, loop_var="i"):
    script = _annotated_script(program)
    return next(line for line in script.splitlines() if f"for {loop_var}" in line and "multi_buffer_eligible" in line)


def _make_reordered_stage_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.Pipelined(
                4,
                num_stages=2,
                annotations={"enable_offset": True},
            ):
                with T.Stage(1):
                    T.copy(ub, C[i * tile : (i + 1) * tile])
                with T.Stage(0):
                    T.copy(A[i * tile : (i + 1) * tile], ub)

    return main


def _make_equivalent_guard_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(4):
                if pred > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                if pred > 0:
                    T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_exhaustive_guard_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        B: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(4):
                if pred > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                if pred <= 0:
                    T.copy(B[i * tile : (i + 1) * tile], ub)
                T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_disjoint_guard_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(4):
                if pred > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                if pred <= 0:
                    T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_vf_internal_write_first_program():
    @T.prim_func
    def main():
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            for _i in T.serial(4):
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vdup(T.float32(0), "float32", mask)
                    T.simd.vsts(ub[0], value, mask)
                    loaded = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], loaded, mask)

    return main


def _make_explicit_and_automatic_sibling_program():
    @T.prim_func
    def main(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((1,), "float32")
            for i in T.serial(
                2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                ub[0] = A[i]
                C[i] = ub[0]
            for j in T.serial(2):
                ub[0] = A[j + 2]
                C[j + 2] = ub[0]

    return main


def test_manual_stage_order_drives_write_first_analysis():
    assert "ub" in _owner_line(_make_reordered_stage_program())


def test_equivalent_flattened_guards_preserve_write_first_order():
    assert "ub" in _owner_line(_make_equivalent_guard_program())


def test_opposite_flattened_guards_cover_unconditional_read():
    assert "ub" in _owner_line(_make_exhaustive_guard_program())


def test_disjoint_flattened_guards_remain_read_first():
    assert "ub" not in _owner_line(_make_disjoint_guard_program())


def test_vf_task_is_classified_from_internal_statement_order():
    assert "ub" in _owner_line(_make_vf_internal_write_first_program(), loop_var="_i")


def test_explicit_claim_disables_automatic_analysis_for_storage():
    script = _annotated_script(_make_explicit_and_automatic_sibling_program())
    explicit = next(line for line in script.splitlines() if "for i" in line)
    automatic = next(line for line in script.splitlines() if "for j" in line)
    assert "ub" in explicit
    assert "ub" not in automatic
