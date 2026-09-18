import re

import pytest
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm

from tilelang.engine.lower import lower


TILE = 256
FLAG = 3


def _core_branch(source: str, token: str) -> str:
    position = source.index(token)
    aic = source.rfind("if ASC_IS_AIC", 0, position)
    aiv = source.rfind("if ASC_IS_AIV", 0, position)
    if max(aic, aiv) >= 0:
        return "AIC" if aic > aiv else "AIV"

    function_start = source.rfind('extern "C"', 0, position)
    signature = source[function_start : source.find("{", function_start)]
    if "__cube__" in signature:
        return "AIC"
    if "__vector__" in signature:
        return "AIV"
    raise AssertionError(f"cannot determine core branch for {token}")


def _make_per_core_mte2_program(to_l1: bool, mode: int = 0):
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        X: T.Tensor((64,), "float32"),
        C: T.Tensor((TILE, TILE), "float32"),
        Y: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            x_ub = T.alloc_shared((64,), "float32")
            if to_l1:
                with T.PerCoreTask():
                    with T.Task():
                        T.copy(A, a_l1)
                    T.ascend_cross_core_set_flag(mode, "PIPE_MTE2", FLAG)
                    T.ascend_cross_core_wait_flag(mode, "PIPE_MTE2", FLAG)
            else:
                T.copy(A, a_l1)
            T.copy(B, b_l1)
            if not to_l1:
                with T.PerCoreTask():
                    with T.Task():
                        T.copy(X, x_ub)
                    T.ascend_cross_core_set_flag(mode, "PIPE_MTE2", FLAG)
                    T.ascend_cross_core_wait_flag(mode, "PIPE_MTE2", FLAG)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, C)
            if to_l1:
                T.copy(X, x_ub)
            T.copy(x_ub, Y)

    return main


def test_per_core_mte2_uses_copy_destination_core():
    marker = f"asc_sync_inter_wait(PIPE_MTE2, {FLAG});"
    for to_l1, expected in ((True, "AIC"), (False, "AIV")):
        source = lower(_make_per_core_mte2_program(to_l1, 0), target="ascend").kernel_source
        assert _core_branch(source, marker) == expected


def _make_supported_cross_core_mode_program():
    @T.prim_func
    def main():
        with T.Kernel(1):
            # Flag IDs share one hardware domain across modes. The concrete
            # PIPE_V endpoint places the ambiguous PIPE_MTE2 endpoint on AIV.
            T.ascend_cross_core_set_flag(0, "PIPE_MTE2", FLAG)
            T.ascend_cross_core_set_flag(1, "PIPE_V", FLAG)
            T.ascend_cross_core_set_flag(2, "PIPE_MTE2", FLAG)
            T.ascend_cross_core_set_flag(4, "PIPE_V", FLAG)

    return main


def test_cross_core_placement_accepts_supported_modes():
    source = lower(_make_supported_cross_core_mode_program(), target="ascend").kernel_source

    sync_mode_0 = f"asc_sync_inter_arrive(PIPE_MTE2, {FLAG});"
    sync_mode_1 = f"asc_sync_subblock_arrive(PIPE_V, {FLAG});"
    sync_mode_2 = f"asc_sync_block_arrive(PIPE_MTE2, {FLAG});"
    sync_mode_4 = f"asc_sync_intra_arrive(PIPE_V, {FLAG});"

    assert _core_branch(source, sync_mode_0) == "AIV"
    assert _core_branch(source, sync_mode_1) == "AIV"
    assert _core_branch(source, sync_mode_2) == "AIV"
    assert _core_branch(source, sync_mode_4) == "AIV"


def _make_per_core_candidate_program(explicit: bool, leave_second_unmarked: bool = False):
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
    ):
        with T.Kernel(2) as bx:
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            a_l0 = T.alloc_l0a((TILE, TILE), "bfloat16")
            with T.PerCoreTask():
                if bx == 0:
                    if explicit:
                        with T.Task():
                            T.copy(A, a_l1)
                    else:
                        T.copy(A, a_l1)
                if bx != 0:
                    if explicit and not leave_second_unmarked:
                        with T.Task():
                            T.copy(B, a_l1)
                    else:
                        T.copy(B, a_l1)
            T.copy(a_l1, a_l0)

    return main


def _assert_per_core_mte2_mte1_sync(source: str):
    set_flag = "asc_sync_notify(PIPE_MTE2, PIPE_MTE1,"
    wait_flag = "asc_sync_wait(PIPE_MTE2, PIPE_MTE1,"
    assert source.count(set_flag) == 2
    assert source.count(wait_flag) == 1
    candidate_branches = re.findall(r"  if \([^\n]*block_idx[^\n]*\) \{\n(.*?)\n  \}", source, re.DOTALL)
    assert len(candidate_branches) == 2
    assert all(set_flag in branch for branch in candidate_branches)


def test_per_core_explicit_tasks_drive_sync_migration():
    source = lower(_make_per_core_candidate_program(explicit=True), target="ascend").kernel_source
    _assert_per_core_mte2_mte1_sync(source)


def test_per_core_syncs_preserve_candidate_task_markers():
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureScheduledIR:
        def run_after_pass(self, mod, info):
            if info.name in {"tl.AutoSchedule", "tl.AssignCore", "tl.InsertSync"}:
                snapshots.setdefault(info.name, []).append(mod.script())

    with tvm.transform.PassContext(opt_level=3, instruments=[CaptureScheduledIR()]):
        tilelang.lower(_make_per_core_candidate_program(explicit=True), target="ascend")

    scheduled = snapshots["tl.AutoSchedule"][0]
    assert len(snapshots["tl.AssignCore"]) == 1
    first_assigned = snapshots["tl.AssignCore"][0]
    synchronized = snapshots["tl.InsertSync"][0]
    assert '"core_mask"' not in scheduled
    assert '"core_mask"' in first_assigned
    assert synchronized.count('"tl.ascend_per_core_task"') == 1
    # InsertSync wraps each migrated synchronization statement in its own
    # compiler-generated T.Task carrying compiler-owned core metadata.
    assert synchronized.count('"tl.ascend_task"') > first_assigned.count('"tl.ascend_task"')
    assert '"core_mask"' in synchronized
    assert "T.ascend_set_flag" in synchronized


def test_per_core_implicit_candidate_inference_remains_supported():
    source = lower(_make_per_core_candidate_program(explicit=False), target="ascend").kernel_source
    _assert_per_core_mte2_mte1_sync(source)


def test_per_core_mixes_explicit_and_inferred_tasks():
    program = _make_per_core_candidate_program(explicit=True, leave_second_unmarked=True)
    source = lower(program, target="ascend").kernel_source
    _assert_per_core_mte2_mte1_sync(source)


def _make_per_core_dual_copy_store_program():
    @T.prim_func
    def main(C: T.Tensor((TILE, TILE), "float32")):
        with T.Kernel(1):
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.dual_copy(accum, temp)
            with T.PerCoreTask(), T.Task():
                T.dual_copy(temp, C)

    return main


def test_per_core_task_rewrites_standalone_ub_to_gm_dual_copy():
    source = lower(_make_per_core_dual_copy_store_program(), target="ascend").kernel_source
    store = re.search(r"asc_copy_ub2gm_align\(\(__gm__ uint8_t\*\)\(([^,]+)\),", source)
    assert store is not None
    assert "sid" in store.group(1)


def test_per_core_task_rejects_nesting_inside_task():
    @T.prim_func
    def main(
        A: T.Tensor((16,), "float32"),
        B: T.Tensor((16,), "float32"),
    ):
        with T.Kernel(1):
            temp = T.alloc_shared((16,), "float32")
            with T.Task(), T.PerCoreTask():
                T.copy(A, temp)
            T.copy(temp, B)

    with pytest.raises(tvm.error.InternalError, match=r"T\.PerCoreTask cannot be nested inside T\.Task"):
        lower(main, target="ascend")


if __name__ == "__main__":
    tilelang.testing.main()
