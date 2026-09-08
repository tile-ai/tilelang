import re

import pytest
import tilelang.ascend.language as T
import tilelang.testing
import tvm
from tilelang.ascend import transform as ascend_transform
from tilelang.engine.lower import lower


def _make_many_flags_program(annotate_versions: bool = True):
    tile = 64
    stages = 2

    @T.prim_func
    def main(A: T.Buffer((4096,), "float32"), C: T.Buffer((4096,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            if annotate_versions:
                T.annotate_buffer_versions({ub: stages})

            for w in T.Pipelined(8, num_stages=stages):
                base = w * 5 * tile

                T.copy(A[base : base + tile], ub)
                with T.SimdVF():
                    mask0 = T.simd.pset(32)
                    value0 = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value0, mask0)
                T.copy(ub, C[base : base + tile])

                T.copy(A[base + tile : base + 2 * tile], ub)
                with T.SimdVF():
                    mask1 = T.simd.pset(32)
                    value1 = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value1, mask1)
                T.copy(ub, C[base + tile : base + 2 * tile])

                T.copy(A[base + 2 * tile : base + 3 * tile], ub)
                with T.SimdVF():
                    mask2 = T.simd.pset(32)
                    value2 = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value2, mask2)
                T.copy(ub, C[base + 2 * tile : base + 3 * tile])

                T.copy(A[base + 3 * tile : base + 4 * tile], ub)
                with T.SimdVF():
                    mask3 = T.simd.pset(32)
                    value3 = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value3, mask3)
                T.copy(ub, C[base + 3 * tile : base + 4 * tile])

                T.copy(A[base + 4 * tile : base + 5 * tile], ub)
                with T.SimdVF():
                    mask4 = T.simd.pset(32)
                    value4 = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value4, mask4)
                T.copy(ub, C[base + 4 * tile : base + 5 * tile])

    return main


def _make_lexical_flag_reuse_program():
    # Keep lexical flag-reuse coverage independent of counter multi-buffering.
    tile = 64
    stages = 2

    @T.macro
    def protocol(A, C, ub, offset):
        T.copy(A[offset : offset + tile], ub)
        with T.SimdVF():
            mask = T.simd.pset(32)
            value = T.simd.vld(ub[0])
            T.simd.vsts(ub[0], value, mask)
        T.copy(ub, C[offset : offset + tile])

    @T.prim_func
    def main(A: T.Buffer((4096,), "float32"), C: T.Buffer((4096,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: stages})

            for w in T.Pipelined(4, num_stages=stages):
                base = w * 9 * tile
                protocol(A, C, ub, base)
                protocol(A, C, ub, base + tile)
                protocol(A, C, ub, base + 2 * tile)
                protocol(A, C, ub, base + 3 * tile)
                protocol(A, C, ub, base + 4 * tile)
                protocol(A, C, ub, base + 5 * tile)
                protocol(A, C, ub, base + 6 * tile)
                protocol(A, C, ub, base + 7 * tile)
                protocol(A, C, ub, base + 8 * tile)

    return main


def _make_cross_iter_reuse_program():
    tile = 64
    stages = 2
    inner = 4

    @T.prim_func
    def main(A: T.Buffer((4096,), "float32"), C: T.Buffer((4096,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: stages})

            for w in T.Pipelined(4, num_stages=stages):
                base = w * 4 * inner * tile

                for i in T.serial(inner):
                    offset = base + i * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask0 = T.simd.pset(32)
                        value0 = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value0, mask0)
                    T.copy(ub, C[offset : offset + tile])

                for i in T.serial(inner):
                    offset = base + (inner + i) * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask1 = T.simd.pset(32)
                        value1 = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value1, mask1)
                    T.copy(ub, C[offset : offset + tile])

                for i in T.serial(inner):
                    offset = base + (2 * inner + i) * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask2 = T.simd.pset(32)
                        value2 = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value2, mask2)
                    T.copy(ub, C[offset : offset + tile])

                for i in T.serial(inner):
                    offset = base + (3 * inner + i) * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask3 = T.simd.pset(32)
                        value3 = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value3, mask3)
                    T.copy(ub, C[offset : offset + tile])

    return main


def _make_multi_owner_counter_channel_program():
    tile = 64
    versions = 8
    iterations = 8

    @T.macro
    def protocol(A, C, ub, offset, row):
        T.copy(A[offset : offset + tile], ub[row, :])
        with T.SimdVF():
            mask = T.simd.pset(32)
            value = T.simd.vld(ub[row, 0])
            T.simd.vsts(ub[row, 0], value, mask)
        T.copy(ub[row, :], C[offset : offset + tile])

    @T.prim_func
    def main(
        A: T.Buffer((4 * iterations * tile,), "float32"),
        C: T.Buffer((4 * iterations * tile,), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((2, tile), "float32")
            T.annotate_buffer_versions({ub: (versions, "counter")})

            for i in T.Pipelined(
                iterations,
                num_stages=versions,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                base = i * 2 * tile
                protocol(A, C, ub, base, 0)
                protocol(A, C, ub, base + tile, 1)

            for j in T.Pipelined(
                iterations,
                num_stages=versions,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                base = (2 * iterations + j * 2) * tile
                protocol(A, C, ub, base, 0)
                protocol(A, C, ub, base + tile, 1)

    return main


def _make_mixed_cross_iter_versions_program():
    tile = 64
    inner = 6
    outer = 4
    segment = inner * tile

    @T.prim_func
    def main(
        state: T.Buffer(((outer + 1) * segment,), "float32"),
        middle: T.Buffer((outer * segment,), "float32"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            ub2 = T.alloc_shared((tile,), "float32")
            ub3 = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub2: 2, ub3: 3})

            for w in T.Pipelined(outer, num_stages=3):
                if enabled > 0:
                    for i in T.serial(inner):
                        offset = w * segment + i * tile
                        T.copy(state[offset : offset + tile], ub2)
                        with T.SimdVF():
                            mask0 = T.simd.pset(32)
                            value0 = T.simd.vld(ub2[0])
                            T.simd.vsts(ub2[0], value0, mask0)
                        T.copy(ub2, middle[offset : offset + tile])

                if enabled > 0:
                    for i in T.serial(inner):
                        offset = w * segment + i * tile
                        next_offset = (w + 1) * segment + i * tile
                        T.copy(middle[offset : offset + tile], ub3)
                        with T.SimdVF():
                            mask1 = T.simd.pset(32)
                            value1 = T.simd.vld(ub3[0])
                            T.simd.vsts(ub3[0], value1, mask1)
                        T.copy(ub3, state[next_offset : next_offset + tile])

    return main


def _make_mixed_version_same_iter_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((1024,), "float32"),
        C: T.Buffer((1024,), "float32"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})

            if enabled > 0:
                for w in T.Pipelined(4, num_stages=2):
                    offset = w * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask0 = T.simd.pset(32)
                        value0 = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value0, mask0)
                    T.copy(ub, C[offset : offset + tile])

            T.copy(A[4 * tile : 5 * tile], ub)
            with T.SimdVF():
                mask1 = T.simd.pset(32)
                value1 = T.simd.vld(ub[0])
                T.simd.vsts(ub[0], value1, mask1)
            T.copy(ub, C[4 * tile : 5 * tile])

    return main


def _make_inner_loop_program(extent: int):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(extent):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
            T.copy(ub, C)

    return main


def _make_single_iteration_nested_loop_program():
    tile = 64
    stages = 3

    @T.prim_func
    def main(A: T.Buffer((8 * tile,), "float32"), C: T.Buffer((8 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: stages})

            for w in T.Pipelined(8, num_stages=stages):
                for _i in T.serial(1):
                    offset = w * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[offset : offset + tile])

    return main


def _make_noneligible_single_iteration_nested_loop_program():
    tile = 64
    stages = 3

    @T.prim_func
    def main(A: T.Buffer((16 * tile,), "float32"), C: T.Buffer((8 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: stages})

            for w in T.Pipelined(8, num_stages=stages):
                T.copy(A[2 * w * tile : (2 * w + 1) * tile], ub)
                for _i in T.serial(1):
                    T.copy(A[(2 * w + 1) * tile : (2 * w + 2) * tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[w * tile : (w + 1) * tile])

    return main


def _make_unavoidable_overflow_program():
    tile = 64
    stages = 2
    load_cost = {"latency": 95, "ii": 3}
    store_cost = {"latency": 185, "ii": 3}

    @T.prim_func
    def main(A: T.Buffer((4096,), "float32"), C: T.Buffer((4096,), "float32")):
        with T.Kernel(1):
            ub0 = T.alloc_shared((tile,), "float32")
            ub1 = T.alloc_shared((tile,), "float32")
            ub2 = T.alloc_shared((tile,), "float32")
            ub3 = T.alloc_shared((tile,), "float32")
            ub4 = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub0: stages, ub1: stages, ub2: stages, ub3: stages, ub4: stages})

            for w in T.Pipelined(8, num_stages=stages):
                base = w * 5 * tile

                with T.Task(**load_cost):
                    T.copy(A[base : base + tile], ub0)
                with T.SimdVF():
                    mask0 = T.simd.pset(32)
                    value0 = T.simd.vld(ub0[0])
                    T.simd.vsts(ub0[0], value0, mask0)
                with T.Task(**store_cost):
                    T.copy(ub0, C[base : base + tile])

                with T.Task(**load_cost):
                    T.copy(A[base + tile : base + 2 * tile], ub1)
                with T.SimdVF():
                    mask1 = T.simd.pset(32)
                    value1 = T.simd.vld(ub1[0])
                    T.simd.vsts(ub1[0], value1, mask1)
                with T.Task(**store_cost):
                    T.copy(ub1, C[base + tile : base + 2 * tile])

                with T.Task(**load_cost):
                    T.copy(A[base + 2 * tile : base + 3 * tile], ub2)
                with T.SimdVF():
                    mask2 = T.simd.pset(32)
                    value2 = T.simd.vld(ub2[0])
                    T.simd.vsts(ub2[0], value2, mask2)
                with T.Task(**store_cost):
                    T.copy(ub2, C[base + 2 * tile : base + 3 * tile])

                with T.Task(**load_cost):
                    T.copy(A[base + 3 * tile : base + 4 * tile], ub3)
                with T.SimdVF():
                    mask3 = T.simd.pset(32)
                    value3 = T.simd.vld(ub3[0])
                    T.simd.vsts(ub3[0], value3, mask3)
                with T.Task(**store_cost):
                    T.copy(ub3, C[base + 3 * tile : base + 4 * tile])

                with T.Task(**load_cost):
                    T.copy(A[base + 4 * tile : base + 5 * tile], ub4)
                with T.SimdVF():
                    mask4 = T.simd.pset(32)
                    value4 = T.simd.vld(ub4[0])
                    T.simd.vsts(ub4[0], value4, mask4)
                with T.Task(**store_cost):
                    T.copy(ub4, C[base + 4 * tile : base + 5 * tile])

    return main


def _make_unapplied_buffer_version_annotation_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((tile,), "float32"), C: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})

            T.copy(A, ub)
            for w in T.Pipelined(4, num_stages=2):
                T.copy(ub, C[w * tile : (w + 1) * tile])

    return main


def _make_same_name_buffer_version_annotation_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((8 * tile,), "float32"), C: T.Buffer((8 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})

            T.copy(A[:tile], ub)
            for w in T.Pipelined(4, num_stages=2):
                T.copy(ub, C[w * tile : (w + 1) * tile])

            ub = T.alloc_shared((tile,), "float32")
            for w in T.Pipelined(4, num_stages=2):
                T.copy(A[(w + 4) * tile : (w + 5) * tile], ub)
                T.copy(ub, C[(w + 4) * tile : (w + 5) * tile])

    return main


def _make_bound_flag_spill_program():
    @T.prim_func
    def main():
        with T.Kernel(1):
            for outer in T.Serial(2):
                for inner in T.Serial(2):
                    event_id = T.bind(outer * 2 + inner)
                    T.ascend_set_flag("MTE2_V", event_id)
                    T.ascend_wait_flag("MTE2_V", event_id)
            T.ascend_set_flag("MTE2_V", 8)
            T.ascend_wait_flag("MTE2_V", 8)
            T.ascend_set_flag("MTE2_V", 10)
            T.ascend_wait_flag("MTE2_V", 10)
            T.ascend_set_flag("MTE2_V", 12)
            T.ascend_wait_flag("MTE2_V", 12)
            T.ascend_set_flag("MTE2_V", 14)
            T.ascend_wait_flag("MTE2_V", 14)
            T.ascend_set_flag("MTE2_V", 16)
            T.ascend_wait_flag("MTE2_V", 16)

    return main


def test_rewrite_flag_to_buf_tracks_bind_context():
    mod = tvm.IRModule({"main": _make_bound_flag_spill_program()})
    rewritten = ascend_transform.RewriteFlagToBuf()(mod)

    assert "T.ascend_get_buf" in rewritten["main"].script()


def test_multi_buffer_metadata_flows_through_split_passes():
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureScheduledIR:
        def run_after_pass(self, mod, info):
            if info.name in {
                "tl.AutoSchedule",
                "tl.AssignCore",
                "tl.PrepareMultiBuffer",
                "tl.ResolveCore",
                "tl.InsertSync",
                "tl.MaterializeMultiBuffer",
                "tl.LowerScheduledTIR",
            }:
                snapshots.setdefault(info.name, []).append(mod.script())

    with tvm.transform.PassContext(opt_level=3, instruments=[CaptureScheduledIR()]):
        lower(_make_many_flags_program(), target="ascend")

    scheduled = snapshots["tl.AutoSchedule"][0]
    assert scheduled.count("tl.buffer_versions_map") == 1
    assert scheduled.count("multi_buffer_eligible") == 1
    assert "tl.auto_schedule.buffer_num_versions" not in scheduled

    assignments = snapshots["tl.AssignCore"]
    assert len(assignments) == 1
    first_assigned = assignments[0]
    assert first_assigned.count("tl.buffer_versions_map") == 1
    assert first_assigned.count("multi_buffer_eligible") == 1

    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    assert prepared.count("tl.buffer_versions_map") == 1
    assert prepared.count("multi_buffer_eligible") == 1

    synchronized = snapshots["tl.InsertSync"][0]
    assert synchronized.count("tl.buffer_versions_map") == 1
    assert synchronized.count("multi_buffer_eligible") == 1

    materialized = snapshots["tl.MaterializeMultiBuffer"][0]
    assert "tl.buffer_versions_map" not in materialized
    assert materialized.count("tl.manual_multi_buffer") == 1
    assert materialized.count("multi_buffer_eligible") == 1

    lowered = snapshots["tl.LowerScheduledTIR"][0]
    assert "multi_buffer_eligible" not in lowered


def test_auto_schedule_stages_solver_selected_buffer_versions_globally():
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureScheduledIR:
        def run_after_pass(self, mod, info):
            if info.name in {"tl.AutoSchedule", "tl.AssignCore"}:
                snapshots.setdefault(info.name, []).append(mod.script())

    with tvm.transform.PassContext(opt_level=3, instruments=[CaptureScheduledIR()]):
        lower(_make_many_flags_program(annotate_versions=False), target="ascend")

    assert len(snapshots["tl.AssignCore"]) == 1
    for snapshot in (snapshots["tl.AutoSchedule"][0], snapshots["tl.AssignCore"][0]):
        assert snapshot.count("tl.buffer_versions_map") == 1
        assert "tl.auto_schedule.buffer_num_versions" not in snapshot
        assert snapshot.count("multi_buffer_eligible") == 1


def _make_cross_core_reuse_program(
    versions: int,
    add_vector_stage: bool = False,
    reserve_last_slot: bool = False,
    counter: bool = False,
):
    tile_m = 128
    tile_n = 128
    tile_k = 64
    tiles = 8

    @T.prim_func
    def main(
        A: T.Buffer((tiles * tile_m, tile_k), "bfloat16"),
        B: T.Buffer((tile_n, tile_k), "bfloat16"),
        C: T.Buffer((tiles * tile_m, tile_n), "float32"),
    ):
        with T.Kernel(1) as bx:
            a_l1 = T.alloc_l1((tile_m, tile_k), "bfloat16")
            b_l1 = T.alloc_l1((tile_n, tile_k), "bfloat16")
            accum = T.alloc_l0c((tile_m, tile_n), "float32")
            output = T.alloc_shared((tile_m // 2, tile_n), "float32")
            T.annotate_buffer_versions({output: (versions, "counter") if counter else versions})

            for tile in T.Persistent([tiles], 1, bx):
                T.copy(A[tile * tile_m : (tile + 1) * tile_m, :], a_l1)
                T.copy(B[:, :], b_l1)
                T.gemm(
                    a_l1,
                    b_l1,
                    accum,
                    transpose_B=True,
                    clear_accum=True,
                    unit_flag_ctrl=3,
                )
                if reserve_last_slot:
                    with T.PerCoreTask():
                        T.dual_copy(accum, output, unit_flag_ctrl=3)
                        T.ascend_sync_inter_arrive("PIPE_FIX", 15)
                        T.ascend_sync_inter_wait("PIPE_FIX", 15)
                else:
                    T.dual_copy(accum, output, unit_flag_ctrl=3)
                if add_vector_stage:
                    with T.SimdVF():
                        for i, j in T.Parallel(tile_m // 2, tile_n):
                            output[i, j] = output[i, j] + 1.0
                T.dual_copy(output, C[tile * tile_m : (tile + 1) * tile_m, :])

    return main


def _constant_flag_ids(source: str, hard_event: str) -> set[int]:
    source_pipe, target_pipe = hard_event.split("_", maxsplit=1)
    pattern = re.compile(
        rf"asc_sync_(?:notify|wait)\(PIPE_{source_pipe}, PIPE_{target_pipe}, "
        rf"static_cast<event_t>\(([0-9]+)\)\)"
    )
    return {int(value) for value in pattern.findall(source)}


def _initial_flag_ids(source: str, hard_event: str) -> set[int]:
    kernel_body = source.split("__global__", maxsplit=1)[1]
    prologue = kernel_body.split("for (", maxsplit=1)[0]
    return _constant_flag_ids(prologue, hard_event)


def _constant_cross_core_flag_ids(source: str) -> set[int]:
    pattern = re.compile(r"asc_sync_intra_(?:arrive|wait)\(PIPE_[A-Z0-9]+, (\d+)\)")
    return {int(value) for value in pattern.findall(source)}


def _constant_tir_flag_ids(source: str, hard_event: str) -> set[int]:
    pattern = re.compile(rf'T\.ascend_(?:set|wait)_flag\("{hard_event}", (\d+)\)')
    return {int(value) for value in pattern.findall(source)}


def test_auto_schedule_keeps_distinct_flags_within_limit():
    artifact = lower(_make_many_flags_program(), target="ascend")
    source = artifact.kernel_source

    for hard_event in ("MTE2_V", "V_MTE2", "MTE3_V", "V_MTE3"):
        ids = _constant_flag_ids(source, hard_event)
        assert not ids or max(ids) < 8, f"{hard_event} exceeds the dav-3510 flag limit: {sorted(ids)}"

    # Five ids already fit, so no otherwise-safe reuse is introduced.
    assert _constant_flag_ids(source, "MTE2_V") == set(range(5))


def test_counter_sibling_protocol_stays_within_flag_limit():
    artifact = lower(_make_cross_iter_reuse_program(), target="ascend")
    source = artifact.kernel_source

    ids = _constant_flag_ids(source, "MTE3_MTE2")
    assert ids
    assert max(ids) < 8
    events = re.findall(
        r"asc_sync_(?:notify|wait)\(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>\(([^;]+)\)\);",
        source,
    )
    dynamic = [event for event in events if "version_counter" in event]
    assert len(dynamic) == 8
    assert len(set(dynamic)) == 1


def test_multi_owner_counter_channels_keep_exclusive_allocations():
    snapshots = []

    @tvm.ir.instrument.pass_instrument
    class CaptureInsertSync:
        def run_after_pass(self, mod, info):
            if info.name == "tl.InsertSync":
                snapshots.append(mod.script())

    with tvm.transform.PassContext(opt_level=3, instruments=[CaptureInsertSync()]):
        source = lower(_make_multi_owner_counter_channel_program(), target="ascend").kernel_source

    # The two disjoint regions form distinct physical counter channels, and
    # each channel is shared by both sibling owner loops. Each complete channel
    # keeps an exclusive allocation. InsertSync primes and drains cross-iteration
    # rings with one constant ID per version, so the two rings span 0..15 before
    # RewriteFlagToBuf spills the second eight-slot ring.
    assert len(snapshots) == 1
    assert _constant_tir_flag_ids(snapshots[0], "MTE3_MTE2") == set(range(16))
    assert "asc_lock(" in source
    assert "asc_unlock(" in source


def test_auto_schedule_reuses_lexical_flags_only_until_within_limit():
    artifact = lower(_make_lexical_flag_reuse_program(), target="ascend")
    ids = _constant_flag_ids(artifact.kernel_source, "MTE2_V")

    # Nine affine protocols exceed the eight-id hardware limit. The lexical
    # lifetime proof performs the first sufficient safe merge, then stops.
    assert ids == set(range(8))


def test_cross_iter_does_not_reuse_when_within_limit():
    artifact = lower(_make_mixed_cross_iter_versions_program(), target="ascend")
    ids = _initial_flag_ids(artifact.kernel_source, "MTE3_MTE2")

    # All three allocations stay independent because their six ids already fit.
    assert ids == set(range(6))


def test_same_iter_does_not_reuse_when_within_limit():
    artifact = lower(_make_mixed_version_same_iter_program(), target="ascend")
    ids = _constant_flag_ids(artifact.kernel_source, "MTE2_V")

    # The two allocations stay independent because they already fit.
    assert ids == {0, 1}


@pytest.mark.parametrize("versions", [4, 8])
def test_cross_core_handshake_does_not_reuse_within_limit(versions):
    artifact = lower(_make_cross_core_reuse_program(versions), target="ascend")
    source = artifact.kernel_source

    ids = _constant_cross_core_flag_ids(source)
    assert {event_id % 16 for event_id in ids} == set(range(versions))

    dynamic_calls = "\n".join(line for line in source.splitlines() if "asc_sync_intra_" in line and re.search(r"\bw(?:_\d+)?\b", line))
    assert "PIPE_FIX" in dynamic_calls
    assert "PIPE_MTE3" in dynamic_calls
    assert re.search(rf"\+ (?:{versions}|{versions + 16})\)", dynamic_calls)


def test_cross_core_handshake_reuses_when_over_limit():
    versions = 15
    artifact = lower(_make_cross_core_reuse_program(versions), target="ascend")
    source = artifact.kernel_source

    ids = _constant_cross_core_flag_ids(source)
    assert {event_id % 16 for event_id in ids} == set(range(versions))

    dynamic_calls = "\n".join(line for line in source.splitlines() if "asc_sync_intra_" in line and re.search(r"\bw(?:_\d+)?\b", line))
    assert "PIPE_FIX" in dynamic_calls
    assert "PIPE_MTE3" in dynamic_calls
    assert not re.search(rf"\+ (?:{versions}|{versions + 16})\)", dynamic_calls)


def test_cross_core_counter_handshake_reuses_when_over_limit():
    versions = 15
    artifact = lower(_make_cross_core_reuse_program(versions, counter=True), target="ascend")
    source = artifact.kernel_source

    ids = _constant_cross_core_flag_ids(source)
    assert {event_id % 16 for event_id in ids} == set(range(versions))

    dynamic_calls = "\n".join(line for line in source.splitlines() if "asc_sync_intra_" in line and "version_counter" in line)
    assert "PIPE_FIX" in dynamic_calls
    assert "PIPE_MTE3" in dynamic_calls
    assert not re.search(rf"\+ (?:{versions}|{versions + 16})\)", dynamic_calls)


def test_cross_core_handshake_with_different_pipes_reuses_when_over_limit():
    versions = 8
    artifact = lower(
        _make_cross_core_reuse_program(
            versions,
            add_vector_stage=True,
            reserve_last_slot=True,
        ),
        target="ascend",
    )
    source = artifact.kernel_source

    assert "asc_sync_inter_arrive(PIPE_FIX, 15);" in source
    ids = _constant_cross_core_flag_ids(source)
    assert {event_id % 16 for event_id in ids} == set(range(versions))

    dynamic_calls = "\n".join(line for line in source.splitlines() if "asc_sync_intra_" in line and re.search(r"\bw(?:_\d+)?\b", line))
    assert "PIPE_FIX" in dynamic_calls
    assert "PIPE_V" in dynamic_calls
    assert "PIPE_MTE3" in dynamic_calls
    assert not re.search(rf"\+ (?:{versions}|{versions + 16})\)", dynamic_calls)


def test_single_iteration_loop_has_no_reverse_flag():
    one_iter_source = lower(_make_inner_loop_program(1), target="ascend").kernel_source
    many_iter_source = lower(_make_inner_loop_program(4), target="ascend").kernel_source

    reverse_flag = "asc_sync_notify(PIPE_V, PIPE_MTE2,"
    assert reverse_flag not in one_iter_source
    assert reverse_flag in many_iter_source


def test_nested_single_iteration_loop_uses_outer_multi_buffer_versions():
    artifact = lower(_make_single_iteration_nested_loop_program(), target="ascend")
    ids = _initial_flag_ids(artifact.kernel_source, "MTE3_MTE2")

    assert ids == {0, 1, 2}


def test_nested_single_iteration_loop_does_not_promote_noneligible_buffer():
    artifact = lower(_make_noneligible_single_iteration_nested_loop_program(), target="ascend")
    source = artifact.kernel_source
    dynamic_wait = "asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>((w % 3)))"
    dynamic_set = "asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>((w % 3)))"

    assert source.count(dynamic_wait) == 1
    assert source.count(dynamic_set) == 1


def test_auto_schedule_spills_unavoidable_flag_overflow_to_buf_mutexes(capfd):
    source = lower(_make_unavoidable_overflow_program(), target="ascend").kernel_source
    stderr = capfd.readouterr().err

    assert "asc_lock(" in source
    assert "asc_unlock(" in source
    assert re.search(
        r"allocated 10 flag ids for [A-Z0-9_]+.*exceeding the dav-3510 limit of 8"
        r".*RewriteFlagToBuf.*get_buf/rls_buf",
        stderr,
    )


def test_auto_schedule_warns_on_unapplied_buffer_version_annotation(capfd):
    lower(_make_unapplied_buffer_version_annotation_program(), target="ascend")
    stderr = capfd.readouterr().err

    assert re.search(r'did not enable multi-buffering for buffer "ub".*requested 2 versions', stderr)


def test_auto_schedule_distinguishes_same_name_buffer_version_annotations(capfd):
    lower(_make_same_name_buffer_version_annotation_program(), target="ascend")
    stderr = capfd.readouterr().err

    assert re.search(r'did not enable multi-buffering for buffer "ub".*requested 2 versions', stderr)


if __name__ == "__main__":
    tilelang.testing.main()
