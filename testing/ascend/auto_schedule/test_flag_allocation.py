"""Flag allocation and spill contracts on a fixed scheduled input.

Buffer counts and source order belong to these fixtures. The allocator may
choose any legal numbering or reuse strategy.
"""

import pytest
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import transform as ascend_transform
from tvm import tirx
from testing.ascend._ir import calls, schedule


def _make_cross_iter_reuse_program():
    tile = 64
    stages = 2
    inner = 4

    @T.prim_func
    def main(A: T.Tensor((4096,), "float32"), C: T.Tensor((4096,), "float32")):
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
        # Preserve the same row order on MTE2 and MTE3 so both counter
        # channels survive sync-edge pruning, independently of task costs.
        with T.Stage(0):
            T.copy(A[offset : offset + tile], ub[row, :])
            with T.SimdVF():
                mask = T.simd.pset(32)
                value = T.simd.vld(ub[row, 0])
                T.simd.vsts(ub[row, 0], value, mask)
            T.copy(ub[row, :], C[offset : offset + tile])

    @T.prim_func
    def main(
        A: T.Tensor((4 * iterations * tile,), "float32"),
        C: T.Tensor((4 * iterations * tile,), "float32"),
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


def _make_inner_loop_program(extent: int):
    tile = 64

    @T.prim_func
    def main(A: T.Tensor((4 * tile,), "float32"), C: T.Tensor((tile,), "float32")):
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
    def main(A: T.Tensor((8 * tile,), "float32"), C: T.Tensor((8 * tile,), "float32")):
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
    def main(A: T.Tensor((16 * tile,), "float32"), C: T.Tensor((8 * tile,), "float32")):
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


def _make_bound_flag_spill_program():
    @T.prim_func
    def main():
        with T.Kernel(1):
            for outer in T.serial(2):
                for inner in T.serial(2):
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


def _make_cross_core_reuse_program(
    versions: int,
    add_vector_stage: bool = False,
    reserve_last_slot: bool = False,
    counter: bool = False,
):
    @T.prim_func
    def main(C: T.Tensor((8, 32, 16), "float32")):
        with T.MixedKernel(1):
            accum = T.alloc_l0c((32, 16), "float32")
            output = T.alloc_shared((16, 16), "float32")
            T.annotate_buffer_versions({output: (versions, "counter") if counter else versions})
            for tile in T.serial(8):
                # Pass input: Fixpipe produces UB. GEMM and layout inference
                # are not prerequisites for this synchronization contract.
                T.dual_copy(accum, output, unit_flag_ctrl=3)
                if reserve_last_slot:
                    T.ascend_sync_inter_arrive("PIPE_FIX", 15)
                    T.ascend_sync_inter_wait("PIPE_FIX", 15)
                if add_vector_stage:
                    with T.SimdVF():
                        for offset in T.serial(4):
                            value = T.simd.vld(output[offset * 4, 0])
                            T.simd.vsts(output[offset * 4, 0], value)
                T.dual_copy(output, C[tile, :, :])

    return main


def _make_serial_protocols(count):
    @T.prim_func
    def main(A: T.Tensor((4 * count * 64,), "float32"), C: T.Tensor((4 * count * 64,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({ub: (2, "iteration")})
            for i in T.Pipelined(4, num_stages=2):
                for j in T.Unroll(count, explicit=True):
                    offset = (i * count + j) * 64
                    T.copy(A[offset : offset + 64], ub)
                    with T.SimdVF():
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value)
                    T.copy(ub, C[offset : offset + 64])

    return main


def _insert_sync(program):
    before = schedule(program, through="ResolveCore", auto_schedule=False)
    return ascend_transform.InsertSync()(before)


def _events(mod, operation, hard_event):
    return [call.args[1] for call in calls(mod, "tl.ascend_" + operation + "_flag") if str(call.args[0].value) == hard_event]


def _constant_ids(mod, operation, hard_event):
    return {int(expr) for expr in _events(mod, operation, hard_event) if isinstance(expr, tirx.IntImm)}


@pytest.mark.parametrize("count", [5, 9], ids=["fits", "requires-reuse"])
def test_serial_protocols_fit_hardware_flag_budget(count):
    mod = _insert_sync(_make_serial_protocols(count))
    for event in ("MTE2_V", "V_MTE3", "MTE3_MTE2"):
        sets = _events(mod, "set", event)
        waits = _events(mod, "wait", event)
        assert sets and waits
        assert sorted(str(expr) for expr in sets) == sorted(str(expr) for expr in waits)
        analyzer = tvm.arith.Analyzer()
        assert all(analyzer.can_prove(expr >= 0) and analyzer.can_prove(expr < 8) for expr in [*sets, *waits])


def test_sibling_counter_owners_share_one_release_ring():
    mod = _insert_sync(_make_cross_iter_reuse_program())
    for operation in ("set", "wait"):
        dynamic = [expr for expr in _events(mod, operation, "MTE3_MTE2") if not isinstance(expr, tirx.IntImm)]
        assert len(dynamic) == 4
        for expr in dynamic[1:]:
            tvm.ir.assert_structural_equal(expr, dynamic[0])
        assert len(_constant_ids(mod, operation, "MTE3_MTE2")) == 2


def test_distinct_counter_channels_keep_exclusive_rings_and_spill():
    mod = _insert_sync(_make_multi_owner_counter_channel_program())
    # Two disjoint rows have two exclusive eight-version release rings. These
    # span 16 slots regardless of allocator numbering; one ring must spill.
    for operation in ("set", "wait"):
        assert len(_constant_ids(mod, operation, "MTE3_MTE2")) == 16
    spilled = ascend_transform.RewriteFlagToBuf()(mod)
    assert calls(spilled, "tl.ascend_get_buf")
    assert calls(spilled, "tl.ascend_rls_buf")


def test_flag_spill_preserves_bound_dynamic_ids():
    mod = tvm.IRModule({"main": _make_bound_flag_spill_program()})
    rewritten = ascend_transform.RewriteFlagToBuf()(mod)
    assert calls(rewritten, "tl.ascend_get_buf")
    assert calls(rewritten, "tl.ascend_rls_buf")
    for operation in ("set", "wait"):
        assert _constant_ids(rewritten, operation, "MTE2_V") <= set(range(8))


@pytest.mark.parametrize(
    "versions, counter, vector_stage, reserve_slot",
    [(4, False, False, False), (8, False, False, False), (15, False, False, False), (15, True, False, False), (8, False, True, True)],
    ids=["four-slots", "eight-slots", "iteration-reuse", "counter-reuse", "reserved-slot-and-vector"],
)
def test_cross_core_rings_fit_available_slots(versions, counter, vector_stage, reserve_slot):
    program = _make_cross_core_reuse_program(versions, vector_stage, reserve_slot, counter)
    mod = _insert_sync(program)
    for operation in ("set", "wait"):
        generated = [call.args[2] for call in calls(mod, "tl.ascend_cross_core_" + operation + "_flag") if int(call.args[0]) == 4]
        assert generated
        constants = {int(expr) % 16 for expr in generated if isinstance(expr, tirx.IntImm)}
        assert len(constants) == versions
        assert constants <= set(range(15 if reserve_slot else 16))


@pytest.mark.parametrize("extent", [1, 4])
def test_single_iteration_needs_no_reverse_flag(extent):
    mod = _insert_sync(_make_inner_loop_program(extent))
    assert bool(_events(mod, "set", "V_MTE2")) == (extent > 1)
    assert bool(_events(mod, "wait", "V_MTE2")) == (extent > 1)


@pytest.mark.parametrize("eligible", [True, False])
def test_extent_one_child_uses_only_its_owner_versions(eligible):
    program = (_make_single_iteration_nested_loop_program if eligible else _make_noneligible_single_iteration_nested_loop_program)()
    mod = _insert_sync(program)
    # The outer owner has three versions. The extent-one child must neither
    # lose that ring nor create another outer ring for an unclaimed buffer.
    for operation in ("set", "wait"):
        assert len(_constant_ids(mod, operation, "MTE3_MTE2")) == 3


if __name__ == "__main__":
    tilelang.testing.main()
