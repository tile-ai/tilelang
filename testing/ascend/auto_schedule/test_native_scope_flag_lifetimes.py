"""Retained branches keep exact flag lifetimes and hardware bank capacity."""

import pytest
import tilelang.ascend.language as T
from tilelang.ascend import transform
from testing.ascend._ir import attribute_fields, attribute_key, calls, nodes, schedule
from tvm_tirx import ir, sym, tirx


def _guarded_ring(versions, period, nested):
    if nested:

        @T.prim_func
        def main(C: T.Tensor((181, 16, 16), "float32")):
            with T.MixedKernel(1) as (_, sid):
                accum = T.alloc_l0c((16, 16), "float32")
                storage = T.alloc_shared((16, 16), "float32")
                T.annotate_buffer_versions({storage: (versions, "counter")})
                for i in T.Pipelined(181, num_stages=versions):
                    if sid == (i // period) % 2:  # noqa: SIM102 - exercise nested scopes
                        if i % 3 == 0:
                            T.copy(accum, storage)
                            T.copy(storage, C[i, :, :])
    else:

        @T.prim_func
        def main(C: T.Tensor((181, 16, 16), "float32")):
            with T.MixedKernel(1) as (_, sid):
                accum = T.alloc_l0c((16, 16), "float32")
                storage = T.alloc_shared((16, 16), "float32")
                T.annotate_buffer_versions({storage: (versions, "counter")})
                for i in T.Pipelined(181, num_stages=versions):
                    if sid == (i // period) % 2:
                        T.copy(accum, storage)
                        T.copy(storage, C[i, :, :])

    return main


@pytest.mark.parametrize("versions", [2, 15])
@pytest.mark.parametrize("period", [1, 2])
@pytest.mark.parametrize("nested", [False, True])
def test_counter_flag_ring_survives_native_branch_scopes(versions, period, nested):
    before = schedule(
        _guarded_ring(versions, period, nested),
        through="ResolveCore",
        auto_schedule=False,
        infer_layouts=True,
    )
    before_json = ir.save_json(before)
    after = transform.InsertSync()(before)
    assert ir.save_json(before) == before_json
    assert tirx.analysis.check_well_formed(after["main"])
    assert not tirx.analysis.undefined_vars(after["main"])
    analyzer = sym.Analyzer()
    cube_initial_slots = set()
    vector_initial_slots = {1: set(), 4: set()}
    for task in nodes(after, ir.RegionStmt):
        if attribute_key(task) != "tl.ascend_task":
            continue
        core = int(attribute_fields(task)["node"]["core_mask"])
        for action in ("set", "wait"):
            for call in calls(task.body, "tl.ascend_cross_core_" + action + "_flag"):
                identifier = analyzer.simplify(call.args[2])
                bound = analyzer.const_int_bound(identifier)
                assert 0 <= bound.min_value <= bound.max_value < (32 if core == 2 else 16)
                if isinstance(identifier, ir.IntImm):
                    if core == 2 and action == "wait":
                        cube_initial_slots.add(int(identifier))
                    if core in vector_initial_slots and action == "set":
                        vector_initial_slots[core].add(int(identifier))
    assert any(slot < 16 for slot in cube_initial_slots)
    assert any(slot >= 16 for slot in cube_initial_slots)
    assert all(len(slots) == versions for slots in vector_initial_slots.values())


@pytest.mark.parametrize("versions", [9, 15])
def test_overlapping_counter_flag_lifetimes_cannot_share_a_bank(versions):
    @T.prim_func
    def main(C: T.Tensor((62, 16, 16), "float32")):
        with T.MixedKernel(1) as (_, sid):
            accum = T.alloc_l0c((16, 16), "float32")
            first = T.alloc_shared((16, 16), "float32")
            second = T.alloc_shared((16, 16), "float32")
            T.annotate_buffer_versions({first: (versions, "counter"), second: (versions, "counter")})
            for i in T.Pipelined(31, num_stages=versions):
                if sid == i % 2:
                    T.copy(accum, first)
                    T.copy(accum, second)
                    T.copy(first, C[2 * i, :, :])
                    T.copy(second, C[2 * i + 1, :, :])

    before = schedule(main, through="ResolveCore", auto_schedule=False, infer_layouts=True)
    original = ir.save_json(before)
    with pytest.raises((ValueError, RuntimeError), match="cannot allocate .* contiguous cross-core flag slots"):
        transform.InsertSync()(before)
    assert ir.save_json(before) == original
