"""PrepareMultiBuffer chooses storage clocks and scopes their updates."""

import pytest
import tvm_tirx
from tilelang.ascend import transform
from tvm_tirx import ir
from testing.ascend._ir import allocated_buffer, loop_annotations, nodes, serial, statements, tensor
from testing.ascend.auto_schedule._scheduled_ir import copy_ring


@pytest.mark.parametrize(
    "mode, guarded, owners, uses_counter",
    [
        pytest.param("iteration", False, 1, False, id="affine-iteration"),
        pytest.param("iteration", True, 1, False, id="guarded-iteration"),
        pytest.param("counter", False, 1, True, id="explicit-counter"),
        pytest.param("counter", True, 1, True, id="guarded-counter"),
        pytest.param("auto", False, 1, False, id="auto-affine"),
        pytest.param("auto", True, 1, True, id="auto-sparse"),
        pytest.param("auto", False, 2, True, id="auto-siblings"),
        pytest.param("counter", True, 2, True, id="sparse-siblings"),
    ],
)
@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_storage_clock(mode, guarded, owners, uses_counter, versions):
    before = copy_ring(mode=mode, versions=versions, owners=owners, guarded=guarded)
    after = transform.PrepareMultiBuffer()(before)
    ub = allocated_buffer(after, "ub")
    assert tuple(int(dim) for dim in ub.ty.shape) == (64,)  # Preparation does not expand data.
    loops = [loop for loop in nodes(after, ir.For) if ub in loop_annotations(loop).get("multi_buffer_eligible", [])]
    assert len(loops) == owners
    counters = [loop_annotations(loop).get("tl.multi_buffer_counter_map", {}).get(ub) for loop in loops]
    assert all((counter is not None) == uses_counter for counter in counters)
    if not uses_counter:
        return
    counter = counters[0]
    assert all(other.same_as(counter) for other in counters)
    assert str(counter.ty.dtype) == "int32" and str(counter.ty.scope) == "local.var"
    updates = [
        (store, parents) for store, parents in statements(after) if isinstance(store, ir.TensorStore) and store.dest.same_as(counter)
    ]
    resets = [(store, parents) for store, parents in updates if isinstance(store.value, ir.IntImm)]
    advances = [(store, parents) for store, parents in updates if not isinstance(store.value, ir.IntImm)]
    assert len(resets) == 1 and int(resets[0][0].value) == 0
    assert not any(isinstance(parent, ir.For) for parent in resets[0][1])
    assert len(advances) == owners
    analyzer = tvm_tirx.sym.Analyzer()
    for store, parents in advances:
        assert analyzer.can_prove_equal(store.value, counter[0] + 1)
        owner = next(parent for parent in reversed(parents) if isinstance(parent, ir.For))
        if guarded:
            expected_guard = owner.var % 2 == 0
            actual_guard = loop_annotations(owner)["tl.storage_epoch_guard_map"][ub]
            assert analyzer.can_prove_equal(actual_guard, expected_guard)
            assert any(isinstance(parent, ir.If) and analyzer.can_prove_equal(parent.condition, expected_guard) for parent in parents)


def test_iteration_clock_requires_single_owner():
    with pytest.raises(RuntimeError, match="requires exactly one owner"):
        transform.PrepareMultiBuffer()(copy_ring(mode="iteration", owners=2))


@pytest.mark.parametrize(
    "owners,last_members",
    [(1, "both"), (2, "both"), (2, "data"), (2, "sf"), (1, "data")],
    ids=["single-owner", "sibling-owners", "data-only-sibling", "sf-only-sibling", "unused-sf"],
)
def test_bound_sf_and_data_share_counter_and_union_epoch_guard(owners, last_members):
    from testing.ascend._ir import kernel, seq
    from testing.ascend.auto_schedule._scheduled_ir import unit

    data = tensor((1,), "int32", name="data", scope="shared.l0a")
    sf = tensor((1,), "int32", name="sf", scope="shared.l0a.sf")
    out = tensor((4,), "int32", name="out")
    loops = []
    for owner in range(owners):
        i = ir.Var(f"i{owner}", "int32")
        members = last_members if owner == owners - 1 else "both"
        tasks = []
        for buffer, guard in [(data, None), (sf, i % 2 == 0)]:
            if members != "both" and buffer.name != members:
                continue
            tasks += [
                unit(ir.TensorStore(buffer, [0], i), core=2, guard=guard),
                unit(ir.TensorStore(out, [i], buffer[0]), core=2, guard=guard),
            ]
        loops.append(unit(serial(i, 0, 4, seq(*tasks), annotations={"multi_buffer_eligible": [data, sf]}), core=None))
    before = kernel(
        seq(*loops),
        buffers=[data, sf],
        params=[out],
        annotations={
            "tl.l0_sf_bindings": {sf: data},
            "tl.buffer_versions_map": {data: 2, sf: 2},
            "tl.buffer_version_mode": {data: "counter", sf: "counter"},
        },
        allocation_stage=0,
    )
    after = transform.PrepareMultiBuffer()(before)
    counters = []
    loops = nodes(after, ir.For)
    assert len(loops) == owners
    for index, loop in enumerate(loops):
        mapping = loop_annotations(loop)["tl.multi_buffer_counter_map"]
        assert mapping[data].same_as(mapping[sf])
        counters.append(mapping[data])
        guards = loop_annotations(loop)["tl.storage_epoch_guard_map"]
        expected = loop.var % 2 == 0 if index == owners - 1 and last_members == "sf" else ir.const(True, "bool")
        assert tvm_tirx.sym.Analyzer().can_prove_equal(guards[data], expected)
        assert tvm_tirx.sym.Analyzer().can_prove_equal(guards[data], guards[sf])
    assert all(counter.same_as(counters[0]) for counter in counters)
    updates = [store for store in nodes(after, ir.TensorStore) if store.dest.same_as(counters[0])]
    assert len(updates) == owners + 1  # One reset, one increment per owner.
