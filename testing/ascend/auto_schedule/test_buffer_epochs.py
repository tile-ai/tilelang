"""PrepareMultiBuffer chooses storage clocks and scopes their updates."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import allocated_buffer, nodes, statements
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
    assert tuple(int(dim) for dim in ub.shape) == (64,)  # Preparation does not expand data.
    loops = [loop for loop in nodes(after, tirx.For) if ub.data in loop.annotations.get("multi_buffer_eligible", [])]
    assert len(loops) == owners
    counters = [loop.annotations.get("tl.multi_buffer_counter_map", {}).get(ub.data) for loop in loops]
    assert all((counter is not None) == uses_counter for counter in counters)
    if not uses_counter:
        return
    counter = counters[0]
    assert all(other.same_as(counter) for other in counters)
    assert counter.dtype == "int32" and counter.scope() == "local.var"
    updates = [
        (store, parents) for store, parents in statements(after) if isinstance(store, tirx.BufferStore) and store.buffer.same_as(counter)
    ]
    resets = [(store, parents) for store, parents in updates if isinstance(store.value, tirx.IntImm)]
    advances = [(store, parents) for store, parents in updates if not isinstance(store.value, tirx.IntImm)]
    assert len(resets) == 1 and int(resets[0][0].value) == 0
    assert not any(isinstance(parent, tirx.For) for parent in resets[0][1])
    assert len(advances) == owners
    analyzer = tvm.arith.Analyzer()
    for store, parents in advances:
        assert analyzer.can_prove_equal(store.value, counter[0] + 1)
        owner = next(parent for parent in reversed(parents) if isinstance(parent, tirx.For))
        if guarded:
            expected_guard = owner.loop_var % 2 == 0
            actual_guard = owner.annotations["tl.storage_epoch_guard_map"][ub.data]
            assert analyzer.can_prove_equal(actual_guard, expected_guard)
            assert any(
                isinstance(parent, tirx.IfThenElse) and analyzer.can_prove_equal(parent.condition, expected_guard) for parent in parents
            )


def test_iteration_clock_requires_single_owner():
    with pytest.raises(tvm.error.InternalError, match="requires exactly one owner"):
        transform.PrepareMultiBuffer()(copy_ring(mode="iteration", owners=2))


@pytest.mark.parametrize(
    "owners,last_members",
    [(1, "both"), (2, "both"), (2, "data"), (2, "sf"), (1, "data")],
    ids=["single-owner", "sibling-owners", "data-only-sibling", "sf-only-sibling", "unused-sf"],
)
def test_bound_sf_and_data_share_counter_and_union_epoch_guard(owners, last_members):
    from testing.ascend._ir import kernel, seq
    from testing.ascend.auto_schedule._scheduled_ir import unit

    data = tirx.decl_buffer((1,), "int32", name="data", scope="shared.l0a")
    sf = tirx.decl_buffer((1,), "int32", name="sf", scope="shared.l0a.sf")
    out = tirx.decl_buffer((4,), "int32", name="out")
    loops = []
    for owner in range(owners):
        i = tirx.Var(f"i{owner}", "int32")
        members = last_members if owner == owners - 1 else "both"
        tasks = []
        for buffer, guard in [(data, None), (sf, i % 2 == 0)]:
            if members != "both" and buffer.name != members:
                continue
            tasks += [
                unit(tirx.BufferStore(buffer, i, [0]), core=2, guard=guard),
                unit(tirx.BufferStore(out, buffer[0], [i]), core=2, guard=guard),
            ]
        loops.append(
            unit(
                tirx.For(i, 0, 4, tirx.ForKind.SERIAL, seq(*tasks), annotations={"multi_buffer_eligible": [data.data, sf.data]}), core=None
            )
        )
    before = kernel(
        seq(*loops),
        buffers=[data, sf],
        params=[out],
        annotations={
            "tl.l0_sf_bindings": {sf.data: data.data},
            "tl.buffer_versions_map": {data.data: 2, sf.data: 2},
            "tl.buffer_version_mode": {data.data: "counter", sf.data: "counter"},
        },
    )
    after = transform.PrepareMultiBuffer()(before)
    counters = []
    loops = nodes(after, tirx.For)
    assert len(loops) == owners
    for index, loop in enumerate(loops):
        mapping = loop.annotations["tl.multi_buffer_counter_map"]
        assert mapping[data.data].same_as(mapping[sf.data])
        counters.append(mapping[data.data])
        guards = loop.annotations["tl.storage_epoch_guard_map"]
        expected = loop.loop_var % 2 == 0 if index == owners - 1 and last_members == "sf" else tirx.const(True, "bool")
        assert tvm.arith.Analyzer().can_prove_equal(guards[data.data], expected)
        assert tvm.arith.Analyzer().can_prove_equal(guards[data.data], guards[sf.data])
    assert all(counter.same_as(counters[0]) for counter in counters)
    updates = [store for store in nodes(after, tirx.BufferStore) if store.buffer.same_as(counters[0])]
    assert len(updates) == owners + 1  # One reset, one increment per owner.
