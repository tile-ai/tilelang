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
