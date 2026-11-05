"""Challenge finite DMA lifetime proofs using actual consumer-generated IR."""

from copy import deepcopy

import pytest
import tvm_ffi
from tilelang.ascend import transform
from tvm_tirx import ir, tirx
from testing.ascend.auto_schedule import test_buffer_lifetimes as original
from testing.ascend.auto_schedule._dma_lifetime import certificate
from testing.ascend.auto_schedule._dma_lifetime_events import extract
from testing.ascend.auto_schedule._dma_lifetime_order import prove


CASES = [
    ("test_ring_generation_lifetimes", (2, True, True)),
    ("test_loop_carried_lifetime_preserves_local_reader", (True,)),
    ("test_nested_carried_lifetime_preserves_local_dma_reader", ()),
]


@pytest.fixture(scope="module", params=CASES, ids=["shifted-ring", "carried", "nested"])
def actual_sync(request):
    name, arguments = request.param
    captured = []
    actual_insert = transform.InsertSync

    def observe_insert(*args, **kwargs):
        implementation = actual_insert(*args, **kwargs)

        def apply(module):
            before = ir.save_json(module)
            assert "tirx.noalias" not in module["main"].attrs
            result = implementation(module)
            assert ir.save_json(module) == before
            captured.append((module, result))
            return result

        return apply

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(transform, "InsertSync", observe_insert)
        getattr(original, name)(*arguments)
    assert len(captured) == 1
    before, after = captured[0]
    return name, before, after


def test_actual_dma_protocol_preserves_all_readers_and_separates_allocations(actual_sync):
    name, before, after = actual_sync
    report = certificate(before, after)
    count = 12 if name == "test_ring_generation_lifetimes" else 8
    assert report["reader_counts"] == {"b:S": count, "a:MTE3": count, "a:S": count - (2 if count == 12 else 1)}
    assert len(report["separated_pairs"]) == count * count


@pytest.mark.parametrize("fault", ["lost_seed", "lost_drain", "wrong_token", "late_acquire", "late_dma_reader", "late_scalar_reader"])
def test_incomplete_or_extended_protocol_does_not_prove_allocation_reuse(actual_sync, fault):
    events = extract(actual_sync[2])
    if fault == "lost_seed":
        events.pop(0)
    elif fault == "lost_drain":
        events.pop()
    elif fault == "wrong_token":
        acquire = next(event for event in events if event["kind"] == "wait" and event["pipe"] == "S")
        acquire["token"] = 7
    elif fault == "late_acquire":
        indices = [i for i, event in enumerate(events) if event["kind"] == "wait" and event["pipe"] == "S" and event["loops"]]
        acquire = events.pop(indices[1])
        writer = next(
            i
            for i, event in enumerate(events)
            if event["loops"] == acquire["loops"] and any(access["storage"] == "b" for access in event.get("writes", []))
        )
        events.insert(writer + 1, acquire)
    else:
        reader = deepcopy(next(event for event in events if event["kind"] == "copy"))
        if fault == "late_dma_reader":
            release = next(i for i, event in enumerate(events) if event["kind"] == "set" and event["pipe"] == "MTE3" and event["loops"])
            events.insert(release + 1, reader)
        else:
            writers = [i for i, event in enumerate(events) if any(access["storage"] == "b" for access in event.get("writes", []))]
            reader.update(kind="scalar", pipe="S", writes=[], loops=events[writers[1]]["loops"])
            events.insert(writers[1] + 1, reader)
    report = prove(events)
    assert not report["safe"]


def test_acquire_on_only_one_aiv_core_is_not_a_common_completion_proof(actual_sync):
    events = extract(actual_sync[2])
    next(event for event in events if event["kind"] == "wait" and event["loops"])["core"] = 1
    with pytest.raises(AssertionError, match="common AIV mask"):
        prove(events)


def test_runtime_guarded_acquire_is_not_silently_treated_as_unconditional(actual_sync):
    module = actual_sync[2]
    flag = ir.TensorLoad(module["main"].params[0], [0]) > 0
    changed = False

    def guard(statement):
        nonlocal changed
        if not changed and isinstance(statement.value, ir.Call) and statement.value.op.name == "tl.ascend_wait_flag":
            changed = True
            return ir.If(flag, [statement], None)
        return statement

    guarded = tvm_ffi.structural_map(module, [(ir.Evaluate, guard)])
    assert changed and tirx.analysis.check_well_formed(guarded)
    with pytest.raises(AssertionError, match="unmodeled control or statement"):
        extract(guarded)
