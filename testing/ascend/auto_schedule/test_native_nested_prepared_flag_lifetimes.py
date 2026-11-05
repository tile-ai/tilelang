"""Prepared nested scopes preserve counter epochs and physical flag lifetimes."""

import pytest
from tilelang.ascend import transform
from testing.ascend._ir import (
    attribute,
    attribute_fields,
    attribute_key,
    calls,
    copy,
    kernel,
    nodes,
    tensor,
)
from testing.ascend.auto_schedule._scheduled_ir import unit
from tvm_tirx import ir, sym, tirx


def _prepared_ring(versions, period, depth, precise=True):
    output = tensor((181, 16, 16), name="output")
    accum = tensor((16, 16), name="accum", scope="shared.l0c")
    storage = [tensor((16, 16), name=f"storage_{sid}", scope="shared.dyn") for sid in range(2)]
    counter = [tensor((1,), "int32", name=f"epoch_{sid}", scope="local.var") for sid in range(2)]
    index = ir.Var("i", "int32")
    body = []
    guards = {}
    for sid, core in ((0, 1), (1, 4)):
        terms = [(index // period) % 2 == sid, index % 3 == 0]
        if depth == 3:
            terms.append(index % 5 == 0)
        active = ir.prim.all(*terms)
        guards[storage[sid]] = active if precise else ir.const(True, "bool")
        work = [
            unit(copy(accum, storage[sid], annotations={"sub_blockid": sid}), core=2),
            unit(copy(storage[sid], output, dst_indices=[index, 0, 0], dst_shape=[1, 16, 16]), core=core),
        ]
        for term in reversed(terms):
            work = [ir.If(term, work, None)]
        body.extend(work)
        body.append(
            unit(ir.TensorStore(counter[sid], [0], ir.TensorLoad(counter[sid], [0]) + 1), core=core | 2, guard=guards[storage[sid]])
        )
    metadata = {
        "multi_buffer_eligible": storage,
        "tl.multi_buffer_counter_map": dict(zip(storage, counter)),
        "tl.storage_epoch_guard_map": guards,
    }
    body = attribute(None, "tl.loop.annotations", metadata, body)
    loop = ir.For(index, ir.const(0), ir.const(181), ir.const(1), ir.ForKind.DEFAULT, [body])
    return kernel(
        [*[unit(ir.TensorStore(count, [0], ir.const(0)), core=7) for count in counter], unit(loop, core=None)],
        buffers=[accum, *storage, *counter],
        params=[output],
        allocation_stage=0,
        annotations={
            "vector_count": 2,
            "tl.buffer_versions_map": dict.fromkeys(storage, versions),
            "tl.aiv_storage_map": dict(zip(storage, range(2))),
        },
    )


@pytest.mark.parametrize("versions", [2, 15])
@pytest.mark.parametrize("period", [1, 2])
@pytest.mark.parametrize("depth", [2, 3])
def test_prepared_nested_counter_epochs_share_only_alternating_flags(versions, period, depth):
    before = _prepared_ring(versions, period, depth)
    assert tirx.analysis.check_well_formed(before["main"])
    assert not tirx.analysis.undefined_vars(before["main"])
    snapshot = ir.save_json(before)
    after = transform.InsertSync()(before)
    assert ir.save_json(before) == snapshot
    assert tirx.analysis.check_well_formed(after["main"])
    assert not tirx.analysis.undefined_vars(after["main"])
    analyzer = sym.Analyzer()
    initial = {1: set(), 4: set()}
    for task in nodes(after, ir.RegionStmt):
        if attribute_key(task) != "tl.ascend_task":
            continue
        core = int(attribute_fields(task)["node"]["core_mask"])
        for call in calls(task.body, "tl.ascend_cross_core_set_flag") + calls(task.body, "tl.ascend_cross_core_wait_flag"):
            flag = analyzer.simplify(call.args[2])
            bound = analyzer.const_int_bound(flag)
            assert 0 <= bound.min_value <= bound.max_value < (32 if core == 2 else 16)
        if core in initial:
            for call in calls(task.body, "tl.ascend_cross_core_set_flag"):
                flag = analyzer.simplify(call.args[2])
                if isinstance(flag, ir.IntImm):
                    initial[core].add(int(flag))
    assert all(len(slots) == versions for slots in initial.values())


@pytest.mark.parametrize("depth", [2, 3])
def test_unconditional_epoch_does_not_prove_a_conditional_handshake(depth):
    before = _prepared_ring(15, 1, depth, precise=False)
    snapshot = ir.save_json(before)
    with pytest.raises((ValueError, RuntimeError), match="cannot allocate .* contiguous cross-core flag slots"):
        transform.InsertSync()(before)
    assert ir.save_json(before) == snapshot
