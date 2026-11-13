"""Guarded producer/consumer chains: activation-qualified covering."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import calls, copy, kernel, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


def _events(mod, operation, pipe):
    return [call for call in calls(mod, "tl.ascend_" + operation + "_flag") if call.args[0].value == pipe]


def _guarded_events(mod, operation, pipe):
    """(call, [IfThenElse conditions]) for each flag event of `pipe`."""
    result = []
    for stmt, parents in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if call.op.name != "tl.ascend_" + operation + "_flag" or call.args[0].value != pipe:
            continue
        guards = [parent.condition for parent in parents if isinstance(parent, tirx.IfThenElse)]
        result.append((call, guards))
    return result


def _decode_tail(*, guarded):
    """P -> Q -> R round trip; P and Q are guarded, R runs every round."""
    tile = 64
    rows = 4
    k = tirx.decl_buffer((rows * tile,), "float16", name="K")
    ks = tirx.decl_buffer((rows * tile,), "float32", name="KS")
    seq_len = tirx.decl_buffer((1,), "int32", name="seq_len")
    o = tirx.decl_buffer((rows * tile,), "float16", name="O")
    os = tirx.decl_buffer((rows * tile,), "float32", name="OS")
    ub = tirx.decl_buffer((tile,), "float16", name="ub", scope="shared.dyn")
    gscale = tirx.decl_buffer((tile,), "float32", name="gscale", scope="shared.dyn")
    gacc = tirx.decl_buffer((tile,), "float32", name="gacc", scope="shared.dyn")
    epoch = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var")
    i = tirx.Var("i", "int32")
    cond = i * tile < seq_len[0]
    guard = cond if guarded else None
    load = seq(
        copy(k, ub, src_indices=[i * tile], src_shape=[tile]),
        copy(ks, gscale, src_indices=[i * tile], src_shape=[tile]),
    )
    compute = tirx.SBlock([], [], [], "SIMD_VF", tirx.BufferStore(gacc, tirx.Cast("float32", ub[0]), [0]))
    store = seq(
        copy(gacc, o, dst_indices=[i * tile], dst_shape=[tile]),
        copy(gscale, os, dst_indices=[i * tile], dst_shape=[tile]),
    )
    body = seq(
        unit(load, guard=guard),
        unit(compute, guard=guard),
        unit(store),
        *([unit(tirx.BufferStore(epoch, epoch[0] + 1, [0]), guard=guard)] if guarded else []),
    )
    annotations = {
        "multi_buffer_eligible": [ub.data],
        "tl.storage_epoch_guard_map": {ub.data: cond if guarded else tirx.const(True, "bool")},
    }
    if guarded:
        annotations["tl.multi_buffer_counter_map"] = {ub.data: epoch}
    owner = unit(tirx.For(i, 0, rows, tirx.ForKind.SERIAL, body, annotations=annotations), core=None)
    before = kernel(
        seq(*([unit(tirx.BufferStore(epoch, 0, [0]))] if guarded else []), owner),
        buffers=[ub, gscale, gacc] + ([epoch] if guarded else []),
        params=[k, ks, seq_len, o, os],
        annotations={"tl.buffer_versions_map": {ub.data: 2}},
    )
    return before, cond


def _guarded_by(analyzer, guards, cond):
    """True if one guard is structurally the hazard predicate."""
    return any(tvm.ir.structural_equal(guard, cond) for guard in guards)


@pytest.mark.parametrize("guarded", [False, True], ids=["unguarded", "guarded"])
def test_decode_tail_culls_cross_epoch_handshakes(guarded):
    before, cond = _decode_tail(guarded=guarded)
    after = transform.InsertSync()(before)
    analyzer = tvm.arith.Analyzer()

    # Both cross-epoch handshakes are covered by the remaining chain.
    assert not _events(after, "set", "MTE3_V")
    assert not _events(after, "wait", "MTE3_V")
    assert not _events(after, "set", "MTE2_MTE3")
    assert not _events(after, "wait", "MTE2_MTE3")

    if not guarded:
        # Unguarded variant: every handshake unconditional and pairwise matched.
        for operation in ("set", "wait"):
            for pipe in ("MTE2_V", "V_MTE3", "MTE3_MTE2"):
                for _, guards in _guarded_events(after, operation, pipe):
                    assert not guards
        for pipe in ("MTE2_V", "V_MTE3", "MTE3_MTE2"):
            released = _events(after, "set", pipe)
            acquired = _events(after, "wait", pipe)
            assert released and acquired
            assert len(released) == len(acquired)
        return

    # Same-round pair: one set/wait, both carrying the hazard guard.
    released = _guarded_events(after, "set", "V_MTE3")
    acquired = _guarded_events(after, "wait", "V_MTE3")
    assert len(released) == len(acquired) == 1
    for _, guards in released + acquired:
        assert guards and _guarded_by(analyzer, guards, cond)

    # The ub ring uses the counter clock as event id.
    released = _events(after, "set", "MTE2_V")
    acquired = _events(after, "wait", "MTE2_V")
    assert len(released) == len(acquired) == 1
    for event in released + acquired:
        assert not isinstance(event.args[1], tirx.IntImm)
        base = analyzer.simplify(event.args[1] - (epoch_of(after) % 2))
        assert isinstance(base, tirx.IntImm)

    # Read-completion and backpressure rings keep set/wait ids paired.
    released = _events(after, "set", "V_MTE2")
    acquired = _events(after, "wait", "V_MTE2")
    assert released and acquired
    constant_ids = lambda events: sorted(  # noqa: E731
        int(event.args[1]) for event in events if isinstance(event.args[1], tirx.IntImm)
    )
    assert constant_ids(released) == constant_ids(acquired)
    assert any(not isinstance(event.args[1], tirx.IntImm) for event in released + acquired)
    for pipe in ("MTE3_MTE2",):
        released = _events(after, "set", pipe)
        acquired = _events(after, "wait", pipe)
        assert released and acquired
        assert all(isinstance(event.args[1], tirx.IntImm) for event in released + acquired)
        assert sorted(int(event.args[1]) for event in released) == sorted(int(event.args[1]) for event in acquired)


def epoch_of(mod):
    """The counter buffer load referenced by the flag events."""
    events = [call for call in calls(mod, "tl.ascend_set_flag") if not isinstance(call.args[1], tirx.IntImm)]
    assert events, "expected a dynamic flag event (counter clock)"
    loads = []
    tirx.stmt_functor.post_order_visit(events[0], lambda node: loads.append(node) if isinstance(node, tirx.BufferLoad) else None)
    return loads[-1]
