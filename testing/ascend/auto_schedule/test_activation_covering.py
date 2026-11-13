"""Activation-qualified covering and guarded sync emission."""

import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import calls, schedule, statements


def _insert_sync(program):
    before = schedule(program, through="ResolveCore", auto_schedule=False)
    return transform.InsertSync()(before)


def _events(mod, operation, pipe):
    return [call for call in calls(mod, "tl.ascend_" + operation + "_flag") if call.args[0].value == pipe]


def _guarded_syncs(mod, pipe=None):
    """Flag/barrier calls carrying an IfThenElse guard."""
    result = []
    for stmt, parents in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if not isinstance(call.op, tvm.ir.Op):
            continue
        if (
            call.op.name in ("tl.ascend_set_flag", "tl.ascend_wait_flag", "tl.ascend_pipe_barrier")
            and (pipe is None or call.args[0].value == pipe)
            and any(isinstance(parent, tirx.IfThenElse) for parent in parents)
        ):
            result.append(call)
    return result


def _event_guards(mod, pipe):
    guards = []
    for stmt, parents in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if not isinstance(call.op, tvm.ir.Op):
            continue
        if call.op.name in ("tl.ascend_set_flag", "tl.ascend_wait_flag") and call.args[0].value == pipe:
            guards.extend(parent.condition for parent in parents if isinstance(parent, tirx.IfThenElse))
    return guards


def _wait_guards(mod, pipe):
    """IfThenElse conditions guarding the pipe's wait flags."""
    guards = []
    for stmt, parents in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if not isinstance(call.op, tvm.ir.Op):
            continue
        if call.op.name == "tl.ascend_wait_flag" and call.args[0].value == pipe:
            guards.extend(parent.condition for parent in parents if isinstance(parent, tirx.IfThenElse))
    return guards


def _is_true(expr):
    return isinstance(expr, tirx.IntImm) and expr.value == 1


def _chain_kernel(guard_cond, counter_ub=False):
    """P -> Q -> R chain: ub is guarded, gub is read every round."""
    tile = 64
    n = 4

    @T.prim_func
    def kernel(
        A: T.Tensor((n * tile,), "float32"),
        C: T.Tensor((n * 2 * tile,), "float32"),
        seq_len: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            gub = T.alloc_shared((2 * tile,), "float32")
            if counter_ub:
                T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.serial(n):
                if guard_cond(i, seq_len):
                    with T.Task():
                        T.copy(A[i * tile : (i + 1) * tile], ub[0:tile])
                        T.copy(A[i * tile : (i + 1) * tile], gub[0:tile])
                    T.copy(ub[0:tile], gub[tile : 2 * tile])
                T.copy(gub[0 : 2 * tile], C[i * 2 * tile : (i + 1) * 2 * tile])

    return kernel


def test_activation_covering_eliminates_shortcut():
    # Both shortcut edges are covered by the remaining chain.
    kernel = _chain_kernel(lambda i, s: i % 2 == 0)
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert _events(after, "set", "MTE2_MTE3") == []
    assert _events(after, "set", "MTE3_V") == []
    assert _events(after, "set", "V_MTE3")
    assert _events(after, "set", "MTE3_MTE2")


def test_complementary_guards_drop_same_round_keep_cross_round():
    tile = 64
    n = 4

    @T.prim_func
    def kernel(A: T.Tensor((n * tile,), "float32"), C: T.Tensor((n * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(n):
                if i % 2 == 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub[0:tile])
                if i % 2 == 1:
                    T.copy(ub[0:tile], C[i * tile : (i + 1) * tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert len(_events(after, "set", "MTE2_MTE3")) == 2  # ring only
    assert len(_events(after, "wait", "MTE2_MTE3")) == 2


def test_same_guard_dependency_is_kept():
    tile = 64
    n = 4

    @T.prim_func
    def kernel(A: T.Tensor((n * tile,), "float32"), C: T.Tensor((n * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(n):
                if i % 2 == 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub[0:tile])
                    T.copy(ub[0:tile], C[i * tile : (i + 1) * tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert len(_events(after, "set", "MTE2_MTE3")) == 1
    assert len(_events(after, "wait", "MTE2_MTE3")) == 1
    guards = _event_guards(after, "MTE2_MTE3")
    assert guards and not all(_is_true(g) for g in guards)


def test_counter_chain_segment_enables_covering():
    # The counter protocol point acts as a chain segment with dynamic ids.
    kernel = _chain_kernel(lambda i, s: i % 2 == 0, counter_ub=True)
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert _events(after, "set", "MTE2_MTE3") == []
    assert _events(after, "set", "MTE3_V") == []
    released = _events(after, "set", "MTE2_V")
    acquired = _events(after, "wait", "MTE2_V")
    assert released and acquired
    assert all(not isinstance(e.args[1], tirx.IntImm) for e in released + acquired)


def test_same_guard_cross_iteration_edge_is_retained():
    # Same-guard cross-round edges stand in for wider hazards and are kept.
    tile = 64
    n = 4

    @T.prim_func
    def kernel(A: T.Tensor((n * tile,), "float32"), C: T.Tensor((n * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(n):
                if i % 2 == 0:
                    T.copy(ub[0:tile], C[i * tile : (i + 1) * tile])
                    T.copy(A[i * tile : (i + 1) * tile], ub[0:tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    # The cross-round RAW ring is retained.
    assert len(_events(after, "set", "MTE2_MTE3")) >= 2
    assert len(_events(after, "wait", "MTE2_MTE3")) >= 2
    # The same-round WAR pair is kept as well.
    assert len(_events(after, "set", "MTE3_MTE2")) == 1


def test_same_round_pair_carries_hazard_guard():
    # Mixed pair: the wait beside the unguarded consumer carries the hazard
    # guard.
    kernel = _chain_kernel(lambda i, s: i % 2 == 0)
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    guards = _wait_guards(after, "V_MTE3")
    assert guards and not all(_is_true(g) for g in guards)


def test_runtime_scalar_guard_resolves():
    kernel = _chain_kernel(lambda i, s: i * 64 < s[0])
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert _events(after, "set", "MTE2_MTE3") == []
    guards = _wait_guards(after, "V_MTE3")
    assert guards and not all(_is_true(g) for g in guards)


def test_scope_written_guard_variable_stays_conservative():
    # A guard reading a scope-written variable is rejected; the pair stays
    # unguarded.
    tile = 64
    n = 4

    @T.prim_func
    def kernel(A: T.Tensor((n * tile,), "float32"), C: T.Tensor((n * tile,), "float32")):
        with T.Kernel(1):
            gub = T.alloc_shared((tile,), "float32")
            cnt = T.alloc_shared((1,), "int32")
            for i in T.serial(n):
                cnt[0] = cnt[0] + 1
                if cnt[0] > 0:
                    T.copy(A[i * tile : (i + 1) * tile], gub[0:tile])
                T.copy(gub[0:tile], C[i * tile : (i + 1) * tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert _events(after, "set", "MTE2_MTE3")
    assert not _wait_guards(after, "MTE2_MTE3")


def test_unguarded_kernel_has_no_activation_traces():
    kernel = _chain_kernel(lambda i, s: True)
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert _guarded_syncs(after) == []


def test_elimination_preserves_kept_sync_counts():
    kernel = _chain_kernel(lambda i, s: i % 2 == 0)
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert len(_events(after, "set", "V_MTE3")) == 1
    assert len(_events(after, "wait", "V_MTE3")) == 1
    released = _events(after, "set", "MTE3_MTE2")
    acquired = _events(after, "wait", "MTE3_MTE2")
    assert len(released) >= 2
    assert len(acquired) == len(released)


def test_chain_segment_point_fires_unconditionally():
    # A same-round pair carrying a contained candidate's ordering is not
    # statement-guarded; it fires unconditionally.
    tile = 64
    n = 4

    @T.prim_func
    def kernel(A: T.Tensor((n * tile,), "float32"), C: T.Tensor((n * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(n):
                T.copy(A[i * tile : (i + 1) * tile], ub[0:tile])
                if i % 2 == 0:
                    T.copy(ub[0:tile], C[i * tile : (i + 1) * tile])
                if i % 2 == 1:
                    T.copy(ub[0:tile], C[i * tile : (i + 1) * tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    released = _events(after, "set", "MTE2_MTE3")
    acquired = _events(after, "wait", "MTE2_MTE3")
    assert released and acquired
    assert len(released) == len(acquired)
    for operation in ("set", "wait"):
        for _, guards in _guarded_events_of(after, operation, "MTE2_MTE3"):
            assert not guards


def _guarded_events_of(mod, operation, pipe):
    result = []
    for stmt, parents in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if not isinstance(call.op, tvm.ir.Op):
            continue
        if call.op.name != "tl.ascend_" + operation + "_flag" or call.args[0].value != pipe:
            continue
        guards = [p.condition for p in parents if isinstance(p, tirx.IfThenElse)]
        result.append((call, guards))
    return result


def test_nested_guards_widen_instead_of_drop():
    # Nested guards: the surviving point unions both guards instead of losing
    # its statement guard.
    tile = 64
    n = 4

    @T.prim_func
    def kernel(A: T.Tensor((n * tile,), "float32"), C: T.Tensor((n * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(n):
                T.copy(A[i * tile : (i + 1) * tile], ub[0:tile])
                if i < 2:
                    T.copy(ub[0:tile], C[i * tile : (i + 1) * tile])
                if i < 3:
                    T.copy(ub[0:tile], C[i * tile : (i + 1) * tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    released = _events(after, "set", "MTE2_MTE3")
    acquired = _events(after, "wait", "MTE2_MTE3")
    assert released and acquired
    assert len(released) == len(acquired)
    for operation in ("set", "wait"):
        for _, guards in _guarded_events_of(after, operation, "MTE2_MTE3"):
            assert any(isinstance(guard, tirx.Or) for guard in guards)
