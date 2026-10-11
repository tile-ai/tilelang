"""PrepareMultiBuffer groups equivalent clocks and respects lexical scope."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


@pytest.mark.parametrize("relation, share_clock", [("same", True), ("equivalent", True), ("different", False), ("mutated-memory", False)])
def test_guard_equivalence_preserves_storage_epochs(relation, share_clock):
    a = tirx.decl_buffer((64,), "float32", name="A")
    c = tirx.decl_buffer((64,), "float32", name="C")
    pred = tirx.decl_buffer((1,), "int32", name="pred")
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    y = tirx.decl_buffer((64,), "float32", name="y", scope="shared.dyn")
    p, q = tirx.Var("p", "int32"), tirx.Var("q", "int32")
    i = tirx.Var("i", "int32")
    first = p > 0
    second = p > 0 if relation == "same" else (p >= 1 if relation == "equivalent" else q > 0)
    prefix, middle = [], []
    if relation == "mutated-memory":
        first, second = tirx.Var("before_write", "bool"), tirx.Var("after_write", "bool")
        prefix = [unit(tirx.Bind(first, pred[0] > 0), core=3)]
        middle = [unit(tirx.BufferStore(pred, 1, [0]), core=3), unit(tirx.Bind(second, pred[0] > 0), core=3)]
    body = seq(
        *prefix,
        unit(copy(a, x), guard=first),
        unit(copy(x, c), guard=first),
        *middle,
        unit(copy(a, y), guard=second),
        unit(copy(y, c), guard=second),
    )
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data, y.data]})
    before = kernel(
        unit(loop, core=None),
        buffers=[x, y],
        params=[a, c, pred, p, q],
        annotations={"tl.buffer_versions_map": {x.data: 2, y.data: 3}, "tl.buffer_version_mode": {x.data: "counter", y.data: "counter"}},
    )
    after = transform.PrepareMultiBuffer()(before)
    (owner,) = nodes(after, tirx.For)
    clocks = owner.annotations["tl.multi_buffer_counter_map"]
    assert clocks[x.data].same_as(clocks[y.data]) == share_clock
    for storage, clock in clocks.items():
        updates = [
            (stmt, parents)
            for stmt, parents in statements(after)
            if isinstance(stmt, tirx.BufferStore) and stmt.buffer.same_as(clock) and not isinstance(stmt.value, tirx.IntImm)
        ]
        assert len(updates) == 1
        expected = owner.annotations["tl.storage_epoch_guard_map"][storage]
        assert any(
            isinstance(parent, tirx.IfThenElse) and tvm.arith.Analyzer().can_prove_equal(parent.condition, expected)
            for parent in updates[0][1]
        )


@pytest.mark.parametrize("case", ["inner-guard", "mutated-bound"])
def test_nested_condition_cannot_escape_into_outer_epoch(case):
    a = tirx.decl_buffer((64,), "float32", name="A")
    c = tirx.decl_buffer((64,), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    extent = tirx.Var("extent", "int32")
    bound = tirx.decl_buffer((1,), "int32", name="bound")
    guard = j % 2 == 0 if case == "inner-guard" else None
    tasks = [unit(copy(a, ub), guard=guard)]
    if case == "mutated-bound":
        tasks.append(unit(tirx.BufferStore(bound, 0, [0])))
    tasks.append(unit(copy(ub, c), guard=guard))
    child = unit(tirx.For(j, 0, extent, tirx.ForKind.SERIAL, seq(*tasks)), core=None)
    body = seq(unit(tirx.Bind(extent, bound[0])), child) if case == "mutated-bound" else child
    owner = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [ub.data]})
    before = kernel(
        unit(owner, core=None),
        buffers=[ub],
        params=[a, c, bound if case == "mutated-bound" else extent],
        annotations={
            "tl.buffer_versions_map": {ub.data: 1 if case == "mutated-bound" else 2},
            "tl.buffer_version_mode": {ub.data: "counter"},
        },
    )
    after = transform.PrepareMultiBuffer()(before)
    (owner,) = [loop for loop in nodes(after, tirx.For) if loop.loop_var.same_as(i)]
    assert tvm.arith.Analyzer().can_prove(owner.annotations["tl.storage_epoch_guard_map"][ub.data])
    clock = owner.annotations["tl.multi_buffer_counter_map"][ub.data]
    advances = [
        (stmt, parents)
        for stmt, parents in statements(after)
        if isinstance(stmt, tirx.BufferStore) and stmt.buffer.same_as(clock) and not isinstance(stmt.value, tirx.IntImm)
    ]
    assert len(advances) == 1
    assert not any(isinstance(parent, tirx.For) and parent.loop_var.same_as(j) for parent in advances[0][1])


@pytest.mark.parametrize("case", ["union", "cross-stage", "late-definition"])
def test_lexical_epoch_guard_is_available_at_first_access(case):
    a = tirx.decl_buffer((64,), "float32", name="A")
    c = tirx.decl_buffer((64,), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    p, q = tirx.Var("p", "bool"), tirx.Var("q", "bool")
    late = tirx.Var("late", "bool")
    i = tirx.Var("i", "int32")
    body = [unit(copy(a, ub), guard=p)]
    if case == "late-definition":
        body.append(unit(tirx.Bind(late, q), core=3))
    body.append(unit(copy(ub, c), guard=late if case == "late-definition" else q, stage=1 if case == "cross-stage" else 0))
    before = kernel(unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, seq(*body)), core=None), buffers=[ub], params=[a, c, p, q])
    after = transform.PrepareMultiBuffer()(before)
    (loop,) = nodes(after, tirx.For)
    guard = loop.annotations["tl.storage_epoch_guard_map"].get(ub.data, tirx.const(True, "bool"))
    expected = tirx.Or(p, q) if case == "union" else tirx.const(True, "bool")
    assert tvm.arith.Analyzer().can_prove_equal(guard, expected)


@pytest.mark.parametrize("mode, owners", [("auto", 1), ("counter", 1), ("auto", 2)])
def test_cross_stage_storage_cannot_use_counter_clock(mode, owners):
    a = tirx.decl_buffer((64,), "float32", name="A")
    c = tirx.decl_buffer((64,), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    guard = i % 2 == 0
    body = seq(unit(copy(a, ub), guard=guard), unit(copy(ub, c), guard=guard, stage=1))
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [ub.data]})
    owner_loops = []
    for index in range(owners):
        iv = tirx.Var(f"owner{index}", "int32")
        owner_loops.append(
            unit(
                tirx.For(iv, 0, 4, tirx.ForKind.SERIAL, tirx.stmt_functor.substitute(body, {i: iv}), annotations=loop.annotations),
                core=None,
            )
        )
    before = kernel(
        seq(*owner_loops),
        buffers=[ub],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {ub.data: 2}, "tl.buffer_version_mode": {ub.data: mode}},
    )
    if mode == "counter" or owners > 1:
        with pytest.raises(tvm.error.InternalError, match="Align their T.Stage values" if owners > 1 else "multiple schedule stages"):
            transform.PrepareMultiBuffer()(before)
    else:
        after = transform.PrepareMultiBuffer()(before)
        assert all("tl.multi_buffer_counter_map" not in loop.annotations for loop in nodes(after, tirx.For))
