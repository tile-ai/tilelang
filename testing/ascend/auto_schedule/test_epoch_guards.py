"""PrepareMultiBuffer groups equivalent clocks and respects lexical scope."""

import pytest
import tvm_tirx
import tvm_ffi
from tilelang.ascend import transform
from tvm_tirx import ir, tirx
from testing.ascend._ir import copy, kernel, loop_annotations, nodes, seq, serial, statements, tensor
from testing.ascend.auto_schedule._scheduled_ir import unit


@pytest.mark.parametrize("relation, share_clock", [("same", True), ("equivalent", True), ("different", False), ("mutated-memory", False)])
def test_guard_equivalence_preserves_storage_epochs(relation, share_clock):
    a = tensor((64,), "float32", name="A")
    c = tensor((64,), "float32", name="C")
    pred = tensor((1,), "int32", name="pred")
    x = tensor((64,), "float32", name="x", scope="shared.dyn")
    y = tensor((64,), "float32", name="y", scope="shared.dyn")
    p, q = ir.Var("p", "int32"), ir.Var("q", "int32")
    i = ir.Var("i", "int32")
    first = p > 0
    second = p > 0 if relation == "same" else (p >= 1 if relation == "equivalent" else q > 0)
    prefix, middle = [], []
    if relation == "mutated-memory":
        first, second = ir.Var("before_write", "bool"), ir.Var("after_write", "bool")
        prefix = [unit(ir.Bind(first, pred[0] > 0), core=7)]
        middle = [unit(ir.TensorStore(pred, [0], 1), core=7), unit(ir.Bind(second, pred[0] > 0), core=7)]
    body = seq(
        *prefix,
        unit(copy(a, x), guard=first),
        unit(copy(x, c), guard=first),
        *middle,
        unit(copy(a, y), guard=second),
        unit(copy(y, c), guard=second),
    )
    loop = serial(i, 0, 4, body, annotations={"multi_buffer_eligible": [x, y]})
    before = kernel(
        unit(loop, core=None),
        buffers=[x, y],
        params=[a, c, pred, p, q],
        annotations={"tl.buffer_versions_map": {x: 2, y: 3}, "tl.buffer_version_mode": {x: "counter", y: "counter"}},
        allocation_stage=0,
    )
    after = transform.PrepareMultiBuffer()(before)
    (owner,) = nodes(after, ir.For)
    clocks = loop_annotations(owner)["tl.multi_buffer_counter_map"]
    assert clocks[x].same_as(clocks[y]) == share_clock
    for storage, clock in clocks.items():
        updates = [
            (stmt, parents)
            for stmt, parents in statements(after)
            if isinstance(stmt, ir.TensorStore) and stmt.dest.same_as(clock) and not isinstance(stmt.value, ir.IntImm)
        ]
        assert len(updates) == 1
        expected = loop_annotations(owner)["tl.storage_epoch_guard_map"][storage]
        assert any(
            isinstance(parent, ir.If) and tvm_tirx.sym.Analyzer().can_prove_equal(parent.condition, expected) for parent in updates[0][1]
        )


@pytest.mark.parametrize("case", ["inner-guard", "mutated-bound"])
def test_nested_condition_cannot_escape_into_outer_epoch(case):
    a = tensor((64,), "float32", name="A")
    c = tensor((64,), "float32", name="C")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    extent = ir.Var("extent", "int32")
    bound = tensor((1,), "int32", name="bound")
    guard = j % 2 == 0 if case == "inner-guard" else None
    tasks = [unit(copy(a, ub), guard=guard)]
    if case == "mutated-bound":
        tasks.append(unit(ir.TensorStore(bound, [0], 0)))
    tasks.append(unit(copy(ub, c), guard=guard))
    child = unit(serial(j, 0, extent, seq(*tasks)), core=None)
    body = seq(unit(ir.Bind(extent, bound[0])), child) if case == "mutated-bound" else child
    owner = serial(i, 0, 4, body, annotations={"multi_buffer_eligible": [ub]})
    before = kernel(
        unit(owner, core=None),
        buffers=[ub],
        params=[a, c, bound if case == "mutated-bound" else extent],
        annotations={"tl.buffer_versions_map": {ub: 1 if case == "mutated-bound" else 2}, "tl.buffer_version_mode": {ub: "counter"}},
        allocation_stage=0,
    )
    after = transform.PrepareMultiBuffer()(before)
    (owner,) = [loop for loop in nodes(after, ir.For) if loop.var.same_as(i)]
    assert tvm_tirx.sym.Analyzer().can_prove(loop_annotations(owner)["tl.storage_epoch_guard_map"][ub])
    clock = loop_annotations(owner)["tl.multi_buffer_counter_map"][ub]
    advances = [
        (stmt, parents)
        for stmt, parents in statements(after)
        if isinstance(stmt, ir.TensorStore) and stmt.dest.same_as(clock) and not isinstance(stmt.value, ir.IntImm)
    ]
    assert len(advances) == 1
    assert not any(isinstance(parent, ir.For) and parent.var.same_as(j) for parent in advances[0][1])


@pytest.mark.parametrize("case", ["union", "cross-stage", "late-definition", "late-memory"])
def test_lexical_epoch_guard_is_available_at_first_access(case):
    a = tensor((64,), "float32", name="A")
    c = tensor((64,), "float32", name="C")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    p, q = ir.Var("p", "bool"), ir.Var("q", "bool")
    late = ir.Var("late", "bool")
    i = ir.Var("i", "int32")
    params = [a, c, p, q]
    body = [unit(copy(a, ub), guard=p)]
    if case == "late-definition":
        body.append(unit(ir.Bind(late, q), core=7))
    elif case == "late-memory":
        pred = tensor((1,), "int32", name="pred")
        params.append(pred)
        body.extend([unit(ir.TensorStore(pred, [0], 1), core=7), unit(ir.Bind(late, pred[0] > 0), core=7)])
    body.append(unit(copy(ub, c), guard=late if case.startswith("late-") else q, stage=1 if case == "cross-stage" else 0))
    before = kernel(unit(serial(i, 0, 4, seq(*body)), core=None), buffers=[ub], params=params, allocation_stage=0)
    after = transform.PrepareMultiBuffer()(before)
    assert tirx.analysis.check_well_formed(after)
    assert not tirx.analysis.undefined_vars(after["main"])
    (loop,) = nodes(after, ir.For)
    guard = loop_annotations(loop)["tl.storage_epoch_guard_map"].get(ub, ir.const(True, "bool"))
    # A pure late SSA alias can be replayed using the dominating parameter q;
    # a post-write memory snapshot and a different stage cannot be replayed.
    expected = tirx.Or(p, q) if case in ("union", "late-definition") else ir.const(True, "bool")
    assert tvm_tirx.sym.Analyzer().can_prove_equal(guard, expected)


@pytest.mark.parametrize("mode, owners", [("auto", 1), ("counter", 1), ("auto", 2)])
def test_cross_stage_storage_cannot_use_counter_clock(mode, owners):
    a = tensor((64,), "float32", name="A")
    c = tensor((64,), "float32", name="C")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    i = ir.Var("i", "int32")
    guard = i % 2 == 0
    body = seq(unit(copy(a, ub), guard=guard), unit(copy(ub, c), guard=guard, stage=1))
    loop = serial(i, 0, 4, body, annotations={"multi_buffer_eligible": [ub]})
    owner_loops = []
    for index in range(owners):
        iv = ir.Var(f"owner{index}", "int32")
        owner_loops.append(
            unit(
                serial(
                    iv,
                    0,
                    4,
                    tvm_ffi.structural_map(body, (ir.Var, lambda value, iv=iv: iv if value.same_as(i) else value), order="pre"),
                    annotations=loop_annotations(loop),
                ),
                core=None,
            )
        )
    before = kernel(
        seq(*owner_loops),
        buffers=[ub],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {ub: 2}, "tl.buffer_version_mode": {ub: mode}},
        allocation_stage=0,
    )
    if mode == "counter" or owners > 1:
        with pytest.raises(RuntimeError, match="Align their T.Stage values" if owners > 1 else "multiple schedule stages"):
            transform.PrepareMultiBuffer()(before)
    else:
        after = transform.PrepareMultiBuffer()(before)
        assert all("tl.multi_buffer_counter_map" not in loop_annotations(loop) for loop in nodes(after, ir.For))
