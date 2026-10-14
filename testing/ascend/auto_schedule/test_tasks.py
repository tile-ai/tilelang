"""Atomic task boundaries and per-core candidate synchronization."""

import pytest
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import calls, copy, kernel, nodes, region, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


@pytest.mark.parametrize(
    "case, message",
    [
        ("cross-pipe", "exactly one Ascend hardware pipe"),
        ("cross-core", "both AIC and AIV HBM paths"),
        ("scalar-fill", "exactly one Ascend hardware pipe"),
        ("vector-copy", "exactly one Ascend hardware pipe"),
        ("two-syncs", "must be the task's only statement"),
        ("nested-per-core", "cannot be nested inside"),
    ],
)
def test_invalid_atomic_task(case, message):
    @T.prim_func
    def before(A: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 16), "float32")
            l0 = T.alloc_l0a((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            other = T.alloc_shared((16, 16), "float32")
            with T.Task():
                if case == "cross-pipe":
                    T.copy(A, l1)
                    T.copy(l1, l0)
                elif case == "cross-core":
                    T.copy(A, l1)
                    T.copy(A, ub)
                elif case == "scalar-fill":
                    T.copy(A, ub)
                    T.fill(ub, 0)
                elif case == "vector-copy":
                    T.copy(A, ub)
                    T.copy(ub, other)
                elif case == "two-syncs":
                    T.ascend_sync_inter_arrive("PIPE_V", 3)
                    T.ascend_sync_inter_wait("PIPE_V", 3)
                else:
                    with T.PerCoreTask():
                        T.copy(A, ub)

    with pytest.raises(tvm.error.InternalError, match=message):
        transform.MaterializeScheduleUnits()(tvm.IRModule({"main": before}))


@pytest.mark.parametrize("explicit, mixed", [(True, False), (False, False), (True, True)], ids=["explicit", "inferred", "mixed"])
def test_per_core_candidates_preserve_conditions(explicit, mixed):
    @T.prim_func
    def before(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32"), enabled: T.bool):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 16), "float32")
            with T.PerCoreTask():
                if enabled:
                    if explicit:
                        with T.Task():
                            T.copy(A, l1)
                    else:
                        T.copy(A, l1)
                if not enabled:
                    if explicit and not mixed:
                        with T.Task():
                            T.copy(B, l1)
                    else:
                        T.copy(B, l1)

    after = transform.MaterializeScheduleUnits()(tvm.IRModule({"main": before}))
    groups = [node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.ascend_per_core_task"]
    assert len(groups) == 1
    candidates = [node for node in nodes(groups[0].body, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]
    assert len(candidates) == 2
    assert len(nodes(groups[0].body, tirx.IfThenElse)) == 2


def test_sync_set_is_migrated_into_each_per_core_candidate():
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    b = tirx.decl_buffer((16, 16), "float32", name="B")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    l0 = tirx.decl_buffer((16, 16), "float32", name="l0", scope="shared.l0a")
    enabled = tirx.Var("enabled", "bool")
    candidates = seq(
        tirx.IfThenElse(enabled, unit(copy(a, l1), core=2).body, None),
        tirx.IfThenElse(tirx.Not(enabled), unit(copy(b, l1), core=2).body, None),
    )
    group = tirx.AttrStmt({"core_mask": tirx.IntImm("int64", 2)}, "tl.ascend_per_core_task", 1, candidates)
    before = kernel(seq(unit(group, core=None), unit(copy(l1, l0), core=2)), buffers=[l1, l0], params=[a, b, enabled])
    after = transform.InsertSync()(before)
    sets = [
        stmt.value
        for stmt, _ in statements(after)
        if isinstance(stmt, tirx.Evaluate)
        and isinstance(stmt.value, tirx.Call)
        and stmt.value.op.name == "tl.ascend_set_flag"
        and stmt.value.args[0].value == "MTE2_MTE1"
    ]
    waits = [call for call in calls(after, "tl.ascend_wait_flag") if call.args[0].value == "MTE2_MTE1"]
    assert len(sets) == 2 and len(waits) == 1
    guards = []
    for stmt, parents in statements(after):
        if isinstance(stmt, tirx.Evaluate) and any(stmt.value.same_as(event) for event in sets):
            guards.append(next(parent.condition for parent in reversed(parents) if isinstance(parent, tirx.IfThenElse)))
    assert len(guards) == 2
    assert tvm.arith.Analyzer().can_prove(tirx.Or(*guards))
    assert all(tvm.arith.Analyzer().can_prove_equal(event.args[1], waits[0].args[1]) for event in sets)


@pytest.mark.parametrize("dynamic", [False, True], ids=["static-fill", "dynamic-fill"])
def test_fill_outside_vf_synchronizes_on_scalar_pipe(dynamic):
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((64,), "float32", name="out")
    extent = tirx.Var("extent", "int32")
    fill = tirx.Evaluate(
        tirx.Call("handle", tvm.ir.Op.get("tl.tileop.fill"), [region(ub, 2, shape=[extent if dynamic else 64]), tirx.const(0, "float32")])
    )
    before = kernel(seq(unit(fill), unit(copy(ub, out))), buffers=[ub], params=[out, extent])
    after = transform.InsertSync()(before)
    for op in ("set", "wait"):
        assert len([call for call in calls(after, "tl.ascend_" + op + "_flag") if call.args[0].value == "S_MTE3"]) == 1
