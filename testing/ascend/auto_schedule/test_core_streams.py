"""Lowering preserves each AIV's program with a single subcore selection."""

import pytest
import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


@pytest.mark.parametrize("active", [(0, 1), (0,), (1,)])
def test_subcore_selection_preserves_bindings_and_order(active):
    sid = tirx.Var("sid", "int32")
    route = tirx.Var("route", "int32")
    i = tirx.Var("i", "int32")
    tasks = [unit(tirx.Bind(route, sid + 10), core=5)]
    for subcore in active:
        for label in ("before", "after"):
            tasks.append(
                unit(
                    tirx.Evaluate(tirx.call_extern("int32", f"{label}_{subcore}", i, route, sid)),
                    core=1 if subcore == 0 else 4,
                )
            )
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, seq(*tasks))
    before = kernel(unit(loop, core=None))
    func = before["main"]
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before = tvm.IRModule({"main": func.with_body(tirx.AttrStmt(thread, "thread_extent", 2, func.body))})
    after = transform.LowerScheduledTIR()(before)
    assert tirx.analysis.verify_ssa(after["main"])
    (vector,) = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "VECTOR"]
    (selection,) = [
        node for node in nodes(vector.body, tirx.IfThenElse) if any(var.same_as(sid) for var in nodes(node.condition, tirx.Var))
    ]
    assert isinstance(selection.condition, tirx.EQ) and int(selection.condition.b) == 0
    analyzer = tvm.arith.Analyzer()
    for subcore, branch in enumerate((selection.then_case, selection.else_case)):
        assert branch is not None
        (loop,) = nodes(branch, tirx.For)
        # The selector encloses the whole loop. No per-task sid predicates or
        # references to another AIV's scalar binding survive inside it.
        assert not any(var.same_as(sid) for var in nodes(loop, tirx.Var))
        records = calls(loop, "tirx.call_extern")
        assert [call.args[0].value for call in records] == ([f"before_{subcore}", f"after_{subcore}"] if subcore in active else [])
        bindings = nodes(loop, tirx.Bind)
        assert len(bindings) == 1
        assert int(analyzer.simplify(bindings[0].value)) == subcore + 10
        for call in records:
            assert call.args[2].same_as(bindings[0].var)
            assert int(call.args[3]) == subcore


def test_shared_simd_body_keeps_distinct_local_definitions():
    @T.prim_func
    def main(A: T.Tensor((2, 128), "float32"), B: T.Tensor((2, 128), "float32")):
        with T.MixedKernel(1) as (_, sid):
            a = T.alloc_shared((128,), "float32")
            b = T.alloc_shared((128,), "float32")
            T.copy(A[sid, :], a)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in range(2):
                    value = T.simd.vld(a[i * 64])
                    doubled = T.simd.vadd(value, value, mask)
                    T.simd.vsts(b[i * 64], doubled, mask)
            T.copy(b, B[sid, :])

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.InsertSync", "tl.LowerScheduledTIR"):
                snapshots[info.name] = mod

    # VF-local bindings and loops live inside one atomic scheduling task, so
    # projecting only the outer schedule units must also freshen these locals.
    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(main, target="ascend")
    for mod in snapshots.values():
        assert tirx.analysis.verify_ssa(mod["main"])
    after = snapshots["tl.LowerScheduledTIR"]
    (vector,) = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "VECTOR"]
    (selection,) = nodes(vector.body, tirx.IfThenElse)
    for branch in (selection.then_case, selection.else_case):
        assert len(calls(branch, "tl.simd.vadd")) == 1
        assert len(nodes(branch, tirx.For)) == 1
    assert "__simd_vf__" in artifact.kernel_source
