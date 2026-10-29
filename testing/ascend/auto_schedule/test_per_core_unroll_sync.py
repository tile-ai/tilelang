"""Explicit unrolling must not multiply a PerCoreTask's external handshakes."""

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from testing.ascend._ir import nodes, schedule, statements
from tvm import tirx


def _program(explicit):
    @T.prim_func
    def main(A: T.Tensor((16, 16), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 16), "bfloat16")
            l0 = T.alloc_l0a((16, 16), "bfloat16")
            with T.Stage(0):
                T.copy(A, l1)
                with T.PerCoreTask():
                    for _i in T.unroll(2, explicit=explicit):
                        with T.Task():
                            T.copy(l1, l0)
                T.copy(A, l1)

    return main


def _assert_balanced_once(mod):
    # The migrator reuses synchronization IR nodes across candidates. Count
    # executed statement sites, rather than deduplicating nodes by identity.
    sites = [
        stmt.value
        for stmt, _ in statements(mod)
        if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call) and isinstance(stmt.value.op, tvm.ir.Op)
    ]
    for pipe in ("MTE2_MTE1", "MTE1_MTE2"):
        events = {
            operation: [call for call in sites if call.op.name == "tl.ascend_" + operation + "_flag" and call.args[0].value == pipe]
            for operation in ("set", "wait")
        }
        assert len(events["set"]) == len(events["wait"]) == 1, events
        tvm.ir.assert_structural_equal(events["set"][0].args[1], events["wait"][0].args[1])


@pytest.mark.parametrize("explicit", [False, True], ids=["retained-loop", "explicitly-unrolled-loop"])
def test_per_core_handshake_after_unroll_prefix(explicit):
    # This prefix runs UnrollLoopSkipVF before MaterializeScheduleUnits; a
    # prepared InsertSync input would miss the earlier loss of the For node.
    after = schedule(_program(explicit), through="InsertSync", auto_schedule=False, infer_layouts=True)
    _assert_balanced_once(after)


@pytest.mark.parametrize("explicit", [False, True], ids=["retained-loop", "explicitly-unrolled-loop"])
def test_per_core_handshake_and_boundary_metadata_survive_lowering(explicit):
    snapshots = {}
    boundaries = {"tl.UnrollLoopSkipVF", "tl.AssignCore", "tl.InsertSync"}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in boundaries:
                snapshots[info.name] = mod

    with tvm.target.Target("ascend"), tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(_program(explicit), target="ascend", enable_device_compile=False)
    assert artifact.kernel_source
    assert snapshots.keys() == boundaries
    for mod in snapshots.values():
        (group,) = [node for node in nodes(mod, tirx.AttrStmt) if node.attr_key == "tl.ascend_per_core_task"]
        assert int(group.node["boundary_sync"]) == 1
        assert bool(nodes(group.body, tirx.For)) == (not explicit)
    _assert_balanced_once(snapshots["tl.InsertSync"])
