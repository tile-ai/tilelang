import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def make_multi_kernel_program():
    @T.prim_func
    def main(
        a: T.Buffer((64,), "float32"),
        b: T.Buffer((64,), "float32"),
        c: T.Buffer((64,), "float32"),
    ):
        with T.Kernel(1):
            T.annotate_unlimit_memory("shared")
            ub0 = T.alloc_shared((64,), "float32")
            T.copy(a, ub0)
            T.copy(ub0, c)

        with T.Kernel(1):
            T.annotate_unlimit_memory("shared.l1")
            ub1 = T.alloc_shared((64,), "float32")
            T.copy(c, ub1)
            T.copy(ub1, b)

    return main


def test_auto_schedule_rewrites_every_tilelang_kernel():
    snapshots = {}
    snapshot_modules = {}
    wanted = {
        "tl.EstimateLatency",
        "tl.AutoSchedule",
        "tl.AssignCore",
        "tl.PrepareMultiBuffer",
        "tl.ResolveCore",
        "tl.InsertSync",
        "tl.MaterializeMultiBuffer",
        "tl.LowerScheduledTIR",
    }

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots.setdefault(info.name, []).append(mod.script())
                snapshot_modules.setdefault(info.name, []).append(mod)

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        artifact = tilelang.lower(make_multi_kernel_program(), target="ascend")

    assert artifact.target.kind.name == "ascend"
    assert artifact.kernel_source.count("__global__ __vector__") == 2
    assert {gvar.name_hint for gvar in artifact.device_mod.functions} == {
        "main_kernel",
        "main_kernel_1",
    }
    host_ir = artifact.host_mod.script()
    assert host_ir.index('T.call_packed("main_kernel"') < host_ir.index('T.call_packed("main_kernel_1"')

    assert snapshots.keys() >= wanted
    assert len(snapshots["tl.AssignCore"]) == 1
    assert len(snapshots["tl.ResolveCore"]) == 1
    for pass_snapshots in snapshots.values():
        for snapshot in pass_snapshots:
            assert snapshot.count('T.sblock("tilelang_root")') == 2

    for snapshot in snapshots["tl.EstimateLatency"]:
        assert snapshot.count("tl.unlimit_memory_scopes") == 2
        assert '"shared"' in snapshot
        assert '"shared.l1"' in snapshot

    scheduled_and_later = (
        snapshots["tl.AutoSchedule"]
        + snapshots["tl.AssignCore"]
        + snapshots["tl.PrepareMultiBuffer"]
        + snapshots["tl.ResolveCore"]
        + snapshots["tl.InsertSync"]
        + snapshots["tl.MaterializeMultiBuffer"]
        + snapshots["tl.LowerScheduledTIR"]
    )
    for snapshot in scheduled_and_later:
        assert "tl.unlimit_memory_scopes" not in snapshot

    scheduled = snapshots["tl.AutoSchedule"][0]
    first_assigned = snapshots["tl.AssignCore"][0]
    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    synchronized = snapshots["tl.InsertSync"][0]
    materialized = snapshots["tl.MaterializeMultiBuffer"][0]
    lowered = snapshots["tl.LowerScheduledTIR"][0]
    assert synchronized.count("tl.buffer_alias_map") == 2
    assert lowered.count("tl.buffer_alias_map") == 2
    assert '"tl.schedule_unit"' in scheduled
    assert '"tl.schedule_unit"' in first_assigned
    assert '"tl.schedule_unit"' in prepared
    assert '"tl.schedule_unit"' in synchronized
    assert '"tl.schedule_unit"' in materialized
    assert '"tl.schedule_unit"' not in lowered
    assert '"tl.ascend_task"' not in lowered
    assert '"tl.ascend_per_core_task"' not in lowered

    task_lines = [line for line in scheduled.splitlines() if '"tl.ascend_task"' in line]
    assert task_lines
    assert all('"stage"' not in line and '"core_mask"' not in line for line in task_lines)

    assigned_task_lines = [line for line in first_assigned.splitlines() if '"tl.ascend_task"' in line]
    assert len(assigned_task_lines) == len(task_lines)
    assert all('"core_mask"' in line and '"stage"' not in line for line in assigned_task_lines)
    assigned_unit_lines = [line for line in first_assigned.splitlines() if '"tl.schedule_unit"' in line]
    assert assigned_unit_lines
    assert all('"stage"' in line and '"core_mask"' not in line for line in assigned_unit_lines)

    assigned_task_payloads = []
    assigned_unit_payloads = []

    def collect_assigned_payloads(node):
        if not isinstance(node, tirx.AttrStmt):
            return
        if node.attr_key == "tl.ascend_task":
            assigned_task_payloads.append(node.node)
        elif node.attr_key == "tl.schedule_unit":
            assigned_unit_payloads.append(node.node)

    post_order_visit(
        snapshot_modules["tl.AssignCore"][0]["main"].body,
        collect_assigned_payloads,
    )
    assert assigned_task_payloads
    assert assigned_unit_payloads
    for payload in assigned_task_payloads:
        assert "core_mask" in payload
        assert "__tl_internal__" not in payload

    synchronized_task_payloads = []
    synchronized_unit_payloads = []

    def collect_synchronized_payloads(node):
        if not isinstance(node, tirx.AttrStmt):
            return
        if node.attr_key == "tl.ascend_task":
            synchronized_task_payloads.append(node.node)
        elif node.attr_key == "tl.schedule_unit":
            synchronized_unit_payloads.append(node.node)

    post_order_visit(
        snapshot_modules["tl.InsertSync"][0]["main"].body,
        collect_synchronized_payloads,
    )
    assert len(synchronized_task_payloads) > len(assigned_task_payloads)
    assert synchronized_unit_payloads
    num_new_tasks = len(synchronized_task_payloads) - len(assigned_task_payloads)
    num_new_units = len(synchronized_unit_payloads) - len(assigned_unit_payloads)
    # Ordinary compiler-generated sync tasks each own a new ScheduleUnit.
    assert num_new_tasks == num_new_units
    for payload in synchronized_task_payloads:
        assert "core_mask" in payload
        assert "__tl_internal__" not in payload
    assert all("core_mask" not in payload for payload in synchronized_unit_payloads)
    assert all(int(payload["stage"]) >= 0 for payload in synchronized_unit_payloads)
