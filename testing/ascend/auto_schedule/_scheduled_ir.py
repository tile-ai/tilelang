"""Explicit pass inputs: these builders do not run any compiler passes.

A unit carries a stage and one atomic task; a loop is a unit without a task.
Assigned core masks are Vector=1, Cube=2, flexible scalar=3. core=0 creates
an unassigned task without a core_mask key; core=None omits the task wrapper.
Tests choose the order, stages, version counts and guards independently of
AutoSchedule's solution.
"""

from testing.ascend._ir import copy, kernel, seq
from tvm import tirx


def unit(body, *, core=1, stage=0, guard=None, cost=None):
    if core is not None:
        metadata = {"core_mask": tirx.IntImm("int64", core)} if core else {}
        if cost is not None:
            metadata.update({"latency": tirx.IntImm("int64", cost[0]), "ii": tirx.IntImm("int64", cost[1])})
        body = tirx.AttrStmt(metadata, "tl.ascend_task", 1, body)
    if guard is not None:
        body = tirx.IfThenElse(guard, body, None)
    return tirx.AttrStmt({"stage": stage}, "tl.schedule_unit", 1, body)


def copy_ring(*, mode="counter", versions=2, owners=1, guarded=False, prepared=False, core=1):
    """A GM->UB->GM ring, either before Prepare or before Sync/Materialize.

    Prepared inputs explicitly carry the shared counter and epoch guards. This
    lets the three passes be tested separately, without deriving one fixture
    from the output of another pass.
    """
    a = tirx.decl_buffer((owners * 4, 64), "float32", name="A")
    c = tirx.decl_buffer((owners * 4, 64), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    counter = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var") if prepared and mode == "counter" else None
    bodies = []
    if counter is not None:
        bodies.append(unit(tirx.BufferStore(counter, 0, [0]), core=core))
    for owner in range(owners):
        i = tirx.Var(f"i{owner}", "int32")
        guard = i % 2 == 0 if guarded else None
        tasks = [
            unit(copy(a, ub, src_indices=[owner * 4 + i, 0], src_shape=[1, 64]), guard=guard),
            unit(copy(ub, c, dst_indices=[owner * 4 + i, 0], dst_shape=[1, 64]), guard=guard),
        ]
        annotations = {"multi_buffer_eligible": [ub.data]}
        if prepared:
            annotations["tl.storage_epoch_guard_map"] = {ub.data: guard if guarded else tirx.const(True, "bool")}
            if counter is not None:
                annotations["tl.multi_buffer_counter_map"] = {ub.data: counter}
                tasks.append(unit(tirx.BufferStore(counter, counter[0] + 1, [0]), core=core, guard=guard))
        bodies.append(unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, seq(*tasks), annotations=annotations), core=None))
    annotations = {"tl.buffer_versions_map": {ub.data: versions}}
    if not prepared:
        annotations["tl.buffer_version_mode"] = {ub.data: mode}
    return kernel(seq(*bodies), buffers=[ub] + ([counter] if counter is not None else []), params=[a, c], annotations=annotations)
