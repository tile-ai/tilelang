"""Explicit native pass inputs; these builders do not run compiler passes.

Statement units use RegionStmt metadata, while SSA definitions use the native
scheduled_bind value carrier. Guards wrap units in their actual lexical scope.
Core masks: Vector0=1, Cube=2, Vector1=4, Vector=5, All=7; zero is unassigned.
"""

from tvm_tirx import ir
from testing.ascend._ir import attribute, copy, kernel, scheduled_bind, seq, tensor


def unit(body, *, core=5, stage=0, guard=None, cost=None):
    metadata = {"core_mask": ir.IntImm("int64", core)} if core else {}
    if cost is not None:
        metadata.update({"latency": ir.IntImm("int64", cost[0]), "ii": ir.IntImm("int64", cost[1])})
    if isinstance(body, ir.Bind):
        result = scheduled_bind(body, stage, metadata if core is not None else {})
    else:
        if core is not None:
            body = attribute(metadata, "tl.ascend_task", 1, [body])
        result = attribute({"stage": stage}, "tl.schedule_unit", 1, [body])
    # Native scheduling represents the branch as a lexical scope containing
    # complete units. A branch inside a task instead makes it atomic.
    return ir.If(guard, [result], None) if guard is not None else result


def copy_ring(*, mode="counter", versions=2, owners=1, guarded=False, prepared=False, core=5):
    """A GM->UB->GM ring before Prepare or before Sync/Materialize.

    Prepared cases explicitly contain their counter and epoch guard; no pass
    output is reused as the oracle for a different pass.
    """
    a = tensor((owners * 4, 64), "float32", name="A")
    c = tensor((owners * 4, 64), "float32", name="C")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    counter = tensor((1,), "int32", name="epoch", scope="local.var") if prepared and mode == "counter" else None
    bodies = []
    if counter is not None:
        bodies.append(unit(ir.TensorStore(counter, [0], ir.const(0)), core=core))
    for owner in range(owners):
        i = ir.Var(f"i{owner}", "int32")
        guard = i % 2 == 0 if guarded else None
        tasks = [
            unit(copy(a, ub, src_indices=[owner * 4 + i, 0], src_shape=[1, 64]), guard=guard),
            unit(copy(ub, c, dst_indices=[owner * 4 + i, 0], dst_shape=[1, 64]), guard=guard),
        ]
        annotations = {"multi_buffer_eligible": [ub]}
        if prepared:
            annotations["tl.storage_epoch_guard_map"] = {ub: guard if guarded else ir.const(True, "bool")}
            if counter is not None:
                annotations["tl.multi_buffer_counter_map"] = {ub: counter}
                tasks.append(unit(ir.TensorStore(counter, [0], ir.TensorLoad(counter, [0]) + 1), core=core, guard=guard))
        loop_body = attribute(None, "tl.loop.annotations", annotations, tasks)
        bodies.append(unit(ir.For(i, ir.const(0), ir.const(4), ir.const(1), ir.ForKind.DEFAULT, [loop_body]), core=None))
    annotations = {"tl.buffer_versions_map": {ub: versions}}
    if not prepared:
        annotations["tl.buffer_version_mode"] = {ub: mode}
    return kernel(
        seq(*bodies),
        buffers=[ub] + ([counter] if counter is not None else []),
        params=[a, c],
        annotations=annotations,
        allocation_stage=0,
    )
