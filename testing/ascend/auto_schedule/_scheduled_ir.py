"""Explicit pass inputs: these builders do not run any compiler passes.

A unit carries a stage and one atomic task; a loop is a unit without a task.
Core masks are Vector0=1, Cube=2, Vector1=4, Vector=5, All=7. core=0 creates
an unassigned task without a core_mask key; core=None omits the task wrapper.
Tests choose the order, stages, version counts and guards independently of
AutoSchedule's solution.
"""

from testing.ascend._ir import copy, kernel, seq
import tvm_ffi
from tvm_tirx import ir
from tilelang._compat.tensor import TensorView


def _definitions(statement):
    if isinstance(statement, ir.Bind):
        return [statement.var]
    if isinstance(statement, ir.RegionStmt):
        return list(statement.result_vars)
    if isinstance(statement, ir.SeqStmt):
        return [variable for child in statement.seq for variable in _definitions(child)]
    return []


def _marker(key, payload, body):
    definitions = _definitions(body)
    replacements = {}

    def rename(variable, engine):
        return replacements[variable] if variable in replacements else engine.default_mutate(variable)

    for variable in definitions:
        variable_type = tvm_ffi.structural_mutate(variable.ty, (ir.Var, rename))
        replacements[variable] = ir.Var(variable.name + "_task", variable_type)
    attributes = {key: payload}
    if definitions:
        body = tvm_ffi.structural_mutate(body, (ir.Var, rename))
        attributes["tl.schedule_yield"] = [replacements[variable] for variable in definitions]
    return ir.RegionStmt("tl.ascend.schedule_scope", [], [], attributes, body, definitions)


def task(body, *, core=5, cost=None):
    metadata = {"core_mask": ir.IntImm("int64", core)} if core else {}
    if cost is not None:
        metadata.update({"latency": ir.IntImm("int64", cost[0]), "ii": ir.IntImm("int64", cost[1])})
    return _marker("tl.ascend_task", metadata, body)


def unit(body, *, core=5, stage=0, guard=None, cost=None):
    if core is not None:
        body = task(body, core=core, cost=cost)
    if guard is not None:
        body = _marker("tl.schedule_condition", guard, body)
    return _marker("tl.schedule_unit", {"stage": stage}, body)


def copy_ring(*, mode="counter", versions=2, owners=1, guarded=False, prepared=False, core=5):
    """A GM->UB->GM ring, either before Prepare or before Sync/Materialize.

    Prepared inputs explicitly carry the shared counter and epoch guards. This
    lets the three passes be tested separately, without deriving one fixture
    from the output of another pass.
    """
    source = TensorView.create("A", (owners * 4, 64), "float32").tensor
    output = TensorView.create("C", (owners * 4, 64), "float32").tensor
    shared = TensorView.create("ub", (64,), "float32", scope="shared.dyn").tensor
    counter = TensorView.create("epoch", (1,), "int32", scope="local.var").tensor if prepared and mode == "counter" else None
    bodies = []
    if counter is not None:
        bodies.append(unit(ir.TensorStore(counter, [0], 0), core=core))
    for owner in range(owners):
        index = ir.Var(f"i{owner}", "int32")
        guard = index % 2 == 0 if guarded else None
        tasks = [
            unit(copy(source, shared, src_indices=[owner * 4 + index, 0], src_shape=[1, 64]), core=core, guard=guard),
            unit(copy(shared, output, dst_indices=[owner * 4 + index, 0], dst_shape=[1, 64]), core=core, guard=guard),
        ]
        annotations = {"multi_buffer_eligible": [shared]}
        if prepared:
            annotations["tl.storage_epoch_guard_map"] = {shared: guard if guarded else ir.const(True, "bool")}
            if counter is not None:
                annotations["tl.multi_buffer_counter_map"] = {shared: counter}
                tasks.append(unit(ir.TensorStore(counter, [0], counter[0] + 1), core=core, guard=guard))
        bodies.append(unit(ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, seq(*tasks), annotations=annotations), core=None))
    annotations = {"tl.buffer_versions_map": {shared: versions}}
    if not prepared:
        annotations["tl.buffer_version_mode"] = {shared: mode}
    return kernel(
        seq(*bodies), buffers=[shared] + ([counter] if counter is not None else []), params=[source, output], annotations=annotations
    )
