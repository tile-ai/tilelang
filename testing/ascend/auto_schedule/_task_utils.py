"""Minimal task input shared by scheduling and latency tests."""

import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
import tilelang
from tilelang.backend.target import determine_target
from tvm_tirx import ir, tirx
from testing.ascend._ir import attribute_fields, scheduled_fields, statements


def _make_program(latency=None, ii=None):
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((16,), "float32")
            with T.Task(latency=latency, ii=ii):
                T.copy(A, temp)
            T.copy(temp, B)

    return main


def _bind_target(program):
    mod = ir.IRModule({"main": program.with_attr("global_symbol", "main")})
    mod = tilelang.transform.NormalizeAliasAnnotations()(mod)
    mod = tilelang.transform.LowerTensorSpec()(mod)
    return tirx.transform.BindTarget(determine_target("ascend"))(mod)


def _materialize_schedule_units(mod):
    return ascend_transform.MaterializeScheduleUnits()(mod)


def _is_allocation_task(node):
    if scheduled_fields(node) is None or not isinstance(node.var.ty, tirx.TensorType):
        return False
    value = node.value.args[0]
    if not isinstance(value, ir.Call) or not isinstance(value.op, ir.Op) or value.op.name != "tirx.alloc_tensor":
        return False
    assert not value.args and len(value.ty_args) == 1
    ir.assert_structural_equal(value.ty_args[0], node.var.ty)
    ir.assert_structural_equal(value.ty, node.var.ty)
    return True


def _collect_allocation_tasks(mod):
    """Return flat scheduled allocation definitions, retaining every runtime Bind."""
    return [node for node, _ in statements(mod) if _is_allocation_task(node)]


def _collect_task_metadata(mod, *, include_allocations=True):
    metadata = {"latency": [], "ii": [], "core_mask": []}
    for node, _ in statements(mod):
        fields = None
        if isinstance(node, ir.RegionStmt) and node.op.name == "tl.attr":
            values = attribute_fields(node)
            if values["key"] == "tl.ascend_task":
                fields = values["node"]
        elif (values := scheduled_fields(node)) is not None:
            fields = values["task"]
        if fields is not None and (include_allocations or not _is_allocation_task(node)):
            for key in metadata:
                if key in fields:
                    metadata[key].append(int(fields[key]))
    return metadata


def _collect_schedule_units(mod):
    units = []
    for node, _ in statements(mod):
        if (
            isinstance(node, ir.RegionStmt) and node.op.name == "tl.attr" and attribute_fields(node)["key"] == "tl.schedule_unit"
        ) or scheduled_fields(node) is not None:
            units.append(node)
    return units
