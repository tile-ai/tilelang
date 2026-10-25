"""Minimal task input shared by scheduling and latency tests."""

import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang._compat import tirx
from tilelang._compat.stmt_functor import post_order_visit
from tvm_tirx import ir


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
    mod = tvm.IRModule.from_expr(program.with_attr("global_symbol", "main"))
    return tirx.transform.BindTarget(determine_target("ascend"))(mod)


def _materialize_schedule_units(mod):
    return ascend_transform.MaterializeScheduleUnits()(mod)


def _collect_task_metadata(mod, *, include_allocations=True):
    metadata = {"latency": [], "ii": [], "core_mask": []}

    def visit(node):
        if isinstance(node, ir.RegionStmt) and "tl.ascend_task" in node.attrs.dict:
            if not include_allocations:
                body = node.body
                while isinstance(body, ir.SeqStmt) and len(body.seq) == 1:
                    body = body.seq[0]
                if isinstance(body, ir.Bind) and isinstance(body.value, ir.Call) and body.value.op.name == "tirx.alloc_tensor":
                    return
            attributes = node.attrs.dict["tl.ascend_task"]
            for key in metadata:
                if key in attributes:
                    metadata[key].append(int(attributes[key]))

    post_order_visit(mod["main"].body, visit)
    return metadata


def _collect_schedule_units(mod):
    units = []

    def visit(node):
        if isinstance(node, ir.RegionStmt) and "tl.schedule_unit" in node.attrs.dict:
            units.append(node)

    post_order_visit(mod["main"].body, visit)
    return units
