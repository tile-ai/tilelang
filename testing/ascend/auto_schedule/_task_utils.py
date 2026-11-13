"""Shared task programs and helpers for Ascend scheduling tests."""

import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.backend.target import determine_target
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _make_per_core_strided_store_program(*, explicit=False):
    num_tokens = T.dynamic("num_tokens")

    @T.macro
    def store_stages(ub, out, sid, count):
        for stage in T.serial(count):
            begin = stage * 128 + sid * 64
            extent = T.min(num_tokens - begin, 64)
            T.copy(ub[stage * 64 : stage * 64 + extent], out[begin : begin + extent])

    @T.prim_func
    def main(src: T.Tensor((512,), "float32"), out: T.Tensor((num_tokens,), "float32")):
        with T.MixedKernel(1) as (_, sid):
            ub = T.alloc_shared((256,), "float32")
            T.assume(num_tokens <= 512)
            T.copy(src[sid * 256 : sid * 256 + 256], ub)
            count = T.ceildiv(num_tokens - sid * 64, 128)
            with T.PerCoreTask():
                if explicit:
                    with T.Task():
                        store_stages(ub, out, sid, count)
                else:
                    store_stages(ub, out, sid, count)

    return main


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


def _collect_task_metadata(mod):
    metadata = {"latency": [], "ii": [], "core_mask": []}

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.ascend_task":
            for key in metadata:
                if key in node.node:
                    metadata[key].append(int(node.node[key]))

    post_order_visit(mod["main"].body, visit)
    return metadata


def _collect_schedule_units(mod):
    units = []

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit":
            units.append(node)

    post_order_visit(mod["main"].body, visit)
    return units
