"""Small IR builders and inspection helpers for host-only Ascend pass tests.

The scheduling prefix ends at the requested pass. It deliberately excludes
device lowering, code generation, and compilation. Tests of a pass with an
already prepared input should call that pass directly instead.
"""

import tilelang
import tvm_tirx as tvm
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tilelang.ascend import transform as ascend
from tilelang._compat.stmt_functor import ir_transform, post_order_visit


def seq(*stmts):
    return stmts[0] if len(stmts) == 1 else ir.SeqStmt(stmts)


def unseq(statement):
    while isinstance(statement, ir.SeqStmt) and len(statement.seq) == 1:
        statement = statement.seq[0]
    return statement


def root_scope(module):
    return next(region for region in nodes(module, ir.RegionStmt) if region.attrs.get("name_hint") == "tilelang_root")


def guard_conditions(parents):
    return [
        parent.condition if isinstance(parent, ir.If) else parent.attrs["tl.schedule_condition"]
        for parent in parents
        if isinstance(parent, ir.If) or isinstance(parent, ir.RegionStmt) and "tl.schedule_condition" in parent.attrs
    ]


def simd_scope(body):
    return ir.RegionStmt("tl.scope", [], [], {"name_hint": "SIMD_VF"}, body)


def region(buffer, mask, indices=None, shape=None):
    indices = [0] * len(buffer.shape) if indices is None else indices
    shape = buffer.shape if shape is None else shape
    return ir.TensorRegion(
        buffer, [ir.Range.from_min_extent(index, extent) for index, extent in zip(indices, shape)], ir.TensorRegionType()
    )


def copy(src, dst, *, src_indices=None, dst_indices=None, src_shape=None, dst_shape=None, annotations=None):
    return ir.Evaluate(
        ir.Call(
            "tl.tileop.ascend_copy",
            [region(src, 1, src_indices, src_shape), region(dst, 2, dst_indices, dst_shape)],
            attrs=annotations or {},
            ty="handle",
        )
    )


def kernel(body, *, buffers=(), params=(), annotations=None):
    if isinstance(body, (list, tuple)):
        body = ir.SeqStmt(body)
    allocations = [ir.Bind(buffer, ir.Call("tirx.alloc_tensor", [], ty_args=[buffer.ty])) for buffer in buffers]
    markers = [region for region in nodes(body, ir.RegionStmt) if "tl.schedule_unit" in region.attrs]
    if markers:
        from testing.ascend.auto_schedule._scheduled_ir import unit

        children = list(body.seq) if isinstance(body, ir.SeqStmt) else [body]
        root_stages = [
            child.attrs["tl.schedule_unit"]["stage"]
            for child in children
            if isinstance(child, ir.RegionStmt) and "tl.schedule_unit" in child.attrs
        ]
        allocation_stage = -1 if root_stages and all(int(stage) == -1 for stage in root_stages) else 0
        unassigned = any(
            "core_mask" not in region.attrs["tl.ascend_task"] for region in nodes(body, ir.RegionStmt) if "tl.ascend_task" in region.attrs
        )
        allocations = [unit(allocation, core=0 if unassigned else 7, stage=allocation_stage, cost=(1, 1)) for allocation in allocations]
    root = ir.RegionStmt("tl.scope", [], [], {"name_hint": "tilelang_root", **(annotations or {})}, [*allocations, body])
    func = tirx.Function(list(params), root, ret_type=ir.TupleType([])).with_attr(
        {"global_symbol": "main", "target": tvm.target.Target("ascend")}
    )
    return tvm.IRModule({"main": func})


def nodes(ir, kind):
    if isinstance(ir, tvm.IRModule):
        ir = ir["main"].body
    elif isinstance(ir, tirx.Function):
        ir = ir.body
    found = []
    post_order_visit(ir, lambda node: found.append(node) if isinstance(node, kind) else None)
    return found


def calls(ir, name):
    return [node for node in nodes(ir, tvm.ir.Call) if isinstance(node.op, tvm.ir.Op) and node.op.name == name]


def access_ptr_offset(pointer):
    def byte_offset(expression):
        if expression.op.name == "tirx.ptr_byte_offset":
            return byte_offset(expression.args[0]) + prim.Cast("int64", expression.args[1])
        if expression.op.name == "tirx.reinterpret":
            return byte_offset(expression.args[0])
        if expression.op.name == "tirx.tensor_data_ptr":
            return ir.const(0, "int64")
        assert expression.op.name == "tl.access_ptr"
        load = expression.args[0]
        offset = ir.const(0, "int64")
        indices = [region.begin for region in load.region] if isinstance(load, ir.TensorRegion) else load.indices
        for extent, index in zip(load.source.shape, indices, strict=True):
            index = index.base if isinstance(index, prim.Ramp) else index
            offset = offset * prim.Cast("int64", extent) + prim.Cast("int64", index)
        if load.source.ty.layout is not None:
            offset = load.source.ty.layout.apply(*indices, shape=load.source.shape)["m"]
        return offset * load.source.dtype.bits // 8

    element_bits = pointer.ty.element_type.dtype.bits
    return tvm.sym.Analyzer().simplify(byte_offset(pointer) * 8 // element_bits)


def schedule_snapshots(program, pass_names, *, auto_schedule=True, infer_layouts=False):
    """Run the scheduling prerequisites and stop at the last requested boundary.

    This helper accepts frontend programs, including explicit unrolls and dual
    copies. Tests needing physical layouts can opt into layout inference or
    supply an input whose layouts are already canonical.
    """
    pipeline = (
        ascend.NormalizeControlFlowForSchedule,
        ascend.NormalizeConflictHints,
        ascend.MaterializeScheduleUnits,
        ascend.AssignCore,
        ascend.AnnotateMultiBufferEligible,
        ascend.EstimateLatency,
        ascend.AutoSchedule,
        ascend.PrepareMultiBuffer,
        ascend.ResolveCore,
        ascend.InsertSync,
        ascend.MaterializeMultiBuffer,
        ascend.LowerScheduledTIR,
        ascend.RestoreWhileLoops,
    )
    if not auto_schedule:
        pipeline = tuple(factory for factory in pipeline if factory != ascend.EstimateLatency)

    # For sync-only tests, the fixture supplies source order, explicit stages,
    # and explicit buffer counts. Fill only unspecified stages with zero; no
    # latency estimate or solver is needed to construct this scheduled input.
    def use_source_order(mod):
        def assign_stage(node):
            if isinstance(node, ir.RegionStmt) and "tl.schedule_unit" in node.attrs:
                metadata = dict(node.attrs["tl.schedule_unit"])
                if int(metadata["stage"]) < 0:
                    metadata["stage"] = 0
                    attributes = {**dict(node.attrs), "tl.schedule_unit": metadata}
                    return ir.RegionStmt(node.op, node.args, node.body_params, attributes, node.body, node.result_vars, node.loc)

        return tvm.IRModule({gvar: func.with_body(ir_transform(func.body, None, assign_stage)) for gvar, func in mod.functions.items()})

    requested = set(pass_names)
    available = {"tl." + factory.__name__ for factory in pipeline}
    if not requested:
        raise ValueError("At least one scheduling pass must be requested")
    if not requested <= available:
        raise ValueError(f"Unknown or disabled scheduling passes: {sorted(requested - available)}")
    snapshots = {}
    target = tvm.target.Target("ascend")
    with target:
        mod = tvm.IRModule({"main": program.with_attr("global_symbol", "main")})
        mod = tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch(
            lower_grid_binding=True,
            lower_thread_binding=False,
            default_threads=None,
            unsupported_annotations=["cluster_dims"],
            launch_dim_tags=["cthread"],
        )(mod)
        mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
        mod = tilelang.transform.InjectAssumes()(mod)
        mod = ascend.RewriteDualCopy()(mod)
        mod = ascend.UnrollLoopSkipVF()(mod)
        mod = tilelang.transform.Simplify()(mod)
        if infer_layouts:
            mod = ascend.AscendLayoutInference()(mod)
            mod = ascend.NormalizeAscendFractalStorage()(mod)
        for factory in pipeline:
            if not auto_schedule and factory == ascend.AutoSchedule:
                mod = use_source_order(mod)
            else:
                mod = factory()(mod)
            name = "tl." + factory.__name__
            if name in requested:
                snapshots[name] = mod
                if snapshots.keys() == requested:
                    return snapshots


def schedule(program, through="InsertSync", *, auto_schedule=True, infer_layouts=False):
    name = "tl." + through
    return schedule_snapshots(program, {name}, auto_schedule=auto_schedule, infer_layouts=infer_layouts)[name]


def statements(ir, ancestors=()):
    """Yield statements with their lexical parents, without inspecting text."""
    if isinstance(ir, tvm.IRModule):
        ir = ir["main"].body
    elif isinstance(ir, tirx.Function):
        ir = ir.body
    yield ir, ancestors
    parents = (*ancestors, ir)
    if isinstance(ir, tvm.ir.SeqStmt):
        children = ir.seq
    elif isinstance(ir, tvm.ir.If):
        children = [ir.then_case] + ([ir.else_case] if ir.else_case is not None else [])
    elif isinstance(ir, (tvm.ir.RegionStmt, tvm.ir.For, tvm.ir.While)):
        children = [ir.body]
    else:
        children = []
    for child in children:
        yield from statements(child, parents)


def allocated_buffers(ir):
    exports = {
        value: variable
        for region in nodes(ir, tvm.ir.RegionStmt)
        for variable, value in zip(region.result_vars, region.attrs.get("tl.schedule_yield", []))
    }
    allocations = []
    for binding in nodes(ir, tvm.ir.Bind):
        if (
            isinstance(binding.value, tvm.ir.Call)
            and isinstance(binding.value.op, tvm.ir.Op)
            and binding.value.op.name == "tirx.alloc_tensor"
        ):
            variable = binding.var
            while variable in exports:
                variable = exports[variable]
            allocations.append(variable)
    return allocations


def allocated_buffer(ir, name):
    return next(variable for variable in allocated_buffers(ir) if variable.name == name)


def storage_root(module, tensor):
    bindings = {binding.var: binding.value for binding in nodes(module, tvm.ir.Bind)}
    for region in nodes(module, tvm.ir.RegionStmt):
        bindings.update(zip(region.result_vars, region.attrs.get("tl.schedule_yield", [])))
    visited = set()
    while True:
        if isinstance(tensor, tvm.ir.Var):
            if tensor not in bindings or tensor in visited:
                return tensor
            visited.add(tensor)
            value = bindings[tensor]
            if isinstance(value, tvm.ir.Call) and value.op.name == "tirx.alloc_tensor":
                return tensor
            tensor = value
        elif isinstance(tensor, tvm.ir.Call) and tensor.op.name in (
            "tirx.decl_tensor",
            "tirx.tensor_data_ptr",
            "tirx.ptr_byte_offset",
            "tirx.reinterpret",
        ):
            tensor = tensor.args[0]
        else:
            raise ValueError(f"Unsupported native storage alias: {tensor}")


def task_core_masks(ir, op_name):
    return {
        int(task.attrs["tl.ascend_task"]["core_mask"])
        for task in nodes(ir, tvm.ir.RegionStmt)
        if "tl.ascend_task" in task.attrs and calls(task.body, op_name)
    }
