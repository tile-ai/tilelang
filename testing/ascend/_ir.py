"""Small IR builders and inspection helpers for host-only Ascend pass tests.

The scheduling prefix ends at the requested pass. It deliberately excludes
device lowering, code generation, and compilation. Tests of a pass with an
already prepared input should call that pass directly instead.
"""

import tilelang
from tilelang import tvm
from tilelang.ascend import transform as ascend
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform, post_order_visit


def seq(*stmts):
    return stmts[0] if len(stmts) == 1 else tirx.SeqStmt(stmts)


def region(buffer, mask, indices=None, shape=None):
    return tirx.Call(
        "handle", tvm.ir.Op.get("tl.region"), [buffer[tuple(indices or [0] * len(buffer.shape))], mask, *(shape or buffer.shape)]
    )


def copy(src, dst, *, src_indices=None, dst_indices=None, src_shape=None, dst_shape=None, annotations=None):
    return tirx.Evaluate(
        tirx.Call(
            "handle",
            tvm.ir.Op.get("tl.tileop.ascend_copy"),
            [region(src, 1, src_indices, src_shape), region(dst, 2, dst_indices, dst_shape)],
            annotations=annotations,
        )
    )


def kernel(body, *, buffers=(), params=(), annotations=None):
    root = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=buffers, annotations=annotations or {})
    buffer_params = [param for param in params if isinstance(param, tirx.Buffer)]
    func = tirx.PrimFunc(
        [param.data if isinstance(param, tirx.Buffer) else param for param in params],
        tirx.SBlockRealize([], True, root),
        buffer_map={buffer.data: buffer for buffer in buffer_params},
    ).with_attr({"global_symbol": "main", "target": tvm.target.Target("ascend")})
    return tvm.IRModule({"main": func})


def nodes(ir, kind):
    if isinstance(ir, tvm.IRModule):
        ir = ir["main"].body
    elif isinstance(ir, tirx.PrimFunc):
        ir = ir.body
    found = []
    post_order_visit(ir, lambda node: found.append(node) if isinstance(node, kind) else None)
    return found


def calls(ir, name):
    return [node for node in nodes(ir, tirx.Call) if isinstance(node.op, tvm.ir.Op) and node.op.name == name]


def schedule_snapshots(program, pass_names, *, auto_schedule=True):
    """Run the scheduling prerequisites and stop at the last requested boundary.

    This helper accepts frontend programs, including explicit unrolls and dual
    copies. It does not run layout inference: tests needing physical layouts
    must prepare those themselves, or belong in the lowering integration suite.
    """
    pipeline = (
        ascend.NormalizeControlFlowForSchedule,
        ascend.NormalizeConflictHints,
        ascend.MaterializeScheduleUnits,
        ascend.AnnotateMultiBufferEligible,
        ascend.EstimateLatency,
        ascend.AutoSchedule,
        ascend.AssignCore,
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
            if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit" and int(node.node["stage"]) < 0:
                return tirx.AttrStmt({"stage": 0}, node.attr_key, node.value, node.body)

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


def schedule(program, through="InsertSync", *, auto_schedule=True):
    name = "tl." + through
    return schedule_snapshots(program, {name}, auto_schedule=auto_schedule)[name]


def statements(ir, ancestors=()):
    """Yield statements with their lexical parents, without inspecting text."""
    if isinstance(ir, tvm.IRModule):
        ir = ir["main"].body
    elif isinstance(ir, tirx.PrimFunc):
        ir = ir.body
    yield ir, ancestors
    parents = (*ancestors, ir)
    if isinstance(ir, tirx.SeqStmt):
        children = ir.seq
    elif isinstance(ir, tirx.SBlockRealize):
        children = [ir.block]
    elif isinstance(ir, tirx.IfThenElse):
        children = [ir.then_case] + ([ir.else_case] if ir.else_case is not None else [])
    elif isinstance(ir, (tirx.AttrStmt, tirx.For, tirx.While, tirx.SBlock)):
        children = [ir.body]
    else:
        children = []
    for child in children:
        yield from statements(child, parents)


def allocated_buffer(ir, name):
    return next(buffer for block in nodes(ir, tirx.SBlock) for buffer in block.alloc_buffers if buffer.name == name)


def task_core_masks(ir, op_name):
    return {
        int(task.node["core_mask"]) for task in nodes(ir, tirx.AttrStmt) if task.attr_key == "tl.ascend_task" and calls(task.body, op_name)
    }
