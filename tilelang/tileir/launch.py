"""Launch metadata extraction for TileIR backend dispatch."""

from __future__ import annotations

from typing import Any

from tvm import tirx
from .artifact import TileIRLaunchMetadata, TileIRTemporaryBuffer
from .errors import TileIRLoweringError
from .semantic import SemanticBuffer
from .tir_analysis import (
    _as_launch_extent,
    _as_static_int,
    _kernel_launch_stmts,
    _op_name,
)


def extract_launch_metadata(prim_func: tirx.PrimFunc) -> TileIRLaunchMetadata:
    """Extract TileIR launch metadata from a TileLang kernel PrimFunc.

    CUDA Tile kernels use the CUDA launch grid as the TileIR tile-block grid.
    The CUDA thread block dimensions are unused by TileIR and must remain
    ``(1, 1, 1)``; logical TileLang thread extents are lowered inside TileIR.
    """

    grid = [1, 1, 1]
    dynamic_smem_bytes = 0
    attrs = prim_func.attrs or {}

    def record_thread_extent(tag: str, extent: Any) -> None:
        if tag.startswith("blockIdx."):
            grid["xyz".index(tag[-1])] = _as_launch_extent(extent, field=f"thread_extent[{tag}]", minimum=1)
        elif tag.startswith("threadIdx."):
            _as_static_int(extent, field=f"thread_extent[{tag}]", minimum=1)

    if "thread_extent" in attrs:
        for tag, extent in attrs["thread_extent"].items():
            record_thread_extent(str(tag), extent)
    else:
        seen_extents: dict[str, str] = {}

        def visit(node):
            if not isinstance(node, tirx.AttrStmt) or node.attr_key != "thread_extent":
                return
            tag = str(getattr(node.node, "thread_tag", ""))
            if not tag:
                return
            if tag in seen_extents:
                # A repeated tag with the SAME extent is a nested SIMT-style
                # thread binding inside the one kernel (e.g. an explicit
                # ``T.thread_binding(threadIdx.x)`` under T.Kernel) — not a
                # second kernel launch. Different extents (or a repeated
                # blockIdx) mean a genuine second launch nest.
                if str(node.value) == seen_extents[tag] and tag.startswith("threadIdx."):
                    return
                raise TileIRLoweringError(
                    "TileIR backend requires exactly one TileLang kernel launch per PrimFunc before "
                    f"multi-kernel host orchestration is implemented; duplicate thread extent `{tag}`."
                )
            seen_extents[tag] = str(node.value)
            record_thread_extent(tag, node.value)

        tirx.stmt_functor.post_order_visit(prim_func.body, visit)

        if not seen_extents:
            # MaterializeKernelLaunch has not run yet (e.g. callers that read the
            # frontend PrimFunc directly): T.Kernel still encodes the launch nest
            # as ForKind.THREAD_BINDING loops rather than thread_extent AttrStmts.
            # Read the grid extents straight from the kThreadBinding loop nest.
            def visit_binding(node):
                if not isinstance(node, tirx.For) or node.kind != tirx.ForKind.THREAD_BINDING or node.thread_binding is None:
                    return
                tag = str(node.thread_binding.thread_tag)
                if not tag or tag in seen_extents:
                    return
                seen_extents[tag] = str(node.extent)
                record_thread_extent(tag, node.extent)

            tirx.stmt_functor.post_order_visit(prim_func.body, visit_binding)

    if "dyn_shared_memory_buf" in attrs:
        dynamic_smem_bytes = _as_static_int(attrs["dyn_shared_memory_buf"], field="dyn_shared_memory_buf", minimum=0)

    return TileIRLaunchMetadata(grid=tuple(grid), block=(1, 1, 1), dynamic_smem_bytes=dynamic_smem_bytes)


def _buffer_argument_names(prim_func: tirx.PrimFunc) -> tuple[str, ...]:
    names = []
    for param in prim_func.params:
        buffer = prim_func.buffer_map.get(param)
        if buffer is not None:
            names.append(buffer.name)
    return tuple(names)


def _argument_names(prim_func: tirx.PrimFunc) -> tuple[str, ...]:
    names = []
    for param in prim_func.params:
        buffer = prim_func.buffer_map.get(param)
        names.append(buffer.name if buffer is not None else param.name)
    return tuple(names)


def _global_temporary_buffers(prim_func: tirx.PrimFunc) -> tuple[tirx.Buffer, ...]:
    if not isinstance(prim_func.body, tirx.SBlockRealize):
        return ()
    root = prim_func.body.block
    return tuple(buffer for buffer in root.alloc_buffers if buffer.scope() == "global")


def _temporary_buffer_metadata_from_semantics(buffers: tuple[SemanticBuffer, ...]) -> tuple[TileIRTemporaryBuffer, ...]:
    metadata = []
    for buffer in buffers:
        shape = []
        for dim in buffer.shape:
            if not isinstance(dim, int):
                raise TileIRLoweringError(f"TileIR global temporary `{buffer.name}` requires static shape; got {buffer.shape}.")
            shape.append(dim)
        metadata.append(TileIRTemporaryBuffer(name=buffer.name, shape=tuple(shape), dtype=buffer.dtype))
    return tuple(metadata)


def _used_buffer_data_vars(stmt: tirx.Stmt) -> set[tirx.Var]:
    used: set[tirx.Var] = set()

    def visit(node):
        if isinstance(node, (tirx.BufferLoad, tirx.BufferStore)):
            used.add(node.buffer.data)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return used


def _used_vars(stmt: tirx.Stmt) -> set[tirx.Var]:
    used: set[tirx.Var] = set()

    def visit(node):
        if isinstance(node, tirx.Var):
            used.add(node)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return used


def _split_host_orchestrated_primfunc(prim_func: tirx.PrimFunc) -> tuple[tirx.PrimFunc, ...]:
    if not isinstance(prim_func.body, tirx.SBlockRealize):
        raise TileIRLoweringError("TileIR multi-kernel lowering expects a root BlockRealize.")
    root = prim_func.body.block
    if not isinstance(root.body, tirx.SeqStmt):
        raise TileIRLoweringError("TileIR multi-kernel lowering expects root body to be a SeqStmt of TileLang kernels.")

    temporary_buffers = _global_temporary_buffers(prim_func)
    temporary_params = [buffer.data for buffer in temporary_buffers]
    temporary_buffer_map = {buffer.data: buffer for buffer in temporary_buffers}
    base_params = list(prim_func.params)
    base_buffer_map = dict(prim_func.buffer_map)
    kernel_name = str(prim_func.attrs.get("global_symbol", "main"))
    kernels = []

    for index, stmt in enumerate(_kernel_launch_stmts(prim_func)):
        used_buffer_vars = _used_buffer_data_vars(stmt)
        used_vars = _used_vars(stmt)
        sub_params = [
            param
            for param in base_params
            if (
                (param in base_buffer_map and base_buffer_map[param].data in used_buffer_vars)
                or (param not in base_buffer_map and param in used_vars)
            )
        ]
        sub_params.extend(param for param in temporary_params if param in used_buffer_vars)
        sub_buffer_map = {param: buffer for param, buffer in {**base_buffer_map, **temporary_buffer_map}.items() if param in sub_params}
        sub_func = tirx.PrimFunc(
            sub_params,
            stmt,
            ret_type=prim_func.ret_type,
            buffer_map=sub_buffer_map,
        ).with_attr("global_symbol", f"{kernel_name}_{index}")
        kernels.append(sub_func)

    if len(kernels) < 2:
        raise TileIRLoweringError("TileIR multi-kernel lowering requires at least two TileLang kernel launches.")
    return tuple(kernels)


def _is_grid_sync_stmt(stmt: tirx.Stmt) -> bool:
    if not isinstance(stmt, tirx.Evaluate):
        return False
    value = stmt.value
    return isinstance(value, tirx.Call) and _op_name(value) == "tl.sync_grid"


def _seq_from_phase(stmts: list[tirx.Stmt], span=None) -> tirx.Stmt:
    if not stmts:
        return tirx.Evaluate(0)
    if len(stmts) == 1:
        return stmts[0]
    return tirx.SeqStmt(stmts, span=span)


def _clone_block_with_body(block: tirx.SBlock, body: tirx.Stmt) -> tirx.SBlock:
    return tirx.SBlock(
        block.iter_vars,
        block.reads,
        block.writes,
        block.name_hint,
        body,
        init=block.init,
        alloc_buffers=block.alloc_buffers,
        match_buffers=block.match_buffers,
        annotations=block.annotations,
        span=getattr(block, "span", None),
    )


def _split_stmt_on_grid_sync(stmt: tirx.Stmt) -> list[tirx.Stmt] | None:
    if isinstance(stmt, tirx.SeqStmt):
        phases: list[list[tirx.Stmt]] = [[]]
        saw_sync = False
        for child in stmt.seq:
            if _is_grid_sync_stmt(child):
                saw_sync = True
                phases.append([])
                continue
            child_phases = _split_stmt_on_grid_sync(child)
            if child_phases is None:
                phases[-1].append(child)
                continue
            saw_sync = True
            phases[-1].append(child_phases[0])
            for phase in child_phases[1:]:
                phases.append([phase])
        if not saw_sync:
            return None
        return [_seq_from_phase(phase, span=getattr(stmt, "span", None)) for phase in phases if phase]

    if isinstance(stmt, tirx.AttrStmt):
        split_body = _split_stmt_on_grid_sync(stmt.body)
        if split_body is None:
            return None
        return [tirx.AttrStmt(stmt.node, stmt.attr_key, stmt.value, body, span=getattr(stmt, "span", None)) for body in split_body]

    if isinstance(stmt, tirx.SBlockRealize):
        split_body = _split_stmt_on_grid_sync(stmt.block.body)
        if split_body is None:
            return None
        return [
            tirx.SBlockRealize(
                stmt.iter_values,
                stmt.predicate,
                _clone_block_with_body(stmt.block, body),
                span=getattr(stmt, "span", None),
            )
            for body in split_body
        ]

    return None


def _split_grid_sync_primfunc(prim_func: tirx.PrimFunc) -> tirx.PrimFunc:
    if not isinstance(prim_func.body, tirx.SBlockRealize):
        return prim_func
    split_body = _split_stmt_on_grid_sync(prim_func.body.block.body)
    if split_body is None or len(split_body) < 2:
        return prim_func

    root = prim_func.body.block
    new_root = _clone_block_with_body(root, tirx.SeqStmt(split_body, span=getattr(root.body, "span", None)))
    new_body = tirx.SBlockRealize(
        prim_func.body.iter_values,
        prim_func.body.predicate,
        new_root,
        span=getattr(prim_func.body, "span", None),
    )
    return tirx.PrimFunc(
        prim_func.params,
        new_body,
        ret_type=prim_func.ret_type,
        buffer_map=prim_func.buffer_map,
        attrs=prim_func.attrs,
        span=getattr(prim_func, "span", None),
    )
