"""Structured semantic extraction from TileLang TIR for TileIR lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tvm import tirx

from .tir_analysis import _kernel_launch_stmts


class TileLangSemanticError(RuntimeError):
    """Raised when frontend TIR cannot be interpreted as TileLang semantics."""


@dataclass(frozen=True)
class SemanticBuffer:
    name: str
    shape: tuple[int | str, ...]
    dtype: str
    scope: str


@dataclass(frozen=True)
class SemanticRegion:
    buffer: str
    access: str
    indices: tuple[str, ...]
    shape: tuple[int | str, ...]


@dataclass(frozen=True)
class SemanticStmt:
    kind: str
    attrs: tuple[tuple[str, str], ...] = ()
    children: tuple[SemanticStmt, ...] = ()
    regions: tuple[SemanticRegion, ...] = ()
    # Structured expression payloads used by semantic-to-TileIR lowering.
    # Raw PrimExprs remain typed expression leaves; whole TIR statements do
    # not cross this boundary.
    call_args: tuple[Any, ...] = ()
    call_annotations: tuple[tuple[str, Any], ...] = ()
    binding_var: Any | None = None
    value: Any | None = None
    condition: Any | None = None
    indices: tuple[Any, ...] = ()
    loop_min: Any | None = None
    loop_extent: Any | None = None
    loop_kind: Any | None = None


@dataclass(frozen=True)
class SemanticKernel:
    name: str
    grid: tuple[str, str, str]
    threads: tuple[str, str, str]
    alloc_buffers: tuple[SemanticBuffer, ...]
    body: SemanticStmt
    # Reshape views (T.reshape / T.view): (alias_name, base_name, alias_shape,
    # dtype) — a body Buffer sharing an alloc buffer's data Var under a new
    # shape and, for T.view, possibly a new dtype.  The frontend guarantees
    # equal storage size; sem_to_ir validates it again before constructing the
    # typed alias.
    buffer_aliases: tuple[tuple[str, str, tuple, str], ...] = ()


@dataclass(frozen=True)
class SemanticScalarParam:
    """A non-buffer (scalar) entry parameter of a TileLang PrimFunc.

    These are PrimFunc params that do not appear in ``buffer_map`` (e.g. an
    ``int`` or ``float`` scale factor).  They are distinct from
    ``SemanticBuffer`` params (which have a shape / scope).  The ``name``
    comes from the TIR ``Var.name`` and ``dtype`` from ``Var.dtype``.

    ``position`` is the 0-based index of this param in ``prim_func.params``
    (needed by the pipeline to interleave scalars with buffer params in the
    correct PrimFunc order when building ``entry_args``).
    """

    name: str
    dtype: str
    position: int  # index in prim_func.params


@dataclass(frozen=True)
class SemanticProgram:
    name: str
    params: tuple[SemanticBuffer, ...]
    global_alloc_buffers: tuple[SemanticBuffer, ...]
    kernels: tuple[SemanticKernel, ...]
    # Scalar (non-buffer) entry params.
    scalar_params: tuple[SemanticScalarParam, ...] = ()
    # Buffer and scalar entry parameter names in the original PrimFunc order.
    # Keeping this explicit avoids retaining the entire frontend PrimFunc in
    # the semantic program just to recover ABI ordering during lowering.
    param_order: tuple[str, ...] = ()


def _expr_text(expr: Any) -> str:
    if isinstance(expr, tirx.IntImm):
        return str(int(expr))
    if isinstance(expr, tirx.FloatImm):
        return str(float(expr))
    if isinstance(expr, tirx.StringImm):
        return expr.value
    return str(expr)


def _shape(values: Any) -> tuple[int | str, ...]:
    result = []
    for value in values:
        if isinstance(value, tirx.IntImm):
            result.append(int(value))
        else:
            result.append(str(value))
    return tuple(result)


def _semantic_buffer(buffer: tirx.Buffer) -> SemanticBuffer:
    return SemanticBuffer(
        name=buffer.name,
        shape=_shape(buffer.shape),
        dtype=str(buffer.dtype),
        scope=buffer.scope(),
    )


def _op_name(call: tirx.Call) -> str:
    name = getattr(call.op, "name", str(call.op))
    if name.startswith("tirx."):
        return "tir." + name[len("tirx.") :]
    return name


def _extern_name(call: tirx.Call) -> str | None:
    if not call.args or not isinstance(call.args[0], tirx.StringImm):
        return None
    return call.args[0].value


def _attrs(**kwargs: Any) -> tuple[tuple[str, str], ...]:
    return tuple((key, _expr_text(value)) for key, value in kwargs.items() if value is not None)


def _annotation_attrs(annotations: Any) -> tuple[tuple[str, str], ...]:
    if not annotations:
        return ()
    return tuple((f"annotation.{key}", _expr_text(value)) for key, value in annotations.items())


def _call_payload(call: tirx.Call) -> dict[str, Any]:
    """Extract the expression-level payload needed after semantic extraction."""
    annotations = getattr(call, "annotations", None)
    return {
        "call_args": tuple(call.args),
        "call_annotations": tuple(annotations.items()) if annotations else (),
    }


def _annotation_is_truthy(annotations: Any, key: str) -> bool:
    """Presence-with-truthy-value check for a raw (pre-stringification)
    annotations dict.

    Mirrors the lowering layer's ``_parse_bool``
    (``tilelang/tileir/lowering/sem_to_ir/_base.py``) so the semantic-
    extraction gate here and the lowering-time annotation check
    (``_lower_copy`` in ``tile_ops.py``) agree on what counts as "set": a
    *falsy* value (``"0"``, ``0``, ``False``, ...) is
    treated the same as "not set", not as "set". Not importing
    ``_parse_bool`` directly to avoid a semantic.py -> lowering import
    (lowering already imports from semantic.py).
    """
    if not annotations or key not in annotations:
        return False
    s = _expr_text(annotations[key]).strip()
    if s.isdigit():
        return bool(int(s))
    return s.lower() in ("true", "1", "yes")


def _let_body(stmt: tirx.Stmt) -> tirx.Stmt | None:
    return getattr(stmt, "body", None)


def _semantic_region(expr: tirx.PrimExpr) -> SemanticRegion:
    if not isinstance(expr, tirx.Call) or _op_name(expr) != "tl.region":
        raise TileLangSemanticError(f"Expected TileLang tile region, got `{type(expr).__name__}`.")
    load = expr.args[0]
    if not isinstance(load, tirx.BufferLoad):
        raise TileLangSemanticError("TileLang tile region expects a BufferLoad base.")
    access = _expr_text(expr.args[1]) if len(expr.args) > 1 else ""
    return SemanticRegion(
        buffer=load.buffer.name,
        access=access,
        indices=tuple(_expr_text(index) for index in load.indices),
        shape=_shape(expr.args[2:]),
    )


def _semantic_load_region(expr: tirx.PrimExpr, *, access: str) -> SemanticRegion:
    if isinstance(expr, tirx.Call) and _op_name(expr) == "tl.region":
        region = _semantic_region(expr)
        return SemanticRegion(
            buffer=region.buffer,
            access=access,
            indices=region.indices,
            shape=region.shape,
        )
    if not isinstance(expr, tirx.BufferLoad):
        raise TileLangSemanticError(f"Expected TileLang BufferLoad region, got `{type(expr).__name__}`.")
    shape = []
    for index in expr.indices:
        if isinstance(index, tirx.Ramp):
            shape.append(index.lanes)
        else:
            shape.append(tirx.IntImm("int32", 1))
    return SemanticRegion(
        buffer=expr.buffer.name,
        access=access,
        indices=tuple(_expr_text(index) for index in expr.indices),
        shape=_shape(shape),
    )


def _semantic_region_or_load(expr: tirx.PrimExpr, *, access: str) -> SemanticRegion:
    if isinstance(expr, tirx.Call) and _op_name(expr) == "tl.region":
        return _semantic_region(expr)
    if isinstance(expr, tirx.BufferLoad):
        return _semantic_load_region(expr, access=access)
    raise TileLangSemanticError(f"Expected TileLang tile region or BufferLoad, got `{type(expr).__name__}`.")


def _semantic_tile_op(call: tirx.Call) -> SemanticStmt:
    op = _op_name(call)
    attrs = _attrs(op=op) + _annotation_attrs(getattr(call, "annotations", None))
    if op == "tir.call_extern":
        extern_name = _extern_name(call)
        supported_externs = {
            "debug_print_msg",
            "debug_print_var",
            "debug_print_buffer_value",
            "decode_i4u_to_f16",
            "decode_i2u_to_i8s",
            "decode_fp4_to_bf16_twiddling",
            "DP4A",
        }
        if extern_name in supported_externs:
            return SemanticStmt("tile_op", attrs=attrs + _attrs(extern_name=extern_name), **_call_payload(call))
        # Name the extern in the rejection: CUDA Tile IR has no inline-CUDA /
        # inline-PTX escape hatch, so an arbitrary `T.call_extern` helper
        # (usually a C source registered via `T.import_source`) can never be
        # compiled by this backend.  Well-known pure-bit-twiddling decode
        # helpers are ported to structured tile ops instead (whitelist above).
        raise TileLangSemanticError(
            f"Unsupported extern call `{extern_name}` (tir.call_extern): the TileIR backend "
            f"cannot compile arbitrary CUDA C/PTX extern helpers (CUDA Tile IR has no inline-asm "
            f"escape hatch). Supported extern helpers: {sorted(supported_externs)}. "
            f"Rewrite the computation with structured TileLang tile ops (T.copy / T.Parallel "
            f"elementwise expressions), or use the CUDA backend."
        )
    if op in {"tl.annotate_consumer_reg_alloc", "tl.annotate_producer_reg_dealloc", "tl.no_set_max_nreg"}:
        return SemanticStmt("register_control", attrs=attrs, **_call_payload(call))
    if op == "tl.set_max_nreg":
        return SemanticStmt(
            "register_control",
            attrs=attrs + _attrs(reg_count=call.args[0], is_inc=call.args[1]),
            **_call_payload(call),
        )
    if op in {
        "tl.atomic_add_elem_op",
        # x2/x4 vector widths are a CUDA-instruction detail; the tile-level
        # AtomicRMW subsumes them (same access_ptr dst/val arg layout).
        "tl.atomic_addx2_elem_op",
        "tl.atomic_addx4_elem_op",
        "tl.atomic_add_ret_elem_op",
        "tl.atomic_load_elem_op",
        "tl.atomic_max_elem_op",
        "tl.atomic_min_elem_op",
        "tl.atomic_store_elem_op",
        "tl.tileop.atomicadd",
        "tl.tileop.atomicmax",
        "tl.tileop.atomicmin",
    }:
        return SemanticStmt("atomic_rmw", attrs=attrs, **_call_payload(call))
    if op == "tl.tileop.fill":
        return SemanticStmt(
            "tile_op",
            attrs=attrs + _attrs(value=call.args[1]),
            regions=(_semantic_region(call.args[0]),),
            **_call_payload(call),
        )
    if op == "tl.tileop.tma_copy":
        # The 3rd (index) region extraction below is gather/scatter-only
        # and only ``tl.tileop.copy`` carries those annotations -- tma_copy
        # always has exactly 2 regions (src, dst); its lowering
        # (``_lower_tma_copy``) never reads a 3rd region, so never extract
        # one here even if a caller somehow attached a gather/scatter-shaped
        # annotation to a tma_copy call.
        return SemanticStmt(
            "tile_op",
            attrs=attrs,
            regions=(_semantic_region(call.args[0]), _semantic_region(call.args[1])),
            **_call_payload(call),
        )
    if op == "tl.tileop.copy":
        # T.tma_gather4 / T.tma_scatter4 (is_gather4 / is_scatter4) carry
        # their 4 row indices + column offset in `annotations`, not in a 3rd
        # region -- the plain 2-region extraction below is already correct
        # for them.  `_annotation_attrs` (invoked generically above, in
        # `attrs`) already stringifies every annotation key, including
        # is_gather4/is_scatter4/gather4_rows/gather4_col/barrier/
        # eviction_policy, so no special-casing is needed here; the lowering
        # layer (`_lower_copy` in `tile_ops.py`) keys routing directly off
        # `attrs.get("annotation.is_gather4"/"annotation.is_scatter4")` and
        # reads the untouched PrimExpr values from `stmt.call_annotations`.
        return SemanticStmt(
            "tile_op",
            attrs=attrs,
            regions=(_semantic_region(call.args[0]), _semantic_region(call.args[1])),
            **_call_payload(call),
        )
    if op == "tl.tileop.transpose":
        return SemanticStmt(
            "tile_op",
            attrs=attrs,
            regions=(_semantic_region(call.args[0]), _semantic_region(call.args[1])),
            **_call_payload(call),
        )
    # Dense GEMM and its instruction-specific aliases share the 13-slot
    # frontend ABI. Instruction selection belongs to the TileIR compiler.
    # 2:4 structured-sparse GEMM and its wgmma/tcgen05 aliases have no CUDA
    # Tile IR sparse-MMA counterpart. A dense-GEMM fallback would be
    # numerically wrong, not just slow: gemm_sp's B operand is stored 2:4
    # compressed (half the K extent) with a separate metadata tensor E, so
    # feeding it to a dense mma computes garbage.  Reject with a clear reason.
    if op in {"tl.tileop.gemm_sp", "tl.tileop.wgmma_gemm_sp", "tl.tileop.tcgen05_gemm_sp"}:
        raise TileLangSemanticError(
            f"Unsupported TileLang tile operation `{op}`: 2:4 structured-sparse GEMM has no "
            f"counterpart in CUDA Tile IR, which provides dense MMA operations only. "
            f"A dense fallback is not "
            f"possible because the B operand is 2:4-compressed and paired with metadata E. "
            f"Use the CUDA backend for gemm_sp kernels."
        )
    if op in {
        "tl.tileop.gemm",
        "tl.tileop.wgmma_gemm",
        "tl.tileop.tcgen05_gemm",
        "tl.tileop.gemm_blockscaled",
        "tl.tileop.tcgen05_gemm_blockscaled",
    }:
        regions = (
            _semantic_region(call.args[0]),
            _semantic_region(call.args[1]),
            _semantic_region(call.args[2]),
        )
        annotations = getattr(call, "annotations", None)
        extra_attrs: tuple[tuple[str, str], ...] = ()
        # Blockscaled GEMM appends SFA, SFB and k_start to the 13 dense
        # slots. Preserve the raw k_start expression for lowering validation.
        if annotations and "sf_a_granularity_k" in annotations:
            if len(call.args) <= 15:
                raise TileLangSemanticError(
                    f"Blockscaled `{op}` requires SFA/SFB/k_start at call args 13/14/15, but only {len(call.args)} args are present."
                )
            try:
                sfa_region = _semantic_region(call.args[13])
                sfb_region = _semantic_region(call.args[14])
            except TileLangSemanticError as exc:
                raise TileLangSemanticError(f"Blockscaled `{op}` has malformed SFA/SFB regions at call args 13/14: {exc}") from exc
            regions = regions + (sfa_region, sfb_region)
            extra_attrs = _attrs(k_start=call.args[15])
        return SemanticStmt(
            "tile_op",
            attrs=attrs
            + _attrs(
                transpose_A=call.args[3],
                transpose_B=call.args[4],
                M=call.args[5],
                N=call.args[6],
                K=call.args[7],
                policy=call.args[8],
                clear_accum=call.args[9],
            )
            + extra_attrs,
            regions=regions,
            **_call_payload(call),
        )
    if op == "tl.mbarrier_wait_parity":
        return SemanticStmt(
            "tile_op",
            attrs=attrs + _attrs(parity=call.args[1] if len(call.args) > 1 else None),
            **_call_payload(call),
        )
    if op in {
        "tir.ptx_arrive_barrier",
        "tir.ptx_commit_group",
        "tir.ptx_wait_group",
        "tir.tvm_storage_sync",
        "tir.tvm_thread_allreduce",
        "tir.break_loop",
        "tir.continue_loop",
        # `T.loop_break()` emits `tl.loop_break` directly. This is distinct
        # from `tir.break_loop` above, which comes from a Python `break` in a
        # TIR-scripted `while`. `T.Persistent` uses `tl.loop_break` to stop its
        # serial wave loop once the counter exceeds the padded tile-grid size.
        "tl.loop_break",
        # `T.wait_wgmma(id)` (tilelang/language/builtin.py) pairs with
        # T.wgmma_gemm on Hopper.  In the TileIR backend GEMM issue/completion
        # ordering is handled by TKO tokens, so this wait is a scheduling hint
        # exactly like tir.ptx_wait_group above (lowered to the no-op Barrier).
        "tl.wait_wgmma",
        # TMA stores use the same ordered TKO copy path. Their explicit
        # completion wait is represented by the existing barrier node.
        "tl.tma_store_wait",
    }:
        return SemanticStmt("tile_op", attrs=attrs, **_call_payload(call))
    if op in {"tl.device_assert", "tl.device_assert_with_msg"}:
        return SemanticStmt("tile_op", attrs=attrs, **_call_payload(call))
    if op == "tl.tileop.reduce":
        return SemanticStmt(
            "tile_op",
            attrs=attrs + _attrs(reduce_type=call.args[2], dim=call.args[3], clear=call.args[4]),
            regions=(
                _semantic_load_region(call.args[0], access="r"),
                _semantic_load_region(call.args[1], access="w"),
            ),
            **_call_payload(call),
        )
    if op == "tl.tileop.cumsum":
        return SemanticStmt(
            "tile_op",
            attrs=attrs + _attrs(dim=call.args[2], reverse=call.args[3]),
            regions=(
                _semantic_region_or_load(call.args[0], access="r"),
                _semantic_region_or_load(call.args[1], access="w"),
            ),
            **_call_payload(call),
        )
    if op == "tl.tileop.cummax":
        return SemanticStmt(
            "tile_op",
            attrs=attrs + _attrs(dim=call.args[2], reverse=call.args[3]),
            regions=(
                _semantic_region_or_load(call.args[0], access="r"),
                _semantic_region_or_load(call.args[1], access="w"),
            ),
            **_call_payload(call),
        )
    raise TileLangSemanticError(f"Unsupported TileLang tile operation `{op}`.")


def _semantic_stmt(stmt: tirx.Stmt) -> SemanticStmt:
    if isinstance(stmt, tirx.SeqStmt):
        return SemanticStmt("seq", children=tuple(_semantic_stmt(child) for child in stmt.seq))

    if isinstance(stmt, tirx.SBlockRealize):
        allocs = ",".join(buffer.name for buffer in stmt.block.alloc_buffers)
        return SemanticStmt(
            "block",
            attrs=_attrs(name=stmt.block.name_hint, alloc_buffers=allocs),
            children=(_semantic_stmt(stmt.block.body),),
        )

    if isinstance(stmt, tirx.AttrStmt):
        if stmt.attr_key == "thread_extent":
            tag = str(getattr(stmt.node, "thread_tag", ""))
            var = getattr(getattr(stmt.node, "var", None), "name", "")
            return SemanticStmt(
                "thread_extent",
                attrs=_attrs(tag=tag, var=var, extent=stmt.value),
                children=(_semantic_stmt(stmt.body),),
                binding_var=getattr(stmt.node, "var", None),
                value=stmt.value,
            )
        if stmt.attr_key == "threadblock_swizzle_pattern":
            return SemanticStmt(
                "threadblock_swizzle_pattern",
                attrs=_attrs(value=stmt.value),
                children=(_semantic_stmt(stmt.body),),
                value=stmt.value,
            )
        if stmt.attr_key == "reduce_scope":
            return SemanticStmt(
                "reduce_scope",
                attrs=_attrs(value=stmt.value),
                children=(_semantic_stmt(stmt.body),),
                value=stmt.value,
            )
        if stmt.attr_key in {"tl.ws_pipeline_depth", "tl.ws_op_id"}:
            # These annotate the sequential source program consumed by CUDA's
            # WS scheduler; they do not introduce barriers or concurrent roles.
            return SemanticStmt(
                "ws_schedule_hint",
                attrs=_attrs(key=stmt.attr_key, value=stmt.value),
                children=(_semantic_stmt(stmt.body),),
                value=stmt.value,
            )
        if stmt.attr_key == "warp_specialize":
            # warp_specialize marks warp-group sections in a producer/consumer kernel.
            # The cuda_tile dialect owns warp specialization scheduling downstream;
            # at the TileIR semantic level we treat this as a transparent pass-through
            # and lower the body normally.  The integer value is the warp-group id
            # (0 = consumer/compute, 1 = producer/TMA) — accept both; cuTile uses
            # the surrounding if(tx) guards to route them.
            return SemanticStmt(
                "warp_specialize",
                attrs=_attrs(value=stmt.value),
                children=(_semantic_stmt(stmt.body),),
                value=stmt.value,
            )
        raise TileLangSemanticError(f"Unsupported TileLang AttrStmt `{stmt.attr_key}`.")

    if isinstance(stmt, tirx.Bind):
        body = _let_body(stmt)
        return SemanticStmt(
            "let",
            attrs=_attrs(var=stmt.var.name, value=stmt.value),
            children=(_semantic_stmt(body),) if body is not None else (),
            binding_var=stmt.var,
            value=stmt.value,
        )

    if isinstance(stmt, tirx.For):
        kind = {
            tirx.ForKind.SERIAL: "serial",
            tirx.ForKind.PARALLEL: "parallel",
            tirx.ForKind.VECTORIZED: "vectorized",
            tirx.ForKind.UNROLLED: "unrolled",
            tirx.ForKind.THREAD_BINDING: "thread_binding",
        }.get(stmt.kind, str(stmt.kind))
        annotations = getattr(stmt, "annotations", None)
        if annotations and any(key in annotations for key in ("num_stages", "tl_pipeline_order", "tl_pipeline_stage", "tl_pipeline_group")):
            kind = "pipelined"
        return SemanticStmt(
            "for",
            attrs=_attrs(kind=kind, var=stmt.loop_var.name, min=stmt.min, extent=stmt.extent) + _annotation_attrs(annotations),
            children=(_semantic_stmt(stmt.body),),
            binding_var=stmt.loop_var,
            loop_min=stmt.min,
            loop_extent=stmt.extent,
            loop_kind=stmt.kind,
        )

    if isinstance(stmt, tirx.While):
        return SemanticStmt(
            "while",
            attrs=_attrs(condition=stmt.condition),
            children=(_semantic_stmt(stmt.body),),
            condition=stmt.condition,
        )

    if isinstance(stmt, tirx.IfThenElse):
        children = [_semantic_stmt(stmt.then_case)]
        if stmt.else_case is not None:
            children.append(_semantic_stmt(stmt.else_case))
        return SemanticStmt(
            "if",
            attrs=_attrs(condition=stmt.condition),
            children=tuple(children),
            condition=stmt.condition,
        )

    if isinstance(stmt, tirx.BufferStore):
        return SemanticStmt(
            "buffer_store",
            attrs=_attrs(buffer=stmt.buffer.name, value=stmt.value),
            value=stmt.value,
            indices=tuple(stmt.indices),
        )

    if isinstance(stmt, tirx.Evaluate):
        value = stmt.value
        if isinstance(value, tirx.Call):
            return _semantic_tile_op(value)
        raise TileLangSemanticError(f"Unsupported TileLang evaluate node `{type(value).__name__}`.")

    raise TileLangSemanticError(f"Unsupported TileLang semantic statement `{type(stmt).__name__}`.")


def _collect_launch(stmt: tirx.Stmt) -> tuple[dict[str, str], dict[str, str]]:
    grid = {"blockIdx.x": "1", "blockIdx.y": "1", "blockIdx.z": "1"}
    threads = {"threadIdx.x": "1", "threadIdx.y": "1", "threadIdx.z": "1"}

    def visit(node):
        if not isinstance(node, tirx.AttrStmt) or node.attr_key != "thread_extent":
            return
        tag = str(getattr(node.node, "thread_tag", ""))
        if tag in grid:
            grid[tag] = _expr_text(node.value)
        elif tag in threads:
            threads[tag] = _expr_text(node.value)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return grid, threads


def _collect_buffer_aliases(stmt: tirx.Stmt, alloc_buffers: tuple[SemanticBuffer, ...]) -> tuple[tuple[str, str, tuple, str], ...]:
    """Collect reshape views: body Buffers sharing an alloc buffer's data Var.

    ``T.reshape(src, shape)`` / ``T.view(src, ...)`` build a NEW tirx.Buffer
    over ``src.data``; region extraction only sees the alias NAME, so the
    lowering needs the (alias → base, shape, dtype) relation to keep a single
    tile as the source of truth.  Same-dtype aliases lower as reshapes;
    dtype-changing ``T.view`` aliases lower as storage-preserving
    reinterprets.
    """
    del alloc_buffers  # base names come from the raw SBlock alloc buffers below

    data_to_base: dict = {}

    def collect_allocs(node):
        if isinstance(node, tirx.SBlock):
            for buffer in node.alloc_buffers:
                data_to_base[buffer.data] = buffer

    tirx.stmt_functor.post_order_visit(stmt, collect_allocs)

    aliases: dict[str, tuple[str, str, tuple, str]] = {}

    def record_alias(buf: tirx.Buffer) -> None:
        base = data_to_base.get(buf.data)
        if base is None or buf.name == base.name or buf.name in aliases:
            return
        shape = tuple(int(d) if isinstance(d, tirx.IntImm) else _expr_text(d) for d in buf.shape)
        aliases[buf.name] = (buf.name, base.name, shape, str(buf.dtype))

    def visit(node):
        if isinstance(node, (tirx.BufferLoad, tirx.BufferStore, tirx.BufferRegion)):
            record_alias(node.buffer)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return tuple(aliases.values())


def _block_alloc_buffers(stmt: tirx.Stmt) -> tuple[SemanticBuffer, ...]:
    allocs: list[SemanticBuffer] = []

    def collect_allocs(node):
        if isinstance(node, tirx.SBlock):
            allocs.extend(_semantic_buffer(buffer) for buffer in node.alloc_buffers)

    tirx.stmt_functor.post_order_visit(stmt, collect_allocs)
    return tuple(allocs)


def materialize_launch_nest(prim_func: tirx.PrimFunc) -> tirx.PrimFunc:
    """Lower the target-neutral ``T.Kernel`` launch nest into the SIMT
    ``thread_extent`` form the TileIR backend relies on.

    Since d2388370 ``T.Kernel`` traces the launch as a ``ForKind.THREAD_BINDING``
    nest and each backend runs ``tl.MaterializeKernelLaunch`` to lower it (see
    ``tilelang/cuda/pipeline.py``). Callers that hand the frontend PrimFunc
    straight to TileIR readers (semantic extraction, grid-sync splitting, launch
    metadata) would otherwise see ``thread_binding`` loops instead of
    ``thread_extent`` attrs and lose the launch grid. The pass is idempotent, so
    this is a no-op once the launch has already been materialized.
    """
    from tvm import IRModule
    from tilelang.transform import MaterializeKernelLaunch

    global_symbol = "main"
    if prim_func.attrs and "global_symbol" in prim_func.attrs:
        global_symbol = str(prim_func.attrs["global_symbol"])
    mod = MaterializeKernelLaunch()(IRModule({global_symbol: prim_func}))
    return next(func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc))


def extract_semantic_program(prim_func: tirx.PrimFunc) -> SemanticProgram:
    prim_func = materialize_launch_nest(prim_func)
    name = str(prim_func.attrs.get("global_symbol", "main")) if prim_func.attrs else "main"
    params = tuple(_semantic_buffer(prim_func.buffer_map[param]) for param in prim_func.params if param in prim_func.buffer_map)
    # Extract scalar (non-buffer) entry params with their position in
    # prim_func.params so the pipeline can reconstruct the correct argument order.
    scalar_params = tuple(
        SemanticScalarParam(name=param.name, dtype=str(param.dtype), position=pos)
        for pos, param in enumerate(prim_func.params)
        if param not in prim_func.buffer_map
    )
    param_order = tuple(prim_func.buffer_map[param].name if param in prim_func.buffer_map else param.name for param in prim_func.params)
    global_allocs: tuple[SemanticBuffer, ...] = ()
    if isinstance(prim_func.body, tirx.SBlockRealize):
        global_allocs = tuple(_semantic_buffer(buffer) for buffer in prim_func.body.block.alloc_buffers if buffer.scope() == "global")

    kernel_stmts = _kernel_launch_stmts(prim_func)
    kernels = []
    for index, stmt in enumerate(kernel_stmts):
        grid, threads = _collect_launch(stmt)
        alloc_buffers = _block_alloc_buffers(stmt)
        kernels.append(
            SemanticKernel(
                name=f"{name}_{index}" if len(kernel_stmts) > 1 else name,
                grid=(grid["blockIdx.x"], grid["blockIdx.y"], grid["blockIdx.z"]),
                threads=(threads["threadIdx.x"], threads["threadIdx.y"], threads["threadIdx.z"]),
                alloc_buffers=alloc_buffers,
                body=_semantic_stmt(stmt),
                buffer_aliases=_collect_buffer_aliases(stmt, alloc_buffers),
            )
        )

    return SemanticProgram(
        name=name,
        params=params,
        global_alloc_buffers=global_allocs,
        kernels=tuple(kernels),
        scalar_params=scalar_params,
        param_order=param_order,
    )
