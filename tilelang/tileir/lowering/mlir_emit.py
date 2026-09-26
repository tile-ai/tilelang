"""Emit typed TileIR blocks as CUDA Tile IR MLIR."""

from __future__ import annotations

import dataclasses
from typing import Any

from tilelang.tileir.ir.types import TileType, MemSpace
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import TileOp
from tilelang.tileir.emission_utils import (
    _BufferInfo,
    _as_tile,
    _buffer_arg_types,
    _materialize_buffer,
    _mlir_element_type,
    _mlir_tile_type,
    _reshape_tile_to,
    _static_contiguous_strides,
)
from tilelang.tileir.errors import TileIRLoweringError, TileIRLoweringNotImplementedError
from tilelang.tileir.scratch import scratch_layout


__all__ = ["EmitContext", "emit_module"]


def _referenced_values(block: Block) -> set[Value]:
    """Collect values used by ops, including index attributes and nested bodies."""
    values = set()

    def visit(obj):
        if isinstance(obj, Value):
            values.add(obj)
        elif isinstance(obj, Block):
            for op in obj.ops:
                for field in dataclasses.fields(op):
                    visit(getattr(op, field.name))
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                visit(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)

    visit(block)
    return values


# EmitContext


class EmitContext:
    """Bound MLIR handles + Value→mlir.Value mapping for one emission session.

    Attributes
    ----------
    ct : module
        CUDA Tile IR builder module.
    ct_gen : module
        Generated CUDA Tile IR operation builders.
    ir : module
        MLIR IR module.
    loc : ir.Location
        The active MLIR source location (``ir.Location.unknown()`` for
        synthesised code).
    value_map : dict[Value, mlir_value]
        Maps TileIR ``Value`` objects to the MLIR ``BlockArgument`` or
        op result they were lowered to.
    _buffer_map : dict[Value, _BufferInfo]
        Maps GLOBAL buffer entry-param ``Value`` objects to their
        materialised ``_BufferInfo`` (ptr, view, shape/stride tiles).
        Populated by the entry-arg flattening in ``emit_module``.
    _token_map : dict[Value, Any]
        Maps buffer ``Value`` to the latest MLIR token for that buffer.
        Updated by Load/Store/Copy emitters.
        When token_plan is absent, this conservative per-buffer chain is the
        fallback path.
    token_plan : TokenPlan | None
        When present (wired from pipeline.py via ``ctx.results["token_order"]``),
        ``_ensure_token`` consults this plan to compute precise RAW/WAW deps
        instead of the conservative per-buffer chain.  ``None`` disables the
        plan-based path.
    current_op : TileOp | None
        The TileIR op currently being emitted.  Set by the op-walk loop in
        ``emit_module`` and ``_walk_block`` immediately before calling
        ``op.emit_mlir(ctx)``.  Required by ``_ensure_token`` to look up
        the op's deps in ``token_plan``.
    _op_token : dict[int, Any]
        Maps ``id(op)`` → the MLIR out-token produced by that memory op.
        Populated by ``_ensure_token`` (via ``_record_op_token``) after each
        TKO load/store emits its result token.  Used by ``_ensure_token`` to
        join dep-ops' out-tokens when computing the input token for the
        current op.
    """

    def __init__(self, ct: Any, ct_gen: Any, ir: Any, loc: Any) -> None:
        self.ct = ct
        self.ct_gen = ct_gen
        self.ir = ir
        self.loc = loc
        self.value_map: dict[Value, Any] = {}
        self._buffer_map: dict[Value, _BufferInfo] = {}
        self._token_map: dict[Value, Any] = {}
        # Tile map for SHARED/REGISTER (non-GLOBAL) buffers.
        # Keyed by the same Value objects that ops reference via LoweringScope.
        # Values are MLIR ct.Tile objects (zero-initialized on entry).
        # Updated by Copy.emit_mlir when storing into a SHARED buffer.
        self._tile_map: dict[Value, Any] = {}
        # A single root token shared by all non-buffer ops; see token_for().
        self._root_token: Any = None
        # token_plan wiring fields.
        self.token_plan: Any = None  # TokenPlan | None
        # Reshape-view aliases: alias buffer Value → base buffer Value
        # (from block.buffer_aliases; wired by emit_module).
        self.buffer_aliases: dict[Value, Value] = {}
        # SIMT-demoted scratch buffers (per-tile-block workspace): their atomics use
        # tile-block scope instead of device scope.
        self.alloca_values: set[Value] = set()
        self.current_op: Any = None  # TileOp | None — set by op-walk loop
        self._op_token: dict[int, Any] = {}  # id(op) → MLIR out-token
        # arch string for optimization hints (latency/allow_tma).
        self.arch: str | None = None  # e.g. "sm_90a"
        # Global TL_DISABLE_TMA_LOWER flag.  When True it forces
        # allow_tma=False on every copy's load/store optimization hint
        # (OR-ed with the per-copy T.copy(..., disable_tma=True) annotation),
        # mirroring how the CUDA backend honors the global pass config.
        self.disable_tma: bool = False
        # TL_ENABLE_FAST_MATH: when True, emit the approximate-reciprocal div
        # (ct.div approx + flush_to_zero) and flush_to_zero transcendentals,
        # matching what cuTile emits — faster, slightly lower precision.
        self.fast_math: bool = False

    # value_map helpers

    def lookup(self, v: Value) -> Any:
        """Return the MLIR value bound to *v*.

        Raises
        ------
        KeyError
            If *v* has not been bound yet.
        """
        try:
            return self.value_map[v]
        except KeyError:
            raise KeyError(f"Value id={v.id} (name={v.name!r}) is not in the EmitContext value_map") from None

    def bind(self, v: Value, mlir_v: Any) -> None:
        """Associate TileIR ``Value`` *v* with MLIR value *mlir_v*.

        Overwrites a previous binding if one exists (e.g. during re-emission).
        """
        self.value_map[v] = mlir_v

    # Token seam

    def token_for(self, op: TileOp) -> Any:
        """Return the MLIR token that must precede *op*'s execution.

        Returns the single ``_root_token`` for every op.  The seam is
        intentionally clean — callers need only call ``ctx.token_for(op)``
        and pass the result to the op's emitter; the implementation detail
        is hidden here.
        """
        return self._root_token

    def _get_token(self, buf_val: Value) -> Any:
        """Return the current MLIR token for *buf_val*, or a fresh token if none.

        Per-buffer token seam.  After each load/store the emitter calls
        ``_set_token`` to record the result token so the next op on the same
        buffer chains correctly.
        """
        return self._token_map.get(buf_val)

    def _set_token(self, buf_val: Value, token: Any) -> None:
        """Record *token* as the current token for *buf_val*.

        Called by Load/Store/Copy emitters after they emit their TKO op.
        Also records the token on the current_op in _op_token so that
        dependent ops can join this token via the token_plan.
        """
        self._token_map[buf_val] = token
        # When token_plan is active, record the out-token for the current op
        # so downstream ops can join it.
        if self.current_op is not None:
            op_key = id(self.current_op)
            # Use the LAST token recorded for this op (overwrite is fine —
            # for multi-buffer ops like Copy the last _set_token call wins,
            # which is conservative and correct since the op is not done until
            # its final token is produced).
            self._op_token[op_key] = token

    # Buffer helpers

    def get_buffer_info(self, buf_val: Value) -> _BufferInfo:
        """Return the ``_BufferInfo`` for a GLOBAL buffer entry param.

        Raises
        ------
        KeyError
            If *buf_val* is not a known GLOBAL buffer entry param.
        """
        try:
            return self._buffer_map[buf_val]
        except KeyError:
            raise KeyError(f"Value id={buf_val.id} (name={buf_val.name!r}) is not a GLOBAL buffer param in _buffer_map") from None

    # Tile map helpers — SHARED / REGISTER buffers

    def is_tile_buffer(self, buf_val: Value) -> bool:
        """Return True if *buf_val* is a SHARED/REGISTER tile-based buffer.

        These buffers are materialised as zero-constant tiles before the op
        walk rather than as MLIR block arguments (they are kernel-local
        alloc_shared / alloc_fragment allocations, not entry params).
        Reshape-view aliases resolve through their base buffer.
        """
        base = self.buffer_aliases.get(buf_val)
        return (base if base is not None else buf_val) in self._tile_map

    def get_tile(self, buf_val: Value) -> Any:
        """Return the current MLIR tile value for a SHARED/REGISTER buffer.

        The tile is a zero-constant ct.Tile on entry; Copy/Store ops update
        it via ``set_tile`` to simulate shared-memory writes.  A reshape-view
        alias (T.reshape / T.view) reads the BASE buffer's tile reshaped or
        storage-bitcast to the alias type — one tile is the single source of
        truth.

        Raises
        ------
        KeyError
            If *buf_val* is not a known SHARED/REGISTER buffer.
        """
        base = self.buffer_aliases.get(buf_val)
        if base is not None:
            base_tile = self.get_tile(base)
            return self._reinterpret_buffer_tile(base_tile, base.type, buf_val.type)
        try:
            return self._tile_map[buf_val]
        except KeyError:
            raise KeyError(f"Value id={buf_val.id} (name={buf_val.name!r}) is not a SHARED/REGISTER buffer in _tile_map") from None

    def _reinterpret_buffer_tile(self, tile: Any, source_type: TileType, target_type: TileType) -> Any:
        """Reinterpret one equal-capacity buffer tile without changing bits.

        CUDA Tile IR's elementwise ``bitcast`` requires equal element widths.
        ``T.view`` also permits a width change (for example bf16[64] to
        fp32[32]), so that case goes through the standard rank-1 ``pack`` /
        ``unpack`` byte representation.  Both paths remain ordinary CUDA Tile
        IR and introduce no Native SIMT provider boundary. Semantic lowering
        verifies logical capacity and that padding is confined to the tail;
        these types describe padded physical storage, not logical view extents.
        """

        source = _as_tile(self, tile)
        target_shape = list(target_type.shape)
        if source_type.dtype is target_type.dtype:
            return _reshape_tile_to(self.ct, source, target_shape, self.loc)

        source_elements = 1
        for extent in source_type.shape:
            source_elements *= int(extent)
        target_elements = 1
        for extent in target_type.shape:
            target_elements *= int(extent)
        source_bits = source_elements * int(source_type.dtype.bitwidth)
        target_bits = target_elements * int(target_type.dtype.bitwidth)
        if source_bits != target_bits:
            raise TileIRLoweringError(
                f"buffer reinterpret has incompatible physical storage capacity: source has {source_bits} bits, target has {target_bits} bits"
            )

        flat_source = _reshape_tile_to(self.ct, source, [source_elements], self.loc)
        target_element_type = _mlir_element_type(self, target_type)
        if source_type.dtype.bitwidth == target_type.dtype.bitwidth:
            retyped = self.ct.bitcast(target_element_type, flat_source, loc=self.loc)
        else:
            retyped = self.ct.unpack(
                target_element_type,
                self.ct.pack(flat_source, loc=self.loc),
                loc=self.loc,
            )
        return _reshape_tile_to(self.ct, retyped, target_shape, self.loc)

    def set_tile(self, buf_val: Value, mlir_tile: Any) -> None:
        """Update the MLIR tile value for a SHARED/REGISTER buffer.

        Called by Copy / Store emitters when writing into a tile-based buffer
        to keep the functional simulation state consistent.  Writing through a
        reshape/reinterpret-view alias updates the BASE buffer's tile
        (reshaped or storage-bitcast back).
        """
        base = self.buffer_aliases.get(buf_val)
        if base is not None:
            base_tile = self._reinterpret_buffer_tile(mlir_tile, buf_val.type, base.type)
            self.set_tile(base, base_tile)
            return
        self._tile_map[buf_val] = mlir_tile


# emit_module


def _optimization_hints_text(hints: tuple) -> str:
    """Render the frozen per-arch hints tuple as the CUDA Tile IR assembly form.

    ``hints`` is ``((arch_key, ((hint_key, value), ...)), ...)`` (the
    ``TileIRLoweringOptions.hints`` representation, pre-sorted by _validated_hints).
    Returns the full ``#cuda_tile.optimization_hints<sm_100 = {num_cta_in_cga = 2}, ...>``
    attribute text — ``OptimizationHintsAttr.parse`` requires the dialect
    prefix, the bare ``<...>`` form is rejected.

    All entry-scoped hint values are plain ints (``_validated_hints`` in
    lowering/__init__.py runs every value through ``_as_opt_int``, and
    rejects the only bool-typed hint key -- ``allow_tma`` -- outright, since
    it is load/store-scoped and has no effect on the kernel entry); there is
    no bool value to render here.

    Arch keys are rendered in sorted order (default last); hint keys within each
    arch are pre-sorted by _validated_hints so this renders them canonically.
    """
    arch_dicts = ", ".join(
        f"{arch_key} = {{" + ", ".join(f"{key} = {value}" for key, value in entries) + "}" for arch_key, entries in hints
    )
    return f"#cuda_tile.optimization_hints<{arch_dicts}>"


def emit_module(
    root: Block,
    *,
    kernel_name: str,
    entry_args: list[tuple[str, TileType]],
    arch: str | None = None,
    num_cta: int | None = None,
    occupancy: int | None = None,
    num_worker_warps: int | None = None,
    hints: tuple | None = None,
    return_ctx: bool = False,
    token_plan: Any = None,
    disable_tma: bool = False,
    fast_math: bool = False,
) -> Any:
    """Lower a TileIR ``Block`` to a ``cuda_tile.ModuleOp``.

    Creates an MLIR context, registers the dialect, builds the
    ``cuda_tile.module`` and an ``entry`` function whose arguments are
    derived from ``entry_args``.  Each entry argument's MLIR
    ``BlockArgument`` is bound into the ``EmitContext.value_map`` keyed by
    the corresponding ``root.params`` Value (position-aligned:
    ``root.params[i]`` ↔ ``entry_args[i]``).

    GLOBAL buffer args (``MemSpace.GLOBAL``) are *flattened*: instead of
    emitting one shaped tile arg they emit ``ptr + N×shape + N×stride``
    MLIR args and materialise a ``_BufferInfo`` stored in
    ``ctx._buffer_map[param]``.

    After binding entry arguments the function walks ``root.ops`` calling
    ``op.emit_mlir(ctx)`` for each op; if an op returns a non-None mlir
    value and the op has results, the returned value is bound to
    ``op.results[0]``.  Ops that return a sequence of mlir values have each
    element bound to the corresponding result (``op.results[i]``).

    Parameters
    ----------
    root : Block
        Root block of the TileIR program.  ``root.params`` must align with
        ``entry_args`` (same length, same positional order).
    kernel_name : str
        Symbol name for the CUDA Tile IR ``entry`` function.
    entry_args : list[tuple[str, TileType]]
        ``(param_name, tile_type)`` for each entry function argument.
        ``len(entry_args)`` must equal ``len(root.params)``.
    arch : str | None
        Target SM architecture string (e.g. ``"sm_90a"``).  Pass ``None``
        (the default) to omit the arch hint from the entry op — useful for
        tests that do not target a real GPU.
    num_cta : int | None
        Number of CTAs in a thread-block cluster (CGA). When provided, emitted
        as the ``num_cta`` optimization hint on the entry function.  ``None``
        omits the hint (default, no hint for single-CTA kernels without the knob).
    occupancy : int | None
        Per-SM CTA occupancy hint.  When provided, emitted as the ``occupancy``
        optimization hint on the entry function.  ``None`` omits the hint.
    num_worker_warps : int | None
        Number of worker warps per CTA, in {4, 8}. When provided, emitted as the
        ``num_worker_warps`` optimization hint on the entry function.  ``None``
        omits the hint.
    hints : tuple | None
        Per-arch ``optimization_hints`` dictionary in the frozen
        ``((arch_key, ((hint_key, value), ...)), ...)`` form (validated by
        ``_lowering_options``).  When provided, the entry is constructed via
        the raw ``ct_gen.EntryOp`` with an ``OptimizationHintsAttr.parse``-d
        per-arch attribute (``ct.entry`` can only express single-arch hints).
        Mutually exclusive with ``num_cta`` / ``occupancy`` /
        ``num_worker_warps``.  ``None`` keeps the ``ct.entry`` path.
    return_ctx : bool
        If True, return ``(module, ctx)`` instead of just ``module``.
        The EmitContext can then be inspected by callers (e.g. tests).
    token_plan : TokenPlan | None
        When provided (by ``pipeline.py`` from ``ctx.results["token_order"]``),
        stored on ``emit_ctx.token_plan`` so ``_ensure_token`` can compute
        precise RAW/WAW token dependencies instead of the conservative
        per-buffer chain.  When ``None`` (default), the fallback chain is
        used unchanged.

    Returns
    -------
    cuda_tile.ModuleOp
        The emitted module, or ``(ModuleOp, EmitContext)`` if
        ``return_ctx=True``.

    Raises
    ------
    ImportError
        If ``cuda_tile._mlir`` Python bindings are unavailable.
    ValueError
        If ``len(root.params) != len(entry_args)``.
    """
    if len(root.params) != len(entry_args):
        raise ValueError(
            f"emit_module: root.params has {len(root.params)} entries but entry_args has {len(entry_args)}; they must align 1-to-1."
        )

    try:
        from cuda_tile._mlir import ir
        from cuda_tile._mlir._mlir_libs._cuda_tile import register_dialect
        from cuda_tile._mlir.dialects import _cuda_tile_ops_gen as ct_gen
        from cuda_tile._mlir.dialects import cuda_tile_ops as ct
    except ImportError as exc:
        raise ImportError("emit_module: cuda_tile MLIR Python bindings are unavailable.") from exc

    mlir_ctx = ir.Context()
    with mlir_ctx:
        register_dialect(mlir_ctx, load=True)
        loc = ir.Location.unknown()
        with loc:
            module = ct.ModuleOp("tilelang", loc=loc)
            emit_ctx = EmitContext(ct=ct, ct_gen=ct_gen, ir=ir, loc=loc)
            # Wire the token_plan so _ensure_token can use precise deps.
            emit_ctx.token_plan = token_plan
            # Reshape/reinterpret-view aliases (get_tile/set_tile redirection).
            emit_ctx.buffer_aliases = dict(getattr(root, "buffer_aliases", {}) or {})
            # Store arch for optimization hints in Copy.emit_mlir.
            emit_ctx.arch = arch
            # Store global disable_tma so Copy.emit_mlir can force
            # allow_tma=False on every copy's load/store hint.
            emit_ctx.disable_tma = bool(disable_tma)
            # Forward TL_ENABLE_FAST_MATH so div/transcendental emit can use the
            # approximate (faster) cuTile primitives.
            emit_ctx.fast_math = bool(fast_math)

            # Build the flat MLIR arg-type list from entry_args.
            # GLOBAL buffer args are expanded to ptr + N×shape + N×stride.
            # Non-buffer args are a single scalar tile.
            flat_arg_types: list[Any] = []
            for _name, tile_ty in entry_args:
                if tile_ty.space == MemSpace.GLOBAL:
                    flat_arg_types.extend(_buffer_arg_types(emit_ctx, tile_ty))
                else:
                    flat_arg_types.append(_mlir_tile_type(emit_ctx, tile_ty))

            scratch_bytes, scratch_offsets = scratch_layout(root.alloca_buffers)
            if scratch_bytes:
                # One hidden, flat uint8 tensor: pointer, shape and stride.
                flat_arg_types.extend(
                    [ct.TileType.get([], ct.PointerType.get(ir.IntegerType.get_signless(8)))]
                    + [ct.TileType.get([], ir.IntegerType.get_signless(32))] * 2
                )

            with ir.InsertionPoint(module.body):
                function_type = ir.TypeAttr.get(ir.FunctionType.get(flat_arg_types, []))
                if hints is not None:
                    # Per-arch optimization_hints dictionary (T.Kernel tileir_hints).
                    # ct.entry can only build SINGLE-arch hints (via
                    # OptimizationHintsAttr.getEntryOpHint), so parse the multi-arch
                    # assembly form and construct the entry via the raw ct_gen.EntryOp.
                    # This replicates ct.entry's non-hint behavior exactly: it passes
                    # sym_name + function_type (already an ir.TypeAttr, same as the
                    # ct.entry path) straight to ct_gen.EntryOp with arg_attrs=None;
                    # sym_visibility is set by us below on both paths.
                    if num_cta is not None or occupancy is not None or num_worker_warps is not None:
                        raise ValueError(
                            "emit_module: per-arch `hints` and single-value entry knobs "
                            "(num_cta/occupancy/num_worker_warps) are mutually exclusive; "
                            "_lowering_options should have rejected this combination."
                        )
                    hints_text = _optimization_hints_text(hints)
                    try:
                        parsed_hints = ct.OptimizationHintsAttr.parse(hints_text, mlir_ctx)
                    except Exception as exc:
                        raise TileIRLoweringError(
                            f"emit_module: cuda_tile failed to parse the per-arch optimization_hints attribute text `{hints_text}`: {exc}"
                        ) from exc
                    entry_op = ct_gen.EntryOp(kernel_name, function_type, optimization_hints=parsed_hints, loc=loc)
                else:
                    entry_kwargs: dict[str, Any] = {"loc": loc}
                    if arch is not None:
                        entry_kwargs["arch"] = arch
                    # Forward num_cta / occupancy / num_worker_warps hints so the tileiras assembler
                    # can use the CGA / occupancy / num_worker_warps knobs (read from the
                    # tileir.num_ctas / tileir.occupancy / tileir.num_worker_warps kernel-launch annotations).
                    if num_cta is not None:
                        entry_kwargs["num_cta"] = num_cta
                    if occupancy is not None:
                        entry_kwargs["occupancy"] = occupancy
                    if num_worker_warps is not None:
                        entry_kwargs["num_worker_warps"] = num_worker_warps
                    entry_op = ct.entry(kernel_name, function_type, **entry_kwargs)
                entry_op.sym_visibility = ir.StringAttr.get("public")

                # Create the entry block with the flattened argument types.
                entry_block = entry_op.body.blocks.append(*flat_arg_types)

                with ir.InsertionPoint(entry_block):
                    # Iterate over root.params aligned with entry_args.
                    # Consume flat block.arguments in lock-step.
                    block_arg_iter = iter(entry_block.arguments)

                    for param_value, (_param_name, tile_ty) in zip(root.params, entry_args):
                        if tile_ty.space == MemSpace.GLOBAL:
                            # Consume 1 + 2*ndim block arguments for this buffer
                            # (ptr + N shape + N stride — full cuTile runtime ABI).
                            # _materialize_buffer uses constants for static shapes.
                            ndim = len(tile_ty.shape)
                            n_buf_args = 1 + ndim * 2
                            buf_args = [next(block_arg_iter) for _ in range(n_buf_args)]
                            _materialize_buffer(emit_ctx, param_value, tile_ty, buf_args)
                            # param_value is not bound in value_map; buffer
                            # params are accessed via _buffer_map exclusively.
                        else:
                            # Scalar / register tile: consume one block argument.
                            block_arg = next(block_arg_iter)
                            emit_ctx.bind(param_value, block_arg)

                    # Bind dynamic shape symbol placeholders (block.shape_bindings,
                    # populated by lower_kernel) to the materialized buffers'
                    # runtime shape tiles — the ABI already carries them.
                    _shape_bindings: dict = getattr(root, "shape_bindings", {})
                    for _ph_val, (_buf_val, _dim_idx) in _shape_bindings.items():
                        _binfo = emit_ctx._buffer_map.get(_buf_val)
                        if _binfo is None or _dim_idx >= len(_binfo.shape_tiles):
                            raise TileIRLoweringError(
                                f"emit_module: dynamic shape symbol placeholder for buffer "
                                f"`{getattr(_buf_val, 'name', _buf_val)}` dim {_dim_idx} has no "
                                "materialized shape tile."
                            )
                        emit_ctx.bind(_ph_val, _binfo.shape_tiles[_dim_idx])

                    # CUDA 13.4 can rematerialize alloca separately in producer
                    # and consumer warp groups. Use a runtime-owned workspace
                    # so every group addresses the same per-tile-block slice.
                    if scratch_bytes:
                        workspace = _as_tile(emit_ctx, next(block_arg_iter))
                        block_ids = [ct.exti(ct.Int64, v, loc=loc) for v in ct.get_tile_block_id(loc=loc)]
                        grid_dims = [ct.exti(ct.Int64, v, loc=loc) for v in ct.get_num_tile_blocks(loc=loc)]
                        yz = ct.add(block_ids[1], ct.mul(grid_dims[1], block_ids[2], loc=loc), loc=loc)
                        block_index = ct.add(block_ids[0], ct.mul(grid_dims[0], yz, loc=loc), loc=loc)
                        block_offset = ct.mul(block_index, ct.constant(scratch_bytes, ct.Int64, loc=loc), loc=loc)
                        scratch_base = ct.add(ct.ptr_to_int(workspace, loc=loc), block_offset, loc=loc)

                    _alloca_buffers: dict = getattr(root, "alloca_buffers", {}) or {}
                    for _aval, (_ashape, _adtype) in _alloca_buffers.items():
                        _elem_ty = _mlir_element_type(emit_ctx, _aval.type)
                        _ptr_ty = ct.PointerType.get(_elem_ty)
                        _scratch_ptr = ct.int_to_ptr(
                            _ptr_ty, ct.add(scratch_base, ct.constant(scratch_offsets[_aval], ct.Int64, loc=loc), loc=loc), loc=loc
                        )
                        _shape_tiles = [ct.constant(int(_d), ct.Int32, loc=loc) for _d in _ashape]
                        _strides = _static_contiguous_strides(tuple(int(_d) for _d in _ashape))
                        _stride_tiles = [ct.constant(int(_s), ct.Int32, loc=loc) for _s in _strides]
                        emit_ctx._buffer_map[_aval] = _BufferInfo(
                            dtype_name=_adtype,
                            ndim=len(_ashape),
                            ptr=_scratch_ptr,
                            shape_tiles=_shape_tiles,
                            stride_tiles=_stride_tiles,
                            view=None,
                        )
                        emit_ctx.alloca_values.add(_aval)

                    # Materialise SHARED/REGISTER alloc_buffers as zero-constant
                    # tiles in emit_ctx._tile_map.  These are kernel-local buffers
                    # (alloc_shared / alloc_fragment) that never appear as entry
                    # params — they are functional tile values held in SSA registers.
                    for alloc_val in getattr(root, "alloc_buffers", []):
                        tile_ty = alloc_val.type
                        elem_ty = _mlir_element_type(emit_ctx, tile_ty)
                        tile_shape = list(tile_ty.shape)
                        mlir_tile_type = ct.TileType.get(tile_shape, elem_ty)
                        zero_tile = ct.constant(0, tile_type=mlir_tile_type, loc=loc)
                        emit_ctx.set_tile(alloc_val, zero_tile)

                    # Bind block/thread index placeholder Values to MLIR values.
                    # block.index_values is populated by sem_to_ir._lower_thread_extent
                    # for each blockIdx / threadIdx variable that appears in the kernel.
                    # "bx"=blockIdx.x (index 0), "by"=blockIdx.y (1), "bz"=blockIdx.z (2).
                    # "tx"=threadIdx.x, "ty"=threadIdx.y, "tz"=threadIdx.z (bound to 0).
                    _index_values: dict = getattr(root, "index_values", {})
                    _block_axes = {"bx": 0, "by": 1, "bz": 2}
                    _block_key_names = [k for k in _block_axes if k in _index_values]
                    if _block_key_names:
                        # Emit get_tile_block_id() once to get (x_tile, y_tile, z_tile).
                        _block_id_tiles = ct.get_tile_block_id(loc=loc)
                        for _axis_name in _block_key_names:
                            _axis_idx = _block_axes[_axis_name]
                            _placeholder_val = _index_values[_axis_name]
                            emit_ctx.bind(_placeholder_val, _block_id_tiles[_axis_idx])
                    # Bind threadIdx.* vars to real per-thread lane indices using
                    # ct.iota(extent, Int32).  ct.iota(N, Int32) emits a 1D tile
                    # [0, 1, ..., N-1] so each "thread" in the SPMD tile model
                    # sees its own lane index.  This fixes per-thread SIMT scatter
                    # kernels (e.g. dequant_gemm) where tx is used in address/data
                    # computation — constant-0 would give all threads the same data.
                    # For ty/tz (extent 1), iota(1) = [0] ≡ scalar-0.
                    _thread_extents: dict = getattr(root, "thread_extents", {})
                    _thread_axes = {"tx", "ty", "tz"}
                    _i32_ty = ir.IntegerType.get_signless(32)
                    _i32_tile_type = ct.TileType.get([], _i32_ty)
                    _used_values = _referenced_values(root)
                    for _tkey in _thread_axes:
                        if _tkey in _index_values and _index_values[_tkey] in _used_values:
                            _extent = _thread_extents.get(_tkey, 0)
                            if _extent > 1:
                                # emit iota(extent) → tile<extent x i32> = [0..extent-1]
                                _iota_tile = ct.iota(_extent, ct.Int32, loc=loc)
                                emit_ctx.bind(_index_values[_tkey], _iota_tile)
                            else:
                                # extent ≤ 1 (ty/tz single-thread): use scalar constant 0
                                _zero_tile = ct.constant(0, tile_type=_i32_tile_type, loc=loc)
                                emit_ctx.bind(_index_values[_tkey], _zero_tile)

                    # Walk the op list and emit each op.
                    # Set current_op before each emit_mlir call so _ensure_token
                    # can look up this op's deps in token_plan.
                    for op in root.ops:
                        emit_ctx.current_op = op
                        try:
                            mlir_result = op.emit_mlir(emit_ctx)
                        except KeyError as _ke:
                            # Buffer-lookup KeyErrors from emit_mlir mean the op
                            # pattern is unsupported (e.g. a REGISTER buffer where the
                            # op only handles GLOBAL buffers).  Re-raise as
                            # TileIRLoweringNotImplementedError so callers that expect a
                            # hard rejection (not a silent miscompile) see the right type.
                            raise TileIRLoweringNotImplementedError(f"Emit of {type(op).__name__} failed: {_ke}") from _ke

                        # Bind returned mlir values to op.results.
                        # Handles three return shapes:
                        #   None / empty   — side-effect-only op, no binding.
                        #   single object  — bind to op.results[0].
                        #   sequence       — bind op.results[i] ↔ mlir_result[i].
                        if mlir_result is None:
                            pass
                        elif isinstance(mlir_result, (list, tuple)):
                            if len(mlir_result) != len(op.results):
                                raise ValueError(
                                    f"emit_module: {type(op).__name__}.emit_mlir() returned "
                                    f"{len(mlir_result)} mlir values but the op declares "
                                    f"{len(op.results)} result(s); they must match 1-to-1."
                                )
                            for res_val, mlir_val in zip(op.results, mlir_result):
                                emit_ctx.bind(res_val, mlir_val)
                        else:
                            # Single mlir value — bind to first result if present.
                            if op.results:
                                emit_ctx.bind(op.results[0], mlir_result)

                    # Terminate the entry block.
                    ct.ret([], loc=loc)

            # Return inside the MLIR context.
            if return_ctx:
                return module, emit_ctx
            return module
