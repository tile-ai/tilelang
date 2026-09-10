"""Shared MLIR emission helpers for TileIR operations.

Operation modules depend on these helpers, so this module works with
duck-typed operations and does not import ``tilelang.tileir.ir.ops``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from tilelang.tileir.errors import TileIRLoweringError, TileIRLoweringNotImplementedError

__all__ = [
    "_BufferInfo",
    "_mlir_element_type",
    "_mlir_tile_type",
    "_dtype_supports_tensor_view",
    "_buffer_arg_types",
    "_materialize_buffer",
    "_ensure_token",
    "_make_i32_index_tiles",
    "_as_tile",
    "_as_token",
    "_walk_block",
    "_block_ends_with_terminator",
    # Elementwise / Cast / Select emit helpers
    "_emit_elementwise",
    "_emit_cast",
    "_emit_select",
    # Shared buffer load/store helpers
    "_load_buffer_tile",
    "_store_buffer_tile",
    # gather/scatter view construction (CopyGather / CopyScatter)
    "_make_gather_scatter_view",
    # cast helpers exposed for Copy/TmaCopy cross-dtype support
    "_cast_tile",
    "_dtype_from_mlir_type",
    # static-shape helpers
    "_all_static",
    "_static_contiguous_strides",
    "_power_of_two_divisor",
    "_elements_per_16_bytes",
    # high-rank (4D) copy/partition_view helpers
    "_squeeze_shape",
    "_reshape_tile_to",
    # ptr broadcast helper (rank-safe ct.broadcast for scalar ptrs)
    "_broadcast_ptr",
]


# _BufferInfo — materialised entry-arg state for a GLOBAL buffer


@dataclasses.dataclass
class _BufferInfo:
    """Holds the MLIR values materialised for one GLOBAL buffer entry arg.

    Attributes
    ----------
    dtype_name : str
        Canonical dtype string (e.g. ``"float16"``).
    ndim : int
        Number of buffer dimensions.
    ptr : mlir.Tile
        0-d scalar tile holding the base pointer (after ``assume_div_by``).
    shape_tiles : list[mlir.Tile]
        Per-dimension shape tiles (may be static-bounded i32 scalars).
    stride_tiles : list[mlir.Tile]
        Per-dimension stride tiles.
    view : mlir.TensorView | None
        ``make_tensor_view`` result, or ``None`` for dtypes that do not
        support TensorView (e.g. int4 — see ``_dtype_supports_tensor_view``).
    """

    dtype_name: str
    ndim: int
    ptr: Any
    shape_tiles: list
    stride_tiles: list
    view: Any  # mlir.TensorView | None


# _mlir_element_type  — TileType → cuda_tile MLIR element type


def _mlir_element_type(ctx: Any, tile_ty: Any) -> Any:
    """Return the MLIR element type corresponding to *tile_ty.dtype*.

    Handles the full dtype table.
    """
    ir = ctx.ir
    name = tile_ty.dtype.name
    # Float types
    _FLOAT_IR_TYPES: dict[str, str] = {
        "float16": "F16Type",
        "bfloat16": "BF16Type",
        "tf32": "FloatTF32Type",
        "float32": "F32Type",
        "float64": "F64Type",
        "float8_e4m3fn": "Float8E4M3FNType",
        "float8_e4m3": "Float8E4M3FNType",  # TVM alias without 'fn' suffix
        "float8_e5m2": "Float8E5M2Type",
        "float8_e8m0fnu": "Float8E8M0FNUType",
        "float4_e2m1fn": "Float4E2M1FNType",
    }
    if name in _FLOAT_IR_TYPES:
        return getattr(ir, _FLOAT_IR_TYPES[name]).get()
    # Integer types (signless)
    _INT_BITS: dict[str, int] = {
        "bool": 1,
        "int4": 4,
        "int8": 8,
        "uint8": 8,
        "int16": 16,
        "uint16": 16,
        "int32": 32,
        "uint32": 32,
        "int64": 64,
        "uint64": 64,
    }
    if name in _INT_BITS:
        return ir.IntegerType.get_signless(_INT_BITS[name])
    raise ValueError(f"emit_module: unsupported dtype for entry arg: {name!r}")


def _mlir_tile_type(ctx: Any, tile_ty: Any) -> Any:
    """Return ``cuda_tile.TileType`` for *tile_ty* (scalar or shaped).

    For GLOBAL buffer types this returns the *logical* shaped tile type, not
    the pointer-based arg type.  The actual entry-arg flattening (ptr + shape +
    stride) is handled separately in ``_materialize_buffer_args``.

    Shape comes from ``tile_ty.shape``; an empty shape maps to a 0-d
    (scalar) tile ``!cuda_tile.tile<elem_type>``.
    """
    elem_ty = _mlir_element_type(ctx, tile_ty)
    return ctx.ct.TileType.get(list(tile_ty.shape), elem_ty)


# _dtype_supports_tensor_view


def _dtype_supports_tensor_view(dtype_name: str) -> bool:
    """Return True if dtype supports TensorView (i4 does not)."""
    return dtype_name not in {"int4"}


# high-rank (4D) copy helpers


def _squeeze_shape(tile_shape: list) -> list:
    """Remove size-1 dimensions from *tile_shape*.

    The 4D ``tile_shape`` on a Copy from a 4D buffer (e.g. ``(1,64,1,128)``)
    carries singleton batch/head dims used only for TMA addressing.  After
    loading via ``make_partition_view`` the loaded tile has this 4D shape, but
    the consumer (SHARED buffer or Gemm operand) expects the squeezed 2D shape
    ``(64, 128)``.  The squeezed non-singleton dims must match the target tile
    shape for the view path to be valid.

    Returns
    -------
    list[int]
        A new list with all size-1 dims removed.  If no dims remain (all-ones
        shape) returns ``[1]`` to avoid an empty shape.
    """
    squeezed = [d for d in tile_shape if d != 1]
    return squeezed if squeezed else [1]


def _reshape_tile_to(ct: Any, tile: Any, target_shape: list, loc: Any) -> Any:
    """Reshape *tile* to *target_shape* if the shapes differ.

    A no-op (returns *tile* unchanged) when shapes already match.  Used to
    reconcile the 4D shape emitted by ``load_view_tko`` with the 2D shape
    expected by SHARED buffers and Gemm operands.

    Parameters
    ----------
    ct :
        ``cuda_tile._mlir.dialects.cuda_tile_ops`` builder.
    tile :
        ``ct.Tile`` value to reshape.
    target_shape :
        The desired shape as a list of ints.
    loc :
        MLIR location for the reshape op.
    """
    current = list(tile.tile_type.shape)
    if current == target_shape:
        return tile
    return ct.reshape(target_shape, tile, loc=loc)


def _build_gather_ptrs(ctx: Any, buf_val: Any, result_shape: list, dim_kinds: tuple, dim_values: tuple, dim_axes: tuple, loc: Any) -> Any:
    """Build a per-element pointer tile for gather/scatter ops.

    ``ptr[e] = base + Σ_k idx_k[e] * stride_k`` over *result_shape*, where each
    buffer dim's index contribution is described by
    ``dim_kinds[k]`` (``"const" | "scalar" | "iota" | "tile"``) — see
    ``GatherLoad`` for the spec.  Strides come from the materialized
    ``_BufferInfo.stride_tiles`` (static constants or runtime ABI args).
    """
    ct = ctx.ct
    buf_info = ctx.get_buffer_info(buf_val)
    ndim_res = len(result_shape)
    stride_tiles = list(getattr(buf_info, "stride_tiles", ()) or ())
    if len(stride_tiles) != len(dim_kinds):
        raise TileIRLoweringNotImplementedError(
            f"gather pointers for `{buf_val.name}`: {len(dim_kinds)} dim specs but {len(stride_tiles)} buffer strides."
        )

    def _bcast_scalar(scalar_tile: Any) -> Any:
        return ct.broadcast(result_shape, ct.reshape([1] * ndim_res, scalar_tile, loc=loc), loc=loc)

    offsets = None
    for kind, val, axis, stride_scalar in zip(dim_kinds, dim_values, dim_axes, stride_tiles):
        if kind in ("const", "scalar"):
            idx_scalar = _make_i32_index_tiles(ctx, (val,))[0]
            contrib_idx = _bcast_scalar(idx_scalar)
        elif kind == "iota":
            base_scalar = _make_i32_index_tiles(ctx, (val,))[0]
            base_full = _bcast_scalar(base_scalar)
            ext = result_shape[axis]
            iota_1d = ct.iota(ext, ct.Int32, loc=loc)
            rank_shape = [1] * ndim_res
            rank_shape[axis] = ext
            iota_full = ct.broadcast(result_shape, ct.reshape(rank_shape, iota_1d, loc=loc), loc=loc)
            contrib_idx = ct.add(base_full, iota_full, loc=loc)
        elif kind == "tile":
            idx_tile = _as_tile(ctx, ctx.lookup(val))
            if list(idx_tile.tile_type.shape) != list(result_shape):
                raise TileIRLoweringNotImplementedError(
                    f"gather pointers for `{buf_val.name}`: index tile shape "
                    f"{list(idx_tile.tile_type.shape)} != result shape {list(result_shape)}."
                )
            contrib_idx = idx_tile
        else:
            raise TileIRLoweringNotImplementedError(f"gather pointers for `{buf_val.name}`: unknown dim kind {kind!r}.")
        stride_full = _bcast_scalar(_as_tile(ctx, stride_scalar))
        contrib = ct.mul(contrib_idx, stride_full, loc=loc)
        offsets = contrib if offsets is None else ct.add(offsets, contrib, loc=loc)

    ptr_base = _broadcast_ptr(ct, buf_info.ptr, result_shape, loc=loc)
    # Keep the operation builder from the same dialect module as the values.
    # CUDA Tile 13.3 exposes both ``cuda_tile`` and ``cuda_tile_ops`` module
    # names in some builds.  Importing ``_offset`` from the latter while the
    # values were built by the former creates distinct Python ``Tile`` classes,
    # so its ``isinstance`` check mistakes a tile offset for a Python scalar.
    return ct._offset(ptr_base, offsets, loc=loc)


def _make_tile_view(ct: Any, tensor_view: Any, tile_shape: list, *, elem_view: bool, padding_value: Any = None, loc: Any) -> Any:
    """Build a TileView over *tensor_view* for load/store/atomic TKO ops.

    ``elem_view=False`` — ``make_partition_view``: indices are tile-granular
    (index k addresses elements ``[k*extent, (k+1)*extent)``).

    ``elem_view=True`` — ``make_strided_view`` with unit traversal strides:
    indices are raw ELEMENT offsets (a sliding window advancing one element
    per index step). Used when a region's start offset is not statically
    divisible by the tile extent, e.g. varlen attention's
    ``cu_seqlens[b] + bx*block_M`` row bases.
    """
    try:
        static_shape = [int(dim) for dim in tile_shape]
    except (TypeError, ValueError) as exc:
        raise TileIRLoweringNotImplementedError(f"TileView requires a static tile shape, got {tile_shape!r}.") from exc
    invalid_dims = [dim for dim in static_shape if dim <= 0 or dim & (dim - 1)]
    if invalid_dims:
        raise TileIRLoweringNotImplementedError(
            "CUDA Tile IR TileView dimensions must be power-of-two positive integers; "
            f"got tile shape {static_shape} with unsupported dimensions {invalid_dims}."
        )

    if elem_view:
        return ct.make_strided_view(
            tensor_view,
            static_shape,
            [1] * len(static_shape),
            padding_value=padding_value,
            loc=loc,
        )
    return ct.make_partition_view(
        tensor_view,
        static_shape,
        padding_value=padding_value,
        loc=loc,
    )


def _broadcast_ptr(ct: Any, ptr: Any, target_shape: list, loc: Any) -> Any:
    """Broadcast a pointer scalar tile to *target_shape*.

    ``ct.broadcast`` requires source and result to have the same rank.
    When *ptr* is a rank-0 scalar (shape ``[]``) and *target_shape* is non-empty,
    we must first reshape to ``[1]*len(target_shape)`` then broadcast.

    This avoids ``'cuda_tile.broadcast' op failed to verify that all of
    {source, result} have same rank`` when loading from a GLOBAL ptr-based buffer
    (no TensorView, e.g. int4 or scalar loads) into a non-scalar tile.
    """
    if not target_shape:
        return ptr  # ptr is already scalar; no broadcast needed
    src_shape = list(ptr.tile_type.shape)
    rank = len(target_shape)
    if src_shape == target_shape:
        return ptr
    if src_shape != [1] * rank:
        ptr = ct.reshape([1] * rank, ptr, loc=loc)
    return ct.broadcast(target_shape, ptr, loc=loc)


# _static_contiguous_strides — compute row-major strides from a shape


def _static_contiguous_strides(shape: tuple) -> tuple:
    """Return row-major (C-order) strides for a static shape.

    E.g. shape=(M, K) → strides=(K, 1), shape=(1024, 512) → (512, 1).
    """
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _power_of_two_divisor(value: int, *, limit: int) -> int:
    """Largest power-of-two that divides *value*, capped at *limit*."""
    divisor = 1
    while divisor < limit and value % (divisor * 2) == 0:
        divisor *= 2
    return divisor


def _elements_per_16_bytes(dtype_name: str) -> int:
    """Number of elements that fit in 16 bytes for *dtype_name*."""
    _BIT_WIDTHS: dict[str, int] = {
        "bool": 1,  # a bool is 1 bit (perf hint only)
        "int4": 4,
        "float4_e2m1fn": 4,
        "int8": 8,
        "uint8": 8,
        "float8_e4m3fn": 8,
        "float8_e5m2": 8,
        "float8_e8m0fnu": 8,
        "int16": 16,
        "uint16": 16,
        "float16": 16,
        "bfloat16": 16,
        "tf32": 32,
        "tfloat32": 32,
        "float32": 32,
        "int32": 32,
        "uint32": 32,
        "float64": 64,
        "int64": 64,
        "uint64": 64,
    }
    bit_width = _BIT_WIDTHS.get(dtype_name)
    if bit_width is None:
        return 1
    return max(1, 128 // bit_width)


def _all_static(shape: tuple) -> bool:
    """Return whether every dimension is a positive Python integer."""
    return all(isinstance(d, int) and d > 0 for d in shape)


# _buffer_arg_types  — compute the flat MLIR arg types for a GLOBAL buffer


def _buffer_arg_types(ctx: Any, tile_ty: Any) -> list:
    """Return the flat MLIR arg-type list for one GLOBAL buffer.

    Always emits the full ABI layout expected by the native dispatcher:
      - 1 scalar tile holding the base pointer  (``tile<ptr<dtype>>``)
      - N scalar i32 tiles for the N shape dims (``tile<i32>``)
      - N scalar i32 tiles for the N stride dims (``tile<i32>``)

    For a 2-D fp16 buffer this yields 5 arg types.

    Static buffers retain shape and stride arguments for ABI compatibility;
    ``_materialize_buffer`` replaces those runtime values with constants.
    """
    ct = ctx.ct
    ir = ctx.ir
    elem_ty = _mlir_element_type(ctx, tile_ty)
    ptr_type = ct.PointerType.get(elem_ty)
    tile_ptr_type = ct.TileType.get([], ptr_type)
    tile_i32_type = ct.TileType.get([], ir.IntegerType.get_signless(32))
    ndim = len(tile_ty.shape)
    return [tile_ptr_type] + [tile_i32_type] * (ndim * 2)


# _materialize_buffer  — bind block args to _BufferInfo for a GLOBAL param


def _materialize_buffer(
    ctx: Any,
    param_val: Any,
    tile_ty: Any,
    block_args: list,  # pre-consumed block.arguments slice for this buffer
) -> None:
    """Build the ``_BufferInfo`` for one GLOBAL buffer param and store it.

    ``block_args`` must always contain 1 + 2*N MLIR block arguments
    (ptr + N shape + N stride), matching the cuTile runtime ABI.

    Static shapes and strides become constants with divisibility hints.
    Dynamic values retain their runtime arguments and boundedness hints. Global
    buffers are registered in ``ctx._buffer_map`` rather than ``ctx.value_map``.
    """
    ct = ctx.ct
    ir = ctx.ir
    loc = ctx.loc
    elem_ty = _mlir_element_type(ctx, tile_ty)
    ndim = len(tile_ty.shape)
    dtype_name = tile_ty.dtype.name

    ptr_type = ct.PointerType.get(elem_ty)
    tile_ptr_type = ct.TileType.get([], ptr_type)
    tile_i32_type = ct.TileType.get([], ir.IntegerType.get_signless(32))

    # ptr arg (always first) — used in both static and dynamic paths.
    raw_ptr = ct.Tile(block_args[0], tile_ptr_type)
    ptr = ct.assume_div_by(raw_ptr, 16, loc=loc)

    if _all_static(tile_ty.shape):
        # Static path: constants for shape + integer literal strides
        # Consume (and discard) the N shape + N stride block args so the
        # block-argument iterator in emit_module stays aligned.
        # (block_args[1 .. 1+ndim]      = shape args — ignored)
        # (block_args[1+ndim .. 1+2*ndim] = stride args — ignored)

        static_shape = tuple(int(d) for d in tile_ty.shape)
        static_strides = _static_contiguous_strides(static_shape)

        # Shape tiles: MLIR constant i32 scalars.
        shape_tiles = [ct.constant(dim, ct.Int32, loc=loc) for dim in static_shape]

        # Stride tiles: MLIR constant i32 scalars with assume_div_by hints.
        # These are stored in _BufferInfo for callers that need tile<i32> values
        # (e.g. direct ptr-based paths). The assume_div_by divisor is
        #   min(_power_of_two_divisor(stride, 16), elements_per_16_bytes(dtype))
        stride_tiles = []
        for stride_val in static_strides:
            raw_stride = ct.constant(stride_val, ct.Int32, loc=loc)
            divisor = min(
                _power_of_two_divisor(stride_val, limit=16),
                _elements_per_16_bytes(dtype_name),
            )
            if divisor > 1:
                stride_tile = ct.assume_div_by(raw_stride, divisor, loc=loc)
            else:
                stride_tile = raw_stride
            stride_tiles.append(stride_tile)

        # Pass integer strides so the static layout is encoded in the TensorView
        # type instead of represented as dynamic tile values.
        view = None
        if _dtype_supports_tensor_view(dtype_name):
            view = ct.make_tensor_view(ptr, elem_ty, shape_tiles, list(static_strides), loc=loc)
        ctx._buffer_map[param_val] = _BufferInfo(
            dtype_name=dtype_name,
            ndim=ndim,
            ptr=ptr,
            shape_tiles=shape_tiles,
            stride_tiles=stride_tiles,
            view=view,
        )
        return  # static path complete

    else:
        # Preserve statically known trailing strides even when leading
        # dimensions are dynamic. In row-major order,
        # stride[i] = product(shape[i+1:]).
        shape = tile_ty.shape  # may contain -1 sentinels for dynamic dims

        def _trailing_product_is_static(dim_idx: int) -> tuple:
            """Return (True, product) if all dims after dim_idx are static, else (False, None)."""
            product = 1
            for k in range(dim_idx + 1, ndim):
                d = shape[k]
                if not (isinstance(d, int) and d > 0):
                    return False, None
                product *= int(d)
            return True, product

        shape_tiles = []
        for i in range(ndim):
            raw_dim = ct.Tile(block_args[1 + i], tile_i32_type)
            dim_val = shape[i]
            if isinstance(dim_val, int) and dim_val > 0:
                # Static dimension: emit as constant (matching assume_bounded logic).
                shape_tiles.append(ct.constant(int(dim_val), ct.Int32, loc=loc))
            else:
                dim = ct.assume_bounded(raw_dim, 0, None, loc=loc)
                shape_tiles.append(dim)

        stride_tiles = []
        # Compute which strides are static (row-major rule).
        static_strides_int: list = []  # None if dynamic, int if static
        for i in range(ndim):
            is_static, product = _trailing_product_is_static(i)
            static_strides_int.append(product if is_static else None)

        for i in range(ndim):
            raw_str = ct.Tile(block_args[1 + ndim + i], tile_i32_type)
            sv = static_strides_int[i]
            if sv is not None:
                # Static stride: emit as constant with assume_div_by hint.
                raw_stride = ct.constant(sv, ct.Int32, loc=loc)
                divisor = min(
                    _power_of_two_divisor(sv, limit=16),
                    _elements_per_16_bytes(dtype_name),
                )
                if divisor > 1:
                    stride_tile = ct.assume_div_by(raw_stride, divisor, loc=loc)
                else:
                    stride_tile = raw_stride
                stride_tiles.append(stride_tile)
            else:
                s = ct.assume_bounded(raw_str, 0, None, loc=loc)
                stride_tiles.append(s)

    # Build TensorView.  Pass strides as Python INT LITERALS for fully-static strides
    # (so the MLIR type shows strides=[N,1] not strides=[?,?]).
    # Pass tile<i32> stride values for dynamic strides.
    # This mixed approach requires assembling the stride arg list carefully:
    # if ALL strides are static → pass as int list (fully static type).
    # if ANY stride is dynamic → pass mixed list via tile<i32> values.
    view = None
    if _dtype_supports_tensor_view(dtype_name):
        # Determine whether all strides are statically-known.
        _all_static_strides = all(sv is not None for sv in static_strides_int)
        if _all_static_strides:
            # Pass strides as Python ints so the MLIR type embeds them (strides=[N,1]).
            int_strides = [sv for sv in static_strides_int]
            view = ct.make_tensor_view(ptr, elem_ty, shape_tiles, int_strides, loc=loc)
        else:
            view = ct.make_tensor_view(ptr, elem_ty, shape_tiles, stride_tiles, loc=loc)

    ctx._buffer_map[param_val] = _BufferInfo(
        dtype_name=dtype_name,
        ndim=ndim,
        ptr=ptr,
        shape_tiles=shape_tiles,
        stride_tiles=stride_tiles,
        view=view,
    )


# _ensure_token  — obtain or create a fresh token for a buffer


def _ensure_token(ctx: Any, buf_val: Any) -> Any:
    """Return the MLIR input-token for the current op touching *buf_val*.

    Two modes:

    **Plan-based (when ``ctx.token_plan`` is present):**
        Look up ``ctx.current_op``'s dependency set in the plan.  Join the
        out-tokens of all dep-ops recorded in ``ctx._op_token``.  Dep-ops
        that have not yet produced a token (e.g. non-memory ops or first op)
        are skipped; if no dep tokens exist, a fresh ``make_token`` is used.
        This path enables load-load overlap (deps set is empty for RAW-only
        ops with no prior store) while preserving store→store / store→load
        ordering.

    **Conservative fallback (when ``ctx.token_plan`` is absent):**
        Per-buffer chain: the first op on a buffer gets a fresh
        ``make_token``; subsequent ops chain off the previous op's out-token
        stored in ``ctx._token_map[buf_val]``.
    """
    plan = getattr(ctx, "token_plan", None)
    current_op = getattr(ctx, "current_op", None)

    if plan is not None and current_op is not None:
        deps = plan.deps_for(current_op)
        dep_tokens = []
        op_token_map = getattr(ctx, "_op_token", {})
        for dep_op in deps:
            dep_tok = op_token_map.get(id(dep_op))
            if dep_tok is not None:
                dep_tokens.append(dep_tok)

        if not dep_tokens:
            # Unconstrained operations share a block-local root token. Inside a
            # pipelined loop, the loop-entry token preserves cross-iteration order
            # while allowing operations within an iteration to overlap.
            pipelined_tok_arg = getattr(ctx, "_pipelined_loop_token_entry_arg", None)
            if pipelined_tok_arg is not None:
                return pipelined_tok_arg

            global_root_tok = getattr(ctx, "_global_root_token", None)
            if global_root_tok is None:
                global_root_tok = ctx.ct.make_token(loc=ctx.loc)
                ctx._global_root_token = global_root_tok
            return global_root_tok
        elif len(dep_tokens) == 1:
            return dep_tokens[0]
        else:
            ct = ctx.ct
            try:
                joined = ct.join_tokens(*dep_tokens, loc=ctx.loc)
            except (AttributeError, TypeError) as exc:
                # Falling back to one dependency would drop ordering edges.
                raise TileIRLoweringNotImplementedError(
                    f"join_tokens unavailable on this toolchain; cannot serialize "
                    f"{len(dep_tokens)} dependency tokens without dropping edges"
                ) from exc
            return joined

    # Conservative per-buffer-chain fallback
    tok = ctx._get_token(buf_val)
    if tok is None:
        tok = ctx.ct.make_token(loc=ctx.loc)
        ctx._set_token(buf_val, tok)
    return tok


# _make_i32_index_tiles — build constant i32 index tiles


def _make_i32_index_tiles(ctx: Any, indices: tuple) -> list:
    """Build a list of 0-d i32 constant tiles from an index tuple.

    Each element in *indices* may be:
    - a plain ``int`` → emit a ``ct.constant(n, tile_type=tile<i32>)``
    - a TileIR ``Value`` object → look up in ``ctx.value_map`` and return
      the MLIR value produced by semantic-to-TileIR expression lowering.

    This dual handling is needed for runtime partition indices (e.g.
    ``bx``, ``k``) computed from loop/block variables.
    """
    ct = ctx.ct
    ir = ctx.ir
    i32_tile_type = ct.TileType.get([], ir.IntegerType.get_signless(32))
    result = []
    # Keep the import local because TileIR operation modules import this module.
    from tilelang.tileir.ir.value import Value as TileIRValue

    for idx in indices:
        if isinstance(idx, TileIRValue):
            # Runtime index: look up the MLIR value and wrap as tile.
            mlir_val = ctx.lookup(idx)
            tile_val = _as_tile(ctx, mlir_val)
            # load_view_tko requires rank-0 (scalar) index tiles.  When a
            # thread-index variable (e.g. ``tn`` / ``tx``) is used as a
            # partition index, ``mlir_emit.py`` binds it to
            # ``ct.iota(extent, Int32)`` — a rank-1 tile<N x i32>.  Reduce such
            # rank-1 (or higher) tiles to rank-0 via extract element-0 then reshape.
            # ``ct.reshape([], tile)`` is only valid when tile has exactly 1 element
            # (same total element count required).  For an N-element tile we first
            # extract element [0] as a tile<1 x i32> then reshape that to scalar.
            idx_shape = list(tile_val.tile_type.shape)
            if idx_shape:  # non-scalar (rank >= 1)
                i32 = ctx.ir.IntegerType.get_signless(32)
                i32_scalar_ty = ct.TileType.get([], i32)
                one_ty = ct.TileType.get([1], i32)
                zero_tile = ct.constant(0, tile_type=i32_scalar_ty, loc=ctx.loc)
                elem = ct.extract(one_ty, tile_val, [zero_tile], loc=ctx.loc)
                tile_val = ct.reshape([], elem, loc=ctx.loc)
            result.append(tile_val)
        elif hasattr(idx, "tile_type"):
            # Already an MLIR tile (e.g. a computed element base from the
            # no-view gather path) — pass through.
            result.append(idx)
        else:
            result.append(ct.constant(int(idx), tile_type=i32_tile_type, loc=ctx.loc))
    return result


# _block_ends_with_terminator — detect explicit block termination


def _block_ends_with_terminator(block: Any) -> bool:
    """Return True if the last op in *block* is a terminator (Break/Continue).

    Used by control-flow emit_mlir to decide whether to auto-insert
    loop_continue at the end of a loop body.
    """
    if not block.ops:
        return False
    last = block.ops[-1]
    return getattr(last, "_terminator", False)


# _walk_block — walk a TileIR Block's ops inside the current InsertionPoint


def _walk_block(block: Any, ctx: Any) -> None:
    """Emit all ops in *block* into the current MLIR InsertionPoint.

    Mirrors the op-walking loop in ``emit_module`` but operates on a nested
    ``Block`` (loop body / if branch) rather than the root block.  Result
    binding follows the same three-case logic:

    - ``None`` / empty → side-effect only, no binding.
    - single object     → bind to ``op.results[0]``.
    - sequence          → bind ``op.results[i] ↔ mlir_result[i]``.

    This module does not import ``ir.ops``; it relies on the
    duck-typed ``op.emit_mlir(ctx)`` protocol and the ``results`` attribute
    that every TileOp carries.

    The global root token is reset (and restored) at each block boundary.
    The root token is scoped to the current block so that ops in different blocks
    (e.g. serial loop iterations or if-else branches) use fresh tokens. Operations
    within one block share a root token so independent loads can overlap.
    """
    # Reset the global root token for this block scope.
    saved_root = getattr(ctx, "_global_root_token", None)
    ctx._global_root_token = None

    for op in block.ops:
        # Set current_op before each emit_mlir call so that _ensure_token can
        # look up this op's deps in ctx.token_plan.  This mirrors the same
        # threading in emit_module's root op-walk.
        ctx.current_op = op
        mlir_result = op.emit_mlir(ctx)

        if mlir_result is None:
            pass
        elif isinstance(mlir_result, (list, tuple)):
            if len(mlir_result) != len(op.results):
                raise ValueError(
                    f"_walk_block: {type(op).__name__}.emit_mlir() returned "
                    f"{len(mlir_result)} mlir values but the op declares "
                    f"{len(op.results)} result(s); they must match 1-to-1."
                )
            for res_val, mlir_val in zip(op.results, mlir_result):
                ctx.bind(res_val, mlir_val)
        else:
            if op.results:
                ctx.bind(op.results[0], mlir_result)

    # Restore the parent block's root token so outer scopes are unaffected.
    ctx._global_root_token = saved_root


# _as_tile — ensure an MLIR value is wrapped as ct.Tile


def _as_tile(ctx: Any, mlir_val: Any) -> Any:
    """Ensure *mlir_val* is a ``ct.Tile``.

    MLIR block arguments and some op results are plain ``mlir.ir.Value``
    objects, not the ``ct.Tile`` wrapper that TKO helpers expect.  Wrapping
    them with ``ct.Tile(v, v.type)`` is always safe and idempotent for
    ``ct.Tile`` objects (they are already wrapped).
    """
    ct = ctx.ct
    if isinstance(mlir_val, ct.Tile):
        return mlir_val
    return ct.Tile(mlir_val, mlir_val.type)


# _as_token — ensure an MLIR value is wrapped as ``ct.Token``


def _as_token(ctx: Any, mlir_val: Any) -> Any:
    """Ensure *mlir_val* is a ``ct.Token``.

    MLIR block arguments carried as loop iter-args (token-typed) are plain
    ``mlir.ir.Value`` / ``mlir.ir.BlockArgument`` objects, not the ``ct.Token``
    wrapper that the TKO helpers (``store_view_tko``, ``load_view_tko``, …)
    require for their ``input_token`` argument.  Wrapping them with
    ``ct.Token(v)`` is safe and idempotent for objects already of type
    ``ct.Token`` (returned unchanged).  ``None`` is passed through so callers
    can treat "no token" uniformly.
    """
    if mlir_val is None:
        return None
    ct = ctx.ct
    if isinstance(mlir_val, ct.Token):
        return mlir_val
    return ct.Token(mlir_val)


# Elementwise / Cast / Select emit helpers

# _UNARY_MATH_TABLE
#   Maps TileIR unary-math function name → lambda(ct, tile, loc) → mlir_tile.
#   "abs", "neg", "sigmoid", "not" use wrapper lambdas for the multi-step helpers.


def _build_unary_math_table():
    """Build and return the {fn_name: (ct, tile, loc) -> mlir_tile} table.

    Calling this once at import time to avoid re-building every call.
    The table is built lazily (inside the function) so that cuda_tile is not
    imported at module load time — it is only needed when emit_mlir runs.
    """
    # The table is built as a plain dict of lambdas.  Each lambda takes
    # (ct, tile, loc) and returns an MLIR tile value.
    #
    # Multi-step ops ("abs", "sigmoid", "not") are handled by delegating to
    # the private wrappers defined below the table.
    return {
        # Simple single-call ops
        "ceil": lambda ct, t, loc: ct.ceil(t, loc=loc),
        "cos": lambda ct, t, loc: ct.cos(t, loc=loc),
        "cosh": lambda ct, t, loc: ct.cosh(t, loc=loc),
        "exp": lambda ct, t, loc: ct.exp(t, loc=loc),
        # exp2 uses the builder's default flush-to-zero behavior.
        "exp2": lambda ct, t, loc: ct.exp2(t, loc=loc),
        # exp10 = ct.pow(10.0 tile, value).
        "exp10": lambda ct, t, loc: _unary_exp10(ct, t, loc),
        "floor": lambda ct, t, loc: ct.floor(t, loc=loc),
        "log": lambda ct, t, loc: ct.log(t, loc=loc),
        "log10": lambda ct, t, loc: ct.log10(t, loc=loc),
        "log1p": lambda ct, t, loc: ct.log1p(t, loc=loc),
        "log2": lambda ct, t, loc: ct.log2(t, loc=loc),
        "rsqrt": lambda ct, t, loc: ct.rsqrt(t, loc=loc),
        "sin": lambda ct, t, loc: ct.sin(t, loc=loc),
        "sinh": lambda ct, t, loc: ct.sinh(t, loc=loc),
        "sqrt": lambda ct, t, loc: ct.sqrt(t, loc=loc),
        "tan": lambda ct, t, loc: ct.tan(t, loc=loc),
        "tanh": lambda ct, t, loc: ct.tanh(t, loc=loc),
        "negf": lambda ct, t, loc: ct.negf(t, loc=loc),
        "neg": lambda ct, t, loc: ct.negf(t, loc=loc),
        # Multi-step ops via wrapper lambdas that call the helpers below
        "abs": lambda ct, t, loc: _unary_abs(ct, t, loc),
        "sigmoid": lambda ct, t, loc: _unary_sigmoid(ct, t, loc),
        "not": lambda ct, t, loc: _unary_not(ct, t, loc),  # i1 logical NOT
        "bitwise_not": lambda ct, t, loc: _unary_bitwise_not(ct, t, loc),  # integer ~x
    }


# Module-level cache — built once, reused across calls.
_UNARY_MATH_TABLE: dict | None = None


def _get_unary_math_table() -> dict:
    global _UNARY_MATH_TABLE
    if _UNARY_MATH_TABLE is None:
        _UNARY_MATH_TABLE = _build_unary_math_table()
    return _UNARY_MATH_TABLE


# Unary helper wrappers (abs / sigmoid / not) — multi-step ops


def _tile_shape(tile: Any) -> list:
    return list(tile.tile_type.shape)


def _dtype_from_mlir_type(ir: Any, mlir_type: Any) -> str:
    """Reverse-map an MLIR type to a dtype name string."""
    if isinstance(mlir_type, ir.F16Type):
        return "float16"
    if isinstance(mlir_type, ir.BF16Type):
        return "bfloat16"
    if isinstance(mlir_type, ir.F32Type):
        return "float32"
    if isinstance(mlir_type, ir.F64Type):
        return "float64"
    if isinstance(mlir_type, ir.FloatTF32Type):
        return "tf32"
    # Float8 / Float4 types — fall back to str-based check
    type_str = str(mlir_type)
    if "f8E4M3FN" in type_str:
        return "float8_e4m3fn"
    if "f8E5M2" in type_str:
        return "float8_e5m2"
    if "f8E8M0FNU" in type_str:
        return "float8_e8m0fnu"
    if "f4E2M1FN" in type_str:
        return "float4_e2m1fn"
    if isinstance(mlir_type, ir.IntegerType):
        bits = mlir_type.width
        if bits == 1:
            return "bool"
        return f"int{bits}"
    raise ValueError(f"_dtype_from_mlir_type: unknown MLIR type {mlir_type!r}")


def _is_float_mlir(ir: Any, mlir_type: Any) -> bool:
    return isinstance(
        mlir_type,
        (
            ir.F16Type,
            ir.BF16Type,
            ir.F32Type,
            ir.F64Type,
            ir.FloatTF32Type,
        ),
    )


def _unary_abs(ct: Any, tile: Any, loc: Any) -> Any:
    """ct.absf for float, ct.absi for int, identity for bool."""
    # Inspect the MLIR element type via tile.element_type (a property).
    elem = tile.element_type
    # Float types have 'width' attribute only sometimes; use string check.
    type_str = str(elem)
    if type_str.startswith("i1"):
        return tile  # bool: abs is identity
    if any(x in type_str for x in ("f16", "bf16", "f32", "f64", "tf32", "f8", "f4")):
        return ct.absf(tile, loc=loc)
    return ct.absi(tile, loc=loc)


def _unary_sigmoid(ct: Any, tile: Any, loc: Any, fast: bool = False) -> Any:
    """1 / (1 + exp(-x)).  ``fast`` uses the approximate reciprocal (fast_math)."""
    shape = _tile_shape(tile)
    elem = tile.element_type
    # Build 1.0 constant of the same dtype as tile
    tile_type_1 = ct.TileType.get(shape, elem)
    one = ct.constant(1.0, tile_type=tile_type_1, loc=loc)
    neg = ct.negf(tile, loc=loc)
    exp_neg = ct.exp(neg, loc=loc)
    denom = ct.add(one, exp_neg, loc=loc)
    if fast:
        return ct.div(one, denom, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX, loc=loc)
    return ct.div(one, denom, loc=loc)


def _unary_not(ct: Any, tile: Any, loc: Any) -> Any:
    """Boolean NOT: xori(tile, 1).

    Correct only for i1 (boolean) tiles, where flipping bit 0 IS the logical
    negation.  For wider integers use ``_unary_bitwise_not``.
    """
    shape = _tile_shape(tile)
    elem = tile.element_type
    one_tile_type = ct.TileType.get(shape, elem)
    one = ct.constant(1, tile_type=one_tile_type, loc=loc)
    return ct.xori(tile, one, loc=loc)


def _unary_bitwise_not(ct: Any, tile: Any, loc: Any) -> Any:
    """Integer bitwise NOT: ~x == xori(x, -1) (all-ones).

    ``"not"`` (``xori(x, 1)``) only flips bit 0 and is correct for i1; a real
    integer ``~x`` must flip every bit, which ``xori`` against the all-ones
    constant (-1 in two's complement) does for any integer width.
    """
    shape = _tile_shape(tile)
    elem = tile.element_type
    all_ones_type = ct.TileType.get(shape, elem)
    all_ones = ct.constant(-1, tile_type=all_ones_type, loc=loc)
    return ct.xori(tile, all_ones, loc=loc)


def _unary_exp10(ct: Any, tile: Any, loc: Any) -> Any:
    """10 ** x via ct.pow(10.0 tile, x)."""
    shape = _tile_shape(tile)
    elem = tile.element_type
    ten_tile_type = ct.TileType.get(shape, elem)
    ten = ct.constant(10.0, tile_type=ten_tile_type, loc=loc)
    return ct.pow(ten, tile, loc=loc)


# _BINARY_OP_TABLE
#   Maps fn_name → lambda(ct, lhs, rhs, loc) → mlir_tile.


def _build_binary_op_table() -> dict:
    # Comparison helpers: ct.cmp is the generic form (handles int and float).
    # It takes (predicate, lhs, rhs, signedness).
    # We import the enums lazily (inside the lambdas) to avoid importing
    # cuda_tile at module load time.
    def _cmp(pred_name, ct, l, r, loc, signedness):
        from cuda_tile._mlir.dialects.cuda_tile_ops import ComparisonPredicates

        pred = getattr(ComparisonPredicates, pred_name)
        return ct.cmp(pred, l, r, signedness, loc=loc)

    # Every lambda takes a uniform (ct, l, r, loc, signedness) signature so
    # `_emit_elementwise` can dispatch without special-casing which fns care
    # about signedness: "max"/"min"/comparisons
    # consult `signedness` (derived from `Elementwise.unsigned`); everything
    # else ignores it (signedness-agnostic in two's complement, or -- like
    # "shr_unsigned" -- already has an explicit unsigned variant fn name).
    return {
        "add": lambda ct, l, r, loc, signedness: ct.add(l, r, loc=loc),
        "sub": lambda ct, l, r, loc, signedness: ct.sub(l, r, loc=loc),
        "mul": lambda ct, l, r, loc, signedness: ct.mul(l, r, loc=loc),
        "div": lambda ct, l, r, loc, signedness: ct.div(l, r, loc=loc),
        "floordiv": lambda ct, l, r, loc, signedness: ct.floordivi(l, r, loc=loc),
        "floormod": lambda ct, l, r, loc, signedness: ct.rem(l, r, loc=loc),
        "max": lambda ct, l, r, loc, signedness: ct.max(l, r, signedness=signedness, loc=loc),
        "min": lambda ct, l, r, loc, signedness: ct.min(l, r, signedness=signedness, loc=loc),
        "pow": lambda ct, l, r, loc, signedness: ct.pow(l, r, loc=loc),
        "atan2": lambda ct, l, r, loc, signedness: ct.atan2(l, r, loc=loc),
        "mulhi": lambda ct, l, r, loc, signedness: ct.mulhii(l, r, loc=loc),
        "andi": lambda ct, l, r, loc, signedness: ct.andi(l, r, loc=loc),
        "ori": lambda ct, l, r, loc, signedness: ct.ori(l, r, loc=loc),
        "xori": lambda ct, l, r, loc, signedness: ct.xori(l, r, loc=loc),
        # comparison ops (produce bool result)
        "eq": lambda ct, l, r, loc, signedness: _cmp("EQUAL", ct, l, r, loc, signedness),
        "ne": lambda ct, l, r, loc, signedness: _cmp("NOT_EQUAL", ct, l, r, loc, signedness),
        "lt": lambda ct, l, r, loc, signedness: _cmp("LESS_THAN", ct, l, r, loc, signedness),
        "le": lambda ct, l, r, loc, signedness: _cmp("LESS_THAN_OR_EQUAL", ct, l, r, loc, signedness),
        "gt": lambda ct, l, r, loc, signedness: _cmp("GREATER_THAN", ct, l, r, loc, signedness),
        "ge": lambda ct, l, r, loc, signedness: _cmp("GREATER_THAN_OR_EQUAL", ct, l, r, loc, signedness),
        # bitwise shift ops (integer only; ct.shli / ct.shri)
        "shl": lambda ct, l, r, loc, signedness: ct.shli(l, r, loc=loc),
        "shr": lambda ct, l, r, loc, signedness: ct.shri(l, r, loc=loc),
        # unsigned right-shift (uint types use Signedness.UNSIGNED) — already
        # has its own explicit fn name selected at the lowering call site.
        "shr_unsigned": lambda ct, l, r, loc, signedness: ct.shri(l, r, signedness=ct.Signedness.UNSIGNED, loc=loc),
    }


_BINARY_OP_TABLE: dict | None = None


def _get_binary_op_table() -> dict:
    global _BINARY_OP_TABLE
    if _BINARY_OP_TABLE is None:
        _BINARY_OP_TABLE = _build_binary_op_table()
    return _BINARY_OP_TABLE


# _align_binary_tiles — ensure shapes and element types match


def _broadcast_source_shape(source: tuple, target: tuple, *, prefix_axes: bool = True) -> tuple:
    """Plan the existing tile broadcast, preserving Parallel's prefix axes."""
    if len(source) < len(target):
        ones = (1,) * (len(target) - len(source))
        source = source + ones if prefix_axes and source == target[: len(source)] else ones + source
    if len(source) != len(target) or any(s != 1 and s != t for s, t in zip(source, target)):
        raise TileIRLoweringError(
            f"cannot broadcast tile shape {source} to {target}; "
            "non-unit extents differ. The semantic lowering must preserve the operand's parallel-axis placement."
        )
    return source


def _binary_result_shape(lhs: tuple, rhs: tuple) -> tuple:
    """Share emission's target selection with typed expression construction."""
    if lhs == rhs:
        return lhs
    target = lhs if len(lhs) > len(rhs) or (len(lhs) == len(rhs) and max(lhs) >= max(rhs)) else rhs
    _broadcast_source_shape(lhs, target)
    _broadcast_source_shape(rhs, target)
    return target


def _select_result_shape(cond: tuple, true: tuple, false: tuple) -> tuple:
    """Select aligns branches first, then broadcasts its condition with leading axes."""
    branches = _binary_result_shape(true, false)
    target = branches or cond
    _broadcast_source_shape(cond, target, prefix_axes=False)
    return target


def _align_binary_tiles(ct: Any, lhs: Any, rhs: Any, ir: Any, loc: Any = None) -> tuple:
    """Ensure lhs and rhs have identical shapes and element types.

    Broadcasting: if one operand is a scalar ``[]`` or has fewer dims than the
    other, broadcast it to the larger shape via ``ct.broadcast``.  This handles
    patterns like ``A_f[i,j] * s[i]`` where A_f is ``[1,256]`` and s is ``[1]``.

    If element types differ, cast rhs to lhs dtype (conservative).
    """
    lhs_shape = _tile_shape(lhs)
    rhs_shape = _tile_shape(rhs)
    if lhs_shape != rhs_shape:
        # Attempt broadcasting: broadcast the smaller-rank tile to the larger shape.
        # cuda_tile.broadcast requires same rank between source and result, so
        # when ranks differ we first reshape to same rank (padding dims of 1),
        # then broadcast.
        def _broadcast_to(tile: Any, target_shape: list) -> Any:
            src_shape = _tile_shape(tile)
            if src_shape == target_shape:
                return tile
            pad = list(_broadcast_source_shape(tuple(src_shape), tuple(target_shape)))
            if pad != src_shape:
                tile = ct.reshape(pad, tile, loc=loc)
            return ct.broadcast(target_shape, tile, loc=loc)

        target_shape = list(_binary_result_shape(tuple(lhs_shape), tuple(rhs_shape)))
        lhs = _broadcast_to(lhs, target_shape)
        rhs = _broadcast_to(rhs, target_shape)
        # After broadcast, shapes must match.
        if _tile_shape(lhs) != _tile_shape(rhs):
            raise ValueError(f"_align_binary_tiles: shape mismatch after broadcast {_tile_shape(lhs)} vs {_tile_shape(rhs)}")
    if lhs.element_type != rhs.element_type:
        # cast rhs to lhs dtype
        rhs_dtype = _dtype_from_mlir_type(ir, rhs.element_type)
        lhs_dtype = _dtype_from_mlir_type(ir, lhs.element_type)
        rhs = _cast_tile(ct, ir, rhs, rhs_dtype, lhs_dtype)
    return lhs, rhs


# _cast_tile — apply the appropriate CUDA Tile IR cast


def _cast_tile(ct: Any, ir: Any, tile: Any, src_dtype: str, tgt_dtype: str, loc: Any = None, src_unsigned: bool = False) -> Any:
    """Convert a tile from ``src_dtype`` to ``tgt_dtype``.

    Maps dtype names to MLIR types and picks ftof / itof / ftoi / exti / trunci.

    Parameters
    ----------
    loc :
        Optional bytecode-compatible MLIR location to attach to the cast op.
    """
    _FLOAT_NAMES = {
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "tf32",
        "float8_e4m3fn",
        "float8_e4m3",  # float8_e4m3 is TVM alias for float8_e4m3fn
        "float8_e5m2",
        "float8_e8m0fnu",
        "float4_e2m1fn",
    }
    _DTYPE_TO_MLIR_ATTR: dict[str, Any] = {
        # We need the element wrapper (dtype class) for cast ops.
        # These are the cuda_tile element-type class names.
        # cuda_tile has no UInt wrappers, so unsigned integer types share the
        # same bit-width class as their signed counterparts (Int4/Int8/
        # Int16/Int32/Int64). Signedness is carried on the operation itself
        # (via Signedness.SIGNED / Signedness.UNSIGNED), not the type wrapper.
        # uint{N} therefore maps to the same wrapper as int{N}.
        "float16": "Float16",
        "bfloat16": "BFloat16",
        "tf32": "TFloat32",
        "float32": "Float32",
        "float64": "Float64",
        "float8_e4m3fn": "Float8E4M3FN",
        "float8_e4m3": "Float8E4M3FN",  # TVM alias without 'fn' suffix
        "float8_e5m2": "Float8E5M2",
        "float8_e8m0fnu": "Float8E8M0FNU",
        "float4_e2m1fn": "Float4E2M1FN",
        "bool": "Bool",
        "int4": "Int4",
        "int8": "Int8",
        "uint8": "Int8",  # unsigned → same-width signed wrapper
        "int16": "Int16",
        "uint16": "Int16",  # unsigned → same-width signed wrapper
        "int32": "Int32",
        "uint32": "Int32",  # unsigned → same-width signed wrapper
        "int64": "Int64",
        "uint64": "Int64",  # unsigned → same-width signed wrapper
    }
    attr_name = _DTYPE_TO_MLIR_ATTR.get(tgt_dtype)
    if attr_name is None:
        raise ValueError(f"_cast_tile: unsupported target dtype {tgt_dtype!r}")

    tgt_wrapper = getattr(ct, attr_name, None)
    if tgt_wrapper is None:
        raise ValueError(f"_cast_tile: cuda_tile has no attribute {attr_name!r}")

    src_is_float = src_dtype in _FLOAT_NAMES
    tgt_is_float = tgt_dtype in _FLOAT_NAMES

    loc_kw: dict = {} if loc is None else {"loc": loc}

    if src_is_float and tgt_is_float:
        return ct.ftof(tgt_wrapper, tile, **loc_kw)
    if not src_is_float and tgt_is_float:
        return ct.itof(tgt_wrapper, tile, **loc_kw)
    if src_is_float and not tgt_is_float:
        return ct.ftoi(tgt_wrapper, tile, **loc_kw)
    # int → int: check widths for exti vs trunci
    src_type = tile.element_type
    tgt_type = _mlir_element_type_from_name(ir, tgt_dtype)
    if hasattr(src_type, "width") and hasattr(tgt_type, "width"):
        if src_type.width < tgt_type.width:
            # Widening. The TileType alias collapses uint{N} -> signless
            # Int{N}, so tile.element_type has lost the source signedness — the
            # caller passes src_unsigned (derived from the lowering-time source
            # dtype name) so we can zero-extend.  Without this, ct.exti defaults
            # to SIGNED and sign-extends, so uint8(200) becomes int(-56).
            ext_kw = dict(loc_kw)
            if src_unsigned or src_dtype.startswith("uint"):
                ext_kw["signedness"] = ct.Signedness.UNSIGNED
            return ct.exti(tgt_wrapper, tile, **ext_kw)
        if src_type.width > tgt_type.width:
            return ct.trunci(tgt_wrapper, tile, **loc_kw)
        return tile  # same width, same type would have been caught earlier
    tgt_tile_type = ct.TileType.get(_tile_shape(tile), tgt_type)
    return ct.bitcast(tgt_tile_type, tile, **loc_kw)


def _mlir_element_type_from_name(ir: Any, dtype_name: str) -> Any:
    """Return an MLIR element type from a dtype name string."""
    _FLOAT_IR = {
        "float16": ir.F16Type.get,
        "bfloat16": ir.BF16Type.get,
        "tf32": ir.FloatTF32Type.get,
        "float32": ir.F32Type.get,
        "float64": ir.F64Type.get,
    }
    if dtype_name in _FLOAT_IR:
        return _FLOAT_IR[dtype_name]()
    _INT_BITS = {
        "bool": 1,
        "int4": 4,
        "int8": 8,
        "uint8": 8,
        "int16": 16,
        "uint16": 16,
        "int32": 32,
        "uint32": 32,
        "int64": 64,
        "uint64": 64,
    }
    if dtype_name in _INT_BITS:
        return ir.IntegerType.get_signless(_INT_BITS[dtype_name])
    raise ValueError(f"_mlir_element_type_from_name: unknown dtype {dtype_name!r}")


# _emit_elementwise — main dispatch for Elementwise.emit_mlir


def _emit_elementwise(op: Any, ctx: Any) -> Any:
    """Emit MLIR for an Elementwise op.

    Dispatches on op.fn:
      - Unary (1 operand): look up in _UNARY_MATH_TABLE.
      - Binary (2 operands): look up in _BINARY_OP_TABLE.
      - 'fma' (3 operands): call ct.fma directly.

    All operands are resolved from ctx.value_map, wrapped as ct.Tile,
    then shape-aligned before calling the relevant CUDA Tile IR primitive.
    """
    ct = ctx.ct
    ir = ctx.ir
    loc = ctx.loc
    fn = op.fn

    # Resolve operand MLIR values from value_map.
    # op.inputs is a tuple of Value objects (named `inputs` rather than
    # `operands` so it does not shadow the inherited TileOp.operands() method).
    operands_raw = op.inputs
    if operands_raw is None or operands_raw == ():
        raise ValueError(f"Elementwise.emit_mlir: op.inputs is empty for fn={fn!r}. Elementwise requires at least 1 operand.")
    tiles = [_as_tile(ctx, ctx.lookup(v)) for v in operands_raw]

    n = len(tiles)

    # FMA (ternary)
    if fn == "fma":
        if n != 3:
            raise ValueError(f"Elementwise 'fma' requires 3 operands, got {n}")
        a, b, c = tiles
        # Align all three
        a, b = _align_binary_tiles(ct, a, b, ir, loc=loc)
        a, c = _align_binary_tiles(ct, a, c, ir, loc=loc)
        a, b = _align_binary_tiles(ct, a, b, ir, loc=loc)
        return ct.fma(a, b, c, loc=loc)

    fast_math = getattr(ctx, "fast_math", False)

    # Unary (1 operand)
    if n == 1:
        if fast_math:
            # Approximate transcendentals (match cuTile / torch's fast paths):
            # the hardware MUFU approximations instead of the full-precision
            # (polynomial / Newton-Raphson-refined) lowerings.
            if fn == "sigmoid":
                return _unary_sigmoid(ct, tiles[0], loc, fast=True)
            if fn == "exp2":
                return ct.exp2(tiles[0], flush_to_zero=True, loc=loc)
            if fn == "tanh":
                # gelu's per-element tanh; default RoundingMode.FULL is a precise
                # polynomial — APPROX is the hardware tanh (matches torch
                # approximate='tanh'). Per-element, so this is the dominant cost.
                return ct.tanh(tiles[0], rounding_mode=ct.RoundingMode.APPROX, loc=loc)
            if fn == "rsqrt":
                return ct.rsqrt(tiles[0], flush_to_zero=True, loc=loc)
        tbl = _get_unary_math_table()
        fn_lambda = tbl.get(fn)
        if fn_lambda is None:
            raise ValueError(f"Elementwise: unknown unary fn {fn!r}. Known: {sorted(tbl.keys())}")
        return fn_lambda(ct, tiles[0], loc)

    # Binary (2 operands)
    if n == 2:
        lhs, rhs = _align_binary_tiles(ct, tiles[0], tiles[1], ir, loc=loc)
        if fast_math and fn == "div":
            # Approximate reciprocal (MUFU.RCP, no Newton-Raphson refinement),
            # matching cuTile's ct.truediv(rounding_mode=APPROX, flush_to_zero=True).
            return ct.div(lhs, rhs, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX, loc=loc)
        tbl = _get_binary_op_table()
        fn_lambda = tbl.get(fn)
        if fn_lambda is None:
            raise ValueError(f"Elementwise: unknown binary fn {fn!r}. Known: {sorted(tbl.keys())}")
        # uint signedness: "max"/"min"/comparisons consult this;
        # every other table entry ignores it (see _build_binary_op_table).
        signedness = ct.Signedness.UNSIGNED if getattr(op, "unsigned", False) else ct.Signedness.SIGNED
        return fn_lambda(ct, lhs, rhs, loc, signedness)

    expected = "Expected 1 (unary), 2 (binary), or 3 (fma)."
    raise ValueError(f"Elementwise.emit_mlir: fn={fn!r} with {n} operands — unsupported arity. {expected}")


# _emit_cast — emit_mlir for Cast


def _emit_cast(op: Any, ctx: Any) -> Any:
    """Emit MLIR for a Cast op.

    Resolves op.src from value_map, determines src dtype from MLIR element
    type, then calls the appropriate CUDA Tile IR cast via _cast_tile.
    """
    ct = ctx.ct
    ir = ctx.ir
    src_tile = _as_tile(ctx, ctx.lookup(op.src))
    src_dtype = _dtype_from_mlir_type(ir, src_tile.element_type)
    tgt_dtype = op.dtype
    if src_dtype == tgt_dtype:
        return src_tile  # identity cast — return as-is
    # ``tir.reinterpret`` preserves bits; a width-changing bitcast is invalid.
    if getattr(op, "bitcast", False):
        tgt_el_type = _mlir_element_type_from_name(ir, tgt_dtype)
        return ct.bitcast(tgt_el_type, src_tile, loc=ctx.loc)
    # The MLIR element type lost uint signedness (uint{N} -> signless
    # Int{N}), so derive the unsigned-ness from the lowering-time source dtype
    # name (op.src_dtype) and pass it explicitly. Do not replace src_dtype
    # outright — the raw TIR name may be e.g. "custom[tfloat32]" which would
    # break the float/int classification in _cast_tile.
    op_src_dtype = (getattr(op, "src_dtype", "") or "").strip()
    src_unsigned = op_src_dtype.startswith("uint")
    return _cast_tile(ct, ir, src_tile, src_dtype, tgt_dtype, loc=ctx.loc, src_unsigned=src_unsigned)


# _emit_select — emit_mlir for Select


def _emit_select(op: Any, ctx: Any) -> Any:
    """Emit MLIR for a Select op via ``ct.select``.

    Resolves cond, true_val, false_val from value_map.

    ct.select requires cond to have the same shape as true/false.
    When cond is a scalar bool (shape [] or [1]) but true/false are shaped tiles,
    broadcast cond to the target shape first.

    Handles the reverse case too — when cond is a shaped tile
    (e.g. tile<16x64xi1> from a causal-mask comparison involving an Iota loop var)
    but true/false are scalars (e.g. 0 and -inf).  In that case broadcast
    true_val and false_val up to cond_shape.  This occurs in NSA's causal mask:
        acc_s[row,col] = T.if_then_else(i_t >= (i_s + col), 0, -T.infinity(...))
    where `col` is a T.Parallel loop var that becomes an Iota tile<16x64xi32>,
    making the condition tile<16x64xi1> while the values are scalar f32.
    """
    ct = ctx.ct
    ir = ctx.ir
    loc = ctx.loc
    cond = _as_tile(ctx, ctx.lookup(op.cond))
    true_val = _as_tile(ctx, ctx.lookup(op.true_val))
    false_val = _as_tile(ctx, ctx.lookup(op.false_val))
    # Align true/false shapes and dtypes
    true_val, false_val = _align_binary_tiles(ct, true_val, false_val, ir, loc=loc)
    target_shape = _tile_shape(true_val)
    cond_shape = _tile_shape(cond)
    _select_result_shape(tuple(cond_shape), tuple(target_shape), tuple(target_shape))
    if cond_shape == target_shape:
        # Already aligned — nothing to do.
        pass
    elif target_shape and cond_shape != target_shape:
        # Case A: cond is scalar/lower-rank, true/false are shaped.
        # Broadcast cond (bool tile) up to target_shape.
        rank = len(target_shape)
        cond_rank = len(cond_shape)
        if cond_rank == 0:
            cond = ct.reshape([1] * rank, cond, loc=loc)
        elif cond_rank < rank:
            pad = [1] * (rank - cond_rank) + list(cond_shape)
            cond = ct.reshape(pad, cond, loc=loc)
        cond = ct.broadcast(target_shape, cond, loc=loc)
    elif cond_shape and not target_shape:
        # Case B: cond is shaped, true/false are scalar.
        # Broadcast true_val and false_val up to cond_shape.
        def _bcast_scalar_to(tile: Any, shape: list) -> Any:
            src = _tile_shape(tile)
            if src == shape:
                return tile
            rank = len(shape)
            src_rank = len(src)
            if src_rank == 0:
                tile = ct.reshape([1] * rank, tile, loc=loc)
            elif src_rank < rank:
                pad = [1] * (rank - src_rank) + list(src)
                tile = ct.reshape(pad, tile, loc=loc)
            return ct.broadcast(shape, tile, loc=loc)

        true_val = _bcast_scalar_to(true_val, list(cond_shape))
        false_val = _bcast_scalar_to(false_val, list(cond_shape))
    return ct.select(cond, true_val, false_val, loc=loc)


# shared buffer load/store helpers


def _load_buffer_tile(ctx: Any, buf_val: Any, loc: Any) -> Any:
    """Load a tile from the full buffer extent (partition index 0 on all dims).

    Shared by _emit_gemm_impl, Reduce.emit_mlir, and Cumsum.emit_mlir.

    SHARED/REGISTER buffers are in ``ctx._tile_map``; return the current tile
    value directly without a TKO load op.  For GLOBAL buffers, reshape from
    the full tile_shape (may be 4D) to the squeezed shape.

    Parameters
    ----------
    ctx :
        EmitContext holding ct, ir, token maps, buffer_map, and tile_map.
    buf_val :
        The TileIR ``Value`` whose type carries shape/dtype/space info and
        which is keyed in either ``ctx._buffer_map`` (GLOBAL) or
        ``ctx._tile_map`` (SHARED/REGISTER).
    loc :
        MLIR location for all emitted ops.

    Returns
    -------
    The loaded ``ct.Tile`` value (with squeezed shape if tile_shape had singleton dims).
    """
    ct = ctx.ct

    # SHARED / REGISTER tile-based buffer: return the current tile value.
    if ctx.is_tile_buffer(buf_val):
        return _as_tile(ctx, ctx.get_tile(buf_val))

    buf_info = ctx.get_buffer_info(buf_val)
    elem_ty = _mlir_element_type(ctx, buf_val.type)
    tile_shape = list(buf_val.type.shape)
    squeezed_shape = _squeeze_shape(tile_shape)
    tok = _ensure_token(ctx, buf_val)
    if buf_info.view is not None:
        partition = ct.make_partition_view(
            buf_info.view,
            tile_shape,
            padding_value=ct.PaddingValue.ZERO,
            loc=loc,
        )
        ndim = len(tile_shape)
        indices = _make_i32_index_tiles(ctx, (0,) * ndim)
        tile, out_tok = ct.load_view_tko(
            view=partition,
            indices=indices,
            input_token=tok,
            return_token=True,
            loc=loc,
        )
        # Reshape from full (possibly 4D) view tile to squeezed shape.
        tile = _reshape_tile_to(ct, tile, squeezed_shape, loc)
    else:
        ptr_shaped = _broadcast_ptr(ct, buf_info.ptr, squeezed_shape, loc=loc)
        tile_ty = ct.TileType.get(squeezed_shape, elem_ty)
        tile, out_tok = ct.load_ptr_tko(
            result=tile_ty,
            source=ptr_shaped,
            input_token=tok,
            return_token=True,
            loc=loc,
        )
    ctx._set_token(buf_val, out_tok)
    return tile


def _store_buffer_tile(ctx: Any, buf_val: Any, tile: Any, loc: Any) -> None:
    """Store a tile back to a buffer (GLOBAL or SHARED/REGISTER).

    Shared by _emit_gemm_impl, Reduce.emit_mlir, and Cumsum.emit_mlir.

    SHARED/REGISTER buffers are in ``ctx._tile_map``; update the tile value
    directly without a TKO store op.  For GLOBAL buffers, reshape the tile to
    the full tile_shape before store_view_tko.

    Parameters
    ----------
    ctx :
        EmitContext holding ct, ir, token maps, buffer_map, and tile_map.
    buf_val :
        The TileIR ``Value`` keyed in either ``ctx._buffer_map`` (GLOBAL) or
        ``ctx._tile_map`` (SHARED/REGISTER).
    tile :
        The ``ct.Tile`` value to store.
    loc :
        MLIR location for all emitted ops.
    """
    ct = ctx.ct

    # SHARED / REGISTER tile-based buffer: update _tile_map.
    if ctx.is_tile_buffer(buf_val):
        ctx.set_tile(buf_val, tile)
        return

    buf_info = ctx.get_buffer_info(buf_val)
    tile_shape = list(buf_val.type.shape)
    squeezed_shape = _squeeze_shape(tile_shape)
    tok = _ensure_token(ctx, buf_val)
    if buf_info.view is not None:
        partition = ct.make_partition_view(
            buf_info.view,
            tile_shape,
            loc=loc,
        )
        ndim = len(tile_shape)
        indices = _make_i32_index_tiles(ctx, (0,) * ndim)
        # Reshape tile to full tile_shape before store_view_tko.
        store_tile = _reshape_tile_to(ct, tile, tile_shape, loc)
        out_tok = ct.store_view_tko(
            tile=store_tile,
            view=partition,
            indices=indices,
            input_token=tok,
            loc=loc,
        )
    else:
        ptr_shaped = _broadcast_ptr(ct, buf_info.ptr, squeezed_shape, loc=loc)
        store_tile = _reshape_tile_to(ct, tile, squeezed_shape, loc)
        out_tok = ct.store_ptr_tko(
            destination=ptr_shaped,
            value=store_tile,
            input_token=tok,
            loc=loc,
        )
    ctx._set_token(buf_val, out_tok)


# _make_gather_scatter_view — build a !cuda_tile.gather_scatter_view + the op
# that creates it (CopyGather / CopyScatter).


def _make_gather_scatter_view(ctx: Any, buf_info: _BufferInfo, tile_shape: tuple, sparse_dim: int, loc: Any) -> Any:
    """Build a CUDA Tile IR gather/scatter view over ``buf_info.view``.

    The Python bindings do not expose a high-level helper for this view, so the
    type and operation are constructed with the registered dialect bindings.
    The raw MLIR value is returned because generic tile-view helpers only accept
    scalar indices.

    Parameters
    ----------
    ctx :
        ``EmitContext`` (needs ``.ir`` and ``.ct_gen``).
    buf_info :
        The ``_BufferInfo`` of the GLOBAL buffer to view (must have a
        ``TensorView`` — ``buf_info.view is not None``).
    tile_shape :
        The gsview's tile shape (one dim per ``buf_info.view`` rank).  For a
        rank-2 buffer this is ``(rows, row_width)``.
    sparse_dim :
        Which tensor_view dimension is gathered/scattered over (0 for the
        row-granularity ``T.tma_gather4`` / ``T.tma_scatter4`` convention).
    loc :
        MLIR location for the emitted op.

    Returns
    -------
    mlir.ir.Value
        The raw gsview SSA value (an ``OpResult``, not wrapped in a
        ``ct.TileView`` subclass — see above for why).
    """
    ir = ctx.ir
    ct_gen = ctx.ct_gen

    if buf_info.view is None:
        raise TileIRLoweringNotImplementedError("_make_gather_scatter_view: buffer has no TensorView (dtype does not support TensorView).")

    tensor_view_str = str(buf_info.view.type)
    _PREFIX = "!cuda_tile."
    if tensor_view_str.startswith(_PREFIX):
        tensor_view_str = tensor_view_str[len(_PREFIX) :]
    tile_shape_str = "x".join(str(int(d)) for d in tile_shape)
    gsview_text = f"!cuda_tile.gather_scatter_view<tile=({tile_shape_str}), {tensor_view_str}, sparse_dim={int(sparse_dim)}>"
    # ir.Type.parse raises an opaque MLIR-internal error (no gsview_text
    # context) on a malformed type string -- e.g. a dtype without a
    # TensorView representation producing garbage in `tensor_view_str`, or a
    # future dialect syntax change. Re-raise with the constructed text so the
    # failure is diagnosable without re-deriving it from a stack trace.
    try:
        gsview_type = ir.Type.parse(gsview_text)
    except Exception as exc:
        raise TileIRLoweringError(
            f"_make_gather_scatter_view: failed to parse gather_scatter_view MLIR type {gsview_text!r}: {exc}"
        ) from exc

    gsv_op = ct_gen.MakeGatherScatterViewOp(gsview_type, buf_info.view, loc=loc)
    return gsv_op.result
