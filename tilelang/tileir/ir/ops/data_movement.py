"""TileIR data-movement ops (copies, loads/stores, fills, views)."""

from __future__ import annotations

import dataclasses
import functools
import operator
from typing import Any

from tilelang.tileir.emission_utils import (
    _as_tile,
    _broadcast_ptr,
    _build_gather_ptrs,
    _cast_tile,
    _dtype_from_mlir_type,
    _ensure_token,
    _load_buffer_tile,
    _make_gather_scatter_view,
    _make_i32_index_tiles,
    _make_tile_view,
    _mlir_element_type,
    _reshape_tile_to,
    _squeeze_shape,
    _store_buffer_tile,
)
from tilelang.tileir.errors import (
    TileIRLoweringError,
    TileIRLoweringNotImplementedError,
    _UnsupportedTileIRNode,
)
from tilelang.tileir.ir.value import Value
from tilelang.tileir.ir.ops._base import (
    Effect,
    TileOp,
    attribute,
    buffer_operand,
    operand,
)


def _copy_hint_kwargs(ctx: Any, latency: Any, allow_tma: Any) -> dict:
    """Build the ``arch``/``allow_tma``/``latency`` kwargs for a copy's
    ``load_view_tko`` / ``store_view_tko`` optimization hint.

    Per-copy ``latency`` / ``allow_tma`` come from
    ``T.copy(..., latency=, disable_tma=)``.  The GLOBAL
    ``TL_DISABLE_TMA_LOWER`` flag (``ctx.disable_tma``), when set, forces
    ``allow_tma=False`` on every copy, OR-ed with the per-copy hint — exactly
    how the CUDA backend honors the global pass config.  The hint requires an
    ``arch`` string; without one the assembler decides, so we omit the hint.
    """
    arch = getattr(ctx, "arch", None)
    # OR the global disable_tma with the per-copy allow_tma.
    effective_allow_tma = allow_tma
    if getattr(ctx, "disable_tma", False):
        effective_allow_tma = False
    if arch is None or (latency is None and effective_allow_tma is None):
        return {}
    kw: dict = {"arch": arch}
    if effective_allow_tma is not None:
        kw["allow_tma"] = effective_allow_tma
    if latency is not None:
        kw["latency"] = latency
    return kw


def _elem_ptrs_for_tile(ctx: Any, buf_val: Any, load_shape: list, idx_dims: list, tile_dims: list, elem_view: bool, loc: Any) -> Any:
    """Per-element pointer tile for a rectangular tile access on a
    ptr-only (no TensorView) buffer: ``base + Σ_k (elem_base_k + iota_k)·stride_k``.

    ``idx_dims`` are per-buffer-dim indices (tile-granular unless
    ``elem_view``); ``tile_dims`` the per-dim tile extents. Dims with extent
    > 1 contribute an iota along the corresponding squeezed result axis.
    """
    ct = ctx.ct
    kinds, vals, axes = [], [], []
    res_axis = 0
    for idx, ext in zip(idx_dims, tile_dims):
        if isinstance(ext, int) and ext > 1:
            if elem_view or (isinstance(idx, int) and idx == 0):
                base = idx
            elif isinstance(idx, int):
                base = idx * ext
            else:
                mul = ct.constant(int(ext), el_type=ct.Int32, loc=loc)
                base = ct.mul(_make_i32_index_tiles(ctx, (idx,))[0], mul, loc=loc)
            kinds.append("iota")
            vals.append(base)
            axes.append(res_axis)
            res_axis += 1
        else:
            kinds.append("const" if isinstance(idx, int) else "scalar")
            vals.append(idx)
            axes.append(-1)
    return _build_gather_ptrs(ctx, buf_val, load_shape, tuple(kinds), tuple(vals), tuple(axes), loc)


def _merge_subtile_into(
    ctx: Any, old_tile: Any, sub_tile: Any, *, dst_region_shape: list, dst_indices: tuple, dst_name: str, loc: Any
) -> Any:
    """Merge a sub-region tile into a full SHARED/REGISTER tile.

    Embeds *sub_tile* (the copied region, e.g. column ``k`` of ``S_shared``)
    into *old_tile* via reshape + broadcast + iota-mask select:
    ``select(axis_pos == index, sub, old)``.  Each partial axis must have
    region extent 1 (a slice-scatter); wider partial spans cannot be placed by
    a broadcast and are rejected loudly.
    """
    ct = ctx.ct
    buf_shape = list(old_tile.tile_type.shape)
    ndim = len(buf_shape)
    region = list(dst_region_shape)
    if len(region) != ndim or len(dst_indices) < ndim:
        raise TileIRLoweringNotImplementedError(
            f"sub-region copy into `{dst_name}`: region {region} / indices {dst_indices} do not match the buffer rank {ndim}."
        )
    for axis, (rd, bd) in enumerate(zip(region, buf_shape)):
        if rd not in (1, bd):
            raise TileIRLoweringNotImplementedError(
                f"sub-region copy into `{dst_name}`: axis {axis} writes a span of "
                f"{rd} of {bd} elements — only single-slice (extent 1) or full-axis "
                "sub-region stores are supported."
            )

    sub = _reshape_tile_to(ct, sub_tile, region, loc)
    sub_b = ct.broadcast(buf_shape, sub, loc=loc) if region != buf_shape else sub

    mask = None
    for axis, (rd, bd) in enumerate(zip(region, buf_shape)):
        if rd == bd:
            continue
        iota_1d = ct.iota(bd, ct.Int32, loc=loc)
        rank_shape = [1] * ndim
        rank_shape[axis] = bd
        iota_t = ct.broadcast(buf_shape, ct.reshape(rank_shape, iota_1d, loc=loc), loc=loc)
        idx_scalar = _make_i32_index_tiles(ctx, (dst_indices[axis],))[0]
        idx_b = ct.broadcast(buf_shape, ct.reshape([1] * ndim, idx_scalar, loc=loc), loc=loc)
        axis_mask = ct.cmp(ct.ComparisonPredicates.EQUAL, iota_t, idx_b, loc=loc)
        mask = axis_mask if mask is None else ct.andi(mask, axis_mask, loc=loc)
    if mask is None:
        # Region equals the buffer shape — plain replacement.
        return sub_b
    return ct.select(mask, sub_b, old_tile, loc=loc)


@dataclasses.dataclass(eq=False)
class GatherLoad(TileOp, opcode="gather_load", effect=Effect.READ):
    """Data-dependent gather from a GLOBAL buffer via a pointer tile.

    Lowers ``KV[b, Indices[b, s, g, i*BI + bi], g, d]``-style reads:
    per-element pointers are built as
    ``ptr[e] = base + Σ_k dim_index_k[e] * stride_k`` and loaded with
    ``load_ptr_tko`` (the dialect's documented "load and gather" form).

    Per BUFFER dim ``k`` the index contribution over ``result_shape`` is
    described by ``dim_kinds[k]`` / ``dim_values[k]`` / ``dim_axes[k]``:

    - ``"const"``  — ``dim_values[k]`` is a Python int.
    - ``"scalar"`` — ``dim_values[k]`` is a scalar (0-d) index Value.
    - ``"iota"``   — ``dim_values[k]`` is the scalar base (int | Value);
                     the index varies as ``base + iota`` along RESULT axis
                     ``dim_axes[k]``.
    - ``"tile"``   — ``dim_values[k]`` is a tile Value of ``result_shape``
                     (e.g. a loaded-and-broadcast ``Indices`` tile).
    """

    src: Any = buffer_operand(effect=Effect.READ)
    result_shape: tuple = attribute(default=())
    dim_kinds: tuple = attribute(default=())
    dim_values: tuple = attribute(default=())
    dim_axes: tuple = attribute(default=())

    def emit_mlir(self, ctx: Any) -> Any:
        ct = ctx.ct
        loc = ctx.loc

        result_shape = list(self.result_shape)
        ptrs = _build_gather_ptrs(ctx, self.src, result_shape, self.dim_kinds, self.dim_values, self.dim_axes, loc)

        elem_ty = _mlir_element_type(ctx, self.src.type)
        tile_type = ct.TileType.get(result_shape, elem_ty)
        tok = _ensure_token(ctx, self.src)
        tile, out_tok = ct.load_ptr_tko(
            result=tile_type,
            source=ptrs,
            input_token=tok,
            return_token=True,
            loc=loc,
        )
        ctx._set_token(self.src, out_tok)
        return tile


@dataclasses.dataclass(eq=False)
class Copy(TileOp, opcode="copy", effect=Effect.READWRITE):
    """Tile-level copy between two memory regions (T.copy / tl.tileop.copy).

    Fields carry the tile shape and per-buffer indices required by the emit
    path.  ``dst_tile_shape`` supports SHARED → 4D GLOBAL copies where the src
    tile shape (2D) differs from the dst region shape (4D, e.g.
    ``(1,64,1,128)``):

    ``tile_shape``      — the src-side tile shape (used for src make_partition_view).
    ``dst_tile_shape``  — the dst-side tile shape (default ``()`` → same as
                          ``tile_shape``).  Set to the full 4D region shape when
                          the dst is a 4D GLOBAL buffer and src is 2D SHARED.
    ``src_indices``     — integer partition indices into src (one per dim, default 0).
    ``dst_indices``     — integer partition indices into dst (one per dim, default 0).
    ``src_elem_view``   — when True, ``src_indices`` are raw ELEMENT offsets and the
                          src view is a unit-stride strided view (misaligned region
                          bases, e.g. varlen ``cu_seqlens[b] + bx*block_M``).
    ``dst_elem_view``   — same for the dst side.
    """

    src: Any = buffer_operand(effect=Effect.READ)
    dst: Any = buffer_operand(effect=Effect.WRITE)
    tile_shape: tuple = attribute(default=())
    dst_tile_shape: tuple = attribute(default=())
    src_indices: tuple = attribute(default=())
    dst_indices: tuple = attribute(default=())
    src_elem_view: bool = attribute(default=False)
    dst_elem_view: bool = attribute(default=False)
    # Static semantic fact used to omit unnecessary load padding.
    src_in_bounds: bool = attribute(default=False)
    # per-copy load/store optimization hints.
    latency: Any = attribute(default=None)  # int | None
    allow_tma: Any = attribute(default=None)  # bool | None

    def emit_mlir(self, ctx: Any) -> None:
        """Lower Copy to MLIR: load-from-src then store-to-dst.

        Handles all four src/dst combinations:
          GLOBAL → GLOBAL: load_view_tko/load_ptr_tko then store_view_tko/store_ptr_tko.
          GLOBAL → SHARED: load from GLOBAL view/ptr, update SHARED tile in _tile_map.
          SHARED → GLOBAL: read from _tile_map, store to GLOBAL view/ptr.
          SHARED → SHARED: read from _tile_map, write to _tile_map (tile-to-tile).

        High-rank handling: when ``tile_shape`` has singleton dims (e.g. a 4D
        slice ``(1,64,1,128)`` from a 4D BSHD buffer), ``make_partition_view`` must
        receive the full 4D shape for TMA addressing, but ``load_view_tko`` returns a
        4D tile which must be reshaped to the target (squeezed) shape before use.
        Symmetrically, a 2D tile being stored to a GLOBAL 4D buffer must be reshaped
        to the 4D ``tile_shape`` before ``store_view_tko``.
        """
        ct = ctx.ct
        ir = ctx.ir
        loc = ctx.loc

        src_is_tile = ctx.is_tile_buffer(self.src)
        dst_is_tile = ctx.is_tile_buffer(self.dst)

        # src_tile_shape: shape for src make_partition_view (may be 4D for BSHD GLOBAL)
        src_tile_shape = list(self.tile_shape)
        # dst_tile_shape: shape for dst make_partition_view. If dst_tile_shape is
        # set, use it; otherwise fall back to src_tile_shape. This handles
        # SHARED(2D) → GLOBAL(4D) copies where the dst region has a 4D shape.
        dst_tile_shape = list(self.dst_tile_shape) if self.dst_tile_shape else src_tile_shape

        src_elem_ty = _mlir_element_type(ctx, self.src.type)
        dst_elem_ty = _mlir_element_type(ctx, self.dst.type)

        src_ndim = len(src_tile_shape)
        dst_ndim = len(dst_tile_shape)
        src_idx_tuple = self.src_indices if self.src_indices else (0,) * src_ndim
        dst_idx_tuple = self.dst_indices if self.dst_indices else (0,) * dst_ndim

        # Squeezed tile shape: remove size-1 dims that exist purely for TMA
        # addressing in 4D BSHD buffers (e.g. (1,64,1,128) → (64,128)).
        # This is the shape that SHARED buffers and compute ops (Gemm) expect.
        squeezed_shape = _squeeze_shape(src_tile_shape)

        # Load phase
        if src_is_tile:
            # SHARED / REGISTER source: return the current tile value.
            # Partial-slice path: when tile_shape is smaller than the
            # buffer's full alloc shape (e.g. acc_o[:4, :] from a [64,128]
            # buffer), use ct.extract to pull out the slice.  The extract
            # indices are the tile-level coordinates from src_idx_tuple
            # (base / slice_extent, already computed by _compute_view_indices).
            full_tile = _as_tile(ctx, ctx.get_tile(self.src))
            buf_shape = list(self.src.type.shape)
            if src_tile_shape and src_tile_shape != buf_shape:
                # Element counts differ — extract the sub-tile.
                # ct.extract requires source and result of the SAME rank, so
                # extract at full rank (e.g. (1,128) row-slice of a (4,128)
                # fragment) and reshape to the squeezed shape afterwards.
                stored_shape = list(full_tile.tile_type.shape)
                extract_shape = list(src_tile_shape)
                if len(extract_shape) != len(stored_shape):
                    raise TileIRLoweringNotImplementedError(
                        f"Copy from `{self.src.name}`: slice rank {len(extract_shape)} does not "
                        f"match the stored tile rank {len(stored_shape)} ({extract_shape} vs {stored_shape})."
                    )
                full_rank_type = ct.TileType.get(extract_shape, _mlir_element_type(ctx, self.src.type))
                extract_indices = _make_i32_index_tiles(ctx, src_idx_tuple)
                loaded_tile = ct.extract(full_rank_type, full_tile, extract_indices, loc=loc)
                loaded_tile = _reshape_tile_to(ct, _as_tile(ctx, loaded_tile), squeezed_shape, loc)
            else:
                loaded_tile = full_tile
        else:
            # GLOBAL source: use TKO load.
            src_info = ctx.get_buffer_info(self.src)
            src_tok = _ensure_token(ctx, self.src)
            if src_info.view is not None:
                src_partition = _make_tile_view(
                    ct,
                    src_info.view,
                    src_tile_shape,
                    elem_view=self.src_elem_view,
                    padding_value=None if self.src_in_bounds else ct.PaddingValue.ZERO,
                    loc=loc,
                )
                src_indices = _make_i32_index_tiles(ctx, src_idx_tuple)
                # Pass per-copy optimization hints to the load TKO.
                _src_hint_kw = _copy_hint_kwargs(ctx, self.latency, self.allow_tma)
                loaded_tile, src_out_tok = ct.load_view_tko(
                    view=src_partition,
                    indices=src_indices,
                    input_token=src_tok,
                    return_token=True,
                    loc=loc,
                    **_src_hint_kw,
                )
                # Reshape from 4D view tile to squeezed shape.
                # load_view_tko returns tile with shape == src_tile_shape (may be 4D);
                # downstream consumers (SHARED buffer, Gemm) expect squeezed_shape.
                loaded_tile = _reshape_tile_to(ct, loaded_tile, squeezed_shape, loc)
            else:
                src_ptr_shaped = _elem_ptrs_for_tile(
                    ctx, self.src, squeezed_shape, list(src_idx_tuple), list(src_tile_shape), self.src_elem_view, loc
                )
                squeezed_tile_type = ct.TileType.get(squeezed_shape, src_elem_ty)
                loaded_tile, src_out_tok = ct.load_ptr_tko(
                    result=squeezed_tile_type,
                    source=src_ptr_shaped,
                    input_token=src_tok,
                    return_token=True,
                    loc=loc,
                )
            ctx._set_token(self.src, src_out_tok)

        # Cast if src/dst dtypes differ (use _cast_tile helper, no ct.cast).
        if src_elem_ty != dst_elem_ty:
            src_dtype = _dtype_from_mlir_type(ir, src_elem_ty)
            dst_dtype = _dtype_from_mlir_type(ir, dst_elem_ty)
            loaded_tile = _cast_tile(ct, ir, _as_tile(ctx, loaded_tile), src_dtype, dst_dtype, loc=loc)

        # Store phase
        # dst_squeezed: the squeezed shape expected by the dst (SHARED or ptr paths).
        dst_squeezed_shape = _squeeze_shape(dst_tile_shape)

        if dst_is_tile:
            # SHARED / REGISTER destination: update the tile in _tile_map.
            # The loaded tile has squeezed_shape (e.g. (16,64) for a
            # 3D src region (1,16,64)). The SHARED buffer's declared shape may be
            # larger (e.g. [64,64]).  Only reshape if the element counts match.
            dst_shape = list(self.dst.type.shape)
            cur_tile = _as_tile(ctx, loaded_tile)
            cur_elems = functools.reduce(operator.mul, cur_tile.tile_type.shape, 1)
            dst_elems = functools.reduce(operator.mul, dst_shape, 1)
            if cur_elems == dst_elems:
                store_tile = _reshape_tile_to(ct, cur_tile, dst_shape, loc)
            else:
                # The copy fills only a SUB-REGION of the SHARED/REGISTER tile
                # (e.g. ``T.copy(scores_max, S_shared[:, k])`` writes column k).
                # Replacing the tile-map entry with the smaller tile corrupts
                # the buffer (and breaks loop carries); merge into the existing
                # tile via an iota-mask select instead.
                old_tile = _as_tile(ctx, ctx.get_tile(self.dst))
                store_tile = _merge_subtile_into(
                    ctx,
                    old_tile,
                    cur_tile,
                    dst_region_shape=dst_tile_shape,
                    dst_indices=dst_idx_tuple,
                    dst_name=self.dst.name,
                    loc=loc,
                )
            ctx.set_tile(self.dst, store_tile)
        else:
            # GLOBAL destination: use TKO store.
            dst_info = ctx.get_buffer_info(self.dst)
            dst_tok = _ensure_token(ctx, self.dst)
            if dst_info.view is not None:
                dst_partition = _make_tile_view(
                    ct,
                    dst_info.view,
                    dst_tile_shape,
                    elem_view=self.dst_elem_view,
                    loc=loc,
                )
                dst_indices = _make_i32_index_tiles(ctx, dst_idx_tuple)
                # Reshape tile back to full dst_tile_shape before store_view_tko.
                store_tile = _reshape_tile_to(ct, _as_tile(ctx, loaded_tile), dst_tile_shape, loc)
                # Pass per-copy optimization hints to the store TKO.
                _dst_hint_kw = _copy_hint_kwargs(ctx, self.latency, self.allow_tma)
                out_tok = ct.store_view_tko(
                    tile=store_tile,
                    view=dst_partition,
                    indices=dst_indices,
                    input_token=dst_tok,
                    loc=loc,
                    **_dst_hint_kw,
                )
            else:
                dst_ptr_shaped = _elem_ptrs_for_tile(
                    ctx, self.dst, dst_squeezed_shape, list(dst_idx_tuple), list(dst_tile_shape), self.dst_elem_view, loc
                )
                store_tile = _reshape_tile_to(ct, _as_tile(ctx, loaded_tile), dst_squeezed_shape, loc)
                out_tok = ct.store_ptr_tko(
                    destination=dst_ptr_shaped,
                    value=store_tile,
                    input_token=dst_tok,
                    loc=loc,
                )
            ctx._set_token(self.dst, out_tok)

        return None  # side-effect only


@dataclasses.dataclass(eq=False)
class TmaCopy(TileOp, opcode="tma_copy", effect=Effect.READWRITE):
    """TMA-accelerated copy (tl.tileop.tma_copy).

    Semantically identical to ``Copy`` but the hint layer may lower it via
    TMA hardware.  Same field extensions as ``Copy``.
    """

    src: Any = buffer_operand(effect=Effect.READ)
    dst: Any = buffer_operand(effect=Effect.WRITE)
    tile_shape: tuple = attribute(default=())
    dst_tile_shape: tuple = attribute(default=())
    src_indices: tuple = attribute(default=())
    dst_indices: tuple = attribute(default=())

    src_elem_view: bool = attribute(default=False)
    dst_elem_view: bool = attribute(default=False)

    def emit_mlir(self, ctx: Any) -> None:
        """Use Copy's indexing, partial-tile updates, and token handling."""
        return Copy(
            src=self.src,
            dst=self.dst,
            tile_shape=self.tile_shape,
            dst_tile_shape=self.dst_tile_shape,
            src_indices=self.src_indices,
            dst_indices=self.dst_indices,
            src_elem_view=self.src_elem_view,
            dst_elem_view=self.dst_elem_view,
        ).emit_mlir(ctx)


@dataclasses.dataclass(eq=False)
class TransposeCopy(TileOp, opcode="transpose_copy", effect=Effect.READWRITE):
    """Rank-2 tile transpose (``T.transpose`` / ``tl.tileop.transpose``):
    ``dst[j, i] = src[i, j]``.

    Two lowering paths, chosen by ``src``'s memory space:

    SHARED / REGISTER src
        ``tile = ctx.get_tile(src)`` -> ``ct.permute(tile, [1, 0])`` -> store
        to ``dst`` (``_store_buffer_tile`` handles both SHARED/REGISTER and
        GLOBAL destinations).

    GLOBAL src
        A full-tile transposed load via ``cuda_tile.make_strided_view``. Under
        the CUDA Tile IR strided-view contract, a view whose
        declared ``tile_shape``/``traversal_strides`` equal the OUTPUT
        (post-transpose) shape ``dst_shape = (N, M)`` and whose
        ``dim_map=[1, 0]`` (declared tile dim 0 -> tensor_view dim 1, tile
        dim 1 -> tensor_view dim 0) reads exactly ``src.T`` in one shot at
        index ``(0, 0)`` — no separate ``ct.permute`` needed on this path.
        ``cuda_tile.strided_view`` requires power-of-two tile dims (verified
        dialect constraint), so this path only supports src/dst shapes that
        satisfy that.

    Only a full-buffer, rank-2 transpose is supported on EITHER path: the
    SHARED/REGISTER path moves the whole resident tile, and the GLOBAL path's
    strided_view/partition_view is always read/written at a hardcoded
    ``(0, 0)`` index.  ``_lower_transpose`` (the ``tl.tileop.transpose``
    lowering handler in ``lowering/sem_to_ir/tile_ops.py``) is the
    authoritative site that proves both the src and dst regions are their
    buffer's full extent, at a statically-zero offset, before constructing
    this op — a partial or dynamically-sized region raises there rather than
    silently transposing the wrong window.  This class's ``emit_mlir`` keeps
    a secondary defensive re-check of that invariant.

    ``src_shape`` / ``dst_shape`` — the buffers' declared rank-2 shapes
    ``(M, N)`` / ``(N, M)``, verified equal to the full buffer extent by
    ``_lower_transpose``.
    """

    src: Any = buffer_operand(effect=Effect.READ)
    dst: Any = buffer_operand(effect=Effect.WRITE)
    src_shape: tuple = attribute(default=())
    dst_shape: tuple = attribute(default=())

    def emit_mlir(self, ctx: Any) -> None:
        """Lower TransposeCopy to MLIR via ``ct.permute`` (SHARED/REGISTER
        src) or ``ct.make_strided_view`` + ``ct.load_view_tko`` (GLOBAL src).
        """
        ct = ctx.ct
        loc = ctx.loc

        # Secondary defensive check.  The authoritative validation lives in
        # ``_lower_transpose`` / ``_require_full_buffer_region``
        # (tilelang/tileir/lowering/sem_to_ir/tile_ops.py), which proves
        # BOTH src and dst regions are the buffer's full extent, rank-2, at
        # a statically-zero offset, before ever constructing a
        # ``TransposeCopy``.  Both emission paths below assume that: the
        # SHARED/REGISTER path moves the WHOLE resident tile
        # (``get_tile``/``set_tile``), and the GLOBAL path reads/writes the
        # WHOLE ``TensorView`` at a hardcoded ``(0, 0)`` index.  A partial
        # region on either side — most dangerously a partial GLOBAL dst,
        # which ``_store_buffer_tile`` would otherwise overwrite at the
        # wrong window — would silently touch the wrong data.  This
        # assertion should be unreachable via the ``T.transpose`` frontend
        # (the lowering handler raises first); it only guards a
        # ``TransposeCopy`` constructed directly, bypassing that handler.
        if list(self.src.type.shape) != list(self.src_shape) or len(self.src_shape) != 2:
            raise TileIRLoweringError(
                "tl.tileop.transpose: src_shape attribute "
                f"{list(self.src_shape)} does not match the declared rank-2 src buffer "
                f"shape {list(self.src.type.shape)} — transpose only supports a "
                "full-buffer rank-2 region."
            )
        if list(self.dst.type.shape) != list(self.dst_shape) or len(self.dst_shape) != 2:
            raise TileIRLoweringError(
                "tl.tileop.transpose: dst_shape attribute "
                f"{list(self.dst_shape)} does not match the declared rank-2 dst buffer "
                f"shape {list(self.dst.type.shape)} — transpose only supports a "
                "full-buffer rank-2 region."
            )

        if ctx.is_tile_buffer(self.src):
            # SHARED / REGISTER source: permute the resident tile.
            src_tile = _load_buffer_tile(ctx, self.src, loc)
            transposed = ct.permute(src_tile, [1, 0], loc=loc)
            _store_buffer_tile(ctx, self.dst, transposed, loc)
            return None

        # GLOBAL source: full-tile transposed load via a strided_view.
        src_info = ctx.get_buffer_info(self.src)
        if src_info.view is None:
            raise TileIRLoweringError(
                "tl.tileop.transpose: GLOBAL src has no TensorView (dtype without "
                "TensorView support) — the strided_view transpose path requires one."
            )

        dst_shape = list(self.dst_shape)

        for dim in dst_shape:
            if dim <= 0 or (dim & (dim - 1)) != 0:
                raise TileIRLoweringError(
                    "tl.tileop.transpose: cuda_tile.strided_view requires power-of-two "
                    f"tile dimensions (StridedViewType dialect constraint); got tile "
                    f"shape {dst_shape}."
                )

        strided_view = ct.make_strided_view(
            src_info.view,
            dst_shape,
            dst_shape,
            dim_map=[1, 0],
            loc=loc,
        )
        src_tok = _ensure_token(ctx, self.src)
        indices = _make_i32_index_tiles(ctx, (0, 0))
        loaded_tile, src_out_tok = ct.load_view_tko(
            view=strided_view,
            indices=indices,
            input_token=src_tok,
            return_token=True,
            loc=loc,
        )
        ctx._set_token(self.src, src_out_tok)

        _store_buffer_tile(ctx, self.dst, loaded_tile, loc)
        return None


def _cat_index_tile(ct: Any, pieces: list, loc: Any) -> Any:
    """Build a rank-1 ``[len(pieces)]`` tile from rank-0 scalar pieces via
    reshape-to-``[1]`` + pairwise tree-``ct.cat`` (the template used by
    ``DecodeI4``/``DecodeI2`` in ``misc.py``)."""
    reshaped = [ct.reshape([1], p, loc=loc) for p in pieces]
    while len(reshaped) > 1:
        next_level = []
        for i in range(0, len(reshaped), 2):
            if i + 1 < len(reshaped):
                next_level.append(ct.cat(reshaped[i], reshaped[i + 1], 0, loc=loc))
            else:
                next_level.append(reshaped[i])
        reshaped = next_level
    return reshaped[0]


@dataclasses.dataclass(eq=False)
class CopyGather(TileOp, opcode="gather4", effect=Effect.READWRITE):
    """Row-granularity 4-row gather copy (``T.tma_gather4`` /
    ``tl.tileop.copy`` with ``annotation.is_gather4``).

    ``dst[i, :] = src[row_indices[i], col : col + K_box]`` for ``i in range(4)``.

    ``src``         — GLOBAL source buffer, rank-2 ``(n, k)``, viewed through
                       a ``cuda_tile.gather_scatter_view`` (``sparse_dim=0``)
                       so the gather indices select entire rows.
    ``dst``         — SHARED destination buffer, rank-2 ``(4, K_box)`` — also
                       the gsview's ``tile_shape``.
    ``row_indices`` — 4 scalar (rank-0) int32 ``Value``s, the rows of ``src``
                       to gather.
    ``col``         — scalar (rank-0) int32 ``Value`` giving the column
                       OFFSET of the ``K_box``-wide box within each gathered
                       row (``None`` → constant 0, i.e. columns
                       ``[0, K_box)``).

    src and dst must share a dtype (no cast path, unlike Copy). Capability
    note: general index-BUFFER gather (an arbitrary-length runtime index
    buffer, as opposed to these 4 literal/dynamic scalar rows) awaits a
    universal TileLang API.

    Out-of-range row indices or col windows (col + K_box > buffer columns)
    are undefined behavior — no bounds check is emitted. Duplicate row
    indices in gather4 are safe (a pure read is not a race).
    """

    src: Any = buffer_operand(effect=Effect.READ)
    dst: Any = buffer_operand(effect=Effect.WRITE)
    row_indices: Any = operand(default=())
    col: Any = operand(default=None)

    def emit_mlir(self, ctx: Any) -> None:
        """Gather four rows from ``src`` and store the resulting tile."""
        ct = ctx.ct
        ct_gen = ctx.ct_gen
        ir = ctx.ir
        loc = ctx.loc

        dst_shape = tuple(int(d) for d in self.dst.type.shape)

        src_info = ctx.get_buffer_info(self.src)
        gsview = _make_gather_scatter_view(ctx, src_info, dst_shape, sparse_dim=0, loc=loc)

        row_pieces = [_as_tile(ctx, ctx.lookup(v)) for v in self.row_indices]
        index_tile = _cat_index_tile(ct, row_pieces, loc)

        i32 = ir.IntegerType.get_signless(32)
        if self.col is not None:
            col_tile = _as_tile(ctx, ctx.lookup(self.col))
        else:
            col_tile = ct.constant(0, tile_type=ct.TileType.get([], i32), loc=loc)

        elem_ty = _mlir_element_type(ctx, self.src.type)
        loaded_tile_type = ct.TileType.get(list(dst_shape), elem_ty)
        result_token_type = ct.TokenType.get()
        sem_attr = ct.get_memory_ordering_semantics_attr(ct.MemoryOrderingSemantics.WEAK)

        src_tok = _ensure_token(ctx, self.src)
        load_op = ct_gen.LoadViewTkoOp(
            tile=loaded_tile_type,
            result_token=result_token_type,
            memory_ordering_semantics=sem_attr,
            inbounds=ir.DenseBoolArrayAttr.get([False, False]),
            view=gsview,
            index=[index_tile, col_tile],
            token=src_tok,
            loc=loc,
        )
        loaded_tile = ct.Tile(load_op.tile, load_op.tile.type)
        ctx._set_token(self.src, ct.Token(load_op.result_token))

        _store_buffer_tile(ctx, self.dst, loaded_tile, loc)
        return None  # side-effect only


@dataclasses.dataclass(eq=False)
class CopyScatter(TileOp, opcode="scatter4", effect=Effect.READWRITE):
    """Row-granularity 4-row scatter copy (``T.tma_scatter4`` /
    ``tl.tileop.copy`` with ``annotation.is_scatter4``).

    ``dst[row_indices[i], col : col + K_box] = src[i, :]`` for ``i in range(4)``.

    ``src``         — SHARED source buffer, rank-2 ``(4, K_box)`` — also the
                       gsview's ``tile_shape``.
    ``dst``         — GLOBAL destination buffer, rank-2 ``(n, k)``, viewed
                       through a ``cuda_tile.gather_scatter_view``
                       (``sparse_dim=0``) so the scatter indices select
                       entire rows.
    ``row_indices`` — 4 scalar (rank-0) int32 ``Value``s, the rows of ``dst``
                       to scatter into.
    ``col``         — scalar (rank-0) int32 ``Value`` giving the column
                       OFFSET of the ``K_box``-wide box within each scattered
                       row (``None`` → constant 0, i.e. columns
                       ``[0, K_box)``).

    src and dst must share a dtype (no cast path, unlike Copy). Capability
    note: general index-BUFFER scatter (an arbitrary-length runtime index
    buffer, as opposed to these 4 literal/dynamic scalar rows) awaits a
    universal TileLang API.

    Out-of-range row indices or col windows (col + K_box > buffer columns)
    are undefined behavior — no bounds check is emitted. Duplicate row
    indices in scatter4 are a data race; the surviving row is unspecified.
    """

    src: Any = buffer_operand(effect=Effect.READ)
    dst: Any = buffer_operand(effect=Effect.WRITE)
    row_indices: Any = operand(default=())
    col: Any = operand(default=None)

    def emit_mlir(self, ctx: Any) -> None:
        """Scatter four rows from ``src`` into ``dst``."""
        ct = ctx.ct
        ct_gen = ctx.ct_gen
        ir = ctx.ir
        loc = ctx.loc

        src_shape = tuple(int(d) for d in self.src.type.shape)

        src_tile = _load_buffer_tile(ctx, self.src, loc)
        row_pieces = [_as_tile(ctx, ctx.lookup(v)) for v in self.row_indices]
        index_tile = _cat_index_tile(ct, row_pieces, loc)

        dst_info = ctx.get_buffer_info(self.dst)
        gsview = _make_gather_scatter_view(ctx, dst_info, src_shape, sparse_dim=0, loc=loc)

        i32 = ir.IntegerType.get_signless(32)
        if self.col is not None:
            col_tile = _as_tile(ctx, ctx.lookup(self.col))
        else:
            col_tile = ct.constant(0, tile_type=ct.TileType.get([], i32), loc=loc)
        sem_attr = ct.get_memory_ordering_semantics_attr(ct.MemoryOrderingSemantics.WEAK)

        dst_tok = _ensure_token(ctx, self.dst)
        store_op = ct_gen.StoreViewTkoOp(
            memory_ordering_semantics=sem_attr,
            inbounds=ir.DenseBoolArrayAttr.get([False, False]),
            tile=src_tile,
            view=gsview,
            index=[index_tile, col_tile],
            token=dst_tok,
            results=[ct.TokenType.get()],
            loc=loc,
        )
        ctx._set_token(self.dst, ct.Token(store_op.results[0]))
        return None  # side-effect only


@dataclasses.dataclass(eq=False)
class Load(TileOp, opcode="load", effect=Effect.READ):
    """Load a tile from a memory buffer into registers.

    ``tile_shape`` — the shape of the tile to load.
    ``indices``    — integer partition indices (one per buffer dim).
    ``elem_view``  — when True, ``indices`` are raw ELEMENT offsets and the src
                     view is a unit-stride strided view (misaligned bases,
                     e.g. varlen cu_seqlens).
    """

    src: Any = buffer_operand(effect=Effect.READ)
    tile_shape: tuple = attribute(default=())
    indices: tuple = attribute(default=())
    elem_view: bool = attribute(default=False)
    # Reshape the loaded tile to this shape before returning (same element
    # count). Used for linearized reads like ``h[v // 32, v % 32]`` under a
    # 1D T.Parallel(1024): the whole (32,32) tile viewed as (1024,).
    reshape_to: tuple = attribute(default=())

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower Load to MLIR: load_view_tko or load_ptr_tko.

        Returns the MLIR tile value so emit_module can bind it to results[0].
        SHARED/REGISTER buffers are in _tile_map; return directly.
        A 4D view-loaded tile is reshaped to its squeezed shape.
        """
        if self.reshape_to:
            _result = self._emit_mlir_inner(ctx)
            return _reshape_tile_to(ctx.ct, _as_tile(ctx, _result), list(self.reshape_to), ctx.loc)
        return self._emit_mlir_inner(ctx)

    def _emit_mlir_inner(self, ctx: Any) -> Any:
        ct = ctx.ct
        loc = ctx.loc

        # SHARED/REGISTER tile-based buffer: return the current tile value.
        # When tile_shape=() AND indices are present,
        # emit a per-element indexed gather via ct.extract so that
        # B_shared[vi, vj] (where vi/vj depend on threadIdx.x) correctly reads
        # the element at the runtime-computed position instead of the whole tile.
        if ctx.is_tile_buffer(self.src):
            full_tile = _as_tile(ctx, ctx.get_tile(self.src))
            if self.elem_view:
                # A flat slice can cross an extract partition boundary (e.g.
                # elements [8:24] of a 32-element tile). Split into aligned
                # power-of-two pieces, then concatenate in source order.
                if len(self.tile_shape) != 1 or len(self.indices) != 1 or not isinstance(self.indices[0], int):
                    raise _UnsupportedTileIRNode("register element-view loads require a constant one-dimensional slice")
                start, size = self.indices[0], self.tile_shape[0]
                if len(full_tile.tile_type.shape) != 1 or start < 0 or start + size > full_tile.tile_type.shape[0]:
                    raise _UnsupportedTileIRNode(
                        f"register element-view slice {self.src.name}[{start}:{start + size}] is outside source tile {full_tile.tile_type.shape}"
                    )

                def extract_slice(offset, extent):
                    if offset % extent == 0:
                        ty = ct.TileType.get([extent], full_tile.element_type)
                        return ct.extract(ty, full_tile, [ct.constant(offset // extent, ct.Int32, loc=loc)], loc=loc)
                    half = extent // 2
                    return ct.cat(extract_slice(offset, half), extract_slice(offset + half, half), 0, loc=loc)

                return extract_slice(start, size)
            is_scalar_load = not bool(self.tile_shape)
            if is_scalar_load and self.indices:
                # Indexed scalar gather: ct.extract([1]*ndim, full_tile, idx_tiles)
                # then reshape [1]*ndim → [] (scalar).
                ndim = len(self.indices)
                extract_shape = [1] * ndim
                elem_ty = _mlir_element_type(ctx, self.src.type)
                result_tile_type = ct.TileType.get(extract_shape, elem_ty)

                # ct.extract requires rank-0 (scalar) indices.
                # When indices are rank-1 (e.g. derived from threadIdx.x which is bound
                # to ct.iota(128, Int32) = tile<128xi32>), the SIMT per-thread gather
                # pattern cannot be expressed in the collective tile model.
                # Check raw values before _make_i32_index_tiles because that helper
                # scalarizes partition indices by extracting element zero.
                def _raw_idx_rank(idx: Any) -> int:
                    try:
                        if isinstance(idx, Value):
                            idx_tile = _as_tile(ctx, ctx.lookup(idx))
                            return len(list(idx_tile.tile_type.shape))
                    except KeyError:
                        pass  # not in the value map → not a tile → rank 0 (scalar)
                    return 0

                if any(_raw_idx_rank(idx) > 0 for idx in self.indices):
                    raise _UnsupportedTileIRNode(
                        "per-thread SIMT gather (rank-1 / threadIdx-dependent index into a "
                        "tile) is not expressible in the CUDA Tile IR execution model"
                    )
                idx_tiles = _make_i32_index_tiles(ctx, self.indices)
                elem_tile = ct.extract(result_tile_type, full_tile, idx_tiles, loc=loc)
                return _reshape_tile_to(ct, elem_tile, [], loc)

            # Sub-tile slice: tile_shape smaller than the buffer — e.g.
            # ``frag[k, j]`` with a serial-loop ``k`` (tile_shape=[1, N],
            # indices=[k, 0]) or a range-clamped read ``acc[i, j]`` under
            # ``if i < 4`` (tile_shape=[4, N], indices=[0, 0]). Extract the
            # slice and squeeze the size-1 sliced dims; returning the full
            # tile would broaden the consumer (element-count mismatch at the
            # store reshape). Full-tile loads keep the fast aliasing path.
            _buf_shape = list(self.src.type.shape)
            if self.tile_shape and list(self.tile_shape) != _buf_shape:
                elem_ty = _mlir_element_type(ctx, self.src.type)
                extract_ty = ct.TileType.get(list(self.tile_shape), elem_ty)
                idx_tiles = _make_i32_index_tiles(ctx, self.indices)
                sub = ct.extract(extract_ty, full_tile, idx_tiles, loc=loc)
                squeezed = _squeeze_shape(list(self.tile_shape))
                if squeezed != list(self.tile_shape):
                    sub = _reshape_tile_to(ct, sub, squeezed, loc)
                return sub
            return full_tile

        buf_info = ctx.get_buffer_info(self.src)
        elem_ty = _mlir_element_type(ctx, self.src.type)
        tile_shape = list(self.tile_shape)
        squeezed_shape = _squeeze_shape(tile_shape)

        tok = _ensure_token(ctx, self.src)

        # Scalar loads (tile_shape=[]) from a GLOBAL buffer with a
        # TensorView must skip make_partition_view (rank-0 tile ≠ view rank-N).
        # Use the ptr-based path: broadcast the ptr to shape [1], load, then reshape
        # back to [] so downstream ops (e.g. _make_i32_index_tiles, Elementwise)
        # see a scalar tile<i32> not tile<1xi32>.
        is_scalar_load = not bool(tile_shape)  # True when tile_shape == [] == ()

        # Indexed scalar load: scalar load with explicit indices — use
        # the TensorView path with [1]*ndim tile_shape and the explicit index, then
        # reshape the result tile<1x...xdtype> → tile<dtype> (scalar).
        # This handles `eid = block_expert[bx] % N` where block_expert is a GLOBAL
        # 1D tensor and bx is a runtime block index.
        if is_scalar_load and self.indices and buf_info.view is not None:
            ndim = len(self.indices)
            scalar_tile_shape = [1] * ndim
            partition = _make_tile_view(
                ct,
                buf_info.view,
                scalar_tile_shape,
                elem_view=self.elem_view,
                padding_value=ct.PaddingValue.ZERO,
                loc=loc,
            )
            indices = _make_i32_index_tiles(ctx, self.indices)
            tile_raw, out_tok = ct.load_view_tko(
                view=partition,
                indices=indices,
                input_token=tok,
                return_token=True,
                loc=loc,
            )
            # Reshape tile<1x...xdtype> → tile<dtype> (scalar rank-0 tile).
            tile = _reshape_tile_to(ct, tile_raw, [], loc)
            ctx._set_token(self.src, out_tok)
            return tile

        if buf_info.view is not None and not is_scalar_load:
            partition = _make_tile_view(
                ct,
                buf_info.view,
                tile_shape,
                elem_view=self.elem_view,
                padding_value=ct.PaddingValue.ZERO,
                loc=loc,
            )
            ndim = len(tile_shape)
            indices_tuple = self.indices if self.indices else (0,) * ndim
            indices = _make_i32_index_tiles(ctx, indices_tuple)
            tile, out_tok = ct.load_view_tko(
                view=partition,
                indices=indices,
                input_token=tok,
                return_token=True,
                loc=loc,
            )
            # Reshape from full tile_shape (may be 4D) to squeezed shape.
            tile = _reshape_tile_to(ct, tile, squeezed_shape, loc)
        else:
            # No TensorView (int4 buffers, alloca scratch) or a scalar load
            # (tile_shape=[]). Load through per-element pointers (base +
            # per-dim iota/offset) — a plain base-pointer broadcast makes
            # every lane read element 0.
            load_shape = squeezed_shape if squeezed_shape else [1]
            _idx_dims = list(self.indices) if self.indices else [0] * (buf_info.ndim or 1)
            _tshape = list(tile_shape) if tile_shape else [1] * len(_idx_dims)
            ptr_shaped = _elem_ptrs_for_tile(ctx, self.src, load_shape, _idx_dims, _tshape, self.elem_view, loc)
            load_tile_type = ct.TileType.get(load_shape, elem_ty)
            tile, out_tok = ct.load_ptr_tko(
                result=load_tile_type,
                source=ptr_shaped,
                input_token=tok,
                return_token=True,
                loc=loc,
            )
            # Reshape tile<1xi32> → tile<i32> for scalar loads so index/arithmetic
            # ops receive a rank-0 tile (scalar).
            if is_scalar_load:
                tile = _reshape_tile_to(ct, tile, [], loc)

        ctx._set_token(self.src, out_tok)
        return tile  # bound to results[0] by emit_module


@dataclasses.dataclass(eq=False)
class Store(TileOp, opcode="store", effect=Effect.WRITE):
    """Store a tile from registers to a memory buffer.

    ``tile_shape`` — the shape of the tile to store.
    ``indices``    — integer partition indices (one per buffer dim).
    ``mask``       — optional tile-wide boolean Value (tile<Nxi1>).  When set,
                     the stored value is wrapped with ``ct.select(mask, val,
                     zeros)`` so that out-of-bounds lanes write zero.  Used for
                     masked parallel stores (``if idx < n`` inside T.Parallel).
    """

    dst: Any = buffer_operand(effect=Effect.WRITE)
    val: Any = operand()
    tile_shape: tuple = attribute(default=())
    indices: tuple = attribute(default=())
    mask: Any = operand(default=None)
    # When True, ``indices`` are raw ELEMENT offsets and the dst view is a
    # unit-stride strided view (misaligned bases, e.g. varlen cu_seqlens).
    elem_view: bool = attribute(default=False)
    # Axis permutation applied to ``val`` before storing: dst axis k reads
    # val axis ``val_perm[k]``. Set for transposed parallel stores like
    # ``B_t[i_k, i_s] = B[i_s, i_k]`` (a reshape would scramble the data).
    val_perm: tuple = attribute(default=())

    def emit_mlir(self, ctx: Any) -> None:
        """Lower Store to MLIR: store_view_tko or store_ptr_tko.

        SHARED/REGISTER destinations update _tile_map.  GLOBAL view stores
        reshape the tile to full tile_shape before store_view_tko.
        """
        ct = ctx.ct
        loc = ctx.loc

        tile_shape = list(self.tile_shape)
        squeezed_shape = _squeeze_shape(tile_shape)

        # Look up the tile to store from the value_map and wrap if needed.
        tile = _as_tile(ctx, ctx.lookup(self.val))

        # Transposed parallel store: permute the value so dst axis k reads
        # val axis val_perm[k] (a plain reshape would scramble the data).
        if self.val_perm and list(self.val_perm) != list(range(len(self.val_perm))):
            tile = ct.permute(tile, list(self.val_perm), loc=loc)

        if self.tile_shape and not list(tile.tile_type.shape):
            # A scalar RHS fills the selected parallel region in either memory
            # space. Reshaping alone would incorrectly change its element count.
            tile = ct.broadcast(tile_shape, ct.reshape([1] * len(tile_shape), tile, loc=loc), loc=loc)

        # Masked store to GLOBAL: must use a TRUE predicated store — masked
        # lanes are not written. A select(mask, val, zeros) view-store would
        # write ZERO to masked lanes, clobbering valid data whenever those
        # lanes alias live elements (varlen batch neighbours, or a
        # conditional ``if oob: Logits[...] = -inf`` whose masked lanes are
        # the VALID logits). store_view_tko has no mask — go through
        # per-element pointers + store_ptr_tko(mask=...). Partition-granular
        # indices are converted to element offsets (idx * tile_extent).
        if self.mask is not None and not ctx.is_tile_buffer(self.dst):
            self._emit_masked_elem_ptr_store(ctx, tile)
            return None

        # SHARED/REGISTER tile-based destination: update _tile_map.
        # When tile_shape=() AND indices are present,
        # emit a per-element indexed scatter via iota+cmp+select so that
        # B_local[v] = ... (where v depends on the loop variable / threadIdx.x)
        # updates exactly the element at the runtime-computed position rather than
        # overwriting the whole tile, via ct.select(mask, broadcast, old_tile).
        if ctx.is_tile_buffer(self.dst):
            is_scalar_store = not bool(self.tile_shape)
            if is_scalar_store and self.indices:
                # ct.iota+cmp+select scatter requires scalar (rank-0) indices.
                # A threadIdx-derived rank-1 tile cannot select one scalar element
                # in the collective tile model and must fail before scalarization.
                def _idx_rank_store(idx_val: Any) -> int:
                    try:
                        if isinstance(idx_val, Value):
                            mlir_v = ctx.lookup(idx_val)
                            mlir_t = _as_tile(ctx, mlir_v)
                            return len(list(mlir_t.tile_type.shape))
                    except KeyError:
                        pass  # not in the value map → not a tile → rank 0 (scalar)
                    return 0

                if any(_idx_rank_store(idx) > 0 for idx in self.indices):
                    raise _UnsupportedTileIRNode(
                        "per-thread SIMT scatter (rank-1 / threadIdx-dependent index into a "
                        "tile) is not expressible in the CUDA Tile IR execution model"
                    )
                # Scatter a scalar value into a single element of the tile.
                old_tile = _as_tile(ctx, ctx.get_tile(self.dst))
                buf_shape = list(self.dst.type.shape)
                ndim = len(buf_shape)
                # Build a combined element-wise boolean mask: position == idx for each axis.
                mask = None
                for axis, (dim_size, idx) in enumerate(zip(buf_shape, self.indices)):
                    # Iota for this axis: broadcast to full buf_shape.
                    axis_iota_1d = ct.iota(dim_size, ct.Int32, loc=loc)
                    if ndim == 1:
                        iota_tile = axis_iota_1d
                    else:
                        rank_shape = [1] * ndim
                        rank_shape[axis] = dim_size
                        iota_tile = ct.reshape(rank_shape, axis_iota_1d, loc=loc)
                        iota_tile = ct.broadcast(buf_shape, iota_tile, loc=loc)
                    # Build scalar idx tile (shape=[] → broadcast to buf_shape).
                    idx_tiles = _make_i32_index_tiles(ctx, (idx,))
                    idx_scalar = idx_tiles[0]  # tile<i32> scalar
                    idx_scalar_reshaped = ct.reshape([1] * ndim, idx_scalar, loc=loc) if ndim > 1 else idx_scalar
                    idx_bcast = (
                        ct.broadcast(buf_shape, idx_scalar_reshaped, loc=loc)
                        if ndim > 1
                        else ct.broadcast(buf_shape, ct.reshape([1], idx_scalar, loc=loc), loc=loc)
                    )
                    axis_mask = ct.cmp(
                        ct.ComparisonPredicates.EQUAL,
                        iota_tile,
                        idx_bcast,
                        loc=loc,
                    )
                    if mask is None:
                        mask = axis_mask
                    else:
                        mask = ct.andi(mask, axis_mask, loc=loc)
                # Broadcast the scalar value to the full tile shape.
                scalar_reshaped = ct.reshape([1] * ndim, tile, loc=loc)
                new_val_bcast = ct.broadcast(buf_shape, scalar_reshaped, loc=loc)
                # Merge: select(mask, new_val, old_tile).
                new_tile = ct.select(mask, new_val_bcast, old_tile, loc=loc)
                ctx.set_tile(self.dst, new_tile)
                return None
            # Masked REGISTER store — use select(mask, val, old_tile)
            # so that a parallel-body `if cond: alloc_var = x` only updates the
            # elements where cond is True, leaving other elements unchanged.
            # This is needed for `alloc_var` inside T.Parallel with if/else:
            #   if cond: value = 1.0 else: value = 0.0
            # lowers as two masked REGISTER stores; without select the second
            # (else) store with mask=None overwrites the whole tile with 0.0.
            if self.mask is not None:
                # Reconcile the value and destination with the mask shape by
                # broadcasting scalar-like tiles. A genuine shape mismatch is
                # rejected by ``ct.select``.
                mask_tile = _as_tile(ctx, ctx.lookup(self.mask))
                old_tile = _as_tile(ctx, ctx.get_tile(self.dst))
                mask_shape = list(mask_tile.tile_type.shape)

                def _broadcast_to_mask(t: Any) -> Any:
                    ts = list(t.tile_type.shape)
                    if ts == mask_shape:
                        return t
                    numel = 1
                    for d in ts:
                        numel *= d
                    if numel == 1:  # scalar-like ([], [1], [1, 1], …): reshape rank + broadcast
                        reshaped = ct.reshape([1] * len(mask_shape), t, loc=loc)
                        return ct.broadcast(mask_shape, reshaped, loc=loc)
                    return t  # leave as-is; a genuine shape mismatch raises in ct.select

                tile = _broadcast_to_mask(tile)
                old_tile = _broadcast_to_mask(old_tile)
                new_tile = ct.select(mask_tile, tile, old_tile, loc=loc)
                ctx.set_tile(self.dst, new_tile)
                return None
            ctx.set_tile(self.dst, tile)
            return None

        buf_info = ctx.get_buffer_info(self.dst)
        tok = _ensure_token(ctx, self.dst)

        is_scalar_store = not bool(tile_shape)

        # Scalar store to GLOBAL: mirror Load.emit_mlir's
        # scalar-load path.  When tile_shape=() (scalar store) with explicit
        # indices and a TensorView is present, use a [1]*ndim partition view
        # so make_partition_view sees the correct rank, then reshape the
        # scalar tile to [1]*ndim before store_view_tko and update the token.
        if is_scalar_store and self.indices and buf_info.view is not None:
            ndim = buf_info.ndim
            scalar_tile_shape = [1] * ndim
            partition = _make_tile_view(
                ct,
                buf_info.view,
                scalar_tile_shape,
                elem_view=self.elem_view,
                loc=loc,
            )
            indices = _make_i32_index_tiles(ctx, self.indices)
            # Reshape scalar tile<elem> → tile<1x...x1xelem>
            store_tile = _reshape_tile_to(ct, tile, scalar_tile_shape, loc)
            out_tok = ct.store_view_tko(
                tile=store_tile,
                view=partition,
                indices=indices,
                input_token=tok,
                loc=loc,
            )
            ctx._set_token(self.dst, out_tok)
            return None

        if buf_info.view is not None:
            partition = _make_tile_view(
                ct,
                buf_info.view,
                tile_shape,
                elem_view=self.elem_view,
                loc=loc,
            )
            ndim = len(tile_shape)
            indices_tuple = self.indices if self.indices else (0,) * ndim
            indices = _make_i32_index_tiles(ctx, indices_tuple)
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
            # No TensorView (alloca scratch / int4): store through per-element
            # pointers so every lane addresses its own element.
            _store_shape = squeezed_shape if squeezed_shape else [1]
            _idx_dims = list(self.indices) if self.indices else [0] * (buf_info.ndim or 1)
            _tshape = list(tile_shape) if tile_shape else [1] * len(_idx_dims)
            ptr_shaped = _elem_ptrs_for_tile(ctx, self.dst, _store_shape, _idx_dims, _tshape, self.elem_view, loc)
            store_tile = _reshape_tile_to(ct, tile, _store_shape, loc)
            out_tok = ct.store_ptr_tko(
                destination=ptr_shaped,
                value=store_tile,
                input_token=tok,
                loc=loc,
            )

        ctx._set_token(self.dst, out_tok)
        return None  # side-effect only

    def _emit_masked_elem_ptr_store(self, ctx: Any, tile: Any) -> None:
        """Masked store via per-element pointers.

        Builds ``ptrs[e] = base + Σ_k (elem_base_k + iota_k[e]) * stride_k``
        and emits ``store_ptr_tko(ptrs, value, mask=mask)``. Masked lanes are
        not written because a select-zero view-store would clobber live aliased
        elements).  ``indices`` are element offsets when ``elem_view`` is set,
        tile-granular partition indices otherwise (converted here via
        ``idx * tile_extent``).
        """
        ct = ctx.ct
        loc = ctx.loc

        buf_info = ctx.get_buffer_info(self.dst)
        tok = _ensure_token(ctx, self.dst)

        tile_shape = list(self.tile_shape)
        ndim = len(tile_shape)
        # Per-dim strides come from _BufferInfo.stride_tiles — scalar i32
        # tiles carrying either static constants or the runtime ABI stride
        # args, so dynamically-shaped buffers work too.
        stride_tiles = list(getattr(buf_info, "stride_tiles", ()) or ())
        if len(stride_tiles) != ndim or len(self.indices) != ndim:
            raise TileIRLoweringNotImplementedError(
                f"masked element-offset store to `{self.dst.name}`: rank mismatch "
                f"(tile {ndim}, strides {len(stride_tiles)}, indices {len(self.indices)})."
            )
        squeezed_shape = _squeeze_shape(tile_shape)

        # offsets[e] = Σ_k (elem_base_k + iota_k[e]) * stride_k, built at full
        # rank then reshaped to the squeezed shape.
        offsets = None
        for axis, (ext, stride_tile_scalar) in enumerate(zip(tile_shape, stride_tiles)):
            idx_scalar = _make_i32_index_tiles(ctx, (self.indices[axis],))[0]  # 0-d i32
            if not self.elem_view and ext > 1:
                # Tile-granular partition index → element base (idx * extent).
                ext_scalar = ct.constant(int(ext), el_type=ct.Int32, loc=loc)
                idx_scalar = ct.mul(idx_scalar, ext_scalar, loc=loc)
            idx_full = ct.broadcast(tile_shape, ct.reshape([1] * ndim, idx_scalar, loc=loc), loc=loc)
            if ext > 1:
                iota_1d = ct.iota(ext, ct.Int32, loc=loc)
                rank_shape = [1] * ndim
                rank_shape[axis] = ext
                iota_full = ct.broadcast(tile_shape, ct.reshape(rank_shape, iota_1d, loc=loc), loc=loc)
                dim_idx = ct.add(idx_full, iota_full, loc=loc)
            else:
                dim_idx = idx_full
            stride_tile = ct.broadcast(tile_shape, ct.reshape([1] * ndim, _as_tile(ctx, stride_tile_scalar), loc=loc), loc=loc)
            contrib = ct.mul(dim_idx, stride_tile, loc=loc)
            offsets = contrib if offsets is None else ct.add(offsets, contrib, loc=loc)
        offsets = _reshape_tile_to(ct, _as_tile(ctx, offsets), squeezed_shape, loc)

        ptr_base = _broadcast_ptr(ct, buf_info.ptr, squeezed_shape, loc=loc)
        ptrs = ct._offset(ptr_base, offsets, loc=loc)

        mask_tile = _as_tile(ctx, ctx.lookup(self.mask))
        # A scalar-like value (e.g. ``Logits[bx, idx] = -inf`` under a SIMT
        # thread binding) broadcasts to the store shape; other shapes reshape.
        _tshape = list(tile.tile_type.shape)
        _tnumel = 1
        for _d in _tshape:
            _tnumel *= _d
        if _tnumel == 1 and squeezed_shape and _tshape != squeezed_shape:
            store_tile = ct.broadcast(squeezed_shape, ct.reshape([1] * len(squeezed_shape), tile, loc=loc), loc=loc)
        else:
            store_tile = _reshape_tile_to(ct, tile, squeezed_shape, loc)
        if list(mask_tile.tile_type.shape) != squeezed_shape:
            raise TileIRLoweringNotImplementedError(
                f"masked element-offset store to `{self.dst.name}`: mask shape "
                f"{list(mask_tile.tile_type.shape)} != tile shape {squeezed_shape}."
            )
        out_tok = ct.store_ptr_tko(
            destination=ptrs,
            value=store_tile,
            mask=mask_tile,
            input_token=tok,
            loc=loc,
        )
        ctx._set_token(self.dst, out_tok)


@dataclasses.dataclass(eq=False)
class Fill(TileOp, opcode="fill", effect=Effect.WRITE):
    """Fill a tile buffer with a scalar constant (tl.tileop.fill).

    ``tile_shape`` — the shape of the tile to fill (required for ct.constant).
    ``value``      — the scalar fill value (moved from operand to attribute so
                     it can be a Python float/int literal rather than an SSA
                     Value; emit uses it directly in ct.constant).
    """

    dst: Any = buffer_operand(effect=Effect.WRITE)
    value: Any = attribute(default=0.0)
    tile_shape: tuple = attribute(default=())

    def emit_mlir(self, ctx: Any) -> None:
        """Lower Fill to MLIR: ct.constant tile then store via TensorView.

        No result is bound (Fill is side-effect only).
        Fill always targets partition 0 on all dims (no per-indices field).
        SHARED/REGISTER tile-map destinations update _tile_map; GLOBAL view
        stores reshape the filled tile to full tile_shape before store_view_tko.
        """
        ct = ctx.ct
        loc = ctx.loc

        elem_ty = _mlir_element_type(ctx, self.dst.type)
        tile_shape = list(self.tile_shape)
        squeezed_shape = _squeeze_shape(tile_shape)
        # Constant is always created in squeezed shape; reshaping to full tile_shape
        # happens just before store_view_tko if needed.
        squeezed_tile_type = ct.TileType.get(squeezed_shape, elem_ty)

        # Build the constant fill tile (squeezed shape)
        filled = ct.constant(self.value, tile_type=squeezed_tile_type, loc=loc)

        # SHARED/REGISTER tile-based destination: update _tile_map.
        if ctx.is_tile_buffer(self.dst):
            dst_shape = list(self.dst.type.shape)
            ctx.set_tile(self.dst, _reshape_tile_to(ct, filled, dst_shape, loc))
            return None

        buf_info = ctx.get_buffer_info(self.dst)

        if buf_info.view is not None:
            # Partition-view store path (preferred, mirrors tile_region.py)
            partition = ct.make_partition_view(
                buf_info.view,
                tile_shape,
                padding_value=ct.PaddingValue.ZERO,
                loc=loc,
            )
            # Fill always targets partition zero on every dimension
            ndim = len(tile_shape)
            indices = _make_i32_index_tiles(ctx, (0,) * ndim)
            # Reshape to full tile_shape before store_view_tko.
            store_tile = _reshape_tile_to(ct, filled, tile_shape, loc)
            tok = _ensure_token(ctx, self.dst)
            out_tok = ct.store_view_tko(
                tile=store_tile,
                view=partition,
                indices=indices,
                input_token=tok,
                loc=loc,
            )
            ctx._set_token(self.dst, out_tok)
        else:
            # Pointer store path (alloca scratch / dtypes without TensorView):
            # per-element pointers — a base broadcast would write element 0 only.
            _tshape = list(tile_shape) if tile_shape else [1]
            ptr_shaped = _elem_ptrs_for_tile(
                ctx, self.dst, squeezed_shape if squeezed_shape else [1], [0] * len(_tshape), _tshape, False, loc
            )
            tok = _ensure_token(ctx, self.dst)
            out_tok = ct.store_ptr_tko(
                destination=ptr_shaped,
                value=filled,
                input_token=tok,
                loc=loc,
            )
            ctx._set_token(self.dst, out_tok)

        return None  # side-effect only


@dataclasses.dataclass(eq=False)
class PartitionView(TileOp, opcode="partition_view", effect=Effect.READ):
    """Create a partition view of a global tensor for TMA indexing.

    Effect is READ: ``PartitionView`` itself only calls
    ``ct.make_partition_view`` on ``src`` and does not write memory. The
    actual write happens in a *downstream* ``Store`` / ``Copy`` that consumes
    this view's result, and that op carries the WRITE effect.  The explicit
    ``buffer_operand(effect=Effect.READ)`` declaration below records that role.

    ``tile_shape`` — the tile shape passed to make_partition_view.
    """

    src: Any = buffer_operand(effect=Effect.READ)
    tile_shape: tuple = attribute(default=())

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower PartitionView to MLIR: ``ct.make_partition_view``.

        Returns the MLIR PartitionView object so emit_module binds it to
        results[0].
        """
        ct = ctx.ct
        loc = ctx.loc

        buf_info = ctx.get_buffer_info(self.src)
        tile_shape = list(self.tile_shape)

        if buf_info.view is None:
            raise NotImplementedError(
                f"PartitionView.emit_mlir: buffer {self.src.name!r} has no TensorView (dtype does not support TensorView)."
            )

        partition = ct.make_partition_view(
            buf_info.view,
            tile_shape,
            padding_value=ct.PaddingValue.ZERO,
            loc=loc,
        )
        return partition  # bound to results[0]
