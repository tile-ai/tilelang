"""TileIR atomic memory ops."""

from __future__ import annotations

import dataclasses
from typing import Any

from tilelang.tileir.ir.ops._base import (
    Effect,
    TileOp,
    attribute,
    buffer_operand,
    operand,
)


@dataclasses.dataclass(eq=False)
class AtomicRMW(TileOp, opcode="atomic_rmw", effect=Effect.READWRITE):
    """Atomic read-modify-write on a global buffer (tl.tileop.atomic{add,max,min}).

    ``kind`` is one of ``"add"``, ``"max"``, ``"min"``.
    ``memory_order`` is an integer (0=RELAXED, 2=ACQUIRE, 3=RELEASE, 4=ACQ_REL).
    ``return_prev`` — when True, emit_mlir returns the previous value tile.

    Handles tile-level RMW, parallel-loop RMW, and the return-value path.

    Emit strategy:
    - RELAXED + dst has TensorView → ``ct.atomic_red_view_tko`` (partition-view path)
    - Otherwise → ``ct.atomic_rmw_tko`` with pointer + memory ordering
    """

    dst: Any = buffer_operand(effect=Effect.READWRITE)
    val: Any = operand()
    kind: str = attribute()
    memory_order: int = attribute(default=0)  # 0=RELAXED, 2=ACQUIRE, 3=RELEASE, 4=ACQ_REL
    return_prev: bool = attribute(default=False)
    # Partition indices into dst (one per buffer dim; int or scalar Value),
    # mirroring Copy.dst_indices.  Empty tuple → partition (0, ..., 0).
    dst_indices: tuple = attribute(default=())
    # Full-rank tile shape for the dst partition view.  Empty tuple → use the
    # val tile's shape (rank must then equal the dst buffer rank).
    tile_shape: tuple = attribute(default=())
    # When True, ``dst_indices`` are raw ELEMENT offsets and the dst view is a
    # unit-stride strided view (misaligned bases, e.g. varlen cu_seqlens).
    elem_view: bool = attribute(default=False)
    # Gather-form dst (data-dependent scatter, e.g. ``dKV[b, Indices[...], g,
    # d] += v``): per-dim index specs as in GatherLoad. When ``gather_dim_kinds``
    # is non-empty, ``dst_indices``/``tile_shape``/``elem_view`` are unused and
    # the atomic goes through per-element pointers (atomic_rmw_tko).
    gather_dim_kinds: tuple = attribute(default=())
    gather_dim_values: tuple = attribute(default=())
    gather_dim_axes: tuple = attribute(default=())

    @staticmethod
    def _memory_ordering(ct: Any, order: int) -> Any:
        """Map integer memory order to ct.MemoryOrderingSemantics."""
        from tilelang.tileir.errors import TileIRLoweringNotImplementedError

        if order == 0:
            return ct.MemoryOrderingSemantics.RELAXED
        if order == 2:
            return ct.MemoryOrderingSemantics.ACQUIRE
        if order == 3:
            return ct.MemoryOrderingSemantics.RELEASE
        if order == 4:
            return ct.MemoryOrderingSemantics.ACQ_REL
        raise TileIRLoweringNotImplementedError(
            f"AtomicRMW: unsupported memory order id {order}; supported: relaxed=0, acquire=2, release=3, acq_rel=4."
        )

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower AtomicRMW to MLIR.

        For RELAXED ordering when dst has a TensorView: emits
        ``ct.atomic_red_view_tko`` (partition-view path, no pointer arithmetic).

        Otherwise emits ``ct.atomic_rmw_tko`` with a broadcast pointer and
        the requested memory ordering.

        When ``self.return_prev`` is True, returns the previous-value tile so
        emit_module can bind it to results[0].  Otherwise returns None.
        """
        from tilelang.tileir.emission_utils import (
            _ensure_token,
            _as_tile,
            _mlir_element_type,
            _broadcast_ptr,
            _make_i32_index_tiles,
            _make_tile_view,
            _reshape_tile_to,
        )

        ct = ctx.ct
        loc = ctx.loc

        buf_info = ctx.get_buffer_info(self.dst)
        # Select AtomicRMWMode from kind + dtype.
        dtype_name = self.dst.type.dtype.name
        _UNSIGNED = {"uint8", "uint16", "uint32", "uint64"}
        _FLOAT = {
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "tf32",
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e8m0fnu",
            "float4_e2m1fn",
        }
        kind = self.kind
        if kind == "add":
            mode = ct.AtomicRMWMode.ADDF if dtype_name in _FLOAT else ct.AtomicRMWMode.ADD
        elif kind == "max":
            mode = ct.AtomicRMWMode.UMAX if dtype_name in _UNSIGNED else ct.AtomicRMWMode.MAX
        elif kind == "min":
            mode = ct.AtomicRMWMode.UMIN if dtype_name in _UNSIGNED else ct.AtomicRMWMode.MIN
        else:
            from tilelang.tileir.errors import _UnsupportedTileIRNode

            raise _UnsupportedTileIRNode(f"AtomicRMW.emit_mlir: unsupported kind {kind!r}. Supported: 'add', 'max', 'min'.")

        ordering = self._memory_ordering(ct, self.memory_order)

        # Resolve the val tile.  val may be:
        #   - a SHARED/REGISTER buffer (in _tile_map)
        #   - a GLOBAL buffer (in _buffer_map) — load via view with partition [0]
        #   - a REGISTER result Value (in value_map)
        if ctx.is_tile_buffer(self.val):
            val_tile = _as_tile(ctx, ctx.get_tile(self.val))
        elif self.val in ctx._buffer_map:
            # GLOBAL val: load with load_view_tko (view path) or load_ptr_tko (ptr path).
            val_info = ctx._buffer_map[self.val]
            ndim = val_info.ndim
            dst_tile_shape = list(getattr(self.dst.type, "shape", (1,) * buf_info.ndim))
            val_load_tok = _ensure_token(ctx, self.val)
            if val_info.view is not None:
                # view path: make_partition_view then load_view_tko([0,...,0]).
                val_partition = ct.make_partition_view(val_info.view, dst_tile_shape, loc=loc)
                partition_indices = _make_i32_index_tiles(ctx, (0,) * ndim)
                val_tile_raw, _ = ct.load_view_tko(
                    val_partition,
                    partition_indices,
                    memory_ordering_semantics=ct.MemoryOrderingSemantics.RELAXED,
                    memory_scope=ct.MemoryScope.DEVICE,
                    input_token=val_load_tok,
                    return_token=True,
                    loc=loc,
                )
                val_tile = val_tile_raw
            else:
                # Pointer-based load.
                ptr_shaped = _broadcast_ptr(ct, val_info.ptr, dst_tile_shape, loc=loc)
                elem_ty = _mlir_element_type(ctx, self.val.type)
                tile_type = ct.TileType.get(dst_tile_shape, elem_ty)
                val_tile_raw, _ = ct.load_ptr_tko(
                    tile_type,
                    ptr_shaped,
                    memory_ordering_semantics=ct.MemoryOrderingSemantics.RELAXED,
                    memory_scope=ct.MemoryScope.DEVICE,
                    input_token=val_load_tok,
                    return_token=True,
                    loc=loc,
                )
                val_tile = val_tile_raw
        else:
            val_tile = _as_tile(ctx, ctx.lookup(self.val))
        tile_shape = list(val_tile.tile_type.shape)

        tok = _ensure_token(ctx, self.dst)

        # Gather-form dst: per-element pointers + atomic_rmw_tko.
        if self.gather_dim_kinds:
            from tilelang.tileir.emission_utils import _build_gather_ptrs

            # The scatter domain comes from tile_shape (the parallel extents);
            # a scalar val (e.g. the histogram's `+= 1`) broadcasts to it.
            result_shape = list(self.tile_shape) if self.tile_shape else list(val_tile.tile_type.shape)
            if list(val_tile.tile_type.shape) != result_shape:
                _numel = 1
                for _d in val_tile.tile_type.shape:
                    _numel *= _d
                if _numel == 1:
                    val_tile = ct.broadcast(
                        result_shape,
                        ct.reshape([1] * len(result_shape), val_tile, loc=loc),
                        loc=loc,
                    )
            ptrs = _build_gather_ptrs(ctx, self.dst, result_shape, self.gather_dim_kinds, self.gather_dim_values, self.gather_dim_axes, loc)
            # SIMT-demoted scratch (alloca global) is only shared within the
            # tile block — tile-block scope lets the assembler use cheaper
            # SM-local atomics (REDG.SM vs REDG.GPU).
            _scope = ct.MemoryScope.TL_BLK if self.dst in getattr(ctx, "alloca_values", ()) else ct.MemoryScope.DEVICE
            prev_val, out_tok = ct.atomic_rmw_tko(
                ordering,
                _scope,
                ptrs,
                mode,
                val_tile,
                input_token=tok,
                return_token=True,
                loc=loc,
            )
            ctx._set_token(self.dst, out_tok)
            return prev_val if self.return_prev else None

        idx_tuple = self.dst_indices if self.dst_indices else (0,) * buf_info.ndim
        has_nonzero_idx = any(not (isinstance(i, int) and i == 0) for i in idx_tuple)
        view_tile_shape = list(self.tile_shape) if self.tile_shape else tile_shape

        # Try atomic_red_view_tko (RELAXED + dst has TensorView)
        if ordering is ct.MemoryOrderingSemantics.RELAXED and buf_info.view is not None and not self.return_prev:
            from tilelang.tileir.errors import TileIRLoweringNotImplementedError as _NotImpl

            if len(view_tile_shape) != buf_info.ndim:
                raise _NotImpl(
                    f"AtomicRMW: tile shape rank {len(view_tile_shape)} does not match dst "
                    f"buffer rank {buf_info.ndim} for `{self.dst.name}`; cannot build a partition view."
                )
            # Build a partition/strided view from the TensorView with the tile shape.
            partition = _make_tile_view(ct, buf_info.view, view_tile_shape, elem_view=self.elem_view, loc=loc)
            partition_indices = _make_i32_index_tiles(ctx, idx_tuple)
            store_tile = _reshape_tile_to(ct, _as_tile(ctx, val_tile), view_tile_shape, loc)
            out_tok = ct.atomic_red_view_tko(
                partition,
                partition_indices,
                mode,
                store_tile,
                memory_ordering_semantics=ordering,
                memory_scope=ct.MemoryScope.DEVICE,
                input_token=tok,
                loc=loc,
            )
            ctx._set_token(self.dst, out_tok)
            return None

        if has_nonzero_idx:
            # The pointer fallback broadcasts the buffer BASE pointer over the
            # tile — it cannot address a non-zero partition. Refuse loudly
            # rather than atomically updating the wrong elements.
            from tilelang.tileir.errors import TileIRLoweringNotImplementedError as _NotImpl

            raise _NotImpl(
                f"AtomicRMW on `{self.dst.name}`: non-zero partition indices require the "
                "TensorView path (RELAXED ordering, no return value); "
                f"got memory_order={self.memory_order}, return_prev={self.return_prev}, "
                f"view={'present' if buf_info.view is not None else 'absent'}."
            )

        # Fallback: atomic_rmw_tko with pointer
        ptr_shaped = _broadcast_ptr(ct, buf_info.ptr, tile_shape, loc=loc)
        prev_val, out_tok = ct.atomic_rmw_tko(
            ordering,
            ct.MemoryScope.DEVICE,
            ptr_shaped,
            mode,
            val_tile,
            input_token=tok,
            return_token=True,
            loc=loc,
        )
        ctx._set_token(self.dst, out_tok)
        if self.return_prev:
            return prev_val
        return None  # side-effect only; previous value discarded


@dataclasses.dataclass(eq=False)
class AtomicLoad(TileOp, opcode="atomic_load", effect=Effect.READ):
    """Atomic load from a global buffer with specified memory ordering.

    Emits ``ct.load_ptr_tko`` with the requested memory ordering semantics.
    ``memory_order`` is an integer: 0=RELAXED, 2=ACQUIRE.
    """

    src: Any = buffer_operand(effect=Effect.READ)
    memory_order: int = attribute(default=2)  # default ACQUIRE

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower AtomicLoad to MLIR: ct.load_ptr_tko with memory ordering."""
        from tilelang.tileir.emission_utils import (
            _ensure_token,
            _mlir_element_type,
            _broadcast_ptr,
        )
        from tilelang.tileir.errors import TileIRLoweringNotImplementedError

        ct = ctx.ct
        loc = ctx.loc

        buf_info = ctx.get_buffer_info(self.src)
        # Map memory order integer.
        order = self.memory_order
        if order == 0:
            ordering = ct.MemoryOrderingSemantics.RELAXED
        elif order == 2:
            ordering = ct.MemoryOrderingSemantics.ACQUIRE
        else:
            raise TileIRLoweringNotImplementedError(f"atomic load: unsupported memory order id {order}; supported: relaxed=0, acquire=2.")

        tile_shape = list(getattr(self.src.type, "shape", ()))
        elem_ty = _mlir_element_type(ctx, self.src.type)
        tile_type = ct.TileType.get(tile_shape, elem_ty)
        ptr_shaped = _broadcast_ptr(ct, buf_info.ptr, tile_shape, loc=loc)

        tok = _ensure_token(ctx, self.src)
        result_tile, out_tok = ct.load_ptr_tko(
            tile_type,
            ptr_shaped,
            memory_ordering_semantics=ordering,
            memory_scope=ct.MemoryScope.DEVICE,
            input_token=tok,
            return_token=True,
            loc=loc,
        )
        ctx._set_token(self.src, out_tok)
        return result_tile


@dataclasses.dataclass(eq=False)
class AtomicStore(TileOp, opcode="atomic_store", effect=Effect.WRITE):
    """Atomic store to a global buffer with specified memory ordering.

    Emits ``ct.store_ptr_tko`` with the requested memory ordering semantics.
    ``memory_order`` is an integer: 0=RELAXED, 3=RELEASE.
    """

    dst: Any = buffer_operand(effect=Effect.WRITE)
    val: Any = operand()
    memory_order: int = attribute(default=3)  # default RELEASE

    def emit_mlir(self, ctx: Any) -> None:
        """Lower AtomicStore to MLIR: ct.store_ptr_tko with memory ordering."""
        from tilelang.tileir.emission_utils import (
            _ensure_token,
            _as_tile,
            _broadcast_ptr,
        )
        from tilelang.tileir.errors import TileIRLoweringNotImplementedError

        ct = ctx.ct
        loc = ctx.loc

        buf_info = ctx.get_buffer_info(self.dst)
        # Resolve val tile.
        if ctx.is_tile_buffer(self.val):
            val_tile = _as_tile(ctx, ctx.get_tile(self.val))
        elif self.val in ctx._buffer_map:
            val_info = ctx._buffer_map[self.val]
            tile_shape = list(getattr(self.dst.type, "shape", ()))
            ptr_v = _broadcast_ptr(ct, val_info.ptr, tile_shape, loc=loc)
            from tilelang.tileir.emission_utils import _mlir_element_type

            elem_ty = _mlir_element_type(ctx, self.val.type)
            tile_type = ct.TileType.get(tile_shape, elem_ty)
            load_tok = _ensure_token(ctx, self.val)
            val_raw, _ = ct.load_ptr_tko(
                tile_type,
                ptr_v,
                memory_ordering_semantics=ct.MemoryOrderingSemantics.RELAXED,
                memory_scope=ct.MemoryScope.DEVICE,
                input_token=load_tok,
                return_token=True,
                loc=loc,
            )
            val_tile = val_raw
        else:
            val_tile = _as_tile(ctx, ctx.lookup(self.val))
        tile_shape = list(val_tile.tile_type.shape)

        # Map memory order integer.
        order = self.memory_order
        if order == 0:
            ordering = ct.MemoryOrderingSemantics.RELAXED
        elif order == 3:
            ordering = ct.MemoryOrderingSemantics.RELEASE
        else:
            raise TileIRLoweringNotImplementedError(f"atomic store: unsupported memory order id {order}; supported: relaxed=0, release=3.")

        ptr_shaped = _broadcast_ptr(ct, buf_info.ptr, tile_shape, loc=loc)
        tok = _ensure_token(ctx, self.dst)
        # Note: store_ptr_tko does not support return_token; token threading is
        # via _ensure_token's conservative per-buffer chain (no out-token update needed).
        ct.store_ptr_tko(
            ptr_shaped,
            val_tile,
            memory_ordering_semantics=ordering,
            memory_scope=ct.MemoryScope.DEVICE,
            input_token=tok,
            loc=loc,
        )
        return None


@dataclasses.dataclass(eq=False)
class AtomicCAS(TileOp, opcode="atomic_cas", effect=Effect.READWRITE):
    """Atomic compare-and-swap on a global buffer.

    The CUDA Tile IR dialect exposes ``ct.atomic_cas_tko`` which performs a
    compare-and-swap: if ``*dst == expected``, write ``desired`` and return the
    old value; otherwise return the old value unchanged.

    ``dst``      — GLOBAL buffer to atomically compare-and-swap.
    ``expected`` — tile Value with the compare value.
    ``desired``  — tile Value with the replacement value.
    """

    dst: Any = buffer_operand(effect=Effect.READWRITE)
    expected: Any = operand()
    desired: Any = operand()

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower AtomicCAS to MLIR: ct.atomic_cas_tko.

        Steps
        -----
        1. Resolve ``expected`` and ``desired`` operand tiles from value_map.
        2. Look up the dst GLOBAL buffer via ctx.get_buffer_info.
        3. Build a broadcast pointer from buf_info.ptr over the tile shape.
        4. Call ct.atomic_cas_tko with RELAXED ordering and DEVICE scope.
        5. Update the dst token via ctx._set_token.

        Returns the old-value tile so emit_module can bind it to results[0]
        (matches the CAS return-the-previous-value semantics).
        """
        from tilelang.tileir.emission_utils import (
            _ensure_token,
            _as_tile,
            _broadcast_ptr,
        )

        ct = ctx.ct
        loc = ctx.loc

        buf_info = ctx.get_buffer_info(self.dst)
        cmp_tile = _as_tile(ctx, ctx.lookup(self.expected))
        val_tile = _as_tile(ctx, ctx.lookup(self.desired))
        tile_shape = list(cmp_tile.tile_type.shape)

        # Build a broadcast pointer over the tile shape.
        ptr_shaped = _broadcast_ptr(ct, buf_info.ptr, tile_shape, loc=loc)

        tok = _ensure_token(ctx, self.dst)
        old_val, out_tok = ct.atomic_cas_tko(
            ct.MemoryOrderingSemantics.RELAXED,
            ct.MemoryScope.DEVICE,
            ptr_shaped,
            cmp_tile,
            val_tile,
            input_token=tok,
            return_token=True,
            loc=loc,
        )
        ctx._set_token(self.dst, out_tok)
        return old_val  # old value — bound to results[0] by emit_module
