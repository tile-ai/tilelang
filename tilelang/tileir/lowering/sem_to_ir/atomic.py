"""SemanticIR -> TileIR lowering: atomic + register-control handlers.

Provides the ``@impl`` handlers for the ``atomic_rmw`` and ``register_control``
SemanticStmt kinds (AtomicRMW / AtomicLoad / AtomicStore).  Imports the shared
foundation and the region/static-int helpers from ``tile_ops``.
"""

from __future__ import annotations

from tvm import tirx as _tir

from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.errors import TileIRLoweringNotImplementedError, _UnsupportedTileIRNode
from tilelang.tileir.ir.types import MemSpace, TileType
from tilelang.tileir.ir.ops import AtomicLoad, AtomicRMW, AtomicStore, Barrier
from tilelang.tileir.semantic import SemanticStmt

from ._base import LoweringScope, impl

# tile_ops does not import atomic, so this module-level import is acyclic.
from .tile_ops import _extract_region_buffer_name, _extract_static_int


@impl("atomic_rmw")
def _lower_atomic_rmw(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower atomic element-wise ops to AtomicRMW / AtomicLoad / AtomicStore IR ops.

    Handles all six TileLang element-wise atomic ops:

    tl.atomic_add_elem_op / tl.atomic_max_elem_op / tl.atomic_min_elem_op:
        args[0] = access_ptr(dst, "rw")
        args[1] = value (BufferLoad)
        args[2] = memory_order integer (optional, 0=RELAXED default)

    tl.atomic_add_ret_elem_op:
        Same layout but marks return_prev=True (emits result tile).

    tl.atomic_load_elem_op:
        args[0] = access_ptr(src, "r")
        args[1] = memory_order integer

    tl.atomic_store_elem_op:
        args[0] = access_ptr(dst, "w")
        args[1] = value (BufferLoad)
        args[2] = memory_order integer

    tl.tileop.atomicadd / tl.tileop.atomicmax / tl.tileop.atomicmin:
        Legacy tile-level: args[0]=val_region, args[1]=dst_region; no memory_order.

    Memory ordering IDs: 0=RELAXED, 2=ACQUIRE, 3=RELEASE, 4=ACQ_REL.
    """
    attrs = dict(stmt.attrs)
    op_name = attrs.get("op", "")
    tir_op_name = op_name
    args = list(stmt.call_args)

    # Determine kind (add / max / min)
    if "atomicmax" in tir_op_name or "atomic_max" in tir_op_name:
        kind = "max"
    elif "atomicmin" in tir_op_name or "atomic_min" in tir_op_name:
        kind = "min"
    else:
        kind = "add"

    # tl.atomic_load_elem_op: AtomicLoad
    if "atomic_load_elem_op" in tir_op_name:
        if len(args) < 2:
            raise _UnsupportedTileIRNode(f"atomic_load_elem_op: expected 2 args (src_ptr, memory_order), got {len(args)}.")
        src_name = _extract_region_buffer_name(args[0])
        if src_name is None:
            raise _UnsupportedTileIRNode(f"atomic_load_elem_op: could not extract src buffer name from args[0]={args[0]!r}.")
        try:
            src_val = scope.lookup_buffer(src_name)
        except KeyError as exc:
            raise _UnsupportedTileIRNode(f"atomic_load_elem_op: buffer lookup failed: {exc}") from exc
        memory_order = _extract_static_int(args[1])
        if memory_order is None:
            raise _UnsupportedTileIRNode(f"atomic_load_elem_op: memory_order must be a static integer, got {args[1]!r}.")
        if memory_order not in (0, 2):
            raise TileIRLoweringNotImplementedError(
                f"atomic_load_elem_op: unsupported memory order id {memory_order}; supported: relaxed=0, acquire=2."
            )
        _load_tile_shape = tuple(src_val.type.shape)
        _load_result_ty = TileType(
            dtype=src_val.type.dtype,
            shape=_load_tile_shape,
            space=MemSpace.REGISTER,
            layout=None,
        )
        op = builder.create(AtomicLoad(src=src_val, memory_order=memory_order), result_types=(_load_result_ty,))
        return op.results[0] if op.results else None

    # tl.atomic_store_elem_op: AtomicStore
    if "atomic_store_elem_op" in tir_op_name:
        if len(args) < 3:
            raise _UnsupportedTileIRNode(f"atomic_store_elem_op: expected 3 args (dst_ptr, val, memory_order), got {len(args)}.")
        dst_name = _extract_region_buffer_name(args[0])
        val_name = _extract_region_buffer_name(args[1])
        if dst_name is None or val_name is None:
            raise _UnsupportedTileIRNode("atomic_store_elem_op: could not extract buffer names from args.")
        try:
            dst_val = scope.lookup_buffer(dst_name)
            val_val = scope.lookup_buffer(val_name)
        except KeyError as exc:
            raise _UnsupportedTileIRNode(f"atomic_store_elem_op: buffer lookup failed: {exc}") from exc
        memory_order = _extract_static_int(args[2])
        if memory_order is None:
            raise _UnsupportedTileIRNode(f"atomic_store_elem_op: memory_order must be a static integer, got {args[2]!r}.")
        if memory_order not in (0, 3):
            raise TileIRLoweringNotImplementedError(
                f"atomic_store_elem_op: unsupported memory order id {memory_order}; supported: relaxed=0, release=3."
            )
        builder.create(AtomicStore(dst=dst_val, val=val_val, memory_order=memory_order))
        return None

    # Legacy tile-level ops (tl.tileop.atomicadd / .atomicmax / .atomicmin)
    # TIR layout: args[0]=val_region, args[1]=dst_region (reversed from elem ops).
    if "tileop.atomic" in tir_op_name:
        if len(args) < 2:
            raise _UnsupportedTileIRNode(f"atomic_rmw ({kind!r}): expected TIR call with at least 2 args, got {len(args)}.")
        # tileop.atomicadd: args[0]=src_region, args[1]=dst_region
        dst_name = _extract_region_buffer_name(args[1])
        val_name = _extract_region_buffer_name(args[0])
        if dst_name is None or val_name is None:
            raise _UnsupportedTileIRNode(f"atomic_rmw ({kind!r}): could not extract buffer names from tile-region args.")
        try:
            dst_val = scope.lookup_buffer(dst_name)
            val_val = scope.lookup_buffer(val_name)
        except KeyError as exc:
            raise _UnsupportedTileIRNode(f"atomic_rmw ({kind!r}): buffer lookup failed: {exc}") from exc
        memory_order = 0  # tile-region atomics default to RELAXED
        # Check annotation for memory_order override.
        try:
            ann_mo = attrs.get("annotation.memory_order")
            if ann_mo is not None:
                mo_int = int(ann_mo)
                memory_order = mo_int
        except (ValueError, TypeError):
            pass
        # Resolve real dst view indices from the region base, exactly like
        # _lower_copy: ``T.atomic_add(C[by*bM, bx*bN], C_local)`` reduces the
        # (bM, bN) val tile into partition (by, bx) of C — dropping the base
        # would silently reduce every CTA into partition (0, 0).
        # The dst REGION may have higher rank than the val fragment (e.g.
        # ``dQ[rows, h, :]`` (bM, 1, D) reduced from a (bM, D) fragment) —
        # tile shape comes from the region; emit reshapes the val tile.
        from .parallel import (
            _compute_view_indices,
            _extract_tir_region_indices,
        )

        def _region_call_extents(tir_arg: object) -> tuple | None:
            """Static extents from a ``tl.region(BufferLoad, mask, ext...)`` call."""
            if isinstance(tir_arg, _tir.Call) and len(tir_arg.args) >= 3:
                exts = []
                for e in tir_arg.args[2:]:
                    if isinstance(e, _tir.IntImm):
                        exts.append(int(e))
                    else:
                        return None
                return tuple(exts)
            return None

        dst_tile_shape = None
        if len(args) >= 2:
            dst_tile_shape = _region_call_extents(args[1])
        if dst_tile_shape is None:
            dst_tile_shape = tuple(val_val.type.shape)
        dst_tir_indices = _extract_tir_region_indices(stmt.call_args, 1)
        dst_idx: tuple = ()
        dst_elem = False
        if dst_tir_indices is not None:
            if len(dst_tir_indices) != len(dst_tile_shape):
                raise _UnsupportedTileIRNode(
                    f"atomic_rmw ({kind!r}): dst region rank {len(dst_tir_indices)} != "
                    f"tile shape rank {len(dst_tile_shape)} for `{dst_name}`."
                )
            dst_idx, dst_elem = _compute_view_indices(dst_tir_indices, dst_tile_shape, scope, builder, what=f"atomic {kind} dst")
        builder.create(
            AtomicRMW(
                dst=dst_val,
                val=val_val,
                kind=kind,
                memory_order=memory_order,
                dst_indices=dst_idx,
                tile_shape=dst_tile_shape if dst_idx else (),
                elem_view=dst_elem,
            )
        )
        return None

    # Element-wise atomic RMW (add/max/min): args[0]=dst_ptr, args[1]=val
    # Handles: tl.atomic_add_elem_op, tl.atomic_add_ret_elem_op,
    #          tl.atomic_max_elem_op, tl.atomic_min_elem_op
    if len(args) < 2:
        raise _UnsupportedTileIRNode(f"atomic_rmw ({kind!r}): expected TIR call with at least 2 args (dst_ptr, val), got {len(args)}.")
    dst_name = _extract_region_buffer_name(args[0])
    val_name = _extract_region_buffer_name(args[1])
    if dst_name is None or val_name is None:
        raise _UnsupportedTileIRNode(
            f"atomic_rmw ({kind!r}): could not extract buffer names from elem-op args (args[0]={args[0]!r}, args[1]={args[1]!r})."
        )
    # This generic path emits an AtomicRMW over the WHOLE dst buffer at
    # partition (0, ..., 0).  A dst indexed at a non-zero offset (e.g.
    # ``C[by*128 + i, ...]`` — normally routed through the T.Parallel atomic
    # branch in parallel.py) would silently reduce into the wrong tile, so
    # reject any non-zero dst index loudly.
    from .parallel import _extract_region_buffer_load

    _dst_load = _extract_region_buffer_load(args[0])
    if _dst_load is not None:
        try:
            for _idx in _dst_load.indices:
                if not (isinstance(_idx, _tir.IntImm) and int(_idx) == 0):
                    raise _UnsupportedTileIRNode(
                        f"atomic_rmw ({kind!r}): dst `{dst_name}` is indexed at a non-zero "
                        f"offset `{_idx}` outside a recognized T.Parallel nest; lowering "
                        "would atomically update partition (0, ..., 0) instead."
                    )
        except ImportError:
            pass
    try:
        dst_val = scope.lookup_buffer(dst_name)
        val_val = scope.lookup_buffer(val_name)
    except KeyError as exc:
        raise _UnsupportedTileIRNode(f"atomic_rmw ({kind!r}): buffer lookup failed: {exc}") from exc

    # Extract memory_order from args[2] (optional).
    memory_order = 0
    if len(args) >= 3:
        mo = _extract_static_int(args[2])
        if mo is not None:
            memory_order = mo

    return_prev = "atomic_add_ret_elem_op" in tir_op_name
    if return_prev:
        _ret_tile_shape = tuple(dst_val.type.shape)
        _ret_result_ty = TileType(
            dtype=dst_val.type.dtype,
            shape=_ret_tile_shape,
            space=MemSpace.REGISTER,
            layout=None,
        )
        op = builder.create(
            AtomicRMW(dst=dst_val, val=val_val, kind=kind, memory_order=memory_order, return_prev=True),
            result_types=(_ret_result_ty,),
        )
        return op.results[0] if op.results else None
    builder.create(AtomicRMW(dst=dst_val, val=val_val, kind=kind, memory_order=memory_order))
    return None


# register_control kind


@impl("register_control")
def _lower_register_control(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    # Scheduling / register-allocation hints — no TileIR op needed.
    # Emit a Barrier as a structural no-op so the block is not empty.
    builder.create(Barrier())
