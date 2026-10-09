"""
SIMT cross-workitem all-reduce.

Vendored from the PTOAS project (``ptodsl/ptodsl/_allreduce.py``); upstream
commit f5eff3ee249697f6157088f649c6434fcc9d7c5b ("refactor(ptodsl): remove
the SIMT allreduce helpers") removed the module expecting TileLang to own the
implementation. contrib.ptodsl's integer-signedness compatibility, previously
applied as monkey patches from ``simt.py``, is merged directly into the
reducer tables and the dtype dispatch.

All-reduce ops are emitted **inline** at the current insertion point.
Three reducer variants: ``simt_allreduce_sum``, ``simt_allreduce_max``, ``simt_allreduce_min``.

Dispatch tree (compile-time, since *threads* / *scale* are Python ints)::

    threads <= scale                                       →  identity
    threads ≤ 32,  pow2(threads), pow2(scale)              →  warp_reduce
    threads ≤ 32                                           →  ub_reduce
    threads > 32,  pow2(threads), scale≤32, pow2(scale)   →  cross_warp_reduce
    otherwise                                              →  ub_reduce (fallback)
"""

from __future__ import annotations

import ptodsl._scalar as scalar
from ptodsl import pto
from ptodsl._control_flow import if_, for_
from ptodsl._ops import const as _const, get_laneid, get_tid_x, syncthreads
from ptodsl._surface_values import unwrap_surface_value, wrap_surface_value
from ptodsl._types import (
    _integer_signedness,
    _resolve,
    _restore_integer_signedness,
    _strip_integer_signedness,
    float16 as _f16_dtype,
    float32 as _f32_dtype,
    ui32 as _ui32_dtype,
)
from ptoas.mlir.dialects import pto as _pto
from ptoas.mlir.ir import F16Type, F32Type, IntegerType


# ── helpers ────────────────────────────────────────────────────────────────────


def _is_pow2(n: int) -> bool:
    """Compile-time power-of-two check."""
    return n > 0 and (n & (n - 1)) == 0


def _validate_scratch_buffer(scratch, *, value_type, reducer: str, dtype: str, threads: int, scale: int, thread_offset: int) -> None:
    context = f"all_reduce {reducer}/{dtype}/t{threads}/s{scale}/o{thread_offset}"
    raw_scratch = unwrap_surface_value(scratch)
    try:
        scratch_type = _pto.PtrType(raw_scratch.type)
    except Exception as exc:
        raise TypeError(f"{context} requires a UB scratch buffer pointer, got {raw_scratch.type}") from exc
    if scratch_type.element_type != value_type:
        raise TypeError(f"{context} scratch element type mismatch: expected {value_type}, got {scratch_type.element_type}")
    memory_space = getattr(scratch_type, "memory_space", None)
    vec_attr = _pto.AddressSpaceAttr.get(_pto.AddressSpace.VEC)
    memory_space_value = getattr(memory_space, "value", memory_space)
    scratch_type_text = str(scratch_type)
    if (
        memory_space != vec_attr
        and memory_space_value != _pto.AddressSpace.VEC
        and ", ub>" not in scratch_type_text
        and ", vec>" not in scratch_type_text
    ):
        raise TypeError(f"{context} requires a UB scratch buffer, got {scratch_type}")


# ── signedness-preserving redux/shuffle ────────────────────────────────────────
# PTOAS's redux/shuffle take signless i32 carriers while its allreduce
# dispatches on signed/unsigned types.


def _redux_signedness(type_obj):
    # Maps signed ints to signless pto.iXX (uint keeps pto.uiXX).
    return "unsigned" if _integer_signedness(type_obj) == "unsigned" else "signed"


def _redux_integer_compat(op, value, *, with_signedness):
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return op(value)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    if with_signedness:
        result = op(signless_value, signedness=_redux_signedness(raw_value.type))
    else:
        result = op(signless_value)
    return wrap_surface_value(_restore_integer_signedness(unwrap_surface_value(result), raw_value.type))


def _redux_add(value):
    # redux_add takes no signedness argument.
    return _redux_integer_compat(pto.redux_add, value, with_signedness=False)


def _redux_max(value):
    # redux_max/min require an explicit integer signedness.
    return _redux_integer_compat(pto.redux_max, value, with_signedness=True)


def _redux_min(value):
    return _redux_integer_compat(pto.redux_min, value, with_signedness=True)


def shuffle_bfly(value, offset):
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return pto.shuffle_bfly(value, offset)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    result = pto.shuffle_bfly(signless_value, offset)
    return wrap_surface_value(_restore_integer_signedness(unwrap_surface_value(result), raw_value.type))


# ── reducer dispatch tables ────────────────────────────────────────────────────

_REDUCER_IDENTITY = {
    "sum": {"f32": 0.0, "f16": 0.0, "si32": 0, "ui32": 0},
    "max": {
        "f32": float("-inf"),
        "f16": float("-inf"),
        "si32": -(2**31),
        "ui32": 0,
    },
    "min": {
        "f32": float("inf"),
        "f16": float("inf"),
        "si32": 2**31 - 1,
        "ui32": 2**32 - 1,
    },
}

_REDUCER_COMBINE = {
    "sum": lambda a, b: a + b,
    "max": scalar.max,
    "min": scalar.min,
}

_REDUCER_REDUX = {
    "sum": _redux_add,
    "max": _redux_max,
    "min": _redux_min,
}

# Run the si32 strategy on the signless carrier with identity constants typed
# to match, instead of reinterpreting to si32 and back.
_REDUCER_IDENTITY_DTYPE = {
    "f32": _f32_dtype,
    "f16": _f16_dtype,
    "si32": pto.i32,
    "ui32": _ui32_dtype,
}


# ── butterfly  ──────────────────────────────────────────────────────────────────


def _emit_butterfly(v, *, threads: int, scale: int, reducer: str):
    """Unrolled butterfly shuffle reduce."""
    combine = _REDUCER_COMBINE[reducer]
    cur = threads
    while cur > scale:
        offset = cur // 2
        v = combine(v, shuffle_bfly(v, offset))
        cur //= 2
    return v


# ── warp_hw_reduce  ────────────────────────────────────────────────────────────


def _emit_warp_hw_reduce(x, *, threads: int, lane_in_warp, dtype: str, reducer: str):
    """Warp-level hardware reduce with group masking."""
    redux_fn = _REDUCER_REDUX[reducer]
    groups = 32 // threads

    if groups == 1:
        return redux_fn(x)

    c_identity = _const(
        _REDUCER_IDENTITY[reducer][dtype],
        dtype=_resolve(_REDUCER_IDENTITY_DTYPE[dtype]),
    )
    my_group = lane_in_warp // threads

    for g in range(groups):
        in_group = my_group == g
        masked = scalar.select(in_group, x, c_identity)
        reduced = redux_fn(masked)
        x = scalar.select(in_group, reduced, x)
    return x


# ── warp_reduce  ───────────────────────────────────────────────────────────────


def _emit_warp_reduce(x, *, dtype, threads, scale, thread_offset, reducer):
    """Single-warp all-reduce."""
    extent = threads // scale
    if extent <= 1:
        return x

    if thread_offset:
        lane_in_warp = (get_tid_x() - thread_offset) & 31
    else:
        lane_in_warp = get_laneid()

    if extent >= 16 and scale == 1:
        return _emit_warp_hw_reduce(
            x,
            threads=threads,
            lane_in_warp=lane_in_warp,
            dtype=dtype,
            reducer=reducer,
        )
    return _emit_butterfly(x, threads=threads, scale=scale, reducer=reducer)


# ── cross_warp_reduce  ─────────────────────────────────────────────────────────


def _simt_warp_indexing(*, thread_offset):
    """Derive global tx, warp id, and warp-local lane id for this work-item."""
    tid_x = get_tid_x()
    if thread_offset:
        tx = tid_x - thread_offset
        wid = tx // 32
        lid = tx & 31
    else:
        tx = tid_x
        wid = tx // 32
        lid = get_laneid()
    return tx, wid, lid


def _warp_partial_reduce(x, *, scale: int, reducer: str):
    """Reduce the warp-local value down to ``scale`` partials per warp."""
    if scale == 1:
        return _REDUCER_REDUX[reducer](x)
    return _emit_butterfly(x, threads=32, scale=scale, reducer=reducer)


def _write_warp_partials(warp_val, scratch, *, wid, lid, scale: int) -> None:
    """Have each warp's ``scale`` leaders store their partial result."""
    is_writer = lid < scale
    with if_(is_writer) as br, br.then_:
        slot = wid * scale + lid
        scalar.store(warp_val, scratch, slot)


def _combine_warp_partials(lid, scratch, *, num_warps: int, scale: int, reducer: str, identity):
    """Combine every warp's partials on warp 0 (leader lanes only).

    The caller wraps the emitted ops in the runtime leader-warp region; the
    ``num_warps``/``scale``/``reducer`` dispatch is resolved at trace time.
    """
    if scale == 1:
        loaded = scalar.select(
            lid < num_warps,
            scalar.load(scratch, lid),
            identity,
        )
        return _REDUCER_REDUX[reducer](loaded)

    total = scale * num_warps
    if total <= 32:
        loaded = scalar.select(
            lid < total,
            scalar.load(scratch, lid),
            identity,
        )
        return _emit_butterfly(
            loaded,
            threads=total,
            scale=scale,
            reducer=reducer,
        )

    combine = _REDUCER_COMBINE[reducer]
    is_reducer = lid < scale
    reduced = identity
    my_slot = lid % scale
    for w in range(num_warps):
        idx_val = w * scale + my_slot
        loaded_v = scalar.load(scratch, idx_val)
        reduced = combine(reduced, loaded_v)
    return scalar.select(is_reducer, reduced, identity)


def _leader_warp_reduce(tx, lid, scratch, *, num_warps: int, scale: int, reducer: str, identity):
    """Let warp 0 combine every warp's partial into the cross-warp result."""
    is_leader_warp = tx < 32
    with if_(is_leader_warp) as br:
        with br.then_:
            br.assign(
                stage4_result=_combine_warp_partials(
                    lid,
                    scratch,
                    num_warps=num_warps,
                    scale=scale,
                    reducer=reducer,
                    identity=identity,
                )
            )
        with br.else_:
            br.assign(stage4_result=identity)

    return br.stage4_result


def _broadcast_warp_result(partial_reduced, scratch, *, tx, scale: int):
    """Store the final partial and load the value broadcast to all lanes."""
    is_global_leader = tx < scale
    with if_(is_global_leader) as br5, br5.then_:
        scalar.store(partial_reduced, scratch, tx)

    syncthreads()
    result = scalar.load(scratch, tx % scale)
    syncthreads()
    return result


def _emit_cross_warp_reduce(x, scratch, *, dtype, threads, scale, thread_offset, reducer):
    """Cross-warp all-reduce (threads > 32)."""
    num_warps = threads // 32
    c_identity = _const(
        _REDUCER_IDENTITY[reducer][dtype],
        dtype=_resolve(_REDUCER_IDENTITY_DTYPE[dtype]),
    )
    tx, wid, lid = _simt_warp_indexing(thread_offset=thread_offset)
    warp_val = _warp_partial_reduce(x, scale=scale, reducer=reducer)
    _write_warp_partials(warp_val, scratch, wid=wid, lid=lid, scale=scale)

    syncthreads()

    partial_reduced = _leader_warp_reduce(
        tx,
        lid,
        scratch,
        num_warps=num_warps,
        scale=scale,
        reducer=reducer,
        identity=c_identity,
    )

    return _broadcast_warp_result(
        partial_reduced,
        scratch,
        tx=tx,
        scale=scale,
    )


# ── ub_reduce  ─────────────────────────────────────────────────────────────────


def _emit_ub_reduce(x, scratch, *, dtype, threads, scale, thread_offset, reducer):
    """UB-scratch all-reduce (fallback for non-pow2 or general case)."""
    combine = _REDUCER_COMBINE[reducer]

    # ── thread indexing ──────────────────────────────────────────────────
    tid_x = get_tid_x()
    tx = (tid_x - thread_offset) if thread_offset else tid_x
    group = tx // threads
    lane = tx % threads

    # ── each lane writes x → scratch[tx] ─────────────────────────────────
    scalar.store(x, scratch, tx)
    syncthreads()

    # ── reducers sequentially combine ────────────────────────────────────
    is_reducer = scalar.cmp(lane, scale, "lt")
    with if_(is_reducer) as br:
        with br.then_:
            group_offset = group * threads
            first_elem = group_offset + lane
            acc = scalar.load(scratch, first_elem)

            carry_loop = for_(scale, threads, step=scale).carry(acc=acc)
            with carry_loop:
                prev = carry_loop.acc
                elem = first_elem + carry_loop.iv
                loaded = scalar.load(scratch, elem)
                carry_loop.update(acc=combine(prev, loaded))
            acc = carry_loop.final("acc")

            br.assign(flag=acc)
        with br.else_:
            br.assign(flag=x)

    flag = br.flag
    syncthreads()

    # ── per-class leader writes back ─────────────────────────────────────
    is_leader = scalar.cmp(lane, scale, "lt")
    with if_(is_leader) as br5, br5.then_:
        scalar.store(flag, scratch, group * threads + lane)

    # ── broadcast ────────────────────────────────────────────────────────
    syncthreads()
    result = scalar.load(scratch, group * threads + (tx % scale))
    syncthreads()

    return result


# ── public API  ────────────────────────────────────────────────────────────────


def _check_params(*, threads, scale, thread_offset):
    """Validate allreduce parameters (compile-time checks)."""
    for name, val in (("threads", threads), ("scale", scale), ("thread_offset", thread_offset)):
        if not isinstance(val, int):
            raise ValueError(f"all_reduce: '{name}' must be a Python int, got {type(val).__name__}")
    if threads < 1:
        raise ValueError(f"all_reduce: threads must be >= 1, got {threads}")
    if scale < 1:
        raise ValueError(f"all_reduce: scale must be >= 1, got {scale}")
    if thread_offset < 0:
        raise ValueError(f"all_reduce: thread_offset must be >= 0, got {thread_offset}")
    if threads % scale != 0:
        raise ValueError(f"all_reduce requires threads % scale == 0; got threads={threads}, scale={scale}")


def _simt_allreduce(value, *, threads, scale, thread_offset, scratch, reducer):
    """Unified allreduce dispatch tree."""
    _check_params(threads=threads, scale=scale, thread_offset=thread_offset)

    if threads <= scale:
        return value

    raw_value = unwrap_surface_value(value)
    if raw_value.type == F32Type.get():
        dtype = "f32"
    elif raw_value.type == F16Type.get():
        dtype = "f16"
    elif raw_value.type in (IntegerType.get_signed(32), IntegerType.get_signless(32)):
        dtype = "si32"
    elif raw_value.type == IntegerType.get_unsigned(32):
        dtype = "ui32"
    else:
        raise NotImplementedError(f"all_reduce: unsupported dtype {raw_value.type}")

    args = dict(dtype=dtype, threads=threads, scale=scale, thread_offset=thread_offset, reducer=reducer)

    if threads <= 32 and _is_pow2(threads) and _is_pow2(scale):
        return _emit_warp_reduce(value, **args)

    if scratch is None:
        raise ValueError(f"all_reduce {reducer}/{dtype}/t{threads}/s{scale}/o{thread_offset} requires a UB scratch buffer")
    _validate_scratch_buffer(
        scratch,
        value_type=raw_value.type,
        reducer=reducer,
        dtype=dtype,
        threads=threads,
        scale=scale,
        thread_offset=thread_offset,
    )

    if threads <= 32:
        return _emit_ub_reduce(value, scratch, **args)

    if scale <= 32 and _is_pow2(threads) and _is_pow2(scale):
        return _emit_cross_warp_reduce(value, scratch, **args)

    return _emit_ub_reduce(value, scratch, **args)


def simt_allreduce_sum(value, *, threads, scale=1, thread_offset=0, scratch=None):
    """Sum reduce across SIMT work-items."""
    return _simt_allreduce(value, threads=threads, scale=scale, thread_offset=thread_offset, scratch=scratch, reducer="sum")


def simt_allreduce_max(value, *, threads, scale=1, thread_offset=0, scratch=None):
    """Max reduce across SIMT work-items."""
    return _simt_allreduce(value, threads=threads, scale=scale, thread_offset=thread_offset, scratch=scratch, reducer="max")


def simt_allreduce_min(value, *, threads, scale=1, thread_offset=0, scratch=None):
    """Min reduce across SIMT work-items."""
    return _simt_allreduce(value, threads=threads, scale=scale, thread_offset=thread_offset, scratch=scratch, reducer="min")


__all__ = [
    "simt_allreduce_sum",
    "simt_allreduce_max",
    "simt_allreduce_min",
]
