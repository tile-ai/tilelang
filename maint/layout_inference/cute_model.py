"""Symbolic statement scoring on the CuTe layout algebra (experiment).

The pipeline (validated by design against the exact oracle in oracle.py):

1. PACK the fragment exactly as FragmentNode::InverseWithLevel does
   (src/layout/layout.cc:1133-1141): replication becomes a trailing input
   dim, and the outputs are ordered [thread, slot] so the multi-output
   probe's row-major serialization yields ONE plain strided layout
   computing `thread * slots + slot` — the enumerator's cell index.
2. CONVERT via cute.Layout.from_tilelang (probe-then-prove; None on
   swizzle/non-affine).
3. INVERT via cute.right_inverse; bijectivity check by size (the partial
   inverse of a non-injective candidate is smaller).
4. COMPOSE with the global row-major byte-stride layout, then split into
   (slot, thread) axes with with_shape.
5. VECTOR WIDTH is read off the coalesced slot axis: the innermost mode
   with stride == elem_bytes gives the contiguous run; alignment is
   checked on the remaining mode strides. Pure mode arithmetic.
6. SEGMENTS are counted by evaluating the ALGEBRA-DERIVED composed layout
   once per issued vector lane at warp/step granularity, instead of once
   per logical point and replica. Store gating projects the replica index
   through the inverse. A fully closed-form segment count from mode
   arithmetic remains a possible refinement; this bounded evaluation
   already exercises conversion, inversion, and composition end to end.

Every algebra call is wrapped: cute ops raise tvm InternalError on their
ICHECKs, and conversion returns None — both map to Unconvertible(reason),
the analog of the C++ model's conservative worst case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tvm.error
from tilelang import tvm
from tilelang.layout import Layout as TLLayout
from tilelang.layout import cute
from tvm.tirx.stmt_functor import substitute

from oracle import placeholder_vars


@dataclass
class CuteScore:
    vector: int
    issue: int
    bw: int
    segments: int
    notes: list[str] = field(default_factory=list)  # human-readable derivation


@dataclass
class Unconvertible:
    reason: str


def _flat_modes(layout) -> list[tuple[int, int]]:
    """Coalesced (extent, stride) pairs, innermost first."""
    lay = cute.coalesce(layout)
    shape = cute.flatten_to_tuple(lay.shape)
    stride = cute.flatten_to_tuple(lay.stride)
    return list(zip([int(s) for s in shape], [int(t) for t in stride]))


def pack_fragment(frag) -> TLLayout:
    """(coords..., rep) -> [thread, slot] as a plain multi-output Layout."""
    shape = [int(x) for x in frag.get_input_shape()]
    R = int(frag.replicate_size)
    inputs, rep_var = placeholder_vars(frag)
    thread_expr = frag.forward_thread
    slot_expr = frag.get_forward_index()[0]

    def forward(*coords):
        vmap = {inputs[i]: coords[i] for i in range(len(shape))}
        if rep_var is not None:
            vmap[rep_var] = coords[len(shape)]
        return [substitute(thread_expr, vmap), substitute(slot_expr, vmap)]

    return TLLayout(shape + [R], forward)


def score_statement_cute(
    frag, global_shape, elem_bytes: int, is_store: bool, vector_bits: int = 128, warp_size: int = 32, segment_bytes: int = 128
):
    """Score one fragment<->global copy statement via the CuTe algebra.

    Returns CuteScore or Unconvertible.
    """
    shape = [int(x) for x in frag.get_input_shape()]
    if len(global_shape) != len(shape):
        return Unconvertible("global rank mismatch")
    R = int(frag.replicate_size)
    S = int(frag.get_output_shape()[0])
    T = int(frag.get_thread_size())
    notes = []

    try:
        flat = cute.Layout.from_tilelang(pack_fragment(frag))
    except tvm.error.InternalError as exc:
        return Unconvertible(f"conversion crashed: {exc}")
    if flat is None:
        return Unconvertible("from_tilelang returned None (non-affine/swizzle)")
    notes.append(f"F_flat = {flat}")

    try:
        inv = cute.right_inverse(flat)
    except tvm.error.InternalError as exc:
        return Unconvertible(f"right_inverse crashed: {exc}")
    if int(cute.size(inv)) != int(cute.size(flat)):
        return Unconvertible(f"non-bijective: size(inv)={cute.size(inv)} != {cute.size(flat)}")
    notes.append(f"Finv = {inv}")

    # Global row-major BYTE strides over (coords..., rep) with rep-stride 0.
    gstrides = [0] * len(shape)
    acc = elem_bytes
    for d in range(len(shape) - 1, -1, -1):
        gstrides[d] = acc
        acc *= int(global_shape[d])
    g_text = "(" + ",".join(str(s) for s in shape + [R]) + "):(" + ",".join(str(st) for st in gstrides + [0]) + ")"
    try:
        G = cute.Layout.parse(g_text)
        A = cute.composition(G, inv)  # cell -> byte address
        A2 = A.with_shape((S, T))  # (slot, thread) split
    except tvm.error.InternalError as exc:
        return Unconvertible(f"composition failed: {exc}")
    notes.append(f"A2 = {A2}")

    # --- Vector width from the slot axis, pure mode arithmetic ------------
    slot_modes = _flat_modes(A2[0])
    notes.append(f"slot modes = {slot_modes}")
    run = 1
    if slot_modes and slot_modes[0][1] == elem_bytes:
        run = slot_modes[0][0]
    max_vector = min(S, (vector_bits // 8) // max(1, elem_bytes))
    vector = 1
    cand = 32
    while cand >= 2:
        if (
            cand <= max_vector
            and S % cand == 0
            and run % cand == 0
            and all(st % (cand * elem_bytes) == 0 for e, st in slot_modes[1:] if st != 0)
            and all(st % (cand * elem_bytes) == 0 for e, st in _flat_modes(A2[1]) if st != 0)
        ):
            vector = cand
            break
        cand //= 2
    steps = S // vector
    lane_bytes = vector_bits // 8
    issue = steps * T * lane_bytes

    # --- Segments: bounded evaluation of the derived layout ---------------
    # Base byte address per (step, warp, lane): evaluate A at
    # cell = thread * S + slot with slot = q*vector, thread = w*W + lane.
    # rep gating: rep(cell) = flat logical index // prod(shape) since rep is
    # the slowest packed input (column-major).
    prod_shape = 1
    for s in shape:
        prod_shape *= s
    num_warps = (T + warp_size - 1) // warp_size
    segments = 0
    span = vector * elem_bytes - 1
    for q in range(steps):
        for w in range(num_warps):
            seg_ids = set()
            for lane in range(min(warp_size, T - w * warp_size)):
                t = w * warp_size + lane
                cell = t * S + q * vector
                if is_store and R > 1:
                    rep = int(inv(cell)) // prod_shape
                    if rep != 0:
                        continue  # guarded replica: idle for stores
                base = int(A(cell))
                seg_ids.add(base // segment_bytes)
                seg_ids.add((base + span) // segment_bytes)
            segments += len(seg_ids)
    bw = segments * segment_bytes
    return CuteScore(vector=vector, issue=int(issue), bw=int(bw), segments=int(segments), notes=notes)
