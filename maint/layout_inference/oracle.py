"""Independent exact oracle for fragment<->global statement scoring.

A vectorized (numpy) re-implementation of the retired C++ enumerator:
evaluate the fragment's own forward expressions over the whole
(logical, replica) grid in one pass, build the (thread, slot) -> address
table, and measure vector width / issue / segment bytes with the exact
same formulas (including base alignment and store replication gating).

Independent of both the production C++ scorer and the CuTe conversion —
this is the arbiter the symbolic reference path in cute_model.py is
validated against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tvm_tirx import ir
from tvm_tirx.ir import prim


@dataclass
class OracleScore:
    vector: int
    issue: int
    bw: int
    segments: int


class Unenumerable(Exception):
    """The statement is outside the oracle's model (mirrors the C++
    conservative fallback conditions)."""


def _dtype(expr):
    if not isinstance(expr.ty, ir.PrimType):
        raise Unenumerable("layout coordinates must have a primitive integer type")
    try:
        dtype = np.dtype(str(expr.ty.dtype))
    except TypeError as exc:
        raise Unenumerable(f"unsupported layout coordinate dtype: {expr.ty}") from exc
    if dtype.kind not in "iub":
        raise Unenumerable(f"layout coordinates must be integers, got {dtype}")
    return dtype


def _arithmetic(expr, a, b, operation):
    """Evaluate integer coordinates; do not silently wrap signed arithmetic."""
    dtype = _dtype(expr)
    if dtype.kind == "i":
        # Up to int32, int64 holds every product. Use Python integers for the
        # int64 case so an overflowing intermediate cannot masquerade as valid.
        promoted = np.int64 if dtype.itemsize <= 4 else object
        a, b = np.asarray(a, dtype=promoted), np.asarray(b, dtype=promoted)
    else:
        a, b = np.asarray(a, dtype=dtype), np.asarray(b, dtype=dtype)
    if operation in (np.floor_divide, np.mod) and np.any(b == 0):
        raise Unenumerable("division by zero in layout coordinate")
    result = operation(a, b)
    if dtype.kind == "i":
        bounds = np.iinfo(dtype)
        if np.any(result < bounds.min) or np.any(result > bounds.max):
            raise Unenumerable(f"signed {dtype} overflow in layout coordinate")
    return np.asarray(result).astype(dtype)


def eval_expr_grid(expr, env: dict) -> np.ndarray:
    """Evaluate a quasi-affine PrimExpr over numpy coordinate grids.

    `env` maps VarObj -> ndarray. Vars outside env evaluate to 0 (foreign
    additive offsets, same convention as the C++ model). Supported node
    family mirrors the model's; anything else raises Unenumerable.
    """
    if isinstance(expr, ir.IntImm):
        return np.asarray(expr.value, dtype=_dtype(expr))
    if isinstance(expr, ir.Var):
        for var, arr in env.items():
            if expr.same_as(var):
                return np.asarray(arr).astype(_dtype(expr))
        return np.asarray(0, dtype=_dtype(expr))  # foreign additive offset
    if isinstance(expr, prim.Cast):
        # Integer conversion preserves truncation, signed interpretation and
        # bool conversion. In particular uint32(uint8(256)) evaluates to zero.
        return np.asarray(eval_expr_grid(expr.value, env)).astype(_dtype(expr))
    a = lambda: eval_expr_grid(expr.a, env)  # noqa: E731
    b = lambda: eval_expr_grid(expr.b, env)  # noqa: E731
    for kind, operation in (
        (prim.Add, np.add),
        (prim.Sub, np.subtract),
        (prim.Mul, np.multiply),
        (prim.FloorDiv, np.floor_divide),
        (prim.FloorMod, np.mod),
        (prim.Min, np.minimum),
        (prim.Max, np.maximum),
    ):
        if isinstance(expr, kind):
            return _arithmetic(expr, a(), b(), operation)
    raise Unenumerable(f"unsupported node {type(expr).__name__}: {expr}")


def placeholder_vars(frag):
    """(input placeholders, rep placeholder or None).

    FragmentNode::GetForwardVars PREPENDS ReplicationPlaceholder when R > 1
    (src/layout/layout.cc:1103-1112); the input placeholders are always the
    trailing InputDim entries.
    """
    fwd_vars = list(frag.get_forward_vars())
    ndim = len(frag.get_input_shape())
    inputs = fwd_vars[-ndim:]
    rep = fwd_vars[0] if len(fwd_vars) > ndim else None
    return inputs, rep


def score_statement_oracle(
    frag, global_shape, elem_bytes: int, is_store: bool, vector_bits: int = 128, warp_size: int = 32, segment_bytes: int = 128
) -> OracleScore:
    """Score one fragment<->global copy statement by exact enumeration."""
    shape = [int(x) for x in frag.get_input_shape()]
    if len(global_shape) != len(shape):
        raise Unenumerable("global rank mismatch")
    R = int(frag.replicate_size)
    S = int(frag.get_output_shape()[0])
    T = int(frag.get_thread_size())
    inputs, rep_var = placeholder_vars(frag)

    grids = np.meshgrid(*[np.arange(s, dtype=np.int64) for s in shape], np.arange(R, dtype=np.int64), indexing="ij")
    env = {inputs[d]: grids[d] for d in range(len(shape))}
    if rep_var is not None:
        env[rep_var] = grids[-1]

    thread = np.broadcast_to(eval_expr_grid(frag.forward_thread, env), grids[0].shape)
    slot = np.broadcast_to(eval_expr_grid(frag.get_forward_index()[0], env), grids[0].shape)
    if thread.min() < 0 or thread.max() >= T or slot.min() < 0 or slot.max() >= S:
        raise Unenumerable("thread/slot out of range")

    # Row-major global element address over the logical coords (region
    # offsets drop out: foreign vars are 0 by convention on both paths).
    strides = np.ones(len(shape), dtype=np.int64)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * global_shape[d + 1]
    addr = np.zeros(grids[0].shape, dtype=np.int64)
    for d in range(len(shape)):
        addr = addr + grids[d] * strides[d]

    # Fill the (thread, slot) table; bijectivity via bincount.
    cell = (thread * S + slot).ravel()
    counts = np.bincount(cell, minlength=T * S)
    if counts.max() != 1 or cell.size != T * S:
        raise Unenumerable("candidate is not a (logical, rep) <-> cell bijection")
    table = np.empty(T * S, dtype=np.int64)
    table[cell] = addr.ravel()
    lead = np.zeros(T * S, dtype=bool)
    lead[cell[(grids[-1] == 0).ravel()]] = True
    table = table.reshape(T, S)
    lead = lead.reshape(T, S)

    # Vector width: widest power-of-two aligned contiguous run (the numeric
    # mirror of IndicesCanVectorize, identical to the C++ model).
    max_vector = min(S, (vector_bits // 8) // max(1, elem_bytes))
    vector = 1
    cand = 32
    while cand >= 2:
        if cand <= max_vector and S % cand == 0:
            blocks = table.reshape(T, S // cand, cand)
            contiguous = bool(np.all(blocks == blocks[:, :, :1] + np.arange(cand, dtype=np.int64)))
            aligned = bool(np.all(blocks[:, :, 0] % cand == 0))
            if contiguous and aligned:
                vector = cand
                break
        cand //= 2
    steps = S // vector

    lane_bytes = vector_bits // 8
    issue = steps * T * lane_bytes

    # Segments: per (vector step, warp), distinct segment ids over active
    # lanes' [base, base + vector*elem_bytes - 1] byte spans.
    bases = table[:, ::vector] * elem_bytes  # (T, steps)
    active = lead[:, ::vector] if is_store else np.ones_like(bases, dtype=bool)
    num_warps = (T + warp_size - 1) // warp_size
    segments = 0
    for q in range(steps):
        for w in range(num_warps):
            lanes = slice(w * warp_size, min((w + 1) * warp_size, T))
            base = bases[lanes, q][active[lanes, q]]
            if base.size == 0:
                continue
            first = base // segment_bytes
            last = (base + vector * elem_bytes - 1) // segment_bytes
            segments += np.unique(np.concatenate([first, last])).size
    bw = segments * segment_bytes
    return OracleScore(vector=vector, issue=int(issue), bw=int(bw), segments=int(segments))
