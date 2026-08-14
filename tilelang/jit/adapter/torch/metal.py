"""Metal eager adapter (torch.mps.compile_shader) with a host-call-site launch plan.

This adapter is the supported Metal execution path for the torch backend. It
launches every kernel of a (possibly multi-kernel) module in the exact order
of the lowered host entry's ``tirx.tvm_call_packed`` call sites, binds every
MSL buffer slot to either a public call-signature parameter, a compiler
generated host allocation (``T.alloc_global`` workspace), or a runtime
dynamic symbol (symbolic scalar such as ``N``), allocates ``out_idx`` outputs
and compiler-generated buffers, and pins every buffer of a launch batch with
a completion fence until the enqueued work finishes.

Design notes:

- Scalar interleave: shape resolution only ever consumes actual tensor
  arguments; scalars stay in the positional ``full`` binding but never
  contribute shapes.
- Dynamic ``out_idx``: a runtime symbol table keyed by ``tirx.Var``
  identity is built from caller-supplied tensor inputs only; output /
  workspace dimensions (``IntImm``, ``Var``, or general ``PrimExpr`` such as
  ``N + 1``) are evaluated against it, with an explicit error when a symbol
  cannot be determined.
- Compiler-generated buffers: binding information comes from the
  lowered host call site / allocation semantics (``AllocBuffer`` nodes), not
  from assuming every device parameter back-references a user parameter name.
- Multi-kernel ordering: the launch plan follows the host call sites in
  program order (duplicates preserved, constant-extent host loops expanded
  with the loop variable substituted into call-site arguments and launch
  geometry; statically resolvable conditionals follow their branch;
  runtime-dependent control flow is rejected at plan build time).
- Aborted launch batches: the keepalive fence is established from the
  first successful enqueue via ``try/finally``; an exception mid-batch still
  pins everything already submitted, and event-record failures fall back to
  ``torch.mps.synchronize()``.
- One-shot release: a background reaper thread drops each batch's
  strong refs when its completion event fires, so a single launch does not
  hold tensors until the next launch; the global queue also outlives the
  adapter (destruction path). Reaping is batch-atomic (observation outside
  the lock, removal under the lock with a head re-check), so concurrent
  releasers can never pop a batch they did not observe finished; a query
  exception transitions through ``torch.mps.synchronize()`` (drop) or moves
  the batch to a pinned stuck list out of the head-of-line path.

Call-site contract:

- Every device parameter is bound from the *call site's* actual argument
  (``args[1:1 + len(device_func.params)]``, the same slice the common
  ``wrapper.py`` uses), resolved to public parameters through the FFI
  ``args`` slot index (``tirx.tvm_struct_get(args, i, ...)`` chains) instead
  of device-parameter names, to compiler-generated allocations through the
  ``AllocBuffer`` data-handle ``tirx.Var`` identity, or to constants /
  symbolic scalars / general expressions.
- Grid/block geometry comes from the launch arguments appended after the
  function arguments by ``LowerDeviceKernelLaunch`` (already substituted
  into the caller's scope), mapped to axes by the device function's
  ``tirx.kernel_launch_params`` attribute.
- ``_BufferBinding.symbol`` stores the ``tirx.Var`` object (identity), not a
  name string; the runtime symbol table is keyed by ``tirx.Var`` identity
  with an unambiguous-name fallback only.

Shape and host-loop validation:

- Retained-prefix validation: the trailing-singleton rank relaxation drops
  only trailing declared dims that are all constant 1, then EVERY retained
  dimension of the relaxed param must equal the actual extent -- ``Var``
  dims are bound by identity, constant / general-expression dims are
  validated against the runtime symbol table.  A declared ``(7, 1)`` param
  fed a ``(3,)`` tensor is rejected instead of launching static geometry 7
  over a three-element buffer (GPU out-of-bounds).
- Nested static host loops: ``For`` bounds are substituted with the outer
  constant iterations and simplified before the constant check, so an inner
  bound like ``_i + 1`` is statically enumerable and never mistaken for a
  runtime loop.

Strict extent validation:

- Rank matching is no longer a validation criterion.  EVERY declared
  dimension of EVERY tensor input is validated exactly (constants and
  general expressions must equal the caller's actual extent; ``Var`` dims
  bind by identity) UNLESS the dimension is explicitly declared as a
  *capacity* dimension in the compiled contract (PrimFunc attr
  ``tilelang_capacity_dims``).  Unmarked rank-matched constant dims that
  differ from the actual extent (e.g. declared ``(7,)`` fed a ``(3,)``
  tensor) are rejected before launch.

Explicit capacity dimensions:

- Capacity marking is EXPLICIT opt-in only.  The eager-jit pipeline no
  longer infers capacity dims from tensor annotations that reference
  scalar function parameters (the round-4 HIGH: ``B_q(E, N, (K+1)//2)``
  had its ordinary exact dims ``E``/``N`` auto-exempted).  Eager kernels
  declare capacity dims in the body via
  ``T.annotate_capacity_dims({"A_q": (0,)})``; lazy kernels opt in via
  ``func.with_attr("tilelang_capacity_dims", {"W": (0,)})``.  Everything
  unmarked stays strictly validated.
- The adapter runs an advisory guard audit on every explicitly declared
  capacity dim: it collects structural evidence (condition comparisons
  against non-declared bounds, runtime-bounded loops, min clamps) for
  every access of the marked buffer along the marked dim and warns when a
  dim has no accesses or ANY access lacks guard evidence.  The audit is
  evidence, never a rejection: the explicit declaration is the trust
  boundary. Both caller directions
  are legal for a capacity dim -- declared > actual (padded/masked grouped
  QMM) and declared < actual (active-prefix processing of a larger
  allocation); safety for the former
  rests on the kernel's masks, which the audit advises on.
"""

from __future__ import annotations

import contextlib
import logging
import numbers
import os
import struct
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from tvm import tirx

from tilelang import tvm as tvm

from ..base import BaseKernelAdapter
from tilelang.engine.param import KernelParam

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level keepalive reaper: batches live in a global queue so they
# are reaped on completion regardless of the adapter's lifetime; the reaper
# is a single lazily-started daemon thread.  Reaping is batch-atomic: the
# completion observation happens outside the lock, the removal happens under
# the lock and re-checks that the queue head is still the observed batch, so
# concurrent releasers (reaper / launcher / destructor) can never pop a
# batch they did not observe finished.
# ---------------------------------------------------------------------------
_KEEPALIVE_POLL_SECONDS = float(os.environ.get("TILELANG_METAL_KEEPALIVE_POLL_MS", "20")) / 1000.0

_pending_keepalive: deque[tuple[tuple[torch.Tensor, ...], Any]] = deque()
# Batches whose event query raised and whose synchronize fallback also
# failed (MPS teardown): kept pinned (fail-safe) but out of the head-of-line
# path, retried on later drains.
_stuck_keepalive: list[tuple[tuple[torch.Tensor, ...], Any]] = []
_keepalive_lock = threading.Lock()
_reaper_lock = threading.Lock()
_reaper_thread: threading.Thread | None = None
# Wake event for the reaper: appended batches set it so the reaper drains
# immediately instead of waiting out the poll interval; tests pause reaping
# by setting ``_KEEPALIVE_POLL_SECONDS`` large and wake it by setting this.
_reaper_wakeup = threading.Event()


def _retry_stuck_keepalive() -> None:
    """Release stuck batches once a later query proves them finished.

    Batches in ``_stuck_keepalive`` stay strongly referenced (fail-safe
    direction) until a query succeeds; removal is per batch identity (the
    event object).
    """
    if not _stuck_keepalive:
        return
    for _batch, ev in list(_stuck_keepalive):
        try:
            done = bool(ev.query())
        except Exception:
            done = False
        if done:
            with _keepalive_lock:
                _stuck_keepalive[:] = [e for e in _stuck_keepalive if e[1] is not ev]


def _release_finished_work() -> None:
    """Drop keepalive batches whose completion event has already fired.

    Observation (``ev.query()``) happens *outside* the lock; removal happens
    *under* the lock and re-checks that the queue head is still the observed
    batch.  Two concurrent releasers therefore pop at most one batch each,
    and a batch is only ever removed after it was observed finished (no
    second ``popleft`` can steal the next, still-running batch).

    A query exception is a state transition, not a stall: ``synchronize()``
    proves all enqueued work completed, so the batch can be dropped; if the
    synchronize itself fails (MPS teardown), the batch is moved to the stuck
    list -- still pinned, but no longer blocking the queue head.
    """
    while True:
        with _keepalive_lock:
            if not _pending_keepalive:
                break
            batch, ev = _pending_keepalive[0]
        try:
            done = bool(ev.query())
        except Exception:
            done = None
        if done is True:
            with _keepalive_lock:
                if _pending_keepalive and _pending_keepalive[0][1] is ev:
                    _pending_keepalive.popleft()
            continue
        if done is False:
            break
        # Query error: transition through a full synchronize; the sync
        # completes (or fails for) every enqueued batch.
        try:
            torch.mps.synchronize()
        except Exception:
            with _keepalive_lock:
                if _pending_keepalive and _pending_keepalive[0][1] is ev:
                    _pending_keepalive.popleft()
                    _stuck_keepalive.append((batch, ev))
            continue
        with _keepalive_lock:
            if _pending_keepalive and _pending_keepalive[0][1] is ev:
                _pending_keepalive.popleft()
        continue
    _retry_stuck_keepalive()


def _track_keepalive(refs: list[torch.Tensor]) -> None:
    """Pin ``refs`` (deduplicated) until the enqueued work completes.

    A completion event is recorded on the MPS stream after the batch was
    enqueued; the background reaper drops the strong refs once the event
    fires. If the event cannot be created/recorded, synchronize instead so
    dropping the refs cannot dangle.
    """
    unique: list[torch.Tensor] = []
    seen: set[int] = set()
    for r in refs:
        if isinstance(r, torch.Tensor) and id(r) not in seen:
            seen.add(id(r))
            unique.append(r)
    if not unique:
        return
    try:
        ev = torch.mps.Event()
        ev.record()
    except Exception:
        # Fence unavailable (MPS teardown / allocation failure): block until
        # the enqueued work finishes, then dropping the refs is safe.
        torch.mps.synchronize()
        return
    _pending_keepalive.append((tuple(unique), ev))
    _ensure_reaper()
    # Wake the reaper immediately: it drains on the event instead of
    # waiting out the whole poll interval (also makes test pause/resume
    # deterministic).
    _reaper_wakeup.set()


def _reaper_loop() -> None:
    while True:
        with contextlib.suppress(Exception):
            _release_finished_work()
        # Interruptible sleep: ``_KEEPALIVE_POLL_SECONDS`` is read at call
        # time (tests pause reaping by updating the module global) and a new
        # batch wakes the reaper immediately via ``_reaper_wakeup``.
        _reaper_wakeup.wait(timeout=_KEEPALIVE_POLL_SECONDS)
        _reaper_wakeup.clear()


def _ensure_reaper() -> None:
    global _reaper_thread
    with _reaper_lock:
        if _reaper_thread is None or not _reaper_thread.is_alive():
            _reaper_thread = threading.Thread(
                target=_reaper_loop,
                name="tilelang-metal-keepalive",
                daemon=True,
            )
            _reaper_thread.start()


# ---------------------------------------------------------------------------
# Module-level launch-time resolvers.  These are plain functions (no ``self``)
# so the launcher closure never captures the adapter, keeping the destruction
# path deterministic with provable ``__del__`` cleanup.
# ---------------------------------------------------------------------------
def _symbol_value(var: Any, symtab: dict[Any, int]) -> int:
    """Resolve a dynamic scalar symbol by ``tirx.Var`` identity.

    The call-site symbol is normally the same ``tirx.Var`` object that
    appears in the public parameter shapes (identity key).  When a lowering
    step produced a distinct object, fall back to the name only if it is
    unambiguous -- two same-named symbols of different identity are an error,
    never a silent first match.
    """
    value = symtab.get(var)
    if value is not None:
        return value
    name = str(var)
    matches = [v for k, v in symtab.items() if str(k) == name]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"Metal adapter: dynamic scalar '{name}' is not determined by any caller-supplied tensor input shape")


def _pack_scalar_args(scalar_slots: list[tuple[Any, Any]], device: Any) -> torch.Tensor:
    """Pack runtime scalar kernel arguments into the Metal ``args_t`` buffer.

    The Metal codegen emits every scalar function parameter inside a single
    packed struct bound at ``buffer(num_buffer)``, with one 8-byte slot per
    scalar (32-bit values in the low 4 bytes, 64-bit values in the full
    slot), matching the TVM runtime's ``ArgUnion64`` packing.  The
    ``torch.mps.compile_shader`` launcher, however, binds each positional
    argument to its own buffer slot, so passing ``m`` separate Python
    scalars only ever lands the first one inside the struct and silently
    misreads the rest (for example, ``kern(a, 5, 7)`` can produce ``a + 5000``).
    Packing every scalar into a single tensor reproduces the struct layout
    exactly and binds it to the struct buffer slot.
    """
    if not scalar_slots:
        raise RuntimeError("Metal adapter: internal error, no scalar slots to pack")
    buf = bytearray(len(scalar_slots) * 8)
    for i, (dtype, value) in enumerate(scalar_slots):
        bits = int(dtype.bits)
        if bits == 32:
            if str(dtype) == "int32":
                struct.pack_into("<i", buf, i * 8, int(value))
            elif str(dtype) == "uint32":
                struct.pack_into("<I", buf, i * 8, int(value))
            elif str(dtype) == "float32":
                struct.pack_into("<f", buf, i * 8, float(value))
            else:
                raise RuntimeError(f"Metal adapter: unsupported 32-bit scalar kernel parameter dtype '{dtype}'")
        elif bits == 64:
            if str(dtype) == "int64":
                struct.pack_into("<q", buf, i * 8, int(value))
            elif str(dtype) == "uint64":
                struct.pack_into("<Q", buf, i * 8, int(value))
            elif str(dtype) == "float64":
                struct.pack_into("<d", buf, i * 8, float(value))
            else:
                raise RuntimeError(f"Metal adapter: unsupported 64-bit scalar kernel parameter dtype '{dtype}'")
        else:
            raise RuntimeError(
                f"Metal adapter: unsupported scalar kernel parameter dtype "
                f"'{dtype}' ({bits} bits); the Metal codegen can only "
                "represent 32-bit and 64-bit scalar arguments in the packed "
                "args struct"
            )
    return torch.frombuffer(buf, dtype=torch.uint8).to(device)


def _resolve_int_value(
    expr: Any,
    symtab: dict[Any, int],
    analyzer: Any,
    full: list[Any] | None = None,
    scalar_vars: dict[Any, int] | None = None,
) -> int:
    """Resolve an integer launch / dimension expression at call time.

    ``IntImm`` / ``int`` pass through; a ``tirx.Var`` is resolved from the
    identity-keyed symbol table, from a user scalar call-site argument (via
    ``scalar_vars``/``full``), or from an unambiguous symbol name; general
    ``PrimExpr`` expressions are evaluated by substituting the known values
    and simplifying.
    """
    if isinstance(expr, tirx.IntImm):
        return int(expr)
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tirx.Var):
        if expr in symtab:
            return symtab[expr]
        if scalar_vars is not None and full is not None and expr in scalar_vars:
            return int(full[scalar_vars[expr]])
        name = str(expr)
        matches = [val for k, val in symtab.items() if str(k) == name]
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError(f"Metal adapter: dynamic dimension symbol '{name}' is not determined by any caller-supplied tensor input shape")
    if isinstance(expr, tirx.PrimExpr):
        vmap: dict[Any, Any] = {}
        for var, val in symtab.items():
            vmap[var] = tirx.IntImm(str(var.dtype), int(val))
        if scalar_vars is not None and full is not None:
            for var, slot in scalar_vars.items():
                value = full[slot]
                if isinstance(value, numbers.Integral):
                    vmap[var] = tirx.IntImm(str(var.dtype), int(value))
        substituted = tvm.tirx.stmt_functor.substitute(expr, vmap) if vmap else expr
        simplified = analyzer.simplify(substituted)
        if isinstance(simplified, tirx.IntImm):
            return int(simplified)
        raise RuntimeError(f"Metal adapter: cannot resolve dynamic expression {expr} at launch time (unbound symbols remain)")
    raise RuntimeError(f"Metal adapter: unsupported dimension {expr!r} ({type(expr).__name__})")


def _build_symbol_table(
    tensor_input_shapes: list[tuple[int, ...]],
    params: list[KernelParam],
    result_idx: list[int],
    capacity_dims: dict[int, frozenset[int]] | None = None,
) -> dict[Any, int]:
    """Build a runtime symbol table keyed by ``tirx.Var`` identity.

    Built from caller-supplied tensor inputs only: outputs never contribute,
    shared symbols resolve from the input binding, and conflicting bindings
    of the same symbol are rejected.

    Shape discipline: every declared
    dimension of every tensor input is validated against the caller's actual
    extent -- constants and general expressions must evaluate to the exact
    actual value, ``Var`` dimensions are bound into the symbol table by
    identity (exact by construction).  The only accepted mismatches are:

    - trailing declared dimensions that are all constant 1 (torch
      right-aligned broadcasting, e.g. a ``(T_, 1)`` param fed a flat
      ``(m,)`` tensor): those dimensions are dropped and the retained prefix
      is validated as above -- a declared ``(7, 1)`` param fed a ``(3,)``
      tensor has a 7-element kernel domain over a 3-element buffer and is
      rejected instead of launching out of bounds;
    - dimensions explicitly declared as *capacity* dimensions in the
      compiled contract (``capacity_dims`` map; PrimFunc attr
      ``tilelang_capacity_dims``, explicit opt-in only): the kernel
      contract states the declared extent bounds every access the kernel
      performs (internally masked/offset-guarded), so the caller's actual
      extent may differ -- the legal padded/masked grouped-QMM pattern
      (declared rows=64, actual per-expert slices of 15) and the
      active-prefix pattern (declared active rows, larger allocation).
      Both directions are legal; the guard audit
      advises on mask evidence for the declared > actual direction.

    ``capacity_dims`` maps public parameter positions to the shape-dim
    indices that are contractually capacity declarations; everything else is
    strict. Rank matching is not a criterion: an unmarked
    rank-matched constant dim that differs from the actual extent is a real
    out-of-bounds hazard and is rejected.
    """
    capacity_dims = capacity_dims or {}
    symtab: dict[Any, int] = {}
    tensor_params = [i for i in range(len(params)) if i not in result_idx and not params[i].is_scalar()]
    if len(tensor_params) != len(tensor_input_shapes):
        raise RuntimeError(f"Metal adapter: expected {len(tensor_params)} tensor input shapes, got {len(tensor_input_shapes)}")
    # Phase 1: apply the trailing-singleton relaxation (dropping only
    # trailing declared dims that are all constant 1) and bind ``Var`` dims
    # by identity with conflict rejection.  The truncated declared dims are
    # kept per param for phase-2 validation.
    retained: list[tuple[int, list[Any], tuple[int, ...]]] = []
    for param_idx, shape in zip(tensor_params, tensor_input_shapes):
        param = params[param_idx]
        declared = list(param.shape)
        if len(shape) != len(declared):
            # Allow trailing size-1 declared dims to be implicit (torch
            # right-aligned broadcasting, e.g. a ``(T_, 1)`` param fed a flat
            # ``(m,)`` tensor): the flat memory layout is identical, and the
            # trailing dims are all constant 1.
            if len(shape) < len(declared) and all(
                (isinstance(d, tirx.IntImm) and int(d) == 1) or (isinstance(d, int) and d == 1) for d in declared[len(shape) :]
            ):
                declared = declared[: len(shape)]
            else:
                raise RuntimeError(
                    f"Metal adapter: param {param_idx} declared rank {len(param.shape)} but caller supplied a rank-{len(shape)} tensor"
                )
        for dim, val in zip(declared, shape):
            if isinstance(dim, tirx.Var):
                prev = symtab.get(dim)
                if prev is not None and prev != int(val):
                    raise RuntimeError(
                        f"Metal adapter: dynamic symbol '{dim}' bound to conflicting sizes {prev} and {int(val)} by caller-supplied tensors"
                    )
                symtab[dim] = int(val)
        retained.append((param_idx, declared, tuple(shape)))
    # Phase 2: validate EVERY retained declared dimension of EVERY tensor
    # input (constants and general expressions, e.g. ``N + 1``) against the
    # actual extent, EXCEPT dims explicitly declared as capacity dimensions
    # (``capacity_dims``).  ``Var`` dims were bound in phase 1 (identity) and
    # are exact by construction.  A mismatched extent would launch the
    # static geometry over a buffer of a different size (GPU OOB).  Rank
    # matching or relaxation is NOT a criterion: an unmarked rank-matched
    # constant dim must still match exactly.
    analyzer = tvm.arith.Analyzer()
    for var, val in symtab.items():
        analyzer.bind(var, val)
    for param_idx, declared, shape in retained:
        cap = capacity_dims.get(param_idx, ())
        for dim_idx, (dim, val) in enumerate(zip(declared, shape)):
            if dim_idx in cap:
                # Explicit capacity declaration (PrimFunc attr
                # ``tilelang_capacity_dims``, explicit opt-in): the
                # kernel contract states the declared extent bounds every
                # access (masked/offset-guarded), so the caller's actual
                # extent may differ in EITHER direction -- declared >
                # actual (padded/masked grouped QMM: declared rows=64,
                # actual per-expert slices of 15) or declared < actual
                # (active-prefix processing of a larger allocation).
                # The guard audit advises on mask
                # evidence; the explicit declaration is the trust
                # boundary.
                continue
            if isinstance(dim, tirx.Var):
                continue
            actual = _resolve_int_value(dim, symtab, analyzer)
            if actual != int(val):
                raise RuntimeError(
                    f"Metal adapter: param {param_idx} declared dimension "
                    f"{dim} evaluates to {actual} but caller supplied "
                    f"{int(val)}; declared and actual tensor extents must "
                    "match exactly (a mismatched extent would launch the "
                    "static geometry over a buffer of a different size) "
                    "unless the dimension is explicitly declared as a "
                    "capacity dimension"
                )
    return symtab


def _eval_param_shape(symtab: dict[Any, int], param: Any) -> tuple[int, ...]:
    """Resolve every dimension of ``param`` (KernelParam or tirx.Buffer)."""
    analyzer = tvm.arith.Analyzer()
    for var, val in symtab.items():
        analyzer.bind(var, val)
    return tuple(_resolve_int_value(d, symtab, analyzer) for d in param.shape)


# ---------------------------------------------------------------------------
# Capacity-dimension guard audit
# ---------------------------------------------------------------------------
# Capacity dims are exempt from strict extent validation ONLY when explicitly
# declared (``T.annotate_capacity_dims`` in the eager DSL /
# ``tilelang_capacity_dims`` attr for lazy kernels).  The exemption is a
# contract: the kernel must bound its accesses along the marked dim by a
# runtime mask/offset guard so that a smaller actual buffer is never
# accessed out of bounds.  The audit below gathers STRUCTURAL evidence for
# such guards from the compiled body (best-effort, advisory): it warns when
# a marked dim has no accesses or no guard evidence at all, and never
# rejects -- the explicit declaration is the trust boundary, and the audit
# provides evidence for it.
#
# Evidence rules (an access is guarded when ANY holds):
#   - condition: an enclosing ``Select`` / ``IfThenElse`` condition (only
#     ``And`` conjuncts) UPPER-bounds a subexpression of the access index
#     with a bound that is not the declared extent (not structurally
#     equal to it and not derived from it) and not derived from the index
#     itself. For a
#     capacity dim the danger is an index that runs past the extent, so
#     only comparisons that bound the index from above count (``idx < b``
#     / ``idx <= b`` / ``b > idx`` / ``b >= idx``); lower-bound
#     comparisons such as ``idx >= 0`` or ``idx > m`` do not protect that
#     direction and are not evidence;
#   - loop extent: an enclosing loop whose extent is not the declared
#     extent and whose loop variable occurs in the access index (TIR loops
#     bound their variable from above by construction);
#   - clamp: the access index contains a ``Min`` that bounds it from above
#     with operands all not derived from the declared extent (``Max``
#     clamps from below -- it can only grow the index -- and is not
#     evidence by itself; its children are still searched).
#
# Known limitations (documented): the eager-jit pipeline bakes scalar
# parameters into constants before the adapter sees the PrimFunc, so a
# guard whose bound is ``declared - 1`` (e.g. a clamp to ``rows - 1``) is
# indistinguishable from a runtime bound after baking and counts as weak
# evidence.  The audit is therefore advisory only.
# ---------------------------------------------------------------------------


def _expr_children(expr: Any) -> tuple[Any, ...]:
    """Direct sub-expressions of a TIR expression (best-effort)."""
    if isinstance(expr, tirx.Select):
        return (expr.condition, expr.true_value, expr.false_value)
    if isinstance(expr, tirx.Call):
        return tuple(expr.args)
    if isinstance(expr, tirx.BufferLoad):
        return tuple(expr.indices)
    if isinstance(expr, tirx.Cast):
        return (expr.value,)
    if isinstance(
        expr,
        (
            tirx.Add,
            tirx.Sub,
            tirx.Mul,
            tirx.Div,
            tirx.FloorDiv,
            tirx.FloorMod,
            tirx.Mod,
            tirx.Min,
            tirx.Max,
        ),
    ):
        return (expr.a, expr.b)
    if isinstance(expr, (tirx.LT, tirx.LE, tirx.GT, tirx.GE, tirx.EQ, tirx.NE, tirx.And, tirx.Or)):
        return (expr.a, expr.b)
    if isinstance(expr, tirx.Not):
        return (expr.a,)
    if isinstance(expr, tirx.Let):
        return (expr.value, expr.body)
    if isinstance(expr, tirx.Ramp):
        return (expr.base, expr.stride)
    if isinstance(expr, tirx.Broadcast):
        return (expr.value,)
    return ()


def _struct_eq(a: Any, b: Any) -> bool:
    """Structural equality for the audit's comparisons.

    Vars compare by name + dtype (NOT identity): the eager/lazy pipeline
    redeclares buffers inside the body (``DeclBuffer``), so a symbolic
    shape Var in the body is a different object than the ``buffer_map``
    extent it came from -- a guard against the declared extent is a no-op
    and must not count as evidence, and a runtime bound with a different
    name must stay a runtime bound.  (``tvm.ir.structural_equal(...,
    map_free_vars=True)`` cannot be used: it unifies ANY two free Vars,
    making ``bx`` "contain" ``tx``.)
    """
    if isinstance(a, tirx.Var) and isinstance(b, tirx.Var):
        return a.name == b.name and a.dtype == b.dtype
    try:
        return tvm.ir.structural_equal(a, b)
    except Exception:
        return False


def _structural_contains(expr: Any, sub: Any) -> bool:
    """True iff ``sub`` is structurally equal to ``expr`` or to a subtree
    of ``expr`` (best-effort; analysis failures return False)."""
    if _struct_eq(expr, sub):
        return True
    return any(_structural_contains(child, sub) for child in _expr_children(expr))


def _condition_bounds_access(cond: Any, idx: Any, declared: Any) -> bool:
    """Condition evidence: ``cond`` UPPER-bounds the access index ``idx``
    using direction-aware comparisons. For a capacity dim
    the danger is an index that runs past the extent, so only comparisons
    that bound ``idx`` from above count -- ``idx < b`` / ``idx <= b``
    (index on the left) and ``b > idx`` / ``b >= idx`` (index on the
    right).  Lower-bound comparisons (``idx >= 0``, ``idx > m``,
    ``b < idx``) do not protect that direction and are not evidence.
    ``And`` conjuncts are walked; ``Or`` is not evidence (a disjunct does
    not bound the access by itself)."""
    if isinstance(cond, tirx.And):
        return _condition_bounds_access(cond.a, idx, declared) or _condition_bounds_access(cond.b, idx, declared)
    if isinstance(cond, (tirx.LT, tirx.LE)) and _structural_contains(idx, cond.a):
        # ``a < b`` / ``a <= b``: an index on the left is bounded above.
        return _upper_bound_is_valid(cond.b, idx, declared)
    if isinstance(cond, (tirx.GT, tirx.GE)) and _structural_contains(idx, cond.b):
        # ``a > b`` / ``a >= b``: an index on the right is bounded above.
        return _upper_bound_is_valid(cond.a, idx, declared)
    return False


def _upper_bound_is_valid(bound: Any, idx: Any, declared: Any) -> bool:
    """A comparison bound counts as capacity evidence only when it is not
    the declared extent (``bx < cap`` is a no-op for any smaller actual
    buffer) and not derived from the access index itself (``bx < bx + 1``
    is vacuous)."""
    return not _structural_contains(bound, declared) and not _structural_contains(bound, idx)


def _clamp_evidence(idx: Any, declared: Any) -> bool:
    """Clamp evidence: the access index contains a ``Min`` that bounds it
    from above with operands all not derived from the declared extent
    using direction-aware analysis. ``Max`` clamps from
    below -- it can only grow the index -- so it is not evidence by
    itself; its children are still searched for a genuine ``Min``."""
    if isinstance(idx, tirx.Min):
        if not _structural_contains(idx.a, declared) and not _structural_contains(idx.b, declared):
            return True
        return _clamp_evidence(idx.a, declared) or _clamp_evidence(idx.b, declared)
    if isinstance(idx, tirx.Max):
        return _clamp_evidence(idx.a, declared) or _clamp_evidence(idx.b, declared)
    return any(_clamp_evidence(child, declared) for child in _expr_children(idx))


def _access_guarded(idx: Any, conds: list[Any], loops: list[tuple[Any, Any]], declared: Any) -> bool:
    """Guard evidence for one access index (see module comment)."""
    if _clamp_evidence(idx, declared):
        return True
    for cond in conds:
        if _condition_bounds_access(cond, idx, declared):
            return True
    for var, extent in loops:
        if _structural_contains(extent, declared):
            continue
        if _structural_contains(idx, var):
            return True
    return False


@tirx.functor.visitor
class _CapacityGuardAuditor(tirx.PyStmtExprVisitor):
    """Structural guard-evidence collector for one (buffer, dim) pair.

    Records every access of the target buffer along the marked dim
    (``BufferLoad`` / ``BufferStore`` / ``BufferRegion`` inside tile-op
    calls) together with the enclosing condition / loop context, and
    classifies each access as guarded or unguarded (see module comment for
    the evidence rules).  Runs once per marked dim at adapter construction.

    The hooked node types recurse MANUALLY (no ``super()``): the C++
    ``PyStmtExprVisitor`` default double-traverses a subtree when a
    registered Python hook calls the default for the same node, which would
    double-count every access.  Non-hooked node types keep the C++ default
    (single traversal).
    """

    def __init__(self, buffer_name: str, dim_idx: int, declared: Any):
        super().__init__()
        self._buffer_name = buffer_name
        self._dim_idx = dim_idx
        self._declared = declared
        self._conds: list[Any] = []
        self._loops: list[tuple[Any, Any]] = []
        self.accesses = 0
        self.guarded = 0

    def _is_target(self, buf: Any) -> bool:
        try:
            return str(buf.name) == self._buffer_name
        except Exception:
            return False

    def _record(self, idx: Any) -> None:
        self.accesses += 1
        if _access_guarded(idx, self._conds, self._loops, self._declared):
            self.guarded += 1

    # -- hooks (manual recursion; see class docstring) -------------------

    def visit_if_then_else_(self, op: tirx.IfThenElse) -> None:
        self.visit_expr(op.condition)
        self._conds.append(op.condition)
        try:
            self.visit_stmt(op.then_case)
            if op.else_case is not None:
                self.visit_stmt(op.else_case)
        finally:
            self._conds.pop()

    def visit_select_(self, op: tirx.Select) -> None:
        self.visit_expr(op.condition)
        self._conds.append(op.condition)
        try:
            self.visit_expr(op.true_value)
            self.visit_expr(op.false_value)
        finally:
            self._conds.pop()

    def visit_for_(self, op: tirx.For) -> None:
        self.visit_expr(op.min)
        self.visit_expr(op.extent)
        self._loops.append((op.loop_var, op.extent))
        try:
            self.visit_stmt(op.body)
        finally:
            self._loops.pop()

    def visit_buffer_load_(self, op: tirx.BufferLoad) -> None:
        if self._is_target(op.buffer) and len(op.indices) > self._dim_idx:
            self._record(op.indices[self._dim_idx])
        for idx in op.indices:
            self.visit_expr(idx)

    def visit_buffer_store_(self, op: tirx.BufferStore) -> None:
        if self._is_target(op.buffer) and len(op.indices) > self._dim_idx:
            self._record(op.indices[self._dim_idx])
        for idx in op.indices:
            self.visit_expr(idx)
        self.visit_expr(op.value)

    def visit_call_(self, op: tirx.Call) -> None:
        for arg in op.args:
            if isinstance(arg, tirx.BufferRegion):
                self._record_region(arg)
            elif isinstance(arg, tirx.PrimExpr):
                self.visit_expr(arg)

    def _record_region(self, region: tirx.BufferRegion) -> None:
        if not self._is_target(region.buffer):
            return
        try:
            ranges = region.region
            if len(ranges) > self._dim_idx:
                self._record(ranges[self._dim_idx].min)
            # Region bounds may themselves contain loads; audit them too.
            for rng in ranges:
                self.visit_expr(rng.min)
                self.visit_expr(rng.extent)
        except Exception:
            pass


def _audit_capacity_guards(
    pf: tirx.PrimFunc,
    capacity_dims: dict[int, frozenset[int]],
    name_to_pos: dict[str, int],
) -> None:
    """Advisory guard audit for explicitly declared capacity dims.

    Emits a warning when a marked (param, dim) has no accesses in the
    kernel body or no guard evidence for any of its accesses.  Never
    rejects: the explicit declaration is the trust boundary.
    """
    if not capacity_dims:
        return
    for _var, buf in pf.buffer_map.items():
        bname = str(buf.name)
        pos = name_to_pos.get(bname)
        if pos is None:
            continue
        dims = capacity_dims.get(pos)
        if not dims:
            continue
        for dim_idx in sorted(int(d) for d in dims):
            if dim_idx >= len(buf.shape):
                continue
            auditor = _CapacityGuardAuditor(bname, dim_idx, buf.shape[dim_idx])
            try:
                auditor.visit_stmt(pf.body)
            except Exception as exc:  # advisory: never break compilation
                logger.warning(
                    "Metal adapter: capacity guard audit failed for '%s' dim %d (%s); the declaration remains in effect",
                    bname,
                    dim_idx,
                    exc,
                )
                continue
            if auditor.accesses == 0:
                logger.warning(
                    "Metal adapter: capacity dim %d of '%s' has no accesses "
                    "in the kernel body; the capacity declaration cannot be "
                    "verified against mask/offset guards",
                    dim_idx,
                    bname,
                )
            elif auditor.guarded < auditor.accesses:
                logger.warning(
                    "Metal adapter: capacity dim %d of '%s': %d of %d "
                    "access(es) have no mask/offset guard evidence; the "
                    "declared extent is an upper bound only if every "
                    "access is bounded by a runtime guard",
                    dim_idx,
                    bname,
                    auditor.accesses - auditor.guarded,
                    auditor.accesses,
                )


@dataclass(frozen=True)
class _BufferBinding:
    """One MSL buffer slot of one kernel call site.

    ``kind`` is one of:
      - "user": public call-signature parameter at ``param_index`` (tensor or
        scalar value, both passed through positionally);
      - "alloc": compiler-generated host allocation (``T.alloc_global``
        workspace) described by ``buffer`` (tirx.Buffer with shape/dtype);
      - "symbol": runtime dynamic scalar (e.g. the symbolic ``N`` threaded
        through the device signature); ``symbol`` is the ``tirx.Var`` object
        (identity), resolved from the runtime symbol table at launch time;
      - "const": a constant integer known at plan build time (``value``);
      - "expr": a general expression (``value``) evaluated against the
        runtime symbol table / user scalar values at launch time.
    """

    kind: str
    param_index: int = -1
    buffer: Any = None
    symbol: Any = None
    value: Any = None
    dtype: Any = None


@dataclass(frozen=True)
class _LaunchSite:
    """One host call site: symbol + launch geometry + per-buffer bindings.

    ``block`` / ``grid`` are expressions derived from the *call-site* launch
    arguments (already substituted into the caller's scope by
    ``LowerDeviceKernelLaunch``), evaluated per launch.  ``scalar_vars`` maps
    call-site ``tirx.Var`` objects of user scalar parameters to their public
    parameter index, for expression evaluation.
    """

    symbol: str
    block: tuple[Any, Any, Any]
    grid: tuple[Any, Any, Any]
    bindings: tuple[_BufferBinding, ...]
    scalar_vars: tuple[tuple[Any, int], ...] = ()


class MetalKernelAdapter(BaseKernelAdapter):
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        #  target: Union[str, Target],
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        kernel_global_source: str | None = None,
        verbose: bool = False,
        #  pass_configs: Optional[Dict[str, Any]] = None,
        #  compile_flags: Optional[List[str]] = None
    ):
        self.kernel_global_source = kernel_global_source
        self.host_mod = host_mod
        self.device_mod = device_mod
        if isinstance(func_or_mod, tirx.PrimFunc):
            func_name = func_or_mod.attrs["global_symbol"]
        else:
            func_name = func_or_mod.__name__
        self.kernel_name = func_name + "_kernel"
        self.verbose = verbose
        # Explicit capacity-dimension contract markers: capacity
        # dims are exempt from strict extent validation ONLY when the kernel
        # author explicitly declares them -- eager kernels via
        # ``T.annotate_capacity_dims({"A_q": (0,)})`` in the body, lazy
        # kernels via ``func.with_attr("tilelang_capacity_dims", {"W": (0,)})``.
        # The syntactic auto-inference from tensor annotations was removed
        # because ordinary exact dims that reference scalar parameters must
        # not be auto-exempted.
        # Keyed by public parameter position (the ``params``/call-site
        # order).  Must be set BEFORE ``super().__init__``: the base
        # ``_post_init`` builds the launcher closure, which captures this
        # map.
        self._capacity_dims: dict[int, frozenset[int]] = {}
        attrs = getattr(func_or_mod, "attrs", None) or {}
        cap_attr = attrs.get("tilelang_capacity_dims")
        name_to_pos: dict[str, int] = {}
        if isinstance(func_or_mod, tirx.PrimFunc):
            # Public param names come from the BUFFER (``buffer_map`` value),
            # not from ``params``: the lazy @T.prim_func path renames the FFI
            # params to ``W_handle`` while the buffers keep the public name.
            for i, var in enumerate(func_or_mod.params):
                buf = func_or_mod.buffer_map.get(var)
                if buf is not None:
                    name_to_pos[str(buf.name)] = i
        if cap_attr and isinstance(func_or_mod, tirx.PrimFunc):
            unknown_names = sorted(str(pname) for pname in cap_attr if str(pname) not in name_to_pos)
            if unknown_names:
                raise ValueError(
                    "Metal adapter: tilelang_capacity_dims contains unknown "
                    f"tensor parameter name(s) {unknown_names}; declared "
                    f"tensor parameters are {sorted(name_to_pos)}"
                )
            for pname, dims in cap_attr.items():
                pos = name_to_pos.get(str(pname))
                if pos is not None and not params[pos].is_scalar():
                    self._capacity_dims[pos] = frozenset(int(d) for d in dims)
            # Advisory guard audit: structural mask/offset-guard
            # evidence for every explicitly declared capacity dim (warns on
            # absence; never rejects).
            _audit_capacity_guards(func_or_mod, self._capacity_dims, name_to_pos)
        super().__init__(func_or_mod, result_idx=result_idx, params=params)

    _kernel = None

    def __del__(self) -> None:
        # Best-effort: drop completed batches promptly at adapter destruction.
        # Pending (in-flight) batches stay in the module-global queue and are
        # reaped by the background thread on completion.
        with contextlib.suppress(Exception):
            _release_finished_work()

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        if kernel_only:
            # Return just the kernel function body, stripping Metal
            # module-level boilerplate (includes, structs, etc.).
            idx = self.kernel_global_source.find("kernel void ")
            if idx >= 0:
                return self.kernel_global_source[idx:]
        return self.kernel_global_source

    # ------------------------------------------------------------------
    # Host call-site analysis
    # ------------------------------------------------------------------
    def _host_entry_funcs(self) -> list[Any]:
        """Lowered host functions whose bodies contain the kernel calls."""
        if self.host_mod is None:
            raise RuntimeError(
                "Metal adapter: host_mod is required to build the launch plan (host call-site order); pass artifact.host_mod to the adapter"
            )
        funcs = list(self.host_mod.functions.values())
        entries = [f for f in funcs if f.attrs.get("tirx.is_entry_func")]
        return entries or funcs

    @staticmethod
    def _is_kernel_packed_call(call: Any) -> bool:
        return getattr(call.op, "name", None) == "tirx.tvm_call_packed"

    @staticmethod
    def _const_int(expr: Any) -> int | None:
        if isinstance(expr, tirx.IntImm):
            return int(expr)
        if isinstance(expr, int):
            return expr
        return None

    @staticmethod
    def _entry_args_var(entry: Any) -> Any:
        """The FFI packed-``args`` handle of a host entry function, if any.

        Call-site buffer/scalar arguments are resolved to public parameters
        through the ``args`` slot index of the FFI entry (the packed-ABI
        slot order is the public parameter order), so binding never depends
        on parameter names.
        """
        for param in entry.params:
            if str(param) == "args":
                return param
        return None

    def _loop_substitute(self, expr: Any, loop_vmap: dict[Any, Any]) -> Any:
        """Substitute constant host loop iterations into ``expr``."""
        if not loop_vmap:
            return expr
        return tvm.tirx.stmt_functor.substitute(expr, loop_vmap)

    def _static_condition(self, condition: Any, loop_vmap: dict[Any, Any]) -> bool:
        """Statically resolve an ``IfThenElse`` condition.

        Conditions that depend on runtime values cannot be represented in a
        static launch plan: raise at plan build time instead of silently
        emitting both branches.
        """
        cond = self._loop_substitute(condition, loop_vmap)
        if isinstance(cond, tirx.IntImm):
            return int(cond) != 0
        simplified = tvm.arith.Analyzer().simplify(cond)
        if isinstance(simplified, tirx.IntImm):
            return int(simplified) != 0
        raise RuntimeError(
            f"Metal adapter: host conditional '{condition}' depends on "
            "runtime values and cannot be resolved statically; the Metal "
            "launch plan only supports statically resolvable control flow"
        )

    def _walk_host(
        self,
        node: Any,
        call_sites: list[tuple[str, list[Any], Any, dict[Any, Any]]],
        host_buffers: dict[Any, Any],
        bind_map: dict[Any, Any],
        args_var: Any,
        loop_vmap: dict[Any, Any],
    ) -> None:
        """Collect kernel call sites in program order (duplicates preserved).

        Constant-bounds host loops are expanded with the iteration value
        substituted into every call-site argument and launch argument (the
        ``min`` is honored); statically resolvable ``IfThenElse`` nodes walk
        only their taken branch; runtime-dependent control flow raises.
        ``Bind`` statements are recorded for FFI slot resolution and
        ``AllocBuffer`` nodes for compiler-generated buffer binding.
        """
        if isinstance(node, tirx.Evaluate):
            value = node.value
            if isinstance(value, tirx.Call) and self._is_kernel_packed_call(value):
                args = list(value.args)
                if not args or not isinstance(args[0], tirx.StringImm):
                    return
                if loop_vmap:
                    args = [self._loop_substitute(a, loop_vmap) for a in args]
                # Snapshot the bindings at the call site: a tirx.Bind in a
                # later static loop iteration must not overwrite the bindings
                # used by earlier sites.
                call_sites.append((str(args[0].value), args[1:], args_var, dict(bind_map)))
            return
        if isinstance(node, tirx.SeqStmt):
            for stmt in node.seq:
                self._walk_host(stmt, call_sites, host_buffers, bind_map, args_var, loop_vmap)
            return
        if isinstance(node, tirx.AttrStmt):
            self._walk_host(node.body, call_sites, host_buffers, bind_map, args_var, loop_vmap)
            return
        if isinstance(node, tirx.Bind):
            # Flat tirx Bind: record the value behind the local var (with the
            # current loop iteration substituted, so loop-local bindings that
            # reference the loop variable resolve per iteration).
            bind_map[node.var] = self._loop_substitute(node.value, loop_vmap)
            return
        if isinstance(node, tirx.For):
            # Substitute the outer constant iterations into the bounds and
            # simplify BEFORE the constant check: a nested
            # loop whose bound references the outer loop variable
            # (``T.serial(_i + 1)`` inside ``T.serial(2, 4)``) is statically
            # enumerable and must not be mistaken for a runtime loop.  Bounds
            # that are still non-constant after substitution depend on
            # runtime values and are rejected.
            min_e = node.min if isinstance(node.min, int) else self._loop_substitute(node.min, loop_vmap)
            extent_e = node.extent if isinstance(node.extent, int) else self._loop_substitute(node.extent, loop_vmap)
            analyzer = tvm.arith.Analyzer()
            min_v = self._const_int(analyzer.simplify(min_e))
            extent = self._const_int(analyzer.simplify(extent_e))
            if min_v is None or extent is None:
                raise RuntimeError(
                    f"Metal adapter: host loop with non-constant bounds "
                    f"(min={node.min}, extent={node.extent}) around kernel "
                    "call sites depends on runtime values and cannot be "
                    "represented in the static launch plan; only "
                    "constant-bounds loops are supported"
                )
            for i in range(min_v, min_v + extent):
                inner = dict(loop_vmap)
                inner[node.loop_var] = tirx.IntImm(str(node.loop_var.dtype), i)
                self._walk_host(node.body, call_sites, host_buffers, bind_map, args_var, inner)
            return
        if isinstance(node, tirx.IfThenElse):
            if self._static_condition(node.condition, loop_vmap):
                self._walk_host(node.then_case, call_sites, host_buffers, bind_map, args_var, loop_vmap)
            elif node.else_case is not None:
                self._walk_host(node.else_case, call_sites, host_buffers, bind_map, args_var, loop_vmap)
            return
        # Leaf nodes in this tirx version (no body to descend into):
        #   AllocBuffer (compiler-generated global buffer), DeclBuffer,
        #   AssertStmt. AllocBuffer is recorded for binding resolution
        #   keyed by its data-handle Var identity.
        if isinstance(node, tirx.AllocBuffer):
            host_buffers.setdefault(node.buffer.data, node.buffer)
            return
        if isinstance(node, (tirx.DeclBuffer, tirx.AssertStmt)):
            return
        raise RuntimeError(f"Metal adapter: unsupported host node {type(node).__name__} while collecting kernel call sites")

    def _resolve_slot(self, expr: Any, bind_map: dict[Any, Any], args_var: Any, depth: int = 0) -> int | None:
        """Map an FFI call-site argument back to its public-parameter slot.

        Follows the host lowering's unpacking chain:
        ``X = tvm_struct_get(X_handle, 0, 1, "handle")`` where
        ``X_handle = Select(_, handle_add_byte_offset(tvm_struct_get(args, i, 15)), tvm_struct_get(args, i, 15))``
        (and the scalar variant ``S = Cast(_, tvm_struct_get(args, i, 15))``)
        -- the slot ``i`` is the packed-ABI argument index, i.e. the public
        parameter index.  Returns ``None`` when the expression cannot be
        traced to an ``args`` slot.
        """
        if depth > 24:
            return None
        if isinstance(expr, tirx.Var):
            if args_var is not None and expr.same_as(args_var):
                return None
            value = bind_map.get(expr)
            if value is None:
                return None
            return self._resolve_slot(value, bind_map, args_var, depth + 1)
        if isinstance(expr, tirx.Cast):
            return self._resolve_slot(expr.value, bind_map, args_var, depth + 1)
        if isinstance(expr, tirx.Select):
            then_slot = self._resolve_slot(expr.true_value, bind_map, args_var, depth + 1)
            else_slot = self._resolve_slot(expr.false_value, bind_map, args_var, depth + 1)
            if then_slot is not None and then_slot == else_slot:
                return then_slot
            return None
        if isinstance(expr, tirx.Call):
            name = expr.op.name
            if name == "tirx.tvm_struct_get" and len(expr.args) == 3:
                struct, index, field = expr.args
                index_i = self._const_int(index)
                field_i = self._const_int(field)
                if index_i is None or field_i is None:
                    return None
                if args_var is not None and struct.same_as(args_var):
                    # FFI value slot: field 15 is the union value member
                    # (data handle or scalar int64).
                    if field_i == 15:
                        return index_i
                    return None
                # Data-pointer dereference: tvm_struct_get(handle, 0, 1).
                if index_i == 0 and field_i == 1:
                    return self._resolve_slot(struct, bind_map, args_var, depth + 1)
                return None
            if name == "tirx.handle_add_byte_offset" and len(expr.args) == 2:
                return self._resolve_slot(expr.args[0], bind_map, args_var, depth + 1)
        return None

    def _bind_function_arg(
        self,
        kparam: Any,
        call_arg: Any,
        host_buffers: dict[Any, Any],
        bind_map: dict[Any, Any],
        args_var: Any,
    ) -> tuple[_BufferBinding, Any | None]:
        """Bind one device parameter from its per-call-site argument.

        The device parameter order equals the MSL buffer order, and the host
        call site passes its arguments in that same order (the
        ``args[1:1+len(device_func.params)]`` contract of the common
        wrapper).  Returns ``(binding, scalar_var)`` where ``scalar_var`` is
        the call-site ``tirx.Var`` of a user scalar parameter (needed to
        evaluate launch expressions at call time).
        """
        slot = self._resolve_slot(call_arg, bind_map, args_var)
        if slot is not None:
            scalar_var = call_arg if isinstance(call_arg, tirx.Var) and str(kparam.dtype) != "handle" else None
            return (
                _BufferBinding(
                    kind="user",
                    param_index=slot,
                    dtype=None if str(kparam.dtype) == "handle" else kparam.dtype,
                ),
                scalar_var,
            )
        if isinstance(call_arg, tirx.Var):
            buffer = host_buffers.get(call_arg)
            if buffer is not None:
                return _BufferBinding(kind="alloc", buffer=buffer), None
            if str(kparam.dtype) != "handle":
                # Runtime dynamic scalar (e.g. the symbolic N threaded through
                # the device signature); resolved per launch by Var identity.
                return (
                    _BufferBinding(kind="symbol", symbol=call_arg, dtype=kparam.dtype),
                    None,
                )
            raise RuntimeError(
                f"Metal adapter: kernel parameter '{kparam}' receives buffer "
                f"'{call_arg}' that is neither a public call-signature "
                "argument nor a compiler-generated host allocation; cannot "
                "bind it"
            )
        const = self._const_int(call_arg)
        if const is not None:
            return _BufferBinding(kind="const", value=const, dtype=kparam.dtype), None
        if str(kparam.dtype) != "handle":
            return _BufferBinding(kind="expr", value=call_arg, dtype=kparam.dtype), None
        raise RuntimeError(
            f"Metal adapter: kernel parameter '{kparam}' receives unsupported "
            f"argument {call_arg!r} ({type(call_arg).__name__}); cannot bind it"
        )

    def _launch_plan(self) -> list[_LaunchSite]:
        """Build a per-call-site launch plan from the lowered host module.

        Every kernel is represented once per host call site, in program
        order, with its DCE'd buffer signature (call-site args are in the
        device-parameter order == MSL buffer order) and its own grid/block
        derived from the call-site launch arguments.
        """
        if self.device_mod is None:
            raise RuntimeError("Metal adapter: device_mod is required to build the launch plan")
        host_buffers: dict[Any, Any] = {}
        # Each call site records the entry context (FFI args handle + Bind
        # map) it was collected under, so its arguments resolve against the
        # right entry's slot chain.
        call_sites: list[tuple[str, list[Any], Any, dict[Any, Any]]] = []
        for entry in self._host_entry_funcs():
            args_var = self._entry_args_var(entry)
            bind_map: dict[Any, Any] = {}
            self._walk_host(entry.body, call_sites, host_buffers, bind_map, args_var, {})
        self._host_alloc_buffers = host_buffers

        plan: list[_LaunchSite] = []
        for symbol, call_args, args_var, entry_bind_map in call_sites:
            if symbol not in self.device_mod:
                # Non-kernel packed calls (e.g. __tvm_set_device).
                continue
            func = self.device_mod[symbol]
            kparams = list(func.params)
            if len(call_args) < len(kparams):
                raise RuntimeError(
                    f"Metal adapter: host call site '{symbol}' passes "
                    f"{len(call_args)} buffers but the device function has "
                    f"{len(kparams)} parameters"
                )
            function_args = call_args[: len(kparams)]
            launch_args = call_args[len(kparams) :]
            launch_tags = func.attrs.get("tirx.kernel_launch_params")
            if launch_tags is None:
                raise RuntimeError(
                    f"Metal adapter: device function '{symbol}' has no "
                    "'tirx.kernel_launch_params' attribute; cannot map the "
                    "call-site launch arguments to grid/block geometry"
                )
            launch_tags = list(launch_tags)
            if len(launch_args) != len(launch_tags):
                raise RuntimeError(
                    f"Metal adapter: host call site '{symbol}' passes "
                    f"{len(launch_args)} launch arguments but the device "
                    f"function declares {len(launch_tags)} launch parameters "
                    f"{launch_tags}"
                )

            bindings: list[_BufferBinding] = []
            scalar_vars: list[tuple[Any, int]] = []
            for kp, ca in zip(kparams, function_args):
                binding, scalar_var = self._bind_function_arg(kp, ca, host_buffers, entry_bind_map, args_var)
                if scalar_var is not None:
                    scalar_vars.append((scalar_var, binding.param_index))
                bindings.append(binding)

            # Geometry from the call-site launch arguments (already
            # substituted into the caller's scope by
            # LowerDeviceKernelLaunch), mapped to axes by the launch params.
            # The dynamic-shared-memory tag is skipped exactly like the Metal
            # codegen does (Metal has no dynamic shared memory; the size is
            # baked into the MSL kernel's static threadgroup allocation).
            block = [1, 1, 1]
            grid = [1, 1, 1]
            for tag, arg in zip(launch_tags, launch_args):
                if tag == "tirx.use_dyn_shared_memory":
                    continue
                if tag == "blockIdx.x":
                    grid[0] = arg
                elif tag == "blockIdx.y":
                    grid[1] = arg
                elif tag == "blockIdx.z":
                    grid[2] = arg
                elif tag == "threadIdx.x":
                    block[0] = arg
                elif tag == "threadIdx.y":
                    block[1] = arg
                elif tag == "threadIdx.z":
                    block[2] = arg
                else:
                    raise RuntimeError(f"Metal adapter: unsupported launch parameter tag '{tag}' on device function '{symbol}'")
            plan.append(
                _LaunchSite(
                    symbol=symbol,
                    block=tuple(block),
                    grid=tuple(grid),
                    bindings=tuple(bindings),
                    scalar_vars=tuple(scalar_vars),
                )
            )
        if not plan:
            raise AssertionError(f"no kernel call sites with prefix {self.kernel_name} in host module")
        return plan

    # ------------------------------------------------------------------
    # Dynamic shape resolution
    # ------------------------------------------------------------------
    def _symbol_table(self, tensor_input_shapes: list[tuple[int, ...]]) -> dict[Any, int]:
        return _build_symbol_table(tensor_input_shapes, self.params, self.result_idx, self._capacity_dims)

    def _eval_dim(self, dim: Any, symtab: dict[Any, int], analyzer: Any) -> int:
        return _resolve_int_value(dim, symtab, analyzer)

    def _eval_param_shape(self, symtab: dict[Any, int], param: Any) -> tuple[int, ...]:
        return _eval_param_shape(symtab, param)

    def _resolve_output_shapes(self, tensor_input_shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        """Resolve ``out_idx`` output tensor shapes."""
        symtab = self._symbol_table(tensor_input_shapes)
        return [self._eval_param_shape(symtab, self.params[idx]) for idx in self.result_idx]

    # ------------------------------------------------------------------
    # Launcher
    # ------------------------------------------------------------------
    def _convert_torch_func(self) -> Callable:
        if self._kernel is None:
            _mps_module = torch.mps.compile_shader(self.kernel_global_source)
            _launch_plan = self._launch_plan()
            _symbol_table = _build_symbol_table
            _eval_param_shape_fn = _eval_param_shape
            _resolve_int_value_fn = _resolve_int_value
            _symbol_value_fn = _symbol_value
            _result_idx = list(self.result_idx)
            _params = list(self.params)
            _capacity_dims = dict(self._capacity_dims)
            _input_param_idx = [i for i in range(len(_params)) if i not in _result_idx]

            def launcher(*args: Any) -> Any:
                # `args` are the user-supplied non-output arguments in
                # positional order and may mix tensors and scalars; only
                # actual tensor arguments ever contribute shapes.
                if len(args) != len(_input_param_idx):
                    raise RuntimeError(f"Metal adapter: kernel expects {len(_input_param_idx)} non-output arguments, got {len(args)}")
                tensor_args: list[torch.Tensor] = []
                for i, arg in zip(_input_param_idx, args):
                    if _params[i].is_scalar() == isinstance(arg, torch.Tensor):
                        raise RuntimeError(
                            f"Metal adapter: argument {i} for param "
                            f"{_params[i]} must be a "
                            f"{'tensor' if not _params[i].is_scalar() else 'scalar value'}, "
                            f"got {type(arg).__name__}"
                        )
                    if isinstance(arg, torch.Tensor):
                        tensor_args.append(arg)

                _release_finished_work()

                symtab = _symbol_table(
                    [tuple(a.shape) for a in tensor_args],
                    _params,
                    _result_idx,
                    _capacity_dims,
                )
                analyzer = tvm.arith.Analyzer()
                for var, val in symtab.items():
                    analyzer.bind(var, val)
                out_device = tensor_args[0].device if tensor_args else torch.device("mps")
                full: list[Any] = [None] * len(_params)
                for i, arg in zip(_input_param_idx, args):
                    full[i] = arg

                # Allocate out_idx outputs (mirrors the TVMFFI adapter
                # contract; shapes may be dynamic).
                outputs: list[torch.Tensor] = []
                for idx in _result_idx:
                    shape = _eval_param_shape_fn(symtab, _params[idx])
                    tensor = torch.empty(*shape, dtype=_params[idx].torch_dtype(), device=out_device)
                    full[idx] = tensor
                    outputs.append(tensor)

                # Compiler-generated workspaces: one allocation per host
                # AllocBuffer per launch, shared across call sites (keyed by
                # the tirx.Buffer identity, not its name).
                gen_buffers: dict[Any, torch.Tensor] = {}
                launch_refs: list[torch.Tensor] = []
                submitted = False
                try:
                    for site in _launch_plan:
                        kfn = getattr(_mps_module, site.symbol)
                        scalar_vars = dict(site.scalar_vars)
                        # MSL buffer order is handles first, then a single
                        # packed scalar-args struct at ``buffer(num_buffer)``.
                        # ``torch.mps.compile_shader`` binds each positional
                        # argument to its own buffer slot, so runtime scalars
                        # must be packed into ONE tensor at the struct slot:
                        # passing them individually only lands the first
                        # scalar in the struct and silently misreads the rest
                        # due to the multi-runtime-scalar ABI layout.
                        tensor_kargs: list[torch.Tensor] = []
                        scalar_slots: list[tuple[Any, Any]] = []
                        seen_scalar = False
                        for binding in site.bindings:
                            if binding.kind == "user":
                                value = full[binding.param_index]
                                if binding.dtype is None:
                                    if seen_scalar:
                                        raise RuntimeError(
                                            "Metal adapter: buffer kernel "
                                            "parameter appears after a scalar "
                                            "parameter; the Metal ABI "
                                            "requires handles first"
                                        )
                                    tensor_kargs.append(value)
                                else:
                                    seen_scalar = True
                                    scalar_slots.append((binding.dtype, value))
                            elif binding.kind == "alloc":
                                if seen_scalar:
                                    raise RuntimeError(
                                        "Metal adapter: buffer kernel "
                                        "parameter appears after a scalar "
                                        "parameter; the Metal ABI requires "
                                        "handles first"
                                    )
                                tensor = gen_buffers.get(binding.buffer)
                                if tensor is None:
                                    shape = _eval_param_shape_fn(symtab, binding.buffer)
                                    tensor = torch.empty(
                                        *shape,
                                        dtype=KernelParam(
                                            binding.buffer.dtype,
                                            list(binding.buffer.shape),
                                        ).torch_dtype(),
                                        device=out_device,
                                    )
                                    gen_buffers[binding.buffer] = tensor
                                tensor_kargs.append(tensor)
                            elif binding.kind == "symbol":
                                seen_scalar = True
                                scalar_slots.append(
                                    (
                                        binding.dtype,
                                        _symbol_value_fn(binding.symbol, symtab),
                                    )
                                )
                            elif binding.kind == "const":
                                seen_scalar = True
                                scalar_slots.append((binding.dtype, binding.value))
                            elif binding.kind == "expr":
                                seen_scalar = True
                                scalar_slots.append(
                                    (
                                        binding.dtype,
                                        _resolve_int_value_fn(
                                            binding.value,
                                            symtab,
                                            analyzer,
                                            full,
                                            scalar_vars,
                                        ),
                                    )
                                )
                            else:
                                raise RuntimeError(f"Metal adapter: unknown binding kind {binding.kind!r}")
                        if scalar_slots:
                            tensor_kargs.append(_pack_scalar_args(scalar_slots, out_device))
                        launch_refs.extend(tensor_kargs)
                        block = [_resolve_int_value_fn(b, symtab, analyzer, full, scalar_vars) for b in site.block]
                        grid = [_resolve_int_value_fn(g, symtab, analyzer, full, scalar_vars) for g in site.grid]
                        kfn(
                            *tensor_kargs,
                            threads=[a * b for a, b in zip(block, grid)],
                            group_size=block,
                        )
                        submitted = True
                finally:
                    # From the first successful enqueue on, an exception
                    # mid-batch still establishes the completion fence and
                    # keeps every submitted buffer pinned.
                    if submitted:
                        _track_keepalive(launch_refs)

                if len(outputs) == 1:
                    return outputs[0]
                if outputs:
                    return outputs
                return None

            self._kernel = launcher

        return self._kernel
