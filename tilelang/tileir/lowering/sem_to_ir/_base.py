"""SemanticIR -> TileIR lowering: shared foundation.

This module holds the registries (``IMPL`` / ``TILE_OP_IMPL``), the ``@impl`` /
``@tile_op_impl`` decorators, the ``LoweringScope`` buffer/binding table, and the
small dtype / TileType helper functions shared by every other module in this
package.  It imports no sibling module (it is the bottom of the import graph).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from collections.abc import Callable

from tvm import tirx as _tirx

from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.errors import _UnsupportedTileIRNode
from tilelang.tileir.ir.types import MemSpace, TileType, dtype as _lookup_dtype_raw
from tilelang.tileir.ir.value import Value, fresh_value
from tilelang.tileir.tir_analysis import _canonical_dtype_name
from tilelang.tileir.semantic import SemanticBuffer, SemanticKernel, SemanticProgram, SemanticStmt


def lookup_dtype(name: Any):
    """Look up a TileIR ``DType`` from a TIR dtype string.

    Normalizes the TVM ``custom[<name>]`` wrapper first (e.g. TVM prints
    tfloat32 as ``custom[tfloat32]``) via ``_canonical_dtype_name`` so the
    alias map in ``ir.types`` resolves it (``tfloat32`` -> ``float32``
    storage; tf32 is a compute-only format materialized by the Gemm emit).
    Without this, ``custom[tfloat32]`` raised KeyError.
    """
    return _lookup_dtype_raw(_canonical_dtype_name(name))


def _canonical_dtype_str(name: Any) -> str:
    """Canonical dtype NAME string for op attributes (e.g. ``Cast.dtype``).

    Resolves the ``custom[...]`` wrapper and dtype aliases through the same
    table as ``lookup_dtype`` (``custom[tfloat32]`` → ``float32``); unknown
    names pass through unchanged so downstream raises stay specific.
    """
    try:
        return lookup_dtype(name).name
    except KeyError:
        return str(name)


IMPL: dict[str, Callable] = {}
TILE_OP_IMPL: dict[str, Callable] = {}


def impl(kind: str) -> Callable:
    """Decorator: ``@impl("for")`` registers a SemanticStmt kind handler."""

    def decorator(fn: Callable) -> Callable:
        IMPL[kind] = fn
        return fn

    return decorator


def tile_op_impl(*names: str) -> Callable:
    """Decorator: ``@tile_op_impl("tl.tileop.copy", ...)`` registers a tile-op handler."""

    def decorator(fn: Callable) -> Callable:
        for name in names:
            TILE_OP_IMPL[name] = fn
        return fn

    return decorator


# Helpers


def _parse_bool(value: Any, default: bool = False) -> bool:
    """Parse a TIR attr string to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip()
    if s.isdigit():
        return bool(int(s))
    return s.lower() in ("true", "1", "yes")


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _sem_buffer_to_tile_type(buf: SemanticBuffer) -> TileType:
    """Map a SemanticBuffer to a TileType.

    Rank handling
    -------------
    Dynamic-shape GLOBAL buffers (rank mismatch):
        For GLOBAL buffers, preserve the FULL rank by using ``-1`` as a
        sentinel for dynamic (non-integer) dimensions.  This ensures
        ``ndim = len(shape)`` matches the original TIR buffer rank so that
        ``_materialize_buffer`` creates a TensorView of the correct rank.
        ``_all_static`` in emission_utils.py checks ``d > 0`` and therefore
        treats ``-1`` as dynamic (falls into the dynamic path).

    Non-power-of-2 SHARED/REGISTER alloc dims:
        For SHARED and REGISTER buffers, pad each dimension to the next
        power of 2.  This prevents the CUDA Tile IR optimizer from rejecting
        tile ops with non-power-of-2 shapes (e.g. ``tile<9xi64>`` for mbars).
    """
    # Detect unsigned-ness before alias collapse (uint8 → int8 in the registry).
    try:
        dt = lookup_dtype(buf.dtype)
    except KeyError as exc:
        raise _UnsupportedTileIRNode(f"unsupported dtype {buf.dtype!r} on buffer {buf.name!r}") from exc

    scope_str = buf.scope
    if "shared" in scope_str:
        space = MemSpace.SHARED
    elif "local" in scope_str or "fragment" in scope_str:
        space = MemSpace.REGISTER
    else:
        space = MemSpace.GLOBAL

    if space == MemSpace.GLOBAL:
        # Preserve full rank — use -1 as sentinel for dynamic dims.
        # emission_utils._all_static treats non-positive dims as dynamic.
        shape = tuple(d if isinstance(d, int) and d > 0 else -1 for d in buf.shape)
    else:
        # Keep only integer dims (all-static for alloc buffers), then pad
        # each to the next power of 2.
        int_dims = tuple(d for d in buf.shape if isinstance(d, int))
        shape = tuple(_next_power_of_two(d) for d in int_dims)

    return TileType(dtype=dt, shape=shape, space=space, layout=None)


def _is_unsigned_dtype(dtype_str: str) -> bool:
    """Return True if the raw dtype string denotes an unsigned integer type."""
    return dtype_str in ("uint8", "uint16", "uint32", "uint64", "uint4")


def _buffers_requiring_alloca(sem_kernel: SemanticKernel) -> set[str]:
    """SHARED buffers that must be demoted to ``alloca global`` scratch.

    A SHARED buffer written or atomically updated at a DATA-DEPENDENT index
    (the index expression itself loads from a buffer, e.g. the histogram
    scatter ``s_histogram[inval_int16] += 1``) cannot be an SSA register
    tile — conflicting lanes need real addressable memory with atomics.
    Reads alone never demote: gathers from an SSA tile are handled by
    extract/iota paths, and demoting on reads would pessimize every kernel
    that indexes shared tiles.
    """
    shared_names = {b.name for b in sem_kernel.alloc_buffers if str(b.scope).startswith("shared")}
    if not shared_names:
        return set()

    def _has_buffer_load(e: Any) -> bool:
        found = False

        def _v(n):
            nonlocal found
            if isinstance(n, _tirx.BufferLoad):
                found = True

        _tirx.stmt_functor.post_order_visit(e, _v)
        return found

    _ATOMIC_ELEM_OPS = {
        "tl.atomic_add_elem_op",
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
    }

    demote: set[str] = set()

    def _dst_buffer_load(args: tuple[Any, ...]):
        """BufferLoad under args[0] of an atomic elem-op call."""
        if not args:
            return None
        a0 = args[0]
        if isinstance(a0, _tirx.BufferLoad):
            return a0
        inner = getattr(a0, "args", None)
        if inner and isinstance(inner[0], _tirx.BufferLoad):
            return inner[0]
        return None

    def _demote_from_atomic_args(args: tuple[Any, ...]) -> None:
        bl = _dst_buffer_load(args)
        if bl is not None and bl.buffer.name in shared_names and any(_has_buffer_load(i) for i in bl.indices):
            demote.add(bl.buffer.name)

    def _scan_expr_atomics(e: Any) -> None:
        """Demote shared targets of atomic elem-op calls nested in an
        expression (e.g. ``pos = T.atomic_add(hist[idx], 1, return_prev=True)``,
        where the atomic is the RHS value of a store into another buffer)."""

        def _v(n):
            if isinstance(n, _tirx.Call) and getattr(getattr(n, "op", None), "name", None) in _ATOMIC_ELEM_OPS:
                _demote_from_atomic_args(tuple(n.args))

        _tirx.stmt_functor.post_order_visit(e, _v)

    def _walk(stmt: SemanticStmt) -> None:
        if stmt.kind == "buffer_store":
            buffer_name = dict(stmt.attrs).get("buffer", "")
            if buffer_name in shared_names and any(_has_buffer_load(i) for i in stmt.indices):
                demote.add(buffer_name)
            # A return-value atomic on a shared buffer can sit in the RHS.
            if stmt.value is not None:
                _scan_expr_atomics(stmt.value)
        elif stmt.kind == "atomic_rmw":
            _demote_from_atomic_args(stmt.call_args)
        for child in stmt.children:
            _walk(child)

    _walk(sem_kernel.body)
    return demote


def _binding_var(stmt: Any) -> Any | None:
    """Return the explicit TIR ``Var`` bound by a semantic statement.

    Returns None when no live Var is reachable, in which case callers bind by
    the serialized name instead.

    Binding by the Var OBJECT lets the name-keyed scope distinguish two
    same-named Vars (TVM's legal shadowing), e.g. a kernel's ``blockIdx.x``
    IterVar and a ``T.Persistent`` coordinate the user also writes ``bx``.
    """
    return getattr(stmt, "binding_var", None)


def _make_placeholder(builder: IRBuilder, ty: TileType, name: str | None = None) -> Value:
    """Stamp a fresh placeholder Value from the builder's counter."""
    return fresh_value(builder._counter, ty, name)


def _scalar_bool_type() -> TileType:
    return TileType(dtype=lookup_dtype("bool"), shape=(), space=MemSpace.REGISTER, layout=None)


def _scalar_i32_type() -> TileType:
    return TileType(dtype=lookup_dtype("int32"), shape=(), space=MemSpace.REGISTER, layout=None)


# LoweringScope


class LoweringScope:
    """Buffer table + scalar bindings + loop frames for lowering.

    Parameters
    ----------
    sem_kernel :
        The SemanticKernel being lowered (provides ``alloc_buffers``).
    program :
        The parent SemanticProgram (provides ``params`` for GLOBAL buffers).
        May be ``None`` when lowering a kernel in isolation.
    builder :
        The IRBuilder whose counter is used to assign monotone SSA ids to
        buffer Values.  Required so that buffer ids are >= 0 and unique
        (downstream passes may assume id >= 0).
    """

    def __init__(
        self,
        sem_kernel: SemanticKernel,
        program: SemanticProgram | None = None,
        builder: IRBuilder | None = None,
        fast_math: bool = False,
    ) -> None:
        # fast_math: when True, a*b+c patterns are lowered to FMA (fused multiply-add).
        # When False (precise mode), they are left as separate mul + add operations.
        self.fast_math: bool = fast_math
        # Buffer table: name -> Value
        self._buffers: dict[str, Value] = {}

        # Scalar entry params (non-buffer).  name -> Value (0-d TileType).
        # These are bound into the top-level _binding_stack frame so that lower_expr
        # can resolve them via scope.lookup() when they appear in kernel expressions.
        self._scalar_params: dict[str, Value] = {}

        # Scalar binding stack: list of dicts (innermost last)
        self._binding_stack: list[dict[str, Value]] = [{}]

        # TIR PrimExpr bindings for let-bound variables.
        # Maps var_name → raw TIR PrimExpr (the RHS of the let).
        # Populated by _lower_let so that _compute_partition_indices can
        # substitute let-bound variables (e.g. m_start = bx * block_M)
        # when computing partition indices for Copy/Load ops.
        self._tir_expr_bindings: dict[str, Any] = {}

        # Replay bindings: var_name -> raw TIR PrimExpr, for `tirx.Bind` nodes
        # whose value expression could not be eagerly lowered because it
        # references a variable not yet in scope (e.g. `T.Persistent`'s
        # `Bind(bx, ...)` placed textually before the `for w in
        # range(waves)` loop whose induction variable `w` the value
        # references).  Populated by `_lower_let` on an `_UnboundScopeVariable`
        # from the eager attempt; consumed by `lower_expr`'s `Var` branch,
        # which re-lowers the stored expression on demand at each reference
        # (by which point the loop var is bound).  See
        # `_UnboundScopeVariable` (tilelang/tileir/errors.py) for the full
        # rationale.  Keyed separately from `_tir_expr_bindings` above
        # because that dict's invariant (populated only alongside a
        # concrete `_binding_stack` entry) must not be disturbed.
        self._replay_bindings: dict[str, Any] = {}

        # Raw dtype strings (before alias collapse) for unsigned-ness checks.
        # e.g. "uint8" stays "uint8" here; lookup_dtype("uint8") → int8 (aliased).
        self._raw_dtypes: dict[str, str] = {}

        # Track the actual TIR variable names that correspond to
        # threadIdx.* extents so that _is_thread_index_only_cond can identify
        # warp-specialize guards regardless of the variable name used (e.g. "tid"
        # in minference, not the hardcoded "tx"/"ty"/"tz").
        self._thread_var_names: set[str] = set()

        # Block/thread launch axis ("bx"/"by"/"bz"/...) -> the scope binding key
        # (a TIR Var when available) used by _lower_thread_extent, so the swizzle
        # pass can re-bind the SAME key (identity) instead of shadowing by name.
        self._axis_bind_vars: dict[str, Any] = {}

        # Populate global param buffers
        if program is not None:
            for buf in program.params:
                ty = _sem_buffer_to_tile_type(buf)
                # Override to GLOBAL (params are always global)
                ty = TileType(dtype=ty.dtype, shape=ty.shape, space=MemSpace.GLOBAL, layout=None)
                # Use fresh_value so ids are monotone/unique (>= 0).
                if builder is not None:
                    self._buffers[buf.name] = fresh_value(builder._counter, ty, name=buf.name)
                else:
                    self._buffers[buf.name] = Value(id=-1, type=ty, name=buf.name)
                self._raw_dtypes[buf.name] = buf.dtype

            # Bind scalar (non-buffer) entry params
            # Scalar params are not buffers — represent them as 0-d REGISTER Values
            # and bind them into the top-level scope frame so lower_expr can find them.
            for sp in getattr(program, "scalar_params", ()):
                scalar_ty = _tir_dtype_to_tile_type(sp.dtype)
                if builder is not None:
                    sv = fresh_value(builder._counter, scalar_ty, name=sp.name)
                else:
                    sv = Value(id=-1, type=scalar_ty, name=sp.name)
                self._scalar_params[sp.name] = sv
                # Bind into the outermost frame so expressions in the kernel body
                # that reference this scalar param can resolve it via scope.lookup().
                self._binding_stack[0][sp.name] = sv

        # Populate kernel-local alloc buffers. SHARED buffers with
        # tile-inexpressible access patterns (data-dependent scatter/atomic
        # indices) are DEMOTED to addressable ``alloca global`` scratch: their
        # Value gets GLOBAL space so every pointer path (gather / scatter /
        # ptr atomics) applies unchanged; emit materializes the alloca.
        _alloca_names = _buffers_requiring_alloca(sem_kernel)
        self.alloca_buffer_values: dict[Value, tuple[tuple, str]] = {}
        for buf in sem_kernel.alloc_buffers:
            ty = _sem_buffer_to_tile_type(buf)
            if buf.name in _alloca_names:
                if not all(isinstance(d, int) and d > 0 for d in ty.shape):
                    raise _UnsupportedTileIRNode(f"alloca demotion of `{buf.name}`: requires a static shape, got {buf.shape}.")
                ty = TileType(dtype=ty.dtype, shape=ty.shape, space=MemSpace.GLOBAL, layout=None)
                if builder is not None:
                    val = fresh_value(builder._counter, ty, name=buf.name)
                else:
                    val = Value(id=-1, type=ty, name=buf.name)
                self._buffers[buf.name] = val
                self._raw_dtypes[buf.name] = buf.dtype
                self.alloca_buffer_values[val] = (tuple(int(d) for d in ty.shape), ty.dtype.name)
                continue
            # Use fresh_value so ids are monotone/unique (>= 0).
            if builder is not None:
                self._buffers[buf.name] = fresh_value(builder._counter, ty, name=buf.name)
            else:
                self._buffers[buf.name] = Value(id=-1, type=ty, name=buf.name)
            self._raw_dtypes[buf.name] = buf.dtype

        # Register reshape views (T.reshape / T.view): an alias gets its own
        # buffer Value with the VIEW shape but shares the base buffer's data —
        # lower_kernel records the (alias Value → base Value) relation in
        # block.buffer_aliases and emit redirects tile reads/writes through
        # the base tile with reshapes, keeping a single source of truth.
        self.buffer_alias_values: dict[Value, Value] = {}
        for alias_name, base_name, alias_shape, alias_dtype in getattr(sem_kernel, "buffer_aliases", ()):
            base_val = self._buffers.get(base_name)
            if base_val is None or alias_name in self._buffers:
                continue
            alias_buf = SemanticBuffer(name=alias_name, shape=tuple(alias_shape), dtype=alias_dtype, scope="")
            ty = _sem_buffer_to_tile_type(alias_buf)
            ty = TileType(dtype=ty.dtype, shape=ty.shape, space=base_val.type.space, layout=None)
            if builder is not None:
                alias_val = fresh_value(builder._counter, ty, name=alias_name)
            else:
                alias_val = Value(id=-1, type=ty, name=alias_name)
            self._buffers[alias_name] = alias_val
            self._raw_dtypes[alias_name] = alias_dtype
            self.buffer_alias_values[alias_val] = base_val

    # Frame scoping

    @contextmanager
    def frame(self):
        """Push/pop a binding scope (for let/block scoping)."""
        self._binding_stack.append({})
        try:
            yield
        finally:
            self._binding_stack.pop()

    # Buffer access

    def lookup_buffer(self, name: str) -> Value:
        """Return the Value for a named buffer.

        Raises KeyError if the buffer is not known.
        """
        if name not in self._buffers:
            raise KeyError(f"LoweringScope: unknown buffer {name!r}")
        return self._buffers[name]

    def lookup_raw_dtype(self, name: str) -> str | None:
        """Return the ORIGINAL dtype string for a buffer (before alias collapse).

        uint8 aliases to int8 via lookup_dtype, so unsigned-ness must
        be read from the raw SemanticBuffer.dtype string stored here.
        Returns None if the buffer is unknown.
        """
        return self._raw_dtypes.get(name)

    # Scalar bindings

    # Scalar bindings are keyed by TIR ``Var`` OBJECT IDENTITY when a Var is
    # available, falling back to the variable NAME string otherwise.  TVM lets
    # two distinct Vars share a name (legal shadowing) — e.g. a kernel's
    # ``blockIdx.x`` IterVar and a ``T.Persistent`` coordinate the user writes
    # ``for bx, by in ...`` are BOTH named ``bx`` but are different objects.
    # Keying by identity keeps them distinct so a coordinate never silently
    # resolves to the block axis (a wrong-coordinate miscompile).  Name-keyed
    # bindings (scalar entry params, which never shadow) still resolve via the
    # ``.name`` fallback below.
    @staticmethod
    def _key_and_name(key: Any) -> tuple[Any, str | None]:
        """Split *key* into (identity-key-or-None, name-or-None).

        A ``Var`` yields (the Var, its ``.name``); a plain string yields
        (None, the string)."""
        if isinstance(key, str):
            return None, key
        return key, getattr(key, "name", None)

    @classmethod
    def _lookup_in(cls, store: dict, key: Any) -> Any | None:
        """Look up *key* in a single dict: identity first, then name."""
        ident, name = cls._key_and_name(key)
        if ident is not None and ident in store:
            return store[ident]
        if name is not None and name in store:
            return store[name]
        return None

    def bind(self, key: Any, value: Value) -> None:
        """Bind a scalar (a TIR ``Var`` for identity, or a name string) to a
        Value in the current frame."""
        self._binding_stack[-1][key] = value

    def lookup(self, key: Any) -> Value | None:
        """Look up a scalar binding by ``Var`` identity (name fallback) or name
        (innermost scope wins)."""
        for frame in reversed(self._binding_stack):
            hit = self._lookup_in(frame, key)
            if hit is not None:
                return hit
        return None

    def get_scalar_expr_binding(self, key: Any) -> Any | None:
        """Return the raw TIR PrimExpr for a let-bound variable, or None.

        Populated by _lower_let so that _compute_partition_indices
        can substitute let-bound variables (e.g. ``m_start = bx * block_M``)
        and then call ``_try_divide_expr`` on the substituted expression to
        derive the correct tile-level partition index.
        """
        return self._lookup_in(self._tir_expr_bindings, key)

    def set_replay_binding(self, key: Any, value: Any) -> None:
        """Record *key* as a deferred (replay) binding -- see `_replay_bindings`."""
        self._replay_bindings[key] = value

    def get_replay_binding(self, key: Any) -> Any | None:
        """Return the raw TIR PrimExpr to replay for *key*, or None."""
        return self._lookup_in(self._replay_bindings, key)


# Expression lowering

# Mapping from TIR binary op class to Elementwise fn string.
_BINARY_OP_FN: dict = {}  # populated lazily below to avoid import at module load


def _get_binary_op_fn() -> dict:
    global _BINARY_OP_FN
    if not _BINARY_OP_FN:
        _BINARY_OP_FN = {
            _tirx.Add: "add",
            _tirx.Sub: "sub",
            _tirx.Mul: "mul",
            _tirx.Div: "div",
            _tirx.FloorDiv: "floordiv",
            _tirx.FloorMod: "floormod",
            _tirx.Mod: "floormod",
            _tirx.Min: "min",
            _tirx.Max: "max",
            # Comparisons
            _tirx.EQ: "eq",
            _tirx.NE: "ne",
            _tirx.LT: "lt",
            _tirx.LE: "le",
            _tirx.GT: "gt",
            _tirx.GE: "ge",
            # Logical
            _tirx.And: "andi",
            _tirx.Or: "ori",
        }
    return _BINARY_OP_FN


# TIR Call op-name → Elementwise fn string for unary math.
_UNARY_CALL_FN: dict[str, str] = {
    "tir.exp": "exp",
    "tir.exp2": "exp2",
    "tir.exp10": "exp10",
    "tir.log": "log",
    "tir.log2": "log2",
    "tir.log10": "log10",
    "tir.log1p": "log1p",
    "tir.sqrt": "sqrt",
    "tir.rsqrt": "rsqrt",
    "tir.sin": "sin",
    "tir.cos": "cos",
    "tir.tan": "tan",
    "tir.sinh": "sinh",
    "tir.cosh": "cosh",
    "tir.tanh": "tanh",
    "tir.ceil": "ceil",
    "tir.floor": "floor",
    "tir.fabs": "abs",
    "tir.sigmoid": "sigmoid",
    # "tir.abs" also appears in some TIR variants
    "tir.abs": "abs",
    # Negation
    "tir.neg": "neg",
    "tir.negf": "negf",
}

# Binary comparison fn names → result dtype is bool
_CMP_FNS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


def _tir_dtype_to_tile_type(tir_dtype_str: str) -> TileType:
    """Convert a TIR dtype string to a scalar TileType."""
    dtype_str = str(tir_dtype_str)
    try:
        dt = lookup_dtype(dtype_str)
    except KeyError as exc:
        raise _UnsupportedTileIRNode(f"unsupported dtype {dtype_str!r} in scalar TileType conversion") from exc
    return TileType(dtype=dt, shape=(), space=MemSpace.REGISTER, layout=None)


def _op_name_from_call(call: Any) -> str:
    """Extract the canonical op name from a TIR Call node (matches tir_analysis._op_name)."""
    name = getattr(call.op, "name", str(call.op))
    if name.startswith("tirx."):
        return "tir." + name[len("tirx.") :]
    return name
