"""TileLang VMI namespace for Ascend PTO kernels."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps

from tvm import tirx
from tvm.tirx import Buffer, BufferLoad, BufferRegion
from tvm.tirx.script.builder.ir import bind as _bind, sblock_attr as _sblock_attr
from tvm.script.ir_builder import IRBuilder

from tilelang.language.builtin import access_ptr
from tilelang.language.dtypes import dtype as _dtype
from .frame import inside_simdvf as _inside_simdvf

_Op = tirx.op.Op.get


# Numeric element type codes accepted for VMI register arrays.  These are the
# DLDataType codes int=0, uint=1, float=2, bfloat=4, float8=7..14,
# float6=15..16, float4=17, tfloat32=130 and float4_e2m1_unpacked=131; bool,
# handle and void are excluded.  The set mirrors IsDataArrayElementType in
# transform/common/vmi_alloc.h: every numeric width is allowed and
# instruction-level width limits belong to the individual operations.
_NUMERIC_ELEMENT_TYPE_CODES = frozenset({0, 1, 2, 4, *range(7, 18), 130, 131})


def _is_numeric_element(dtype) -> bool:
    return int(dtype.type_code) in _NUMERIC_ELEMENT_TYPE_CODES


def alloc_local(shape, vreg_type):
    """Allocate a statically indexed array of VMI vector registers.

    ``vreg_type`` is a numeric vector dtype produced by :func:`vreg`; bool/mask
    and handle element types are rejected.  The resulting buffer may only be
    indexed by compile-time values, such as induction variables from
    ``T.unroll(..., explicit=True)`` on the PTO backend.

    A block-level allocation marker records that the array is an explicit VMI
    data allocation, so lowering and code generation do not infer that intent
    from the vector dtype of an ordinary SIMD allocation.
    """
    require_vmi_scope("T.vmi.alloc_local(...)")
    dtype = _dtype(vreg_type)
    if int(getattr(dtype, "lanes", 1)) <= 1:
        raise TypeError(f"T.vmi.alloc_local(...) expects a VMI vector type produced by T.vmi.vreg(...), got {dtype}")
    if not _is_numeric_element(dtype):
        raise TypeError(f"T.vmi.alloc_local(...) requires a numeric non-bool vector element type, got {dtype}")

    dims = shape if _is_sequence(shape) else (shape,)
    normalized_shape = []
    for dim in dims:
        value = _constant_int(dim)
        if value is None:
            raise TypeError(f"T.vmi.alloc_local(...) requires every shape dimension to be a compile-time integer, got {dim!r}")
        if value <= 0:
            raise ValueError(f"T.vmi.alloc_local(...) requires positive shape dimensions, got {value}")
        normalized_shape.append(value)

    if not normalized_shape:
        raise ValueError("T.vmi.alloc_local(...) requires at least one shape dimension")

    from tilelang.language.allocate import alloc_local as _alloc_local

    buffer = _alloc_local(tuple(normalized_shape), dtype, scope="local")
    _sblock_attr({"tl.vmi.alloc_kind": {buffer.data: "data_array"}})
    return buffer


def alloc_var(dtype, *, size):
    """Allocate a mutable VMI data register slot.

    The TIR buffer keeps the numeric element dtype (a single lane is the
    scalar dtype, more lanes are a vector dtype).  A block-level allocation
    marker records that the slot is an explicit VMI data register, so lowering
    and code generation do not have to infer that intent from the vector
    dtype or from a later store.
    """
    lanes = _normalize_lanes(size, context="T.vmi.alloc_var(...)")
    elem = _scalar_dtype(dtype, context="T.vmi.alloc_var(...)")
    from tilelang.language.allocate import alloc_var as _alloc_var

    buffer = _alloc_var(f"{elem}x{lanes}", scope="local.var")
    _sblock_attr({"tl.vmi.alloc_kind": {buffer.data: "data"}})
    return buffer


def _normalize_lanes(lanes: int, *, context: str) -> int:
    if isinstance(lanes, bool) or not isinstance(lanes, int):
        raise TypeError(f"{context} expects lanes to be a positive Python int, got {type(lanes)}")
    if lanes <= 0:
        raise ValueError(f"{context} expects lanes to be positive, got {lanes!r}")
    return lanes


# Formal PTODSL VMI vreg/mask lane counts (see ptodsl VMI_LANE_COUNTS).
_VMI_LANE_COUNTS = (1, 2, 4, 8, 64, 128, 256)


def _require_vmi_lane_count(lanes: int, *, context: str) -> int:
    if lanes not in _VMI_LANE_COUNTS:
        raise ValueError(f"{context} requires lanes to be one of {_VMI_LANE_COUNTS}; got {lanes}")
    return lanes


# PTODSL signed integer names. TileLang/TIR distinguishes signed and unsigned
# integer dtypes; codegen preserves that distinction in the generated PTO type.
_PTO_SIGNED_DTYPE = {
    "si8": "int8",
    "si16": "int16",
    "si32": "int32",
    "si64": "int64",
}


def _scalar_dtype(dtype_like, *, context: str):
    if isinstance(dtype_like, str) and dtype_like in _PTO_SIGNED_DTYPE:
        dtype_like = _PTO_SIGNED_DTYPE[dtype_like]
    dt = _dtype(dtype_like)
    if getattr(dt, "lanes", 1) != 1:
        raise ValueError(f"{context} expects a scalar element dtype, got {dt}")
    return dt


def _pto_to_dtype_annotation(dtype_like) -> str:
    if isinstance(dtype_like, str) and dtype_like in _PTO_SIGNED_DTYPE:
        return dtype_like
    return str(_scalar_dtype(dtype_like, context="T.vmi to_dtype"))


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _wrap_pair(pair):
    if not IRBuilder.is_in_scope():
        raise RuntimeError("VMI pair results require an active IRBuilder")
    return VmiPair(_bind(pair))


def _dtype_of(value):
    if isinstance(value, VmiPair):
        return value.dtype
    if hasattr(value, "dtype"):
        return _dtype(value.dtype)
    if hasattr(value, "buffer") and hasattr(value.buffer, "dtype"):
        return _dtype(value.buffer.dtype)
    return _dtype(tirx.convert(value).dtype)


def _element_dtype_of(value):
    dt = _dtype_of(value)
    return _dtype(str(dt).split("x", 1)[0])


def _lanes_of(value) -> int:
    return int(getattr(_dtype_of(value), "lanes", 1))


def _require_mask(mask, *, context: str):
    if mask is None:
        raise TypeError(f"{context} requires a mask operand")
    return mask


def _require_same_vreg_type(*values, context: str, allow_packed_fp4: bool = False):
    dtypes = [_dtype_of(value) for value in values]
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError(f"{context} requires identical VMI vector types")
    if not allow_packed_fp4:
        _reject_packed_fp4(*values, context=context)
    return dtypes[0]


def _reject_packed_fp4(*values, context: str):
    """Reject FP4 values from VMI operations without packed-FP4 semantics."""
    if any(str(_element_dtype_of(value)) == "float4_e2m1fn" for value in values):
        raise TypeError(f"{context} does not support packed FP4 vectors")


def _require_integer_vector(value, *, context: str):
    if not str(_element_dtype_of(value)).startswith(("int", "uint")):
        raise TypeError(f"{context} requires an integer VMI vector")
    return _dtype_of(value)


def _require_compatible_mask(mask, lanes: int, *, context: str):
    mask = _require_mask(mask, context=context)
    if not str(_dtype_of(mask)).startswith("bool"):
        raise TypeError(f"{context} expects a VMI mask operand")
    if _lanes_of(mask) != lanes:
        raise TypeError(f"{context} requires mask and vector lane counts to match")
    return mask


def _access_ptr_base_load(value):
    """Return the immediate BufferLoad behind a frontend ``tl.access_ptr`` call.

    This is a local type recovery for an explicit pointer expression, not
    general pointer provenance analysis: only the call's own base argument
    (through a direct frontend ``Bind`` alias) is inspected.
    """
    if isinstance(value, tirx.Var):
        from tilelang.language.frame import has_let_value, get_let_value

        if has_let_value(value):
            bound = get_let_value(value)
            if bound is not None and not isinstance(bound, tirx.Var):
                return _access_ptr_base_load(bound)
        return None
    if isinstance(value, tirx.Call):
        op = value.op
        if getattr(op, "name", None) == "tl.access_ptr" and value.args and isinstance(value.args[0], BufferLoad):
            return value.args[0]
    return None


def _require_ub_address(value, *, context: str):
    buffer = None
    base_load = _access_ptr_base_load(value)
    if base_load is not None:
        buffer = base_load.buffer
    elif isinstance(value, (BufferLoad, BufferRegion)):
        buffer = value.buffer
    elif isinstance(value, Buffer):
        buffer = value
    if buffer is not None and buffer.scope() not in {"shared", "shared.dyn", "ub"}:
        raise TypeError(f"{context} requires a UB pointer, got buffer scope {buffer.scope()!r}")
    type_annotation = getattr(value, "type_annotation", None)
    storage_scope = getattr(type_annotation, "storage_scope", None)
    if storage_scope and storage_scope not in {"shared", "shared.dyn", "ub"}:
        raise TypeError(f"{context} requires a UB pointer, got pointer scope {storage_scope!r}")
    return value


def _address_element_dtype(value):
    base_load = _access_ptr_base_load(value)
    if base_load is not None:
        return _dtype(base_load.buffer.dtype)
    if isinstance(value, BufferLoad):
        return _dtype(value.buffer.dtype)
    if isinstance(value, BufferRegion):
        return _dtype(value.buffer.dtype)
    if isinstance(value, Buffer):
        return _dtype(value.dtype)
    type_annotation = getattr(value, "type_annotation", None)
    element_type = getattr(type_annotation, "element_type", None)
    dtype = getattr(element_type, "dtype", None)
    if dtype is not None and str(dtype):
        return _dtype(dtype)
    value_dtype = getattr(value, "dtype", None)
    if value_dtype is not None and str(value_dtype) not in {"", "handle"}:
        try:
            return _element_dtype_of(value)
        except (TypeError, ValueError):
            return None
    return None


def _require_address_element_dtype(value, *, context: str):
    dtype = _address_element_dtype(value)
    if dtype is None:
        raise TypeError(f"{context} requires an address with a known element dtype")
    return dtype


def _normalize_compare_predicate(cmp, *, context: str):
    allowed = {"eq", "ne", "lt", "le", "gt", "ge", "oeq", "one", "olt", "ole", "ogt", "oge"}
    if not isinstance(cmp, str) or cmp not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{context} does not support comparison {cmp!r}; expected one of {expected}")
    return cmp


def _constant_int(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, tirx.IntImm):
        return int(value.value)
    return None


def _annotation_value(value):
    if isinstance(value, str):
        return tirx.StringImm(value)
    return value


def _call_vmi(op_name: str, result_dtype, *args, **annotations):
    attrs = {key: _annotation_value(value) for key, value in annotations.items() if value is not None and key not in {"loc", "ip"}}
    args = tuple(arg for arg in args if arg is not None)
    return tirx.call_intrin(str(result_dtype), _Op(f"tl.vmi.{op_name}"), *args, annotations=attrs or None)


def _scope_guarded(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        require_vmi_scope(f"T.vmi.{fn.__name__}")
        return fn(*args, **kwargs)

    return wrapper


def _normalize_pto_vcvt_rounding(mode, *, context: str, allowed=None):
    token = mode
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    allowed_modes = set(allowed or {"R", "A", "H", "Z"})
    if normalized not in allowed_modes:
        expected = ", ".join(sorted(allowed_modes))
        raise ValueError(f"{context} does not support rounding {mode!r}; expected one of {expected}")
    return normalized


def _validate_pto_load_modes(
    context: str,
    *,
    dist_mode,
    group,
    stride,
    block_stride,
    allow_group_brc: bool,
    allowed_dist_modes,
):
    if dist_mode is not None and dist_mode not in allowed_dist_modes:
        expected = ", ".join(repr(mode) for mode in sorted(allowed_dist_modes, key=str))
        raise TypeError(f"{context} does not support dist_mode={dist_mode!r}; expected one of {expected}")

    if group is not None:
        if dist_mode is not None and (not allow_group_brc or dist_mode != "brc"):
            raise TypeError(f"{context} does not allow dist_mode together with group")
        if block_stride is not None:
            raise TypeError(f"{context} does not allow block_stride together with group")
        if stride is None:
            raise TypeError(f"{context} with group=... requires stride")
        return

    if block_stride is not None:
        if dist_mode is not None:
            raise TypeError(f"{context} does not allow dist_mode together with block_stride")
        if stride is not None:
            raise TypeError(f"{context} does not allow stride together with block_stride")
        return

    if stride is not None:
        raise TypeError(f"{context} accepts stride only when group is provided")


def _zero_offset():
    return tirx.IntImm("int32", 0)


def _looks_like_mask(value) -> bool:
    if value is None:
        return False
    try:
        return str(_dtype_of(value)).startswith("bool")
    except Exception:  # pylint: disable=broad-except
        return False


def _is_unit_stride_1d(buffer) -> bool:
    if len(buffer.shape) != 1:
        return False
    if buffer.strides:
        return len(buffer.strides) == 1 and _constant_int(buffer.strides[0]) == 1
    return True


_UB_SCOPES = frozenset({"shared", "shared.dyn", "ub"})


def _is_tight_buffer(buffer) -> bool:
    """Whether *buffer* is tightly packed row-major.

    Explicit strides must equal the row-major strides; default (empty) strides
    are row-major by construction.  The comparison uses Python integers only:
    a symbolic row-major product could narrow or overflow in its index dtype,
    so it is not treated as an equality proof.
    """
    ndim = len(buffer.shape)
    if ndim == 0:
        return False
    if not buffer.strides:
        return True
    if len(buffer.strides) != ndim:
        return False
    expected = 1
    for dim in range(ndim - 1, -1, -1):
        stride = _constant_int(buffer.strides[dim])
        if stride is None or stride != expected:
            return False
        if dim > 0:
            length = _constant_int(buffer.shape[dim])
            if length is None or length <= 0:
                return False
            expected *= length
    return True


def _whole_buffer_span(buffer, proof):
    """Whole-buffer element span when it can be declared without narrowing.

    A multi-dimensional span must be the product of static positive lengths, so
    the descriptor extent stays an exact Python integer.  A one-dimensional
    span is the length itself, so a symbolic length is accepted when it is
    provably positive; no product is formed and nothing can overflow.
    """
    if not _is_tight_buffer(buffer):
        return None
    if len(buffer.shape) == 1:
        length = _constant_int(buffer.shape[0])
        if length is not None:
            return length if length > 0 else None
        return buffer.shape[0] if proof.can_prove(buffer.shape[0] > 0) else None
    span = 1
    for dim in buffer.shape:
        length = _constant_int(dim)
        if length is None or length <= 0:
            return None
        span *= length
    return span


def _source_relative_offset(source):
    """Element offset of *source* relative to its buffer's logical origin.

    ``Buffer.offset_of`` already folds strides and ``elem_offset``; the origin
    offset (which is exactly ``elem_offset``) is subtracted so the result can
    be added to an ``access_ptr(buffer, ...)`` whose lowering adds
    ``elem_offset`` once.
    """
    if isinstance(source, BufferLoad):
        buffer = source.buffer
        indices = list(source.indices)
    elif isinstance(source, BufferRegion):
        buffer = source.buffer
        indices = [region.min for region in source.region]
    elif isinstance(source, Buffer):
        return _zero_offset()
    else:
        return None
    if not buffer.shape or len(indices) != len(buffer.shape):
        return None
    source_offsets = buffer.offset_of(indices)
    origin_offsets = buffer.offset_of([0] * len(buffer.shape))
    # A single flattened index is the only form whose linear address this
    # frontend contract can relate to one descriptor.  Multi-address /
    # axis-separator buffers have no single linear offset, so return None and
    # let every relocation candidate fail instead of silently reducing to
    # offsets[0].
    if len(source_offsets) != 1 or len(origin_offsets) != 1:
        return None
    return source_offsets[0] - origin_offsets[0]


def _provably_non_negative_starts(source, proof) -> bool:
    """Whether every source axis start is a scalar proven non-negative.

    Relocating a descriptor moves the source position into the instruction
    offset, which bypasses the normal negative-index legalization.  Only
    sources whose starts are provably non-negative may take that path.  The
    caller's per-access proof supplies active loop domains and stable lets.
    """
    if isinstance(source, Buffer):
        return True
    if isinstance(source, BufferLoad):
        starts = list(source.indices)
    elif isinstance(source, BufferRegion):
        starts = [region.min for region in source.region]
    else:
        return False
    return all(getattr(start.dtype, "lanes", 1) == 1 and proof.can_prove(start >= 0) for start in starts)


def _proven_sign(analyzer, expr) -> int:
    """Return 1 for a proven non-negative scalar, -1 for proven negative, else 0."""
    if getattr(expr.dtype, "lanes", 1) != 1:
        return 0
    if analyzer.can_prove(expr >= 0):
        return 1
    if analyzer.can_prove(expr < 0):
        return -1
    return 0


def _vmi_op_name(is_load: bool) -> str:
    return "T.vmi.vload" if is_load else "T.vmi.vstore"


def _descriptor_error(
    source,
    *,
    is_load,
    reason,
    buffer=None,
    size=None,
    group=None,
    stride=None,
    block_stride=None,
    start=None,
    extent=None,
    low=None,
    high=None,
):
    """Build the frontend diagnostic for an unprovable physical footprint."""
    if buffer is None:
        buffer = source if isinstance(source, Buffer) else source.buffer
    strides = tuple(buffer.strides) if buffer.strides else ()
    context = f"buffer {buffer.name!r}: shape={tuple(buffer.shape)}, strides={strides}, elem_offset={buffer.elem_offset}"
    params = []
    if size is not None:
        params.append(f"size={size}")
    if group is not None:
        params.append(f"group={group}")
    if stride is not None:
        params.append(f"stride={stride}")
    if block_stride is not None:
        params.append(f"block_stride={block_stride}")
    if start is not None:
        params.append(f"descriptor_start={tuple(start)}")
    if extent is not None:
        params.append(f"descriptor_extent={extent}")
    if low is not None or high is not None:
        params.append(f"access_envelope=[{low}, {high})")
    if params:
        context += ", " + ", ".join(params)
    return ValueError(
        f"{_vmi_op_name(is_load)} cannot construct a descriptor covering the complete physical access: "
        f"{reason} ({context}). No conservative range was generated; use an actual root buffer with a "
        "known finite range, or an explicit pointer whose declared extent covers this access."
    )


# ---------------------------------------------------------------------------
# Local proof environment and ordinary-access coverage proofs
# ---------------------------------------------------------------------------
# Node types that may participate in a stable scalar integer definition.  A
# BufferLoad (for example a mutable local.var read) or any call is deliberately
# not bound: the frontend only consumes definitions whose value is fixed by the
# surrounding integer arithmetic.  Cast nodes are kept as-is so narrowed or
# overflowing arithmetic is never silently strengthened.
_STABLE_SCALAR_NODE_TYPES = (
    tirx.Var,
    tirx.IntImm,
    tirx.Cast,
    tirx.Add,
    tirx.Sub,
    tirx.Mul,
    tirx.FloorDiv,
    tirx.FloorMod,
    tirx.Min,
    tirx.Max,
)

_COVERAGE_PREFIX = "the existing logical pointer-region analysis does not cover the complete access"


def _is_integer_scalar(expr) -> bool:
    dtype = getattr(expr, "dtype", None)
    if dtype is None or int(getattr(dtype, "lanes", 1)) != 1:
        return False
    return str(dtype).startswith(("int", "uint"))


def _is_stable_integer_arithmetic(expr) -> bool:
    """Whether *expr* is a scalar integer definition safe to bind in a proof."""
    if not isinstance(expr, tirx.PrimExpr) or not _is_integer_scalar(expr):
        return False
    nodes = []
    tirx.stmt_functor.post_order_visit(expr, nodes.append)
    for node in nodes:
        if isinstance(node, (tirx.BufferLoad, tirx.Call)):
            return False
        if not isinstance(node, _STABLE_SCALAR_NODE_TYPES):
            return False
    return True


def _vars_in(expr):
    found = []
    tirx.stmt_functor.post_order_visit(expr, lambda node: found.append(node) if isinstance(node, tirx.Var) else None)
    return found


def _local_analyzer(expressions):
    """Build the per-access analyzer from active loop domains and stable lets.

    Active ``ForFrame`` ``vars/doms`` are bound outer-to-inner; a stable scalar
    integer definition is bound only after its own dependencies, and only when
    the accessed expressions actually reference it.  No global table is kept and
    no definition containing a mutable read or a call is treated as an equality.
    """
    from tvm import arith
    from tvm.script.ir_builder import tirx as _builder_tirx
    from tilelang.language.frame import get_let_value, has_let_value

    proof = arith.Analyzer()
    bound = set()
    frames = IRBuilder.current().frames if IRBuilder.is_in_scope() else ()
    for frame in frames:
        if not isinstance(frame, _builder_tirx.frame.ForFrame):
            continue
        for var, dom in zip(getattr(frame, "vars", ()) or (), getattr(frame, "doms", ()) or ()):
            if var in bound:
                continue
            proof.bind(var, dom)
            bound.add(var)

    done = set()

    def bind_definition(var):
        if var in done:
            return
        done.add(var)
        if var in bound or not has_let_value(var):
            return
        rhs = get_let_value(var)
        if not _is_stable_integer_arithmetic(rhs):
            return
        for dependency in _vars_in(rhs):
            bind_definition(dependency)
        if var in bound:
            return
        proof.bind(var, rhs)
        bound.add(var)

    for expr in expressions:
        if not isinstance(expr, tirx.PrimExpr):
            continue
        for var in _vars_in(expr):
            bind_definition(var)
    return proof


def _coverage_starts(buffer, starts):
    """Normalize only indices the negative-index pass provably rewrites.

    The proof may use ``shape + index`` only when a bare analyzer (no loop or
    let facts) proves the index negative, because the pass applies its own
    rewrite with at least those facts.  A more restricted normalization is not
    allowed: the frontend must not claim a coordinate the pass will not produce.
    """
    from tvm import arith

    bare = arith.Analyzer()
    normalized = []
    for dim, start in enumerate(starts):
        length = _constant_int(buffer.shape[dim])
        if length is not None and _is_integer_scalar(start) and bare.can_prove(start < 0):
            normalized.append(start + length)
        else:
            normalized.append(start)
    return normalized


def _local_coverage_reason(proof, buffer, starts, extent):
    """Return None when the existing row-major carry provably covers ``extent``.

    This mirrors ``LogicalPointerRegion`` for a descriptor that starts at
    ``starts``: a static positive extent is distributed innermost-first, each
    non-outermost dimension consumes ``ceil(rem / length)`` rows, and the start
    offset of every dimension that carries must fit the rows the existing
    analysis declares.  The outermost dimension has no further outward carry, so it is
    never given a mechanical bounds check: a legal outer coordinate (for
    example a dynamic row start) keeps the original expression.  A
    non-constant extent is only accepted for one dimension or when the whole
    declaration is provably inside the innermost dimension, so the proof stays
    valid if a later pass constantizes the extent.  Non-tight layouts have no
    local proof.
    """
    ndim = len(buffer.shape)
    if ndim == 0:
        return f"{_COVERAGE_PREFIX}: the buffer has no dimensions"
    if len(starts) != ndim:
        return f"{_COVERAGE_PREFIX}: the descriptor start arity does not match the buffer rank"
    if not _is_tight_buffer(buffer):
        return (
            f"{_COVERAGE_PREFIX}: the address is a local coordinate of a buffer whose layout is not "
            "proven tight row-major, so no accepted local coverage proof applies"
        )
    normalized = _coverage_starts(buffer, starts)
    constant = _constant_int(extent)
    if constant is None:
        if not proof.can_prove(extent > 0):
            return f"{_COVERAGE_PREFIX}: cannot prove the descriptor extent is positive"
        if ndim > 1:
            dim = ndim - 1
            start = normalized[dim]
            if not proof.can_prove(start >= 0) or not proof.can_prove(start + extent <= buffer.shape[dim]):
                return (
                    f"{_COVERAGE_PREFIX}: a non-constant extent must stay inside the innermost "
                    f"dimension {dim} (length {buffer.shape[dim]}, start {start}), but that inclusion "
                    "cannot be proven"
                )
        return None
    if constant <= 0:
        return f"{_COVERAGE_PREFIX}: the descriptor extent must be positive, got {constant}"
    rem = constant
    # Dim 0 is the outermost axis: there is no further outward carry to declare,
    # so only dimensions that carry into an outer axis are checked.
    for dim in range(ndim - 1, 0, -1):
        if rem <= 1:
            return None
        start = normalized[dim]
        length = _constant_int(buffer.shape[dim])
        if length is None:
            if not proof.can_prove(start >= 0) or not proof.can_prove(start + rem <= buffer.shape[dim]):
                return (
                    f"{_COVERAGE_PREFIX}: {rem} remaining elements must stay inside the non-constant "
                    f"dimension {dim} (start {start}), but that inclusion cannot be proven"
                )
            return None
        if length <= 0:
            return f"{_COVERAGE_PREFIX}: dimension {dim} has non-positive length {length}"
        rnext = (rem + length - 1) // length
        if not proof.can_prove(start >= 0) or not proof.can_prove(start < length) or not proof.can_prove(start + rem <= rnext * length):
            return (
                f"{_COVERAGE_PREFIX}: dimension {dim} (length {length}) starts at {start} and carries "
                f"{rem} elements, but the existing row-major carry only declares {rnext} dimension(s) from "
                "the start"
            )
        rem = rnext
    return None


def _origin_prefix_reason(proof, buffer, extent, *, require_static_shape):
    """Return None when an origin-based descriptor is a stable dense prefix."""
    if len(buffer.shape) == 0:
        return f"{_COVERAGE_PREFIX}: the buffer has no dimensions"
    for dim, length in enumerate(buffer.shape):
        value = _constant_int(length)
        if value is not None and value <= 0:
            return f"{_COVERAGE_PREFIX}: dimension {dim} has non-positive length {value}"
        if require_static_shape and value is None:
            return f"{_COVERAGE_PREFIX}: the origin descriptor has a non-constant dimension {dim}, so no stable dense-prefix proof applies"
    constant = _constant_int(extent)
    if constant is not None:
        if constant <= 0:
            return f"{_COVERAGE_PREFIX}: the descriptor extent must be positive, got {constant}"
        return None
    if not proof.can_prove(extent > 0):
        return f"{_COVERAGE_PREFIX}: cannot prove the descriptor extent is positive"
    return None


def _whole_dense_permutation_reason(buffer, starts, extent):
    """Return None for an exact whole dense-permutation descriptor.

    The layout is accepted when every shape and stride is a static positive
    integer, all descriptor starts are zero, the extent is exactly the whole
    dense span, and the strides are the successive products of the non-singleton
    dimensions sorted by stride.  Singleton axes do not change coverage.
    """
    if not buffer.strides or len(buffer.strides) != len(buffer.shape):
        return "the buffer has no explicit permutation strides"
    if not all(_constant_int(start) == 0 for start in starts):
        return "the descriptor start is not the origin"
    dims = []
    for length in buffer.shape:
        value = _constant_int(length)
        if value is None or value <= 0:
            return "the shape is not static and positive"
        dims.append(value)
    total = 1
    for value in dims:
        total *= value
    if _constant_int(extent) != total:
        return f"the descriptor extent is not exactly the whole dense span {total}"
    axes = []
    for dim, value in enumerate(dims):
        stride = _constant_int(buffer.strides[dim])
        if stride is None or stride < 0:
            return "a stride is not a static non-negative integer"
        if value == 1:
            continue
        axes.append((stride, value))
    axes.sort()
    expected = 1
    for stride, value in axes:
        if stride != expected:
            return f"stride {stride} does not continue the dense permutation at {expected}"
        expected *= value
    return None


def _scalar_start_indices(source, *, buffer, context, ramp_base=False):
    """Extract one scalar integer start per buffer dimension.

    An automatic BufferLoad address follows the public
    ``get_buffer_region_from_load``/``T.access_ptr`` convention: a Ramp index
    contributes only its scalar ``base``.  The Ramp lanes and stride describe
    the vector value, not the VMI physical envelope, so they are never used as
    address bounds.  Explicit ``tl.access_ptr`` descriptors stay scalar-only
    (that frontend already reduced a Ramp base to a scalar).
    """
    if isinstance(source, Buffer):
        dtype = "int32" if not buffer.shape else str(buffer.shape[0].dtype)
        starts = [tirx.IntImm(dtype, 0) for _ in buffer.shape]
    elif isinstance(source, BufferLoad):
        starts = list(source.indices)
    elif isinstance(source, BufferRegion):
        starts = [region.min for region in source.region]
    else:
        starts = []
    if len(starts) != len(buffer.shape):
        raise TypeError(f"{context} requires a full-rank buffer address, got {len(starts)} indices for rank {len(buffer.shape)}")
    if ramp_base:
        starts = [start.base if isinstance(start, tirx.Ramp) else start for start in starts]
    for start in starts:
        if not _is_integer_scalar(start):
            raise TypeError(f"{context} requires scalar integer buffer indices; vector indices cannot be reduced to one address start")
    return starts


class _VmiAddress:
    """The single parsed address record used by an ordinary VMI access."""

    __slots__ = ("kind", "source", "buffer", "starts", "extent", "rw_mask", "call")

    def __init__(self, kind, *, source, buffer, starts, extent=None, rw_mask=None, call=None):
        self.kind = kind
        self.source = source
        self.buffer = buffer
        self.starts = starts
        self.extent = extent
        self.rw_mask = rw_mask
        self.call = call

    @property
    def is_pointer(self) -> bool:
        return self.kind == "pointer"


def _validate_ub_buffer(buffer, *, context):
    """Require the accessed buffer to live in UB with one scalar linear address."""
    scope = buffer.scope()
    if scope not in _UB_SCOPES:
        raise TypeError(f"{context} requires a UB pointer, got buffer scope {scope!r}")
    if buffer.axis_separators:
        raise TypeError(
            f"{context} requires a single scalar linear buffer address, but buffer {buffer.name!r} "
            f"declares axis separators {tuple(buffer.axis_separators)}"
        )


def _validate_linear_address(buffer, starts, *, context):
    """Require ``buffer.offset_of`` to yield exactly one scalar linear offset."""
    try:
        offsets = buffer.offset_of(list(starts))
        origin = buffer.offset_of([0] * len(buffer.shape))
    except Exception as error:  # pylint: disable=broad-except
        raise TypeError(f"{context} requires a single scalar linear buffer address, but offset_of failed: {error}") from error
    if len(offsets) != 1 or len(origin) != 1:
        raise TypeError(
            f"{context} requires a single scalar linear buffer address, got {len(offsets)} offsets "
            "(axis separators and multiple address spaces are not supported)"
        )


def _validate_pointer_annotation(source, buffer, *, context):
    """Check the original handle's PointerType against the parsed buffer, if any.

    Plain handle annotations carry no scope or element information; the buffer
    itself was already validated by :func:`_validate_ub_buffer`.
    """
    annotation = getattr(source, "type_annotation", None)
    storage_scope = getattr(annotation, "storage_scope", None)
    if storage_scope and storage_scope not in _UB_SCOPES:
        raise TypeError(f"{context} requires a UB pointer, got pointer scope {storage_scope!r}")
    if storage_scope and storage_scope != buffer.scope():
        raise TypeError(f"{context} requires the pointer scope {buffer.scope()!r} to match the accessed buffer, got {storage_scope!r}")
    element_dtype = getattr(getattr(annotation, "element_type", None), "dtype", None)
    if element_dtype is not None and str(element_dtype) not in {"", "void"} and str(element_dtype) != str(buffer.dtype):
        raise TypeError(f"{context} requires the pointer element dtype {buffer.dtype} to match the accessed buffer, got {element_dtype}")


def _parse_access_ptr_call(call, *, source, is_load, context):
    args = list(call.args)
    if len(args) != 3:
        raise TypeError(f"{context} requires a three-argument tl.access_ptr descriptor, got {len(args)}")
    base, extent, mask = args
    if not isinstance(base, BufferLoad):
        raise TypeError(f"{context} requires tl.access_ptr to be constructed from a BufferLoad base")
    if not isinstance(mask, tirx.IntImm) or isinstance(mask.value, bool) or int(mask.value) not in (1, 2, 3):
        raise TypeError(f"{context} requires a valid tl.access_ptr rw_mask (1, 2, or 3), got {mask!r}")
    rw_mask = int(mask.value)
    if is_load and not (rw_mask & 1):
        raise TypeError(f"{context} requires the explicit descriptor to declare read access, got rw_mask={rw_mask}")
    if not is_load and not (rw_mask & 2):
        raise TypeError(f"{context} requires the explicit descriptor to declare write access, got rw_mask={rw_mask}")
    if not _is_stable_integer_arithmetic(extent):
        raise TypeError(f"{context} requires a stable scalar integer tl.access_ptr extent, got {extent!r}")
    buffer = base.buffer
    _validate_ub_buffer(buffer, context=context)
    starts = _scalar_start_indices(base, buffer=buffer, context=context)
    _validate_linear_address(buffer, starts, context=context)
    if isinstance(source, tirx.Var):
        _validate_pointer_annotation(source, buffer, context=context)
    return _VmiAddress("pointer", source=source, buffer=buffer, starts=starts, extent=extent, rw_mask=rw_mask, call=call)


def _parse_vmi_address(source, *, is_load, context):
    """Resolve one ordinary address form exactly once.

    Accepted forms are ``Buffer``, ``BufferLoad``, ``BufferRegion``, a direct
    ``tl.access_ptr`` call, and a handle ``Var`` whose direct let binding is that
    call.  Every form must address a UB buffer with one scalar linear offset and
    no axis separators; a numeric ``Var`` is never followed to its definition
    and treated as a pointer.  Python aliases of the same object resolve
    identically; genuine multi-level handle chains, raw ``data`` handles, and
    unknown handles are rejected with a controlled diagnostic.
    """
    if isinstance(source, tirx.Var):
        from tilelang.language.frame import get_let_value, has_let_value

        if str(source.dtype) != "handle":
            raise TypeError(f"{context} requires a Buffer address or an explicit tl.access_ptr; got a scalar {source.dtype} variable")
        if not has_let_value(source):
            raise TypeError(
                f"{context} requires a Buffer address or an explicit tl.access_ptr; got a handle with no recoverable definition"
            )
        bound = get_let_value(source)
        if isinstance(bound, tirx.Var):
            raise TypeError(f"{context} does not support multi-level pointer aliases; bind the tl.access_ptr result directly")
        if isinstance(bound, tirx.Call) and getattr(bound.op, "name", None) == "tl.access_ptr":
            if str(bound.dtype) != "handle":
                raise TypeError(f"{context} requires a bound tl.access_ptr to produce a handle, got {bound.dtype}")
            return _parse_access_ptr_call(bound, source=source, is_load=is_load, context=context)
        if isinstance(bound, (Buffer, BufferLoad, BufferRegion)):
            raise TypeError(f"{context} requires a pointer-producing tl.access_ptr; got a value alias of {type(bound).__name__}")
        raise TypeError(f"{context} requires a Buffer address or an explicit tl.access_ptr; got an unsupported handle definition")
    if isinstance(source, tirx.Call):
        if getattr(source.op, "name", None) != "tl.access_ptr" or str(source.dtype) != "handle":
            raise TypeError(f"{context} requires a Buffer address or an explicit tl.access_ptr")
        return _parse_access_ptr_call(source, source=source, is_load=is_load, context=context)
    if isinstance(source, (Buffer, BufferLoad, BufferRegion)):
        buffer = source if isinstance(source, Buffer) else source.buffer
        _validate_ub_buffer(buffer, context=context)
        starts = _scalar_start_indices(source, buffer=buffer, context=context, ramp_base=True)
        _validate_linear_address(buffer, starts, context=context)
        if isinstance(source, BufferLoad) and any(isinstance(index, tirx.Ramp) for index in source.indices):
            # Consume the same scalar start everywhere: descriptor construction,
            # address-preservation and coverage all use the normalized load, the
            # same position the public access_ptr convention selects.
            source = tirx.BufferLoad(buffer, starts)
        return _VmiAddress("buffer", source=source, buffer=buffer, starts=starts)
    raise TypeError(f"{context} requires a Buffer, BufferLoad, BufferRegion, or an explicit tl.access_ptr")


def _explicit_descriptor_error(address, *, is_load, reason, delta=None, low=None, high=None):
    buffer = address.buffer
    strides = tuple(buffer.strides) if buffer.strides else ()
    context = f"buffer {buffer.name!r}: shape={tuple(buffer.shape)}, strides={strides}, elem_offset={buffer.elem_offset}"
    params = [f"descriptor_start={tuple(address.starts)}", f"descriptor_extent={address.extent}"]
    if delta is not None:
        params.append(f"instruction_offset={delta}")
    if low is not None or high is not None:
        params.append(f"access_envelope=[{low}, {high})")
    context += ", " + ", ".join(params)
    return ValueError(f"{_vmi_op_name(is_load)} cannot verify the explicit access_ptr descriptor: {reason} ({context})")


class _DescriptorPlan:
    """One candidate descriptor construction awaiting the shared acceptance check.

    ``base`` is the object the descriptor is built from (``None`` keeps the
    original source), ``starts`` are its logical start indices, ``extent`` is
    its declared element length, ``offset`` is the VMI instruction offset and
    ``relative`` is the descriptor start position measured from the buffer's
    logical origin.  ``relative`` is only set for candidates that move the
    descriptor base; the shared check uses it to prove address preservation.
    """

    __slots__ = ("base", "starts", "extent", "offset", "relative", "kind", "require_static_shape")

    def __init__(self, *, base, starts, extent, kind, offset=None, relative=None, require_static_shape=False):
        self.base = base
        self.starts = starts
        self.extent = extent
        self.kind = kind
        self.offset = offset if offset is not None else _zero_offset()
        self.relative = relative
        self.require_static_shape = require_static_shape


def _plan_coverage_reason(proof, address, plan, extent):
    """Logical-coverage proof of one candidate under the existing consumer."""
    buffer = address.buffer
    if plan.kind == "explicit":
        return _explicit_coverage_reason(proof, address, extent)
    if plan.kind == "local":
        return _local_coverage_reason(proof, buffer, plan.starts, extent)
    if plan.kind == "origin":
        return _origin_prefix_reason(proof, buffer, extent, require_static_shape=plan.require_static_shape)
    raise AssertionError(f"unknown descriptor plan kind {plan.kind!r}")


def _accept_plan(proof, address, plan, *, low, high):
    """The single acceptance exit shared by every descriptor candidate.

    Returns ``None`` when the candidate preserves the original access address,
    contains the complete physical envelope ``[low, high)`` relative to the
    instruction offset and covers its declared extent under the existing logical
    pointer-region analysis.  Otherwise returns ``(kind, reason)`` so the
    caller can choose the diagnostic: ``kind`` is one of ``extent``,
    ``address``, ``low``, ``high`` or ``coverage``.
    """
    delta = plan.offset if isinstance(plan.offset, tirx.PrimExpr) else tirx.convert(plan.offset)
    extent = plan.extent if isinstance(plan.extent, tirx.PrimExpr) else tirx.convert(plan.extent)
    constant = _constant_int(plan.extent)
    if constant is not None:
        if constant <= 0:
            return "extent", f"the descriptor extent must be positive, got {constant}"
    elif not proof.can_prove(extent > 0):
        return "extent", "cannot prove the descriptor extent is positive"
    if plan.relative is not None:
        source_relative = _source_relative_offset(address.source)
        if source_relative is None or not proof.can_prove_equal(plan.relative + delta, source_relative):
            return "address", "the descriptor construction does not preserve the original access address"
    if not proof.can_prove(delta + low >= 0):
        return (
            "low",
            "the complete physical access envelope starts below the descriptor: 0 <= instruction_offset + envelope_low cannot be proven",
        )
    if not proof.can_prove(delta + high <= extent):
        return (
            "high",
            "the complete physical access envelope exceeds the descriptor: "
            "instruction_offset + envelope_high <= descriptor_extent cannot be proven",
        )
    reason = _plan_coverage_reason(proof, address, plan, plan.extent)
    if reason is not None:
        return "coverage", reason
    return None


def _explicit_coverage_reason(proof, address, extent):
    buffer = address.buffer
    starts = address.starts
    if _is_tight_buffer(buffer):
        return _local_coverage_reason(proof, buffer, starts, extent)
    origin = all(_constant_int(start) == 0 for start in starts)
    if origin and (buffer.strides and len(buffer.strides) == len(buffer.shape) and _padded_row_major(buffer, proof)):
        return _origin_prefix_reason(proof, buffer, extent, require_static_shape=True)
    permutation_reason = _whole_dense_permutation_reason(buffer, starts, extent)
    if permutation_reason is None:
        return None
    return (
        f"{_COVERAGE_PREFIX}: the descriptor starts at a local coordinate of a buffer whose layout is not "
        "proven tight row-major, and neither an origin-prefix nor an exact whole dense permutation proof "
        f"applies ({permutation_reason})"
    )


def _validate_explicit_descriptor(address, proof, *, low, high, delta, is_load, context):
    """Validate a caller-declared descriptor without rewriting it."""
    if delta is None:
        delta = _zero_offset()
    elif not isinstance(delta, tirx.PrimExpr):
        delta = tirx.convert(delta)
    if not _is_integer_scalar(delta):
        raise TypeError(f"{context} requires a scalar integer instruction offset, got {delta!r}")
    plan = _DescriptorPlan(base=address.source, starts=address.starts, extent=address.extent, offset=delta, kind="explicit")
    rejection = _accept_plan(proof, address, plan, low=low, high=high)
    if rejection is not None:
        raise _explicit_descriptor_error(address, is_load=is_load, reason=rejection[1], delta=delta, low=low, high=high)
    return delta


def _try_whole_buffer_plan(source, proof, *, low, high):
    """Build a whole tight-buffer descriptor candidate.

    The plan uses the buffer origin as the descriptor and carries the source
    position as the instruction offset.  It is only offered when a single
    linear relative offset is available, the whole span is exactly
    representable and every source axis start is provably non-negative
    (otherwise moving the start would bypass the normal negative-index
    legalization).  Containment and logical coverage are decided by the shared
    acceptance exit, not here.
    """
    buffer = source if isinstance(source, Buffer) else source.buffer
    if not _provably_non_negative_starts(source, proof):
        return None
    span = _whole_buffer_span(buffer, proof)
    relative = _source_relative_offset(source)
    if span is None or relative is None:
        return None
    return span, buffer, relative


def _padded_row_major(buffer, proof) -> bool:
    """Whether every explicit stride is at least its dense row-major stride.

    This proves that every legal non-negative coordinate maps at or above its
    dense row-major offset, which is what keeps the logical prefix declared by
    :func:`_padded_origin_plan` conservative.  It does not characterize the
    layout as ordinary row-major padding; any view whose strides merely satisfy
    the inequality is accepted.  Views with a smaller stride on some dimension
    are rejected.  The dense stride itself is accumulated in Python integers
    (inner dimensions must be static), so a narrowed symbolic product is never
    used as the comparison basis.
    """
    ndim = len(buffer.shape)
    if ndim == 0 or not buffer.strides or len(buffer.strides) != ndim:
        return False
    dense_stride = 1
    for dim in range(ndim - 1, -1, -1):
        if not proof.can_prove(buffer.strides[dim] >= dense_stride):
            return False
        if dim > 0:
            length = _constant_int(buffer.shape[dim])
            if length is None or length <= 0:
                return False
            dense_stride *= length
    return True


def _padded_origin_plan(source, proof, *, low, high):
    """Build a padded-origin descriptor candidate.

    The accessed envelope is ``[relative + low, relative + high)`` in element
    addresses, where ``relative`` is the source position from the buffer
    origin.  When each stride is at least its dense row-major stride, each
    logical element of the buffer maps at or above its row-major position, so
    every element whose address falls in the envelope sorts below
    ``relative + high`` in row-major order.  The declared logical prefix
    ``[0, relative + high)`` therefore covers the whole envelope, and the VMI
    instruction offset carries ``relative`` once from the descriptor origin.
    Containment and logical coverage are decided by the shared acceptance exit.
    """
    if not isinstance(source, (BufferLoad, BufferRegion)):
        return None
    buffer = source.buffer
    relative = _source_relative_offset(source)
    if relative is None or not _padded_row_major(buffer, proof):
        return None
    # A negative or unknown start must keep the normal negative-index
    # legalization instead of being moved into the instruction offset.
    if not _provably_non_negative_starts(source, proof):
        return None
    return proof.simplify(relative + high), buffer, relative


def _group_envelope(address, *, lanes, group, stride, dist_mode, is_load, elem_dtype):
    """Complete physical envelope ``[low, high)`` of an equal-distance group access.

    The first group starts at the original access address ``A``.  For stride
    ``S`` and physical group width ``W`` the contiguous envelope is
    ``[A + min(0,(G-1)*S), A + max(0,(G-1)*S) + W)``.  Candidate descriptors
    for an envelope below the source position are chosen by
    :func:`_negative_group_plan`.

    A group load is unmasked and therefore reads whole physical vectors: when
    the logical per-group width ``N/G`` is not a multiple of the physical VL,
    the last chunk still reads a full VL (e.g. f32 G=1/N=80 lowers to two
    full vlds covering 128 address elements).  A group store keeps the logical
    width because the backend's generated tail mask bounds the writes; do not
    copy the load rounding onto stores.

    A group or stride with no proven bounds is reported to the caller instead
    of being replaced by a fabricated range.
    """
    source = address.source
    buffer = address.buffer
    group_const = _constant_int(group)
    if group_const is not None and group_const <= 0:
        raise ValueError(f"{_vmi_op_name(is_load)}(...) requires group to be a positive integer, got {group_const}")
    if group_const is None:
        raise _descriptor_error(
            source,
            buffer=buffer,
            is_load=is_load,
            reason="group has no proven bounds for this buffer view",
            size=lanes,
            group=group,
            stride=stride,
        )
    # Slot (one result/input element per group) and group broadcast read or
    # write exactly one address element per group.  Ordinary groups cover
    # ceil(N/G) elements; the ceiling keeps a non-divisible shape conservative.
    if dist_mode == "brc" or lanes <= group_const:
        per_group = 1
    else:
        per_group = (lanes + group_const - 1) // group_const
        if is_load:
            bits = int(getattr(elem_dtype, "bits", 0) or 0)
            if bits <= 0:
                raise _descriptor_error(
                    source,
                    buffer=buffer,
                    is_load=is_load,
                    reason="the physical element width is unknown, so the complete physical vector width cannot be computed",
                    size=lanes,
                    group=group,
                    stride=stride,
                )
            vector_elements = max(1, 2048 // bits)
            per_group = (per_group + vector_elements - 1) // vector_elements * vector_elements
    stride_const = _constant_int(stride)
    if stride_const is None:
        if group_const == 1:
            # One group has no inter-group displacement: the stride cannot move
            # any access, so the envelope is already complete.
            return 0, per_group
        raise _descriptor_error(
            source,
            buffer=buffer,
            is_load=is_load,
            reason="group stride has no proven bounds for this buffer view",
            size=lanes,
            group=group,
            stride=stride,
        )
    span = (group_const - 1) * stride_const
    return min(0, span), max(0, span) + per_group


def _block_envelope(address, *, lanes, elem_dtype, block_stride, is_load):
    """Complete physical envelope ``[low, high)`` of a block-stride access (C.3.2).

    ``block_stride`` is measured in 32-byte blocks.  A full physical vector
    covers eight blocks; physical chunks beyond one vector start at the next
    block group, and the tail chunk of a masked store conservatively covers
    its whole block group as well.  A dynamic or negative control word has no
    proven physical bounds, and an unknown element width cannot size a block;
    those cases are reported instead of being replaced by a fabricated range.
    """
    source = address.source
    buffer = address.buffer
    bits = int(getattr(elem_dtype, "bits", 0) or 0)
    if bits <= 0:
        raise _descriptor_error(
            source,
            buffer=buffer,
            is_load=is_load,
            reason="the physical element width is unknown, so the complete block-stride footprint cannot be computed",
            size=lanes,
            block_stride=block_stride,
        )
    elements_per_block = max(1, 256 // bits)
    vector_elements = max(1, 2048 // bits)
    chunks = (lanes + vector_elements - 1) // vector_elements
    blocks = chunks * max(1, vector_elements // elements_per_block)
    block_const = _constant_int(block_stride)
    if block_const is None:
        raise _descriptor_error(
            source,
            buffer=buffer,
            is_load=is_load,
            reason="block stride has no proven bounds for this buffer view",
            size=lanes,
            block_stride=block_stride,
        )
    if block_const < 0:
        raise _descriptor_error(
            source,
            buffer=buffer,
            is_load=is_load,
            reason="negative block-stride control words have no proven physical bounds",
            size=lanes,
            block_stride=block_stride,
        )
    return 0, (blocks - 1) * block_const * elements_per_block + elements_per_block


def _vmi_access_envelope(address, *, lanes, elem_dtype, dist_mode, group, stride, block_stride, is_load):
    """Complete physical envelope ``[low, high)`` of one ordinary VMI access.

    The envelope is relative to the instruction address.  Loads are unmasked
    and read whole physical vectors, so their contiguous envelope is rounded up
    to the physical VL; stores keep the logical lane count because the
    generated tail mask bounds the writes.  ``dintlv`` touches both interleaved
    inputs, group modes use :func:`_group_envelope`, block strides use
    :func:`_block_envelope`, and genuine FP4 descriptors stay in logical
    element units until ``RewriteFp4ToFp4x2`` performs the single packing.
    """
    if dist_mode == "dintlv":
        # Two interleaved inputs/outputs: the complete 2*N span, regardless of
        # the (ineffective) store mask.
        return 0, 2 * lanes
    if group is not None:
        return _group_envelope(
            address,
            lanes=lanes,
            group=group,
            stride=stride,
            dist_mode=dist_mode,
            is_load=is_load,
            elem_dtype=elem_dtype,
        )
    if block_stride is not None:
        return _block_envelope(address, lanes=lanes, elem_dtype=elem_dtype, block_stride=block_stride, is_load=is_load)
    if str(elem_dtype) == "float4_e2m1fn":
        # Genuine FP4 descriptors stay in logical FP4 element units;
        # RewriteFp4ToFp4x2 performs the single physical packing.
        return 0, lanes
    if dist_mode == "brc":
        # An ordinary broadcast reads exactly one source element.  The
        # automatic descriptor declares that single element; an explicit
        # descriptor keeps the caller's extent and only has to contain it.
        return 0, 1
    if is_load:
        bits = int(getattr(elem_dtype, "bits", 0) or 0)
        if bits <= 0:
            raise _descriptor_error(
                address.source,
                buffer=address.buffer,
                is_load=True,
                reason="the physical element width is unknown, so the complete physical vector width cannot be computed",
                size=lanes,
            )
        vector_elements = max(1, 2048 // bits)
        # An unmasked load reads whole physical vectors: cover the tail chunk.
        return 0, (lanes + vector_elements - 1) // vector_elements * vector_elements
    return 0, lanes


def _local_descriptor_plan(address, proof, *, low, high, extent, is_load, size, group, stride, block_stride):
    """Declare the descriptor at the source position with extent ``extent``.

    The descriptor keeps the caller's own start indices, so the logical
    coverage proof must show that the existing row-major carry from those
    starts includes every element of the declared extent.  ``extent`` is the
    envelope extent (a single element for an ordinary broadcast); the shared
    acceptance exit re-proves containment of the physical envelope against it.
    A one-dimensional buffer has no outer dimension the carry can miss and
    only needs a positive extent, which keeps runtime store offsets working.
    """
    plan = _DescriptorPlan(base=None, starts=address.starts, extent=extent, kind="local")
    rejection = _accept_plan(proof, address, plan, low=low, high=high)
    if rejection is not None:
        raise _descriptor_error(
            address.source,
            buffer=address.buffer,
            is_load=is_load,
            reason=rejection[1],
            size=size,
            group=group,
            stride=stride,
            block_stride=block_stride,
            start=address.starts,
            extent=extent,
            low=low,
            high=high,
        )
    return plan


def _negative_group_plan(address, proof, *, low, high, is_load, size, group, stride):
    """Cover an envelope that starts below the source position.

    A negative static group stride is the only ordinary access that reaches
    below its own start.  Three address-preserving constructions are tried, in
    this order: relocating a one-dimensional unit-stride descriptor to the
    lowest address while the VMI instruction offset carries ``-low``; a padded
    origin prefix for a view whose strides are each at least dense; and the
    whole tight buffer origin.  Every candidate goes through the shared
    acceptance exit, which re-proves address preservation, physical containment
    and logical coverage.  Relocations keep the normal negative-index
    legalization behavior: the original and shifted positions must have the
    same proven sign, so the pass rewrites both or neither and their difference
    stays ``low``.  That sign test deliberately uses a bare analyzer: it must
    model the facts LegalizeNegativeIndex itself has, not the stronger
    per-access proof.
    """
    source = address.source
    buffer = address.buffer
    extent = high - low
    coverage_reason = None
    if isinstance(source, BufferLoad) and len(source.indices) == 1 and _is_unit_stride_1d(buffer):
        from tvm import arith

        bare = arith.Analyzer()
        original_index = source.indices[0]
        shifted_index = original_index + low
        original_sign = _proven_sign(bare, original_index)
        if original_sign != 0 and original_sign == _proven_sign(bare, shifted_index):
            shifted = tirx.BufferLoad(buffer, [shifted_index])
            plan = _DescriptorPlan(
                base=shifted,
                starts=[shifted_index],
                extent=extent,
                offset=-low,
                relative=_source_relative_offset(shifted),
                kind="local",
            )
            rejection = _accept_plan(proof, address, plan, low=low, high=high)
            if rejection is None:
                return plan
            if rejection[0] == "coverage":
                coverage_reason = rejection[1]
    origin_candidate = _padded_origin_plan(source, proof, low=low, high=high)
    if origin_candidate is not None:
        plan = _DescriptorPlan(
            base=origin_candidate[1],
            starts=[_zero_offset() for _ in buffer.shape],
            extent=origin_candidate[0],
            offset=origin_candidate[2],
            relative=_zero_offset(),
            kind="origin",
            require_static_shape=True,
        )
        rejection = _accept_plan(proof, address, plan, low=low, high=high)
        if rejection is None:
            return plan
        if rejection[0] == "coverage":
            coverage_reason = coverage_reason or rejection[1]
    whole_candidate = _try_whole_buffer_plan(source, proof, low=low, high=high)
    if whole_candidate is not None:
        plan = _DescriptorPlan(
            base=whole_candidate[1],
            starts=[_zero_offset() for _ in buffer.shape],
            extent=whole_candidate[0],
            offset=whole_candidate[2],
            relative=_zero_offset(),
            kind="origin",
            require_static_shape=len(buffer.shape) > 1,
        )
        rejection = _accept_plan(proof, address, plan, low=low, high=high)
        if rejection is None:
            return plan
        if rejection[0] == "coverage":
            coverage_reason = coverage_reason or rejection[1]
    relative = _source_relative_offset(source)
    if relative is None:
        reason = "the access address cannot be reduced to a single linear element offset"
    elif not _provably_non_negative_starts(source, proof):
        reason = (
            "the source index is negative or has no proven sign, so relocating the descriptor "
            "cannot preserve the address selected by negative-index legalization"
        )
    elif proof.can_prove(relative + low < 0):
        reason = "the finite access envelope starts below the buffer view origin and no constructible descriptor covers it"
    else:
        reason = "cannot prove that any constructible descriptor covers the finite access envelope"
    raise _descriptor_error(
        source,
        buffer=buffer,
        is_load=is_load,
        reason=coverage_reason or reason,
        size=size,
        group=group,
        stride=stride,
        start=address.starts,
        extent=extent,
        low=low,
        high=high,
    )


def _vmi_access_plan(address, proof, *, low, high, is_load, size, group, stride, block_stride):
    """Choose one descriptor construction for an ordinary buffer access.

    A non-negative envelope keeps the caller's start as the descriptor origin
    and declares the envelope extent (a single element for an ordinary
    broadcast, whose true physical read is exactly that element).  An envelope
    below the source position can only come from a negative group stride and
    uses :func:`_negative_group_plan`.  There is deliberately no whole-buffer
    fallback for a failed local descriptor: that would hide a genuinely
    incomplete logical region instead of reporting it.
    """
    if low >= 0:
        return _local_descriptor_plan(
            address,
            proof,
            low=low,
            high=high,
            extent=high,
            is_load=is_load,
            size=size,
            group=group,
            stride=stride,
            block_stride=block_stride,
        )
    return _negative_group_plan(address, proof, low=low, high=high, is_load=is_load, size=size, group=group, stride=stride)


def _resolve_ptr_and_offset(source, offset=None, *, access_type: str, extent=None, descriptor=None, instruction_offset=None):
    """Normalize VMI address syntax into PTODSL's explicit ptr+offset ABI."""
    if isinstance(source, (BufferLoad, Buffer, BufferRegion)) and offset is not None:
        raise TypeError("T.vmi buffer addresses do not accept an offset; express the offset in the buffer index")
    base = descriptor if descriptor is not None else source
    if isinstance(base, (BufferLoad, Buffer, BufferRegion)):
        ptr = access_ptr(base, access_type, extent=extent, offset=0)
        return ptr, instruction_offset if instruction_offset is not None else _zero_offset()
    if offset is None:
        return source, _zero_offset()
    return source, offset


def _pair_result_dtype(value):
    return _dtype_of(value)


def _vector_result_dtype(value, size=None, *, elem_dtype=None):
    elem = _dtype(elem_dtype) if elem_dtype is not None else _element_dtype_of(value)
    lanes = _lanes_of(value) if size is None else _normalize_lanes(size, context="T.vmi result size")
    return _dtype(f"{elem}x{lanes}")


def _vgather_result_element_dtype(source_elem, offsets):
    """Derive the result element type for the A5 byte-widening gather path."""
    source_elem = _dtype(source_elem)
    offsets_elem = _element_dtype_of(offsets)
    source_name = str(source_elem)
    offsets_name = str(offsets_elem)
    if source_name not in {"uint8", "int8"}:
        return source_elem

    # The A5 b8->b16 gather path requires unsigned 16-bit element offsets and
    # widens the source to a matching 16-bit result type.
    if offsets_name != "uint16":
        raise TypeError("T.vmi.vgather(...) requires uint16 offsets for 8-bit integer sources")
    return _dtype("uint16" if source_name == "uint8" else "int16")


class VmiPair:
    """Lazy pair wrapper for multi-result VMI calls."""

    def __init__(self, pair):
        self._pair = pair

    @property
    def dtype(self):
        return self._pair.dtype

    def __iter__(self):
        yield _pair_get(self, 0)
        yield _pair_get(self, 1)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index not in (0, 1):
            raise IndexError("VmiPair only contains two values")
        return _pair_get(self, index)


def inside_vmi() -> bool:
    """Return True when executing inside ``with T.SimdVF():``."""
    return _inside_simdvf()


def require_vmi_scope(context: str = "T.vmi") -> None:
    """Raise if a VMI op is used outside ``with T.SimdVF():``."""
    if not inside_vmi():
        raise RuntimeError(f"{context} operations require a `T.SimdVF()` scope")


def vreg(lanes: int, dtype_like):
    """Return a logical VMI vector dtype descriptor."""
    lanes = _normalize_lanes(lanes, context="T.vmi.vreg(...)")
    elem = _scalar_dtype(dtype_like, context="T.vmi.vreg(...)")
    return _dtype(f"{elem}x{lanes}")


def mask(lanes: int):
    """Return a logical VMI predicate dtype descriptor."""
    lanes = _normalize_lanes(lanes, context="T.vmi.mask(...)")
    return _dtype(f"boolx{lanes}")


@_scope_guarded
def _pair_get(pair, index):
    """Extract one value from a VMI pair."""
    inner = pair._pair if isinstance(pair, VmiPair) else pair
    return tirx.call_intrin(str(inner.dtype), _Op("tl.vmi.pair_get"), inner, index)


def _pack_fp4_element_offset(value, *, context: str):
    """Convert a logical FP4 element offset to packed-pair units."""
    constant = _constant_int(value)
    if constant is not None:
        if constant % 2:
            raise ValueError(f"{context} requires an even FP4 element offset, got {constant}")
        return constant // 2
    # Dynamic offsets are expressed in logical FP4 element units and lowered
    # to packed-pair units.
    return value // 2


@_scope_guarded
def vload(
    source,
    offset=None,
    *,
    size,
    stride=None,
    block_stride=None,
    dist_mode=None,
    group=None,
    loc=None,
    ip=None,
):
    """Load a VMI vector from either `a_ub[0]`-style buffer loads or ptr+offset form.

    A VMI load always reads the full ``size`` lanes starting at the load
    address: the ISA intentionally keeps load instructions unmasked for
    performance, so masks apply to computation and stores, never to the
    load itself. Reading a full vector past a logical tail is still safe
    on device — the top 8K of UB is reserved as stack space, so the extra
    read-only lanes cannot fault; they carry undefined values that the
    compute and store masks must keep out of valid outputs.
    """
    _validate_pto_load_modes(
        "T.vmi.vload(...)",
        dist_mode=dist_mode,
        group=group,
        stride=stride,
        block_stride=block_stride,
        allow_group_brc=True,
        allowed_dist_modes={None, "continuous", "dintlv", "brc"},
    )
    if size is None:
        raise TypeError("T.vmi.vload(...) requires size")
    context = "T.vmi.vload(...)"
    address = _parse_vmi_address(source, is_load=True, context=context)
    source_elem = _dtype(address.buffer.dtype)
    is_packed_fp4 = str(source_elem) == "float4_e2m1fn"
    if is_packed_fp4:
        if size % 2:
            raise ValueError("T.vmi.vload(...) requires an even FP4 lane count for packed FP4 storage")
        if dist_mode == "dintlv":
            raise TypeError('T.vmi.vload(...) does not support packed FP4 with dist_mode="dintlv"')
        if group is not None or block_stride is not None or dist_mode == "brc":
            raise TypeError("T.vmi.vload(...) only supports contiguous packed FP4 loads")

    result_dtype = _vector_result_dtype(source, size, elem_dtype=source_elem)
    low, high = _vmi_access_envelope(
        address,
        lanes=size,
        elem_dtype=source_elem,
        dist_mode=dist_mode,
        group=group,
        stride=stride,
        block_stride=block_stride,
        is_load=True,
    )
    proof = _local_analyzer([*address.starts, *address.buffer.shape, low, high, offset, address.extent])
    if address.is_pointer:
        ptr = address.source
        extra_offset = _validate_explicit_descriptor(address, proof, low=low, high=high, delta=offset, is_load=True, context=context)
    else:
        plan = _vmi_access_plan(
            address,
            proof,
            low=low,
            high=high,
            is_load=True,
            size=size,
            group=group,
            stride=stride,
            block_stride=block_stride,
        )
        ptr, extra_offset = _resolve_ptr_and_offset(
            address.source,
            offset,
            access_type="r",
            extent=plan.extent,
            descriptor=plan.base,
            instruction_offset=plan.offset,
        )
    if is_packed_fp4:
        extra_offset = _pack_fp4_element_offset(extra_offset, context="T.vmi.vload(...)")
    attrs = {
        "size": size,
        "stride": stride,
        "block_stride": block_stride,
        "dist_mode": dist_mode,
        "group": group,
        "loc": loc,
        "ip": ip,
    }
    call = _call_vmi("vload", result_dtype, ptr, extra_offset, **attrs)
    if dist_mode == "dintlv":
        return _wrap_pair(call)
    return call


@_scope_guarded
def vstore(
    values,
    destination,
    offset=None,
    mask=None,
    *,
    stride=None,
    block_stride=None,
    dist_mode=None,
    group=None,
    loc=None,
    ip=None,
):
    """Store a VMI vector to either `a_ub[0]`-style buffer loads or ptr+offset form."""
    if mask is None and _looks_like_mask(offset):
        mask = offset
        offset = None
    _validate_pto_load_modes(
        "T.vmi.vstore(...)",
        dist_mode=dist_mode,
        group=group,
        stride=stride,
        block_stride=block_stride,
        allow_group_brc=False,
        allowed_dist_modes={None, "continuous", "dintlv"},
    )
    if group is not None and mask is not None:
        raise TypeError("T.vmi.vstore(...) group mode does not take a mask operand")
    value_dtype = None
    if dist_mode == "dintlv":
        if isinstance(values, VmiPair):
            value_dtype = _element_dtype_of(values)
            values = tuple(values)
        elif not _is_sequence(values) or len(values) != 2:
            raise TypeError('T.vmi.vstore(...) with dist_mode="dintlv" requires an (even, odd) pair')
        else:
            _require_same_vreg_type(values[0], values[1], context="T.vmi.vstore(...)", allow_packed_fp4=True)
    elif isinstance(values, VmiPair) or _is_sequence(values):
        raise TypeError('T.vmi.vstore(...) expects a single VMI vector unless dist_mode="dintlv"')

    store_lanes = _lanes_of(values[0]) if _is_sequence(values) else _lanes_of(values)
    value_args = list(values) if _is_sequence(values) else [values]
    if value_dtype is None:
        value_dtype = _element_dtype_of(values[0]) if _is_sequence(values) else _element_dtype_of(values)
    if str(value_dtype) == "float4_e2m1fn":
        if store_lanes % 2:
            raise ValueError("T.vmi.vstore(...) requires an even FP4 lane count for packed FP4 storage")
        if dist_mode == "dintlv":
            raise TypeError('T.vmi.vstore(...) does not support packed FP4 with dist_mode="dintlv"')
        if group is not None or block_stride is not None:
            raise TypeError("T.vmi.vstore(...) only supports contiguous packed FP4 stores")
    elif mask is not None:
        mask = _require_mask(mask, context="T.vmi.vstore(...)")

    context = "T.vmi.vstore(...)"
    address = _parse_vmi_address(destination, is_load=False, context=context)
    # The destination element dtype comes from the parsed buffer, not from a
    # second lookup through the source handle.  Codegen stores one value
    # element per destination element, so the type must match exactly.
    destination_elem = _dtype(address.buffer.dtype)
    if str(destination_elem) != str(value_dtype):
        raise TypeError(
            f"{context} requires the destination element dtype {destination_elem} to match the value element "
            f"dtype {value_dtype}; convert the value before storing"
        )
    low, high = _vmi_access_envelope(
        address,
        lanes=store_lanes,
        elem_dtype=value_dtype,
        dist_mode=dist_mode,
        group=group,
        stride=stride,
        block_stride=block_stride,
        is_load=False,
    )
    proof = _local_analyzer([*address.starts, *address.buffer.shape, low, high, offset, address.extent])
    if address.is_pointer:
        ptr = address.source
        extra_offset = _validate_explicit_descriptor(address, proof, low=low, high=high, delta=offset, is_load=False, context=context)
    else:
        plan = _vmi_access_plan(
            address,
            proof,
            low=low,
            high=high,
            is_load=False,
            size=store_lanes,
            group=group,
            stride=stride,
            block_stride=block_stride,
        )
        ptr, extra_offset = _resolve_ptr_and_offset(
            address.source,
            offset,
            access_type="w",
            extent=plan.extent,
            descriptor=plan.base,
            instruction_offset=plan.offset,
        )
    if str(value_dtype) == "float4_e2m1fn":
        extra_offset = _pack_fp4_element_offset(extra_offset, context="T.vmi.vstore(...)")
        if mask is not None:
            mask = _require_mask(mask, context="T.vmi.vstore(...) of FP4")
            if not str(_dtype_of(mask)).startswith("bool"):
                raise TypeError("T.vmi.vstore(...) of FP4 expects a VMI mask operand")
            physical_lanes = store_lanes // 2
            if _lanes_of(mask) != physical_lanes:
                raise TypeError(
                    "T.vmi.vstore(...) of FP4 requires a physical FP4x2 mask with "
                    f"half as many lanes as its {store_lanes}-lane logical FP4 value "
                    f"(expected {physical_lanes}, got {_lanes_of(mask)})"
                )
    return _call_vmi(
        "vstore",
        "void",
        *value_args,
        ptr,
        extra_offset,
        mask,
        stride=stride,
        block_stride=block_stride,
        dist_mode=dist_mode,
        group=group,
        loc=loc,
        ip=ip,
    )


@_scope_guarded
def vci(base, *, size, order=None, loc=None, ip=None):
    if size is None:
        raise TypeError("T.vmi.vci(...) requires size")
    _reject_packed_fp4(base, context="T.vmi.vci(...)")
    result_dtype = vreg(size, _element_dtype_of(base))
    return _call_vmi("vci", result_dtype, base, size=size, order=order, loc=loc, ip=ip)


def _binary_same_dtype(name):
    @_scope_guarded
    def wrapper(lhs, rhs, mask=None, *, pmode=None, loc=None, ip=None):
        result_dtype = _require_same_vreg_type(lhs, rhs, context=f"T.vmi.{name}(...)")
        args = [lhs, rhs]
        if mask is not None:
            args.append(_require_compatible_mask(mask, _lanes_of(lhs), context=f"T.vmi.{name}(...)"))
        return _call_vmi(name, result_dtype, *args, pmode=pmode, loc=loc, ip=ip)

    return wrapper


def _unary_same_dtype(name):
    @_scope_guarded
    def wrapper(source, mask=None, *, pmode=None, loc=None, ip=None):
        _reject_packed_fp4(source, context=f"T.vmi.{name}(...)")
        args = [source]
        if mask is not None:
            args.append(mask)
        return _call_vmi(name, _dtype_of(source), *args, pmode=pmode, loc=loc, ip=ip)

    return wrapper


def _vec_scalar_same_dtype(name):
    @_scope_guarded
    def wrapper(source, scalar, mask, *, pmode=None, loc=None, ip=None):
        _reject_packed_fp4(source, context=f"T.vmi.{name}(...)")
        return _call_vmi(
            name,
            _dtype_of(source),
            source,
            scalar,
            _require_mask(mask, context=f"T.vmi.{name}(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    return wrapper


vadd = _binary_same_dtype("vadd")
vsub = _binary_same_dtype("vsub")
vmul = _binary_same_dtype("vmul")
vmax = _binary_same_dtype("vmax")
vmin = _binary_same_dtype("vmin")
vand = _binary_same_dtype("vand")
vor = _binary_same_dtype("vor")
vxor = _binary_same_dtype("vxor")
vshl = _binary_same_dtype("vshl")
vshr = _binary_same_dtype("vshr")

vabs = _unary_same_dtype("vabs")
vneg = _unary_same_dtype("vneg")
vrelu = _unary_same_dtype("vrelu")
vexp = _unary_same_dtype("vexp")
vln = _unary_same_dtype("vln")
vsqrt = _unary_same_dtype("vsqrt")
vnot = _unary_same_dtype("vnot")

vadds = _vec_scalar_same_dtype("vadds")
vmuls = _vec_scalar_same_dtype("vmuls")
vmaxs = _vec_scalar_same_dtype("vmaxs")
vmins = _vec_scalar_same_dtype("vmins")
vshls = _vec_scalar_same_dtype("vshls")
vshrs = _vec_scalar_same_dtype("vshrs")


# Hardware divide is the default. precision='exact' expands the CANN
# vdiv_0ulp_ftz_true residual search. Changing the no-kwarg default would
# rewrite every T.vmi.vdiv call.
_VDIV_PRECISION_ALIASES = {
    None: "hw",
    "ftz_true": "hw",
    "exact": "exact",
    "vdiv_0ulp_ftz_true": "exact",
}


def _vdiv_hw(lhs, rhs, mask=None, *, pmode=None, loc=None, ip=None):
    result_dtype = _require_same_vreg_type(lhs, rhs, context="T.vmi.vdiv(...)")
    args = [lhs, rhs]
    if mask is not None:
        args.append(_require_compatible_mask(mask, _lanes_of(lhs), context="T.vmi.vdiv(...)"))
    return _call_vmi("vdiv", result_dtype, *args, pmode=pmode, loc=loc, ip=ip)


@_scope_guarded
def vdiv(lhs, rhs, mask=None, *, precision=None, pmode=None, loc=None, ip=None):
    """Vector divide. ``precision='exact'`` expands CANN ``vdiv_0ulp_ftz_true``."""
    context = "T.vmi.vdiv(...)"
    key = None if precision is None else str(precision).lower()
    if key not in _VDIV_PRECISION_ALIASES:
        raise ValueError(f"{context} precision {precision!r} is not valid. Valid: exact, ftz_true, vdiv_0ulp_ftz_true")
    if _VDIV_PRECISION_ALIASES[key] == "hw":
        return _vdiv_hw(lhs, rhs, mask, pmode=pmode, loc=loc, ip=ip)

    _require_same_vreg_type(lhs, rhs, context=context)
    if str(_element_dtype_of(lhs)) != "float32":
        raise ValueError(f"{context} precision={precision!r} requires float32 operands (got {_element_dtype_of(lhs)})")
    if pmode is not None:
        raise ValueError(f"{context} precision={precision!r} does not support pmode={pmode!r}")
    mask = _require_compatible_mask(mask, _lanes_of(lhs), context=context)
    lanes = _lanes_of(lhs)
    z = _bind(_vdiv_hw(lhs, rhs, mask))
    z_bits = _bind(vinterpret_cast(z, "uint32"))
    inf_nan = _bind(vor(z_bits, vbrc(tirx.const(0x80000000, "uint32"), size=lanes), mask))
    zero_cmp = _bind(vcmps(z, tirx.const(0.0, "float32"), mask, "eq"))
    inf_nan_cmp = _bind(vcmp(inf_nan, vbrc(tirx.const(0xFF800000, "uint32"), size=lanes), mask, "ge"))
    rhs_bits = _bind(vinterpret_cast(rhs, "uint32"))
    exp = _bind(
        vand(
            vshrs(rhs_bits, tirx.const(23, "uint32"), mask),
            vbrc(tirx.const(0xFF, "uint32"), size=lanes),
            mask,
        )
    )
    one_bits = vbrc(tirx.const(0x3F800000, "uint32"), size=lanes)
    scale_bits = vshls(
        vsub(vbrc(tirx.const(254, "uint32"), size=lanes), exp, mask),
        tirx.const(23, "uint32"),
        mask,
    )
    scale_bits = vsel(vcmps(exp, tirx.const(0, "uint32"), mask, "eq"), one_bits, scale_bits)
    scale_bits = vsel(
        vcmp(exp, vbrc(tirx.const(253, "uint32"), size=lanes), mask, "gt"),
        one_bits,
        scale_bits,
    )
    scale = _bind(vinterpret_cast(scale_bits, "float32"))
    lhs_s = _bind(vmul(lhs, scale, mask))
    rhs_s = _bind(vmul(rhs, scale, mask))
    y = _bind(vmuls(rhs_s, tirx.const(-1.0, "float32"), mask))
    r = _bind(vabs(vmula(lhs_s, z, y, mask), mask))
    z_i = _bind(vinterpret_cast(z, "int32"))
    z_pre = _bind(vinterpret_cast(vadds(z_i, tirx.const(-1, "int32"), mask), "float32"))
    z_next = _bind(vinterpret_cast(vadds(z_i, tirx.const(1, "int32"), mask), "float32"))
    r_pre = _bind(vabs(vmula(lhs_s, z_pre, y, mask), mask))
    r_next = _bind(vabs(vmula(lhs_s, z_next, y, mask), mask))
    z_corr = _bind(vsel(vcmp(r_pre, r, mask, "lt"), z_pre, z))
    r_best = _bind(vmin(r, r_pre, mask))
    z_corr = _bind(vsel(vcmp(r_next, r_best, mask, "lt"), z_next, z_corr))
    z_corr = _bind(vsel(inf_nan_cmp, z, z_corr))
    return vsel(zero_cmp, z, z_corr)


@_scope_guarded
def vcmp(lhs, rhs, seed, cmp, *, pmode=None, loc=None, ip=None):
    context = "T.vmi.vcmp(...)"
    _require_same_vreg_type(lhs, rhs, context=context)
    seed = _require_compatible_mask(seed, _lanes_of(lhs), context=context)
    cmp = _normalize_compare_predicate(cmp, context=context)
    return _call_vmi("vcmp", _dtype_of(seed), lhs, rhs, seed, cmp, pmode=pmode, loc=loc, ip=ip)


@_scope_guarded
def vcmps(source, scalar, seed, cmp, *, pmode=None, loc=None, ip=None):
    context = "T.vmi.vcmps(...)"
    _reject_packed_fp4(source, context=context)
    seed = _require_compatible_mask(seed, _lanes_of(source), context=context)
    cmp = _normalize_compare_predicate(cmp, context=context)
    return _call_vmi("vcmps", _dtype_of(seed), source, scalar, seed, cmp, pmode=pmode, loc=loc, ip=ip)


@_scope_guarded
def vsel(mask, true_value, false_value, *, pmode=None, loc=None, ip=None):
    context = "T.vmi.vsel(...)"
    result_dtype = _require_same_vreg_type(true_value, false_value, context=context)
    mask = _require_compatible_mask(mask, _lanes_of(true_value), context=context)
    return _call_vmi("vsel", result_dtype, mask, true_value, false_value, pmode=pmode, loc=loc, ip=ip)


@_scope_guarded
def vselr(source, index, *, loc=None, ip=None):
    _reject_packed_fp4(source, context="T.vmi.vselr(...)")
    _require_integer_vector(index, context="T.vmi.vselr(...)")
    if _lanes_of(source) != _lanes_of(index):
        raise TypeError("T.vmi.vselr(...) requires source and index lane counts to match")
    return _call_vmi("vselr", _dtype_of(source), source, index, loc=loc, ip=ip)


@_scope_guarded
def vbrc(value, *, size, group=None, loc=None, ip=None):
    if size is None:
        raise TypeError("T.vmi.vbrc(...) requires size")
    _reject_packed_fp4(value, context="T.vmi.vbrc(...)")
    if group is not None:
        if isinstance(group, bool) or not isinstance(group, int):
            raise TypeError("T.vmi.vbrc(...) requires group to be a positive Python integer")
        if group <= 0:
            raise ValueError(f"T.vmi.vbrc(...) requires group to be positive, got {group!r}")
        if not hasattr(_dtype_of(value), "lanes"):
            raise TypeError("T.vmi.vbrc(...) with group=... requires a VMI vector input")
        if _lanes_of(value) != group:
            raise ValueError("T.vmi.vbrc(...) with group=... requires the input lane count to match group")
    return _call_vmi(
        "vbrc",
        vreg(size, _element_dtype_of(value)),
        value,
        size=size,
        group=group,
        loc=loc,
        ip=ip,
    )


@_scope_guarded
def vcadd(source, mask, *, group=None, pmode=None, reassoc=None, loc=None, ip=None):
    context = "T.vmi.vcadd(...)"
    _reject_packed_fp4(source, context=context)
    elem_dtype = _element_dtype_of(source)
    if reassoc is not None and not isinstance(reassoc, bool):
        raise TypeError(f"{context} requires reassoc to be the Python boolean True or False; received {reassoc!r}")
    if reassoc is False:
        raise TypeError(
            f"{context} does not support reassoc=False: the PTO backend accepts only reassoc=True (integer vectors may omit reassoc)"
        )
    if reassoc is None and str(elem_dtype).startswith(("float", "bfloat")):
        raise TypeError(f"{context} on floating-point vectors requires an explicit reassoc=True argument; the PTO backend rejects omission")
    lanes = 1 if group is None else _normalize_lanes(group, context="T.vmi.vcadd(..., group=...)")
    if group is not None and _lanes_of(source) % group != 0:
        raise ValueError(f"{context} requires source lanes to be divisible by group")
    mask = _require_compatible_mask(mask, _lanes_of(source), context=context)
    return _call_vmi(
        "vcadd",
        vreg(lanes, elem_dtype),
        source,
        mask,
        group=group,
        pmode=pmode,
        reassoc=reassoc,
        loc=loc,
        ip=ip,
    )


@_scope_guarded
def vcmax(source, mask, *, group=None, pmode=None, loc=None, ip=None):
    _reject_packed_fp4(source, context="T.vmi.vcmax(...)")
    lanes = 1 if group is None else _normalize_lanes(group, context="T.vmi.vcmax(..., group=...)")
    if group is not None and _lanes_of(source) % group != 0:
        raise ValueError("T.vmi.vcmax(...) requires source lanes to be divisible by group")
    mask = _require_compatible_mask(mask, _lanes_of(source), context="T.vmi.vcmax(...)")
    return _call_vmi("vcmax", vreg(lanes, _element_dtype_of(source)), source, mask, group=group, pmode=pmode, loc=loc, ip=ip)


@_scope_guarded
def vcmin(source, mask, *, group=None, pmode=None, loc=None, ip=None):
    _reject_packed_fp4(source, context="T.vmi.vcmin(...)")
    lanes = 1 if group is None else _normalize_lanes(group, context="T.vmi.vcmin(..., group=...)")
    if group is not None and _lanes_of(source) % group != 0:
        raise ValueError("T.vmi.vcmin(...) requires source lanes to be divisible by group")
    mask = _require_compatible_mask(mask, _lanes_of(source), context="T.vmi.vcmin(...)")
    return _call_vmi("vcmin", vreg(lanes, _element_dtype_of(source)), source, mask, group=group, pmode=pmode, loc=loc, ip=ip)


@_scope_guarded
def vcvt(source, to_dtype=None, mask=None, *, rounding=None, saturate=None, pmode=None, loc=None, ip=None):
    if mask is not None:
        raise TypeError("T.vmi.vcvt does not support masked form")
    if to_dtype is None:
        raise TypeError("T.vmi.vcvt(...) requires to_dtype")
    to_dtype = _scalar_dtype(to_dtype, context="T.vmi.vcvt(..., to_dtype=...)")
    src_dt = _element_dtype_of(source)
    if str(to_dtype) == "float4_e2m1fn":
        if str(src_dt) != "bfloat16":
            raise TypeError("T.vmi.vcvt(...) supports packed FP4 only for bfloat16 to float4_e2m1fn")
        if _lanes_of(source) % 2:
            raise ValueError("T.vmi.vcvt(...) requires an even BF16 lane count for packed FP4")
        if rounding is not None:
            rounding = _normalize_pto_vcvt_rounding(
                rounding,
                context="T.vmi.vcvt(..., rounding=...)",
                allowed={"R", "A", "F", "Z", "C"},
            )
        if saturate is not None:
            raise ValueError("T.vmi.vcvt(...) does not support saturate for bfloat16 to packed FP4 conversion")
    elif str(src_dt) == "float4_e2m1fn":
        if str(to_dtype) not in {"bfloat16", "float32"}:
            raise TypeError("T.vmi.vcvt(...) supports packed FP4 sources only for float4_e2m1fn to bfloat16 or float32")
        if _lanes_of(source) % 2:
            raise ValueError("T.vmi.vcvt(...) requires an even packed FP4 logical lane count")
        # Result is vreg(logical, to_dtype). Check that width, not the
        # physical packed-pair count (logical // 2), which is not a VMI size.
        _require_vmi_lane_count(
            _lanes_of(source),
            context="T.vmi.vcvt(...) packed FP4 source lane count",
        )
        if rounding is not None:
            raise ValueError("T.vmi.vcvt(...) does not support rounding for packed FP4 source conversion")
        if saturate is not None:
            raise ValueError("T.vmi.vcvt(...) does not support saturate for packed FP4 source conversion")
        if pmode is not None:
            raise ValueError("T.vmi.vcvt(...) does not support pmode for packed FP4 source conversion")
    elif rounding is not None:
        rounding = _normalize_pto_vcvt_rounding(rounding, context="T.vmi.vcvt(..., rounding=...)")
    return _call_vmi(
        "vcvt",
        vreg(_lanes_of(source), to_dtype),
        source,
        to_dtype=str(to_dtype),
        rounding=rounding,
        saturate=saturate,
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


@_scope_guarded
def vinterpret_cast(source, to_dtype=None, *, loc=None, ip=None):
    if to_dtype is None:
        raise TypeError("T.vmi.vinterpret_cast(...) requires to_dtype")
    source_dt = _element_dtype_of(source)
    target_dt = _scalar_dtype(to_dtype, context="T.vmi.vinterpret_cast(..., to_dtype=...)")
    src_bits = getattr(source_dt, "bits", None)
    tgt_bits = getattr(target_dt, "bits", None)
    src_lanes = _lanes_of(source)
    if src_bits is None or tgt_bits is None:
        raise TypeError("T.vmi.vinterpret_cast(...) requires sized element dtypes")
    # Same element width keeps lane count; otherwise require bit-total match
    # (ASC vintlv pack: 128xbf16 → 64xf32).
    if src_bits == tgt_bits:
        out_lanes = src_lanes
    else:
        total = src_lanes * src_bits
        if total % tgt_bits != 0:
            raise TypeError("T.vmi.vinterpret_cast(...) requires source/target bit totals to match")
        out_lanes = total // tgt_bits
    # PTOAS rejects lane counts outside the formal VMI set (e.g. int8x128→si64
    # yields int64x16, which is not legal).
    _require_vmi_lane_count(out_lanes, context="T.vmi.vinterpret_cast(...)")
    return _call_vmi(
        "vinterpret_cast",
        vreg(out_lanes, target_dt),
        source,
        to_dtype=_pto_to_dtype_annotation(to_dtype),
        loc=loc,
        ip=ip,
    )


@_scope_guarded
def vexpdif(x, max_value, mask, *, pmode=None, loc=None, ip=None):
    _reject_packed_fp4(x, max_value, context="T.vmi.vexpdif(...)")
    return _call_vmi(
        "vexpdif", _dtype_of(max_value), x, max_value, _require_mask(mask, context="T.vmi.vexpdif(...)"), pmode=pmode, loc=loc, ip=ip
    )


@_scope_guarded
def vaxpy(x, acc, alpha, mask, *, pmode=None, loc=None, ip=None):
    _reject_packed_fp4(x, acc, context="T.vmi.vaxpy(...)")
    return _call_vmi("vaxpy", _dtype_of(acc), x, acc, alpha, _require_mask(mask, context="T.vmi.vaxpy(...)"), pmode=pmode, loc=loc, ip=ip)


@_scope_guarded
def vlrelu(x, slope, mask, *, pmode=None, loc=None, ip=None):
    _reject_packed_fp4(x, context="T.vmi.vlrelu(...)")
    return _call_vmi("vlrelu", _dtype_of(x), x, slope, _require_mask(mask, context="T.vmi.vlrelu(...)"), pmode=pmode, loc=loc, ip=ip)


@_scope_guarded
def vprelu(x, alpha, mask, *, pmode=None, loc=None, ip=None):
    _reject_packed_fp4(x, context="T.vmi.vprelu(...)")
    return _call_vmi("vprelu", _dtype_of(x), x, alpha, _require_mask(mask, context="T.vmi.vprelu(...)"), pmode=pmode, loc=loc, ip=ip)


@_scope_guarded
def vmull(a, b, mask, *, pmode=None, loc=None, ip=None):
    context = "T.vmi.vmull(...)"
    result_dtype = _require_same_vreg_type(a, b, context=context)
    elem_dtype = _element_dtype_of(a)
    if not str(elem_dtype).startswith(("int", "uint")) or elem_dtype.bits != 32:
        raise TypeError(f"{context} requires identical 32-bit integer vectors")
    mask = _require_compatible_mask(mask, _lanes_of(a), context=context)
    pair = _call_vmi("vmull", result_dtype, a, b, mask, pmode=pmode, loc=loc, ip=ip)
    return _wrap_pair(pair)


@_scope_guarded
def vmula(acc, lhs, rhs, mask, *, pmode=None, loc=None, ip=None):
    _reject_packed_fp4(acc, lhs, rhs, context="T.vmi.vmula(...)")
    return _call_vmi("vmula", _dtype_of(acc), acc, lhs, rhs, _require_mask(mask, context="T.vmi.vmula(...)"), pmode=pmode, loc=loc, ip=ip)


@_scope_guarded
def vdhist(acc, source, mask, *, loc=None, ip=None):
    if str(_dtype_of(acc)) != "uint16x256":
        raise TypeError("T.vmi.vdhist(...) requires a uint16x256 accumulator")
    if str(_element_dtype_of(source)) != "uint8":
        raise TypeError("T.vmi.vdhist(...) requires a uint8 source vector")
    mask = _require_compatible_mask(mask, _lanes_of(source), context="T.vmi.vdhist(...)")
    return _call_vmi("vdhist", _dtype_of(acc), acc, source, mask, loc=loc, ip=ip)


@_scope_guarded
def vchist(acc, source, mask, *, loc=None, ip=None):
    if str(_dtype_of(acc)) != "uint16x256":
        raise TypeError("T.vmi.vchist(...) requires a uint16x256 accumulator")
    if str(_element_dtype_of(source)) != "uint8":
        raise TypeError("T.vmi.vchist(...) requires a uint8 source vector")
    mask = _require_compatible_mask(mask, _lanes_of(source), context="T.vmi.vchist(...)")
    return _call_vmi("vchist", _dtype_of(acc), acc, source, mask, loc=loc, ip=ip)


@_scope_guarded
def vgather(source, offsets, mask, *, pmode=None, loc=None, ip=None):
    _require_ub_address(source, context="T.vmi.vgather(...)")
    source_elem = _require_address_element_dtype(source, context="T.vmi.vgather(...)")
    if str(source_elem) == "float4_e2m1fn":
        raise TypeError("T.vmi.vgather(...) does not support packed FP4 source buffers")
    _require_integer_vector(offsets, context="T.vmi.vgather(...)")
    mask = _require_compatible_mask(mask, _lanes_of(offsets), context="T.vmi.vgather(...)")
    ptr, offset = _resolve_ptr_and_offset(source, access_type="r", extent=_lanes_of(offsets))
    return _call_vmi(
        "vgather",
        _vector_result_dtype(
            offsets,
            elem_dtype=_vgather_result_element_dtype(source_elem, offsets),
        ),
        ptr,
        offset,
        offsets,
        _require_mask(mask, context="T.vmi.vgather(...)"),
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


@_scope_guarded
def vgatherb(source, offsets, mask, *, pmode=None, loc=None, ip=None):
    _require_ub_address(source, context="T.vmi.vgatherb(...)")
    source_elem = _require_address_element_dtype(source, context="T.vmi.vgatherb(...)")
    if str(source_elem) == "float4_e2m1fn":
        raise TypeError("T.vmi.vgatherb(...) does not support packed FP4 source buffers")
    _require_integer_vector(offsets, context="T.vmi.vgatherb(...)")
    mask = _require_compatible_mask(mask, _lanes_of(offsets), context="T.vmi.vgatherb(...)")
    ptr, offset = _resolve_ptr_and_offset(source, access_type="r", extent=_lanes_of(mask))
    return _call_vmi(
        "vgatherb",
        _vector_result_dtype(mask, elem_dtype=source_elem),
        ptr,
        offset,
        offsets,
        _require_mask(mask, context="T.vmi.vgatherb(...)"),
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


@_scope_guarded
def vscatter(value, destination, offsets, mask, *, pmode=None, loc=None, ip=None):
    _require_ub_address(destination, context="T.vmi.vscatter(...)")
    _reject_packed_fp4(value, context="T.vmi.vscatter(...)")
    destination_dtype = _address_element_dtype(destination)
    if destination_dtype is not None and destination_dtype != _element_dtype_of(value):
        raise TypeError("T.vmi.vscatter(...) requires value and destination element types to match")
    _require_integer_vector(offsets, context="T.vmi.vscatter(...)")
    if _lanes_of(value) != _lanes_of(offsets):
        raise TypeError("T.vmi.vscatter(...) requires value and offset lane counts to match")
    mask = _require_compatible_mask(mask, _lanes_of(value), context="T.vmi.vscatter(...)")
    ptr, offset = _resolve_ptr_and_offset(destination, access_type="w", extent=_lanes_of(value))
    return _call_vmi(
        "vscatter",
        "void",
        value,
        ptr,
        offset,
        offsets,
        _require_mask(mask, context="T.vmi.vscatter(...)"),
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


@_scope_guarded
def create_mask(active_lanes, *, size, group=None, loc=None, ip=None):
    if size is None:
        raise TypeError("T.vmi.create_mask(...) requires size")
    if group is None:
        return _call_vmi("create_mask", mask(size), active_lanes, size=size, loc=loc, ip=ip)
    if isinstance(group, bool) or not isinstance(group, int):
        raise TypeError("T.vmi.create_mask(...) requires group to be a positive Python integer")
    if group <= 0:
        raise ValueError(f"T.vmi.create_mask(...) requires group to be positive, got {group!r}")
    if size % group != 0:
        raise ValueError("T.vmi.create_mask(...) requires size to be divisible by group")
    group_size = size // group
    active_lanes_const = _constant_int(active_lanes)
    if active_lanes_const is not None and active_lanes_const > group_size:
        raise ValueError("T.vmi.create_mask(...) requires active_lanes to be <= the inferred group_size")
    return _call_vmi(
        "create_mask",
        mask(size),
        active_lanes,
        group=group,
        size=size,
        loc=loc,
        ip=ip,
    )


@_scope_guarded
def vintlv(lhs, rhs, mask, *, pmode=None, loc=None, ip=None):
    context = "T.vmi.vintlv(...)"
    result_dtype = _require_same_vreg_type(lhs, rhs, context=context)
    mask = _require_compatible_mask(mask, _lanes_of(lhs), context=context)
    pair = _call_vmi("vintlv", result_dtype, lhs, rhs, mask, pmode=pmode, loc=loc, ip=ip)
    return _wrap_pair(pair)


@_scope_guarded
def vdintlv(lhs, rhs, mask, *, pmode=None, loc=None, ip=None):
    context = "T.vmi.vdintlv(...)"
    result_dtype = _require_same_vreg_type(lhs, rhs, context=context)
    mask = _require_compatible_mask(mask, _lanes_of(lhs), context=context)
    pair = _call_vmi("vdintlv", result_dtype, lhs, rhs, mask, pmode=pmode, loc=loc, ip=ip)
    return _wrap_pair(pair)


@_scope_guarded
def vunzip(source, to_dtype=None, *, loc=None, ip=None):
    """Unzip every element into its low and high half-width halves.

    Returns (low, high) with the source lane count preserved and the element
    width halved; lane i of each result is the low/high half of source lane i.
    to_dtype selects the half element type and defaults to the unsigned
    half-width integer of the source element (uint16 for a 32-bit source).  It is
    forwarded to the emitted PTO call, where PTODSL derives the half-width result
    element type from it and re-checks it against the source width, so it must
    agree with the result dtype this wrapper declares.
    """
    context = "T.vmi.vunzip(...)"
    source_dtype = _dtype_of(source)
    lanes = int(getattr(source_dtype, "lanes", 1))
    src_bits = getattr(_element_dtype_of(source), "bits", None)
    if src_bits is None:
        raise TypeError(context + " requires a sized source element dtype")
    if src_bits not in (16, 32):
        raise ValueError(context + " requires a 16- or 32-bit source element so the half is 8 or 16 bits; got " + str(src_bits))
    half_bits = int(src_bits) // 2
    if to_dtype is None:
        half_elem = _scalar_dtype("uint" + str(half_bits), context=context)
    else:
        half_elem = _scalar_dtype(to_dtype, context=context)
        if getattr(half_elem, "bits", None) != half_bits:
            raise TypeError(context + " requires the half element width to be exactly half the source element width")
    pair = _call_vmi(
        "vunzip",
        vreg(lanes, half_elem),
        source,
        to_dtype=_pto_to_dtype_annotation(half_elem),
        loc=loc,
        ip=ip,
    )
    return _wrap_pair(pair)


@_scope_guarded
def vzip(low, high, to_dtype=None, *, loc=None, ip=None):
    """Zip low/high half-width halves back (inverse of vunzip).

    to_dtype selects the wide element type and defaults to the unsigned
    double-width integer of the half element (uint32 for uint16 halves).  As in
    vunzip it is forwarded to the emitted PTO call and must agree with the result
    dtype this wrapper declares.
    """
    context = "T.vmi.vzip(...)"
    lanes = _lanes_of(low)
    high_lanes = _lanes_of(high)
    low_elem = _element_dtype_of(low)
    high_elem = _element_dtype_of(high)
    if high_lanes != lanes or high_elem != low_elem:
        raise ValueError(
            context + " requires low and high to share one lane count and "
            "element type; got " + str(lanes) + "x" + str(low_elem) + " and " + str(high_lanes) + "x" + str(high_elem)
        )
    half_bits = getattr(low_elem, "bits", None)
    if half_bits is None:
        raise TypeError(context + " requires a sized half element dtype")
    if half_bits not in (8, 16):
        raise ValueError(context + " requires an 8- or 16-bit half element so the wide element is 16 or 32 bits; got " + str(half_bits))
    wide_bits = int(half_bits) * 2
    if to_dtype is None:
        wide_elem = _scalar_dtype("uint" + str(wide_bits), context=context)
    else:
        wide_elem = _scalar_dtype(to_dtype, context=context)
        if getattr(wide_elem, "bits", None) != wide_bits:
            raise TypeError(context + " requires the wide element width to be exactly twice the half element width")
    result_dtype = vreg(lanes, wide_elem)
    return _call_vmi(
        "vzip",
        result_dtype,
        low,
        high,
        to_dtype=_pto_to_dtype_annotation(wide_elem),
        loc=loc,
        ip=ip,
    )


__all__ = [
    "VmiPair",
    "alloc_local",
    "alloc_var",
    "vzip",
    "vunzip",
    "create_mask",
    "inside_vmi",
    "mask",
    "require_vmi_scope",
    "vabs",
    "vadd",
    "vadds",
    "vaxpy",
    "vbrc",
    "vchist",
    "vcadd",
    "vcmax",
    "vcmin",
    "vcmp",
    "vci",
    "vcmps",
    "vcvt",
    "vdiv",
    "vdhist",
    "vdintlv",
    "vexp",
    "vexpdif",
    "vgather",
    "vgatherb",
    "vintlv",
    "vinterpret_cast",
    "vln",
    "vlrelu",
    "vload",
    "vmax",
    "vmaxs",
    "vmin",
    "vmins",
    "vmul",
    "vmula",
    "vmull",
    "vmuls",
    "vneg",
    "vnot",
    "vor",
    "vprelu",
    "vscatter",
    "vsel",
    "vselr",
    "vshl",
    "vshls",
    "vshr",
    "vshrs",
    "vstore",
    "vrelu",
    "vreg",
    "vxor",
    "vand",
    "vsqrt",
]
