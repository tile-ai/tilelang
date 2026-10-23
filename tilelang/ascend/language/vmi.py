"""TileLang VMI namespace for Ascend PTO kernels."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps

from tvm import tirx
from tvm.tirx import Buffer, BufferLoad, BufferRegion
from tvm.tirx.script.builder.ir import bind as _bind, sblock_attr as _sblock_attr
from tvm.script.ir_builder import IRBuilder

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


_UB_SCOPES = frozenset({"shared", "shared.dyn"})


def _require_ub_address(value, *, context: str):
    buffer = None
    base_load = _access_ptr_base_load(value)
    if base_load is not None:
        buffer = base_load.buffer
    elif isinstance(value, (BufferLoad, BufferRegion)):
        buffer = value.buffer
    elif isinstance(value, Buffer):
        buffer = value
    if buffer is not None and buffer.scope() not in _UB_SCOPES:
        raise TypeError(f"{context} requires a UB pointer, got buffer scope {buffer.scope()!r}")
    type_annotation = getattr(value, "type_annotation", None)
    storage_scope = getattr(type_annotation, "storage_scope", None)
    if storage_scope and storage_scope not in _UB_SCOPES:
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


def _is_integer_scalar(expr) -> bool:
    dtype = getattr(expr, "dtype", None)
    if dtype is None or int(getattr(dtype, "lanes", 1)) != 1:
        return False
    return str(dtype).startswith(("int", "uint"))


# Node types accepted in an explicit ``tl.access_ptr`` extent expression.  A
# BufferLoad (for example a mutable local.var read) or any call is rejected:
# an extent that can change at run time or depends on a call is not a stable
# structural declaration for the descriptor.  Cast nodes are kept as-is so
# narrowed or overflowing arithmetic is never silently reinterpreted.
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


def _is_stable_integer_arithmetic(expr) -> bool:
    """Whether *expr* is a stable scalar integer extent declaration.

    This is a frontend API/type check on the caller-declared extent, not a
    range proof: containment and logical coverage are decided by
    ``LegalizePTOVMIMemoryAccess``.
    """
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
    """The parsed address information needed by VMI lowering."""

    __slots__ = ("source", "buffer", "starts")

    def __init__(self, *, source, buffer, starts):
        self.source = source
        self.buffer = buffer
        self.starts = starts


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
    # ``tl.access_ptr`` is a public frontend API with a fixed (base, extent,
    # rw_mask) shape; the frontend decodes that shape rather than re-auditing
    # a contract the constructor already guarantees.
    base, extent, mask = call.args
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
    return _VmiAddress(source=source, buffer=buffer, starts=starts)


def _parse_vmi_address(source, *, is_load, context):
    """Resolve one VMI address form exactly once.

    Accepted forms are ``Buffer``, ``BufferLoad``, ``BufferRegion``, a direct
    ``tl.access_ptr`` call, and a handle ``Var`` whose direct let binding is that
    call.  Every form must address a UB buffer with one scalar linear offset and
    no axis separators; a numeric ``Var`` is never followed to its definition
    and treated as a pointer.  Python aliases of the same object resolve
    identically; genuine multi-level handle chains, raw ``data`` handles, and
    unknown handles are rejected with a controlled diagnostic.  The frontend
    keeps the resulting address raw; the descriptor Pass builds an ordinary
    descriptor from it and rebuilds an indexed gather/scatter
    descriptor at the real owner origin.
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
        return _VmiAddress(source=source, buffer=buffer, starts=starts)
    raise TypeError(f"{context} requires a Buffer, BufferLoad, BufferRegion, or an explicit tl.access_ptr")


def _normalize_vmi_address_operand(source, *, context, offset=None):
    """Return the raw VMI address operand and its scalar instruction offset.

    Automatic buffer addresses become the ``BufferLoad(buffer, scalar_starts)``
    consumed by ``LegalizePTOVMIMemoryAccess``; the descriptor itself is built
    later in TIR.  An ordinary explicit pointer operand keeps the original
    expression and caller offset so the pass can verify the declaration without
    rewriting it.  For indexed gather/scatter, an explicit or bound
    pointer is only the address-snapshot input: the pass rejects a rebuild that
    would re-read memory, then emits the owner-origin descriptor and the D
    offset.  The instruction offset remains a frontend API/type check (scalar
    integer): it is part of the indexed address / the position inside an
    ordinary explicit descriptor, not a coverage proof.
    """
    if isinstance(source, (BufferLoad, Buffer, BufferRegion)):
        if offset is not None:
            raise TypeError("T.vmi buffer addresses do not accept an offset; express the offset in the buffer index")
        buffer = source if isinstance(source, Buffer) else source.buffer
        starts = _scalar_start_indices(source, buffer=buffer, context=context, ramp_base=True)
        return tirx.BufferLoad(buffer, starts), _zero_offset()
    if offset is None:
        return source, _zero_offset()
    offset = offset if isinstance(offset, tirx.PrimExpr) else tirx.convert(offset)
    if not _is_integer_scalar(offset):
        raise TypeError(f"{context} requires a scalar integer instruction offset, got {offset!r}")
    return source, offset


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
    ptr, extra_offset = _normalize_vmi_address_operand(address.source, context=context, offset=offset)
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
    ptr, extra_offset = _normalize_vmi_address_operand(address.source, context=context, offset=offset)
    if str(value_dtype) == "float4_e2m1fn" and mask is not None:
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
    address = _parse_vmi_address(source, is_load=True, context="T.vmi.vgather(...)")
    ptr, offset = _normalize_vmi_address_operand(address.source, context="T.vmi.vgather(...)")
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
    address = _parse_vmi_address(destination, is_load=False, context="T.vmi.vscatter(...)")
    ptr, offset = _normalize_vmi_address_operand(address.source, context="T.vmi.vscatter(...)")
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
