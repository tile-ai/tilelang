"""TileLang VMI namespace for Ascend PTO kernels."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps

from tvm import tirx
from tvm.tirx import Buffer, BufferLoad, BufferRegion
from tvm.tirx.script.builder.ir import bind as _bind
from tvm.script.ir_builder import IRBuilder

from tilelang.language.builtin import access_ptr
from tilelang.language.dtypes import dtype as _dtype
from .frame import inside_simdvf as _inside_simdvf

_Op = tirx.op.Op.get


def alloc_local(shape, vreg_type):
    """Allocate a statically indexed array of VMI vector registers.

    ``vreg_type`` is a vector dtype produced by :func:`vreg`.  The resulting
    buffer may only be indexed by compile-time values, such as induction
    variables from ``T.unroll(..., explicit=True)`` on the PTO backend.
    """
    require_vmi_scope("T.vmi.alloc_local(...)")
    dtype = _dtype(vreg_type)
    if int(getattr(dtype, "lanes", 1)) <= 1:
        raise TypeError(f"T.vmi.alloc_local(...) expects a VMI vector type produced by T.vmi.vreg(...), got {dtype}")

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

    return _alloc_local(tuple(normalized_shape), dtype, scope="local")


def alloc_var(dtype, *, size):
    """Allocate a mutable VMI vector register variable.

    This mirrors ``T.simd.alloc_var`` but preserves the VMI lane count in the
    TIR vector dtype, allowing loop-carried VMI accumulators to be rebound
    without eager-builder immutable-value diagnostics.
    """
    lanes = _normalize_lanes(size, context="T.vmi.alloc_var(...)")
    elem = _scalar_dtype(dtype, context="T.vmi.alloc_var(...)")
    from tilelang.language.allocate import alloc_var as _alloc_var

    return _alloc_var(f"{elem}x{lanes}", scope="local.var")


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


def _require_ub_address(value, *, context: str):
    buffer = None
    if isinstance(value, (BufferLoad, BufferRegion)):
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
    repeat_stride,
    allow_group_brc: bool,
    allowed_dist_modes,
):
    if dist_mode is not None and dist_mode not in allowed_dist_modes:
        expected = ", ".join(repr(mode) for mode in sorted(allowed_dist_modes, key=str))
        raise TypeError(f"{context} does not support dist_mode={dist_mode!r}; expected one of {expected}")

    if group is not None:
        if dist_mode is not None and (not allow_group_brc or dist_mode != "brc"):
            raise TypeError(f"{context} does not allow dist_mode together with group")
        if block_stride is not None or repeat_stride is not None:
            raise TypeError(f"{context} does not allow block_stride together with group")
        if stride is None:
            raise TypeError(f"{context} with group=... requires stride")
        return

    if block_stride is not None or repeat_stride is not None:
        if dist_mode is not None:
            raise TypeError(f"{context} does not allow dist_mode together with block_stride")
        if block_stride is None or repeat_stride is None:
            raise TypeError(f"{context} requires block_stride and repeat_stride together")
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


def _resolve_ptr_and_offset(source, offset=None, *, access_type: str, extent=None):
    """Normalize VMI address syntax into PTODSL's explicit ptr+offset ABI."""
    if isinstance(source, (BufferLoad, Buffer, BufferRegion)) and offset is not None:
        raise TypeError("T.vmi buffer addresses do not accept an offset; express the offset in the buffer index")
    if isinstance(source, BufferLoad):
        if len(source.indices) != 1:
            # Keep the linear element offset outside access_ptr.  In particular,
            # FP4 callers convert this logical offset to f4e2m1x2 units before
            # PTO emission; leaving it embedded in access_ptr would make PTO
            # interpret it as an already-packed offset.
            strides = source.buffer.strides
            if len(strides) != len(source.indices):
                strides = []
                for index in range(len(source.indices)):
                    stride = 1
                    for extent_dim in source.buffer.shape[index + 1 :]:
                        stride = stride * extent_dim
                    strides.append(stride)
            linear_offset = None
            for index, stride in zip(source.indices, strides):
                term = index * stride
                linear_offset = term if linear_offset is None else linear_offset + term
            ptr = access_ptr(source.buffer, access_type, extent=extent, offset=0)
            return ptr, linear_offset if linear_offset is not None else _zero_offset()
        ptr = access_ptr(source.buffer, access_type, extent=extent, offset=0)
        return ptr, source.indices[0]
    if isinstance(source, (Buffer, BufferRegion)):
        ptr = access_ptr(source, access_type, extent=extent, offset=0)
        return ptr, _zero_offset()
    if offset is None:
        return source, _zero_offset()
    return source, offset


def _pair_result_dtype(value):
    return _dtype_of(value)


def _vector_result_dtype(value, size=None, *, elem_dtype=None):
    elem = _dtype(elem_dtype) if elem_dtype is not None else _element_dtype_of(value)
    lanes = _lanes_of(value) if size is None else _normalize_lanes(size, context="T.vmi result size")
    return _dtype(f"{elem}x{lanes}")


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
    to_dtype=None,
    stride=None,
    block_stride=None,
    repeat_stride=None,
    dist_mode=None,
    group=None,
    loc=None,
    ip=None,
):
    """Load a VMI vector from either `a_ub[0]`-style buffer loads or ptr+offset form."""
    _validate_pto_load_modes(
        "T.vmi.vload(...)",
        dist_mode=dist_mode,
        group=group,
        stride=stride,
        block_stride=block_stride,
        repeat_stride=repeat_stride,
        allow_group_brc=True,
        allowed_dist_modes={None, "continuous", "dintlv", "unpack", "brc"},
    )
    if size is None:
        raise TypeError("T.vmi.vload(...) requires size")
    if to_dtype is not None and dist_mode != "unpack":
        raise TypeError('T.vmi.vload(...) accepts to_dtype only when dist_mode="unpack"')
    source_elem = _require_address_element_dtype(source, context="T.vmi.vload(...)")
    # Keep si*/ui* spelling for PTODSL annotations. Unpack vload itself is not
    # legalized on the current VPTO path; this preserves a correct annotation
    # if enabled later.
    to_dtype_annot = None
    is_packed_fp4 = str(source_elem) == "float4_e2m1fn"
    if is_packed_fp4:
        if size % 2:
            raise ValueError("T.vmi.vload(...) requires an even FP4 lane count for packed FP4 storage")
        if to_dtype is not None or dist_mode == "unpack":
            raise TypeError("T.vmi.vload(...) does not unpack or convert packed FP4 source vectors")
        if dist_mode == "dintlv":
            raise TypeError('T.vmi.vload(...) does not support packed FP4 with dist_mode="dintlv"')
        if group is not None or block_stride is not None or repeat_stride is not None or dist_mode == "brc":
            raise TypeError("T.vmi.vload(...) only supports contiguous packed FP4 loads")
    if dist_mode == "unpack":
        if to_dtype is None:
            raise TypeError('T.vmi.vload(...) requires to_dtype when dist_mode="unpack"')
        to_dtype_annot = _pto_to_dtype_annotation(to_dtype)
        to_dtype = _scalar_dtype(to_dtype, context="T.vmi.vload(..., to_dtype=...)")
        if getattr(to_dtype, "bits", None) != 2 * getattr(source_elem, "bits", None):
            raise TypeError("T.vmi.vload(...) unpack must widen by exactly one step")

    result_dtype = _vector_result_dtype(
        source,
        size,
        elem_dtype=to_dtype if to_dtype is not None else source_elem,
    )
    ptr, extra_offset = _resolve_ptr_and_offset(source, offset, access_type="r", extent=size)
    if is_packed_fp4:
        extra_offset = _pack_fp4_element_offset(extra_offset, context="T.vmi.vload(...)")
    attrs = {
        "size": size,
        "to_dtype": to_dtype_annot,
        "stride": stride,
        "block_stride": block_stride,
        "repeat_stride": repeat_stride,
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
    repeat_stride=None,
    dist_mode=None,
    group=None,
    pmode=None,
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
        repeat_stride=repeat_stride,
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

    store_extent = _lanes_of(values[0]) if _is_sequence(values) else _lanes_of(values)
    ptr, extra_offset = _resolve_ptr_and_offset(destination, offset, access_type="w", extent=store_extent)
    value_args = list(values) if _is_sequence(values) else [values]
    if value_dtype is None:
        value_dtype = _element_dtype_of(values[0]) if _is_sequence(values) else _element_dtype_of(values)
    if str(value_dtype) == "float4_e2m1fn":
        if store_extent % 2:
            raise ValueError("T.vmi.vstore(...) requires an even FP4 lane count for packed FP4 storage")
        if dist_mode == "dintlv":
            raise TypeError('T.vmi.vstore(...) does not support packed FP4 with dist_mode="dintlv"')
        if group is not None or block_stride is not None or repeat_stride is not None:
            raise TypeError("T.vmi.vstore(...) only supports contiguous packed FP4 stores")
        extra_offset = _pack_fp4_element_offset(extra_offset, context="T.vmi.vstore(...)")
        if mask is not None:
            mask = _require_mask(mask, context="T.vmi.vstore(...) of FP4")
            if not str(_dtype_of(mask)).startswith("bool"):
                raise TypeError("T.vmi.vstore(...) of FP4 expects a VMI mask operand")
            physical_lanes = store_extent // 2
            if _lanes_of(mask) != physical_lanes:
                raise TypeError(
                    "T.vmi.vstore(...) of FP4 requires a physical FP4x2 mask with "
                    f"half as many lanes as its {store_extent}-lane logical FP4 value "
                    f"(expected {physical_lanes}, got {_lanes_of(mask)})"
                )
    elif mask is not None:
        mask = _require_mask(mask, context="T.vmi.vstore(...)")
    return _call_vmi(
        "vstore",
        "void",
        *value_args,
        ptr,
        extra_offset,
        mask,
        stride=stride,
        block_stride=block_stride,
        repeat_stride=repeat_stride,
        dist_mode=dist_mode,
        group=group,
        pmode=pmode,
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
vdiv = _binary_same_dtype("vdiv")
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
    if reassoc is None and str(elem_dtype).startswith(("float", "bfloat")):
        raise TypeError(
            f"{context} on floating-point vectors requires an explicit reassoc argument; spell out reassoc=True or reassoc=False"
        )
    if reassoc is not None and not isinstance(reassoc, bool):
        raise TypeError(f"{context} requires reassoc to be the Python boolean True or False; received {reassoc!r}")
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
        raise TypeError("T.vmi.vcvt(...) does not support packed FP4 source vectors")
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
        _vector_result_dtype(offsets, elem_dtype=source_elem),
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


__all__ = [
    "VmiPair",
    "alloc_local",
    "alloc_var",
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
