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


def _tight_buffer_span(buffer):
    """Return ``prod(shape)`` when *buffer* is tightly packed, else ``None``.

    A whole-buffer descriptor is only sound when the logical buffer covers one
    contiguous physical span.  Explicit strides must equal the row-major
    strides; default (empty) strides are row-major by construction.
    """
    ndim = len(buffer.shape)
    if ndim == 0:
        return None
    if buffer.strides:
        if len(buffer.strides) != ndim:
            return None
        from tvm import arith

        analyzer = arith.Analyzer()
        expected = 1
        for dim in range(ndim - 1, -1, -1):
            if not analyzer.can_prove_equal(buffer.strides[dim], expected):
                return None
            expected = expected * buffer.shape[dim]
    span = 1
    for dim in buffer.shape:
        span = span * dim
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


def _provably_non_negative_starts(source) -> bool:
    """Whether every source axis start is a scalar proven non-negative.

    Relocating a descriptor moves the source position into the instruction
    offset, which bypasses the normal negative-index legalization.  Only
    sources whose starts are provably non-negative may take that path.
    """
    if isinstance(source, Buffer):
        return True
    if isinstance(source, BufferLoad):
        starts = list(source.indices)
    elif isinstance(source, BufferRegion):
        starts = [region.min for region in source.region]
    else:
        return False
    from tvm import arith

    analyzer = arith.Analyzer()
    return all(getattr(start.dtype, "lanes", 1) == 1 and analyzer.can_prove(start >= 0) for start in starts)


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


def _descriptor_error(source, *, is_load, reason, size=None, group=None, stride=None, block_stride=None):
    """Build the frontend diagnostic for an unprovable physical footprint."""
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
    if params:
        context += ", " + ", ".join(params)
    return ValueError(
        f"{_vmi_op_name(is_load)} cannot construct a descriptor covering the complete physical access: "
        f"{reason} ({context}). No conservative range was generated; use an actual root buffer with a "
        "known finite range, or an explicit pointer whose declared extent covers this access."
    )


def _try_whole_buffer_plan(source, *, low, high):
    """Try a whole tight-buffer descriptor that provably covers ``[low, high)``.

    ``low`` and ``high`` are the complete instruction envelope relative to the
    source address.  The plan uses the buffer origin as the descriptor and
    carries the source position as the instruction offset, so it is only
    returned when a single linear relative offset is available, every source
    axis start is provably non-negative (otherwise moving the start would
    bypass the normal negative-index legalization), and the envelope lies
    inside ``[0, tight_span)``.  Otherwise the caller keeps trying other
    constructions and diagnoses after all of them fail.
    """
    buffer = source if isinstance(source, Buffer) else source.buffer
    if not _provably_non_negative_starts(source):
        return None
    span = _tight_buffer_span(buffer)
    relative = _source_relative_offset(source)
    if span is None or relative is None:
        return None
    from tvm import arith

    analyzer = arith.Analyzer()
    if not analyzer.can_prove(relative + low >= 0):
        return None
    if not analyzer.can_prove(relative + high <= span):
        return None
    return span, buffer, relative


def _padded_row_major(buffer) -> bool:
    """Whether every explicit stride is at least its dense row-major stride.

    This proves that every legal non-negative coordinate maps at or above its
    dense row-major offset, which is what keeps the logical prefix declared by
    :func:`_padded_origin_plan` conservative.  It does not characterize the
    layout as ordinary row-major padding; any view whose strides merely satisfy
    the inequality is accepted.  Views with a smaller stride on some dimension
    are rejected.
    """
    ndim = len(buffer.shape)
    if ndim == 0 or not buffer.strides or len(buffer.strides) != ndim:
        return False
    from tvm import arith

    analyzer = arith.Analyzer()
    dense_stride = 1
    for dim in range(ndim - 1, -1, -1):
        if not analyzer.can_prove(buffer.strides[dim] >= dense_stride):
            return False
        dense_stride = dense_stride * buffer.shape[dim]
    return True


def _padded_origin_plan(source, *, low, high):
    """Declare a static negative group stride from the padded buffer origin.

    The accessed envelope is ``[relative + low, relative + high)`` in element
    addresses, where ``relative`` is the source position from the buffer
    origin.  When each stride is at least its dense row-major stride, each
    logical element of the buffer maps at or above its row-major position, so
    every element whose address falls in the envelope sorts below
    ``relative + high`` in row-major order.  The declared logical prefix
    ``[0, relative + high)`` therefore covers the whole envelope, and the VMI
    instruction offset carries ``relative`` once from the descriptor origin.
    """
    if not isinstance(source, (BufferLoad, BufferRegion)):
        return None
    buffer = source.buffer
    relative = _source_relative_offset(source)
    if relative is None or not _padded_row_major(buffer):
        return None
    # A negative or unknown start must keep the normal negative-index
    # legalization instead of being moved into the instruction offset.
    if not _provably_non_negative_starts(source):
        return None
    from tvm import arith

    analyzer = arith.Analyzer()
    if not analyzer.can_prove(relative + low >= 0):
        return None
    return analyzer.simplify(relative + high), buffer, relative


def _group_access_plan(source, *, lanes, group, stride, dist_mode, is_load, elem_dtype):
    """May-access span of an equal-distance group access (C.2/C.3.1).

    The first group starts at the original access address ``A``.  For stride
    ``S`` and physical group width ``W`` the contiguous envelope is
    ``[A + min(0,(G-1)*S), A + max(0,(G-1)*S) + W)``.  A negative static
    stride on a 1-D unit-stride buffer relocates the descriptor to the lowest
    address and compensates with the VMI instruction offset when the original
    and shifted positions have the same proven sign, so negative-index
    legalization rewrites both or neither.  On a view whose strides are each
    at least their dense row-major stride and whose envelope stays at or above
    the buffer origin, the descriptor covers the logical prefix from the
    origin and the instruction offset carries the source position once.

    A group load is unmasked and therefore reads whole physical vectors: when
    the logical per-group width ``N/G`` is not a multiple of the physical VL,
    the last chunk still reads a full VL (e.g. f32 G=1/N=80 lowers to two
    full vlds covering 128 address elements).  A group store keeps the logical
    width because the backend's generated tail mask bounds the writes; do not
    copy the load rounding onto stores.

    A group or stride with no proven bounds, and a finite envelope that no
    candidate descriptor can express, are reported to the caller instead of
    being replaced by a fabricated range.  The 1-D relocation above requires
    the original and shifted positions to have the same proven sign; the
    origin and whole-tight candidates require every axis start to be provably
    non-negative.  A start that fails those conditions keeps the diagnostic
    because moving it would change the address selected by negative-index
    legalization.
    """
    group_const = _constant_int(group)
    if group_const is not None and group_const <= 0:
        raise ValueError(f"{_vmi_op_name(is_load)}(...) requires group to be a positive integer, got {group_const}")
    if group_const is None:
        raise _descriptor_error(
            source,
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
            return per_group, None, None
        raise _descriptor_error(
            source,
            is_load=is_load,
            reason="group stride has no proven bounds for this buffer view",
            size=lanes,
            group=group,
            stride=stride,
        )
    span = (group_const - 1) * stride_const
    low = min(0, span)
    high = max(0, span) + per_group
    extent = high - low
    if low == 0:
        return extent, None, None
    if isinstance(source, BufferLoad) and len(source.indices) == 1 and _is_unit_stride_1d(source.buffer):
        from tvm import arith

        analyzer = arith.Analyzer()
        original_index = source.indices[0]
        shifted_index = original_index + low
        # LegalizeNegativeIndex rewrites a provably negative index by adding
        # the buffer extent.  Relocation is address-preserving when the
        # original and shifted positions have the same proven sign: with both
        # non-negative neither is rewritten, and with both negative the same
        # pass rewrites both, so their difference ``low`` is unchanged.
        original_sign = _proven_sign(analyzer, original_index)
        if original_sign != 0 and original_sign == _proven_sign(analyzer, shifted_index):
            shifted = tirx.BufferLoad(source.buffer, [shifted_index])
            return extent, shifted, -low
    origin_plan = _padded_origin_plan(source, low=low, high=high)
    if origin_plan is not None:
        return origin_plan
    plan = _try_whole_buffer_plan(source, low=low, high=high)
    if plan is not None:
        return plan
    relative = _source_relative_offset(source)
    if relative is None:
        reason = "the access address cannot be reduced to a single linear element offset"
    elif not _provably_non_negative_starts(source):
        reason = (
            "the source index is negative or has no proven sign, so relocating the descriptor "
            "cannot preserve the address selected by negative-index legalization"
        )
    else:
        from tvm import arith

        analyzer = arith.Analyzer()
        if analyzer.can_prove(relative + low < 0):
            reason = "the finite access envelope starts below the buffer view origin and no constructible descriptor covers it"
        else:
            reason = "cannot prove that any constructible descriptor covers the finite access envelope"
    raise _descriptor_error(
        source,
        is_load=is_load,
        reason=reason,
        size=lanes,
        group=group,
        stride=stride,
    )


def _block_access_plan(source, *, lanes, elem_dtype, block_stride, is_load):
    """May-access span of a block-stride access (C.3.2).

    ``block_stride`` is measured in 32-byte blocks.  A full physical vector
    covers eight blocks; physical chunks beyond one vector start at the next
    block group, and the tail chunk of a masked store conservatively covers
    its whole block group as well.  A dynamic or negative control word has no
    proven physical bounds, and an unknown element width cannot size a block;
    those cases are reported instead of being replaced by a fabricated range.
    """
    bits = int(getattr(elem_dtype, "bits", 0) or 0)
    if bits <= 0:
        raise _descriptor_error(
            source,
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
            is_load=is_load,
            reason="block stride has no proven bounds for this buffer view",
            size=lanes,
            block_stride=block_stride,
        )
    if block_const < 0:
        raise _descriptor_error(
            source,
            is_load=is_load,
            reason="negative block-stride control words have no proven physical bounds",
            size=lanes,
            block_stride=block_stride,
        )
    extent = (blocks - 1) * block_const * elements_per_block + elements_per_block
    return extent, None, None


def _vmi_access_plan(source, *, lanes, elem_dtype, dist_mode, group, stride, block_stride, is_load):
    """Compute (descriptor extent, descriptor base, instruction offset).

    ``descriptor base`` is ``None`` when the original source is already the
    right ``access_ptr`` base.  The instruction offset is ``None`` for zero.
    Only buffer sources are analysed; explicit pointers keep their caller
    declared ``ptr + offset`` ABI untouched.
    """
    if not isinstance(source, (BufferLoad, Buffer, BufferRegion)):
        return None, None, None
    if dist_mode == "dintlv":
        # Two interleaved inputs/outputs: the complete 2*N span, regardless of
        # the (ineffective) store mask.
        return 2 * lanes, None, None
    if group is not None:
        return _group_access_plan(
            source,
            lanes=lanes,
            group=group,
            stride=stride,
            dist_mode=dist_mode,
            is_load=is_load,
            elem_dtype=elem_dtype,
        )
    if block_stride is not None:
        return _block_access_plan(source, lanes=lanes, elem_dtype=elem_dtype, block_stride=block_stride, is_load=is_load)
    if str(elem_dtype) == "float4_e2m1fn":
        # Genuine FP4 descriptors stay in logical FP4 element units;
        # RewriteFp4ToFp4x2 performs the single physical packing.
        return lanes, None, None
    if dist_mode == "brc":
        # An ordinary broadcast reads one source element; keeping the old
        # conservative `size` declaration is allowed and does not change the
        # instruction semantics.
        return lanes, None, None
    if is_load:
        bits = int(getattr(elem_dtype, "bits", 0) or 0)
        if bits <= 0:
            raise _descriptor_error(
                source,
                is_load=True,
                reason="the physical element width is unknown, so the complete physical vector width cannot be computed",
                size=lanes,
            )
        vector_elements = max(1, 2048 // bits)
        # An unmasked load reads whole physical vectors: cover the tail chunk.
        return (lanes + vector_elements - 1) // vector_elements * vector_elements, None, None
    return lanes, None, None


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
    source_elem = _require_address_element_dtype(source, context="T.vmi.vload(...)")
    is_packed_fp4 = str(source_elem) == "float4_e2m1fn"
    if is_packed_fp4:
        if size % 2:
            raise ValueError("T.vmi.vload(...) requires an even FP4 lane count for packed FP4 storage")
        if dist_mode == "dintlv":
            raise TypeError('T.vmi.vload(...) does not support packed FP4 with dist_mode="dintlv"')
        if group is not None or block_stride is not None or dist_mode == "brc":
            raise TypeError("T.vmi.vload(...) only supports contiguous packed FP4 loads")

    result_dtype = _vector_result_dtype(source, size, elem_dtype=source_elem)
    extent, descriptor, instruction_offset = _vmi_access_plan(
        source,
        lanes=size,
        elem_dtype=source_elem,
        dist_mode=dist_mode,
        group=group,
        stride=stride,
        block_stride=block_stride,
        is_load=True,
    )
    ptr, extra_offset = _resolve_ptr_and_offset(
        source,
        offset,
        access_type="r",
        extent=extent,
        descriptor=descriptor,
        instruction_offset=instruction_offset,
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

    extent, descriptor, instruction_offset = _vmi_access_plan(
        destination,
        lanes=store_lanes,
        elem_dtype=value_dtype,
        dist_mode=dist_mode,
        group=group,
        stride=stride,
        block_stride=block_stride,
        is_load=False,
    )
    ptr, extra_offset = _resolve_ptr_and_offset(
        destination,
        offset,
        access_type="w",
        extent=extent,
        descriptor=descriptor,
        instruction_offset=instruction_offset,
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
