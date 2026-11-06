from __future__ import annotations

from tilelang.language import _construction
from tilelang.language import _tensor_spec
from tvm_tirx.ir import prim as _prim

from tilelang._typing import BufferLikeType
from tilelang.language._tensor_spec import TensorValue as Buffer
from tvm_tirx.ir import TensorRegion as BufferRegion
from tvm_tirx.ir import Expr as PrimExpr
from tilelang.language.utils import region as _make_region_call
from tilelang.language.utils import buffer_region_to_tile_region
from tilelang.language.utils import get_buffer_region_from_load
from functools import reduce
import tvm_ffi
from tvm_tirx.ir import IRModule
from tvm_ffi import dtype as DataType
from tvm_tirx.tirx import Function as PrimFunc
from tvm_tirx import ir

from tvm_tirx.ir import CallEffectKind
# Scope Checkers for TVM Buffers
# These utility functions check the memory scope of a given TVM buffer.


def _get_buffer(buffer_or_load_or_region: BufferLikeType) -> Buffer:
    """
    Extract Buffer from Buffer, BufferLoad, or BufferRegion.

    Args:
        buffer_or_load_or_region: Can be Buffer, BufferLoad, or BufferRegion

    Returns:
        Buffer: The underlying buffer object
    """
    if _tensor_spec.is_buffer_like(buffer_or_load_or_region):
        return buffer_or_load_or_region
    elif isinstance(buffer_or_load_or_region, (ir.TensorLoad, ir.TensorRegion)):
        return buffer_or_load_or_region.source
    else:
        raise TypeError(f"Expected Buffer, BufferLoad, or BufferRegion, got {type(buffer_or_load_or_region)}")


def is_global(buffer: BufferLikeType) -> bool:
    """
    Check if the buffer is in the global memory scope.

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in global memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return str(buffer.scope()) == "global"


def is_shared(buffer: BufferLikeType, allow_dynamic: bool = True) -> bool:
    """
    Check if the buffer is in the shared memory scope.

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in shared memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    conditions = [False]
    conditions.append(str(buffer.scope()) == "shared")
    if allow_dynamic:
        conditions.append(is_shared_dynamic(buffer))
    return any(conditions)


def is_shared_dynamic(buffer: BufferLikeType) -> bool:
    """
    Check if the buffer is in the dynamic shared memory scope.

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in dynamic shared memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return str(buffer.scope()) == "shared.dyn"


def is_tensor_memory(buffer: BufferLikeType) -> bool:
    """
    Check if the buffer is in tensor memory scope (e.g., shared.tmem).

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in tensor memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return str(buffer.scope()).startswith("shared.tmem")


def is_local(buffer: BufferLikeType) -> bool:
    """
    Check if the buffer is in the local memory scope.

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in local memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return str(buffer.scope()) == "local"


def is_fragment(buffer: BufferLikeType) -> bool:
    """
    Check if the buffer is a fragment (e.g., for matrix multiplication operations).

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is a fragment, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return str(buffer.scope()).startswith("local.fragment")


def is_reducer(buffer: BufferLikeType) -> bool:
    """
    Check if the buffer is a reducer v2 accumulator (scope local.reducer).

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is a reducer accumulator, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return str(buffer.scope()) == "local.reducer"


def is_local_var(buffer: BufferLikeType) -> bool:
    """
    Check if the buffer is in the local.var memory scope.

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in local.var memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return str(buffer.scope()) == "local.var"


def get_buffer_elems(buffer: Buffer) -> int:
    """
    Get the number of elements in the buffer.
    """
    return reduce(lambda x, y: x * y, buffer.shape)


def array_reduce(array: list[int]) -> int:
    """
    Reduce an array of integers to a single integer.

    Args:
        array (List[int]): The array of integers to reduce.

    Returns:
        int: The reduced integer.
    """
    return reduce(lambda x, y: x * y, array)


def retrieve_func_from_module(ir_module: IRModule) -> PrimFunc:
    """
    Retrieve the single PrimFunc from an IRModule.

    Args:
        ir_module (IRModule): The TVM IRModule to extract the function from.
            The module should contain exactly one global function.

    Returns:
        PrimFunc: The single function contained in the module.

    Raises:
        ValueError: If ir_module is not an IRModule.
        AssertionError: If the module contains more than one global function.
    """
    if not isinstance(ir_module, IRModule):
        raise ValueError("Not supported type: ", type(ir_module))
    assert len(ir_module.get_global_vars()) == 1, "The optimized module should only have one global variable for default schedule."
    func = list(ir_module.functions.values())[0]
    return func


def to_buffer_region(obj: BufferLikeType, access_type: str = "rw", extents: list[PrimExpr] | None = None) -> PrimExpr | BufferRegion:
    """
    Normalize a buffer-like object to BufferRegion, or encode a tl.region call.

    - Buffer/BufferLoad/BufferRegion -> returns BufferRegion when extents is None
    - Buffer/BufferLoad/BufferRegion -> returns a tl.region call when extents is provided
    """
    from tilelang.language.frame import has_let_value, get_let_value

    # Tensor Vars are the native storage/view identity. Only scalar aliases
    # need their source region recovered; unwrapping a tensor would replace
    # its definition with an unbound decl_tensor Call.
    if isinstance(obj, ir.Var) and not _tensor_spec.is_buffer_like(obj) and has_let_value(obj):
        obj = get_let_value(obj)
    # Encode into tl.region call (when extents is provided), otherwise return BufferRegion for analysis
    if isinstance(obj, ir.TensorRegion):
        if extents is None:
            return obj
        mins = [r.begin for r in obj.region]
        exts = [r.extent for r in obj.region]
        assert len(extents) == len(exts)
        exts = [_prim.min(exts[i], extents[i]) for i in range(len(exts))]
        return _make_region_call(
            ir.TensorLoad(obj.source, _construction.as_indices(mins), _construction.current_location()), access_type, *exts
        )
    if _tensor_spec.is_buffer_like(obj):
        mins = [ir.IntImm("int32", 0) for _ in obj.shape]
        if extents is None:
            ranges = [ir.Range.from_min_extent(m, e) for m, e in zip(mins, obj.shape)]
            return ir.TensorRegion(obj, ranges, ir.TensorRegionType(), _construction.current_location())
        exts = list(extents)
        return _make_region_call(ir.TensorLoad(obj, _construction.as_indices(mins), _construction.current_location()), access_type, *exts)
    if isinstance(obj, ir.TensorLoad):
        if extents is None:
            region = get_buffer_region_from_load(obj)
            if region is not None:
                return region
            mins = [idx for idx in obj.indices]
            ones = [ir.IntImm("int32", 1) for _ in obj.indices]
            ranges = [ir.Range.from_min_extent(m, e) for m, e in zip(mins, ones)]
            return ir.TensorRegion(obj.source, ranges, ir.TensorRegionType(), _construction.current_location())
        exts = list(extents)
        if len(obj.indices) > len(exts):
            exts = [ir.IntImm("int32", 1) for _ in range(len(obj.indices) - len(exts))] + exts
        assert len(obj.indices) == len(exts)
        return _make_region_call(obj, access_type, *exts)
    raise ValueError(f"Unsupported argument type for to_buffer_region: {type(obj)}")


def to_tile_region(obj: BufferLikeType, access_type: str = "rw", extents: list[PrimExpr] | None = None) -> PrimExpr:
    """Encode a buffer-like object as a tl.region call for tile-op arguments."""
    buffer_region = to_buffer_region(obj)
    if not isinstance(buffer_region, ir.TensorRegion):
        raise ValueError(f"Expected BufferRegion from to_buffer_region, got {type(buffer_region)}")
    if extents is None:
        extents = [r.extent for r in buffer_region.region]
    return buffer_region_to_tile_region(buffer_region, access_type, extents)


def retrieve_shape(obj: BufferLikeType) -> list:
    """
    Retrieve shape-like extents for a buffer-like object.

    - Buffer -> its `shape`
    - BufferRegion -> list of each range's `extent`
    - BufferLoad -> extents from `get_buffer_region_from_load(obj)`
    """
    if _tensor_spec.is_buffer_like(obj):
        return obj.shape
    if isinstance(obj, ir.TensorRegion):
        return [r.extent for r in obj.region]
    if isinstance(obj, ir.TensorLoad):
        region = get_buffer_region_from_load(obj)
        if region is None:
            raise ValueError("Cannot retrieve shape from scalar BufferLoad without region")
        return [r.extent for r in region.region]
    raise ValueError(f"Unsupported retrieve_shape argument type: {type(obj)} for object {obj}")


def retrieve_stride(obj: BufferLikeType) -> list:
    """
    Retrieve strides for a buffer-like object.

    For BufferRegion and BufferLoad, uses the underlying buffer. Declared
    strides take precedence, with compact row-major strides as the fallback.
    """
    # Imported lazily to avoid a cycle: `tilelang.language` imports this module.

    if _tensor_spec.is_buffer_like(obj):
        buffer = obj
    elif isinstance(obj, (ir.TensorRegion, ir.TensorLoad)):
        buffer = obj.source
    else:
        raise ValueError(f"Unsupported retrieve_stride argument type: {type(obj)} for object {obj}")

    return list(_tensor_spec.tensor_strides(buffer))


def retrive_ptr_from_buffer_region(buffer_or_load_or_region: BufferLikeType, access_type: str = "r") -> PrimExpr:
    """Compatibility spelling for the common native tensor-address helper."""
    return retrieve_ptr(buffer_or_load_or_region, access_type)


def retrieve_ptr(
    obj: BufferLikeType,
    access_type: str = "r",
    ignore_last_ndim: int = 0,
) -> PrimExpr:
    """
    Retrieve a pointer to the start of a (possibly sliced) buffer region.

    - Buffer -> base pointer
    - BufferRegion -> pointer to the logical region minima
    - BufferLoad -> pointer offset computed from indices or derived region

    Args:
        obj: Buffer-like object
        access_type: access mask, e.g. "r", "w", "rw"
        ignore_last_ndim: do not offset the last N dimensions
    """
    # Keep the address in tensor coordinates until physical layout lowering.
    # The interval matches the former Buffer.access_ptr default: the remaining
    # storage span from the chosen start, expressed in tensor elements.
    from tilelang.language.builtin import access_ptr

    if _tensor_spec.is_buffer_like(obj):
        buffer = obj
        mins = [0] * len(buffer.shape)
    elif isinstance(obj, ir.TensorRegion):
        buffer, region = obj.source, obj.region
        mins = [r.begin for r in region]
    elif isinstance(obj, ir.TensorLoad):
        buffer = obj.source
        region = get_buffer_region_from_load(obj)
        if region is not None:
            mins = [r.begin for r in region.region]
        else:
            mins = list(obj.indices)
    else:
        raise ValueError(f"Unsupported retrieve_ptr argument type: {type(obj)} for object {obj}")
    ignore_last_ndim = int(ignore_last_ndim)
    if ignore_last_ndim < 0:
        raise ValueError("ignore_last_ndim must be nonnegative")
    strides = retrieve_stride(buffer)
    upto = max(0, len(mins) - ignore_last_ndim)
    offset = sum(mins[i] * strides[i] for i in range(upto))
    extent = strides[0] * buffer.shape[0] - offset if buffer.shape else 1
    return access_ptr(obj, access_type, extent=extent, ignore_last_ndim=ignore_last_ndim)


def retrieve_buffer_and_offset(obj: BufferLikeType) -> tuple[Buffer, PrimExpr | int]:
    """
    Retrieve the underlying buffer together with its logical element offset.

    - Buffer -> (buffer, 0)
    - BufferRegion -> (buffer, offset from region minima)
    - BufferLoad -> (buffer, offset from indices or derived region minima)

    This is useful when callers need to build custom access patterns from a
    common buffer base rather than materializing a full `access_ptr` directly.
    """
    if _tensor_spec.is_buffer_like(obj):
        return obj, 0

    if isinstance(obj, ir.TensorRegion):
        buffer, region = obj.source, obj.region
        strides = retrieve_stride(obj)
        offset = 0
        for i, r in enumerate(region):
            offset += r.begin * strides[i]
        return buffer, offset

    if isinstance(obj, ir.TensorLoad):
        region = get_buffer_region_from_load(obj)
        if region is not None:
            return retrieve_buffer_and_offset(region)

        buffer = obj.source
        strides = retrieve_stride(obj)
        offset = 0
        for i, idx in enumerate(obj.indices):
            offset += idx * strides[i]
        return buffer, offset

    raise ValueError(f"Unsupported retrieve_buffer_and_offset argument type: {type(obj)} for object {obj}")


def retrieve_offset(obj: BufferLikeType) -> list:
    """
    Retrieve per-dimension minima offsets.

    - Buffer -> [0, 0, ...]
    - BufferRegion -> [r.begin for r in region]
    - BufferLoad -> indices (or derived region minima)
    """
    if _tensor_spec.is_buffer_like(obj):
        return [0] * len(obj.shape)
    if isinstance(obj, ir.TensorRegion):
        return [r.begin for r in obj.region]
    if isinstance(obj, ir.TensorLoad):
        region = get_buffer_region_from_load(obj)
        if region is not None:
            return [r.begin for r in region.region]
        return list(obj.indices)
    raise ValueError(f"Unsupported retrieve_offset argument type: {type(obj)} for object {obj}")


def retrieve_dtype(obj: BufferLikeType) -> str:
    """
    Retrieve the dtype of a buffer-like object.

    - Buffer -> buffer.dtype
    - BufferRegion -> convert to BufferLoad with Ramp indices, then use load.dtype
    - BufferLoad -> load.dtype
    """
    if _tensor_spec.is_buffer_like(obj):
        return _construction.dtype_of(obj)
    if isinstance(obj, ir.TensorRegion):
        # Convert region ranges to indices, using Ramp for vector access
        indices = []
        for r in obj.region:
            extent = r.extent
            if isinstance(extent, ir.IntImm) and extent.value == 1:
                indices.append(r.begin)
            else:
                # Use Ramp for vector access: Ramp(base, stride=1, lanes=extent)
                indices.append(_prim.Ramp(r.begin, 1, extent))
        load = ir.TensorLoad(obj.source, _construction.as_indices(indices), _construction.current_location())
        return _construction.dtype_of(load)
    if isinstance(obj, ir.TensorLoad):
        return _construction.dtype_of(obj)
    raise ValueError(f"Unsupported retrieve_dtype argument type: {type(obj)} for object {obj}")


def bits_product(shape: list[PrimExpr], dtype: str) -> PrimExpr:
    """
    Compute the number of bits in a Buffer (shape with dtype).

    For vector types (e.g. ``bfloat16x2``) ``DataType.bits`` returns the
    per-lane width, not the full element width.  Multiply by ``lanes`` so
    that the total bit count is correct.
    """
    result = ir.IntImm("int32", 1) if len(shape) == 0 else shape[0]
    for i in range(1, len(shape)):
        result = result * shape[i]
    dt = DataType(dtype)
    return result * dt.bits * dt.lanes


def _strip_source_call_annotations(expression):
    """Build a temporary view for the source language's weak Call equality.

    The old shape comparison ignores Call.annotations. Native attrs may carry
    semantics, so only identified source annotations are removed here.
    Keep variables (including tensor types) by identity and never change the
    input IR or the compiler's structural equality/hash rules.
    """

    def visit_call(call, mutator):
        attributes = call.attrs
        if attributes is None:
            return mutator.default_mutate(call)
        source_keys = attributes.get("tl.source_annotation_keys")
        if source_keys is not None:
            if not isinstance(source_keys, tvm_ffi.Array) or any(not isinstance(key, str) or key not in attributes for key in source_keys):
                return mutator.default_mutate(call)
            attributes = {key: value for key, value in attributes.items() if key not in {*source_keys, "tl.source_annotation_keys"}}
            return ir.Call(
                call.op,
                [mutator.mutate(argument) for argument in call.args],
                attrs=attributes or None,
                ty_args=list(call.ty_args),
                ty=call.ty,
                loc=call.loc,
            )
        if attributes.get("schema_version") != 1:
            return mutator.default_mutate(call)
        count = attributes.get("operand_count")
        schema = attributes.get("annotation_schema")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or count + 1 != len(call.args)
            or not isinstance(schema, tvm_ffi.Array)
            or not isinstance(call.args[-1], ir.Tuple)
        ):
            return mutator.default_mutate(call)
        try:
            decoded = tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[-1], schema)
        except (TypeError, ValueError):
            return mutator.default_mutate(call)
        if not isinstance(decoded, tvm_ffi.Map):
            return mutator.default_mutate(call)
        attributes = {
            key: value for key, value in attributes.items() if key not in {"schema_version", "operand_count", "annotation_schema"}
        }
        return ir.Call(
            call.op,
            [mutator.mutate(argument) for argument in call.args[:count]],
            attrs=attributes or None,
            ty_args=list(call.ty_args),
            ty=call.ty,
            loc=call.loc,
        )

    return tvm_ffi.structural_mutate(expression, [(ir.Var, lambda variable, _: variable), (ir.Call, visit_call)])


def prim_expr_equal(lhs, rhs) -> bool:
    """
    Robust equality for PrimExpr shapes/extents.

    Tries structural_equal first, then falls back to expr_deep_equal and the
    source Call-annotation-insensitive comparison used by shape consumers.
    Python ints are converted to IntImm for comparison.
    """
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs == rhs
    if isinstance(lhs, int):
        lhs = ir.IntImm("int32", lhs)
    if isinstance(rhs, int):
        rhs = ir.IntImm("int32", rhs)
    if tvm_ffi.structural_equal(lhs, rhs):
        return True
    if ir.prim.expr_deep_equal(lhs, rhs):
        return True
    return ir.prim.expr_deep_equal(_strip_source_call_annotations(lhs), _strip_source_call_annotations(rhs))


def legalize_pairwise_extents(src_extents: list, dst_extents: list) -> tuple[list, list]:
    """
    Right-align and broadcast two extent lists to be mutually compatible.

    Early-exit rule:
    - If the number of non-1 dimensions in `src_extents` equals that in `dst_extents`,
      no adjustment is made; the original extents are returned unchanged. This
      preserves the per-dimension iteration mapping (one loop var per non-1 dim)
      and avoids creating extra varying axes on either side.

    Otherwise, for each pair of tail-aligned dimensions (x, y):
      - if x == y: keep both
      - elif x == 1: set x = y
      - elif y == 1: set y = x
      - else: promote both to tirx.max(x, y) to handle dynamic-vs-static safely

    Leading unmatched dimensions are kept as-is.

    Returns a tuple of new lists (src_new, dst_new).
    """
    a = list(src_extents)
    b = list(dst_extents)

    # If both sides have the same number of non-1 extents, don't re-broadcast.
    def _num_non_one(exts: list) -> int:
        return sum(0 if prim_expr_equal(x, 1) else 1 for x in exts)

    if _num_non_one(a) == _num_non_one(b):
        return a, b
    k = min(len(a), len(b))
    for i in range(1, k + 1):
        x, y = a[-i], b[-i]
        if prim_expr_equal(x, y):
            continue
        elif prim_expr_equal(x, 1):
            a[-i] = y
        elif prim_expr_equal(y, 1):
            b[-i] = x
        else:
            # Dynamic mismatch: promote to max so downstream clamping/predicates remain safe
            m = _prim.max(x, y)
            a[-i] = m
            b[-i] = m
    return a, b


def is_full_region(buffer_region: BufferRegion) -> bool:
    """
    Check whether a BufferRegion covers the full buffer region.

    A full region means each dimension has start 0 and extent equal to
    the corresponding dimension in the buffer's shape.

    Args:
        buffer_region: The TVM BufferRegion to check.

    Returns:
        bool: True if the region is full; otherwise False.
    """
    if not isinstance(buffer_region, ir.TensorRegion):
        raise TypeError(f"Expected BufferRegion, got {type(buffer_region)}")

    buf = buffer_region.source
    ranges = buffer_region.region

    if len(buf.shape) != len(ranges):
        return False

    expr_equal = prim_expr_equal
    for dim, r in zip(buf.shape, ranges):
        # start == 0 and extent == shape
        if not expr_equal(r.begin, 0):
            return False
        if not expr_equal(r.extent, dim):
            return False
    return True


def get_prim_func_name(func: PrimFunc | None, default: str | None = None) -> str | None:
    """
    Extract a human‑readable function name from a TVM PrimFunc.

    Prefer the `global_symbol` attribute set on the PrimFunc. If it is missing
    (e.g., private PrimFunc without a global symbol), return the provided
    `default` value.

    Args:
        func: TVM PrimFunc instance or None.
        default: Fallback name to return when no name can be determined.

    Returns:
        The function name as a string, or `default` when unavailable.
    """
    if func is None:
        return default
    try:
        name = func.attrs["global_symbol"]
        return str(name) if name is not None else default
    except Exception:
        return default


def side_effect(expr: PrimExpr) -> CallEffectKind:
    from tilelang import _ffi_api

    return _ffi_api.SideEffect(expr)
