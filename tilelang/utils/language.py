from __future__ import annotations
from tvm.tir import Buffer, BufferLoad, BufferRegion, PrimExpr
from tilelang.language.utils import region as _make_region_call
from functools import reduce
from tvm import IRModule, DataType
from tvm.tir import PrimFunc
from tvm import ir, tir
# Scope Checkers for TVM Buffers
# These utility functions check the memory scope of a given TVM buffer.


def _get_buffer(buffer_or_load_or_region: Buffer | BufferLoad | BufferRegion) -> Buffer:
    """
    Extract Buffer from Buffer, BufferLoad, or BufferRegion.

    Args:
        buffer_or_load_or_region: Can be Buffer, BufferLoad, or BufferRegion

    Returns:
        Buffer: The underlying buffer object
    """
    if isinstance(buffer_or_load_or_region, Buffer):
        return buffer_or_load_or_region
    elif isinstance(buffer_or_load_or_region, (tir.BufferLoad, tir.BufferRegion)):
        return buffer_or_load_or_region.buffer
    else:
        raise TypeError(f"Expected Buffer, BufferLoad, or BufferRegion, got {type(buffer_or_load_or_region)}")


def is_global(buffer: Buffer | BufferLoad | BufferRegion) -> bool:
    """
    Check if the buffer is in the global memory scope.

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in global memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return buffer.scope() == "global"


def is_shared(buffer: Buffer | BufferLoad | BufferRegion, allow_dynamic: bool = True) -> bool:
    """
    Check if the buffer is in the shared memory scope.

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in shared memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    conditions = [False]
    conditions.append(buffer.scope() == "shared")
    if allow_dynamic:
        conditions.append(is_shared_dynamic(buffer))
    return any(conditions)


def is_shared_dynamic(buffer: Buffer | BufferLoad | BufferRegion) -> bool:
    """
    Check if the buffer is in the dynamic shared memory scope.

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in dynamic shared memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return buffer.scope() == "shared.dyn"


def is_tensor_memory(buffer: Buffer | BufferLoad | BufferRegion) -> bool:
    """
    Check if the buffer is in tensor memory scope (e.g., shared.tmem).

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in tensor memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return buffer.scope().startswith("shared.tmem")


def is_local(buffer: Buffer | BufferLoad | BufferRegion) -> bool:
    """
    Check if the buffer is in the local memory scope.

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is in local memory, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return buffer.scope() == "local"


def is_fragment(buffer: Buffer | BufferLoad | BufferRegion) -> bool:
    """
    Check if the buffer is a fragment (e.g., for matrix multiplication operations).

    Args:
        buffer: The TVM buffer, BufferLoad, or BufferRegion to check.

    Returns:
        bool: True if the buffer is a fragment, False otherwise.
    """
    buffer = _get_buffer(buffer)
    return buffer.scope().startswith("local.fragment")


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


def get_buffer_region_from_load(buffer_load: tir.BufferLoad, extents: list[PrimExpr] | None = None) -> tir.BufferRegion | None:
    """
    Get the buffer region from a buffer load.

    May encounter buffer load like C[0:128, 0:32], ref to pull request
    for buffer wise op: https://github.com/apache/tvm/pull/14693
    convert load to region
    """
    buffer, indices = buffer_load.buffer, buffer_load.indices
    regions = []
    found_ramp: bool = False

    if extents is not None:
        assert len(extents) == len(indices), "extents should have the same length as indices"
    for i, indice in enumerate(indices):
        if isinstance(indice, tir.Ramp):
            assert extents is None, "extents should be provided for BufferLoad with Ramp indices"
            regions.append(ir.Range.from_min_extent(indice.base, indice.lanes))
            found_ramp = True
        elif isinstance(indice, tir.PrimExpr):
            if extents is not None:
                regions.append(ir.Range.from_min_extent(indice, extents[i]))
                found_ramp = True
            else:
                regions.append(ir.Range.from_min_extent(indice, 1))
        else:
            raise ValueError(f"Unsupported type: {type(indice)} for index {i}")
    if found_ramp:
        return tir.BufferRegion(buffer, regions)
    else:
        return None


def to_buffer_region(
    obj: Buffer | BufferLoad | BufferRegion | tir.Var, access_type: str = "rw", extents: list[PrimExpr] | None = None
) -> PrimExpr | BufferRegion:
    """
    Convert to/from the tl.region representation.

    - Buffer/BufferLoad/BufferRegion -> returns a tl.region call (PrimExpr)
    - tl.region Call -> returns the decoded BufferRegion for analysis
    """
    from tilelang.language.frame import has_let_value, get_let_value

    if isinstance(obj, tir.Var) and has_let_value(obj):
        obj = get_let_value(obj)
    # Encode into tl.region call (when extents is provided), otherwise return BufferRegion for analysis
    if isinstance(obj, tir.BufferRegion):
        if extents is None:
            return obj
        mins = [r.min for r in obj.region]
        exts = [r.extent for r in obj.region]
        assert len(extents) == len(exts)
        exts = [tir.min(exts[i], extents[i]) for i in range(len(exts))]
        return _make_region_call(tir.BufferLoad(obj.buffer, mins), access_type, *exts)
    if isinstance(obj, tir.Buffer):
        mins = [tir.IntImm("int32", 0) for _ in obj.shape]
        if extents is None:
            ranges = [ir.Range.from_min_extent(m, e) for m, e in zip(mins, obj.shape)]
            return tir.BufferRegion(obj, ranges)
        exts = list(extents)
        return _make_region_call(tir.BufferLoad(obj, mins), access_type, *exts)
    if isinstance(obj, tir.BufferLoad):
        if extents is None:
            region = get_buffer_region_from_load(obj)
            if region is not None:
                return region
            mins = [idx for idx in obj.indices]
            ones = [tir.IntImm("int32", 1) for _ in obj.indices]
            ranges = [ir.Range.from_min_extent(m, e) for m, e in zip(mins, ones)]
            return tir.BufferRegion(obj.buffer, ranges)
        exts = list(extents)
        if len(obj.indices) > len(exts):
            exts = [tir.IntImm("int32", 1) for _ in range(len(obj.indices) - len(exts))] + exts
        assert len(obj.indices) == len(exts)
        return _make_region_call(obj, access_type, *exts)
    raise ValueError(f"Unsupported argument type for to_buffer_region: {type(obj)}")


def retrieve_shape(obj: Buffer | BufferRegion | BufferLoad) -> list:
    """
    Retrieve shape-like extents for a buffer-like object.

    - Buffer -> its `shape`
    - BufferRegion -> list of each range's `extent`
    - BufferLoad -> extents from `get_buffer_region_from_load(obj)`
    """
    if isinstance(obj, tir.Buffer):
        return obj.shape
    if isinstance(obj, tir.BufferRegion):
        return [r.extent for r in obj.region]
    if isinstance(obj, tir.BufferLoad):
        region = get_buffer_region_from_load(obj)
        if region is None:
            raise ValueError("Cannot retrieve shape from scalar BufferLoad without region")
        return [r.extent for r in region.region]
    raise ValueError(f"Unsupported retrieve_shape argument type: {type(obj)} for object {obj}")


def retrieve_stride(obj: Buffer | BufferRegion | BufferLoad) -> list:
    """
    Retrieve row-major strides for a buffer-like object based on its buffer.shape.

    For BufferRegion and BufferLoad, uses the underlying buffer's `shape`.
    """
    if isinstance(obj, tir.Buffer):
        shape = obj.shape
    elif isinstance(obj, (tir.BufferRegion, tir.BufferLoad)):
        shape = obj.buffer.shape
    else:
        raise ValueError(f"Unsupported retrieve_stride argument type: {type(obj)} for object {obj}")

    strides = []
    stride = 1
    for s in reversed(shape):
        strides.insert(0, stride)
        stride *= s
    return strides


def retrive_ptr_from_buffer_region(buffer_or_load_or_region: Buffer | BufferLoad | BufferRegion, access_type: str = "r") -> PrimExpr:
    if isinstance(buffer_or_load_or_region, Buffer):
        return buffer_or_load_or_region.access_ptr(access_type)
    elif isinstance(buffer_or_load_or_region, BufferLoad):
        buffer_load = buffer_or_load_or_region
        offset, stride = 0, 1
        buffer = buffer_load.buffer
        for i, shape in enumerate(reversed(buffer.shape)):
            indice = buffer_load.indices[len(buffer_load.indices) - i - 1]
            if isinstance(indice, (tir.IntImm, tir.PrimExpr)):
                offset += indice * stride
            elif isinstance(indice, tir.Ramp):
                offset += indice.base * stride
            else:
                raise ValueError(f"Unsupported index type: {type(indice)}")
            stride *= shape
        return buffer.access_ptr(access_type, offset=offset)
    elif isinstance(buffer_or_load_or_region, BufferRegion):
        buffer_region = buffer_or_load_or_region
        buffer = buffer_region.buffer
        offset, stride = 0, 1
        for i, shape in enumerate(reversed(buffer.shape)):
            offset += buffer_region.region[len(buffer_region.region) - i - 1].min * stride
            stride *= shape
        return buffer.access_ptr(access_type, offset=offset)
    else:
        raise ValueError(f"Unsupported buffer type: {type(buffer_or_load_or_region)}")


def retrieve_ptr(
    obj: Buffer | BufferRegion | BufferLoad,
    access_type: str = "r",
    ignore_last_ndim: int = 0,
) -> PrimExpr:
    """
    Retrieve a pointer to the start of a (possibly sliced) buffer region.

    - Buffer -> base pointer
    - BufferRegion -> pointer with byte offset computed from region minima
    - BufferLoad -> pointer offset computed from indices or derived region

    Args:
        obj: Buffer-like object
        access_type: TVM Buffer access mask, e.g. "r", "w", "rw"
        ignore_last_ndim: do not offset the last N dimensions
    """
    if isinstance(obj, tir.Buffer):
        return obj.access_ptr(access_type)

    if isinstance(obj, tir.BufferRegion):
        buffer, region = obj.buffer, obj.region
        strides = retrieve_stride(obj)
        # offset only over the leading dims, optionally ignoring the tail dims
        upto = max(0, len(region) - int(ignore_last_ndim))
        offset = 0
        for i in range(upto):
            offset += region[i].min * strides[i]
        return buffer.access_ptr(access_type, offset=offset)

    if isinstance(obj, tir.BufferLoad):
        buffer = obj.buffer
        region = get_buffer_region_from_load(obj)
        if region is not None:
            mins = [r.min for r in region.region]
        else:
            mins = list(obj.indices)
        strides = retrieve_stride(obj)
        upto = max(0, len(mins) - int(ignore_last_ndim))
        offset = 0
        for i in range(upto):
            offset += mins[i] * strides[i]
        return buffer.access_ptr(access_type, offset=offset)

    raise ValueError(f"Unsupported retrieve_ptr argument type: {type(obj)} for object {obj}")


def retrieve_offset(obj: Buffer | BufferRegion | BufferLoad) -> list:
    """
    Retrieve per-dimension minima offsets.

    - Buffer -> [0, 0, ...]
    - BufferRegion -> [r.min for r in region]
    - BufferLoad -> indices (or derived region minima)
    """
    if isinstance(obj, tir.Buffer):
        return [0] * len(obj.shape)
    if isinstance(obj, tir.BufferRegion):
        return [r.min for r in obj.region]
    if isinstance(obj, tir.BufferLoad):
        region = get_buffer_region_from_load(obj)
        if region is not None:
            return [r.min for r in region.region]
        return list(obj.indices)
    raise ValueError(f"Unsupported retrieve_offset argument type: {type(obj)} for object {obj}")


def bits_product(shape: list[PrimExpr], dtype: str) -> PrimExpr:
    """
    Compute the number of bits in a Buffer (shape with dtype)."""
    if len(shape) == 0:
        return tir.IntImm("int32", 1)
    result = shape[0]
    for i in range(1, len(shape)):
        result = result * shape[i]
    return result * DataType(dtype).bits


def prim_expr_equal(lhs, rhs) -> bool:
    """
    Robust equality for PrimExpr shapes/extents.

    Tries structural_equal first, then falls back to expr_deep_equal.
    Python ints are converted to IntImm for comparison.
    """
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs == rhs
    if isinstance(lhs, int):
        lhs = tir.IntImm("int32", lhs)
    if isinstance(rhs, int):
        rhs = tir.IntImm("int32", rhs)
    if ir.structural_equal(lhs, rhs):
        return True
    return tir.analysis.expr_deep_equal(lhs, rhs)


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
      - else: promote both to tir.max(x, y) to handle dynamic-vs-static safely

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
            m = tir.max(x, y)
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
    if not isinstance(buffer_region, tir.BufferRegion):
        raise TypeError(f"Expected BufferRegion, got {type(buffer_region)}")

    buf = buffer_region.buffer
    ranges = buffer_region.region

    if len(buf.shape) != len(ranges):
        return False

    expr_equal = tir.analysis.expr_deep_equal
    for dim, r in zip(buf.shape, ranges):
        # start == 0 and extent == shape
        if not expr_equal(r.min, 0):
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
