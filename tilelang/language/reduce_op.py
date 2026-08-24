"""Reduce operations exposed on the TileLang language surface."""

from __future__ import annotations
from typing import Literal
from tvm import arith, tirx
from tilelang._typing import BufferLikeType
from tilelang.language.common import copy, macro, alloc_fragment, evaluate
from tilelang.utils.language import retrieve_shape, to_buffer_region, to_tile_region
from tilelang.utils.language import is_shared, is_fragment, is_local
from tvm.script.ir_builder import IRBuilder
from tilelang.language.utils import _normalize_annotations


def _validate_reduce_region(buffer: BufferLikeType, role: str) -> None:
    if not isinstance(buffer, tirx.BufferRegion):
        return
    if is_fragment(buffer):
        raise ValueError(
            f"T.reduce_* does not support slicing a fragment buffer; got a sliced {role}. "
            "Fragment replication is defined over its complete logical domain. "
            "To reduce selected elements, use T.alloc_reducer, "
            "call T.reducer_update inside T.Parallel, then call T.finalize_reducer."
        )
    if not is_shared(buffer):
        raise ValueError(
            f"T.reduce_* only supports BufferRegion arguments backed by shared memory; got a sliced {role} "
            f"in scope {buffer.buffer.scope()}."
        )


def _legalize_dim(buffer: BufferLikeType, dim: int):
    _validate_reduce_region(buffer, "input")
    if dim < 0:
        dim = len(retrieve_shape(buffer)) + dim
    return dim


def _shape_equal(lhs: list, rhs: list) -> bool:
    analyzer = arith.Analyzer()
    return len(lhs) == len(rhs) and all(analyzer.can_prove_equal(a, b) for a, b in zip(lhs, rhs))


_REDUCE_OP_KEY = "tl.tileop.reduce"

ReduceKind = Literal["sum", "abssum", "max", "absmax", "min", "bitand", "bitor", "bitxor"]


# NOTE(chaofan): T.reduce is implemented as a macro, so no return
def reduce(
    buffer: BufferLikeType,
    out: BufferLikeType,
    reduce_type: ReduceKind,
    dim: int,
    clear: bool,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> None:
    """Perform a reduction operation on a buffer along a specified dimension.

    Args:
        buffer (tirx.Buffer): Input buffer to reduce
        out (tirx.Buffer): Output buffer to store results
        reduce_type (str): Type of reduction ('max', 'min', 'sum', 'abssum')
        dim (int): Dimension along which to perform reduction
        clear (bool): Whether to initialize the output buffer before reduction
        batch (int): Number of output elements per batched AllReduce call
            (default 1 = scalar, current behaviour). When batch > 1 the
            compiler emits ceil(N/batch) batched AllReduce calls each sharing
            a single pair of barriers, reducing total barrier count by batch×.
            batch must evenly divide the per-thread output element count N.
        nan_propagate (bool): Only meaningful for max/min/absmax on
            float16/bfloat16. When True, lower to CUDA __hmax_nan/__hmin_nan so
            NaNs propagate through the reduction. When False (default), use
            __hmax/__hmin which return the non-NaN operand. CUDA-only.
        annotations (dict, optional): Additional lowering controls. On CUDA
            SM100+, FP32 sum/abssum reductions accept
            ``{"enable_fadd2": False}`` to keep the reducer scalar. Packed
            FP32x2 reduction remains enabled by default.

    Shared-memory BufferRegion operands are staged through fragments matching
    the region extents. Fragment BufferRegion operands are rejected because a
    fragment's replication policy applies to its complete logical domain.
    """
    _validate_reduce_region(buffer, "input")
    _validate_reduce_region(out, "output")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    buffer_region = to_buffer_region(buffer)
    out_region = to_buffer_region(out)
    buffer_shape = retrieve_shape(buffer_region)
    out_shape = retrieve_shape(out_region)
    src_buffer = buffer_region.buffer
    out_buffer = out_region.buffer
    if reduce_type in ("bitand", "bitor", "bitxor") and not (out_buffer.dtype.startswith(("int", "uint")) or out_buffer.dtype == "bool"):
        raise ValueError(f"reduce_{reduce_type} requires an integer/bool buffer, got dtype {out_buffer.dtype}")
    # input shape: [X, d, Y], expected output shape: [X, Y] or [X, 1, Y]
    expected_shapes = [
        buffer_shape[:dim] + buffer_shape[dim + 1 :],
        buffer_shape[:dim] + [1] + buffer_shape[dim + 1 :],
    ]
    if not any(_shape_equal(out_shape, expected) for expected in expected_shapes):
        expected_shapes_str = " or ".join(map(str, expected_shapes))
        raise ValueError(
            f"Invalid reduce output shape, buffer shape is {buffer_shape}, dim is {dim}, "
            f"output shape is {out_shape}, expected shapes are {expected_shapes_str}"
        )

    annotations = _normalize_annotations(annotations)
    if batch > 1:
        annotations["batch"] = batch
    if nan_propagate:
        annotations["nan_propagate"] = True

    # Emit local reductions before macro expansion so alloc_var retains its
    # underlying Buffer rather than becoming a scalar expression.
    if is_local(buffer) and out_buffer.scope() in ("local", "local.var"):
        return evaluate(
            tirx.call_intrin(
                "handle",
                tirx.op.Op.get(_REDUCE_OP_KEY),
                to_tile_region(buffer, access_type="r"),
                to_tile_region(out_buffer, access_type="w"),
                reduce_type,
                dim,
                clear,
                annotations=annotations,
            )
        )

    @macro
    def reduce_macro(buffer: BufferLikeType, out: BufferLikeType, reduce_type: str, dim: int, clear: bool) -> None:
        if is_shared(buffer) and is_shared(out):
            red_frag_in = alloc_fragment(buffer_shape, src_buffer.dtype)
            red_frag_out = alloc_fragment(out_shape, out_buffer.dtype)

            # rename buffers
            IRBuilder.name(src_buffer.name + "_frag", red_frag_in)
            IRBuilder.name(out_buffer.name + "_frag", red_frag_out)

            if not clear:
                copy(out, red_frag_out)

            copy(buffer, red_frag_in)
            tirx.call_intrin(
                "handle",
                tirx.op.Op.get(_REDUCE_OP_KEY),
                to_tile_region(red_frag_in, access_type="r"),
                to_tile_region(red_frag_out, access_type="w"),
                reduce_type,
                dim,
                clear,
                annotations=annotations,
            )
            copy(red_frag_out, out)
        elif is_shared(buffer) and is_fragment(out):
            red_frag_in = alloc_fragment(buffer_shape, src_buffer.dtype)
            IRBuilder.name(src_buffer.name + "_frag", red_frag_in)

            copy(buffer, red_frag_in)
            tirx.call_intrin(
                "handle",
                tirx.op.Op.get(_REDUCE_OP_KEY),
                to_tile_region(red_frag_in, access_type="r"),
                to_tile_region(out, access_type="w"),
                reduce_type,
                dim,
                clear,
                annotations=annotations,
            )
        elif is_fragment(buffer) and is_shared(out):
            red_frag_out = alloc_fragment(out_shape, out_buffer.dtype)
            IRBuilder.name(out_buffer.name + "_frag", red_frag_out)

            if not clear:
                copy(out, red_frag_out)

            tirx.call_intrin(
                "handle",
                tirx.op.Op.get(_REDUCE_OP_KEY),
                to_tile_region(buffer, access_type="r"),
                to_tile_region(red_frag_out, access_type="w"),
                reduce_type,
                dim,
                clear,
                annotations=annotations,
            )
            copy(red_frag_out, out)
        elif is_fragment(buffer) and is_fragment(out):
            tirx.call_intrin(
                "handle",
                tirx.op.Op.get(_REDUCE_OP_KEY),
                to_tile_region(buffer, access_type="r"),
                to_tile_region(out, access_type="w"),
                reduce_type,
                dim,
                clear,
                annotations=annotations,
            )
        else:
            raise ValueError(f"Invalid buffer scopes: {src_buffer.scope()} and {out_buffer.scope()}")

    reduce_macro(buffer, out, reduce_type, dim, clear)


def reduce_max(
    buffer: BufferLikeType,
    out: BufferLikeType,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> None:
    """Perform reduce max on input buffer, store the result to output buffer

    Parameters
    ----------
    buffer : Buffer
        The input buffer.
    out : Buffer
        The output buffer.
    dim : int
        The dimension to perform reduce on
    clear : bool
        If set to True, the output buffer will first be initialized to -inf.
    batch : int
        Number of output elements per batched AllReduce call (default 1).
    nan_propagate : bool
        For float16/bfloat16 only. When True, NaN inputs propagate through the
        reduction (CUDA __hmax_nan). When False (default), NaN inputs are
        ignored in favor of the other operand (CUDA __hmax). CUDA-only.
    Returns
    -------
    handle : PrimExpr
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "max", dim, clear, batch=batch, nan_propagate=nan_propagate, annotations=annotations)


def reduce_min(
    buffer: BufferLikeType,
    out: BufferLikeType,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> None:
    """Perform reduce min on input buffer, store the result to output buffer.

    Args:
        buffer (tirx.Buffer): The input buffer
        out (tirx.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        clear (bool, optional): If True, output buffer will be initialized to inf. Defaults to True.
        batch (int): Number of output elements per batched AllReduce call (default 1).
        nan_propagate (bool, optional): For float16/bfloat16 only. When True,
            NaN inputs propagate (CUDA __hmin_nan). When False (default), NaNs
            are ignored (CUDA __hmin). CUDA-only.

    Returns:
        tirx.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "min", dim, clear, batch=batch, nan_propagate=nan_propagate, annotations=annotations)


def reduce_sum(
    buffer: BufferLikeType,
    out: BufferLikeType,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    annotations: dict | None = None,
) -> None:
    """Perform reduce sum on input buffer, store the result to output buffer.

    Args:
        buffer (tirx.Buffer): The input buffer
        out (tirx.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        clear (bool, optional): If True, output buffer will be cleared before reduction.
                              If False, results will be accumulated on existing values.
                              Defaults to True.
        batch (int): Number of output elements per batched AllReduce call (default 1).
        annotations (dict, optional): On CUDA SM100+, set
            ``{"enable_fadd2": False}`` to disable packed FP32x2 accumulation
            for this reduction. It is enabled by default.
    Note: When clear=True, reduce_sum will not compute directly on the output buffer. This is because
          during warp reduction, the same value would be accumulated multiple times (number of threads
          in the warp). Therefore, the implementation with clear=True follows these steps:
        1. create a temp buffer with same shape and dtype as out
        2. copy out to temp buffer
        3. call reduce_sum with temp buffer and out
        4. Add temp buffer to out

    Returns:
        tirx.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "sum", dim, clear, batch=batch, annotations=annotations)


def reduce_abssum(
    buffer: BufferLikeType,
    out: BufferLikeType,
    dim: int = -1,
    batch: int = 1,
    annotations: dict | None = None,
) -> None:
    """Perform reduce absolute sum on input buffer, store the result to output buffer.

    Args:
        buffer (tirx.Buffer): The input buffer
        out (tirx.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        batch (int): Number of output elements per batched AllReduce call (default 1).
        annotations (dict, optional): On CUDA SM100+, set
            ``{"enable_fadd2": False}`` to disable packed FP32x2 accumulation
            for this reduction. It is enabled by default.

    Returns:
        tirx.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "abssum", dim, True, batch=batch, annotations=annotations)


def reduce_absmax(
    buffer: BufferLikeType,
    out: BufferLikeType,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> None:
    """Perform reduce absolute max on input buffer, store the result to output buffer.

    Args:
        buffer (tirx.Buffer): The input buffer
        out (tirx.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        batch (int): Number of output elements per batched AllReduce call (default 1).
        nan_propagate (bool, optional): For float16/bfloat16 only. When True,
            NaN inputs propagate (CUDA __hmax_nan). When False (default), NaNs
            are ignored. CUDA-only.

    Returns:
        tirx.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "absmax", dim, clear, batch=batch, nan_propagate=nan_propagate, annotations=annotations)


def reduce_bitand(
    buffer: BufferLikeType,
    out: BufferLikeType,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    annotations: dict | None = None,
) -> None:
    """Perform reduce bitwise-and on input buffer, store the result to output buffer.

    Args:
        buffer (tirx.Buffer): The input buffer
        out (tirx.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        batch (int): Number of output elements per batched AllReduce call (default 1).

    Returns:
        tirx.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "bitand", dim, clear, batch=batch, annotations=annotations)


def reduce_bitor(
    buffer: BufferLikeType,
    out: BufferLikeType,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    annotations: dict | None = None,
) -> None:
    """Perform reduce bitwise-or on input buffer, store the result to output buffer.

    Args:
        buffer (tirx.Buffer): The input buffer
        out (tirx.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        batch (int): Number of output elements per batched AllReduce call (default 1).

    Returns:
        tirx.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "bitor", dim, clear, batch=batch, annotations=annotations)


def reduce_bitxor(
    buffer: BufferLikeType,
    out: BufferLikeType,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    annotations: dict | None = None,
) -> None:
    """Perform reduce bitwise-xor on input buffer, store the result to output buffer.

    Args:
        buffer (tirx.Buffer): The input buffer
        out (tirx.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tirx.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "bitxor", dim, clear, batch=batch, annotations=annotations)


def reducer_init(reducer: tirx.Buffer, init=None) -> tirx.PrimExpr:
    """Open a reducer epoch, optionally with a logical starting value.

    Must appear exactly once per `T.alloc_reducer` allocation, before any
    `T.reducer_update`. The whole epoch (init, updates, finalize) may sit
    inside thread-uniform serial loops or conditionals — the epoch then
    reopens once per dynamic execution — but init and finalize must share
    the same enclosing loop/branch scope. Without `init`, the reduction
    starts from the combine identity (sum -> 0, max -> dtype lowest, min ->
    dtype highest, bitand -> all ones, bitor/bitxor -> 0).

    `init` is a LOGICAL starting value: the result is as if one extra
    contribution `init` were combined into every logical output, exactly
    once. It is not a physical fill — physical partials always start from
    the identity, and the compiler captures `init` at the init site and
    combines it once per logical output at finalize time, so physical
    replication can never multiply it and later writes to buffers the
    expression reads cannot change the epoch's starting value.

    Args:
        reducer (tirx.Buffer): Handle returned by `T.alloc_reducer`.
        init (PrimExpr | int | float | None): Optional logical starting
            value; converted to the reducer's dtype when given as a Python
            number.

    Returns:
        tirx.Call: Handle to the reducer_init intrinsic call.
    """
    args = [to_tile_region(reducer, access_type="w")]
    if init is not None:
        if isinstance(init, (int, float)):
            init = tirx.const(init, reducer.dtype)
        args.append(init)
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.reducer_init"),
        *args,
    )


def reducer_update(target: tirx.BufferLoad, value) -> tirx.PrimExpr:
    """Contribute `value` to one logical output of a reducer.

    `target` must be written as `acc[indices]` directly in the first argument
    position; it is an update-target descriptor, not a read of the reducer's
    current value. Each dynamic logical iteration of the enclosing
    `T.Parallel` loop contributes exactly once, regardless of how the loop is
    physically replicated over threads.

    Args:
        target (tirx.BufferLoad): `acc[indices]` selecting the logical output.
        value: Contribution expression (cast to the reducer dtype if needed).

    Returns:
        tirx.Call: Handle to the reducer_update intrinsic call.
    """
    if not isinstance(target, tirx.BufferLoad):
        raise ValueError(
            f"reducer_update expects `acc[indices]` as its first argument, got {type(target)}; the reducer cannot be read or aliased."
        )
    dtype = target.buffer.dtype
    if isinstance(value, (int, float)):
        value = tirx.const(value, dtype)
    elif isinstance(value, tirx.PrimExpr) and value.dtype != dtype:
        value = tirx.Cast(dtype, value)
    # A builtin intrinsic, not a tile op: the target rides as a plain
    # BufferLoad (an update descriptor keeping the multi-dim indices for the
    # planner), and the layout story belongs to the enclosing T.Parallel.
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.reducer_update"),
        target,
        value,
    )


def finalize_reducer(
    reducer: tirx.Buffer, dst: tirx.Buffer | None = None, batch: int = 1, annotations: dict | None = None
) -> tirx.PrimExpr:
    """Close a reducer epoch.

    v2 form (``dst`` given): complete the cross-participant communication the
    chosen physical plan requires, combine the optional ``T.reducer_init``
    starting value exactly once per logical output, and write the logical
    result into the independent destination fragment ``dst``. After this
    call the reducer handle is dead; read results from ``dst``.

    Legacy v1 form (``dst`` omitted): in-place finalize of a legacy
    ``alloc_reducer(replication=...)`` fragment reducer. Deprecated.

    Parameters:
        reducer (tirx.Buffer): Reducer handle.
        dst (tirx.Buffer | None): Destination fragment (v2). Same logical
            shape and dtype as the reducer.
        batch (int): Batched AllReduce width: the collective covers `batch`
            output elements per call, sharing one pair of barriers.

    Returns:
        tirx.Call: Handle to the finalize intrinsic call.
    """
    if batch < 1:
        raise ValueError(f"finalize_reducer: batch must be >= 1, got {batch}")
    annotations = _normalize_annotations(annotations)
    if batch > 1:
        annotations["batch"] = batch
    if dst is not None:
        return tirx.call_intrin(
            "handle",
            tirx.op.Op.get("tl.tileop.finalize_reducer_v2"),
            to_tile_region(reducer, access_type="rw"),
            to_tile_region(dst, access_type="w"),
            annotations=annotations,
        )
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.finalize_reducer"),
        to_tile_region(reducer, access_type="w"),
        annotations=annotations,
    )


def warp_reduce_sum(value: tirx.PrimExpr) -> tirx.PrimExpr:
    """Perform warp reduction sum on a register value.

    This function reduces a value across all threads in a warp using shuffle operations.
    Each thread provides a  register `value`, and after the reduction, all threads
    will have the sum of all values across the warp.

    Args:
        value (tirx.PrimExpr): The input register value to reduce

    Returns:
        tirx.PrimExpr: The reduced sum value (same on all threads in the warp)
    """
    return tirx.call_intrin(value.dtype, tirx.op.Op.get("tl.warp_reduce_sum"), value)


def warp_reduce_max(value: tirx.PrimExpr) -> tirx.PrimExpr:
    """Perform warp reduction max on a register value.

    This function reduces a value across all threads in a warp using shuffle operations.
    Each thread provides a  register `value`, and after the reduction, all threads
    will have the max of all values across the warp.

    Args:
        value (tirx.PrimExpr): The input register value to reduce

    Returns:
        tirx.PrimExpr: The reduced max value (same on all threads in the warp)
    """
    return tirx.call_intrin(value.dtype, tirx.op.Op.get("tl.warp_reduce_max"), value)


def warp_reduce_min(value: tirx.PrimExpr) -> tirx.PrimExpr:
    """Perform warp reduction min on a register value.

    This function reduces a value across all threads in a warp using shuffle operations.
    Each thread provides a  register `value`, and after the reduction, all threads
    will have the min of all values across the warp.

    Args:
        value (tirx.PrimExpr): The input register value to reduce

    Returns:
        tirx.PrimExpr: The reduced min value (same on all threads in the warp)
    """
    return tirx.call_intrin(value.dtype, tirx.op.Op.get("tl.warp_reduce_min"), value)


def warp_reduce_bitand(value: tirx.PrimExpr) -> tirx.PrimExpr:
    """Perform warp reduction bitwise-and on a register value.

    This function reduces a value across all threads in a warp using shuffle operations.
    Each thread provides a  register `value`, and after the reduction, all threads
    will have the bitwise-and of all values across the warp.

    Args:
        value (tirx.PrimExpr): The input register value to reduce

    Returns:
        tirx.PrimExpr: The reduced bitwise-and value (same on all threads in the warp)
    """
    return tirx.call_intrin(value.dtype, tirx.op.Op.get("tl.warp_reduce_bitand"), value)


def warp_reduce_bitor(value: tirx.PrimExpr) -> tirx.PrimExpr:
    """Perform warp reduction bitwise-or on a register value.

    This function reduces a value across all threads in a warp using shuffle operations.
    Each thread provides a  register `value`, and after the reduction, all threads
    will have the bitwise-or of all values across the warp.

    Args:
        value (tirx.PrimExpr): The input register value to reduce

    Returns:
        tirx.PrimExpr: The reduced bitwise-or value (same on all threads in the warp)
    """
    return tirx.call_intrin(value.dtype, tirx.op.Op.get("tl.warp_reduce_bitor"), value)
