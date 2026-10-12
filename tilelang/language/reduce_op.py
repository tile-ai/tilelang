"""Reduce operations exposed on the TileLang language surface."""

from __future__ import annotations

import warnings
from typing import Literal
from tilelang._typing import BufferLikeType
from tvm import tir
from tvm.target import Target
from tilelang.language import copy, macro, alloc_shared, alloc_fragment
from tilelang.utils.language import to_buffer_region, retrieve_shape, _get_buffer
from tilelang.utils.language import prim_expr_equal
from tilelang.utils.language import is_shared, is_fragment
from tilelang.utils.target import target_is_sunmmio
from tvm.script.ir_builder import IRBuilder


def _legalize_dim_for_rank(rank: int, dim: int):
    if dim < 0:
        dim = rank + dim
    if dim < 0 or dim >= rank:
        raise ValueError(f"Reduction axis {dim} is out of bounds for rank {rank}")
    return dim


def _legalize_dim(buffer: BufferLikeType, dim: int):
    return _legalize_dim_for_rank(len(retrieve_shape(buffer)), dim)


def _shape_matches(actual, expected):
    if len(actual) != len(expected):
        return False
    return all(prim_expr_equal(lhs, rhs) for lhs, rhs in zip(actual, expected))


def _as_reduce_region_arg(obj: BufferLikeType, access_type: str):
    return to_buffer_region(obj, access_type=access_type, extents=list(retrieve_shape(obj)))


_REDUCE_OP_KEY = "tl.tileop.reduce"

ReduceKind = Literal["sum", "abssum", "max", "absmax", "min", "bitand", "bitor", "bitxor"]


# NOTE(chaofan): T.reduce is implemented as a macro, so no return
def reduce(buffer: BufferLikeType, out: BufferLikeType, reduce_type: ReduceKind, dim: int, clear: bool) -> None:
    """Perform a reduction operation on a buffer along a specified dimension.

    Args:
        buffer: Input buffer or buffer region to reduce
        out: Output buffer or buffer region to store results
        reduce_type (str): Type of reduction ('max', 'min', 'sum', 'abssum')
        dim (int): Dimension along which to perform reduction
        clear (bool): Whether to initialize the output buffer before reduction
    """
    buffer_shape = list(retrieve_shape(buffer))
    out_shape = list(retrieve_shape(out))
    dim = _legalize_dim_for_rank(len(buffer_shape), dim)

    # input shape: [X, d, Y], expected output shape: [X, Y] or [X, 1, Y]
    expected_shapes = [
        buffer_shape[:dim] + buffer_shape[dim + 1 :],
        buffer_shape[:dim] + [tir.IntImm("int32", 1)] + buffer_shape[dim + 1 :],
    ]
    if not any(_shape_matches(out_shape, expected_shape) for expected_shape in expected_shapes):
        expected_shapes_str = " or ".join(map(str, expected_shapes))
        raise ValueError(
            f"Invalid reduce output shape, buffer shape is {buffer_shape}, dim is {dim}, "
            f"output shape is {out_shape}, expected shapes are {expected_shapes_str}"
        )

    @macro
    def reduce_macro(buffer: BufferLikeType, out: BufferLikeType, reduce_type: str, dim: int, clear: bool) -> None:
        target = Target.current()
        # Sunmmio uses direct builtins for ReduceOp in LowerTileOp
        # Check for Sunmmio target or specific Sunmmio shared memory scopes
        src_buffer = _get_buffer(buffer)
        dst_buffer = _get_buffer(out)
        is_sunmmio_scope = any(
            scope in (src_buffer.scope(), dst_buffer.scope()) for scope in ("shared.rsram", "shared.asram", "shared.wsram")
        )
        is_sunmmio_target = target is not None and target_is_sunmmio(target)
        if is_sunmmio_target or is_sunmmio_scope:
            if len(buffer_shape) == 2 and dim == 1 and len(out_shape) == 2:
                warnings.warn(
                    "Sunmmio 2D reduction with a trailing unit-dimension output (N, 1) "
                    "may cause layout issues and prevent the kernel from compiling; "
                    "prefer a rank-1 output (N,). "
                    f"Got input shape {buffer_shape}, dim={dim}, output shape {out_shape}.",
                    UserWarning,
                    stacklevel=5,
                )
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                _as_reduce_region_arg(buffer, access_type="r"),
                _as_reduce_region_arg(out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
            return

        if is_shared(buffer) and is_shared(out):
            red_frag_in = alloc_fragment(buffer_shape, src_buffer.dtype)
            red_frag_out = alloc_fragment(out_shape, dst_buffer.dtype)

            # rename buffers
            IRBuilder.name(src_buffer.name + "_frag", red_frag_in)
            IRBuilder.name(dst_buffer.name + "_frag", red_frag_out)

            if not clear:
                copy(out, red_frag_out)

            copy(buffer, red_frag_in)
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                _as_reduce_region_arg(red_frag_in, access_type="r"),
                _as_reduce_region_arg(red_frag_out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
            copy(red_frag_out, out)
        elif is_shared(buffer) and is_fragment(out):
            red_frag_in = alloc_fragment(buffer_shape, src_buffer.dtype)
            IRBuilder.name(src_buffer.name + "_frag", red_frag_in)

            copy(buffer, red_frag_in)
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                _as_reduce_region_arg(red_frag_in, access_type="r"),
                _as_reduce_region_arg(out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
        elif is_fragment(buffer) and is_shared(out):
            red_frag_out = alloc_fragment(out_shape, dst_buffer.dtype)
            IRBuilder.name(dst_buffer.name + "_frag", red_frag_out)

            if not clear:
                copy(out, red_frag_out)

            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                _as_reduce_region_arg(buffer, access_type="r"),
                _as_reduce_region_arg(red_frag_out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
            copy(red_frag_out, out)
        elif is_fragment(buffer) and is_fragment(out):
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                _as_reduce_region_arg(buffer, access_type="r"),
                _as_reduce_region_arg(out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
        else:
            raise ValueError(f"Invalid buffer scopes: {src_buffer.scope()} and {dst_buffer.scope()}")

    reduce_macro(buffer, out, reduce_type, dim, clear)


def reduce_max(buffer: BufferLikeType, out: BufferLikeType, dim: int = -1, clear: bool = True) -> None:
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
    Returns
    -------
    handle : PrimExpr
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "max", dim, clear)


def reduce_min(buffer: BufferLikeType, out: BufferLikeType, dim: int = -1, clear: bool = True) -> None:
    """Perform reduce min on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        clear (bool, optional): If True, output buffer will be initialized to inf. Defaults to True.

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "min", dim, clear)


def reduce_sum(buffer: BufferLikeType, out: BufferLikeType, dim: int = -1, clear: bool = True) -> None:
    """Perform reduce sum on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        clear (bool, optional): If True, output buffer will be cleared before reduction.
                              If False, results will be accumulated on existing values.
                              Defaults to True.
    Note: When clear=True, reduce_sum will not compute directly on the output buffer. This is because
          during warp reduction, the same value would be accumulated multiple times (number of threads
          in the warp). Therefore, the implementation with clear=True follows these steps:
        1. create a temp buffer with same shape and dtype as out
        2. copy out to temp buffer
        3. call reduce_sum with temp buffer and out
        4. Add temp buffer to out

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "sum", dim, clear)


def reduce_abssum(buffer: BufferLikeType, out: BufferLikeType, dim: int = -1, clear: bool = True) -> None:
    """Perform reduce absolute sum on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "abssum", dim, clear)


def reduce_absmax(buffer: BufferLikeType, out: BufferLikeType, dim: int = -1, clear: bool = True) -> None:
    """Perform reduce absolute max on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "absmax", dim, clear)


def reduce_bitand(buffer: BufferLikeType, out: BufferLikeType, dim: int = -1, clear: bool = True) -> None:
    """Perform reduce bitwise-and on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "bitand", dim, clear)


def reduce_bitor(buffer: BufferLikeType, out: BufferLikeType, dim: int = -1, clear: bool = True) -> None:
    """Perform reduce bitwise-or on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "bitor", dim, clear)


def reduce_bitxor(buffer: BufferLikeType, out: BufferLikeType, dim: int = -1, clear: bool = True) -> None:
    """Perform reduce bitwise-xor on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    reduce(buffer, out, "bitxor", dim, clear)


@macro
def cumsum_fragment(
    src: BufferLikeType,
    dst: BufferLikeType,
    dim: int,
    reverse: bool,
) -> None:
    """
    Compute cumulative sum for fragment buffers by copying to shared memory first.

    This macro handles cumulative sum operations on fragment buffers by first copying
    the data to shared memory, performing the cumsum operation, and then copying back.

    Args:
        src: Source buffer (Buffer, BufferRegion, or BufferLoad) containing input data.
        dst: Destination buffer (Buffer, BufferRegion, or BufferLoad) for output data.
        dim: Dimension along which to compute cumulative sum.
        reverse: If True, compute cumulative sum in reverse order.
    """
    src_shape = retrieve_shape(src)
    src_buffer = _get_buffer(src)
    # Get dtype from the buffer
    if isinstance(src, tir.Buffer):
        dtype = src.dtype
    else:
        dtype = src_buffer.dtype
    cumsum_smem = alloc_shared(src_shape, dtype, "shared.dyn")
    copy(src, cumsum_smem)
    tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.cumsum"),
        to_buffer_region(cumsum_smem, access_type="r"),
        to_buffer_region(cumsum_smem, access_type="w"),
        dim,
        reverse,
    )
    copy(cumsum_smem, dst)


# NOTE(chaofan): T.cumsum returns None if it goes to macro implementations
def cumsum(
    src: BufferLikeType,
    dst: BufferLikeType | None = None,
    dim: int = 0,
    reverse: bool = False,
) -> tir.PrimExpr | None:
    """
    Compute the cumulative sum of `src` along `dim`, writing results to `dst`.

    Negative `dim` indices are normalized (Python-style). If `dst` is None, the operation is performed in-place into `src`. Raises ValueError when `dim` is out of bounds for `src.shape`. When `src.scope() == "local.fragment"`, this delegates to `cumsum_fragment`; otherwise it emits the `tl.cumsum` intrinsic.

    Supports Buffer, BufferRegion, and BufferLoad inputs, allowing operations on buffer slices/regions.

    Examples:
        A 1D inclusive scan that writes the result into a separate shared-memory buffer:

        >>> import tilelang.language as T
        >>> @T.prim_func
        ... def kernel(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        ...     with T.Kernel(1, threads=128):
        ...         A_shared = T.alloc_shared((128,), "float32")
        ...         T.copy(A, A_shared)
        ...         T.cumsum(src=A_shared, dst=A_shared, dim=0)
        ...         T.copy(A_shared, B)

        A 2D prefix sum along the last dimension with reverse accumulation:

        >>> import tilelang.language as T
        >>> @T.prim_func
        ... def kernel2d(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 64), "float16")):
        ...     with T.Kernel(1, 1, threads=256):
        ...         tile = T.alloc_shared((64, 64), "float16")
        ...         T.copy(A, tile)
        ...         T.cumsum(src=tile, dim=1, reverse=True)
        ...         T.copy(tile, B)

        Operating on a buffer region (slice):

        >>> import tilelang.language as T
        >>> @T.prim_func
        ... def kernel_region(InputG_fragment: T.Tensor((128,), "float32"), chunk_size: T.int32):
        ...     with T.Kernel(1, threads=128):
        ...         i = T.int32(0)
        ...         T.cumsum(InputG_fragment[i * chunk_size:(i + 1) * chunk_size], dim=0)

    Returns:
        tir.Call: A handle to the emitted cumulative-sum operation.
    """

    # Get shape from src (supports Buffer, BufferRegion, BufferLoad)
    shape = retrieve_shape(src)
    if dim >= len(shape) or dim < -len(shape):
        raise ValueError(f"Dimension {dim} is out of bounds for buffer with shape {shape}")
    if dim < 0:
        dim = len(shape) + dim

    if dst is None:
        dst = src
    else:
        # Validate that dst shape matches src shape
        dst_shape = retrieve_shape(dst)
        if len(dst_shape) != len(shape):
            raise ValueError(f"cumsum dst shape {dst_shape} must match src shape {shape} (rank mismatch)")
        # Check each dimension matches
        for i in range(len(shape)):
            if not tir.analysis.expr_deep_equal(dst_shape[i], shape[i]):
                raise ValueError(f"cumsum dst shape {dst_shape} must match src shape {shape} (dim {i} mismatch)")

    # Check if src is a fragment buffer
    if is_fragment(src):
        cumsum_fragment(src, dst, dim, reverse)
        return

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.cumsum"),
        to_buffer_region(src, access_type="r"),
        to_buffer_region(dst, access_type="w"),
        dim,
        reverse,
    )


def finalize_reducer(reducer: tir.Buffer) -> tir.PrimExpr:
    """
    Finalize a reducer buffer by emitting the `tl.tileop.finalize_reducer` intrinsic.

    This returns a TVM `tir.Call` handle that finalizes the given reducer using its writable pointer.
    The call does not modify Python objects directly; it produces the low-level intrinsic call used by the IR.

    Parameters:
        reducer (tir.Buffer): Reducer buffer whose writable pointer will be finalized.

    Returns:
        tir.Call: Handle to the finalize reducer intrinsic call.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.finalize_reducer"),
        to_buffer_region(reducer, access_type="w"),
    )


def warp_reduce_sum(value: tir.PrimExpr) -> tir.PrimExpr:
    """Perform warp reduction sum on a register value.

    This function reduces a value across all threads in a warp using shuffle operations.
    Each thread provides a  register `value`, and after the reduction, all threads
    will have the sum of all values across the warp.

    Args:
        value (tir.PrimExpr): The input register value to reduce

    Returns:
        tir.PrimExpr: The reduced sum value (same on all threads in the warp)
    """
    return tir.call_intrin(value.dtype, tir.op.Op.get("tl.warp_reduce_sum"), value)


def warp_reduce_max(value: tir.PrimExpr) -> tir.PrimExpr:
    """Perform warp reduction max on a register value.

    This function reduces a value across all threads in a warp using shuffle operations.
    Each thread provides a  register `value`, and after the reduction, all threads
    will have the max of all values across the warp.

    Args:
        value (tir.PrimExpr): The input register value to reduce

    Returns:
        tir.PrimExpr: The reduced max value (same on all threads in the warp)
    """
    return tir.call_intrin(value.dtype, tir.op.Op.get("tl.warp_reduce_max"), value)


def warp_reduce_min(value: tir.PrimExpr) -> tir.PrimExpr:
    """Perform warp reduction min on a register value.

    This function reduces a value across all threads in a warp using shuffle operations.
    Each thread provides a  register `value`, and after the reduction, all threads
    will have the min of all values across the warp.

    Args:
        value (tir.PrimExpr): The input register value to reduce

    Returns:
        tir.PrimExpr: The reduced min value (same on all threads in the warp)
    """
    return tir.call_intrin(value.dtype, tir.op.Op.get("tl.warp_reduce_min"), value)


def warp_reduce_bitand(value: tir.PrimExpr) -> tir.PrimExpr:
    """Perform warp reduction bitwise-and on a register value.

    This function reduces a value across all threads in a warp using shuffle operations.
    Each thread provides a  register `value`, and after the reduction, all threads
    will have the bitwise-and of all values across the warp.

    Args:
        value (tir.PrimExpr): The input register value to reduce

    Returns:
        tir.PrimExpr: The reduced bitwise-and value (same on all threads in the warp)
    """
    return tir.call_intrin(value.dtype, tir.op.Op.get("tl.warp_reduce_bitand"), value)


def warp_reduce_bitor(value: tir.PrimExpr) -> tir.PrimExpr:
    """Perform warp reduction bitwise-or on a register value.

    This function reduces a value across all threads in a warp using shuffle operations.
    Each thread provides a  register `value`, and after the reduction, all threads
    will have the bitwise-or of all values across the warp.

    Args:
        value (tir.PrimExpr): The input register value to reduce

    Returns:
        tir.PrimExpr: The reduced bitwise-or value (same on all threads in the warp)
    """
    return tir.call_intrin(value.dtype, tir.op.Op.get("tl.warp_reduce_bitor"), value)
