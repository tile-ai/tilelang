"""Reduce operations exposed on the TileLang language surface."""

from __future__ import annotations
from tvm import tir
from tilelang.language import copy, macro, alloc_shared, alloc_fragment
from tilelang.utils.language import to_buffer_region, retrieve_shape, _get_buffer
from tilelang.utils.language import is_shared, is_fragment
from tvm.script.ir_builder import IRBuilder


def _legalize_dim(buffer: tir.Buffer, dim: int):
    if dim < 0:
        dim = len(buffer.shape) + dim
    return dim


_REDUCE_OP_KEY = "tl.tileop.reduce"


def reduce(buffer: tir.Buffer, out: tir.Buffer, reduce_type: str, dim: int, clear: bool):
    """Perform a reduction operation on a buffer along a specified dimension.

    Args:
        buffer (tir.Buffer): Input buffer to reduce
        out (tir.Buffer): Output buffer to store results
        reduce_type (str): Type of reduction ('max', 'min', 'sum', 'abssum')
        dim (int): Dimension along which to perform reduction
        clear (bool): Whether to initialize the output buffer before reduction

    Returns:
        tir.Call: Handle to the reduction operation
    """
    # input shape: [X, d, Y], expected output shape: [X, Y] or [X, 1, Y]
    expected_shapes = [buffer.shape[:dim] + buffer.shape[dim + 1 :], buffer.shape[:dim] + [1] + buffer.shape[dim + 1 :]]
    if list(out.shape) not in expected_shapes:
        expected_shapes_str = " or ".join(map(str, expected_shapes))
        raise ValueError(
            f"Invalid reduce output shape, buffer shape is {buffer.shape}, dim is {dim}, "
            f"output shape is {out.shape}, expected shapes are {expected_shapes_str}"
        )

    @macro
    def reduce_macro(buffer: tir.Buffer, out: tir.Buffer, reduce_type: str, dim: int, clear: bool):
        if is_shared(buffer) and is_shared(out):
            red_frag_in = alloc_fragment(buffer.shape, buffer.dtype)
            red_frag_out = alloc_fragment(out.shape, out.dtype)

            # rename buffers
            IRBuilder.name(buffer.name + "_frag", red_frag_in)
            IRBuilder.name(out.name + "_frag", red_frag_out)

            copy(buffer, red_frag_in)
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                to_buffer_region(red_frag_in, access_type="r"),
                to_buffer_region(red_frag_out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
            copy(red_frag_out, out)
        elif is_shared(buffer) and is_fragment(out):
            red_frag_in = alloc_fragment(buffer.shape, buffer.dtype)
            IRBuilder.name(buffer.name + "_frag", red_frag_in)

            copy(buffer, red_frag_in)
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                to_buffer_region(red_frag_in, access_type="r"),
                to_buffer_region(out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
        elif is_fragment(buffer) and is_shared(out):
            red_frag_out = alloc_fragment(out.shape, out.dtype)
            IRBuilder.name(out.name + "_frag", red_frag_out)

            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                to_buffer_region(buffer, access_type="r"),
                to_buffer_region(red_frag_out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
            copy(red_frag_out, out)
        elif is_fragment(buffer) and is_fragment(out):
            tir.call_intrin(
                "handle",
                tir.op.Op.get(_REDUCE_OP_KEY),
                to_buffer_region(buffer, access_type="r"),
                to_buffer_region(out, access_type="w"),
                reduce_type,
                dim,
                clear,
            )
        else:
            raise ValueError(f"Invalid buffer scopes: {buffer.scope()} and {out.scope()}")

    return reduce_macro(buffer, out, reduce_type, dim, clear)


def reduce_max(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
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
    return reduce(buffer, out, "max", dim, clear)


def reduce_min(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
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
    return reduce(buffer, out, "min", dim, clear)


def reduce_sum(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
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
    return reduce(buffer, out, "sum", dim, clear)


def reduce_abssum(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1):
    """Perform reduce absolute sum on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return reduce(buffer, out, "abssum", dim, True)


def reduce_absmax(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce absolute max on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return reduce(buffer, out, "absmax", dim, clear)


def reduce_bitand(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce bitwise-and on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return reduce(buffer, out, "bitand", dim, clear)


def reduce_bitor(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce bitwise-or on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return reduce(buffer, out, "bitor", dim, clear)


def reduce_bitxor(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce bitwise-xor on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return reduce(buffer, out, "bitxor", dim, clear)


@macro
def cumsum_fragment(
    src: tir.Buffer,
    dst: tir.Buffer,
    dim: int,
    reverse: bool,
) -> tir.PrimExpr:
    """
    Compute cumulative sum for fragment buffers by copying to shared memory first.

    This macro handles cumulative sum operations on fragment buffers by first copying
    the data to shared memory, performing the cumsum operation, and then copying back.

    Args:
        src: Source buffer (Buffer, BufferRegion, or BufferLoad) containing input data.
        dst: Destination buffer (Buffer, BufferRegion, or BufferLoad) for output data.
        dim: Dimension along which to compute cumulative sum.
        reverse: If True, compute cumulative sum in reverse order.

    Returns:
        tir.PrimExpr: A handle to the cumulative sum operation.
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


def cumsum(
    src: tir.Buffer | tir.BufferRegion | tir.BufferLoad,
    dst: tir.Buffer | tir.BufferRegion | tir.BufferLoad | None = None,
    dim: int = 0,
    reverse: bool = False,
):
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
        return cumsum_fragment(src, dst, dim, reverse)
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.cumsum"),
        to_buffer_region(src, access_type="r"),
        to_buffer_region(dst, access_type="w"),
        dim,
        reverse,
    )


def finalize_reducer(reducer: tir.Buffer):
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


def warp_reduce_sum(value: tir.PrimExpr):
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


def warp_reduce_max(value: tir.PrimExpr):
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


def warp_reduce_min(value: tir.PrimExpr):
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


def warp_reduce_bitand(value: tir.PrimExpr):
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


def warp_reduce_bitor(value: tir.PrimExpr):
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
