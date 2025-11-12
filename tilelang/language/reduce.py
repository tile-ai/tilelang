"""The language interface for tl programs."""
from __future__ import annotations

from tvm import tir
from tilelang.language import copy, macro, alloc_shared, alloc_fragment
from tilelang.utils.language import is_shared, is_fragment
from tvm.script.ir_builder import IRBuilder


def _legalize_dim(buffer: tir.Buffer, dim: int):
    if dim < 0:
        dim = len(buffer.shape) + dim
    return dim


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
    expected_shapes = [
        buffer.shape[:dim] + buffer.shape[dim + 1:],
        buffer.shape[:dim] + [1] + buffer.shape[dim + 1:]
    ]
    if list(out.shape) not in expected_shapes:
        expected_shapes_str = ' or '.join(map(str, expected_shapes))
        raise ValueError(
            f"Invalid reduce output shape, buffer shape is {buffer.shape}, dim is {dim}, "
            f"output shape is {out.shape}, expected shapes are {expected_shapes_str}")

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
                tir.op.Op.get("tl.reduce"),
                red_frag_in.access_ptr("r"),
                red_frag_out.access_ptr("w"),
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
                tir.op.Op.get("tl.reduce"),
                red_frag_in.access_ptr("r"),
                out.access_ptr("w"),
                reduce_type,
                dim,
                clear,
            )
        elif is_fragment(buffer) and is_shared(out):
            red_frag_out = alloc_fragment(out.shape, out.dtype)
            IRBuilder.name(out.name + "_frag", red_frag_out)

            tir.call_intrin(
                "handle",
                tir.op.Op.get("tl.reduce"),
                buffer.access_ptr("r"),
                red_frag_out.access_ptr("w"),
                reduce_type,
                dim,
                clear,
            )
            copy(red_frag_out, out)
        elif is_fragment(buffer) and is_fragment(out):
            tir.call_intrin(
                "handle",
                tir.op.Op.get("tl.reduce"),
                buffer.access_ptr("r"),
                out.access_ptr("w"),
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
def cumsum_fragment(src: tir.Buffer, dst: tir.Buffer, dim: int, reverse: bool) -> tir.PrimExpr:
    cumsum_smem = alloc_shared(src.shape, src.dtype, "shared.dyn")
    copy(src, cumsum_smem)
    tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.cumsum"),
        cumsum_smem.access_ptr("r"),
        cumsum_smem.access_ptr("w"),
        dim,
        reverse,
    )
    copy(cumsum_smem, dst)


def cumsum(src: tir.Buffer, dst: tir.Buffer | None = None, dim: int = 0, reverse: bool = False):
    """
    Compute the cumulative sum of `src` along `dim`, writing results to `dst`.

    Negative `dim` indices are normalized (Python-style). If `dst` is None, the operation is performed in-place into `src`. Raises ValueError when `dim` is out of bounds for `src.shape`. When `src.scope() == "local.fragment"`, this delegates to `cumsum_fragment`; otherwise it emits the `tl.cumsum` intrinsic.

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

    Returns:
        tir.Call: A handle to the emitted cumulative-sum operation.
    """

    shape = src.shape
    if dim >= len(shape) or dim <= -len(shape):
        raise ValueError(f"Dimension {dim} is out of bounds for buffer with shape {shape}")
    if dim < 0:
        dim = len(shape) + dim

    if dst is None:
        dst = src
    if src.scope() == "local.fragment":
        return cumsum_fragment(src, dst, dim, reverse)
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.cumsum"),
        src.access_ptr("r"),
        dst.access_ptr("w"),
        dim,
        reverse,
    )


def finalize_reducer(reducer: tir.Buffer):
    """
    Finalize a reducer buffer by emitting the `tl.finalize_reducer` intrinsic.

    This returns a TVM `tir.Call` handle that finalizes the given reducer using its writable pointer.
    The call does not modify Python objects directly; it produces the low-level intrinsic call used by the IR.

    Parameters:
        reducer (tir.Buffer): Reducer buffer whose writable pointer will be finalized.

    Returns:
        tir.Call: Handle to the finalize reducer intrinsic call.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.finalize_reducer"),
        reducer.access_ptr("w"),
    )
