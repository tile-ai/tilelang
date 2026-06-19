"""Scan operations exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.language import alloc_shared, copy, macro
from tilelang.utils.language import _get_buffer, is_fragment, retrieve_shape, to_tile_region
from tvm import tirx

_CUMSUM_OP_KEY = "tl.tileop.cumsum"
_CUMMAX_OP_KEY = "tl.tileop.cummax"


@macro
def _scan_fragment(
    src: BufferLikeType,
    dst: BufferLikeType,
    dim: int,
    reverse: bool,
    op_key: str,
) -> None:
    src_shape = retrieve_shape(src)
    src_buffer = _get_buffer(src)
    if isinstance(src, tirx.Buffer):
        dtype = src.dtype
    else:
        dtype = src_buffer.dtype
    scan_smem = alloc_shared(src_shape, dtype, "shared.dyn")
    copy(src, scan_smem)
    tirx.call_intrin(
        "handle",
        tirx.op.Op.get(op_key),
        to_tile_region(scan_smem, access_type="r"),
        to_tile_region(scan_smem, access_type="w"),
        dim,
        reverse,
    )
    copy(scan_smem, dst)


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
    _scan_fragment(src, dst, dim, reverse, _CUMSUM_OP_KEY)


def _prepare_scan_args(src: BufferLikeType, dst: BufferLikeType | None, dim: int, op_name: str) -> tuple[BufferLikeType, int]:
    shape = retrieve_shape(src)
    if dim >= len(shape) or dim < -len(shape):
        raise ValueError(f"Dimension {dim} is out of bounds for buffer with shape {shape}")
    if dim < 0:
        dim = len(shape) + dim

    if dst is None:
        dst = src
    else:
        dst_shape = retrieve_shape(dst)
        if len(dst_shape) != len(shape):
            raise ValueError(f"{op_name} dst shape {dst_shape} must match src shape {shape} (rank mismatch)")
        for i in range(len(shape)):
            if not tirx.analysis.expr_deep_equal(dst_shape[i], shape[i]):
                raise ValueError(f"{op_name} dst shape {dst_shape} must match src shape {shape} (dim {i} mismatch)")

    return dst, dim


# NOTE(chaofan): T.cumsum returns None if it goes to macro implementations
def cumsum(
    src: BufferLikeType,
    dst: BufferLikeType | None = None,
    dim: int = 0,
    reverse: bool = False,
) -> tirx.PrimExpr | None:
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
        tirx.Call: A handle to the emitted cumulative-sum operation.
    """

    dst, dim = _prepare_scan_args(src, dst, dim, "cumsum")

    if is_fragment(src):
        cumsum_fragment(src, dst, dim, reverse)
        return

    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get(_CUMSUM_OP_KEY),
        to_tile_region(src, access_type="r"),
        to_tile_region(dst, access_type="w"),
        dim,
        reverse,
    )


@macro
def cummax_fragment(
    src: BufferLikeType,
    dst: BufferLikeType,
    dim: int,
    reverse: bool,
) -> None:
    """
    Compute cumulative maximum for fragment buffers by staging through shared memory.

    Args:
        src: Source buffer (Buffer, BufferRegion, or BufferLoad) containing input data.
        dst: Destination buffer (Buffer, BufferRegion, or BufferLoad) for output data.
        dim: Dimension along which to compute cumulative maximum.
        reverse: If True, compute cumulative maximum in reverse order.
    """
    _scan_fragment(src, dst, dim, reverse, _CUMMAX_OP_KEY)


def cummax(
    src: BufferLikeType,
    dst: BufferLikeType | None = None,
    dim: int = 0,
    reverse: bool = False,
) -> tirx.PrimExpr | None:
    """
    Compute the cumulative maximum of `src` along `dim`, writing results to `dst`.

    Negative `dim` indices are normalized (Python-style). If `dst` is None,
    the operation is performed in-place into `src`. When `src.scope()` is
    "local.fragment", this delegates to `cummax_fragment`; otherwise it emits
    the `tl.cummax` intrinsic.
    """

    dst, dim = _prepare_scan_args(src, dst, dim, "cummax")

    if is_fragment(src):
        cummax_fragment(src, dst, dim, reverse)
        return

    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get(_CUMMAX_OP_KEY),
        to_tile_region(src, access_type="r"),
        to_tile_region(dst, access_type="w"),
        dim,
        reverse,
    )
