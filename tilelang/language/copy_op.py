"""Copy operations exposed on the TileLang language surface."""

from __future__ import annotations
from typing import Literal
from tilelang import language as T
from tilelang.utils.language import (
    to_buffer_region,
    get_buffer_region_from_load,
    legalize_pairwise_extents,
)
from tvm import ir, tir


def copy(
    src: tir.Buffer | tir.BufferLoad | tir.BufferRegion,
    dst: tir.Buffer | tir.BufferLoad | tir.BufferRegion,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
):
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Destination memory region
        coalesced_width (Optional[int], optional): Width for coalesced memory access. Defaults to None.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation

    Range handling notes:
    - Accepts `Buffer`/`BufferRegion`/`BufferLoad` on either side. Extents are
      derived as follows: `Buffer -> shape`, `BufferRegion -> [r.extent]`,
      `BufferLoad -> extents from its inferred/encoded region`.
    - If both `src` and `dst` are scalar `BufferLoad` without region extents,
      lowers to a direct store: `dst[...] = src`.
    - If one side is missing extents, it is treated as all-ones with the other
      side's rank to enable broadcasting.
    - Extents are right-aligned and legalized via `legalize_pairwise_extents`:
      per tail-dimension, equal keeps as-is, a `1` broadcasts to the other,
      otherwise a conservative `tir.max` is used to remain safe for dynamic
      shapes.
    - The finalized extents are encoded with `tl.region` via `to_buffer_region`
      and passed through to the backend; low-level loop construction and any
      scope-specific decisions happen during lowering.
    """
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    def get_extent(data):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return data.shape
        elif isinstance(data, tir.BufferRegion):
            return [x.extent for x in data.region]
        elif isinstance(data, tir.BufferLoad):
            region = get_buffer_region_from_load(data)
            if region is None:
                return None
            return [x.extent for x in region.region]
        else:
            return None

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)
    # Combine the nested if statements into a single if statement as suggested by SIM102
    if src_extent is None and dst_extent is None and isinstance(src, tir.BufferLoad) and isinstance(dst, tir.BufferLoad):
        # check if the case is like this:
        # copy(buffer_a[i], buffer_b[i]) where both are BufferLoad nodes
        # In this case, lower it to a simple BufferStore: buffer_b[i] = buffer_a[i]
        return tir.BufferStore(dst.buffer, src, dst.indices)

    assert src_extent or dst_extent, "Can't deduce copy extents from args"
    # Treat missing extent as length-matched ones to enable broadcasting.
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    # Align and broadcast extents from the right (tail) side.
    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    # Use legalized extents for src and dst respectively.
    src = to_buffer_region(src, access_type="r", extents=src_extent)
    dst = to_buffer_region(dst, access_type="w", extents=dst_extent)

    if coalesced_width is None:
        coalesced_width = -1  # PrimExpr can not be None
    if eviction_policy is None:
        eviction_policy = 0
    else:
        eviction_policy = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}[eviction_policy]
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.copy"), src, dst, coalesced_width, disable_tma, eviction_policy)


def c2d_im2col(
    img: tir.Buffer,
    col: tir.Buffer,
    nhw_step: tir.PrimExpr,
    c_step: tir.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
):
    """Perform im2col transformation for 2D convolution.

    Args:
        img (tir.Buffer): Input image buffer
        col (tir.Buffer): Output column buffer
        nhw_step (tir.PrimExpr): Step size for batch and spatial dimensions
        c_step (tir.PrimExpr): Step size for channel dimension
        kernel (int): Kernel size
        stride (int): Stride of the convolution
        dilation (int): Dilation rate
        pad (int): Padding size

    Returns:
        tir.Call: A handle to the im2col operation
    """
    if eviction_policy is None:
        eviction_policy = 0
    else:
        eviction_policy = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}[eviction_policy]
    img_region = to_buffer_region(img, access_type="r")
    col_region = to_buffer_region(col, access_type="w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.c2d_im2col"),
        img_region,
        col_region,
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
        eviction_policy,
    )
