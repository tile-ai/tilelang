"""The language interface for tl programs."""
from __future__ import annotations

from typing import Literal
from tilelang import language as T
from tilelang.utils.language import get_buffer_region_from_load
from tvm import ir, tir
from tilelang.language.utils import buffer_to_tile_region, buffer_region_to_tile_region, buffer_load_to_tile_region


def copy(src: tir.Buffer | tir.BufferLoad | tir.BufferRegion,
         dst: tir.Buffer | tir.BufferLoad,
         coalesced_width: int | None = None,
         disable_tma: bool = False,
         eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None):
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination memory region
        coalesced_width (Optional[int], optional): Width for coalesced memory access. Defaults to None.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation
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
    if (src_extent is None and dst_extent is None and isinstance(src, tir.BufferLoad) and
            isinstance(dst, tir.BufferLoad)):
        # check if the case is like this:
        # copy(buffer_a[i], buffer_b[i]) where both are BufferLoad nodes
        # In this case, lower it to a simple BufferStore: buffer_b[i] = buffer_a[i]
        return tir.BufferStore(dst.buffer, src, dst.indices)

    assert src_extent or dst_extent, "Can't deduce copy extents from args"
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
    extent = max(src_extent, dst_extent)

    def _to_region(data, access_type):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return buffer_to_tile_region(data, access_type)
        elif isinstance(data, tir.BufferRegion):
            return buffer_region_to_tile_region(data, access_type, extent)
        elif isinstance(data, tir.BufferLoad):
            region = get_buffer_region_from_load(data)
            if region is None:
                return buffer_load_to_tile_region(data, access_type, extent)
            return buffer_region_to_tile_region(region, access_type, extent)
        else:
            return buffer_load_to_tile_region(data, access_type, extent)

    src = _to_region(src, "r")
    dst = _to_region(dst, "w")

    if coalesced_width is None:
        coalesced_width = -1  # PrimExpr can not be None
    if eviction_policy is None:
        eviction_policy = 0
    else:
        eviction_policy = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}[eviction_policy]
    return tir.call_intrin("handle", tir.op.Op.get("tl.copy"), src, dst, coalesced_width,
                           disable_tma, eviction_policy)


def c2d_im2col(img: tir.Buffer,
               col: tir.Buffer,
               nhw_step: tir.PrimExpr,
               c_step: tir.PrimExpr,
               kernel: int,
               stride: int,
               dilation: int,
               pad: int,
               eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None):
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
    return tir.call_intrin("handle", tir.op.Op.get("tl.c2d_im2col"), img.access_ptr("r"),
                           col.access_ptr("w"), nhw_step, c_step, kernel, stride, dilation, pad,
                           eviction_policy)
