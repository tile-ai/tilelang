"""Wrapping Layouts."""

# pylint: disable=invalid-name, unsupported-binary-operation
from __future__ import annotations

import tvm
from tvm.tir import Buffer, BufferLoad, BufferRegion
from tilelang import _ffi_api


def _get_buffer_info(buffer_or_load_or_region: Buffer | BufferLoad | BufferRegion) -> tuple[Buffer, list[int], str]:
    """
    Extract buffer, shape, and dtype from Buffer, BufferLoad, or BufferRegion.

    Args:
        buffer_or_load_or_region: Can be Buffer, BufferLoad, or BufferRegion

    Returns:
        tuple: (buffer, shape, dtype)
    """
    if isinstance(buffer_or_load_or_region, Buffer):
        return buffer_or_load_or_region, buffer_or_load_or_region.shape, buffer_or_load_or_region.dtype
    elif isinstance(buffer_or_load_or_region, (BufferLoad, BufferRegion)):
        buf = buffer_or_load_or_region.buffer
        return buf, buf.shape, buf.dtype
    else:
        raise TypeError(f"Expected Buffer, BufferLoad, or BufferRegion, got {type(buffer_or_load_or_region)}")


def _get_stride_continuous(buffer_or_load_or_region: Buffer | BufferLoad | BufferRegion) -> tuple[int, int]:
    """
    Get stride (last 2nd dimension) and continuous (last dimension) from Buffer, BufferLoad, or BufferRegion.

    Args:
        buffer_or_load_or_region: Can be Buffer, BufferLoad, or BufferRegion

    Returns:
        tuple: (stride, continuous) as integers
    """
    _, shape, _ = _get_buffer_info(buffer_or_load_or_region)
    stride = int(shape[-2])
    continuous = int(shape[-1])
    return stride, continuous


def _get_element_size(buffer_or_load_or_region: Buffer | BufferLoad | BufferRegion) -> int:
    """
    Get element size in bits from Buffer, BufferLoad, or BufferRegion.

    Args:
        buffer_or_load_or_region: Can be Buffer, BufferLoad, or BufferRegion

    Returns:
        int: Element size in bits
    """
    _, _, dtype = _get_buffer_info(buffer_or_load_or_region)
    return int(tvm.DataType(dtype).bits)


# Use a stable swizzled layout to ensure consistent memory access patterns.
# Swizzling should be enabled or disabled based on whether TMA (Tensor Memory Access) is applied.
def make_swizzled_layout(buffer: Buffer | BufferLoad | BufferRegion, k_major: bool = True, allow_pad: bool = True):
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    return _ffi_api.make_swizzled_layout(
        stride,
        continuous,
        element_size,
        k_major,
        allow_pad,
    )


# for Volta Intrinsics
def make_volta_swizzled_layout(buffer: Buffer | BufferLoad | BufferRegion, is_a: bool = True, k_inner: bool = True):
    stride, continuous = _get_stride_continuous(buffer)
    return _ffi_api.make_volta_swizzled_layout(
        stride,
        continuous,
        is_a,
        k_inner,
    )


# for WGMMA Intrinsics
def make_wgmma_swizzled_layout(buffer: Buffer | BufferLoad | BufferRegion, continuity: int = None, k_major: bool = True):
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    if continuity is None:
        continuity = continuous
    return _ffi_api.make_wgmma_swizzled_layout(
        stride,
        continuous,
        continuity,
        element_size,
        k_major,
    )


# for TCGEN05MMA Intrinsics
def make_tcgen05mma_swizzled_layout(buffer: Buffer | BufferLoad | BufferRegion, continuity: int = None, k_major: bool = True):
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    if continuity is None:
        continuity = continuous
    return _ffi_api.make_tcgen05mma_swizzled_layout(
        stride,
        continuous,
        continuity,
        element_size,
        k_major,
    )


# swizzle 128B
# args: buffer or (stride, continuous, element_size)
def make_full_bank_swizzled_layout(*args):
    """
    Args:
        args: buffer/BufferLoad/BufferRegion or (stride, continuous, element_size)
    Examples:
        make_full_bank_swizzled_layout(buffer)
        make_full_bank_swizzled_layout(stride, continuous, element_size)
    """
    if len(args) == 1:
        stride, continuous = _get_stride_continuous(args[0])
        element_size = _get_element_size(args[0])
    elif len(args) == 3:
        stride, continuous, element_size = args
    else:
        raise ValueError(f"Invalid arguments: {args}")
    return _ffi_api.make_full_bank_swizzled_layout(
        stride,
        continuous,
        element_size,
    )


# swizzle 64B
# args: buffer or (stride, continuous, element_size)
def make_half_bank_swizzled_layout(*args):
    """
    Args:
        args: buffer/BufferLoad/BufferRegion or (stride, continuous, element_size)
    Examples:
        make_half_bank_swizzled_layout(buffer)
        make_half_bank_swizzled_layout(stride, continuous, element_size)
    """
    if len(args) == 1:
        stride, continuous = _get_stride_continuous(args[0])
        element_size = _get_element_size(args[0])
    elif len(args) == 3:
        stride, continuous, element_size = args
    else:
        raise ValueError(f"Invalid arguments: {args}")
    return _ffi_api.make_half_bank_swizzled_layout(
        stride,
        continuous,
        element_size,
    )


# swizzle 32B
# args: buffer or (stride, continuous, element_size)
def make_quarter_bank_swizzled_layout(*args):
    """
    Args:
        args: buffer/BufferLoad/BufferRegion or (stride, continuous, element_size)
    Examples:
        make_quarter_bank_swizzled_layout(buffer)
        make_quarter_bank_swizzled_layout(stride, continuous, element_size)
    """
    if len(args) == 1:
        stride, continuous = _get_stride_continuous(args[0])
        element_size = _get_element_size(args[0])
    elif len(args) == 3:
        stride, continuous, element_size = args
    else:
        raise ValueError(f"Invalid arguments: {args}")
    return _ffi_api.make_quarter_bank_swizzled_layout(
        stride,
        continuous,
        element_size,
    )


def make_linear_layout(buffer_or_load_or_region: Buffer | BufferLoad | BufferRegion):
    """
    Create a row-major linear layout for any dimension.

    Args:
        buffer_or_load_or_region: Buffer, BufferLoad, or BufferRegion

    Returns:
        Layout: A row-major linear layout
    """
    _, shape, _ = _get_buffer_info(buffer_or_load_or_region)
    return _ffi_api.make_linear_layout(list(shape))
