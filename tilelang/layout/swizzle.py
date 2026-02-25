"""Wrapping Layouts."""

# pylint: disable=invalid-name, unsupported-binary-operation
from __future__ import annotations

import tvm
from tvm import tir
from tilelang import _ffi_api
from tilelang._typing import BufferLikeType, BufferLikeTypeTuple


def _get_buffer_info(buffer_or_load_or_region: BufferLikeType) -> tuple[tir.Buffer, list[int], str]:
    """
    Extract buffer, shape, and dtype from BufferLikeType.

    Args:
        buffer_or_load_or_region: BufferLikeType

    Returns:
        tuple: (buffer, shape, dtype)
    """
    if isinstance(buffer_or_load_or_region, tir.Buffer):
        return buffer_or_load_or_region, buffer_or_load_or_region.shape, buffer_or_load_or_region.dtype
    elif isinstance(buffer_or_load_or_region, BufferLikeTypeTuple):
        buf = buffer_or_load_or_region.buffer
        return buf, buf.shape, buf.dtype
    else:
        raise TypeError(f"Expected BufferLikeType, got {type(buffer_or_load_or_region)}")


def _get_stride_continuous(buffer_or_load_or_region: BufferLikeType) -> tuple[int, int]:
    """
    Get stride (product of all dims except the last) and continuous (last dimension)
    from BufferLikeType.

    Args:
        buffer_or_load_or_region: BufferLikeType

    Returns:
        tuple: (stride, continuous) as integers
    """
    _, shape, _ = _get_buffer_info(buffer_or_load_or_region)
    stride = 1
    for dim in shape[:-1]:
        stride *= int(dim)
    continuous = int(shape[-1])
    return stride, continuous


def _get_element_size(buffer_or_load_or_region: BufferLikeType) -> int:
    """
    Get element size in bits from BufferLikeType.

    Args:
        buffer_or_load_or_region: BufferLikeType

    Returns:
        int: Element size in bits
    """
    _, _, dtype = _get_buffer_info(buffer_or_load_or_region)
    return int(tvm.DataType(dtype).bits)


# Use a stable swizzled layout to ensure consistent memory access patterns.
# Swizzling should be enabled or disabled based on whether TMA (Tensor Memory Access) is applied.
def make_swizzled_layout(buffer: BufferLikeType, k_major: bool = True, allow_pad: bool = True):
    _, shape, _ = _get_buffer_info(buffer)
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    base = _ffi_api.make_swizzled_layout(
        stride,
        continuous,
        element_size,
        k_major,
        allow_pad,
    )
    return base.reshape(shape)


# for Volta Intrinsics
def make_volta_swizzled_layout(buffer: BufferLikeType, is_a: bool = True, k_inner: bool = True):
    _, shape, _ = _get_buffer_info(buffer)
    stride, continuous = _get_stride_continuous(buffer)
    base = _ffi_api.make_volta_swizzled_layout(
        stride,
        continuous,
        is_a,
        k_inner,
    )
    return base.reshape(shape)


# for WGMMA Intrinsics
def make_wgmma_swizzled_layout(buffer: BufferLikeType, continuity: int = None, k_major: bool = True):
    _, shape, _ = _get_buffer_info(buffer)
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    if continuity is None:
        continuity = continuous
    base = _ffi_api.make_wgmma_swizzled_layout(
        stride,
        continuous,
        continuity,
        element_size,
        k_major,
    )
    return base.reshape(shape)


# for TCGEN05MMA Intrinsics
def make_tcgen05mma_swizzled_layout(buffer: BufferLikeType, continuity: int = None, k_major: bool = True):
    _, shape, _ = _get_buffer_info(buffer)
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    if continuity is None:
        continuity = continuous
    base = _ffi_api.make_tcgen05mma_swizzled_layout(
        stride,
        continuous,
        continuity,
        element_size,
        k_major,
    )
    return base.reshape(shape)


# swizzle 128B
def make_full_bank_swizzled_layout(buffer: BufferLikeType):
    """
    Args:
        buffer: BufferLikeType
    Examples:
        make_full_bank_swizzled_layout(buffer)
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_full_bank_swizzled_layout(buf)


# swizzle 64B
def make_half_bank_swizzled_layout(buffer: BufferLikeType):
    """
    Args:
        buffer: BufferLikeType
    Examples:
        make_half_bank_swizzled_layout(buffer)
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_half_bank_swizzled_layout(buf)


# swizzle 32B
def make_quarter_bank_swizzled_layout(buffer: BufferLikeType):
    """
    Args:
        buffer: BufferLikeType
    Examples:
        make_quarter_bank_swizzled_layout(buffer)
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_quarter_bank_swizzled_layout(buf)


def make_linear_layout(buffer_or_load_or_region: BufferLikeType):
    """
    Create a row-major linear layout for any dimension.

    Args:
        buffer_or_load_or_region: BufferLikeType

    Returns:
        Layout: A row-major linear layout
    """
    _, shape, _ = _get_buffer_info(buffer_or_load_or_region)
    return _ffi_api.make_linear_layout(list(shape))


def make_gemm_fragment_8x8():
    """
    Create a standard 8x8 GEMM fragment layout for ldmatrix/stmatrix.

    This layout matches the warp-level matrix multiplication pattern used in tensor cores.

    Returns:
        Fragment: An 8x8 fragment layout
    """
    return _ffi_api.make_gemm_fragment_8x8()


def make_gemm_fragment_8x8_transposed():
    """
    Create a transposed 8x8 GEMM fragment layout for ldmatrix/stmatrix.

    This layout is the transposed version of make_gemm_fragment_8x8, useful for
    different access patterns in matrix operations.

    Returns:
        Fragment: A transposed 8x8 fragment layout
    """
    return _ffi_api.make_gemm_fragment_8x8_transposed()


def make_fully_replicated_layout_fragment(buffer: BufferLikeType, threads: int):
    """
    Create a fully replicated layout for a fragment buffer.

    A fully replicated fragment means all threads hold identical copies of the
    entire buffer. This is useful for index buffers or masks that need to be
    accessed uniformly across all threads.

    Args:
        buffer: BufferLikeType to get shape information
        threads: Number of threads (replicate extent)

    Returns:
        Fragment: A fully replicated layout where each thread has a complete copy

    Example:
        >>> C_local = T.alloc_fragment((2,), T.float32)
        >>> layout = make_fully_replicated_layout_fragment(C_local, 256)
        >>> T.annotate_layout({C_local: layout})
    """
    _, shape, _ = _get_buffer_info(buffer)
    return _ffi_api.make_fully_replicated_layout_fragment(list(shape), threads)
