"""Wrapping Layouts."""

# pylint: disable=invalid-name, unsupported-binary-operation
from __future__ import annotations

import tvm_ffi
from tvm_tirx import ir
from tilelang import _ffi_api
from ._utils import BufferLikeType, as_tensor_view


def _get_buffer_info(buffer_or_load_or_region: BufferLikeType) -> tuple[ir.Var, list[ir.Expr], tvm_ffi.dtype]:
    """
    Extract buffer, shape, and dtype from BufferLikeType.

    Args:
        buffer_or_load_or_region: BufferLikeType

    Returns:
        tuple: (buffer, shape, dtype)
    """
    view = as_tensor_view(buffer_or_load_or_region)
    return view.tensor, view.shape, view.dtype


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
    return int(tvm_ffi.dtype(dtype).bits)


# Use a stable swizzled layout to ensure consistent memory access patterns.
# Swizzling should be enabled or disabled based on whether TMA (Tensor Memory Access) is applied.
def make_swizzled_layout(buffer: BufferLikeType, k_major: bool = True, allow_pad: bool = True):
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_swizzled_layout(buf, k_major, allow_pad)


# for Volta Intrinsics
def make_volta_swizzled_layout(buffer: BufferLikeType, is_a: bool = True, k_inner: bool = True):
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_volta_swizzled_layout(buf, is_a, k_inner)


# for WGMMA Intrinsics
def make_wgmma_swizzled_layout(buffer: BufferLikeType, continuity: int = None, k_major: bool = True):
    buf, _, _ = _get_buffer_info(buffer)
    if continuity is None:
        continuity = -1
    return _ffi_api.make_wgmma_swizzled_layout(buf, continuity, k_major)


# for TCGEN05MMA Intrinsics
def make_tcgen05mma_swizzled_layout(buffer: BufferLikeType, continuity: int = None, k_major: bool = True):
    buf, _, _ = _get_buffer_info(buffer)
    if continuity is None:
        continuity = -1
    return _ffi_api.make_tcgen05mma_swizzled_layout(buf, continuity, k_major)


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
