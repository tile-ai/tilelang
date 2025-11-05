"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation

import tvm
from tilelang import _ffi_api


# Use a stable swizzled layout to ensure consistent memory access patterns.
# Swizzling should be enabled or disabled based on whether TMA (Tensor Memory Access) is applied.
def make_swizzled_layout(buffer: tvm.tir.Buffer, k_major: bool = True, allow_pad: bool = True):
    assert len(buffer.shape) == 2
    return _ffi_api.make_swizzled_layout(
        int(buffer.shape[0]),
        int(buffer.shape[1]),
        int(tvm.DataType(buffer.dtype).bits),
        k_major,
        allow_pad,
    )


# for Volta Intrinsics
def make_volta_swizzled_layout(buffer: tvm.tir.Buffer, is_a: bool = True, k_inner: bool = True):
    assert len(buffer.shape) == 2
    return _ffi_api.make_volta_swizzled_layout(
        int(buffer.shape[0]),
        int(buffer.shape[1]),
        is_a,
        k_inner,
    )


# for WGMMA Intrinsics
def make_wgmma_swizzled_layout(buffer: tvm.tir.Buffer,
                               continuity: int = None,
                               k_major: bool = True):
    assert len(buffer.shape) == 2
    if continuity is None:
        continuity = int(buffer.shape[1])
    return _ffi_api.make_wgmma_swizzled_layout(
        int(buffer.shape[0]),
        int(buffer.shape[1]),
        continuity,
        int(tvm.DataType(buffer.dtype).bits),
        k_major,
    )


# for TCGEN05MMA Intrinsics
def make_tcgen05mma_swizzled_layout(buffer: tvm.tir.Buffer,
                                    continuity: int = None,
                                    k_major: bool = True):
    assert len(buffer.shape) == 2
    if continuity is None:
        continuity = int(buffer.shape[1])
    return _ffi_api.make_tcgen05mma_swizzled_layout(
        int(buffer.shape[0]),
        int(buffer.shape[1]),
        continuity,
        int(tvm.DataType(buffer.dtype).bits),
        k_major,
    )


# swizzle 128B
# args: buffer or (stride, continuous, element_size)
def make_full_bank_swizzled_layout(*args):
    """
    Args:
        args: buffer or (stride, continuous, element_size)
    Examples:
        make_full_bank_swizzled_layout(buffer)
        make_full_bank_swizzled_layout(stride, continuous, element_size)
    """
    if len(args) == 1:
        buffer = args[0]
        stride, continuous = int(buffer.shape[0]), int(buffer.shape[1])
        element_size = int(tvm.DataType(buffer.dtype).bits)
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
        args: buffer or (stride, continuous, element_size)
    Examples:
        make_half_bank_swizzled_layout(buffer)
        make_half_bank_swizzled_layout(stride, continuous, element_size)
    """
    if len(args) == 1:
        buffer = args[0]
        stride, continuous = int(buffer.shape[0]), int(buffer.shape[1])
        element_size = int(tvm.DataType(buffer.dtype).bits)
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
        args: buffer or (stride, continuous, element_size)
    Examples:
        make_quarter_bank_swizzled_layout(buffer)
        make_quarter_bank_swizzled_layout(stride, continuous, element_size)
    """
    if len(args) == 1:
        buffer = args[0]
        stride, continuous = int(buffer.shape[0]), int(buffer.shape[1])
        element_size = int(tvm.DataType(buffer.dtype).bits)
    elif len(args) == 3:
        stride, continuous, element_size = args
    else:
        raise ValueError(f"Invalid arguments: {args}")
    return _ffi_api.make_quarter_bank_swizzled_layout(
        stride,
        continuous,
        element_size,
    )


def make_linear_layout(*args):
    """
    Args:
        args: buffer or (stride, continuous)
    Examples:
        make_linear_layout(buffer)
        make_linear_layout(stride, continuous)
    """
    if len(args) == 1:
        buffer = args[0]
        stride, continuous = int(buffer.shape[0]), int(buffer.shape[1])
    elif len(args) == 2:
        stride, continuous = args
    else:
        raise ValueError(f"Invalid arguments: {args}")
    return _ffi_api.make_linear_layout(
        stride,
        continuous,
    )
