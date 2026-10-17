"""Compilation settings and tile geometry shared by all kernel families."""

import tilelang.ascend.language as T

from ..config import Major

# Alpha epilogues can fill all 256 KiB of UB. These kernels do not need VF
# stack spills or Ascend C APIs that use reserved UB; retain the full dynamic
# UB reservation at launch and release both compiler-reserved regions.
COMPILE_FLAGS = [
    "--cce-disable-vf-stack-reserved-ubuf",
    "--cce-disable-asc-reserved-ubuf",
]

# Each group starts at a multiple of MK_ALIGNMENT.
MK_ALIGNMENT = 256

# AutoSchedule currently needs distinct cross-core flags for epilogue stages.
MAX_EPILOGUE_STAGES = 8

# Each packed int16 scale pair covers 64 FP8 K elements.
MX_SF_DIVISOR = 64


def clamp(value, lower, upper):
    return T.max(T.min(value, upper), lower)


def physical_shape(mn, k, major):
    return (mn, k) if major == Major.K else (k, mn)


def matrix_region(buffer, row, col, rows, cols, batch_idx=0):
    """Select a matrix tile, preserving an optional batch/group axis and strides."""
    if len(buffer.shape) == 3:
        return buffer[batch_idx, row : row + rows, col : col + cols]
    return buffer[row : row + rows, col : col + cols]


def scale_region(scales, mn_idx, actual_mn, sf_idx, actual_k, tile_idx=0):
    if scales is None:
        return None
    return matrix_region(scales, mn_idx, sf_idx, actual_mn, actual_k // MX_SF_DIVISOR, batch_idx=tile_idx)
