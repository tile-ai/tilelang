# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Warp-level primitives for CuTeDSL backend.
Re-exports from cutlass.cute.arch with TileLang naming conventions.
"""

__all__ = [
    "__activemask",
    "__shfl_down_sync",
    "__shfl_up_sync",
    "__shfl_sync",
    "warp_reduce_sum",
    "warp_reduce_max",
    "warp_reduce_min",
    "warp_reduce_bitand",
    "warp_reduce_bitor",
]

from cutlass._mlir.dialects import llvm, arith
from cutlass.base_dsl.typing import Uint32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.arch import shuffle_sync, shuffle_sync_down, shuffle_sync_up, shuffle_sync_bfly


FULL_MASK = 0xFFFFFFFF
WARP_SIZE = 32


@dsl_user_op
def __activemask(*, loc=None, ip=None) -> Uint32:
    """
    Returns a 32-bit integer mask of all currently active threads in the calling warp.

    PTX: activemask.b32 %mask;
    """
    result = llvm.inline_asm(
        T.i32(),
        [],
        "activemask.b32 $0;",
        "=r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


def __shfl_down_sync(mask, val, delta, width=32):
    """
    Shuffle down within warp.

    Uses CuTeDSL's shuffle_sync_down with proper mask_and_clamp calculation.
    Matches CUDA: c = ((warpSize - width) << 8) | 0x1f
    """
    mask_and_clamp = ((WARP_SIZE - width) << 8) | 0x1F
    return shuffle_sync_down(val, offset=delta, mask=mask, mask_and_clamp=mask_and_clamp)


def __shfl_up_sync(mask, val, delta, width=32):
    """
    Shuffle up within warp.

    Uses CuTeDSL's shuffle_sync_up with proper mask_and_clamp calculation.
    Matches CUDA: c = (warpSize - width) << 8
    """
    mask_and_clamp = (WARP_SIZE - width) << 8
    return shuffle_sync_up(val, offset=delta, mask=mask, mask_and_clamp=mask_and_clamp)


def __shfl_sync(mask, val, srcLane, width=32):
    """
    Broadcast from a specific lane within warp.

    Uses CuTeDSL's shuffle_sync (idx mode) with proper mask_and_clamp.
    Matches CUDA: c = ((warpSize - width) << 8) | (width - 1)
    """
    mask_and_clamp = ((WARP_SIZE - width) << 8) | ((width - 1) & 0x1F)
    return shuffle_sync(val, offset=srcLane, mask=mask, mask_and_clamp=mask_and_clamp)


def _shfl_xor_sync(val, lane_mask):
    """Butterfly (XOR) shuffle within full warp."""
    return shuffle_sync_bfly(val, offset=lane_mask, mask=FULL_MASK, mask_and_clamp=0x1F)


def warp_reduce_sum(value):
    """Warp-level parallel reduction: sum across all 32 lanes."""
    value = value + _shfl_xor_sync(value, 16)
    value = value + _shfl_xor_sync(value, 8)
    value = value + _shfl_xor_sync(value, 4)
    value = value + _shfl_xor_sync(value, 2)
    value = value + _shfl_xor_sync(value, 1)
    return value


def warp_reduce_max(value):
    """Warp-level parallel reduction: max across all 32 lanes."""
    from .reduce import max as tl_max

    value = tl_max(value, _shfl_xor_sync(value, 16))
    value = tl_max(value, _shfl_xor_sync(value, 8))
    value = tl_max(value, _shfl_xor_sync(value, 4))
    value = tl_max(value, _shfl_xor_sync(value, 2))
    value = tl_max(value, _shfl_xor_sync(value, 1))
    return value


def warp_reduce_min(value):
    """Warp-level parallel reduction: min across all 32 lanes."""
    from .reduce import min as tl_min

    value = tl_min(value, _shfl_xor_sync(value, 16))
    value = tl_min(value, _shfl_xor_sync(value, 8))
    value = tl_min(value, _shfl_xor_sync(value, 4))
    value = tl_min(value, _shfl_xor_sync(value, 2))
    value = tl_min(value, _shfl_xor_sync(value, 1))
    return value


@dsl_user_op
def _bitand_i32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(arith.andi(Int32(a).ir_value(), Int32(b).ir_value(), loc=loc, ip=ip))


@dsl_user_op
def _bitor_i32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(arith.ori(Int32(a).ir_value(), Int32(b).ir_value(), loc=loc, ip=ip))


def warp_reduce_bitand(value):
    """Warp-level parallel reduction: bitwise AND across all 32 lanes."""
    value = _bitand_i32(value, _shfl_xor_sync(value, 16))
    value = _bitand_i32(value, _shfl_xor_sync(value, 8))
    value = _bitand_i32(value, _shfl_xor_sync(value, 4))
    value = _bitand_i32(value, _shfl_xor_sync(value, 2))
    value = _bitand_i32(value, _shfl_xor_sync(value, 1))
    return value


def warp_reduce_bitor(value):
    """Warp-level parallel reduction: bitwise OR across all 32 lanes."""
    value = _bitor_i32(value, _shfl_xor_sync(value, 16))
    value = _bitor_i32(value, _shfl_xor_sync(value, 8))
    value = _bitor_i32(value, _shfl_xor_sync(value, 4))
    value = _bitor_i32(value, _shfl_xor_sync(value, 2))
    value = _bitor_i32(value, _shfl_xor_sync(value, 1))
    return value
