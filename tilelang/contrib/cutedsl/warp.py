# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Warp-level primitives for CuTeDSL backend.
TileLang-compatible wrappers over CUTLASS primitives.
"""

__all__ = [
    "__activemask",
    "__shfl_down_sync",
    "__shfl_up_sync",
    "__shfl_sync",
    "__shfl_xor_sync",
    "__match_any_sync",
    "__popc",
    "warp_reduce_sum",
    "warp_reduce_max",
    "warp_reduce_min",
    "warp_reduce_bitand",
    "warp_reduce_bitor",
]

from cutlass.experimental import primitives as prims
from cutlass._mlir.dialects import arith, math as _math
from cutlass._mlir_helpers.arith import ArithValue
from cutlass.base_dsl.typing import BFloat16, Float16, Int16, Int32, Numeric, Uint32
from cutlass.cutlass_dsl import dsl_user_op


FULL_MASK = 0xFFFFFFFF
WARP_SIZE = 32


def _as_shfl_value(val):
    if isinstance(val, ArithValue):
        return Numeric.from_mlir_type(val.type)(val)
    return val


@dsl_user_op
def _shfl_sync_typed(mask, val, offset, mask_and_clamp, kind, *, loc=None, ip=None):
    val = _as_shfl_value(val)
    val_type = type(val)
    if val_type in (Float16, BFloat16):
        val_i16 = val.bitcast(Int16, loc=loc, ip=ip)
        val_i32 = Int32(arith.extui(Int32.mlir_type, val_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
        shuffled_i32 = prims.shfl_sync(mask, val_i32, offset, mask_and_clamp, kind, loc=loc, ip=ip)
        shuffled_i16 = Int16(arith.trunci(Int16.mlir_type, shuffled_i32.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
        return shuffled_i16.bitcast(val_type, loc=loc, ip=ip)
    return prims.shfl_sync(mask, val, offset, mask_and_clamp, kind, loc=loc, ip=ip)


@dsl_user_op
def __activemask(*, loc=None, ip=None) -> Uint32:
    """
    Returns a 32-bit integer mask of all currently active threads in the calling warp.
    """
    return Uint32(
        prims.inline_ptx(
            "activemask.b32 {$w0};",
            write_only_types=[Uint32],
            loc=loc,
            ip=ip,
        )
    )


def __shfl_down_sync(mask, val, delta, width=32):
    """
    Shuffle down within warp.

    Matches CUDA: c = ((warpSize - width) << 8) | 0x1f
    """
    mask_and_clamp = ((WARP_SIZE - width) << 8) | 0x1F
    return _shfl_sync_typed(mask, val, delta, mask_and_clamp, "down")


def __shfl_up_sync(mask, val, delta, width=32):
    """
    Shuffle up within warp.

    Matches CUDA: c = (warpSize - width) << 8
    """
    mask_and_clamp = (WARP_SIZE - width) << 8
    return _shfl_sync_typed(mask, val, delta, mask_and_clamp, "up")


def __shfl_sync(mask, val, srcLane, width=32):
    """
    Broadcast from a specific lane within warp.

    Matches CUDA: c = ((warpSize - width) << 8) | (width - 1)
    """
    mask_and_clamp = ((WARP_SIZE - width) << 8) | ((width - 1) & 0x1F)
    return _shfl_sync_typed(mask, val, srcLane, mask_and_clamp, "idx")


def __shfl_xor_sync(mask, val, lane_mask, width=32):
    """
    Butterfly (XOR) shuffle within warp.

    Uses the same clamp encoding as CUDA __shfl_xor_sync.
    """
    mask_and_clamp = ((WARP_SIZE - width) << 8) | ((width - 1) & 0x1F)
    return _shfl_sync_typed(mask, val, lane_mask, mask_and_clamp, "bfly")


@dsl_user_op
def __match_any_sync(mask, val, *, loc=None, ip=None) -> Uint32:
    """
    Return a bitmask of lanes whose value matches the calling lane.

    Uses the primitive match wrapper.
    """
    return Uint32(prims.match_sync(mask, Int32(val), "any", loc=loc, ip=ip))


@dsl_user_op
def __popc(val, *, loc=None, ip=None) -> Int32:
    """
    Count set bits in a 32-bit value.

    Uses MLIR math.ctpop; no inline PTX is needed for the unpredicated form.
    """
    return Int32(_math.ctpop(Int32(val).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def _shfl_xor_sync(val, lane_mask):
    """Butterfly (XOR) shuffle within full warp."""
    return _shfl_sync_typed(FULL_MASK, val, lane_mask, 0x1F, "bfly")


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
