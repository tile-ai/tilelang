# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Warp-level primitives for CuTeDSL backend.
TileLang-compatible wrappers over CUTLASS primitives.
"""

__all__ = [
    "__activemask",
    "activemask",
    "__any_sync",
    "__all_sync",
    "__ballot_sync",
    "ballot_sync",
    "ballot",
    "__ffs",
    "__ffsll",
    "__fns",
    "__shfl_down_sync",
    "__shfl_up_sync",
    "__shfl_sync",
    "__shfl_xor_sync",
    "__match_any_sync",
    "__match_all_sync",
    "__popc",
    "warp_reduce_sum",
    "warp_reduce_max",
    "warp_reduce_min",
    "warp_reduce_bitand",
    "warp_reduce_bitor",
]

from cutlass.experimental import primitives as prims
from cutlass._mlir.dialects import arith, llvm, math as _math
from cutlass._mlir_helpers.arith import ArithValue
from cutlass.base_dsl.typing import (
    BFloat16,
    Boolean,
    Float16,
    Float32,
    Float64,
    Int16,
    Int32,
    Int64,
    Integer,
    Numeric,
    Uint32,
    Uint64,
)
from cutlass.cutlass_dsl import dsl_user_op


FULL_MASK = 0xFFFFFFFF
WARP_SIZE = 32


def _as_numeric(val):
    if isinstance(val, ArithValue):
        return Numeric.from_mlir_type(val.type)(val)
    return val


@dsl_user_op
def _shfl_sync_typed(mask, val, offset, mask_and_clamp, kind, *, loc=None, ip=None):
    val = _as_numeric(val)
    val_type = type(val)
    if val_type in (Float16, BFloat16):
        val_i16 = val.bitcast(Int16, loc=loc, ip=ip)
        val_i32 = Int32(arith.extui(Int32.mlir_type, val_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
        shuffled_i32 = prims.shfl_sync(mask, val_i32, offset, mask_and_clamp, kind, loc=loc, ip=ip)
        shuffled_i16 = Int16(arith.trunci(Int16.mlir_type, shuffled_i32.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
        return shuffled_i16.bitcast(val_type, loc=loc, ip=ip)
    if isinstance(val, Numeric) and val.width == 64:
        # Two 32-bit shuffles of the low and high words, as CUDA's 64-bit overloads.
        bits = val.bitcast(Int64, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
        shift = arith.constant(Int64.mlir_type, 32, loc=loc, ip=ip)
        words = []
        for word in (bits, arith.shrui(bits, shift, loc=loc, ip=ip)):
            word_i32 = Int32(arith.trunci(Int32.mlir_type, word, loc=loc, ip=ip))
            shuffled = prims.shfl_sync(mask, word_i32, offset, mask_and_clamp, kind, loc=loc, ip=ip)
            words.append(arith.extui(Int64.mlir_type, Int32(shuffled).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
        joined = arith.ori(words[0], arith.shli(words[1], shift, loc=loc, ip=ip), loc=loc, ip=ip)
        return Int64(joined).bitcast(val_type, loc=loc, ip=ip)
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


@dsl_user_op
def _zext_to_u64(val, *, loc=None, ip=None) -> Uint64:
    """Zero-extend a 32-bit lane mask to 64 bits, like ``(unsigned long long)mask``."""
    return Uint64(arith.extui(Uint64.mlir_type, Uint32(val).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def activemask():
    """
    ``T.activemask()``: the active-lane mask zero-extended to 64 bits.

    Matches CUDA: (unsigned long long)__activemask()
    """
    return _zext_to_u64(__activemask())


def _as_cuda_predicate(pred):
    """
    Convert a predicate the way CUDA's ``int predicate`` parameters do.

    Non-bool values are first converted to a 32-bit int (C conversion, so wider
    integers are truncated) and then tested against zero.
    """
    pred = _as_numeric(pred)
    if isinstance(pred, (bool, Boolean)):
        return Boolean(pred)
    return Boolean(Int32(pred))


@dsl_user_op
def __any_sync(mask, pred, *, loc=None, ip=None) -> Int32:
    """
    Return 1 if the predicate is non-zero for any lane in ``mask``, else 0.

    Matches CUDA __any_sync (vote.sync.any.pred).
    """
    return Int32(prims.vote_sync(mask, _as_cuda_predicate(pred), "any", loc=loc, ip=ip))


@dsl_user_op
def __all_sync(mask, pred, *, loc=None, ip=None) -> Int32:
    """
    Return 1 if the predicate is non-zero for all lanes in ``mask``, else 0.

    Matches CUDA __all_sync (vote.sync.all.pred).
    """
    return Int32(prims.vote_sync(mask, _as_cuda_predicate(pred), "all", loc=loc, ip=ip))


@dsl_user_op
def __ballot_sync(mask, pred, *, loc=None, ip=None) -> Uint32:
    """
    Return the 32-bit mask of lanes in ``mask`` whose predicate is non-zero.

    Matches CUDA __ballot_sync (vote.sync.ballot.b32).
    """
    return Uint32(prims.vote_sync(mask, _as_cuda_predicate(pred), "ballot", loc=loc, ip=ip))


def ballot_sync(mask, pred):
    """
    ``T.ballot_sync``: the ballot mask zero-extended to 64 bits.

    Matches CUDA: (unsigned long long)__ballot_sync(mask, pred)
    """
    return _zext_to_u64(__ballot_sync(mask, pred))


def ballot(pred):
    """
    ``T.ballot``: full-warp ballot zero-extended to 64 bits.

    Matches CUDA: (unsigned long long)__ballot_sync(0xFFFFFFFF, pred)
    """
    return ballot_sync(FULL_MASK, pred)


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


def _as_match_operand(val, *, loc=None, ip=None):
    """
    Pick the match.sync operand width the way CUDA's __match_*_sync overloads do.

    64-bit integers use match.sync.b64; narrower integers and bools are promoted
    to 32 bits (C integer promotion) and use match.sync.b32. float and double
    are compared by bit pattern, as CUDA's floating-point overloads do.
    """
    val = _as_numeric(val)
    if isinstance(val, (bool, int)):
        return Int32(val)
    if isinstance(val, Float64):
        return val.bitcast(Int64, loc=loc, ip=ip)
    if isinstance(val, Float32):
        return val.bitcast(Int32, loc=loc, ip=ip)
    if isinstance(val, Integer) and type(val).width == 64:
        return Int64(val)
    if isinstance(val, Integer) and type(val).width <= 32:
        return Int32(val)
    raise TypeError(f"match.sync expects a 32/64-bit integer, bool, float32 or float64 value, but got {type(val).__name__}")


@dsl_user_op
def __match_any_sync(mask, val, *, loc=None, ip=None) -> Uint32:
    """
    Return a bitmask of lanes in ``mask`` whose value matches the calling lane.

    Matches CUDA __match_any_sync, including the 64-bit overloads.
    """
    return Uint32(prims.match_sync(mask, _as_match_operand(val, loc=loc, ip=ip), "any", loc=loc, ip=ip))


@dsl_user_op
def __match_all_sync(mask, val, *, loc=None, ip=None) -> Uint32:
    """
    Return ``mask`` if every lane in ``mask`` holds the same value, else 0.

    Matches CUDA __match_all_sync, including the 64-bit overloads. The
    predicate output is dropped; it equals ``result != 0``.
    """
    matched, _ = prims.match_sync(mask, _as_match_operand(val, loc=loc, ip=ip), "all", loc=loc, ip=ip)
    return Uint32(matched)


@dsl_user_op
def __ffs(val, *, loc=None, ip=None) -> Int32:
    """
    One-based position of the least significant set bit of a 32-bit value, 0 if none.

    Matches CUDA __ffs with the sequence nvcc emits: bfind.shiftamt yields
    0xFFFFFFFF for a zero input, so the final add gives 0.
    """
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [Int32(_as_numeric(val)).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 %tl_rev;\nbrev.b32 %tl_rev, $1;\nbfind.shiftamt.u32 $0, %tl_rev;\nadd.s32 $0, $0, 1;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def __ffsll(val, *, loc=None, ip=None) -> Int32:
    """
    One-based position of the least significant set bit of a 64-bit value, 0 if none.

    Matches CUDA __ffsll with the sequence nvcc emits: isolate the lowest set
    bit, then 64 - clz, which is 0 for a zero input.
    """
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [Int64(_as_numeric(val)).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b64 %tl_lsb;\nneg.s64 %tl_lsb, $1;\nand.b64 %tl_lsb, $1, %tl_lsb;\nclz.b64 $0, %tl_lsb;\nsub.s32 $0, 64, $0;\n}",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def __fns(mask, base, offset, *, loc=None, ip=None) -> Uint32:
    """
    Position of the n-th set bit of ``mask`` counted from ``base``, or 0xFFFFFFFF.

    Matches CUDA __fns(unsigned mask, unsigned base, int offset): a positive
    ``offset`` searches upwards from ``base``, a negative one downwards, and
    ``offset == 0`` returns ``base`` only if that bit is set.
    """
    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [
                Uint32(_as_numeric(mask)).ir_value(loc=loc, ip=ip),
                Uint32(_as_numeric(base)).ir_value(loc=loc, ip=ip),
                Int32(_as_numeric(offset)).ir_value(loc=loc, ip=ip),
            ],
            "fns.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


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
    """Warp-level parallel reduction: max across all 32 lanes (tl::MaxOp semantics)."""
    from .math import fast_max as tl_max

    value = tl_max(value, _shfl_xor_sync(value, 16))
    value = tl_max(value, _shfl_xor_sync(value, 8))
    value = tl_max(value, _shfl_xor_sync(value, 4))
    value = tl_max(value, _shfl_xor_sync(value, 2))
    value = tl_max(value, _shfl_xor_sync(value, 1))
    return value


def warp_reduce_min(value):
    """Warp-level parallel reduction: min across all 32 lanes (tl::MinOp semantics)."""
    from .math import fast_min as tl_min

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
