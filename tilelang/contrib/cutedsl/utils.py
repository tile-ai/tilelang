"""
Utility functions for CuTeDSL backend.

Provides common helpers used across the CuTeDSL codegen:
bitcast, tensor construction, warp election, barrier sync, and FP16 packing.
"""

import cutlass
import cutlass.cute as cute

from cutlass.base_dsl.typing import Int8, Int16, Int32, Uint8, Uint16, Uint32, Float16, Float32, BFloat16
from cutlass._mlir.dialects import llvm, nvvm
from cutlass._mlir import ir as mlir_ir
from cutlass.cutlass_dsl import dsl_user_op

__all__ = [
    "BYTES_PER_TENSORMAP",
    "BYTES_PER_POINTER",
    "type_map",
    "bitcast",
    "make_filled_tensor",
    "make_tensor_at_offset",
    "shuffle_elect",
    "sync_thread_partial",
    "pack_half2",
]

BYTES_PER_TENSORMAP = 128
BYTES_PER_POINTER = 8

# Map dtype to WGMMA types (moved from typing.py)
type_map = {
    "int8": nvvm.WGMMATypes.s8,
    "int32": nvvm.WGMMATypes.s32,
    "uint8": nvvm.WGMMATypes.u8,
    "float16": nvvm.WGMMATypes.f16,
    "fp16": nvvm.WGMMATypes.f16,
    "bfloat16": nvvm.WGMMATypes.bf16,
    "bf16": nvvm.WGMMATypes.bf16,
    "float32": nvvm.WGMMATypes.f32,
    "fp32": nvvm.WGMMATypes.f32,
    "tf32": nvvm.WGMMATypes.tf32,
    "float8_e4m3": nvvm.WGMMATypes.e4m3,
    "float8_e5m2": nvvm.WGMMATypes.e5m2,
    "float8_e4m3fn": nvvm.WGMMATypes.e4m3,
    "e4m3": nvvm.WGMMATypes.e4m3,
    "e5m2": nvvm.WGMMATypes.e5m2,
}

# Map dtype to CuTeDSL type (internal)
_DTYPE_TO_CUTEDSL_TYPE = {
    "int8": Int8,
    "int16": Int16,
    "int32": Int32,
    "uint8": Uint8,
    "uint16": Uint16,
    "uint32": Uint32,
    "float16": Float16,
    "float32": Float32,
    "bfloat16": BFloat16,
}


def bitcast(value, target_dtype):
    """
    Reinterpret the bits of a value as a different type.
    Equivalent to C's (*(target_type *)(&value)).

    Args:
        value: Source value (Numeric type from CuTeDSL)
        target_dtype: Target type (CuTeDSL type like Int8, Float16, etc.)

    Returns:
        Value reinterpreted as target type
    """
    # Get the target MLIR type
    if isinstance(target_dtype, type) or hasattr(target_dtype, "mlir_type"):
        tgt_mlir_type = target_dtype.mlir_type
        tgt_wrapper = target_dtype
    else:
        # Assume it's a string like "int8", "float16", etc.
        tgt_wrapper = _DTYPE_TO_CUTEDSL_TYPE.get(str(target_dtype))
        if tgt_wrapper is None:
            raise ValueError(f"Unknown target dtype: {target_dtype}")
        tgt_mlir_type = tgt_wrapper.mlir_type

    @dsl_user_op
    def bitcast_impl(src_val, *, loc=None, ip=None):
        src_ir = src_val.ir_value(loc=loc, ip=ip) if hasattr(src_val, "ir_value") else src_val
        result = llvm.bitcast(tgt_mlir_type, src_ir, loc=loc, ip=ip)
        return tgt_wrapper(result)

    return bitcast_impl(value)


def make_filled_tensor(shape, value):
    t = cute.make_rmem_tensor(shape, type(value))
    t.fill(value)
    return t


def make_tensor_at_offset(ptr: cute.Pointer, offset, shape, div_by=1):
    from cutlass.cute.typing import is_integer as cute_is_integer

    # Ensure offset is a cute-compatible integer.  Complex arithmetic
    # (e.g. cutlass.Int64 mixed ops) can produce ArithValue / Float types
    # that Pointer.__add__ -> _pack_int_tuple doesn't accept.
    if not isinstance(offset, int) and not cute_is_integer(offset):
        offset = cutlass.Int64(offset)
    if div_by != 1:
        offset = cute.assume(cutlass.as_numeric(offset), divby=div_by)
    return cute.make_tensor(ptr + offset, shape)


def shuffle_elect(thread_extent):
    # thread_extent is the number of threads of a warpgroup
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    if thread_extent == 0:
        return warp_idx == 0
    else:
        return (warp_idx % (thread_extent // 32)) == 0


def sync_thread_partial(barrier_id=None, thread_count=None):
    from .reduce import bar_sync_ptx

    bar_sync_ptx(barrier_id, thread_count)


def pack_half2(x, y):
    """
    Pack two half-precision (fp16) values into a single 32-bit value.
    Corresponds to CUDA's __pack_half2 intrinsic.

    This packs two fp16 values into a single int32 by treating the fp16 bits
    as raw data and concatenating them.
    """

    @dsl_user_op
    def pack_half2_impl(x_val, y_val, *, loc=None, ip=None):
        # Cast fp16 to uint16 (bitcast)
        x_ir = x_val.ir_value(loc=loc, ip=ip) if hasattr(x_val, "ir_value") else x_val
        y_ir = y_val.ir_value(loc=loc, ip=ip) if hasattr(y_val, "ir_value") else y_val

        # Bitcast fp16 to i16
        i16_type = mlir_ir.IntegerType.get_signless(16)
        x_i16 = llvm.bitcast(i16_type, x_ir, loc=loc, ip=ip)
        y_i16 = llvm.bitcast(i16_type, y_ir, loc=loc, ip=ip)

        packed_xy = llvm.inline_asm(
            Int32.mlir_type,
            [x_i16, y_i16],
            "mov.b32 $0, {$1, $2};",
            "=r,h,h",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

        return Int32(packed_xy)

    return pack_half2_impl(x, y)
