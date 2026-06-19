"""
Utility functions for CuTeDSL backend.

Provides common helpers used across the CuTeDSL codegen:
bitcast, tensor construction, warp election, barrier sync, and FP16 packing.
"""

import cutlass
import cutlass.cute as cute

from cutlass.base_dsl._mlir_helpers import arith as arith_helper
from cutlass.base_dsl.typing import (
    BFloat16,
    Float,
    Float16,
    Float32,
    Float4E2M1FN,
    Int4,
    Int8,
    Int16,
    Int32,
    Numeric,
    Uint8,
    Uint16,
    Uint32,
)
from cutlass.cute.tensor import TensorSSA
from cutlass._mlir.dialects import arith, builtin, llvm, nvgpu, nvvm, vector
from cutlass._mlir import ir as mlir_ir
from cutlass.cutlass_dsl import dsl_user_op

__all__ = [
    "BYTES_PER_TENSORMAP",
    "BYTES_PER_POINTER",
    "type_map",
    "bitcast",
    "cast_tensor",
    "as_tensor_ssa",
    "as_rmem_tensor",
    "make_filled_tensor",
    "make_tensor_at_offset",
    "handle_add_byte_offset",
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


@dsl_user_op
def _narrow_to_float16_tensor(value, *, loc=None, ip=None):
    src = value.ir_value(loc=loc, ip=ip)
    res_type = arith_helper.recast_type(src.type, Float16.mlir_type)
    res = nvgpu.cvt_fpext(res_type, src, loc=loc, ip=ip)
    return TensorSSA(res, value.shape, Float16)


@dsl_user_op
def _f4e2m1_to_float32_tensor(value, *, loc=None, ip=None):
    length = cute.size(value.shape, loc=loc, ip=ip)
    if not isinstance(length, int):
        raise ValueError("Float4E2M1FN tensor casts require a static vector shape")

    src = value.ir_value(loc=loc, ip=ip)
    vec_i4 = builtin.unrealized_conversion_cast(
        [mlir_ir.VectorType.get([length], Int4.mlir_type, loc=loc)],
        [src],
        loc=loc,
        ip=ip,
    )
    vec_dst_type = mlir_ir.VectorType.get([length], Float32.mlir_type, loc=loc)
    vec_dst = llvm.mlir_zero(vec_dst_type, loc=loc, ip=ip)

    def i32(value):
        return arith.constant(Int32.mlir_type, value, loc=loc, ip=ip)

    zero = i32(0)
    one = i32(1)
    payload_mask = i32(0x7)
    sign_mask = i32(0x8)
    exp_mask = i32(0x3)
    f32_subnormal_abs = i32(0x3F000000)

    for idx in range(length):
        pos = i32(idx)
        nibble_i4 = vector.extractelement(vec_i4, position=pos, loc=loc, ip=ip)
        nibble = arith.extui(Int32.mlir_type, nibble_i4, loc=loc, ip=ip)

        payload = arith.andi(nibble, payload_mask, loc=loc, ip=ip)
        sign = arith.shli(arith.andi(nibble, sign_mask, loc=loc, ip=ip), i32(28), loc=loc, ip=ip)
        exp = arith.andi(arith.shrui(nibble, one, loc=loc, ip=ip), exp_mask, loc=loc, ip=ip)
        mant = arith.andi(nibble, one, loc=loc, ip=ip)

        normal_exp = arith.shli(arith.addi(exp, i32(126), loc=loc, ip=ip), i32(23), loc=loc, ip=ip)
        normal_mant = arith.shli(mant, i32(22), loc=loc, ip=ip)
        normal_abs = arith.ori(normal_exp, normal_mant, loc=loc, ip=ip)
        is_exp_zero = arith.cmpi(arith.CmpIPredicate.eq, exp, zero, loc=loc, ip=ip)
        nonzero_abs = arith.select(is_exp_zero, f32_subnormal_abs, normal_abs, loc=loc, ip=ip)
        is_zero = arith.cmpi(arith.CmpIPredicate.eq, payload, zero, loc=loc, ip=ip)
        abs_bits = arith.select(is_zero, zero, nonzero_abs, loc=loc, ip=ip)
        bits = arith.ori(sign, abs_bits, loc=loc, ip=ip)
        f32 = llvm.bitcast(Float32.mlir_type, bits, loc=loc, ip=ip)
        vec_dst = vector.insertelement(f32, vec_dst, position=pos, loc=loc, ip=ip)

    return TensorSSA(vec_dst, value.shape, Float32)


@dsl_user_op
def _float32_to_f4e2m1_tensor(value, *, loc=None, ip=None):
    length = cute.size(value.shape, loc=loc, ip=ip)
    if not isinstance(length, int):
        raise ValueError("Float4E2M1FN tensor casts require a static vector shape")

    src = value.ir_value(loc=loc, ip=ip)
    vec_i32_type = mlir_ir.VectorType.get([length], Int32.mlir_type, loc=loc)
    vec_i32 = llvm.bitcast(vec_i32_type, src, loc=loc, ip=ip)
    vec_i4_type = mlir_ir.VectorType.get([length], Int4.mlir_type, loc=loc)
    vec_i4 = llvm.mlir_zero(vec_i4_type, loc=loc, ip=ip)

    def i32(value):
        return arith.constant(Int32.mlir_type, value, loc=loc, ip=ip)

    def f32(value):
        return arith.constant(Float32.mlir_type, value, loc=loc, ip=ip)

    def select_payload(abs_val, payload, predicate, threshold, value):
        cond = arith.cmpf(predicate, abs_val, f32(threshold), loc=loc, ip=ip)
        return arith.select(cond, i32(value), payload, loc=loc, ip=ip)

    sign_mask = i32(0x80000000)
    abs_mask = i32(0x7FFFFFFF)
    shift_sign = i32(28)

    for idx in range(length):
        pos = i32(idx)
        bits = vector.extractelement(vec_i32, position=pos, loc=loc, ip=ip)
        sign = arith.shrui(arith.andi(bits, sign_mask, loc=loc, ip=ip), shift_sign, loc=loc, ip=ip)
        abs_bits = arith.andi(bits, abs_mask, loc=loc, ip=ip)
        abs_val = llvm.bitcast(Float32.mlir_type, abs_bits, loc=loc, ip=ip)

        payload = i32(0)
        payload = select_payload(abs_val, payload, arith.CmpFPredicate.OGT, 0.25, 1)
        payload = select_payload(abs_val, payload, arith.CmpFPredicate.OGE, 0.75, 2)
        payload = select_payload(abs_val, payload, arith.CmpFPredicate.OGT, 1.25, 3)
        payload = select_payload(abs_val, payload, arith.CmpFPredicate.OGE, 1.75, 4)
        payload = select_payload(abs_val, payload, arith.CmpFPredicate.OGT, 2.5, 5)
        payload = select_payload(abs_val, payload, arith.CmpFPredicate.OGE, 3.5, 6)
        payload = select_payload(abs_val, payload, arith.CmpFPredicate.OGT, 5.0, 7)
        is_nan = arith.cmpf(arith.CmpFPredicate.UNO, abs_val, abs_val, loc=loc, ip=ip)
        payload = arith.select(is_nan, i32(7), payload, loc=loc, ip=ip)

        nibble = arith.trunci(Int4.mlir_type, arith.ori(sign, payload, loc=loc, ip=ip), loc=loc, ip=ip)
        vec_i4 = vector.insertelement(nibble, vec_i4, position=pos, loc=loc, ip=ip)

    vec_f4_type = mlir_ir.VectorType.get([length], Float4E2M1FN.mlir_type, loc=loc)
    vec_f4 = builtin.unrealized_conversion_cast([vec_f4_type], [vec_i4], loc=loc, ip=ip)
    return TensorSSA(vec_f4, value.shape, Float4E2M1FN)


def cast_tensor(value, dtype):
    if isinstance(value, TensorSSA):
        if value.dtype is dtype:
            return value
        if value.dtype is Float4E2M1FN and dtype in (Float16, Float32):
            f32 = _f4e2m1_to_float32_tensor(value)
            if dtype is Float32:
                return f32
            return f32.to(Float16)
        if dtype is Float4E2M1FN and value.dtype in (BFloat16, Float16, Float32):
            if value.dtype is not Float32:
                value = value.to(Float32)
            return _float32_to_f4e2m1_tensor(value)
        if (
            dtype is Float16
            and isinstance(value.dtype, type)
            and issubclass(value.dtype, Float)
            and getattr(value.dtype, "width", 0) < Float16.width
        ):
            return _narrow_to_float16_tensor(value)
        elem_type = getattr(value.type, "element_type", None)
        if elem_type is None:
            elem_type = getattr(value.ir_value().type, "element_type", None)
        if elem_type == dtype.mlir_type or str(elem_type) == str(dtype.mlir_type):
            return TensorSSA(value, value.shape, dtype)
        return value.to(dtype)
    if isinstance(value, Numeric):
        if type(value) is dtype:
            return value
        return value.to(dtype)
    return dtype(value)


def as_tensor_ssa(value):
    if isinstance(value, TensorSSA):
        return value
    return value.load()


def as_rmem_tensor(value, shape, dtype):
    if not isinstance(value, TensorSSA):
        return value
    tensor = cute.make_rmem_tensor(shape, dtype)
    tensor.store(value)
    return tensor


def make_filled_tensor(shape, value, dtype=None):
    dtype = dtype or type(value)
    if dtype is Float4E2M1FN:
        if value != 0:
            raise ValueError("Float4E2M1FN filled tensors only support zero values")
        storage_layout = cute.recast_layout(8, Float4E2M1FN.width, cute.make_layout(shape))
        storage = cute.make_rmem_tensor(storage_layout.shape, Uint8)
        storage.fill(Uint8(0))
        return cute.recast_tensor(storage, Float4E2M1FN)

    t = cute.make_rmem_tensor(shape, dtype)
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


def handle_add_byte_offset(handle, byte_offset):
    ptr = handle.iterator if hasattr(handle, "iterator") else handle
    byte_ptr = cute.recast_ptr(ptr, dtype=cutlass.Uint8)
    return make_tensor_at_offset(byte_ptr, byte_offset, (1,))


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
