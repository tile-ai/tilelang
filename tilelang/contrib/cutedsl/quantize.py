# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Quantization/dequantization functions for CuTeDSL backend.
These implement the same functionality as the CUDA templates in tilelang/quantize/lop3.py
using inline PTX via llvm.inline_asm.
"""

__all__ = [
    "BOTTOM_MASK",
    "FP16_TOP_MAGIC_NUM",
    "IMMLUT",
    "MEDIAN_NUM_UNSIGNED",
    "MEDIAN_NUM_SIGNED",
    "decode_i4u_to_f16",
    "decode_i4s_to_f16",
    "decode_fp4_to_bf16_twiddling",
]

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm, arith
from cutlass.base_dsl.typing import Uint32
from cutlass.cutlass_dsl import T, dsl_user_op


# Constants for decode operations
BOTTOM_MASK = 0x000F000F
FP16_TOP_MAGIC_NUM = 0x64006400
IMMLUT = (0xF0 & 0xCC) | 0xAA  # = 0xea = 234
MEDIAN_NUM_UNSIGNED = 0x64006400
MEDIAN_NUM_SIGNED = 0x64086408


@dsl_user_op
def _lop3_sub_f16x2_unsigned(i4s: Uint32, shift: int, *, loc=None, ip=None) -> Uint32:
    """
    LOP3 + sub.f16x2 for unsigned i4 to f16x2 decode.

    PTX equivalent:
        shr.b32 shifted, i4s, shift;
        lop3.b32 h, shifted, BOTTOM_MASK, FP16_TOP_MAGIC_NUM, immLut;
        sub.f16x2 h, h, MEDIAN_NUM;

    Note: lop3.b32's immLut must be an immediate constant.
          sub.f16x2 requires register operands, not immediates.
    """
    # Shift right
    shifted = Uint32(arith.shrui(Uint32(i4s).ir_value(), Uint32(shift).ir_value()))

    # LOP3 + sub in single asm block
    # immLut=234 (0xea), BOTTOM_MASK=0x000f000f, FP16_TOP_MAGIC_NUM=0x64006400
    # MEDIAN_NUM_UNSIGNED=0x64006400
    result = Uint32(
        llvm.inline_asm(
            T.i32(),
            [shifted.ir_value()],
            "{ .reg .b32 tmp, median; "
            "lop3.b32 tmp, $1, 0x000f000f, 0x64006400, 0xea; "
            "mov.b32 median, 0x64006400; "
            "sub.f16x2 $0, tmp, median; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
    return result


@dsl_user_op
def _lop3_sub_f16x2_signed(i4s: Uint32, shift: int, *, loc=None, ip=None) -> Uint32:
    """LOP3 + sub.f16x2 for signed i4 to f16x2 decode."""
    shifted = Uint32(arith.shrui(Uint32(i4s).ir_value(), Uint32(shift).ir_value()))

    # MEDIAN_NUM_SIGNED=0x64086408
    result = Uint32(
        llvm.inline_asm(
            T.i32(),
            [shifted.ir_value()],
            "{ .reg .b32 tmp, median; "
            "lop3.b32 tmp, $1, 0x000f000f, 0x64006400, 0xea; "
            "mov.b32 median, 0x64086408; "
            "sub.f16x2 $0, tmp, median; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
    return result


def decode_i4u_to_f16(src_ptr, dst_ptr, N: int = 8):
    """
    Decode unsigned INT4 to FP16.

    Equivalent to CUDA template:
        decode_i4b_to_f16<T1, T2, false>(_i4u, B_local_decode, N);

    Args:
        src_ptr: Pointer to packed INT4 data (4 bytes for 8 elements)
        dst_ptr: Pointer to FP16 output (16 bytes for 8 elements)
        N: Number of elements to decode (default 8, must be even)
    """
    assert N % 2 == 0, f"N must be even for i4-to-f16 decode, got {N}"
    # Load packed i4 values (32 bits = 8 x 4-bit values)
    # Use make_tensor to create a tensor view, then access element
    src_u32_ptr = cute.recast_ptr(src_ptr, dtype=cutlass.Uint32)
    src_tensor = cute.make_tensor(src_u32_ptr, (1,))
    i4s = src_tensor[0]

    # Output as uint32 pairs (each holds 2 x f16)
    dst_u32_ptr = cute.recast_ptr(dst_ptr, dtype=cutlass.Uint32)
    h = cute.make_tensor(dst_u32_ptr, (N // 2,))

    # Decode 2 elements at a time (N/2 iterations)
    # Note: N must be compile-time constant for unrolling
    for i in range(N // 2):
        shift = 4 * i
        h[i] = _lop3_sub_f16x2_unsigned(i4s, shift)


def decode_i4s_to_f16(src_ptr, dst_ptr, N: int = 8):
    """Decode signed INT4 to FP16. N must be even."""
    assert N % 2 == 0, f"N must be even for i4-to-f16 decode, got {N}"
    src_u32_ptr = cute.recast_ptr(src_ptr, dtype=cutlass.Uint32)
    src_tensor = cute.make_tensor(src_u32_ptr, (1,))
    i4s = src_tensor[0]

    dst_u32_ptr = cute.recast_ptr(dst_ptr, dtype=cutlass.Uint32)
    h = cute.make_tensor(dst_u32_ptr, (N // 2,))

    for i in range(N // 2):
        shift = 4 * i
        h[i] = _lop3_sub_f16x2_signed(i4s, shift)


@dsl_user_op
def _fp4_to_bf16_twiddling_r0(inp: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert 8 FP4 to 8 BF16, return output 0."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(inp).ir_value()],
            "{ .reg .b32 tmp, bias; "
            "prmt.b32 tmp, $1, 0, 0x0123; "
            "mov.b32 bias, 0x7e807e80; "
            "and.b32 $0, tmp, 0b10000001110000001000000111000000; "
            "mul.bf16x2 $0, $0, bias; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fp4_to_bf16_twiddling_r1(inp: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert 8 FP4 to 8 BF16, return output 1."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(inp).ir_value()],
            "{ .reg .b32 tmp, bias, d0; "
            "prmt.b32 tmp, $1, 0, 0x0123; "
            "mov.b32 bias, 0x7e807e80; "
            "shl.b32 d0, tmp, 3; "
            "and.b32 $0, d0, 0b10000001110000001000000111000000; "
            "mul.bf16x2 $0, $0, bias; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fp4_to_bf16_twiddling_r2(inp: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert 8 FP4 to 8 BF16, return output 2."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(inp).ir_value()],
            "{ .reg .b32 tmp, bias, d0; "
            "prmt.b32 tmp, $1, 0, 0x0123; "
            "mov.b32 bias, 0x7e807e80; "
            "shl.b32 d0, tmp, 6; "
            "and.b32 $0, d0, 0b10000001110000001000000111000000; "
            "mul.bf16x2 $0, $0, bias; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fp4_to_bf16_twiddling_r3(inp: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert 8 FP4 to 8 BF16, return output 3."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(inp).ir_value()],
            "{ .reg .b32 tmp, bias, d0, d1, d2, d3, d4, d5; "
            "prmt.b32 tmp, $1, 0, 0x0123; "
            "mov.b32 bias, 0x7e807e80; "
            "shl.b32 d0, tmp, 1; "
            "and.b32 d1, d0, 0b10000000000000001000000000000000; "
            "shr.b32 d2, tmp, 3; "
            "and.b32 d3, d2, 0b00000001100000000000000110000000; "
            "or.b32 d4, d1, d3; "
            "shr.b32 d0, tmp, 7; "
            "and.b32 d5, d0, 0b00000000010000000000000001000000; "
            "or.b32 $0, d4, d5; "
            "mul.bf16x2 $0, $0, bias; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_bf16_high(r0: Uint32, r1: Uint32, *, loc=None, ip=None) -> Uint32:
    """Pack high 16-bits of r0 and r1 into one uint32."""
    # r0[31:16] -> result[15:0], r1[31:16] -> result[31:16]
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(r0).ir_value(), Uint32(r1).ir_value()],
            "prmt.b32 $0, $1, $2, 0x7632;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_bf16_low(r0: Uint32, r1: Uint32, *, loc=None, ip=None) -> Uint32:
    """Pack low 16-bits of r0 and r1 into one uint32."""
    # r0[15:0] -> result[15:0], r1[15:0] -> result[31:16]
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(r0).ir_value(), Uint32(r1).ir_value()],
            "prmt.b32 $0, $1, $2, 0x5410;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def decode_fp4_to_bf16_twiddling(src_ptr, dst_ptr, N: int = 8):
    """
    Decode FP4 to BF16 using twiddling technique.

    Reference: triton/tensor_details/layout_details/hopper_value.py

    For each iteration:
        - Input: 4 bytes (uint32) = 8 FP4 values
        - Output: 8 BF16 values (16 bytes)

    C code output layout:
        B_local_decode[(i << 3) + j] = vec[j].high  (j=0..3)
        B_local_decode[(i << 3) + j + 4] = vec[j].low  (j=0..3)

    So output as uint32:
        dst[i*4 + 0] = {r1.high, r0.high}
        dst[i*4 + 1] = {r3.high, r2.high}
        dst[i*4 + 2] = {r1.low, r0.low}
        dst[i*4 + 3] = {r3.low, r2.low}

    Args:
        src_ptr: Pointer to packed FP4 data
        dst_ptr: Pointer to BF16 output
        N: Number of iterations (default 8, processing 64 FP4 -> 64 BF16)
    """
    # Input: read as uint32 (4 bytes at a time)
    src_u32_ptr = cute.recast_ptr(src_ptr, dtype=cutlass.Uint32)
    src_tensor = cute.make_tensor(src_u32_ptr, (N,))

    # Output: write as uint32 (each holds 2 BF16)
    dst_u32_ptr = cute.recast_ptr(dst_ptr, dtype=cutlass.Uint32)
    dst_tensor = cute.make_tensor(dst_u32_ptr, (N * 4,))  # N iterations * 4 outputs each

    for i in range(N):
        inp = src_tensor[i]
        r0 = _fp4_to_bf16_twiddling_r0(inp)
        r1 = _fp4_to_bf16_twiddling_r1(inp)
        r2 = _fp4_to_bf16_twiddling_r2(inp)
        r3 = _fp4_to_bf16_twiddling_r3(inp)

        # Pack high and low halves according to C output layout
        base = i * 4
        dst_tensor[base + 0] = _pack_bf16_high(r0, r1)  # {r1.high, r0.high}
        dst_tensor[base + 1] = _pack_bf16_high(r2, r3)  # {r3.high, r2.high}
        dst_tensor[base + 2] = _pack_bf16_low(r0, r1)  # {r1.low, r0.low}
        dst_tensor[base + 3] = _pack_bf16_low(r2, r3)  # {r3.low, r2.low}
