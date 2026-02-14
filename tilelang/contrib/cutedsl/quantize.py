# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Quantization/dequantization functions for CuTeDSL backend.
These implement the same functionality as the CUDA templates in tilelang/quantize/lop3.py
using inline PTX via llvm.inline_asm.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm, arith
from cutlass.base_dsl.typing import Uint32, as_numeric
from cutlass.cutlass_dsl import T, dsl_user_op


# Constants for decode operations
BOTTOM_MASK = 0x000f000f
FP16_TOP_MAGIC_NUM = 0x64006400
IMMLUT = (0xf0 & 0xcc) | 0xaa  # = 0xea = 234
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
        N: Number of elements to decode (default 8)
    """
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
    """Decode signed INT4 to FP16."""
    src_u32_ptr = cute.recast_ptr(src_ptr, dtype=cutlass.Uint32)
    src_tensor = cute.make_tensor(src_u32_ptr, (1,))
    i4s = src_tensor[0]
    
    dst_u32_ptr = cute.recast_ptr(dst_ptr, dtype=cutlass.Uint32)
    h = cute.make_tensor(dst_u32_ptr, (N // 2,))
    
    for i in range(N // 2):
        shift = 4 * i
        h[i] = _lop3_sub_f16x2_signed(i4s, shift)
