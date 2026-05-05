from typing import Literal
from tilelang import language as T

# Implementation asm for fp4 to bf16, using twiddling
# Reference: https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/tensor_details/layout_details/hopper_value.py#L11-L18
decode_f4_to_bf16_twiddling = """
// N should be the number of elements processed by one thread
template<typename T1, typename T2>
__device__ void decode_fp4_to_bf16_twiddling(T1 *B_local, T2 *B_local_decode, const int N = 8) {
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    uint B_dequantize_local_vec[4];
    uint tmp, bias, d0, d1, d2, d3, d4, d5, d6;
    asm volatile(
      // To handle the endianness issue
      "prmt.b32 %13, %4, 0, 0x0123;"
      "mov.b32 %12, 0x7e807e80;"
      "and.b32 %0, %13, 0b10000001110000001000000111000000;"
      "mul.bf16x2 %0, %0, %12;"
      "shl.b32 %1, %13, 3;"
      "and.b32 %1, %1, 0b10000001110000001000000111000000;"
      "mul.bf16x2 %1, %1, %12;"
      "shl.b32 %2, %13, 6;"
      "and.b32 %2, %2, 0b10000001110000001000000111000000;"
      "mul.bf16x2 %2, %2, %12;"
      "shl.b32 %5, %13, 1;"
      "and.b32 %6, %5, 0b10000000000000001000000000000000;"
      "shr.b32 %7, %13, 3;"
      "and.b32 %8, %7, 0b00000001100000000000000110000000;"
      "or.b32 %9, %6, %8;"
      "shr.b32 %10, %13, 7;"
      "and.b32 %11, %10, 0b00000000010000000000000001000000;"
      "or.b32 %3, %9, %11;"
      "mul.bf16x2 %3, %3, %12;"
      :"=r"(B_dequantize_local_vec[0])
      ,"=r"(B_dequantize_local_vec[1])
      ,"=r"(B_dequantize_local_vec[2])
      ,"=r"(B_dequantize_local_vec[3])
      :"r"(*(uint*)&B_local[i << 2]), "r"(d0), "r"(d1), "r"(d2), "r"(d3), "r"(d4), "r"(d5), "r"(d6), "r"(bias), "r"(tmp)
    );
    for (int j = 0; j < 4; ++j) {
      // Pay attention to the big-endianness issue
      B_local_decode[(i << 3) + j] = reinterpret_cast<T2*>(&B_dequantize_local_vec[j])[1];
      B_local_decode[(i << 3) + j + 4] = reinterpret_cast<T2*>(&B_dequantize_local_vec[j])[0];
    }
  }
  // Check if the synchronization is needed
}
"""


# AMD HIP version of fp4->bf16 twiddling dequantization (gfx950 / CDNA4).
# Implements the same bit-manipulation algorithm as the CUDA PTX version but
# using portable C++ (no PTX inline assembly) so it compiles with HIP/clang.
#
# The algorithm (matching the CUDA PTX decode_fp4_to_bf16_twiddling above):
#   1. byte-reverse the 32-bit packed word (endianness compensation)
#   2. extract 8 FP4 E2M1 nibbles
#   3. map each nibble to BF16 bits by bit-field placement
#   4. multiply by the bias constant 0x7e80 (BF16 representation of 2^126*…)
# AMD gfx950 / HIP version of fp4->bf16 twiddling dequantization.
# Uses the same bit-manipulation algorithm as the CUDA PTX version.
# BF16 multiplication is done via bfloat16_t (= hip_bfloat16, defined in
# tl_templates/hip/common.h included by all TileLang HIP kernels), which
# supports implicit float conversion so no external bf16 API headers are needed.
# Bias constant 0x7e807e80 = two packed BF16 words each equal to 2^126.
decode_f4_to_bf16_twiddling_hip = """
// N = number of 4-element groups (4 packed bytes = 8 FP4 values each)
// This implementation uses only standard C++ types (uint16_t, uint32_t, float)
// so it compiles without any HIP type headers.
template<typename T1, typename T2>
__device__ void decode_fp4_to_bf16_twiddling(T1 *B_local, T2 *B_local_decode, const int N = 8) {
  // Multiply two packed BF16 values stored as uint16 each.
  // BF16 layout: [sign(1)|exp(8)|mant(7)] -- upper 16 bits of IEEE float32.
  // We convert to float, multiply, then convert back via bit manipulation.
  auto bf16_to_float = [](uint16_t b) -> float {
    uint32_t f = (uint32_t)b << 16u;
    float r;
    __builtin_memcpy(&r, &f, 4);
    return r;
  };
  auto float_to_bf16 = [](float f) -> uint16_t {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16u);
  };
  // Multiply two packed uint32 BF16x2 words element-wise.
  auto bf16x2_mul = [&](uint32_t a, uint32_t b) -> uint32_t {
    uint16_t alo = (uint16_t)(a & 0xFFFFu), ahi = (uint16_t)(a >> 16u);
    uint16_t blo = (uint16_t)(b & 0xFFFFu), bhi = (uint16_t)(b >> 16u);
    uint16_t rlo = float_to_bf16(bf16_to_float(alo) * bf16_to_float(blo));
    uint16_t rhi = float_to_bf16(bf16_to_float(ahi) * bf16_to_float(bhi));
    return (uint32_t)rlo | ((uint32_t)rhi << 16u);
  };

  #pragma unroll
  for (int i = 0; i < N; ++i) {
    uint32_t packed;
    __builtin_memcpy(&packed, (const uint8_t*)B_local + (i << 2), 4);

    // Byte-reverse (endianness compensation).
    uint32_t tmp = ((packed & 0xFFu) << 24u)
                 | (((packed >> 8u)  & 0xFFu) << 16u)
                 | (((packed >> 16u) & 0xFFu) << 8u)
                 |  ((packed >> 24u) & 0xFFu);

    // bias = 0x7e80_7e80 = two packed BF16 words each equal to 2^126.
    const uint32_t bias = 0x7e807e80u;
    // Mask for sign+exp[1] bits in each packed BF16 pair: 0b10000001_11000000_10000001_11000000
    const uint32_t mask_e = 0x81C081C0u;

    uint32_t d[4];
    d[0] = bf16x2_mul(tmp & mask_e, bias);
    d[1] = bf16x2_mul((tmp << 3u) & mask_e, bias);
    d[2] = bf16x2_mul((tmp << 6u) & mask_e, bias);
    {
      // Mantissa bits (from CUDA: shl.b32+and combos for each nibble position)
      uint32_t t1 = (tmp << 1u) & 0x80008000u;
      uint32_t t2 = (tmp >> 3u) & 0x01800180u;
      uint32_t t3 = (tmp >> 7u) & 0x00400040u;
      d[3] = bf16x2_mul(t1 | t2 | t3, bias);
    }

    // Store 8 BF16 results (big-endian nibble order matching CUDA reference).
    for (int j = 0; j < 4; ++j) {
      reinterpret_cast<T2*>(B_local_decode)[(i << 3) + j]     = reinterpret_cast<T2*>(&d[j])[1];
      reinterpret_cast<T2*>(B_local_decode)[(i << 3) + j + 4] = reinterpret_cast<T2*>(&d[j])[0];
    }
  }
}
"""

# Simple (non-twiddling) AMD gfx950 FP4->BF16 dequantization via float LUT.
# Uses a static lookup table to avoid dependency on FP4 hardware intrinsics.
# This is the fallback path when use_twiddling=False on AMD.
decode_f4_to_bf16_simple_hip = """
template<typename T1, typename T2>
__device__ void decode_fp4_to_bf16(T1 *B_local, T2 *B_local_decode, const int N = 8) {
  // FP4 E2M1 lookup: nibble index -> BF16 value (via float).
  // Nibble layout: bit3=sign, bits[2:1]=exp, bit0=mant.
  static const float fp4_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    const uint8_t* src = (const uint8_t*)B_local + (i << 2);
    // Each byte holds 2 nibbles: low nibble first, high nibble second.
    // Output order matches the CUDA twiddling reference (interleaved by 4).
    for (int b = 0; b < 4; ++b) {
      uint8_t byte = src[b];
      reinterpret_cast<T2*>(B_local_decode)[(i << 3) + b]     = (T2)fp4_lut[byte & 0xFu];
      reinterpret_cast<T2*>(B_local_decode)[(i << 3) + b + 4] = (T2)fp4_lut[(byte >> 4u) & 0xFu];
    }
  }
}
"""


def get_mxfp_intrin_group(
    out_dtype: Literal[T.float16, T.bfloat16] = T.bfloat16,
    source_format: Literal[T.int, T.uint] = T.uint,
    source_bit: int = 4,
    storage_dtype: Literal[T.int32, T.int8, T.uint8] = T.uint8,
    use_twiddling: bool = False,
    target=None,
) -> dict[str, str]:
    """
    Return metadata for an MXFP decoding intrinsic: function name and C source string.

    Validates the requested output dtype, source format, and storage dtype, then constructs
    a lookup key of the form `fp{source_bit}_to_{f16|bf16}` (appending `_twiddling` when
    use_twiddling is True) to select the corresponding C source snippet and a matching
    function name `decode_fp{source_bit}_to_{f16|bf16}` (also optionally suffixed with
    `_twiddling`).

    Parameters:
        out_dtype: Target floating-point type for decoded values; either T.float16 or T.bfloat16.
        source_format: Integer source representation; "int" or "uint".
        source_bit: Bit width of the packed source format (e.g., 4).
        storage_dtype: Underlying storage integer dtype (one of T.int32, T.int8, T.uint8).
        use_twiddling: When True, select the twiddling variant of the decoding intrinsic.

    Returns:
        A dict with:
          - "func_name": the generated C function name string for the requested decode intrinsic.
          - "c_source": the C source string for that intrinsic.

    Raises:
        AssertionError: if out_dtype, source_format, or storage_dtype are not supported.
        KeyError: if the constructed key does not match any available C source implementation.
    """
    out_dtype, source_format, storage_dtype = T.dtype(out_dtype), T.dtype(source_format), T.dtype(storage_dtype)
    assert out_dtype in [T.float16, T.bfloat16], f"Invalid out_dtype: {out_dtype}. Expected 'float16' or 'bfloat16'."
    assert source_format in [T.int, T.uint], f"Invalid source_format: {source_format}. Expected 'int' or 'uint'."
    assert storage_dtype in [T.int32, T.int8, T.uint8], f"Invalid storage_dtype: {storage_dtype}. Expected 'int32' or 'int8' or 'uint8'."

    # Detect AMD gfx950 target to select the HIP C++ dequantization implementation.
    # All other targets (NV, RDNA, MI300) use the default CUDA PTX path below.
    _is_gfx950 = False
    if target is not None:
        try:
            from tilelang.utils.target import target_is_gfx950

            _is_gfx950 = target_is_gfx950(target)
        except Exception:
            pass

    dtype_map = {T.float16: "f16", T.bfloat16: "bf16"}
    func_name = f"decode_fp{source_bit}_to_{dtype_map[out_dtype]}"
    if use_twiddling:
        func_name += "_twiddling"

    if _is_gfx950:
        # AMD gfx950 path: use portable HIP C++ implementations.
        # The function name stays the same so the call site is unchanged.
        if use_twiddling and source_bit == 4 and out_dtype == T.bfloat16:
            return {"func_name": func_name, "c_source": decode_f4_to_bf16_twiddling_hip}
        elif not use_twiddling and source_bit == 4 and out_dtype == T.bfloat16:
            return {"func_name": func_name, "c_source": decode_f4_to_bf16_simple_hip}
        else:
            raise AssertionError(
                f"AMD gfx950 MXFP dequant only supports source_bit=4 and out_dtype=bfloat16, "
                f"got source_bit={source_bit}, out_dtype={out_dtype}"
            )

    # CUDA / default path: use PTX inline assembly implementations.
    key = f"fp{source_bit}_to_{dtype_map[out_dtype]}"
    if use_twiddling:
        key += "_twiddling"

    import_c_map = {
        "fp4_to_bf16_twiddling": decode_f4_to_bf16_twiddling,
    }

    return {
        "func_name": func_name,
        "c_source": import_c_map[key],
    }
