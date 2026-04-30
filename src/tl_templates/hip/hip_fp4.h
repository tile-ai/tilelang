#pragma once

#include "common.h"

// FP4 E2M1 support for AMD gfx950 (CDNA4 / MI350).
// All device types and conversion helpers are guarded by __gfx950__ so that
// this header is safe to include on any ROCm target but only activates on
// CDNA4.  The CUDA equivalent is tl_templates/cuda/cuda_fp4.h.
#if defined(__gfx950__)

#include <stdint.h>

// ---------------------------------------------------------------------------
// Scalar FP4 type  (fp4_e2_t)
// Stores one E2M1 value in the low 4 bits of a uint8_t.
// Layout: bit3 = sign, bits[2:1] = exponent, bit0 = mantissa.
// ---------------------------------------------------------------------------
struct fp4_e2_t {
  uint8_t __x;  // only low 4 bits are used

  TL_DEVICE fp4_e2_t() = default;
  TL_DEVICE explicit fp4_e2_t(uint8_t raw) : __x(raw & 0x0Fu) {}

  // Convert FP4 E2M1 to float (pure bit manipulation, no hardware intrinsic).
  // E2M1 encoding: value = (-1)^s * 2^(e-1) * (1 + m*0.5)  for e != 0
  //                value = (-1)^s * 0.5 * m                 for e == 0
  TL_DEVICE operator float() const {
    uint8_t bits = __x & 0x0Fu;
    if (bits == 0u) return 0.0f;
    uint32_t sign = (bits >> 3u) & 0x1u;
    uint32_t exp  = (bits >> 1u) & 0x3u;
    uint32_t mant = bits & 0x1u;
    float result;
    if (exp == 0u) {
      // Denormal: value = (-1)^s * 2^(-1) * (0 + m*0.5) = (-1)^s * m * 0.25
      result = mant ? 0.25f : 0.0f;
    } else {
      // Normal: value = (-1)^s * 2^(e-1) * (1 + m*0.5)
      float mantissa = 1.0f + mant * 0.5f;
      float scale = 1.0f;
      int e = (int)exp - 1;
      if (e >= 0) { for (int i = 0; i < e; ++i) scale *= 2.0f; }
      else        { scale = 0.5f; }
      result = mantissa * scale;
    }
    return sign ? -result : result;
  }

  TL_DEVICE operator half_t() const { return (half_t)(float)(*this); }
  TL_DEVICE operator bfloat16_t() const { return (bfloat16_t)(float)(*this); }
};

// Convert float to FP4 E2M1 (round to nearest, saturate).
TL_DEVICE fp4_e2_t __tl_float_to_fp4(float x) {
  // FP4 E2M1 representable values (positive):
  //   0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
  const float fp4_max = 6.0f;
  const float fp4_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  uint8_t sign = 0u;
  if (x < 0.0f) { sign = 1u; x = -x; }
  if (x > fp4_max) x = fp4_max;
  // Find the closest representable value by brute-force over 8 candidates.
  uint8_t best = 0u;
  float best_diff = x;  // diff from 0
  for (uint8_t i = 1u; i < 8u; ++i) {
    float diff = x - fp4_vals[i];
    if (diff < 0.0f) diff = -diff;
    if (diff < best_diff) { best_diff = diff; best = i; }
  }
  // Encode: bit3=sign, bits[2:1]=exp, bit0=mant
  uint8_t enc = (uint8_t)((sign << 3u) | best);
  fp4_e2_t r;
  r.__x = enc;
  return r;
}

// ---------------------------------------------------------------------------
// Packed 2xFP4 type  (fp4_e2_2_t)
// Two FP4 values stored in one byte: low nibble = first, high nibble = second.
// ---------------------------------------------------------------------------
class fp4_e2_2_t {
public:
  uint8_t __x;  // packed storage

  TL_DEVICE fp4_e2_2_t() = default;
  TL_DEVICE explicit fp4_e2_2_t(uint8_t data) : __x(data) {}

  TL_DEVICE fp4_e2_t x() const { return fp4_e2_t(__x & 0x0Fu); }
  TL_DEVICE fp4_e2_t y() const { return fp4_e2_t((__x >> 4u) & 0x0Fu); }

  TL_DEVICE void set_x(fp4_e2_t val) { __x = (__x & 0xF0u) | (val.__x & 0x0Fu); }
  TL_DEVICE void set_y(fp4_e2_t val) {
    __x = (__x & 0x0Fu) | ((val.__x & 0x0Fu) << 4u);
  }
};

// ---------------------------------------------------------------------------
// Vector FP4 types  (fp4_e2_4_t  ..  fp4_e2_32_t)
// Each stores 2*N elements in N bytes via nested fp4_e2_2_t.
// ---------------------------------------------------------------------------
struct __attribute__((aligned(2))) fp4_e2_4_t {
  fp4_e2_2_t x;
  fp4_e2_2_t y;
};

struct __attribute__((aligned(4))) fp4_e2_8_t {
  fp4_e2_4_t x;
  fp4_e2_4_t y;
};

struct __attribute__((aligned(8))) fp4_e2_16_t {
  fp4_e2_8_t x;
  fp4_e2_8_t y;
};

struct __attribute__((aligned(16))) fp4_e2_32_t {
  fp4_e2_16_t x;
  fp4_e2_16_t y;
};

// ---------------------------------------------------------------------------
// Pack helpers
// ---------------------------------------------------------------------------
TL_DEVICE fp4_e2_2_t make_fp4_e2_2_t(fp4_e2_t x, fp4_e2_t y) {
  return fp4_e2_2_t((uint8_t)((x.__x & 0x0Fu) | ((y.__x & 0x0Fu) << 4u)));
}

TL_DEVICE fp4_e2_4_t make_fp4_e2_4_t(fp4_e2_t x0, fp4_e2_t x1,
                                      fp4_e2_t x2, fp4_e2_t x3) {
  fp4_e2_4_t r;
  r.x = make_fp4_e2_2_t(x0, x1);
  r.y = make_fp4_e2_2_t(x2, x3);
  return r;
}

TL_DEVICE fp4_e2_8_t make_fp4_e2_8_t(fp4_e2_t x0, fp4_e2_t x1,
                                      fp4_e2_t x2, fp4_e2_t x3,
                                      fp4_e2_t x4, fp4_e2_t x5,
                                      fp4_e2_t x6, fp4_e2_t x7) {
  fp4_e2_8_t r;
  r.x = make_fp4_e2_4_t(x0, x1, x2, x3);
  r.y = make_fp4_e2_4_t(x4, x5, x6, x7);
  return r;
}

// ---------------------------------------------------------------------------
// FP4 <-> Half2 conversions
// half2 on HIP is __hip_fp16x2 / float16x2 but is accessed as uint1 (packed).
// We work through float as the intermediate type for correctness.
// ---------------------------------------------------------------------------

// fp4x2 (1 packed byte) -> 2 x half_t, returned as uint1 (HIP half2 storage)
TL_DEVICE uint1 __tl_cvt_fp4x2_to_half2(uint8_t src) {
  fp4_e2_2_t packed(src);
  half_t lo = (half_t)(float)packed.x();
  half_t hi = (half_t)(float)packed.y();
  return uint1{__pack_half2(lo, hi)};
}

// 2 x half_t (as uint1) -> fp4x2 packed byte
TL_DEVICE uint8_t __tl_cvt_half2_to_fp4x2(uint1 src) {
  half_t lo, hi;
  // unpack via reinterpret: HIP stores half2 as two consecutive 16-bit values
  const uint32_t raw = src.x;
  lo = *reinterpret_cast<const half_t *>(&raw);
  const uint16_t raw_hi = (uint16_t)(raw >> 16u);
  hi = *reinterpret_cast<const half_t *>(&raw_hi);
  fp4_e2_t fp4_lo = __tl_float_to_fp4((float)lo);
  fp4_e2_t fp4_hi = __tl_float_to_fp4((float)hi);
  return make_fp4_e2_2_t(fp4_lo, fp4_hi).__x;
}

// ---------------------------------------------------------------------------
// FP4 <-> Float2 conversions
// ---------------------------------------------------------------------------

TL_DEVICE float2 __tl_cvt_fp4x2_to_float2(uint8_t src) {
  fp4_e2_2_t packed(src);
  return float2{(float)packed.x(), (float)packed.y()};
}

TL_DEVICE uint8_t __tl_cvt_float2_to_fp4x2(float2 src) {
  fp4_e2_t lo = __tl_float_to_fp4(src.x);
  fp4_e2_t hi = __tl_float_to_fp4(src.y);
  return make_fp4_e2_2_t(lo, hi).__x;
}

// ---------------------------------------------------------------------------
// FP4 <-> Double2 conversions
// ---------------------------------------------------------------------------

TL_DEVICE double2 __tl_cvt_fp4x2_to_double2(uint8_t src) {
  float2 f = __tl_cvt_fp4x2_to_float2(src);
  return double2{(double)f.x, (double)f.y};
}

TL_DEVICE uint8_t __tl_cvt_double2_to_fp4x2(double2 src) {
  return __tl_cvt_float2_to_fp4x2(float2{(float)src.x, (float)src.y});
}

// ---------------------------------------------------------------------------
// FP4 <-> BFloat162 conversions
// bfloat162 on HIP: we use uint1 (same as half2 storage pattern)
// ---------------------------------------------------------------------------

TL_DEVICE uint1 __tl_cvt_fp4x2_to_bfloat162(uint8_t src) {
  fp4_e2_2_t packed(src);
  bfloat16_t lo = (bfloat16_t)(float)packed.x();
  bfloat16_t hi = (bfloat16_t)(float)packed.y();
  return uint1{__pack_bfloat162(lo, hi)};
}

TL_DEVICE uint8_t __tl_cvt_bfloat162_to_fp4x2(uint1 src) {
  const uint32_t raw = src.x;
  bfloat16_t lo = *reinterpret_cast<const bfloat16_t *>(&raw);
  const uint16_t raw_hi = (uint16_t)(raw >> 16u);
  bfloat16_t hi = *reinterpret_cast<const bfloat16_t *>(&raw_hi);
  fp4_e2_t fp4_lo = __tl_float_to_fp4((float)lo);
  fp4_e2_t fp4_hi = __tl_float_to_fp4((float)hi);
  return make_fp4_e2_2_t(fp4_lo, fp4_hi).__x;
}

// ---------------------------------------------------------------------------
// Packed buffer access helpers
// Mirrors tl_fp4_packed_load / tl_fp4_packed_store from cuda_fp4.h.
// ---------------------------------------------------------------------------

// Load a single FP4 element from a packed fp4_e2_2_t array.
// idx is the logical index of the FP4 element (2 elements per array entry).
TL_DEVICE fp4_e2_t tl_fp4_packed_load(fp4_e2_2_t *packed, int idx) {
  return (idx & 1) ? packed[idx >> 1].y() : packed[idx >> 1].x();
}

// Store a single FP4 element into a packed fp4_e2_2_t array.
TL_DEVICE void tl_fp4_packed_store(fp4_e2_2_t *packed, int idx, fp4_e2_t val) {
  if (idx & 1) {
    packed[idx >> 1].set_y(val);
  } else {
    packed[idx >> 1].set_x(val);
  }
}

#endif  // defined(__gfx950__)
