#pragma once

#include "common.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#include <cuda_fp4.h>

// Wrapper for __nv_fp4_e2m1 with implicit conversions
struct fp4_e2_t {
  __nv_fp4_storage_t __x;

  TL_DEVICE fp4_e2_t() = default;

  // Constructor from __nv_fp4_e2m1
  TL_DEVICE fp4_e2_t(__nv_fp4_e2m1 x) : __x(x.__x) {}

  // Constructor from storage type
  TL_DEVICE fp4_e2_t(__nv_fp4_storage_t x) : __x(x) {}

  // Constructor from float
  TL_DEVICE explicit fp4_e2_t(float x) {
    __nv_fp4_e2m1 tmp(x);
    __x = tmp.__x;
  }

  // Conversion to __nv_fp4_e2m1
  TL_DEVICE operator __nv_fp4_e2m1() const {
    __nv_fp4_e2m1 tmp;
    tmp.__x = __x;
    return tmp;
  }

  // Conversion to float
  TL_DEVICE operator float() const {
    __nv_fp4_e2m1 tmp;
    tmp.__x = __x;
    return float(tmp);
  }

  // Implicit conversion to half_t (cutlass::half_t)
  TL_DEVICE operator half_t() const { return half_t(float(*this)); }

  // Implicit conversion to __half
  TL_DEVICE operator __half() const { return __half(float(*this)); }
};

using fp4_e2x2_t = __nv_fp4x2_e2m1;
using fp4_e2x4_t = __nv_fp4x4_e2m1;

struct fp4_e2x8_t {
  fp4_e2_t data[8];
};

struct fp4_e2x16_t {
  fp4_e2_t data[16];
};

struct __CUDA_ALIGN__(1) fp4_e2_2_t {
  fp4_e2_t x;
  fp4_e2_t y;
};

struct __CUDA_ALIGN__(2) fp4_e2_4_t {
  fp4_e2_t x;
  fp4_e2_t y;
  fp4_e2_t z;
  fp4_e2_t w;
};

struct __CUDA_ALIGN__(4) fp4_e2_8_t {
  fp4_e2_4_t x;
  fp4_e2_4_t y;
};

struct __CUDA_ALIGN__(8) fp4_e2_16_t {
  fp4_e2_8_t x;
  fp4_e2_8_t y;
};

struct __CUDA_ALIGN__(16) fp4_e2_32_t {
  fp4_e2_16_t x;
  fp4_e2_16_t y;

  TL_DEVICE fp4_e2_32_t &operator=(const ulonglong4 &rhs) {
    x.x = *(fp4_e2_8_t *)&rhs.x;
    x.y = *(fp4_e2_8_t *)&rhs.y;
    y.x = *(fp4_e2_8_t *)&rhs.z;
    y.y = *(fp4_e2_8_t *)&rhs.w;
    return *this;
  }
};

struct __CUDA_ALIGN__(32) fp4_e2_64_t {
  fp4_e2_32_t x;
  fp4_e2_32_t y;
};

// Pack two fp4_e2_t values.
TL_DEVICE fp4_e2_2_t make_fp4_e2_2_t(fp4_e2_t x, fp4_e2_t y) {
  fp4_e2_2_t result;
  result.x = x;
  result.y = y;
  return result;
}

// Pack four fp4_e2_t values.
TL_DEVICE fp4_e2_4_t make_fp4_e2_4_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                     fp4_e2_t x3) {
  fp4_e2_4_t result;
  result.x = x0;
  result.y = x1;
  result.z = x2;
  result.w = x3;
  return result;
}

// Pack eight fp4_e2_t values.
TL_DEVICE fp4_e2_8_t make_fp4_e2_8_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                     fp4_e2_t x3, fp4_e2_t x4, fp4_e2_t x5,
                                     fp4_e2_t x6, fp4_e2_t x7) {
  fp4_e2_8_t result;
  result.x = make_fp4_e2_4_t(x0, x1, x2, x3);
  result.y = make_fp4_e2_4_t(x4, x5, x6, x7);
  return result;
}

// Pack sixteen fp4_e2_t values.
TL_DEVICE fp4_e2_16_t make_fp4_e2_16_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                       fp4_e2_t x3, fp4_e2_t x4, fp4_e2_t x5,
                                       fp4_e2_t x6, fp4_e2_t x7, fp4_e2_t y0,
                                       fp4_e2_t y1, fp4_e2_t y2, fp4_e2_t y3,
                                       fp4_e2_t y4, fp4_e2_t y5, fp4_e2_t y6,
                                       fp4_e2_t y7) {
  fp4_e2_16_t result;
  result.x = make_fp4_e2_8_t(x0, x1, x2, x3, x4, x5, x6, x7);
  result.y = make_fp4_e2_8_t(y0, y1, y2, y3, y4, y5, y6, y7);
  return result;
}

// Pack thirty-two fp4_e2_t values.
TL_DEVICE fp4_e2_32_t make_fp4_e2_32_t(
    fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2, fp4_e2_t x3, fp4_e2_t x4,
    fp4_e2_t x5, fp4_e2_t x6, fp4_e2_t x7, fp4_e2_t x8, fp4_e2_t x9,
    fp4_e2_t x10, fp4_e2_t x11, fp4_e2_t x12, fp4_e2_t x13, fp4_e2_t x14,
    fp4_e2_t x15, fp4_e2_t y0, fp4_e2_t y1, fp4_e2_t y2, fp4_e2_t y3,
    fp4_e2_t y4, fp4_e2_t y5, fp4_e2_t y6, fp4_e2_t y7, fp4_e2_t y8,
    fp4_e2_t y9, fp4_e2_t y10, fp4_e2_t y11, fp4_e2_t y12, fp4_e2_t y13,
    fp4_e2_t y14, fp4_e2_t y15) {
  fp4_e2_32_t result;
  result.x = make_fp4_e2_16_t(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                              x12, x13, x14, x15);
  result.y = make_fp4_e2_16_t(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11,
                              y12, y13, y14, y15);
  return result;
}

// ============================================================================
// FP4 <-> Half Precision Conversions
// ============================================================================
// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html

// fp4_e2m1 -> half
TL_DEVICE __half __tl_cvt_fp4_to_half(const __nv_fp4_storage_t src) {
  __half_raw raw = __nv_cvt_fp4_to_halfraw(src, __NV_E2M1);
  __half result;
  result = *reinterpret_cast<__half *>(&raw);
  return result;
}

// fp4_e2m1x2 (1 byte) -> half2
TL_DEVICE half2 __tl_cvt_fp4x2_to_half2(const __nv_fp4x2_storage_t src) {
  __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(src, __NV_E2M1);
  half2 result;
  result = *reinterpret_cast<half2 *>(&raw);
  return result;
}

// half -> fp4_e2m1
TL_DEVICE __nv_fp4_storage_t __tl_cvt_half_to_fp4(const __half src) {
  __half_raw raw = *reinterpret_cast<const __half_raw *>(&src);
  return __nv_cvt_halfraw_to_fp4(raw, __NV_E2M1, cudaRoundZero);
}

// half2 -> fp4_e2m1x2 (1 byte)
TL_DEVICE __nv_fp4x2_storage_t __tl_cvt_half2_to_fp4x2(const half2 src) {
  __half2_raw raw = *reinterpret_cast<const __half2_raw *>(&src);
  return __nv_cvt_halfraw2_to_fp4x2(raw, __NV_E2M1, cudaRoundZero);
}

// ============================================================================
// FP4 <-> Float Conversions
// ============================================================================

// fp4_e2m1 -> float
TL_DEVICE float __tl_cvt_fp4_to_float(const __nv_fp4_storage_t src) {
  return __half2float(__tl_cvt_fp4_to_half(src));
}

// fp4_e2m1x2 (1 byte) -> float2
TL_DEVICE float2 __tl_cvt_fp4x2_to_float2(const __nv_fp4x2_storage_t src) {
  half2 tmp = __tl_cvt_fp4x2_to_half2(src);
  float2 result;
  result.x = __half2float(tmp.x);
  result.y = __half2float(tmp.y);
  return result;
}

// float -> fp4_e2m1
TL_DEVICE __nv_fp4_storage_t __tl_cvt_float_to_fp4(const float src) {
  return __nv_cvt_float_to_fp4(src, __NV_E2M1, cudaRoundZero);
}

// float2 -> fp4_e2m1x2 (1 byte)
TL_DEVICE __nv_fp4x2_storage_t __tl_cvt_float2_to_fp4x2(const float2 src) {
  return __nv_cvt_float2_to_fp4x2(src, __NV_E2M1, cudaRoundZero);
}

// ============================================================================
// FP4 <-> Double Conversions
// ============================================================================

// fp4_e2m1 -> double
TL_DEVICE double __tl_cvt_fp4_to_double(const __nv_fp4_storage_t src) {
  return static_cast<double>(__tl_cvt_fp4_to_float(src));
}

// fp4_e2m1x2 -> double2
TL_DEVICE double2 __tl_cvt_fp4x2_to_double2(const __nv_fp4x2_storage_t src) {
  float2 tmp = __tl_cvt_fp4x2_to_float2(src);
  double2 result;
  result.x = static_cast<double>(tmp.x);
  result.y = static_cast<double>(tmp.y);
  return result;
}

// double -> fp4_e2m1
TL_DEVICE __nv_fp4_storage_t __tl_cvt_double_to_fp4(const double src) {
  return __nv_cvt_double_to_fp4(src, __NV_E2M1, cudaRoundZero);
}

// double2 -> fp4_e2m1x2
TL_DEVICE __nv_fp4x2_storage_t __tl_cvt_double2_to_fp4x2(const double2 src) {
  return __nv_cvt_double2_to_fp4x2(src, __NV_E2M1, cudaRoundZero);
}

// ============================================================================
// FP4 <-> BFloat16 Conversions
// ============================================================================

// fp4_e2m1 -> bfloat16
TL_DEVICE __nv_bfloat16 __tl_cvt_fp4_to_bfloat16(const __nv_fp4_storage_t src) {
  return __float2bfloat16(__tl_cvt_fp4_to_float(src));
}

// fp4_e2m1x2 -> bfloat162
TL_DEVICE __nv_bfloat162
__tl_cvt_fp4x2_to_bfloat162(const __nv_fp4x2_storage_t src) {
  float2 tmp = __tl_cvt_fp4x2_to_float2(src);
  return __floats2bfloat162_rn(tmp.x, tmp.y);
}

// bfloat16 -> fp4_e2m1
TL_DEVICE __nv_fp4_storage_t __tl_cvt_bfloat16_to_fp4(const __nv_bfloat16 src) {
  __nv_bfloat16_raw raw = *reinterpret_cast<const __nv_bfloat16_raw *>(&src);
  return __nv_cvt_bfloat16raw_to_fp4(raw, __NV_E2M1, cudaRoundZero);
}

// bfloat162 -> fp4_e2m1x2
TL_DEVICE __nv_fp4x2_storage_t
__tl_cvt_bfloat162_to_fp4x2(const __nv_bfloat162 src) {
  __nv_bfloat162_raw raw = *reinterpret_cast<const __nv_bfloat162_raw *>(&src);
  return __nv_cvt_bfloat16raw2_to_fp4x2(raw, __NV_E2M1, cudaRoundZero);
}

#endif
