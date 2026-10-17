#pragma once

#include <cstdint>

#include "simt_api/asc_fp16.h"
#include "simt_api/asc_fp8.h"

#ifndef TL_DEVICE
#define TL_DEVICE __SIMT_DEVICE_FUNCTIONS_DECL__ inline
#endif

namespace tl {
namespace detail {

TL_DEVICE uint8_t cast_float_to_fp8_e4m3(float x) {
  alignas(8) float x2_storage[2] = {x, 0.0f};
  __asc_fp8x2_storage_t y2 = __asc_cvt_float2_to_fp8x2(
      *reinterpret_cast<float2 *>(x2_storage), __ASC_SATFINITE, __ASC_E4M3);
  return static_cast<uint8_t>(y2 & 0xff);
}

TL_DEVICE uint8_t cast_float_to_fp8_e5m2(float x) {
  alignas(8) float x2_storage[2] = {x, 0.0f};
  __asc_fp8x2_storage_t y2 = __asc_cvt_float2_to_fp8x2(
      *reinterpret_cast<float2 *>(x2_storage), __ASC_SATFINITE, __ASC_E5M2);
  return static_cast<uint8_t>(y2 & 0xff);
}

TL_DEVICE float cast_fp8_e4m3_to_float(uint8_t x) {
  __asc_fp8x2_storage_t x2_storage = x;
  float8_e4m3x2_t x2 = *reinterpret_cast<float8_e4m3x2_t *>(&x2_storage);
  alignas(8) float y2_storage[2];
  *reinterpret_cast<float2 *>(y2_storage) = __e4m3x22float2(x2);
  return y2_storage[0];
}

TL_DEVICE float cast_fp8_e5m2_to_float(uint8_t x) {
  __asc_fp8x2_storage_t x2_storage = x;
  float8_e5m2x2_t x2 = *reinterpret_cast<float8_e5m2x2_t *>(&x2_storage);
  alignas(8) float y2_storage[2];
  *reinterpret_cast<float2 *>(y2_storage) = __e5m2x22float2(x2);
  return y2_storage[0];
}

TL_DEVICE uint16_t cast_half2_to_fp8_e4m3x2(half2 x) {
  return __asc_cvt_float2_to_fp8x2(__half22float2(x), __ASC_SATFINITE,
                                   __ASC_E4M3);
}

TL_DEVICE uint16_t cast_half2_to_fp8_e5m2x2(half2 x) {
  return __asc_cvt_float2_to_fp8x2(__half22float2(x), __ASC_SATFINITE,
                                   __ASC_E5M2);
}

TL_DEVICE uint16_t cast_bfloat162_to_fp8_e4m3x2(bfloat16x2_t x) {
  return __asc_cvt_float2_to_fp8x2(__bfloat1622float2(x), __ASC_SATFINITE,
                                   __ASC_E4M3);
}

TL_DEVICE uint16_t cast_bfloat162_to_fp8_e5m2x2(bfloat16x2_t x) {
  return __asc_cvt_float2_to_fp8x2(__bfloat1622float2(x), __ASC_SATFINITE,
                                   __ASC_E5M2);
}

TL_DEVICE half2 cast_fp8_e4m3x2_to_half2(float8_e4m3x2_t x) {
  return __float22half2_rn(__e4m3x22float2(x));
}

TL_DEVICE half2 cast_fp8_e5m2x2_to_half2(float8_e5m2x2_t x) {
  return __float22half2_rn(__e5m2x22float2(x));
}

TL_DEVICE bfloat16x2_t cast_fp8_e4m3x2_to_bfloat162(float8_e4m3x2_t x) {
  return __float22bfloat162_rn(__e4m3x22float2(x));
}

TL_DEVICE bfloat16x2_t cast_fp8_e5m2x2_to_bfloat162(float8_e5m2x2_t x) {
  return __float22bfloat162_rn(__e5m2x22float2(x));
}

} // namespace detail

struct float_e4m3_t {
  uint8_t data;

  TL_DEVICE float_e4m3_t() = default;

  TL_DEVICE float_e4m3_t(float x) : data(detail::cast_float_to_fp8_e4m3(x)) {}

  TL_DEVICE float_e4m3_t &operator=(float x) {
    data = detail::cast_float_to_fp8_e4m3(x);
    return *this;
  }

  TL_DEVICE static float_e4m3_t from_bits(uint8_t x) {
    float_e4m3_t result;
    result.data = x;
    return result;
  }

  TL_DEVICE operator float() const {
    return detail::cast_fp8_e4m3_to_float(data);
  }
};

struct float_e5m2_t {
  uint8_t data;

  TL_DEVICE float_e5m2_t() = default;

  TL_DEVICE float_e5m2_t(float x) : data(detail::cast_float_to_fp8_e5m2(x)) {}

  TL_DEVICE float_e5m2_t &operator=(float x) {
    data = detail::cast_float_to_fp8_e5m2(x);
    return *this;
  }

  TL_DEVICE static float_e5m2_t from_bits(uint8_t x) {
    float_e5m2_t result;
    result.data = x;
    return result;
  }

  TL_DEVICE operator float() const {
    return detail::cast_fp8_e5m2_to_float(data);
  }
};

} // namespace tl

// using fp8_e4_t = tl::float_e4m3_t;
using fp8_e4_t = float8_e4m3_t;
using fp8_e5_t = tl::float_e5m2_t;
using fp8_e8_t = uint8_t;

using fp8_e4_2_t = uint16_t;
using fp8_e4_4_t = uint32_t;
using fp8_e4_8_t = uint64_t;

using fp8_e5_2_t = uint16_t;
using fp8_e5_4_t = uint32_t;
using fp8_e5_8_t = uint64_t;

using fp8_e8_2_t = uint16_t;
using fp8_e8_4_t = uint32_t;
using fp8_e8_8_t = uint64_t;
