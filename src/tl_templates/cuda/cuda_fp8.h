#pragma once

#include "common.h"
#include <cuda_fp8.h>
#include <cute/numeric/numeric_types.hpp>

using fp8_e4_t = tl::float_e4m3_t;
using fp8_e5_t = tl::float_e5m2_t;

struct __CUDA_ALIGN__(2) fp8_e4_2_t {
  fp8_e4_t x;
  fp8_e4_t y;
};

struct __CUDA_ALIGN__(4) fp8_e4_4_t {
  fp8_e4_t x;
  fp8_e4_t y;
  fp8_e4_t z;
  fp8_e4_t w;
};

struct __CUDA_ALIGN__(8) fp8_e4_8_t {
  fp8_e4_4_t x;
  fp8_e4_4_t y;
};

struct __CUDA_ALIGN__(16) fp8_e4_16_t {
  fp8_e4_8_t x;
  fp8_e4_8_t y;
};

struct __CUDA_ALIGN__(32) fp8_e4_32_t {
  fp8_e4_16_t x;
  fp8_e4_16_t y;

  __device__ __forceinline__ fp8_e4_32_t &operator=(const ulonglong4 &rhs) {
    x.x = *(fp8_e4_8_t *)&rhs.x;
    x.y = *(fp8_e4_8_t *)&rhs.y;
    y.x = *(fp8_e4_8_t *)&rhs.z;
    y.y = *(fp8_e4_8_t *)&rhs.w;
    return *this;
  }
};

struct __CUDA_ALIGN__(2) fp8_e5_2_t {
  fp8_e5_t x;
  fp8_e5_t y;
};

struct __CUDA_ALIGN__(4) fp8_e5_4_t {
  fp8_e5_t x;
  fp8_e5_t y;
  fp8_e5_t z;
  fp8_e5_t w;
};

struct __CUDA_ALIGN__(8) fp8_e5_8_t {
  fp8_e5_4_t x;
  fp8_e5_4_t y;
};

struct __CUDA_ALIGN__(16) fp8_e5_16_t {
  fp8_e5_8_t x;
  fp8_e5_8_t y;
};

struct __CUDA_ALIGN__(32) fp8_e5_32_t {
  fp8_e5_16_t x;
  fp8_e5_16_t y;

  __device__ __forceinline__ fp8_e5_32_t &operator=(const ulonglong4 &rhs) {
    x.x = *(fp8_e5_8_t *)&rhs.x;
    x.y = *(fp8_e5_8_t *)&rhs.y;
    y.x = *(fp8_e5_8_t *)&rhs.z;
    y.y = *(fp8_e5_8_t *)&rhs.w;
    return *this;
  }
};

// Pack two fp8_e4_t values.
__forceinline__ __device__ fp8_e4_2_t make_fp8_e4_2_t(fp8_e4_t x, fp8_e4_t y) {
  fp8_e4_2_t result;
  result.x = x;
  result.y = y;
  return result;
}

// Pack four fp8_e4_t values.
__forceinline__ __device__ fp8_e4_4_t make_fp8_e4_4_t(fp8_e4_t x0, fp8_e4_t x1,
                                                      fp8_e4_t x2,
                                                      fp8_e4_t x3) {
  fp8_e4_4_t result;
  result.x = x0;
  result.y = x1;
  result.z = x2;
  result.w = x3;
  return result;
}

// Pack eight fp8_e4_t values.
__forceinline__ __device__ fp8_e4_8_t make_fp8_e4_8_t(fp8_e4_t x0, fp8_e4_t x1,
                                                      fp8_e4_t x2, fp8_e4_t x3,
                                                      fp8_e4_t x4, fp8_e4_t x5,
                                                      fp8_e4_t x6,
                                                      fp8_e4_t x7) {
  fp8_e4_8_t result;
  result.x = make_fp8_e4_4_t(x0, x1, x2, x3);
  result.y = make_fp8_e4_4_t(x4, x5, x6, x7);
  return result;
}

// Pack sixteen fp8_e4_t values.
__forceinline__ __device__ fp8_e4_16_t
make_fp8_e4_16_t(fp8_e4_t x0, fp8_e4_t x1, fp8_e4_t x2, fp8_e4_t x3,
                 fp8_e4_t x4, fp8_e4_t x5, fp8_e4_t x6, fp8_e4_t x7,
                 fp8_e4_t y0, fp8_e4_t y1, fp8_e4_t y2, fp8_e4_t y3,
                 fp8_e4_t y4, fp8_e4_t y5, fp8_e4_t y6, fp8_e4_t y7) {
  fp8_e4_16_t result;
  result.x = make_fp8_e4_8_t(x0, x1, x2, x3, x4, x5, x6, x7);
  result.y = make_fp8_e4_8_t(y0, y1, y2, y3, y4, y5, y6, y7);
  return result;
}

// Pack thirty-two fp8_e4_t values.
__forceinline__ __device__ fp8_e4_32_t make_fp8_e4_32_t(
    fp8_e4_t x0, fp8_e4_t x1, fp8_e4_t x2, fp8_e4_t x3, fp8_e4_t x4,
    fp8_e4_t x5, fp8_e4_t x6, fp8_e4_t x7, fp8_e4_t x8, fp8_e4_t x9,
    fp8_e4_t x10, fp8_e4_t x11, fp8_e4_t x12, fp8_e4_t x13, fp8_e4_t x14,
    fp8_e4_t x15, fp8_e4_t y0, fp8_e4_t y1, fp8_e4_t y2, fp8_e4_t y3,
    fp8_e4_t y4, fp8_e4_t y5, fp8_e4_t y6, fp8_e4_t y7, fp8_e4_t y8,
    fp8_e4_t y9, fp8_e4_t y10, fp8_e4_t y11, fp8_e4_t y12, fp8_e4_t y13,
    fp8_e4_t y14, fp8_e4_t y15) {
  fp8_e4_32_t result;
  result.x = make_fp8_e4_16_t(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                              x12, x13, x14, x15);
  result.y = make_fp8_e4_16_t(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11,
                              y12, y13, y14, y15);
  return result;
}

// Pack two fp8_e5_t values.
__forceinline__ __device__ fp8_e5_2_t make_fp8_e5_2_t(fp8_e5_t x, fp8_e5_t y) {
  fp8_e5_2_t result;
  result.x = x;
  result.y = y;
  return result;
}

// Pack four fp8_e5_t values.
__forceinline__ __device__ fp8_e5_4_t make_fp8_e5_4_t(fp8_e5_t x0, fp8_e5_t x1,
                                                      fp8_e5_t x2,
                                                      fp8_e5_t x3) {
  fp8_e5_4_t result;
  result.x = x0;
  result.y = x1;
  result.z = x2;
  result.w = x3;
  return result;
}

// Pack eight fp8_e5_t values.
__forceinline__ __device__ fp8_e5_8_t make_fp8_e5_8_t(fp8_e5_t x0, fp8_e5_t x1,
                                                      fp8_e5_t x2, fp8_e5_t x3,
                                                      fp8_e5_t x4, fp8_e5_t x5,
                                                      fp8_e5_t x6,
                                                      fp8_e5_t x7) {
  fp8_e5_8_t result;
  result.x = make_fp8_e5_4_t(x0, x1, x2, x3);
  result.y = make_fp8_e5_4_t(x4, x5, x6, x7);
  return result;
}

// Pack sixteen fp8_e5_t values.
__forceinline__ __device__ fp8_e5_16_t
make_fp8_e5_16_t(fp8_e5_t x0, fp8_e5_t x1, fp8_e5_t x2, fp8_e5_t x3,
                 fp8_e5_t x4, fp8_e5_t x5, fp8_e5_t x6, fp8_e5_t x7,
                 fp8_e5_t y0, fp8_e5_t y1, fp8_e5_t y2, fp8_e5_t y3,
                 fp8_e5_t y4, fp8_e5_t y5, fp8_e5_t y6, fp8_e5_t y7) {
  fp8_e5_16_t result;
  result.x = make_fp8_e5_8_t(x0, x1, x2, x3, x4, x5, x6, x7);
  result.y = make_fp8_e5_8_t(y0, y1, y2, y3, y4, y5, y6, y7);
  return result;
}

// Pack thirty-two fp8_e5_t values.
__forceinline__ __device__ fp8_e5_32_t make_fp8_e5_32_t(
    fp8_e5_t x0, fp8_e5_t x1, fp8_e5_t x2, fp8_e5_t x3, fp8_e5_t x4,
    fp8_e5_t x5, fp8_e5_t x6, fp8_e5_t x7, fp8_e5_t x8, fp8_e5_t x9,
    fp8_e5_t x10, fp8_e5_t x11, fp8_e5_t x12, fp8_e5_t x13, fp8_e5_t x14,
    fp8_e5_t x15, fp8_e5_t y0, fp8_e5_t y1, fp8_e5_t y2, fp8_e5_t y3,
    fp8_e5_t y4, fp8_e5_t y5, fp8_e5_t y6, fp8_e5_t y7, fp8_e5_t y8,
    fp8_e5_t y9, fp8_e5_t y10, fp8_e5_t y11, fp8_e5_t y12, fp8_e5_t y13,
    fp8_e5_t y14, fp8_e5_t y15) {
  fp8_e5_32_t result;
  result.x = make_fp8_e5_16_t(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                              x12, x13, x14, x15);
  result.y = make_fp8_e5_16_t(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11,
                              y12, y13, y14, y15);
  return result;
}
