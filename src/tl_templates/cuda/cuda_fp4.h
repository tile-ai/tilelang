#pragma once

#include "common.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#include <cuda_fp4.h>

using fp4_e2_t = __nv_fp4_e2m1;
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

#endif
