#pragma once
#include <hip/amd_detail/amd_hip_fp8.h>
#include <stdint.h>

#define HIP_FP8_ENABLED 1

#define TILELANG_FP8_E4M3_VARIANT_FN 0
#define TILELANG_FP8_E4M3_VARIANT_FNUZ 1

#define TILELANG_FP8_E5M2_VARIANT_FN 0
#define TILELANG_FP8_E5M2_VARIANT_FNUZ 1

#ifndef TILELANG_FP8_E4M3_VARIANT
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define TILELANG_FP8_E4M3_VARIANT TILELANG_FP8_E4M3_VARIANT_FNUZ
#else
#define TILELANG_FP8_E4M3_VARIANT TILELANG_FP8_E4M3_VARIANT_FN
#endif
#endif

#ifndef TILELANG_FP8_E5M2_VARIANT
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define TILELANG_FP8_E5M2_VARIANT TILELANG_FP8_E5M2_VARIANT_FNUZ
#else
#define TILELANG_FP8_E5M2_VARIANT TILELANG_FP8_E5M2_VARIANT_FN
#endif
#endif

#if (TILELANG_FP8_E4M3_VARIANT == TILELANG_FP8_E4M3_VARIANT_FN)
using hip_fp8_e4_t = __hip_fp8_e4m3;
using hip_fp8x2_e4_t = __hip_fp8x2_e4m3;
using hip_fp8x4_e4_t = __hip_fp8x4_e4m3;
#else
// FNUZ path (MI300X and universal fallback)
using hip_fp8_e4_t = __hip_fp8_e4m3_fnuz;
using hip_fp8x2_e4_t = __hip_fp8x2_e4m3_fnuz;
using hip_fp8x4_e4_t = __hip_fp8x4_e4m3_fnuz;
#endif

#if (TILELANG_FP8_E5M2_VARIANT == TILELANG_FP8_E5M2_VARIANT_FN)
using hip_fp8_e5_t = __hip_fp8_e5m2;
using hip_fp8x2_e5_t = __hip_fp8x2_e5m2;
using hip_fp8x4_e5_t = __hip_fp8x4_e5m2;
#else
using hip_fp8_e5_t = __hip_fp8_e5m2_fnuz;
using hip_fp8x2_e5_t = __hip_fp8x2_e5m2_fnuz;
using hip_fp8x4_e5_t = __hip_fp8x4_e5m2_fnuz;
#endif

struct fp8_e4_t {
  unsigned char data;
  __device__ fp8_e4_t() {}
  __device__ fp8_e4_t(hip_fp8_e4_t val) {
    data = *reinterpret_cast<unsigned char *>(&val);
  }
  __device__ fp8_e4_t(float val) {
    constexpr __hip_fp8_interpretation_t interp =
#if (TILELANG_FP8_E4M3_VARIANT == TILELANG_FP8_E4M3_VARIANT_FNUZ)
        __HIP_E4M3_FNUZ;
#else
        __HIP_E4M3;
#endif
    data = __hip_cvt_float_to_fp8(val, __HIP_SATFINITE, interp);
  }
  __device__ operator hip_fp8_e4_t() const {
    return *reinterpret_cast<const hip_fp8_e4_t *>(&data);
  }
  __device__ operator float() const {
    return static_cast<float>(static_cast<hip_fp8_e4_t>(*this));
  }
};

using fp8_e4_2_t = hip_fp8x2_e4_t;
using fp8_e4_4_storage_t = uint32_t;

// Additional FP8 types for compatibility
using fp8_e5_2_t = hip_fp8x2_e5_t;

struct fp8_e5_t {
  unsigned char data;
  __device__ fp8_e5_t() {}
  __device__ fp8_e5_t(hip_fp8_e5_t val) {
    data = *reinterpret_cast<unsigned char *>(&val);
  }
  __device__ fp8_e5_t(float val) {
    constexpr __hip_fp8_interpretation_t interp =
#if (TILELANG_FP8_E5M2_VARIANT == TILELANG_FP8_E5M2_VARIANT_FNUZ)
        __HIP_E5M2_FNUZ;
#else
        __HIP_E5M2;
#endif
    data = __hip_cvt_float_to_fp8(val, __HIP_SATFINITE, interp);
  }
  __device__ operator hip_fp8_e5_t() const {
    return *reinterpret_cast<const hip_fp8_e5_t *>(&data);
  }
  __device__ operator float() const {
    return static_cast<float>(static_cast<hip_fp8_e5_t>(*this));
  }
};
// Note: E8M0 types are not supported in current HIP version
// using fp8_e8_t = __hip_fp8_e8m0_fnuz;
// using fp8_e8_2_t = __hip_fp8x2_e8m0_fnuz;

// Simple wrapper that provides member access for generated code
struct __align__(4) fp8_e4_4_t {
  union {
    fp8_e4_4_storage_t data;
    struct {
      fp8_e4_t x;
      fp8_e4_t y;
      fp8_e4_t z;
      fp8_e4_t w;
    };
  };

  __device__ fp8_e4_4_t() {}
  __device__ fp8_e4_4_t(const fp8_e4_4_storage_t &val) : data(val) {}
  __device__ fp8_e4_4_t(const hip_fp8x4_e4_t &val) {
    data = *reinterpret_cast<const fp8_e4_4_storage_t *>(&val);
  }

  __device__ operator hip_fp8x4_e4_t() const {
    return *reinterpret_cast<const hip_fp8x4_e4_t *>(&data);
  }

  __device__ fp8_e4_4_t &operator=(const fp8_e4_4_storage_t &val) {
    data = val;
    return *this;
  }
};

struct __align__(8) fp8_e4_8_t {
  fp8_e4_4_t x;
  fp8_e4_4_t y;
};

struct __align__(16) fp8_e4_16_t {
  fp8_e4_8_t x;
  fp8_e4_8_t y;
};

// FP8 E5M2 vector types
using fp8_e5_4_storage_t = uint32_t;

struct __align__(4) fp8_e5_4_t {
  union {
    fp8_e5_4_storage_t data;
    struct {
      fp8_e5_t x;
      fp8_e5_t y;
      fp8_e5_t z;
      fp8_e5_t w;
    };
  };
  __device__ fp8_e5_4_t() {}
  __device__ fp8_e5_4_t(const hip_fp8x4_e5_t &val) {
    data = *reinterpret_cast<const fp8_e5_4_storage_t *>(&val);
  }
  __device__ operator hip_fp8x4_e5_t() const {
    return *reinterpret_cast<const hip_fp8x4_e5_t *>(&data);
  }
};

struct __align__(8) fp8_e5_8_t {
  fp8_e5_4_t x;
  fp8_e5_4_t y;
};

struct __align__(16) fp8_e5_16_t {
  fp8_e5_8_t x;
  fp8_e5_8_t y;
};

// FP8 E8M0 vector types - not supported in current HIP version
/*
struct fp8_e8_4_t {
  union {
    __hip_fp8x4_e8m0_fnuz data;
    struct {
      fp8_e8_t x, y, z, w;
    };
  };
  __device__ fp8_e8_4_t() = default;
  __device__ fp8_e8_4_t(const __hip_fp8x4_e8m0_fnuz &val) : data(val) {}
  __device__ operator __hip_fp8x4_e8m0_fnuz() const { return data; }
};

struct __align__(8) fp8_e8_8_t {
  fp8_e8_4_t x;
  fp8_e8_4_t y;
};

struct __align__(16) fp8_e8_16_t {
  fp8_e8_8_t x;
  fp8_e8_8_t y;
};
*/

__device__ fp8_e4_4_t make_fp8_e4_4_t(fp8_e4_t x, fp8_e4_t y, fp8_e4_t z,
                                      fp8_e4_t w) {
  // reinterpret the 4 fp8_e4_t values to signed char value and shift
  signed char x_char = *reinterpret_cast<signed char *>(&x);
  signed char y_char = *reinterpret_cast<signed char *>(&y);
  signed char z_char = *reinterpret_cast<signed char *>(&z);
  signed char w_char = *reinterpret_cast<signed char *>(&w);
  int res = (w_char << 24) | (z_char << 16) | (y_char << 8) | x_char;
  return *reinterpret_cast<fp8_e4_4_t *>(&res);
}

__device__ fp8_e4_8_t make_fp8_e4_8_t(fp8_e4_t x, fp8_e4_t y, fp8_e4_t z,
                                      fp8_e4_t w, fp8_e4_t v, fp8_e4_t u,
                                      fp8_e4_t t, fp8_e4_t s) {
  signed char x_char = *reinterpret_cast<signed char *>(&x);
  signed char y_char = *reinterpret_cast<signed char *>(&y);
  signed char z_char = *reinterpret_cast<signed char *>(&z);
  signed char w_char = *reinterpret_cast<signed char *>(&w);
  signed char v_char = *reinterpret_cast<signed char *>(&v);
  signed char u_char = *reinterpret_cast<signed char *>(&u);
  signed char t_char = *reinterpret_cast<signed char *>(&t);
  signed char s_char = *reinterpret_cast<signed char *>(&s);
  int a = (w_char << 24) | (z_char << 16) | (y_char << 8) | x_char;
  int b = (s_char << 24) | (t_char << 16) | (u_char << 8) | v_char;
  fp8_e4_8_t res;
  res.x = *reinterpret_cast<fp8_e4_4_t *>(&a);
  res.y = *reinterpret_cast<fp8_e4_4_t *>(&b);
  return res;
}

__device__ fp8_e4_16_t make_fp8_e4_16_t(fp8_e4_t x0, fp8_e4_t x1, fp8_e4_t x2,
                                        fp8_e4_t x3, fp8_e4_t x4, fp8_e4_t x5,
                                        fp8_e4_t x6, fp8_e4_t x7, fp8_e4_t y0,
                                        fp8_e4_t y1, fp8_e4_t y2, fp8_e4_t y3,
                                        fp8_e4_t y4, fp8_e4_t y5, fp8_e4_t y6,
                                        fp8_e4_t y7) {
  signed char x0_char = *reinterpret_cast<signed char *>(&x0);
  signed char x1_char = *reinterpret_cast<signed char *>(&x1);
  signed char x2_char = *reinterpret_cast<signed char *>(&x2);
  signed char x3_char = *reinterpret_cast<signed char *>(&x3);
  signed char x4_char = *reinterpret_cast<signed char *>(&x4);
  signed char x5_char = *reinterpret_cast<signed char *>(&x5);
  signed char x6_char = *reinterpret_cast<signed char *>(&x6);
  signed char x7_char = *reinterpret_cast<signed char *>(&x7);
  signed char y0_char = *reinterpret_cast<signed char *>(&y0);
  signed char y1_char = *reinterpret_cast<signed char *>(&y1);
  signed char y2_char = *reinterpret_cast<signed char *>(&y2);
  signed char y3_char = *reinterpret_cast<signed char *>(&y3);
  signed char y4_char = *reinterpret_cast<signed char *>(&y4);
  signed char y5_char = *reinterpret_cast<signed char *>(&y5);
  signed char y6_char = *reinterpret_cast<signed char *>(&y6);
  signed char y7_char = *reinterpret_cast<signed char *>(&y7);
  int a = (x3_char << 24) | (x2_char << 16) | (x1_char << 8) | x0_char;
  int b = (x7_char << 24) | (x6_char << 16) | (x5_char << 8) | x4_char;
  int c = (y3_char << 24) | (y2_char << 16) | (y1_char << 8) | y0_char;
  int d = (y7_char << 24) | (y6_char << 16) | (y5_char << 8) | y4_char;
  fp8_e4_8_t res_x;
  res_x.x = *reinterpret_cast<fp8_e4_4_t *>(&a);
  res_x.y = *reinterpret_cast<fp8_e4_4_t *>(&b);
  fp8_e4_8_t res_y;
  res_y.x = *reinterpret_cast<fp8_e4_4_t *>(&c);
  res_y.y = *reinterpret_cast<fp8_e4_4_t *>(&d);
  fp8_e4_16_t res;
  res.x = res_x;
  res.y = res_y;
  return res;
}
