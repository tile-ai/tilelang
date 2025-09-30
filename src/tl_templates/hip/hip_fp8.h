#include <hip/amd_detail/amd_hip_fp8.h>

#define HIP_FP8_ENABLED 1

using fp8_e4_t = __hip_fp8_e4m3_fnuz;
using fp8_e4_2_t = __hip_fp8x2_e4m3_fnuz;

// Additional FP8 types for compatibility
using fp8_e5_t = __hip_fp8_e5m2_fnuz;
using fp8_e5_2_t = __hip_fp8x2_e5m2_fnuz;
// Note: E8M0 types are not supported in current HIP version
// using fp8_e8_t = __hip_fp8_e8m0_fnuz;
// using fp8_e8_2_t = __hip_fp8x2_e8m0_fnuz;

// Simple wrapper that provides member access for generated code
struct fp8_e4_4_t {
  union {
    __hip_fp8x4_e4m3_fnuz data;
    struct {
      fp8_e4_t x, y, z, w;
    };
  };

  // Default constructor
  __device__ fp8_e4_4_t() = default;

  // Constructor from __hip_fp8x4_e4m3_fnuz
  __device__ fp8_e4_4_t(const __hip_fp8x4_e4m3_fnuz &val) : data(val) {}

  // Constructor from float4
  __device__ fp8_e4_4_t(const float4 &val) : data(val) {}

  // Conversion operator to __hip_fp8x4_e4m3_fnuz
  __device__ operator __hip_fp8x4_e4m3_fnuz() const { return data; }

  // Assignment operator
  __device__ fp8_e4_4_t &operator=(const __hip_fp8x4_e4m3_fnuz &val) {
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
struct fp8_e5_4_t {
  union {
    __hip_fp8x4_e5m2_fnuz data;
    struct {
      fp8_e5_t x, y, z, w;
    };
  };
  __device__ fp8_e5_4_t() = default;
  __device__ fp8_e5_4_t(const __hip_fp8x4_e5m2_fnuz &val) : data(val) {}
  __device__ operator __hip_fp8x4_e5m2_fnuz() const { return data; }
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
