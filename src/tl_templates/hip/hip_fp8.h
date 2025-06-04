// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <hip/amd_detail/amd_hip_fp8.h>

using fp8_e4_t = __hip_fp8_e4m3_fnuz;
using fp8_e4_2_t = __hip_fp8x2_e4m3_fnuz;
using fp8_e4_4_t = __hip_fp8x4_e4m3_fnuz;

struct __align__(8) fp8_e4_8_t {
  fp8_e4_4_t x;
  fp8_e4_4_t y;
};

struct __align__(16) fp8_e4_16_t {
  fp8_e4_8_t x;
  fp8_e4_8_t y;
};

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
