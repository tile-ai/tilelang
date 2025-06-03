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
