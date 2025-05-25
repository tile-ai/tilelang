// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <cuda_fp8.h>

using fp8_e4_t = __nv_fp8_e4m3;
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

using fp8_e5_t = __nv_fp8_e5m2;

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
