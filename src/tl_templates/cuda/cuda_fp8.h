// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cuda_fp8.h>
using fp8_e4_t = __nv_fp8_e4m3;
using fp8_e4_2_t = __nv_fp8x2_e4m3;
using fp8_e4_4_t = __nv_fp8x4_e4m3;
struct fp8_e4_8_t {
  fp8_e4_t data[8];
};
struct fp8_e4_16_t {
  fp8_e4_t data[16];
};
using fp8_e5_t = __nv_fp8_e5m2;
using fp8_e5_2_t = __nv_fp8x2_e5m2;
using fp8_e5_4_t = __nv_fp8x4_e5m2;
struct fp8_e5_8_t {
  fp8_e5_t data[8];
};
struct fp8_e5_16_t {
  fp8_e5_t data[16];
};
