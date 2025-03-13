// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cute/numeric/numeric_types.hpp>
using fp8_e4_t = cute::float_e4m3_t;
using fp8_e4_2_t = __nv_fp8x2_e4m3;
using fp8_e4_4_t = __nv_fp8x4_e4m3;
struct fp8_e4_8_t {
  fp8_e4_t data[8];
};
struct fp8_e4_16_t {
  fp8_e4_t data[16];
};
using fp8_e5_t = cute::float_e5m2_t;
using fp8_e5_2_t = __nv_fp8x2_e5m2;
using fp8_e5_4_t = __nv_fp8x4_e5m2;
struct fp8_e5_8_t {
  fp8_e5_t data[8];
};
struct fp8_e5_16_t {
  fp8_e5_t data[16];
};
