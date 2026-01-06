#pragma once

#include "atomic.h"
#include <ck_tile/core.hpp>
#include <hip/amd_detail/amd_warp_functions.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#define HIPRT_INF_F __int_as_float(0x7f800000)
#define HIPRT_NEGINF_F __int_as_float(0xff800000)
#define HIPRT_NAN_F __int_as_float(0x7fffffff)
#define HIPRT_MIN_DENORM_F __int_as_float(0x00000001)
#define HIPRT_MAX_NORMAL_F __int_as_float(0x7f7fffff)
#define HIPRT_NEG_ZERO_F __int_as_float(0x80000000)
#define HIPRT_ZERO_F 0.0f
#define HIPRT_ONE_F 1.0f

/* double precision constants */
#define HIPRT_INF __hiloint2double(0x7ff00000, 0x00000000)
#define HIPRT_NAN __hiloint2double(0xfff80000, 0x00000000)

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE __forceinline__ __device__
#define TL_DEVICE_NOINLINE __noinline__ __device__

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    hipError_t __err = (stmt);                                                 \
    if (__err != hipSuccess) {                                                 \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: %s - %s", __FILE__,          \
               __LINE__, hipGetErrorName(__err), hipGetErrorString(__err));    \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    hipError_t __err = hipGetLastError();                                      \
    if (__err != hipSuccess) {                                                 \
      snprintf(error_buf, ERROR_BUF_SIZE, "kernel_name: %s - %s",              \
               hipGetErrorName(__err), hipGetErrorString(__err));              \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define half _Float16
#define __float2half_rn(x) half(x)

#define hpow __ocml_pown_f16
#define hsqrt __ocml_sqrt_f16

using float16_t = _Float16;
using float16x2 =
    __attribute__((__vector_size__(2 * sizeof(float16_t)))) float16_t;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using float16x8 =
    __attribute__((__vector_size__(8 * sizeof(float16_t)))) float16_t;
using float16x16 =
    __attribute__((__vector_size__(16 * sizeof(float16_t)))) float16_t;

using half_t = float16_t;

using bfloat16_t = hip_bfloat16;

struct bfloat16x2 {
  bfloat16_t x, y;
};

struct bfloat16x4 {
  bfloat16_t data[4];
};

struct bfloat16x8 {
  bfloat16_t data[8];
};

struct bfloat16x16 {
  bfloat16_t data[16];
};

typedef
    __attribute__((__vector_size__(4 * sizeof(short)))) short bfloat16x4_vec;

using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

using int8x4 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_bfloat162(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

namespace tl {

// Any
template <typename T> TL_DEVICE bool Any(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i]) {
      return true;
    }
  }
  return false;
}

// All
template <typename T> TL_DEVICE bool All(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (!a[i]) {
      return false;
    }
  }
  return true;
}

// TODO(gong): support shfl_sync(rocm 7.1.1 provide shfl_sync)
// shfl_sync func
template <typename T> TL_DEVICE T shfl_xor(T val, int delta) {
  return __shfl_xor(val, delta);
}

template <typename T> TL_DEVICE T shfl_down(T val, int delta) {
  return __shfl_down(val, delta);
}

template <typename T> TL_DEVICE T shfl_up(T val, int delta) {
  return __shfl_up(val, delta);
}

template <typename T> TL_DEVICE T shfl(T val, int srcLane) {
  return __shfl(val, srcLane);
}

// specialize half_t
template <> TL_DEVICE half_t shfl_xor(half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_xor(f, delta);
  return half_t(r);
}

template <> TL_DEVICE half_t shfl_down(half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_down(f, delta);
  return half_t(r);
}

template <> TL_DEVICE half_t shfl_up(half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_up(f, delta);
  return half_t(r);
}

template <> TL_DEVICE half_t shfl(half_t val, int srcLane) {
  float f = static_cast<float>(val);
  float r = __shfl(f, srcLane);
  return half_t(r);
}

// specialize bfloat16_t
template <> TL_DEVICE bfloat16_t shfl_xor(bfloat16_t val, int laneMask) {
  float f = static_cast<float>(val);
  float r = __shfl_xor(f, laneMask);
  return bfloat16_t(r);
}

template <> TL_DEVICE bfloat16_t shfl_down(bfloat16_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_down(f, delta);
  return bfloat16_t(r);
}

template <> TL_DEVICE bfloat16_t shfl_up(bfloat16_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_up(f, delta);
  return bfloat16_t(r);
}

template <> TL_DEVICE bfloat16_t shfl(bfloat16_t val, int srcLane) {
  float f = static_cast<float>(val);
  float r = __shfl(f, srcLane);
  return bfloat16_t(r);
}

} // namespace tl
