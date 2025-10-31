#pragma once


#include "core.hpp"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
// #include <rocwmma/rocwmma.hpp>

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

using bfloat16_t = __hip_bfloat16;

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

template <typename T>
struct is_half_type : std::false_type {};

template <>
struct is_half_type<__half> : std::true_type {};

template <>
struct is_half_type<half_t> : std::true_type {};

template <typename T>
inline constexpr bool is_half_v = is_half_type<std::decay_t<T>>::value;

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1* address, T2 val) {
    if constexpr (is_half_v<T1>) {
        __half* addr = reinterpret_cast<__half*>(address);
        __half hval = __float2half(static_cast<float>(val));
        atomicAdd(addr, hval);
    } else {
        atomicAdd(address, static_cast<T1>(val));
    }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 &ref, T2 val) {
    AtomicAdd(&ref, val);
}
template <typename T1, typename T2> TL_DEVICE T1 AtomicAddRet(T1 &ref, T2 val) {
  return atomicAdd(&ref, static_cast<T1>(val));
}

template <typename T>
TL_DEVICE void AtomicAddx4(T* ref, const T val[4]) {
    atomicAdd(&ref[0], val[0]);
    atomicAdd(&ref[1], val[1]);
    atomicAdd(&ref[2], val[2]);
    atomicAdd(&ref[3], val[3]);
}