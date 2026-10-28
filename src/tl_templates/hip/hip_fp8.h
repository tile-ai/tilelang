#pragma once
#include <hip/amd_detail/amd_hip_fp8.h>
#include <stdint.h>
#include <type_traits>

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

// HIP's shuffle builtins do not provide exact overloads for TileLang's FP8
// wrappers. Carry the raw byte through the native unsigned-int overload so the
// exact FP8 bit pattern is preserved without widening through float.
#define TL_DEFINE_FP8_SHFL_OVERLOADS(TYPE)                                     \
  __device__ __forceinline__ TYPE __shfl(TYPE value, int src_lane,             \
                                         int width = warpSize) {               \
    TYPE result;                                                               \
    result.data = static_cast<unsigned char>(                                  \
        __shfl(static_cast<unsigned int>(value.data), src_lane, width));       \
    return result;                                                             \
  }                                                                            \
  __device__ __forceinline__ TYPE __shfl_xor(TYPE value, int lane_mask,        \
                                             int width = warpSize) {           \
    TYPE result;                                                               \
    result.data = static_cast<unsigned char>(                                  \
        __shfl_xor(static_cast<unsigned int>(value.data), lane_mask, width));  \
    return result;                                                             \
  }                                                                            \
  __device__ __forceinline__ TYPE __shfl_down(TYPE value, unsigned int delta,  \
                                              int width = warpSize) {          \
    TYPE result;                                                               \
    result.data = static_cast<unsigned char>(                                  \
        __shfl_down(static_cast<unsigned int>(value.data), delta, width));     \
    return result;                                                             \
  }                                                                            \
  __device__ __forceinline__ TYPE __shfl_up(TYPE value, unsigned int delta,    \
                                            int width = warpSize) {            \
    TYPE result;                                                               \
    result.data = static_cast<unsigned char>(                                  \
        __shfl_up(static_cast<unsigned int>(value.data), delta, width));       \
    return result;                                                             \
  }

TL_DEFINE_FP8_SHFL_OVERLOADS(fp8_e4_t)
TL_DEFINE_FP8_SHFL_OVERLOADS(fp8_e5_t)

#undef TL_DEFINE_FP8_SHFL_OVERLOADS

template <typename ScalarType, typename PackedType>
__device__ __forceinline__ ScalarType
tl_fp8x2_get_lane(const PackedType &packed, int lane) {
  ScalarType result;
  result.data = static_cast<unsigned char>(
      (static_cast<unsigned int>(packed.__x) >> (lane * 8)) & 0xffu);
  return result;
}

template <typename PackedType, typename ScalarType>
__device__ __forceinline__ void tl_fp8x2_set_lane(PackedType &packed, int lane,
                                                  ScalarType value) {
  if (lane == 0) {
    // The result carrier is uninitialized before its first lane is written.
    packed.__x = static_cast<decltype(packed.__x)>(value.data);
    return;
  }
  packed.__x = static_cast<decltype(packed.__x)>(
      (static_cast<unsigned int>(packed.__x) & 0x00ffu) |
      (static_cast<unsigned int>(value.data) << 8));
}

// E8M0 (float8_e8m0fnu): a bare biased power-of-two exponent, 0xFF is NaN.
// HIP ships no native type, so this mirrors __nv_fp8_e8m0, whose float
// constructor uses cudaRoundPosInf and __NV_SATFINITE: |val| rounds up to a
// power of two, results below 2^-127 clip to it, results above 2^127 saturate
// to it, and NaN converts to 0xFF.
struct fp8_e8_t {
  unsigned char data;
  __device__ fp8_e8_t() {}
  __device__ fp8_e8_t(float val) {
    uint32_t bits = __float_as_uint(val);
    uint32_t exponent = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (val != val) {
      data = 0xFF;
      return;
    }
    // Subnormals up to 2^-127 (mantissa <= 2^22) keep exponent 0, i.e. clip to
    // 2^-127.
    if (exponent != 0xFF && mantissa != 0 &&
        (exponent != 0 || mantissa > 0x400000u)) {
      ++exponent;
    }
    data = static_cast<unsigned char>(exponent == 0xFF ? 0xFE : exponent);
  }
  // A double converts directly, as with __nv_fp8_e8m0(double): narrowing it to
  // float first can round it down onto a power of two (1 + 2^-25 to 1.0).
  // Integers have their own constructor below; other types use the float one.
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, double>::value>::type>
  __device__ fp8_e8_t(T val) {
    uint64_t bits = static_cast<uint64_t>(__double_as_longlong(val));
    int exponent = static_cast<int>((bits >> 52) & 0x7FF);
    bool inexact = (bits & 0xFFFFFFFFFFFFFull) != 0;
    if (val != val) {
      data = 0xFF;
      return;
    }
    // Zero and subnormal doubles are far below 2^-127 and clip to it.
    int e8 = exponent == 0 ? 0 : exponent - 1023 + 127 + inexact;
    data = static_cast<unsigned char>(
        exponent == 0x7FF || e8 > 254 ? 0xFE : (e8 < 0 ? 0 : e8));
  }
  // Integers round exactly: narrowing to float can land on a power of two
  // (2^24 + 1 to 2^24), and so can narrowing a 64-bit value to double.
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value &&
                                        !std::is_same<T, bool>::value,
                                    int>::type = 0>
  __device__ fp8_e8_t(T val) {
    using U = typename std::make_unsigned<T>::type;
    // |val| in unsigned arithmetic, which also holds the most negative value.
    U mag = (std::is_signed<T>::value && val < 0)
                ? static_cast<U>(U(0) - static_cast<U>(val))
                : static_cast<U>(val);
    if (mag == 0) {
      data = 0; // zero clips to 2^-127
      return;
    }
    int log2_floor = 63 - __builtin_clzll(static_cast<unsigned long long>(mag));
    data =
        static_cast<unsigned char>(127 + log2_floor + ((mag & (mag - 1)) != 0));
  }
  __device__ operator float() const {
    if (data == 0xFF) {
      return __uint_as_float(0x7FC00000u);
    }
    if (data == 0) {
      return __uint_as_float(0x00400000u); // 2^-127 is a float32 denormal
    }
    return __uint_as_float(static_cast<uint32_t>(data) << 23);
  }
};

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
  // reinterpret the 4 fp8_e4_t values to unsigned char to avoid sign extension
  // on shift
  unsigned char x_char = *reinterpret_cast<unsigned char *>(&x);
  unsigned char y_char = *reinterpret_cast<unsigned char *>(&y);
  unsigned char z_char = *reinterpret_cast<unsigned char *>(&z);
  unsigned char w_char = *reinterpret_cast<unsigned char *>(&w);
  unsigned int res = ((unsigned int)w_char << 24) |
                     ((unsigned int)z_char << 16) |
                     ((unsigned int)y_char << 8) | (unsigned int)x_char;
  return *reinterpret_cast<fp8_e4_4_t *>(&res);
}

__device__ fp8_e4_8_t make_fp8_e4_8_t(fp8_e4_t x, fp8_e4_t y, fp8_e4_t z,
                                      fp8_e4_t w, fp8_e4_t v, fp8_e4_t u,
                                      fp8_e4_t t, fp8_e4_t s) {
  unsigned char x_char = *reinterpret_cast<unsigned char *>(&x);
  unsigned char y_char = *reinterpret_cast<unsigned char *>(&y);
  unsigned char z_char = *reinterpret_cast<unsigned char *>(&z);
  unsigned char w_char = *reinterpret_cast<unsigned char *>(&w);
  unsigned char v_char = *reinterpret_cast<unsigned char *>(&v);
  unsigned char u_char = *reinterpret_cast<unsigned char *>(&u);
  unsigned char t_char = *reinterpret_cast<unsigned char *>(&t);
  unsigned char s_char = *reinterpret_cast<unsigned char *>(&s);
  unsigned int a = ((unsigned int)w_char << 24) | ((unsigned int)z_char << 16) |
                   ((unsigned int)y_char << 8) | (unsigned int)x_char;
  unsigned int b = ((unsigned int)s_char << 24) | ((unsigned int)t_char << 16) |
                   ((unsigned int)u_char << 8) | (unsigned int)v_char;
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
  unsigned char x0_char = *reinterpret_cast<unsigned char *>(&x0);
  unsigned char x1_char = *reinterpret_cast<unsigned char *>(&x1);
  unsigned char x2_char = *reinterpret_cast<unsigned char *>(&x2);
  unsigned char x3_char = *reinterpret_cast<unsigned char *>(&x3);
  unsigned char x4_char = *reinterpret_cast<unsigned char *>(&x4);
  unsigned char x5_char = *reinterpret_cast<unsigned char *>(&x5);
  unsigned char x6_char = *reinterpret_cast<unsigned char *>(&x6);
  unsigned char x7_char = *reinterpret_cast<unsigned char *>(&x7);
  unsigned char y0_char = *reinterpret_cast<unsigned char *>(&y0);
  unsigned char y1_char = *reinterpret_cast<unsigned char *>(&y1);
  unsigned char y2_char = *reinterpret_cast<unsigned char *>(&y2);
  unsigned char y3_char = *reinterpret_cast<unsigned char *>(&y3);
  unsigned char y4_char = *reinterpret_cast<unsigned char *>(&y4);
  unsigned char y5_char = *reinterpret_cast<unsigned char *>(&y5);
  unsigned char y6_char = *reinterpret_cast<unsigned char *>(&y6);
  unsigned char y7_char = *reinterpret_cast<unsigned char *>(&y7);
  unsigned int a = ((unsigned int)x3_char << 24) |
                   ((unsigned int)x2_char << 16) |
                   ((unsigned int)x1_char << 8) | (unsigned int)x0_char;
  unsigned int b = ((unsigned int)x7_char << 24) |
                   ((unsigned int)x6_char << 16) |
                   ((unsigned int)x5_char << 8) | (unsigned int)x4_char;
  unsigned int c = ((unsigned int)y3_char << 24) |
                   ((unsigned int)y2_char << 16) |
                   ((unsigned int)y1_char << 8) | (unsigned int)y0_char;
  unsigned int d = ((unsigned int)y7_char << 24) |
                   ((unsigned int)y6_char << 16) |
                   ((unsigned int)y5_char << 8) | (unsigned int)y4_char;
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
