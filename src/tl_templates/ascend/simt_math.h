#pragma once

#include "tl_templates/ascend/cann_compat_names.h"
#include <simt_api/asc_bf16.h>
#include <simt_api/math_functions.h>
#include <stdint.h>
#include <version/asc_devkit_version.h>

// The host pass also reads this header, but the SDK declares hexp2 and
// hexp10 only for the device pass.
#if defined(__NPU_ARCH__)
namespace tl::cann::simt {

#if TL_CANN_9_1 || TL_CANN_9_2_0_OR_9_3_0
namespace detail {
__simt_callee__ inline uint32_t bits(float x) {
  return __builtin_bit_cast(uint32_t, x);
}
__simt_callee__ inline float from_bits(uint32_t x) {
  return __builtin_bit_cast(float, x);
}
constexpr uint32_t sign_bit = 0x80000000u;
constexpr uint32_t infinity = 0x7f800000u;
constexpr uint32_t min_normal = 0x00800000u;
constexpr float pi = 0x1.921fb6p+1f;

__simt_callee__ inline float with_sign(float magnitude, float sign) {
  return from_bits((bits(magnitude) & ~sign_bit) | (bits(sign) & sign_bit));
}

// Positive, finite, nonzero x = mantissa * 2^exponent, with 1 <= mantissa < 2.
// Normalize subnormals as integers before any SIMT division can flush them.
struct Fraction {
  float mantissa;
  int exponent;
};
__simt_callee__ inline Fraction normalize(uint32_t x) {
  int exponent = int(x >> 23) - 127;
  if (x < min_normal) {
    int shift = __builtin_clz(x) - 8;
    x <<= shift;
    exponent = -126 - shift;
  }
  return {from_bits((x & 0x7fffffu) | 0x3f800000u), exponent};
}

// x must be positive and normal. Round once when scaling into the subnormal
// range.
__simt_callee__ inline float scale(float x, int shift) {
  uint32_t word = bits(x);
  int exponent = int(word >> 23) + shift;
  if (exponent >= 255)
    return from_bits(infinity);
  if (exponent > 0)
    return from_bits((word & 0x7fffffu) | (uint32_t(exponent) << 23));
  if (exponent < -23)
    return 0.0f;
  int right = 1 - exponent;
  uint32_t sig = (word & 0x7fffffu) | min_normal;
  uint32_t truncated = sig >> right;
  uint32_t bias = (1u << (right - 1)) - 1 + (truncated & 1);
  return from_bits((sig + bias) >> right);
}

// Atan on [0,1]. Reduction about pi/4 bounds the alternating series argument
// by tan(pi/8); the first omitted term is smaller than 3e-9 there.
__simt_callee__ inline float atan_unit(float x) {
  if (x < 0x1p-12f)
    return x;
  bool reduce = x > 0.41421356237f;
  float t = reduce ? (x - 1.0f) / (x + 1.0f) : x;
  float z = -t * t;
  float p = 1.0f / 17.0f;
  for (int k = 15; k >= 3; k -= 2)
    p = ::fmaf(z, p, 1.0f / float(k));
  float value = ::fmaf(t * z, p, t);
  return reduce ? pi * 0.25f + value : value;
}

// Both operands are positive finite magnitudes, and small < large.
__simt_callee__ inline float atan_ratio(uint32_t small, uint32_t large,
                                        bool preserve_tiny_angle) {
  Fraction a = normalize(small), b = normalize(large);
  int exponent = a.exponent - b.exponent;
  if (preserve_tiny_angle && exponent <= -126) {
    // Here FP32 values are spaced by 2^-149. Divide in those units to avoid
    // rounding the quotient twice. At a midpoint atan(t) is just below t.
    int shift = exponent + 149;
    if (shift < -1)
      return 0.0f;
    uint32_t ma = (bits(a.mantissa) & 0x7fffffu) | min_normal;
    uint32_t mb = (bits(b.mantissa) & 0x7fffffu) | min_normal;
    uint64_t numerator = uint64_t(ma) << (shift > 0 ? shift : 0);
    uint32_t denominator = shift < 0 ? 2 * mb : mb;
    uint32_t whole = uint32_t(numerator / denominator);
    uint32_t remainder = uint32_t(numerator - uint64_t(whole) * denominator);
    return from_bits(whole + (2 * remainder > denominator));
  }
  return atan_unit(scale(a.mantissa / b.mantissa, exponent));
}
} // namespace detail

__simt_callee__ inline float atan2f(float y, float x) {
  using namespace detail;
  uint32_t ax = bits(x) & ~sign_bit, ay = bits(y) & ~sign_bit;
#if TL_CANN_9_2_0_OR_9_3_0
  // Preserve native special values and ordinary angles. Renormalize only
  // subnormal operands or a positive-x ratio small enough to underflow.
  if (ax == 0 || ay == 0 || ax >= infinity || ay >= infinity ||
      (ax >= min_normal && ay >= min_normal &&
       ((bits(x) & sign_bit) || int(ax >> 23) - int(ay >> 23) < 126)))
    return ::atan2f(y, x);
#endif
  if (ax > infinity || ay > infinity)
    return x + y;
  float angle;
  if (ay == 0 || (ax == infinity && ay != infinity))
    angle = 0.0f;
  else if (ax == 0 || (ay == infinity && ax != infinity))
    angle = pi * 0.5f;
  else if (ax == ay)
    angle = pi * 0.25f;
  else if (ay < ax)
    angle = atan_ratio(ay, ax, (bits(x) & sign_bit) == 0);
  else
    angle = pi * 0.5f - atan_ratio(ax, ay, false);
  return with_sign((bits(x) & sign_bit) ? pi - angle : angle, y);
}

#if TL_CANN_9_1
__simt_callee__ inline float copysignf(float x, float y) {
  return detail::with_sign(x, y);
}

__simt_callee__ inline float atanf(float x) {
  float a = ::fabsf(x);
  if (a <= 1.0f)
    return detail::with_sign(detail::atan_unit(a), x);
  return detail::with_sign(detail::pi * 0.5f - detail::atan_unit(1.0f / a), x);
}

__simt_callee__ inline float asinf(float x) {
  return simt::atan2f(x, ::sqrtf(::fmaf(-x, x, 1.0f)));
}
__simt_callee__ inline float acosf(float x) {
  return simt::atan2f(::sqrtf(::fmaf(-x, x, 1.0f)), x);
}

__simt_callee__ inline float hypotf(float x, float y) {
  using namespace detail;
  uint32_t ax = bits(x) & ~sign_bit, ay = bits(y) & ~sign_bit;
  if (ax == infinity || ay == infinity)
    return from_bits(infinity);
  if (ax > infinity || ay > infinity)
    return x + y;
  if (ax >= min_normal && ay >= min_normal)
    return ::hypotf(x, y);
  uint32_t large = ax > ay ? ax : ay, small = ax > ay ? ay : ax;
  if (small == 0)
    return from_bits(large);
  Fraction a = normalize(large), b = normalize(small);
  if (a.exponent - b.exponent > 13)
    return from_bits(large);
  float scaled = scale(b.mantissa, b.exponent - a.exponent);
  return scale(::sqrtf(::fmaf(a.mantissa, a.mantissa, scaled * scaled)),
               a.exponent);
}

__simt_callee__ inline float fmodf(float x, float y) {
  using namespace detail;
  uint32_t ax = bits(x) & ~sign_bit, ay = bits(y) & ~sign_bit;
  if (ax >= infinity || ay > infinity || ay == 0)
    return ::fmodf(x, y);
  if (ax < ay)
    return x;
  // The native remainder loses low bits for a subnormal divisor.
  if (ay >= min_normal)
    return ::fmodf(x, y);
  Fraction a = normalize(ax), b = normalize(ay);
  uint32_t r = (bits(a.mantissa) & 0x7fffffu) | min_normal;
  uint32_t divisor = (bits(b.mantissa) & 0x7fffffu) | min_normal;
  for (int i = a.exponent; i > b.exponent; --i) {
    if (r >= divisor)
      r -= divisor;
    if (r == 0)
      return with_sign(0.0f, x);
    r <<= 1;
  }
  if (r >= divisor)
    r -= divisor;
  return with_sign(r ? scale(float(r), b.exponent - 23) : 0.0f, x);
}

// Compensate for the rounding of 1+x before taking its logarithm.
__simt_callee__ inline float log1pf(float x) {
  if (::fabsf(x) < 0x1p-23f)
    return x;
  float u = 1.0f + x;
  if (!(u > 0.0f) || u == detail::from_bits(detail::infinity))
    return ::logf(u);
  return ::logf(u) + (x - (u - 1.0f)) / u;
}
__simt_callee__ inline float atanhf(float x) {
  float a = ::fabsf(x);
  if (a < 0x1p-12f)
    return x;
  return detail::with_sign(0.5f * simt::log1pf((2.0f * a) / (1.0f - a)), x);
}
__simt_callee__ inline float asinhf(float x) {
  float a = ::fabsf(x);
  if (a < 0x1p-12f)
    return x;
  float result =
      a >= 0x1p12f
          ? ::logf(a) + 0x1.62e43p-1f
          : simt::log1pf(a + a * a / (1.0f + ::sqrtf(::fmaf(a, a, 1.0f))));
  return detail::with_sign(result, x);
}
__simt_callee__ inline float acoshf(float x) {
  if (x >= 0x1p12f)
    return ::logf(x) + 0x1.62e43p-1f;
  float t = x - 1.0f;
  if (x < 1.0f)
    return detail::from_bits(0x7fc00000u);
  return simt::log1pf(t + ::sqrtf(t * (x + 1.0f)));
}

namespace detail {
// exp(x)/2 without premature overflow for x near log(FLT_MAX).
__simt_callee__ inline float exp_over_two(float x) {
  if (x < 80.0f)
    return 0.5f * ::expf(x);
  float h = ::expf(0.5f * x);
  return (0.5f * h) * h;
}
__simt_callee__ inline float sinh_small(float x) {
  float z = x * x;
  float p = ::fmaf(z, 1.0f / 39916800.0f, 1.0f / 362880.0f);
  p = ::fmaf(z, p, 1.0f / 5040.0f);
  p = ::fmaf(z, p, 1.0f / 120.0f);
  return ::fmaf(x * z, ::fmaf(z, p, 1.0f / 6.0f), x);
}
} // namespace detail
__simt_callee__ inline float sinhf(float x) {
  float a = ::fabsf(x);
  if (a < 0x1p-12f)
    return x;
  if (a <= 1.0f)
    return detail::sinh_small(x);
  float h = detail::exp_over_two(a);
  return detail::with_sign(h - 0.25f / h, x);
}
__simt_callee__ inline float coshf(float x) {
  float h = detail::exp_over_two(::fabsf(x));
  return h + 0.25f / h;
}
__simt_callee__ inline float tanhf(float x) {
  float a = ::fabsf(x);
  if (a < 0x1p-12f)
    return x;
  if (a <= 1.0f) {
    float s = detail::sinh_small(x);
    return s / ::sqrtf(::fmaf(s, s, 1.0f));
  }
  float e = ::expf(-2.0f * a);
  return detail::with_sign((1.0f - e) / (1.0f + e), x);
}

#endif

namespace detail {
// Keep a product's rounding error until after reduction to a small exponent.
// A one-ULP error in the unreduced exponent can cost many ULPs in pow/exp10.
struct Split {
  float hi;
  float lo;
};
// Add a correction smaller in magnitude than the leading term.
__simt_callee__ inline Split renormalize(float lead, float tail) {
  float hi = lead + tail;
  return {hi, (lead - hi) + tail};
}
__simt_callee__ inline Split multiply(Split a, float y) {
  float hi = a.hi * y;
  return renormalize(hi, ::fmaf(a.hi, y, -hi) + a.lo * y);
}
__simt_callee__ inline float exp2_split(Split x) {
  if (x.hi > 129.0f)
    return from_bits(infinity);
  // 2^-150 is halfway between zero and the least positive float.
  if (x.hi < -150.0f || (x.hi == -150.0f && x.lo <= 0.0f))
    return 0.0f;
  if (x.hi < -149.0f)
    return from_bits(1);
  if (x.hi == 128.0f) {
    // Here |lo| <= 2^-17. Evaluate exp(t) as 1 + t + t^2/2 with a
    // fused final add. Rounding up to 1 too early would overflow at scale 128.
    float t = x.lo * 0x1.62e43p-1f;
    return scale(::fmaf(t, 1.0f + 0.5f * t, 1.0f), 128);
  }
  int n = int(::rintf(x.hi));
  float r = (x.hi - float(n)) + x.lo;
  return scale(::expf(r * 0x1.62e43p-1f), n);
}
__simt_callee__ inline Split log2_split(uint32_t ax) {
  Fraction a = normalize(ax);
  float m = a.mantissa;
  if (m > 0x1.6a09e6p0f) {
    m *= 0.5f;
    ++a.exponent;
  }
  // log(m) = 2*(s + s^3/3 + ...), |s| <= 0.172. Keep the low
  // part of s; pow amplifies log errors when the exponent is large.
  float u = m - 1.0f, v = m + 1.0f;
  float s = u / v;
  float s_error = (::fmaf(-s, v, u) - s * (m - (v - 1.0f))) / v;
  float z = s * s;
  Split square = {z, ::fmaf(s, s, -z)};
  float p = 1.0f / 19.0f;
  for (int k = 17; k >= 7; k -= 2)
    p = ::fmaf(square.hi, p, 1.0f / float(k));

  // Keep the rounding errors of the leading terms of 1 + s^2/3 + s^4/5.
  Split poly = {p, 0.0f};
  for (int k = 5; k >= 1; k -= 2) {
    float coeff = 1.0f / float(k);
    float coeff_lo = ::fmaf(-float(k), coeff, 1.0f) / float(k);
    Split prod = multiply(poly, square.hi);
    Split sum = renormalize(coeff, prod.hi);
    poly = {sum.hi, sum.lo + (prod.lo + poly.hi * square.lo + coeff_lo)};
  }
  Split series = multiply(poly, s);
  // atanh(s + ds) differs from atanh(s) by ds / (1 - s^2) to first order.
  series.lo += s_error / (1.0f - square.hi);
  constexpr float log2e_hi = 0x1.715476p0f, log2e_lo = 0x1.4ae0cp-26f;
  Split log = multiply(series, 2.0f * log2e_hi);
  log = renormalize(log.hi, log.lo + (2.0f * series.hi) * log2e_lo);
  if (a.exponent == 0)
    return log;
  Split result = renormalize(float(a.exponent), log.hi);
  return renormalize(result.hi, result.lo + log.lo);
}
} // namespace detail

__simt_callee__ inline float powf(float x, float y) {
  using namespace detail;
  uint32_t ax = bits(x) & ~sign_bit, ay = bits(y) & ~sign_bit;
#if TL_CANN_9_2_0_OR_9_3_0
  // Leave SDK special-value and domain handling intact.
  if (ax == 0 || ax >= infinity || ay >= infinity ||
      ((bits(x) & sign_bit) && ::truncf(y) != y))
    return ::powf(x, y);
#endif
  if (ay == 0 || x == 1.0f)
    return 1.0f;
  if (ax > infinity || ay > infinity)
    return x + y;
  bool positive_y = !(bits(y) & sign_bit);
  if (ay == infinity) {
    if (ax == 0x3f800000u)
      return 1.0f;
    return ((ax > 0x3f800000u) == positive_y) ? from_bits(infinity) : 0.0f;
  }
  // At magnitude 2^24 and above every representable float is an even integer.
  bool integer = ay >= 0x3f800000u && ::truncf(y) == y;
  bool odd = integer && ay < 0x4b800000u && (int(y) & 1);
  float sign = odd ? x : 1.0f;
  if (ax == 0 || ax == infinity) {
    bool infinite = (ax == infinity) == positive_y;
    return with_sign(infinite ? from_bits(infinity) : 0.0f, sign);
  }
  if ((bits(x) & sign_bit) && !integer)
    return from_bits(0x7fc00000u);
  if (ax == 0x3f800000u)
    return with_sign(1.0f, sign);
  // Even the float nearest to one is past overflow or underflow here.
  if (ay >= 0x4f800000u)
    return ((ax > 0x3f800000u) == positive_y) ? from_bits(infinity) : 0.0f;
#if TL_CANN_9_2_0_OR_9_3_0
  // For 2^e <= |x| < 2^(e+1), |log2(|x|)| <= max(-e, e+1).
  // A product below 126 keeps |x^y| strictly between 2^-126 and 2^126.
  int e = normalize(ax).exponent;
  int log2_bound = e < 0 ? -e : e + 1;
  if (from_bits(ay) * float(log2_bound) < 126.0f)
    return ::powf(x, y);
#endif
  return with_sign(exp2_split(multiply(log2_split(ax), y)), sign);
}
#else
using ::atan2f;
using ::powf;
#endif

#if TL_CANN_9_1
__simt_callee__ inline float exp2f(float x) {
  if (::isnan(x))
    return x;
  return detail::exp2_split({x, 0.0f});
}
__simt_callee__ inline float exp10f(float x) {
  if (::isnan(x))
    return x;
  if (x > 40.0f)
    return detail::from_bits(detail::infinity);
  if (x < -46.0f)
    return 0.0f;
  return detail::exp2_split(
      detail::multiply({0x1.a934fp1f, 0x1.2f346ep-24f}, x));
}

namespace detail {
// Round the computed float to BF16 even when its exponent is subnormal.
__simt_callee__ inline bfloat16_t to_bf16(float x) {
  uint32_t u = bits(x);
  if ((u & ~sign_bit) > infinity)
    return __ushort_as_bfloat16((u >> 16) | 0x40u);
  return __ushort_as_bfloat16((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}
} // namespace detail

__simt_callee__ inline bfloat16_t hexp2(bfloat16_t x) {
  return detail::to_bf16(simt::exp2f(__bfloat162float(x)));
}
__simt_callee__ inline bfloat16_t hexp10(bfloat16_t x) {
  return detail::to_bf16(simt::exp10f(__bfloat162float(x)));
}
// The 9.1 correction above is BF16-only. Keep the native FP16 overloads.
__simt_callee__ inline half hexp2(half x) { return ::hexp2(x); }
__simt_callee__ inline half hexp10(half x) { return ::hexp10(x); }
#else
#define TL_CANN_USE_NATIVE(name) using ::name;
TL_CANN_SIMT_MATH_9_1_ONLY_NAMES(TL_CANN_USE_NATIVE)
#undef TL_CANN_USE_NATIVE
#endif

} // namespace tl::cann::simt

#define TL_CANN_BIND_MATH(name) using ::tl::cann::simt::name;
#define TL_CANN_USE_SIMT_MATH_NAMES TL_CANN_SIMT_MATH_NAMES(TL_CANN_BIND_MATH)
#else
#define TL_CANN_USE_SIMT_MATH_NAMES
#endif // defined(__NPU_ARCH__)
