#pragma once

#include "common.h"

#include <cutlass/fast_math.h>

#define hexp cutlass::fast_exp
#define hlog cutlass::fast_log
#define hsqrt cutlass::fast_sqrt
#define hsin cutlass::fast_sin
#define hcos cutlass::fast_cos
#define htanh cutlass::fast_tanh

namespace cutlass {
// CUTLASS lacks 16-bit overloads for these functions except fast_exp and
// fast_tanh on half_t. Evaluate through float and convert back, matching its
// fast_exp fallback. Use fast_* here because the h* aliases would recurse.
TL_DEVICE
bfloat16_t fast_exp(bfloat16_t x) { return bfloat16_t(fast_exp(float(x))); }

TL_DEVICE
half_t fast_log(half_t x) { return half_t(fast_log(float(x))); }

TL_DEVICE
bfloat16_t fast_log(bfloat16_t x) { return bfloat16_t(fast_log(float(x))); }

TL_DEVICE
half_t fast_sqrt(half_t x) { return half_t(fast_sqrt(float(x))); }

TL_DEVICE
bfloat16_t fast_sqrt(bfloat16_t x) { return bfloat16_t(fast_sqrt(float(x))); }

TL_DEVICE
half_t fast_sin(half_t x) { return half_t(fast_sin(float(x))); }

TL_DEVICE
bfloat16_t fast_sin(bfloat16_t x) { return bfloat16_t(fast_sin(float(x))); }

TL_DEVICE
half_t fast_cos(half_t x) { return half_t(fast_cos(float(x))); }

TL_DEVICE
bfloat16_t fast_cos(bfloat16_t x) { return bfloat16_t(fast_cos(float(x))); }

TL_DEVICE
bfloat16_t fast_tanh(bfloat16_t x) { return bfloat16_t(fast_tanh(float(x))); }
} // namespace cutlass

namespace tl {

// Pass operands by value: checking NaN must not re-evaluate the caller's
// expression (which can, for example, be a returning atomic operation).
TL_DEVICE float clamp(float x, float lo, float hi) {
  if (isnan(x))
    return x;
  if (isnan(lo))
    return lo;
  if (isnan(hi))
    return hi;
  return fminf(fmaxf(x, lo), hi);
}

TL_DEVICE double clamp(double x, double lo, double hi) {
  if (isnan(x))
    return x;
  if (isnan(lo))
    return lo;
  if (isnan(hi))
    return hi;
  return fmin(fmax(x, lo), hi);
}

// Narrow formats, including FP8, are exactly representable in float.
template <typename T> TL_DEVICE T clamp(T x, T lo, T hi) {
  return T(clamp(float(x), float(lo), float(hi)));
}

TL_DEVICE half_t clamp(half_t x, half_t lo, half_t hi) {
  return half_t(
      __hmin_nan(__hmax_nan(x.to_half(), lo.to_half()), hi.to_half()));
}

TL_DEVICE bfloat16_t clamp(bfloat16_t x, bfloat16_t lo, bfloat16_t hi) {
  return bfloat16_t(
      __hmin_nan(__hmax_nan(x.to_nv_bfloat16(), lo.to_nv_bfloat16()),
                 hi.to_nv_bfloat16()));
}

TL_DEVICE __half2 clamp2(__half2 x, __half2 lo, __half2 hi) {
  return min2_nan(max2_nan(x, lo), hi);
}

TL_DEVICE __nv_bfloat162 clamp2(__nv_bfloat162 x, __nv_bfloat162 lo,
                                __nv_bfloat162 hi) {
  return min2_nan(max2_nan(x, lo), hi);
}

} // namespace tl
