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

TL_DEVICE float clamp_symmetric(float x, float limit) {
  float result;
  asm("min.NaN.xorsign.abs.f32 %0, %1, %2;"
      : "=f"(result)
      : "f"(x), "f"(limit));
  return result;
}

TL_DEVICE half_t clamp_symmetric(half_t x, half_t limit) {
  uint16_t result;
  asm("min.NaN.xorsign.abs.f16 %0, %1, %2;"
      : "=h"(result)
      : "h"(x.raw()), "h"(limit.raw()));
  return half_t::bitcast(result);
}

TL_DEVICE bfloat16_t clamp_symmetric(bfloat16_t x, bfloat16_t limit) {
  uint16_t result;
  asm("min.NaN.xorsign.abs.bf16 %0, %1, %2;"
      : "=h"(result)
      : "h"(x.raw()), "h"(limit.raw()));
  return bfloat16_t::bitcast(result);
}

TL_DEVICE __half2 clamp_symmetric2(__half2 x, __half2 limit) {
  uint1 result;
  asm("min.NaN.xorsign.abs.f16x2 %0, %1, %2;"
      : "=r"(result.x)
      : "r"(to_uint1(x).x), "r"(to_uint1(limit).x));
  return from_uint1<__half2>(result);
}

TL_DEVICE __nv_bfloat162 clamp_symmetric2(__nv_bfloat162 x,
                                          __nv_bfloat162 limit) {
  uint1 result;
  asm("min.NaN.xorsign.abs.bf16x2 %0, %1, %2;"
      : "=r"(result.x)
      : "r"(to_uint1(x).x), "r"(to_uint1(limit).x));
  return from_uint1<__nv_bfloat162>(result);
}

} // namespace tl
