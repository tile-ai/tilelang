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
