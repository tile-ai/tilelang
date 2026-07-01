#pragma once

#include "common.h"

#include <cutlass/fast_math.h>

#define hexp cutlass::fast_exp
#define hlog cutlass::fast_log
#define hsqrt cutlass::fast_sqrt
#define hsin cutlass::fast_sin
#define hcos cutlass::fast_cos
#define htanh cutlass::fast_tanh
#define hpow powf

namespace cutlass {
// Mirror cutlass's own half_t fast_exp (fast_math.h): route through float.
// A direct `return ::hexp(x)` recurses, since `hexp` is #define'd to this
// same cutlass::fast_exp and x is already bfloat16_t.
TL_DEVICE
bfloat16_t fast_exp(bfloat16_t x) { return bfloat16_t(fast_exp(float(x))); }
} // namespace cutlass
