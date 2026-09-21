#pragma once

#include "common.h"

namespace tl {

template <typename T>
TL_DEVICE T invariant_div_fallback(T x, T d, bool truncating) {
  T q = x / d;
  T r = x % d;
  return !truncating && r != 0 && ((r < 0) != (d < 0)) ? q - 1 : q;
}

template <typename T>
TL_DEVICE T invariant_rem_fallback(T x, T d, bool truncating) {
  T r = x % d;
  return !truncating && r != 0 && ((r < 0) != (d < 0)) ? r + d : r;
}

TL_DEVICE int fast_div(int x, int d, unsigned multiplier, int shift, bool valid,
                       bool truncating) {
  if (!valid) {
    return invariant_div_fallback(x, d, truncating);
  }
  return d == 1 ? x : int(__umulhi(unsigned(x), multiplier) >> shift);
}

TL_DEVICE int fast_rem(int x, int d, unsigned multiplier, int shift, bool valid,
                       bool truncating) {
  if (!valid) {
    return invariant_rem_fallback(x, d, truncating);
  }
  return x - fast_div(x, d, multiplier, shift, true, true) * d;
}

TL_DEVICE unsigned fast_div(unsigned x, unsigned d, unsigned reciprocal,
                            int shift, bool valid, bool truncating) {
  if (!valid) {
    return x / d;
  }
  if (d == 1) {
    return x;
  }
  unsigned q = __umulhi(x, reciprocal);
  unsigned r = x - q * d;
  return q + unsigned(r >= d);
}

TL_DEVICE unsigned barrett_reduce(unsigned x, unsigned d, unsigned reciprocal,
                                  bool valid, bool truncating) {
  if (!valid) {
    return x % d;
  }
  if (d == 1) {
    return 0;
  }
  unsigned q = __umulhi(x, reciprocal);
  unsigned r = x - q * d;
  return r >= d ? r - d : r;
}

TL_DEVICE int barrett_reduce(int x, int d, unsigned reciprocal, bool valid,
                             bool truncating) {
  if (!valid) {
    return invariant_rem_fallback(x, d, truncating);
  }
  return int(barrett_reduce(unsigned(x), unsigned(d), reciprocal, true, true));
}

// The reciprocal word and result type are independent. The pass checks the
// narrow algorithm's range before using it on a wider dividend.
template <typename X, typename D>
TL_DEVICE X fast_div(X x, D d, unsigned multiplier, int shift, bool valid,
                     bool truncating) {
  if (!valid) {
    return invariant_div_fallback(x, X(d), truncating);
  }
  if constexpr (sizeof(D) == 4 && D(-1) < D(0)) {
    return X(fast_div(int(x), int(d), multiplier, shift, true, truncating));
  } else if constexpr (sizeof(D) == 4) {
    return X(fast_div(unsigned(x), unsigned(d), multiplier, shift, true,
                      truncating));
  } else {
    if (d == 1) {
      return x;
    }
    unsigned q = __umulhi(unsigned(x), multiplier);
    // q*d <= x <= UINT32_MAX, even when the divisor itself is wider.
    unsigned r = unsigned(x) - q * unsigned(d);
    return X(q + unsigned(X(r) >= X(d)));
  }
}

template <typename X, typename D>
TL_DEVICE X fast_rem(X x, D d, unsigned multiplier, int shift, bool valid,
                     bool truncating) {
  if (!valid) {
    return invariant_rem_fallback(x, X(d), truncating);
  }
  // Fast32 validity guarantees 0 <= x <= INT32_MAX and d > 0, so q*d
  // and the remainder fit int32. Widen the result, not the multiply/subtract.
  if constexpr (sizeof(D) == 4 && D(-1) < D(0)) {
    return X(fast_rem(int(x), int(d), multiplier, shift, true, truncating));
  } else {
    return x - fast_div(x, d, multiplier, shift, true, truncating) * X(d);
  }
}

template <typename X, typename D>
TL_DEVICE X barrett_reduce(X x, D d, unsigned reciprocal, bool valid,
                           bool truncating) {
  if (!valid) {
    return invariant_rem_fallback(x, X(d), truncating);
  }
  if constexpr (sizeof(D) == 4) {
    return X(
        barrett_reduce(unsigned(x), unsigned(d), reciprocal, true, truncating));
  } else {
    if (d == 1) {
      return 0;
    }
    unsigned q = __umulhi(unsigned(x), reciprocal);
    unsigned r = unsigned(x) - q * unsigned(d);
    return X(X(r) >= X(d) ? r - unsigned(d) : r);
  }
}

// mu = floor(2^64 / d). The high product underestimates the quotient by at
// most one, so a single correction suffices for the entire uint64 range.
template <typename T>
TL_DEVICE T fast_div(T x, T d, uint64_t reciprocal, int shift, bool valid,
                     bool truncating) {
  if (!valid) {
    return invariant_div_fallback(x, d, truncating);
  }
  if (d == 1) {
    return x;
  }
  uint64_t q = __umul64hi(uint64_t(x), reciprocal);
  uint64_t r = uint64_t(x) - q * uint64_t(d);
  return T(q + uint64_t(r >= uint64_t(d)));
}

template <typename T>
TL_DEVICE T barrett_reduce(T x, T d, uint64_t reciprocal, bool valid,
                           bool truncating) {
  if (!valid) {
    return invariant_rem_fallback(x, d, truncating);
  }
  if (d == 1) {
    return 0;
  }
  uint64_t q = __umul64hi(uint64_t(x), reciprocal);
  uint64_t r = uint64_t(x) - q * uint64_t(d);
  return T(r >= uint64_t(d) ? r - uint64_t(d) : r);
}

TL_DEVICE int exact_div(int x, int d, unsigned inverse, unsigned shift,
                        bool valid) {
  // Divisibility is proven by the pass, so floor and truncation coincide.
  return valid ? int(unsigned(x >> shift) * inverse) : x / d;
}

TL_DEVICE unsigned exact_div(unsigned x, unsigned d, unsigned inverse,
                             unsigned shift, bool valid) {
  return valid ? (x >> shift) * inverse : x / d;
}

template <typename X, typename D>
TL_DEVICE X exact_div(X x, D d, unsigned inverse, unsigned shift, bool valid) {
  return valid ? X(unsigned(x >> shift) * inverse) : x / X(d);
}

} // namespace tl
