#pragma once

#include "common.h"
#include <type_traits>

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

// Signed arithmetic is reduced to an unsigned magnitude problem. For floor
// division with opposite signs, divide abs(x)-1 and restore -(q+1). This avoids
// computing a remainder just to round a negative quotient. All magnitude and
// sign restoration arithmetic is unsigned, including INT_MIN and a negative d.
template <bool Remainder, bool Magic, typename X, typename D, typename Word>
TL_DEVICE X invariant_divmod(X x, D divisor, Word reciprocal, int shift,
                             bool valid, bool truncating, bool nonnegative,
                             bool positive_divisor) {
  if (!valid) {
    if constexpr (Remainder) {
      return invariant_rem_fallback(x, X(divisor), truncating);
    } else {
      return invariant_div_fallback(x, X(divisor), truncating);
    }
  }
  if constexpr (sizeof(X) == 8 && sizeof(Word) == 4 && sizeof(D) == 4 &&
                std::is_unsigned_v<D>) {
    // The pass already bounds the dividend magnitude to the reciprocal word.
    // Keep the unsigned core narrow and widen only its result.
    if (nonnegative) {
      return X(invariant_divmod<Remainder, Magic>(
          uint32_t(x), uint32_t(divisor), reciprocal, shift, true, truncating,
          true, true));
    }
  }
  using U = std::make_unsigned_t<X>;
  X d = X(divisor);
  if constexpr (std::is_signed_v<X>) {
    if (!truncating && (positive_divisor || d > 0)) {
      // A launch-uniform positive-divisor path avoids the general sign/bias
      // restoration cost. Complementing negative x also handles INT_MIN.
      U sign = U(0) - U(!nonnegative && x < 0);
      U normalized = U(x) ^ sign;
      U result = invariant_divmod<Remainder, Magic>(
          normalized, U(d), reciprocal, shift, true, true, true, true);
      if constexpr (Remainder) {
        return !nonnegative && x < 0 ? X(U(d) - 1 - result) : X(result);
      } else {
        return X(result ^ sign);
      }
    }
  }
  U ax = U(x), ad = U(d);
  bool negative_q = false, negative_r = false, bias = false;
  if constexpr (std::is_signed_v<X>) {
    bool negative_x = !nonnegative && x < 0;
    bool negative_d = !positive_divisor && d < 0;
    U sx = U(0) - U(negative_x), sd = U(0) - U(negative_d);
    ax = (ax ^ sx) - sx;
    ad = (ad ^ sd) - sd;
    negative_q = negative_x != negative_d;
    negative_r = truncating ? negative_x : negative_d;
    bias = !truncating && negative_q && ax != 0;
  }
  Word n = Word(ax - U(bias));
  Word q, r;
  {
    if constexpr (sizeof(Word) == 8) {
      static_assert(!Magic);
      q = Word(__umul64hi(uint64_t(n), uint64_t(reciprocal)));
    } else {
      q = Word(__umulhi(unsigned(n), unsigned(reciprocal)));
    }
    if constexpr (Magic) {
      q >>= shift;
      if constexpr (Remainder) {
        r = n - q * Word(ad);
      }
    } else {
      r = n - q * Word(ad);
      // Compare in the divisor's full width, even for a narrow reciprocal.
      bool correction = U(r) >= ad;
      q += Word(correction);
      r -= correction ? Word(ad) : Word(0);
    }
  }
  // Select identity-divisor results after the reciprocal arithmetic so its
  // common subexpressions remain visible across div/rem and guarded uses.
  if constexpr (Remainder) {
    r = ad == 1 ? Word(0) : r;
    U magnitude = bias ? ad - 1 - U(r) : U(r);
    U sign = U(0) - U(negative_r);
    return X((magnitude ^ sign) - sign);
  } else {
    q = ad == 1 ? n : q;
    U magnitude = U(q) + U(bias);
    U sign = U(0) - U(negative_q);
    return X((magnitude ^ sign) - sign);
  }
}

TL_DEVICE int fast_div(int x, int d, unsigned multiplier, int shift, bool valid,
                       bool truncating, bool nonnegative,
                       bool positive_divisor) {
  return invariant_divmod<false, true>(x, d, multiplier, shift, valid,
                                       truncating, nonnegative,
                                       positive_divisor);
}

TL_DEVICE int fast_rem(int x, int d, unsigned multiplier, int shift, bool valid,
                       bool truncating, bool nonnegative,
                       bool positive_divisor) {
  return invariant_divmod<true, true>(x, d, multiplier, shift, valid,
                                      truncating, nonnegative,
                                      positive_divisor);
}

TL_DEVICE unsigned fast_div(unsigned x, unsigned d, unsigned reciprocal,
                            int shift, bool valid, bool truncating,
                            bool nonnegative, bool positive_divisor) {
  return invariant_divmod<false, false>(x, d, reciprocal, shift, valid,
                                        truncating, nonnegative,
                                        positive_divisor);
}

TL_DEVICE unsigned barrett_reduce(unsigned x, unsigned d, unsigned reciprocal,
                                  bool valid, bool truncating, bool nonnegative,
                                  bool positive_divisor) {
  return invariant_divmod<true, false>(x, d, reciprocal, 0, valid, truncating,
                                       nonnegative, positive_divisor);
}

TL_DEVICE int barrett_reduce(int x, int d, unsigned reciprocal, bool valid,
                             bool truncating, bool nonnegative,
                             bool positive_divisor) {
  return invariant_divmod<true, false>(x, d, reciprocal, 0, valid, truncating,
                                       nonnegative, positive_divisor);
}

// A signed-int32 divisor identifies the magic algorithm for these overloads.
// Barrett quotient lowering widens that operand when necessary to disambiguate.
// The pass proves the normalized dividend fits the reciprocal's word size.
template <typename X, typename D>
TL_DEVICE X fast_div(X x, D d, unsigned multiplier, int shift, bool valid,
                     bool truncating, bool nonnegative, bool positive_divisor) {
  constexpr bool magic = sizeof(D) == 4 && std::is_signed_v<D>;
  return invariant_divmod<false, magic>(x, d, multiplier, shift, valid,
                                        truncating, nonnegative,
                                        positive_divisor);
}

template <typename X, typename D>
TL_DEVICE X fast_rem(X x, D d, unsigned multiplier, int shift, bool valid,
                     bool truncating, bool nonnegative, bool positive_divisor) {
  constexpr bool magic = sizeof(D) == 4 && std::is_signed_v<D>;
  return invariant_divmod<true, magic>(x, d, multiplier, shift, valid,
                                       truncating, nonnegative,
                                       positive_divisor);
}

template <typename X, typename D>
TL_DEVICE X barrett_reduce(X x, D d, unsigned reciprocal, bool valid,
                           bool truncating, bool nonnegative,
                           bool positive_divisor) {
  return invariant_divmod<true, false>(x, d, reciprocal, 0, valid, truncating,
                                       nonnegative, positive_divisor);
}

// mu = floor(2^64 / abs(d)); signed/unsigned and floor/trunc restoration share
// the unsigned Barrett core, but unsigned instantiations erase all sign logic.
template <typename T>
TL_DEVICE T fast_div(T x, T d, uint64_t reciprocal, int shift, bool valid,
                     bool truncating, bool nonnegative, bool positive_divisor) {
  return invariant_divmod<false, false>(x, d, reciprocal, shift, valid,
                                        truncating, nonnegative,
                                        positive_divisor);
}

template <typename T>
TL_DEVICE T barrett_reduce(T x, T d, uint64_t reciprocal, bool valid,
                           bool truncating, bool nonnegative,
                           bool positive_divisor) {
  return invariant_divmod<true, false>(x, d, reciprocal, 0, valid, truncating,
                                       nonnegative, positive_divisor);
}

template <typename X, typename D>
TL_DEVICE X exact_div(X x, D d, unsigned inverse, unsigned shift, bool valid) {
  if (!valid) {
    return x / X(d);
  }
  // Proven divisibility makes the arithmetic shift exact for signed x too.
  unsigned quotient = unsigned(x >> shift) * inverse;
  if constexpr (std::is_signed_v<D>) {
    quotient = d < 0 ? 0u - quotient : quotient;
  }
  return X(quotient);
}

} // namespace tl
