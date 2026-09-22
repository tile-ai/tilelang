/*!
 * \file magic_div.cc
 * \brief Host-side helpers for magic-number division (CUTLASS FastDivmod).
 *
 * Linked into libtvm_runtime so JIT-compiled host modules (C host via dlopen
 * as well as in-process LLVM JIT) can resolve the symbols as plain extern
 * calls emitted for tir.call_extern.
 */

#include <tvm/ffi/c_api.h>

#include <cstdint>
#include <limits>

namespace {

// CUTLASS FastDivmod magic constants for divisor d >= 1, valid for
// 0 <= x < 2^31: with k = ceil_log2(d), p = 31 + k, M = ceil(2^p / d) < 2^32
// and s = p - 32, floor(x / d) == umulhi(uint32(x), M) >> s for d >= 2.
// Values outside the signed int32 contract return neutral constants. d == 1
// is handled by a device-side select, while all other invalid values use the
// device fallback if the original division executes.
void HostFastDivmodU32(uint32_t d, uint32_t &mul, uint32_t &shift) {
  if (d <= 1 ||
      d > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    mul = 0;
    shift = 0;
    return;
  }
  uint32_t k = 0;
  while ((uint32_t(1) << k) < d) {
    ++k; // ceil_log2(d), k in [1, 31] for d >= 2
  }
  uint32_t p = 31 + k;
  mul = static_cast<uint32_t>(((uint64_t(1) << p) + d - 1) / d);
  shift = p - 32;
}

} // namespace

extern "C" {

/*! \brief Magic multiplier M for divisor d (see HostFastDivmodU32). */
TVM_FFI_DLL_EXPORT uint32_t TileLangHostFastDivmodU32Mul(uint32_t d) {
  uint32_t mul, shift;
  HostFastDivmodU32(d, mul, shift);
  return mul;
}

/*! \brief Magic shift s for divisor d (see HostFastDivmodU32). */
TVM_FFI_DLL_EXPORT int32_t TileLangHostFastDivmodU32Shift(uint32_t d) {
  uint32_t mul, shift;
  HostFastDivmodU32(d, mul, shift);
  return static_cast<int32_t>(shift);
}
}
