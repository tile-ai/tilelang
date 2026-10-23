#pragma once

namespace tvm::codegen {
// Self-contained C definitions: host modules need no TileLang runtime headers.
inline constexpr const char *kCBitIntrinsics = R"TL(
#include <stdint.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

static inline int tl_clz32(uint32_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return value ? __builtin_clz(value) : 32;
#elif defined(_MSC_VER)
  unsigned long index;
  return _BitScanReverse(&index, value) ? 31 - (int)index : 32;
#else
  int count = 32;
  for (; value; value >>= 1) {
    --count;
  }
  return count;
#endif
}

static inline int tl_clz64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return value ? __builtin_clzll(value) : 64;
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
  unsigned long index;
  return _BitScanReverse64(&index, value) ? 63 - (int)index : 64;
#else
  uint32_t high = (uint32_t)(value >> 32);
  return high ? tl_clz32(high) : 32 + tl_clz32((uint32_t)value);
#endif
}

)TL";
} // namespace tvm::codegen
