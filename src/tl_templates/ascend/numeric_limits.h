#pragma once

#include <bit>
#include <cstdint>

namespace tl {
namespace limits {

// --- float (fp32) ---
inline constexpr float kFloatInf = std::bit_cast<float>(0x7F800000u);
inline constexpr float kFloatNInf = std::bit_cast<float>(0xFF800000u);
inline constexpr float kFloatNaN = std::bit_cast<float>(0x7FC00000u);

// --- half (fp16) ---
inline constexpr half kHalfInf = std::bit_cast<half>(uint16_t{0x7C00u});
inline constexpr half kHalfNInf = std::bit_cast<half>(uint16_t{0xFC00u});
inline constexpr half kHalfNaN = std::bit_cast<half>(uint16_t{0x7E00u});

// --- bfloat16 ---
inline constexpr bfloat16_t kBf16Inf =
    std::bit_cast<bfloat16_t>(uint16_t{0x7F80u});
inline constexpr bfloat16_t kBf16NInf =
    std::bit_cast<bfloat16_t>(uint16_t{0xFF80u});
inline constexpr bfloat16_t kBf16NaN =
    std::bit_cast<bfloat16_t>(uint16_t{0x7FC0u});

} // namespace limits
} // namespace tl
