#pragma once

#include "../common.h"
#include <cute/arch/config.hpp>

#ifndef __CUDACC_RTC__
#include <cstdint>
#endif

// The SM120 block-scaled warp MMA also runs on SM121 (GB10). CUTLASS defines
// CUTLASS_ARCH_MMA_SM120A_ENABLED only when __CUDA_ARCH__ == 1200, so accept
// both arch-specific (a) and family-specific (f) SM120 and SM121 targets.
#if (defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) ||                               \
     defined(CUTLASS_ARCH_MMA_SM120F_ENABLED) ||                               \
     defined(CUTLASS_ARCH_MMA_SM121A_ENABLED) ||                               \
     defined(CUTLASS_ARCH_MMA_SM121F_ENABLED))
#define TL_SM120_BLOCK_SCALED_MMA_ENABLED 1
#endif

namespace tl {

enum class SM120MmaBlockScaledKind : int {
  kMxf4nvf4 = 0,
  kMxf8f6f4 = 1,
};

enum class SM120MmaScaleType : int {
  kUE4M3 = 0,
  kUE8M0 = 1,
};

// Operand element types of kind::mxf8f6f4. Each element occupies one byte of
// the A/B registers; E2M1 sits in bits [5:2] and E3M2/E2M3 in bits [7:2].
enum class SM120MmaOperandType : int {
  kE2M1 = 0,
  kE3M2 = 1,
  kE2M3 = 2,
  kE4M3 = 3,
  kE5M2 = 4,
};

template <SM120MmaBlockScaledKind Kind, int ScaleVecSize,
          SM120MmaScaleType SType>
struct SM120MmaBlockScaledConfig {
  static constexpr bool kSupported = false;
};

template <>
struct SM120MmaBlockScaledConfig<SM120MmaBlockScaledKind::kMxf4nvf4, 4,
                                 SM120MmaScaleType::kUE4M3> {
  static constexpr bool kSupported = true;
};

template <>
struct SM120MmaBlockScaledConfig<SM120MmaBlockScaledKind::kMxf4nvf4, 2,
                                 SM120MmaScaleType::kUE8M0> {
  static constexpr bool kSupported = true;
};

template <>
struct SM120MmaBlockScaledConfig<SM120MmaBlockScaledKind::kMxf8f6f4, 1,
                                 SM120MmaScaleType::kUE8M0> {
  static constexpr bool kSupported = true;
};

namespace detail {

// SM120a NVF4 block-scaled warp MMA:
// mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs(
    float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, const float *c, uint32_t scale_a, uint32_t scale_b,
    uint16_t scale_a_byte_id = 0, uint16_t scale_a_thread_id = 0,
    uint16_t scale_b_byte_id = 0, uint16_t scale_b_thread_id = 0) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED) &&                        \
    defined(TL_SM120_BLOCK_SCALED_MMA_ENABLED)
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::"
      "4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13}, "
      "{%14}, {%15, %16}, "
      "{%17}, {%18, %19};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]),
        "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(scale_a), "h"(scale_a_byte_id),
        "h"(scale_a_thread_id), "r"(scale_b), "h"(scale_b_byte_id),
        "h"(scale_b_thread_id));
#else
  CUTE_INVALID_CONTROL_PATH(
      "tl::sm120_mma_sync_blockscaled requires sm_120a/f or sm_121a/f and "
      "CUDA 12.8 or later");
#endif
}

TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3(
    float *d, const uint32_t *a, const uint32_t *b, const float *c,
    uint32_t scale_a, uint32_t scale_b, uint16_t scale_a_byte_id = 0,
    uint16_t scale_a_thread_id = 0, uint16_t scale_b_byte_id = 0,
    uint16_t scale_b_thread_id = 0) {
  sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs(
      d, a[0], a[1], a[2], a[3], b[0], b[1], c, scale_a, scale_b,
      scale_a_byte_id, scale_a_thread_id, scale_b_byte_id, scale_b_thread_id);
}

// SM120a MXFP4 block-scaled warp MMA, one UE8M0 scale per 32 K elements:
// mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X
// The two scales of a K64 atom sit in one half of the scale word; the byte
// selector (0 or 2) picks the half.
TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_2x_ue8m0(
    float *d, const uint32_t *a, const uint32_t *b, const float *c,
    uint32_t scale_a, uint32_t scale_b, uint16_t scale_a_byte_id,
    uint16_t scale_a_thread_id, uint16_t scale_b_byte_id,
    uint16_t scale_b_thread_id) {
#if defined(CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED) &&                        \
    defined(TL_SM120_BLOCK_SCALED_MMA_ENABLED)
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::"
      "2X.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13}, "
      "{%14}, {%15, %16}, "
      "{%17}, {%18, %19};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(scale_a),
        "h"(scale_a_byte_id), "h"(scale_a_thread_id), "r"(scale_b),
        "h"(scale_b_byte_id), "h"(scale_b_thread_id));
#else
  CUTE_INVALID_CONTROL_PATH(
      "tl::sm120_mma_sync_blockscaled kind::mxf4nvf4 scale_vec::2X requires "
      "sm_120a/f or sm_121a/f and CUDA 12.8 or later");
#endif
}

// SM120a MX block-scaled warp MMA, one UE8M0 scale per 32 K elements:
// mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X
// The byte selector picks one of the four K32 scales packed in a scale word.
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED) &&                                 \
    defined(TL_SM120_BLOCK_SCALED_MMA_ENABLED)
#define TL_SM120_MXF8F6F4_MMA(ATYPE, BTYPE)                                    \
  TL_DEVICE void sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##ATYPE##_##BTYPE(       \
      float *d, const uint32_t *a, const uint32_t *b, const float *c,          \
      uint32_t scale_a, uint32_t scale_b, uint16_t scale_a_byte_id,            \
      uint16_t scale_a_thread_id, uint16_t scale_b_byte_id,                    \
      uint16_t scale_b_thread_id) {                                            \
    asm volatile(                                                              \
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4."                    \
        "block_scale.scale_vec::1X.f32." #ATYPE "." #BTYPE ".f32.ue8m0 "       \
        "{%0, %1, %2, %3}, "                                                   \
        "{%4, %5, %6, %7}, "                                                   \
        "{%8, %9}, "                                                           \
        "{%10, %11, %12, %13}, "                                               \
        "{%14}, {%15, %16}, "                                                  \
        "{%17}, {%18, %19};\n"                                                 \
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                       \
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),    \
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(scale_a),            \
          "h"(scale_a_byte_id), "h"(scale_a_thread_id), "r"(scale_b),          \
          "h"(scale_b_byte_id), "h"(scale_b_thread_id));                       \
  }
#else
#define TL_SM120_MXF8F6F4_MMA(ATYPE, BTYPE)                                    \
  TL_DEVICE void sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##ATYPE##_##BTYPE(       \
      float *, const uint32_t *, const uint32_t *, const float *, uint32_t,    \
      uint32_t, uint16_t, uint16_t, uint16_t, uint16_t) {                      \
    CUTE_INVALID_CONTROL_PATH(                                                 \
        "tl::sm120_mma_sync_blockscaled kind::mxf8f6f4 requires sm_120a/f or " \
        "sm_121a/f and CUDA 12.8 or later");                                   \
  }
#endif

#define TL_SM120_MXF8F6F4_MMA_FOR_A(ATYPE)                                     \
  TL_SM120_MXF8F6F4_MMA(ATYPE, e2m1)                                           \
  TL_SM120_MXF8F6F4_MMA(ATYPE, e3m2)                                           \
  TL_SM120_MXF8F6F4_MMA(ATYPE, e2m3)                                           \
  TL_SM120_MXF8F6F4_MMA(ATYPE, e4m3)                                           \
  TL_SM120_MXF8F6F4_MMA(ATYPE, e5m2)

TL_SM120_MXF8F6F4_MMA_FOR_A(e2m1)
TL_SM120_MXF8F6F4_MMA_FOR_A(e3m2)
TL_SM120_MXF8F6F4_MMA_FOR_A(e2m3)
TL_SM120_MXF8F6F4_MMA_FOR_A(e4m3)
TL_SM120_MXF8F6F4_MMA_FOR_A(e5m2)

#undef TL_SM120_MXF8F6F4_MMA_FOR_A
#undef TL_SM120_MXF8F6F4_MMA

#define TL_SM120_MXF8F6F4_DISPATCH_B(ATYPE)                                    \
  if constexpr (BType == SM120MmaOperandType::kE2M1) {                         \
    sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##ATYPE##_e2m1(                       \
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,      \
        scale_b_byte_id, scale_b_thread_id);                                   \
  } else if constexpr (BType == SM120MmaOperandType::kE3M2) {                  \
    sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##ATYPE##_e3m2(                       \
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,      \
        scale_b_byte_id, scale_b_thread_id);                                   \
  } else if constexpr (BType == SM120MmaOperandType::kE2M3) {                  \
    sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##ATYPE##_e2m3(                       \
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,      \
        scale_b_byte_id, scale_b_thread_id);                                   \
  } else if constexpr (BType == SM120MmaOperandType::kE4M3) {                  \
    sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##ATYPE##_e4m3(                       \
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,      \
        scale_b_byte_id, scale_b_thread_id);                                   \
  } else {                                                                     \
    sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##ATYPE##_e5m2(                       \
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,      \
        scale_b_byte_id, scale_b_thread_id);                                   \
  }

template <SM120MmaOperandType AType, SM120MmaOperandType BType>
TL_DEVICE void sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0(
    float *d, const uint32_t *a, const uint32_t *b, const float *c,
    uint32_t scale_a, uint32_t scale_b, uint16_t scale_a_byte_id,
    uint16_t scale_a_thread_id, uint16_t scale_b_byte_id,
    uint16_t scale_b_thread_id) {
  if constexpr (AType == SM120MmaOperandType::kE2M1) {
    TL_SM120_MXF8F6F4_DISPATCH_B(e2m1)
  } else if constexpr (AType == SM120MmaOperandType::kE3M2) {
    TL_SM120_MXF8F6F4_DISPATCH_B(e3m2)
  } else if constexpr (AType == SM120MmaOperandType::kE2M3) {
    TL_SM120_MXF8F6F4_DISPATCH_B(e2m3)
  } else if constexpr (AType == SM120MmaOperandType::kE4M3) {
    TL_SM120_MXF8F6F4_DISPATCH_B(e4m3)
  } else {
    TL_SM120_MXF8F6F4_DISPATCH_B(e5m2)
  }
}

#undef TL_SM120_MXF8F6F4_DISPATCH_B

} // namespace detail

template <SM120MmaBlockScaledKind Kind, int ScaleVecSize,
          SM120MmaScaleType SType,
          SM120MmaOperandType AType = SM120MmaOperandType::kE2M1,
          SM120MmaOperandType BType = SM120MmaOperandType::kE2M1>
TL_DEVICE void sm120_mma_sync_blockscaled(float *d, const uint32_t *a,
                                          const uint32_t *b, const float *c,
                                          uint32_t scale_a, uint32_t scale_b,
                                          uint16_t scale_a_byte_id = 0,
                                          uint16_t scale_a_thread_id = 0,
                                          uint16_t scale_b_byte_id = 0,
                                          uint16_t scale_b_thread_id = 0) {
  static_assert(
      SM120MmaBlockScaledConfig<Kind, ScaleVecSize, SType>::kSupported,
      "Unsupported sm120 mma.block_scale configuration: kind::mxf4nvf4 needs "
      "scale_vec::4X with ue4m3 or scale_vec::2X with ue8m0 scales, "
      "kind::mxf8f6f4 needs scale_vec::1X with ue8m0 scales");
  if constexpr (Kind == SM120MmaBlockScaledKind::kMxf4nvf4) {
    static_assert(AType == SM120MmaOperandType::kE2M1 &&
                      BType == SM120MmaOperandType::kE2M1,
                  "kind::mxf4nvf4 only supports e2m1 operands");
    if constexpr (ScaleVecSize == 4) {
      detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3(
          d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
          scale_b_byte_id, scale_b_thread_id);
    } else {
      detail::sm120_mma_m16n8k64_mxf4nvf4_2x_ue8m0(
          d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
          scale_b_byte_id, scale_b_thread_id);
    }
  } else {
    detail::sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0<AType, BType>(
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
        scale_b_byte_id, scale_b_thread_id);
  }
}

} // namespace tl
