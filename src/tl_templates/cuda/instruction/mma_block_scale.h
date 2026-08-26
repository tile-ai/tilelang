#pragma once

#include "../common.h"
#include <cute/arch/config.hpp>

#ifndef __CUDACC_RTC__
#include <cstdint>
#endif

// The vendored CUTLASS predates the mxf4nvf4 4X+ue8m0 arch gate (added in
// CUTLASS 4.6 for CUDA 13.1); mirror that condition so the instruction is
// usable with either CUTLASS tree once the toolchain allows it.
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE8M0_MMA_ENABLED) ||                        \
    (defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) &&                               \
     (__CUDACC_VER_MAJOR__ > 13 ||                                             \
      (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)))
#define TL_SM120_MXF4NVF4_4X_UE8M0_MMA_ENABLED 1
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

// Operand element types of the block-scaled MMA family. The tail template
// parameters below default to kE2M1 so every existing mxf4nvf4 instantiation
// (and its generated kernel source) stays unchanged.
enum class SM120MmaOperandType : int {
  kE2M1 = 0,
  kE4M3 = 1,
  kE5M2 = 2,
  kE2M3 = 3,
  kE3M2 = 4,
};

template <SM120MmaBlockScaledKind Kind, int ScaleVecSize,
          SM120MmaScaleType SType,
          SM120MmaOperandType AType = SM120MmaOperandType::kE2M1,
          SM120MmaOperandType BType = SM120MmaOperandType::kE2M1>
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
struct SM120MmaBlockScaledConfig<SM120MmaBlockScaledKind::kMxf4nvf4, 4,
                                 SM120MmaScaleType::kUE8M0> {
  static constexpr bool kSupported = true;
};

// kind::mxf8f6f4 (m16n8k32): every {e2m1, e2m3, e3m2, e4m3, e5m2} A/B
// pairing (the full f8f6f4 family, 5x5), ue8m0 scales at scale_vec::1X only
// (PTX allows no other scale_vec for this kind). fp4/fp6 operands live in
// 8-bit register containers: e2m1 at bits[5:2] (loader shifts <<2), fp6 at
// bits[5:0].
template <SM120MmaOperandType AType, SM120MmaOperandType BType>
struct SM120MmaBlockScaledConfig<SM120MmaBlockScaledKind::kMxf8f6f4, 1,
                                 SM120MmaScaleType::kUE8M0, AType, BType> {
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
    defined(CUTLASS_ARCH_MMA_SM120A_ENABLED)
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
      "tl::sm120_mma_sync_blockscaled requires sm_120a and CUDA 12.8 or "
      "later");
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

// SM120a MXFP4 block-scaled warp MMA:
// mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X
// ...ue8m0
//
// Spelled shape-first to match the documented PTX grammar and the 4X ue4m3
// wrapper above. (CUTLASS cute/arch/mma_sm120.hpp spells the same
// instruction kind-first; ptxas accepts both spellings.)
//
// The PTX scale operand register stays b32 but the instruction consumes only
// 16 bits of it. CUTLASS drives this form exclusively with byte_id = 0 and a
// uint16 SF fragment (mma_sm120.hpp SM120_16x8x64_TN_VS, VS=32), so byte
// selection is done in software here: shift the requested byte pair into the
// low half and issue with a hardware byte id of 0. thread_id keeps its
// hardware lane-select semantics, which the 4X path already exercises.
TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_2x_ue8m0_regs(
    float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, const float *c, uint32_t scale_a, uint32_t scale_b,
    uint16_t scale_a_byte_id = 0, uint16_t scale_a_thread_id = 0,
    uint16_t scale_b_byte_id = 0, uint16_t scale_b_thread_id = 0) {
#if defined(CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED) &&                        \
    defined(CUTLASS_ARCH_MMA_SM120A_ENABLED)
  uint32_t sa = scale_a >> (scale_a_byte_id * 8);
  uint32_t sb = scale_b >> (scale_b_byte_id * 8);
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
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]),
        "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sa), "h"(uint16_t(0)),
        "h"(scale_a_thread_id), "r"(sb), "h"(uint16_t(0)),
        "h"(scale_b_thread_id));
#else
  CUTE_INVALID_CONTROL_PATH(
      "tl::sm120_mma_sync_blockscaled scale_vec::2X ue8m0 requires sm_120a "
      "and CUDA 12.8 or later");
#endif
}

TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_2x_ue8m0(
    float *d, const uint32_t *a, const uint32_t *b, const float *c,
    uint32_t scale_a, uint32_t scale_b, uint16_t scale_a_byte_id = 0,
    uint16_t scale_a_thread_id = 0, uint16_t scale_b_byte_id = 0,
    uint16_t scale_b_thread_id = 0) {
  sm120_mma_m16n8k64_mxf4nvf4_2x_ue8m0_regs(
      d, a[0], a[1], a[2], a[3], b[0], b[1], c, scale_a, scale_b,
      scale_a_byte_id, scale_a_thread_id, scale_b_byte_id, scale_b_thread_id);
}

// SM120a NVF4-granularity ue8m0 variant (scale_vec::4X, CUDA 13.1+). Full
// 4-byte scale-word consumption, same operand semantics as the ue4m3 form.
TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_4x_ue8m0_regs(
    float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, const float *c, uint32_t scale_a, uint32_t scale_b,
    uint16_t scale_a_byte_id = 0, uint16_t scale_a_thread_id = 0,
    uint16_t scale_b_byte_id = 0, uint16_t scale_b_thread_id = 0) {
#if defined(TL_SM120_MXF4NVF4_4X_UE8M0_MMA_ENABLED)
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::"
      "4X.f32.e2m1.e2m1.f32.ue8m0 "
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
      "tl::sm120_mma_sync_blockscaled scale_vec::4X ue8m0 requires sm_120a "
      "and CUDA 13.1 or later");
#endif
}

TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_4x_ue8m0(
    float *d, const uint32_t *a, const uint32_t *b, const float *c,
    uint32_t scale_a, uint32_t scale_b, uint16_t scale_a_byte_id = 0,
    uint16_t scale_a_thread_id = 0, uint16_t scale_b_byte_id = 0,
    uint16_t scale_b_thread_id = 0) {
  sm120_mma_m16n8k64_mxf4nvf4_4x_ue8m0_regs(
      d, a[0], a[1], a[2], a[3], b[0], b[1], c, scale_a, scale_b,
      scale_a_byte_id, scale_a_thread_id, scale_b_byte_id, scale_b_thread_id);
}

// SM120a MXFP8 block-scaled warp MMA family:
// mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X
//   .f32.<atype>.<btype>.f32.ue8m0
//
// Register skeleton is identical to the k64 mxf4nvf4 wrappers above (4x b32
// A, 2x b32 B, 4x f32 C/D, per-matrix {scale, byte_id, thread_id} operands).
// scale_vec::1X consumes one ue8m0 byte per matrix; as with the 2X wrapper,
// byte selection is done in software - shift the requested byte into the low
// half and issue with a hardware byte id of 0, the only form CUTLASS
// exercises (mma_sm120.hpp SM120_16x8x32_TN_VS, VS=32).
// Dispatch table: one specialization per (AType, BType) pairing, stamped by
// the macro below together with its asm wrapper.
template <SM120MmaOperandType AType, SM120MmaOperandType BType>
struct SM120Mxf8f6f4MmaImpl;

#define TL_SM120_DEFINE_MXF8F6F4_1X_UE8M0_MMA(suffix, a_enum, b_enum,          \
                                              atype_str, btype_str)            \
  TL_DEVICE void sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##suffix##_regs(         \
      float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,            \
      uint32_t b0, uint32_t b1, const float *c, uint32_t scale_a,              \
      uint32_t scale_b, uint16_t scale_a_byte_id = 0,                          \
      uint16_t scale_a_thread_id = 0, uint16_t scale_b_byte_id = 0,            \
      uint16_t scale_b_thread_id =                                             \
          0){TL_SM120_MXF8F6F4_MMA_BODY(atype_str, btype_str)} TL_DEVICE void  \
      sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##suffix(                           \
          float *d, const uint32_t *a, const uint32_t *b, const float *c,      \
          uint32_t scale_a, uint32_t scale_b, uint16_t scale_a_byte_id = 0,    \
          uint16_t scale_a_thread_id = 0, uint16_t scale_b_byte_id = 0,        \
          uint16_t scale_b_thread_id = 0) {                                    \
    sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##suffix##_regs(                      \
        d, a[0], a[1], a[2], a[3], b[0], b[1], c, scale_a, scale_b,            \
        scale_a_byte_id, scale_a_thread_id, scale_b_byte_id,                   \
        scale_b_thread_id);                                                    \
  }                                                                            \
  template <>                                                                  \
  struct SM120Mxf8f6f4MmaImpl<SM120MmaOperandType::a_enum,                     \
                              SM120MmaOperandType::b_enum> {                   \
    static TL_DEVICE void run(float *d, const uint32_t *a, const uint32_t *b,  \
                              const float *c, uint32_t scale_a,                \
                              uint32_t scale_b, uint16_t scale_a_byte_id,      \
                              uint16_t scale_a_thread_id,                      \
                              uint16_t scale_b_byte_id,                        \
                              uint16_t scale_b_thread_id) {                    \
      sm120_mma_m16n8k32_mxf8f6f4_1x_ue8m0_##suffix(                           \
          d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,    \
          scale_b_byte_id, scale_b_thread_id);                                 \
    }                                                                          \
  };

#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED) &&                                 \
    defined(CUTLASS_ARCH_MMA_SM120A_ENABLED)
#define TL_SM120_MXF8F6F4_MMA_BODY(atype_str, btype_str)                       \
  uint32_t sa = scale_a >> (scale_a_byte_id * 8);                              \
  uint32_t sb = scale_b >> (scale_b_byte_id * 8);                              \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale." \
               "scale_vec::1X.f32." atype_str "." btype_str ".f32.ue8m0 "      \
               "{%0, %1, %2, %3}, "                                            \
               "{%4, %5, %6, %7}, "                                            \
               "{%8, %9}, "                                                    \
               "{%10, %11, %12, %13}, "                                        \
               "{%14}, {%15, %16}, "                                           \
               "{%17}, {%18, %19};\n"                                          \
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),         \
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sa),          \
                 "h"(uint16_t(0)), "h"(scale_a_thread_id), "r"(sb),            \
                 "h"(uint16_t(0)), "h"(scale_b_thread_id));
#else
#define TL_SM120_MXF8F6F4_MMA_BODY(atype_str, btype_str)                       \
  CUTE_INVALID_CONTROL_PATH(                                                   \
      "tl::sm120_mma_sync_blockscaled kind::mxf8f6f4 requires sm_120a and "    \
      "CUDA 12.8 or later");
#endif

// The full f8f6f4-family 5x5 operand matrix (matches CUTLASS 4.7
// mma_sm120.hpp SM120_16x8x32_TN_VS specializations).
#define TL_SM120_FOREACH_MXF8F6F4_B(a_suffix, a_enum, a_str)                   \
  TL_SM120_DEFINE_MXF8F6F4_1X_UE8M0_MMA(a_suffix##_e2m1, a_enum, kE2M1, a_str, \
                                        "e2m1")                                \
  TL_SM120_DEFINE_MXF8F6F4_1X_UE8M0_MMA(a_suffix##_e2m3, a_enum, kE2M3, a_str, \
                                        "e2m3")                                \
  TL_SM120_DEFINE_MXF8F6F4_1X_UE8M0_MMA(a_suffix##_e3m2, a_enum, kE3M2, a_str, \
                                        "e3m2")                                \
  TL_SM120_DEFINE_MXF8F6F4_1X_UE8M0_MMA(a_suffix##_e4m3, a_enum, kE4M3, a_str, \
                                        "e4m3")                                \
  TL_SM120_DEFINE_MXF8F6F4_1X_UE8M0_MMA(a_suffix##_e5m2, a_enum, kE5M2, a_str, \
                                        "e5m2")

TL_SM120_FOREACH_MXF8F6F4_B(e2m1, kE2M1, "e2m1")
TL_SM120_FOREACH_MXF8F6F4_B(e2m3, kE2M3, "e2m3")
TL_SM120_FOREACH_MXF8F6F4_B(e3m2, kE3M2, "e3m2")
TL_SM120_FOREACH_MXF8F6F4_B(e4m3, kE4M3, "e4m3")
TL_SM120_FOREACH_MXF8F6F4_B(e5m2, kE5M2, "e5m2")

#undef TL_SM120_FOREACH_MXF8F6F4_B
#undef TL_SM120_DEFINE_MXF8F6F4_1X_UE8M0_MMA
#undef TL_SM120_MXF8F6F4_MMA_BODY

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
  static_assert(Kind == SM120MmaBlockScaledKind::kMxf4nvf4 ||
                    Kind == SM120MmaBlockScaledKind::kMxf8f6f4,
                "Only kind::mxf4nvf4 and kind::mxf8f6f4 are supported");
  static_assert(SM120MmaBlockScaledConfig<Kind, ScaleVecSize, SType, AType,
                                          BType>::kSupported,
                "Unsupported sm120 mma.block_scale configuration");
  if constexpr (Kind == SM120MmaBlockScaledKind::kMxf8f6f4) {
    detail::SM120Mxf8f6f4MmaImpl<AType, BType>::run(
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
        scale_b_byte_id, scale_b_thread_id);
  } else if constexpr (ScaleVecSize == 4 &&
                       SType == SM120MmaScaleType::kUE4M3) {
    detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3(
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
        scale_b_byte_id, scale_b_thread_id);
  } else if constexpr (ScaleVecSize == 2 &&
                       SType == SM120MmaScaleType::kUE8M0) {
    detail::sm120_mma_m16n8k64_mxf4nvf4_2x_ue8m0(
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
        scale_b_byte_id, scale_b_thread_id);
  } else {
#if defined(TL_SM120_MXF4NVF4_4X_UE8M0_MMA_ENABLED)
    detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue8m0(
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
        scale_b_byte_id, scale_b_thread_id);
#else
    // Instantiated only when a kernel actually selects 4X+ue8m0: fail the
    // JIT compile with a clear message instead of trapping at runtime.
    static_assert(ScaleVecSize != 4,
                  "mxf4nvf4 scale_vec::4X with ue8m0 scale factors requires "
                  "CUDA 13.1 or later");
#endif
  }
}

} // namespace tl
