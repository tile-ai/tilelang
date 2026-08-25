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
};

enum class SM120MmaScaleType : int {
  kUE4M3 = 0,
  kUE8M0 = 1,
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
struct SM120MmaBlockScaledConfig<SM120MmaBlockScaledKind::kMxf4nvf4, 4,
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

// SM120a MXFP4 block-scaled warp MMA (PTX ISA operand order):
// mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64...ue8m0
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
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue8m0 "
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
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue8m0 "
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

} // namespace detail

template <SM120MmaBlockScaledKind Kind, int ScaleVecSize,
          SM120MmaScaleType SType>
TL_DEVICE void sm120_mma_sync_blockscaled(float *d, const uint32_t *a,
                                          const uint32_t *b, const float *c,
                                          uint32_t scale_a, uint32_t scale_b,
                                          uint16_t scale_a_byte_id = 0,
                                          uint16_t scale_a_thread_id = 0,
                                          uint16_t scale_b_byte_id = 0,
                                          uint16_t scale_b_thread_id = 0) {
  static_assert(Kind == SM120MmaBlockScaledKind::kMxf4nvf4,
                "Only kind::mxf4nvf4 is supported");
  static_assert(
      SM120MmaBlockScaledConfig<Kind, ScaleVecSize, SType>::kSupported,
      "Unsupported sm120 mma.block_scale configuration");
  if constexpr (ScaleVecSize == 4 && SType == SM120MmaScaleType::kUE4M3) {
    detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3(
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
        scale_b_byte_id, scale_b_thread_id);
  } else if constexpr (ScaleVecSize == 2 &&
                       SType == SM120MmaScaleType::kUE8M0) {
    detail::sm120_mma_m16n8k64_mxf4nvf4_2x_ue8m0(
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
        scale_b_byte_id, scale_b_thread_id);
  } else {
    detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue8m0(
        d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
        scale_b_byte_id, scale_b_thread_id);
  }
}

} // namespace tl
