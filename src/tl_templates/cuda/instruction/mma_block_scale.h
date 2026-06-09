#pragma once

#include "../common.h"

#ifndef __CUDACC_RTC__
#include <cstdint>
#endif

namespace tl {

enum class MmaBlockScaleKind : int {
  kMxf4nvf4 = 0,
};

enum class ScaleType : int {
  kUE4M3 = 0,
  kUE8M0 = 1,
};

template <MmaBlockScaleKind Kind, int ScaleVecSize, ScaleType SType>
struct MmaBlockScaleConfig {
  static constexpr bool kSupported = false;
};

template <>
struct MmaBlockScaleConfig<MmaBlockScaleKind::kMxf4nvf4, 4,
                           ScaleType::kUE4M3> {
  static constexpr bool kSupported = true;
};

namespace detail {

// SM120a NVF4 block-scaled warp MMA:
// mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
TL_DEVICE void mma_block_scale_m16n8k64_mxf4nvf4_4x_ue4m3(
    float *d, const uint32_t *a, const uint32_t *b, const float *c,
    uint32_t scale_a, uint16_t byte_id_a, uint16_t thread_id_a,
    uint32_t scale_b, uint16_t byte_id_b, uint16_t thread_id_b) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::"
      "4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13}, "
      "%14, {%15, %16}, "
      "%17, {%18, %19};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
        "f"(c[3]), "r"(scale_a), "h"(byte_id_a), "h"(thread_id_a),
        "r"(scale_b), "h"(byte_id_b), "h"(thread_id_b));
}

} // namespace detail

template <MmaBlockScaleKind Kind, int ScaleVecSize, ScaleType SType>
TL_DEVICE void mma_block_scale_sync(
    float *d, const uint32_t *a, const uint32_t *b, const float *c,
    uint32_t scale_a, uint16_t byte_id_a, uint16_t thread_id_a,
    uint32_t scale_b, uint16_t byte_id_b, uint16_t thread_id_b) {
  static_assert(Kind == MmaBlockScaleKind::kMxf4nvf4,
                "Only kind::mxf4nvf4 is supported");
  static_assert(ScaleVecSize == 4, "Only scale_vec::4X is supported");
  static_assert(SType == ScaleType::kUE4M3,
                "kind::mxf4nvf4 only supports ue4m3 scale factors");
  static_assert(MmaBlockScaleConfig<Kind, ScaleVecSize, SType>::kSupported,
                "Unsupported mma.block_scale configuration");
  detail::mma_block_scale_m16n8k64_mxf4nvf4_4x_ue4m3(
      d, a, b, c, scale_a, byte_id_a, thread_id_a, scale_b, byte_id_b,
      thread_id_b);
}

} // namespace tl
