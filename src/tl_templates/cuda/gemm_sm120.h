#pragma once

#include "gemm_mma.h"

#ifndef __CUDACC_RTC__
#include <cstdint>
#endif

namespace tl {
using tl_mma::gemm_rs;
using tl_mma::gemm_sr;
using tl_mma::gemm_ss;

enum class SM120MmaBlockScaledKind : int {
  kMxf4nvf4 = 0,
};

enum class SM120MmaScaleType : int {
  kUE4M3 = 0,
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

namespace detail {

// SM120a NVF4 block-scaled warp MMA:
// mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3(
    float *d, const uint32_t *a, const uint32_t *b, const float *c,
    uint32_t scale_a, uint32_t scale_b) {
  uint16_t const zero = 0;
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
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(scale_a),
        "h"(zero), "h"(zero), "r"(scale_b), "h"(zero), "h"(zero));
}

} // namespace detail

template <SM120MmaBlockScaledKind Kind, int ScaleVecSize,
          SM120MmaScaleType SType>
TL_DEVICE void sm120_mma_sync_blockscaled(float *d, const uint32_t *a,
                                          const uint32_t *b, const float *c,
                                          uint32_t scale_a,
                                          uint32_t scale_b) {
  static_assert(Kind == SM120MmaBlockScaledKind::kMxf4nvf4,
                "Only kind::mxf4nvf4 is supported");
  static_assert(ScaleVecSize == 4, "Only scale_vec::4X is supported");
  static_assert(SType == SM120MmaScaleType::kUE4M3,
                "kind::mxf4nvf4 only supports ue4m3 scale factors");
  static_assert(SM120MmaBlockScaledConfig<Kind, ScaleVecSize, SType>::kSupported,
                "Unsupported sm120 mma.block_scale configuration");
  detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3(d, a, b, c, scale_a, scale_b);
}
} // namespace tl
