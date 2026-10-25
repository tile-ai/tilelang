#pragma once

#include "common.h"
#include <cuda_fp8.h>

// ============================================================================
// Inline PTX FP8 Conversions with Stochastic Rounding
// ============================================================================
//
// PTX ISA: cvt.rs.satfinite.f8x4type.f32 d, {a, b, e, f}, rbits
//   Output layout: d[31:24]=a, d[23:16]=b, d[15:8]=e, d[7:0]=f
//   To get little-endian byte order (byte0=elem0), pass elements in reverse.

// --- float4 -> e4m3x4 stochastic rounding ---

// Full 4-element version (float4 input)
template <bool kDependentFalse = false>
TL_DEVICE __nv_fp8x4_storage_t
__tl_cvt_f32x4_to_e4m3x4_rs_sat(float4 src, unsigned int rbits) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __nv_fp8x4_storage_t result;
  asm("cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;"
      : "=r"(result)
      : "f"(src.w), "f"(src.z), "f"(src.y), "f"(src.x), "r"(rbits));
  return result;
#else
  static_assert(kDependentFalse,
                "Stochastic rounding f32-to-FP8 requires sm_100a or sm_103a");
  return {};
#endif
}

// 1-element version: pass src as f (lowest position), returns byte0
template <bool kDependentFalse = false>
TL_DEVICE __nv_fp8_storage_t
__tl_cvt_f32x1_to_e4m3x1_rs_sat(float src, unsigned int rbits) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __nv_fp8x4_storage_t tmp;
  asm("cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;"
      : "=r"(tmp)
      : "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(src), "r"(rbits));
  return static_cast<__nv_fp8_storage_t>(tmp & 0xFF);
#else
  static_assert(kDependentFalse,
                "Stochastic rounding f32-to-FP8 requires sm_100a or sm_103a");
  return {};
#endif
}

// 2-element version: pass src.x as f, src.y as e, returns lower 2 bytes
template <bool kDependentFalse = false>
TL_DEVICE __nv_fp8x2_storage_t
__tl_cvt_f32x2_to_e4m3x2_rs_sat(float2 src, unsigned int rbits) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __nv_fp8x4_storage_t tmp;
  asm("cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;"
      : "=r"(tmp)
      : "f"(0.0f), "f"(0.0f), "f"(src.y), "f"(src.x), "r"(rbits));
  return static_cast<__nv_fp8x2_storage_t>(tmp & 0xFFFF);
#else
  static_assert(kDependentFalse,
                "Stochastic rounding f32-to-FP8 requires sm_100a or sm_103a");
  return {};
#endif
}

// --- float4 -> e5m2x4 stochastic rounding ---

// Full 4-element version (float4 input)
template <bool kDependentFalse = false>
TL_DEVICE __nv_fp8x4_storage_t
__tl_cvt_f32x4_to_e5m2x4_rs_sat(float4 src, unsigned int rbits) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __nv_fp8x4_storage_t result;
  asm("cvt.rs.satfinite.e5m2x4.f32 %0, {%1, %2, %3, %4}, %5;"
      : "=r"(result)
      : "f"(src.w), "f"(src.z), "f"(src.y), "f"(src.x), "r"(rbits));
  return result;
#else
  static_assert(kDependentFalse,
                "Stochastic rounding f32-to-FP8 requires sm_100a or sm_103a");
  return {};
#endif
}

// 1-element version: pass src as f (lowest position), returns byte0
template <bool kDependentFalse = false>
TL_DEVICE __nv_fp8_storage_t
__tl_cvt_f32x1_to_e5m2x1_rs_sat(float src, unsigned int rbits) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __nv_fp8x4_storage_t tmp;
  asm("cvt.rs.satfinite.e5m2x4.f32 %0, {%1, %2, %3, %4}, %5;"
      : "=r"(tmp)
      : "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(src), "r"(rbits));
  return static_cast<__nv_fp8_storage_t>(tmp & 0xFF);
#else
  static_assert(kDependentFalse,
                "Stochastic rounding f32-to-FP8 requires sm_100a or sm_103a");
  return {};
#endif
}

// 2-element version: pass src.x as f, src.y as e, returns lower 2 bytes
template <bool kDependentFalse = false>
TL_DEVICE __nv_fp8x2_storage_t
__tl_cvt_f32x2_to_e5m2x2_rs_sat(float2 src, unsigned int rbits) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __nv_fp8x4_storage_t tmp;
  asm("cvt.rs.satfinite.e5m2x4.f32 %0, {%1, %2, %3, %4}, %5;"
      : "=r"(tmp)
      : "f"(0.0f), "f"(0.0f), "f"(src.y), "f"(src.x), "r"(rbits));
  return static_cast<__nv_fp8x2_storage_t>(tmp & 0xFFFF);
#else
  static_assert(kDependentFalse,
                "Stochastic rounding f32-to-FP8 requires sm_100a or sm_103a");
  return {};
#endif
}
