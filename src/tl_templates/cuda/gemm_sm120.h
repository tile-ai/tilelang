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

TL_DEVICE uint32_t sm120_smem_addr(void const *const smem_ptr) {
  return smem_ptr_to_uint(smem_ptr);
}

TL_DEVICE uint32_t sm120_cutlass_128x4_sf_word(uint32_t idx, uint32_t ki) {
  return ki * 128u + (idx & 31u) * 4u + (idx >> 5);
}

TL_DEVICE uint32_t sm120_cute_sfa_slot_word(uint32_t lane, uint32_t warp_m,
                                            uint32_t ki, uint32_t m_atom) {
  return ki * 128u + warp_m * 64u + (lane >> 2) * 4u + ((lane & 1u) << 5) +
         m_atom;
}

TL_DEVICE uint32_t sm120_cute_sfb_slot_word(uint32_t lane, uint32_t warp_n,
                                            uint32_t ki, uint32_t n16_col,
                                            uint32_t n8_half) {
  return ki * 128u + warp_n * 64u + (lane >> 2) * 4u + n16_col + n8_half * 32u;
}

TL_DEVICE void sm120_ldscale_v4_u32(void const *const smem_ptr, uint32_t &d0,
                                    uint32_t &d1, uint32_t &d2, uint32_t &d3) {
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
               : "r"(sm120_smem_addr(smem_ptr)));
}

TL_DEVICE uint32_t sm120_select4_u32(uint32_t v0, uint32_t v1, uint32_t v2,
                                     uint32_t v3, uint32_t idx) {
  return idx == 0u ? v0 : (idx == 1u ? v1 : (idx == 2u ? v2 : v3));
}

TL_DEVICE uint32_t sm120_broadcast_u8_from_u32(uint32_t word, uint32_t idx) {
  uint32_t const byte = (word >> ((idx & 3u) * 8u)) & 0xffu;
  return byte * 0x01010101u;
}

TL_DEVICE void sm120_ldmatrix_x4_u32_addr(uint32_t smem_int_ptr, uint32_t &d0,
                                          uint32_t &d1, uint32_t &d2,
                                          uint32_t &d3) {
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(smem_int_ptr));
}

TL_DEVICE void sm120_ldmatrix_x4_u32(void const *const smem_ptr, uint32_t &d0,
                                     uint32_t &d1, uint32_t &d2, uint32_t &d3) {
  sm120_ldmatrix_x4_u32_addr(smem_ptr_to_uint(smem_ptr), d0, d1, d2, d3);
}

// This copy atom consumes a padded/unpacked 4-bit shared layout. It matches the
// F8F6F4/MXF-style path where FP4 nibbles are placed in 8-bit slots with p64
// padding. Dense SM120 NVFP4/mxf4nvf4 uses packed U4 shared storage and the
// ordinary x4.m8n8.shared.b16 ldmatrix path above.
TL_DEVICE void sm120_ldmatrix_x4_fp4_u32_addr(uint32_t smem_int_ptr,
                                              uint32_t &d0, uint32_t &d1,
                                              uint32_t &d2, uint32_t &d3) {
  asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 "
               "{%0, %1, %2, %3}, [%4];\n"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
               : "r"(smem_int_ptr));
#if !defined(TL_SM120_FULLTILE_FP4_LDSM_NO_SHIFT)
  d0 <<= 2;
  d1 <<= 2;
  d2 <<= 2;
  d3 <<= 2;
#endif
}

TL_DEVICE void sm120_ldmatrix_x4_fp4_u32(void const *const smem_ptr,
                                         uint32_t &d0, uint32_t &d1,
                                         uint32_t &d2, uint32_t &d3) {
  sm120_ldmatrix_x4_fp4_u32_addr(smem_ptr_to_uint(smem_ptr), d0, d1, d2, d3);
}

TL_DEVICE void sm120_ldmatrix_x4_blockscaled_operand_addr(uint32_t smem_int_ptr,
                                                          uint32_t &d0,
                                                          uint32_t &d1,
                                                          uint32_t &d2,
                                                          uint32_t &d3) {
#if defined(TL_SM120_FULLTILE_FP4_LDSM)
  sm120_ldmatrix_x4_fp4_u32_addr(smem_int_ptr, d0, d1, d2, d3);
#else
  sm120_ldmatrix_x4_u32_addr(smem_int_ptr, d0, d1, d2, d3);
#if defined(TL_SM120_FULLTILE_FP4_REG_SHIFT)
  d0 <<= 2;
  d1 <<= 2;
  d2 <<= 2;
  d3 <<= 2;
#endif
#endif
}

TL_DEVICE void sm120_ldmatrix_x4_blockscaled_operand(void const *const smem_ptr,
                                                     uint32_t &d0, uint32_t &d1,
                                                     uint32_t &d2,
                                                     uint32_t &d3) {
#if defined(TL_SM120_FULLTILE_FP4_LDSM)
  sm120_ldmatrix_x4_fp4_u32(smem_ptr, d0, d1, d2, d3);
#else
  sm120_ldmatrix_x4_u32(smem_ptr, d0, d1, d2, d3);
#if defined(TL_SM120_FULLTILE_FP4_REG_SHIFT)
  d0 <<= 2;
  d1 <<= 2;
  d2 <<= 2;
  d3 <<= 2;
#endif
#endif
}

TL_DEVICE uint32_t sm120_cute_accum_pair_offset_from_tl_pair(uint32_t tl_pair) {
  uint32_t const i = tl_pair >> 5;
  uint32_t const rem_i = tl_pair & 31u;
  uint32_t const j = rem_i >> 3;
  uint32_t const rem_j = rem_i & 7u;
  uint32_t const half = rem_j >> 2;
  uint32_t const reg_pair = rem_j & 3u;
  return i * 4u + j * 32u + half * 16u + reg_pair;
}

TL_DEVICE void
sm120_store_full_c_fragment_panel64_bf16(float const *const c_fragment,
                                         bfloat16_t *const c_shared,
                                         int const panel) {
  uint32_t const tx = uint32_t(threadIdx.x) & 127u;
  if (int(tx >> 6) != panel) {
    return;
  }

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    uint32_t const tl_pair = uint32_t(i) * 2u;
#if defined(TL_SM120_FULLTILE_CUTE_ACCUM_DIRECT)
    uint32_t const src_pair =
        sm120_cute_accum_pair_offset_from_tl_pair(tl_pair);
#else
    uint32_t const src_pair = tl_pair;
#endif
    float2 const value =
        *reinterpret_cast<float2 const *>(c_fragment + src_pair);
    uint1 packed;
    reinterpret_cast<__nv_bfloat162 *>(&packed)[0] =
        __float22bfloat162_rn(value);
    uint32_t const row = ((tx & 63u) >> 5) * 64u + uint32_t(i >> 4) * 16u +
                         uint32_t(i & 1) * 8u + ((tx & 31u) >> 2);
    uint32_t const col = uint32_t((i & 15) >> 1) * 8u + (tx & 3u) * 2u;
    *reinterpret_cast<uint1 *>(c_shared + row * 64u + col) = packed;
  }
}

TL_DEVICE uint32_t sm120_tma_store_128x32_bf16_swizzle_offset(uint32_t row,
                                                              uint32_t col8) {
  uint32_t const tx = ((row & 31u) << 2) | (col8 >> 3);
  return ((row >> 5) * 1024u) + ((tx >> 2) * 32u) +
         (((((tx & 31u) >> 4) + ((tx & 3u) >> 1)) & 1u) * 16u) +
         (((((tx & 15u) >> 3) + (tx & 1u)) & 1u) * 8u);
}

TL_DEVICE void
sm120_store_full_c_fragment_panel32_tma_bf16(float const *const c_fragment,
                                             bfloat16_t *const c_shared,
                                             int const panel32) {
  uint32_t const tx = uint32_t(threadIdx.x) & 127u;
  uint32_t const panel64 = uint32_t(panel32) >> 1;
  uint32_t const sub32 = uint32_t(panel32) & 1u;
  if ((tx >> 6) != panel64) {
    return;
  }

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    uint32_t const col64 = uint32_t((i & 15) >> 1) * 8u + (tx & 3u) * 2u;
    if ((col64 >> 5) != sub32) {
      continue;
    }

    uint32_t const tl_pair = uint32_t(i) * 2u;
#if defined(TL_SM120_FULLTILE_CUTE_ACCUM_DIRECT)
    uint32_t const src_pair =
        sm120_cute_accum_pair_offset_from_tl_pair(tl_pair);
#else
    uint32_t const src_pair = tl_pair;
#endif
    float2 const value =
        *reinterpret_cast<float2 const *>(c_fragment + src_pair);
    uint1 packed;
    reinterpret_cast<__nv_bfloat162 *>(&packed)[0] =
        __float22bfloat162_rn(value);

    uint32_t const row = ((tx & 63u) >> 5) * 64u + uint32_t(i >> 4) * 16u +
                         uint32_t(i & 1) * 8u + ((tx & 31u) >> 2);
    uint32_t const col32 = col64 & 31u;
    uint32_t const swizzled =
        sm120_tma_store_128x32_bf16_swizzle_offset(row, col32 & ~7u) +
        (col32 & 7u);
    *reinterpret_cast<uint1 *>(c_shared + swizzled) = packed;
  }
}

TL_DEVICE void sm120_store_full_c_fragment_epi64x32_tma_bf16(
    float const *const c_fragment, bfloat16_t *const c_shared, int const row64,
    int const col32_panel) {
  uint32_t const tx = uint32_t(threadIdx.x) & 127u;
  uint32_t const col64_panel = uint32_t(col32_panel) >> 1;
  uint32_t const sub32 = uint32_t(col32_panel) & 1u;
  if ((tx >> 6) != col64_panel) {
    return;
  }

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    uint32_t const col64 = uint32_t((i & 15) >> 1) * 8u + (tx & 3u) * 2u;
    if ((col64 >> 5) != sub32) {
      continue;
    }

    uint32_t const row = ((tx & 63u) >> 5) * 64u + uint32_t(i >> 4) * 16u +
                         uint32_t(i & 1) * 8u + ((tx & 31u) >> 2);
    if ((row >> 6) != uint32_t(row64)) {
      continue;
    }

    uint32_t const tl_pair = uint32_t(i) * 2u;
#if defined(TL_SM120_FULLTILE_CUTE_ACCUM_DIRECT)
    uint32_t const src_pair =
        sm120_cute_accum_pair_offset_from_tl_pair(tl_pair);
#else
    uint32_t const src_pair = tl_pair;
#endif
    float2 const value =
        *reinterpret_cast<float2 const *>(c_fragment + src_pair);
    uint1 packed;
    reinterpret_cast<__nv_bfloat162 *>(&packed)[0] =
        __float22bfloat162_rn(value);

    uint32_t const local_row = row & 63u;
    uint32_t const col32 = col64 & 31u;
    uint32_t const swizzled =
        sm120_tma_store_128x32_bf16_swizzle_offset(local_row, col32 & ~7u) +
        (col32 & 7u);
    *reinterpret_cast<uint1 *>(c_shared + swizzled) = packed;
  }
}

// SM120a NVF4 block-scaled warp MMA:
// mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs(
    float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, const float *c, uint32_t scale_a, uint32_t scale_b,
    uint16_t scale_a_byte_id = 0, uint16_t scale_a_thread_id = 0,
    uint16_t scale_b_byte_id = 0, uint16_t scale_b_thread_id = 0) {
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
}

TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_accum_regs(
    float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, uint32_t scale_a, uint32_t scale_b,
    uint16_t scale_a_byte_id = 0, uint16_t scale_a_thread_id = 0,
    uint16_t scale_b_byte_id = 0, uint16_t scale_b_thread_id = 0) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::"
      "4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3}, "
      "{%10}, {%11, %12}, "
      "{%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a),
        "h"(scale_a_byte_id), "h"(scale_a_thread_id), "r"(scale_b),
        "h"(scale_b_byte_id), "h"(scale_b_thread_id));
}

TL_DEVICE void sm120_mma2_m16n8k64_mxf4nvf4_4x_ue4m3_same_b_regs(
    float *d0, uint32_t a00, uint32_t a01, uint32_t a02, uint32_t a03,
    const float *c0, uint32_t scale_a0, float *d1, uint32_t a10, uint32_t a11,
    uint32_t a12, uint32_t a13, const float *c1, uint32_t scale_a1, uint32_t b0,
    uint32_t b1, uint32_t scale_b) {
  uint16_t const zero = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::"
      "4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, "
      "{%8, %9, %10, %11}, "
      "{%16, %17}, "
      "{%18, %19, %20, %21}, "
      "{%26}, {%29, %29}, "
      "{%28}, {%29, %29};\n"
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::"
      "4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%4, %5, %6, %7}, "
      "{%12, %13, %14, %15}, "
      "{%16, %17}, "
      "{%22, %23, %24, %25}, "
      "{%27}, {%29, %29}, "
      "{%28}, {%29, %29};\n"
      : "=&f"(d0[0]), "=&f"(d0[1]), "=&f"(d0[2]), "=&f"(d0[3]), "=&f"(d1[0]),
        "=&f"(d1[1]), "=&f"(d1[2]), "=&f"(d1[3])
      : "r"(a00), "r"(a01), "r"(a02), "r"(a03), "r"(a10), "r"(a11), "r"(a12),
        "r"(a13), "r"(b0), "r"(b1), "f"(c0[0]), "f"(c0[1]), "f"(c0[2]),
        "f"(c0[3]), "f"(c1[0]), "f"(c1[1]), "f"(c1[2]), "f"(c1[3]),
        "r"(scale_a0), "r"(scale_a1), "r"(scale_b), "h"(zero));
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

TL_DEVICE uint32_t sm120_fulltile_compact_k_swizzle_offset(uint32_t tx,
                                                           int k_block_idx) {
  switch (k_block_idx & 3) {
  case 0:
    return ((tx & 7u) >> 1) * 32u;
  case 1:
    return ((tx & 7u) >> 2) * 64u + (((((tx & 3u) >> 1) + 1u) & 1u) * 32u);
  case 2:
    return (((((tx & 7u) >> 2) + 1u) & 1u) * 64u) + (((tx & 3u) >> 1) * 32u);
  default:
    return (((((tx & 7u) >> 2) + 1u) & 1u) * 64u) +
           (((((tx & 3u) >> 1) + 1u) & 1u) * 32u);
  }
}

TL_DEVICE uint32_t sm120_fulltile_compact_a_offset(uint32_t tx, int k_block_idx,
                                                   int row_idx) {
  return (((tx & 63u) >> 5) * 2048u) + ((tx & 15u) * 128u) +
         sm120_fulltile_compact_k_swizzle_offset(tx, k_block_idx) +
         (((((tx & 31u) >> 4) + (tx & 1u)) & 1u) * 16u) +
         uint32_t(row_idx) * 4096u;
}

TL_DEVICE uint32_t sm120_fulltile_compact_b_offset(uint32_t tx, int k_block_idx,
                                                   int panel_idx) {
  return ((tx >> 6) * 2048u) + (((tx & 31u) >> 4) * 1024u) +
         ((tx & 7u) * 128u) +
         sm120_fulltile_compact_k_swizzle_offset(tx, k_block_idx) +
         (((((tx & 15u) >> 3) + (tx & 1u)) & 1u) * 16u) +
         uint32_t(panel_idx) * 4096u;
}

TL_DEVICE uint32_t sm120_fulltile_rowmajor_a_offset(uint32_t tx,
                                                    int k_block_idx,
                                                    int row_idx) {
  return (((tx & 63u) >> 5) * 8192u) + ((tx & 15u) * 128u) +
         sm120_fulltile_compact_k_swizzle_offset(tx, k_block_idx) +
         (((((tx & 31u) >> 4) + (tx & 1u)) & 1u) * 16u) +
         uint32_t(row_idx) * 2048u;
}

TL_DEVICE uint32_t sm120_fulltile_rowmajor_b_offset(uint32_t tx,
                                                    int k_block_idx,
                                                    int panel_idx) {
  return ((tx >> 6) * 8192u) + (((tx & 31u) >> 4) * 1024u) +
         ((tx & 7u) * 128u) +
         sm120_fulltile_compact_k_swizzle_offset(tx, k_block_idx) +
         (((((tx & 15u) >> 3) + (tx & 1u)) & 1u) * 16u) +
         uint32_t(panel_idx) * 2048u;
}

TL_DEVICE uint32_t sm120_fulltile_package_a_offset(uint32_t tx, int k_block_idx,
                                                   int row_idx) {
#if defined(TL_SM120_FULLTILE_PACKAGE_ROWMAJOR_VIEW)
  return sm120_fulltile_rowmajor_a_offset(tx, k_block_idx, row_idx);
#else
  return sm120_fulltile_compact_a_offset(tx, k_block_idx, row_idx);
#endif
}

TL_DEVICE uint32_t sm120_fulltile_package_b_offset(uint32_t tx, int k_block_idx,
                                                   int panel_idx) {
#if defined(TL_SM120_FULLTILE_PACKAGE_ROWMAJOR_VIEW)
  return sm120_fulltile_rowmajor_b_offset(tx, k_block_idx, panel_idx);
#else
  return sm120_fulltile_compact_b_offset(tx, k_block_idx, panel_idx);
#endif
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
  static_assert(ScaleVecSize == 4, "Only scale_vec::4X is supported");
  static_assert(SType == SM120MmaScaleType::kUE4M3,
                "kind::mxf4nvf4 only supports ue4m3 scale factors");
  static_assert(
      SM120MmaBlockScaledConfig<Kind, ScaleVecSize, SType>::kSupported,
      "Unsupported sm120 mma.block_scale configuration");
  detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3(
      d, a, b, c, scale_a, scale_b, scale_a_byte_id, scale_a_thread_id,
      scale_b_byte_id, scale_b_thread_id);
}

TL_DEVICE void sm120_mma_blockscaled_kblock_fulltile(
    float *c, const void *a_smem_0, const void *a_smem_1, const void *a_smem_2,
    const void *a_smem_3, const void *b_smem_0, const void *b_smem_1,
    const void *b_smem_2, const void *b_smem_3, const uint32_t *sfa_0,
    const uint32_t *sfa_1, const uint32_t *sfa_2, const uint32_t *sfa_3,
    const uint32_t *sfb_0, const uint32_t *sfb_1, const uint32_t *sfb_2,
    const uint32_t *sfb_3, const uint32_t *sfb_rep_0, const uint32_t *sfb_rep_1,
    const uint32_t *sfb_rep_2, const uint32_t *sfb_rep_3, int k_block_idx);

TL_DEVICE void sm120_mma_blockscaled_cute_consumer_bridge(
    float *c, const void *a_smem_base, const void *b_smem_base,
    const uint32_t *sfa_smem_base, const uint32_t *sfb_smem_base,
    int k_block_idx) {
#if defined(TL_SM120_FULLTILE_CUTLASS_SF_BASEPTR)
  uint32_t const tx = uint32_t(int(threadIdx.x) & 127);
  char const *const a_base = static_cast<char const *>(a_smem_base);
  char const *const b_base = static_cast<char const *>(b_smem_base);
  sm120_mma_blockscaled_kblock_fulltile(
      c, a_base + detail::sm120_fulltile_compact_a_offset(tx, k_block_idx, 0),
      a_base + detail::sm120_fulltile_compact_a_offset(tx, k_block_idx, 1),
      a_base + detail::sm120_fulltile_compact_a_offset(tx, k_block_idx, 2),
      a_base + detail::sm120_fulltile_compact_a_offset(tx, k_block_idx, 3),
      b_base + detail::sm120_fulltile_compact_b_offset(tx, k_block_idx, 0),
      b_base + detail::sm120_fulltile_compact_b_offset(tx, k_block_idx, 1),
      b_base + detail::sm120_fulltile_compact_b_offset(tx, k_block_idx, 2),
      b_base + detail::sm120_fulltile_compact_b_offset(tx, k_block_idx, 3),
      sfa_smem_base, sfa_smem_base, sfa_smem_base, sfa_smem_base, sfb_smem_base,
      sfb_smem_base, sfb_smem_base, sfb_smem_base, sfb_smem_base, sfb_smem_base,
      sfb_smem_base, sfb_smem_base, k_block_idx);
#else
  asm volatile("" ::"l"(c), "l"(a_smem_base), "l"(b_smem_base),
               "l"(sfa_smem_base), "l"(sfb_smem_base), "r"(k_block_idx)
               : "memory");
#endif
}

TL_DEVICE void sm120_mma_blockscaled_kblock_fulltile(
    float *c, const void *a_smem_0, const void *a_smem_1, const void *a_smem_2,
    const void *a_smem_3, const void *b_smem_0, const void *b_smem_1,
    const void *b_smem_2, const void *b_smem_3, const uint32_t *sfa_0,
    const uint32_t *sfa_1, const uint32_t *sfa_2, const uint32_t *sfa_3,
    const uint32_t *sfb_0, const uint32_t *sfb_1, const uint32_t *sfb_2,
    const uint32_t *sfb_3, const uint32_t *sfb_rep_0, const uint32_t *sfb_rep_1,
    const uint32_t *sfb_rep_2, const uint32_t *sfb_rep_3, int k_block_idx) {
  uint32_t b00, b01, b02, b03;
  uint32_t b10, b11, b12, b13;
  uint32_t b20, b21, b22, b23;
  uint32_t b30, b31, b32, b33;

#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_A) ||                              \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_B)
  int const warp128 = (int(threadIdx.x) & 127) >> 5;
  int const m_group = warp128 & 1;
  int const n_group = warp128 >> 1;
  constexpr int kCuteRowstartBackstep =
#if defined(TL_SM120_FULLTILE_RAW_UNPACKED_FP4_ACCESS_PTR)
      6144;
#elif defined(TL_SM120_FULLTILE_COMPACT_UNPACKED_FP4_SHARED)
      3072;
#else
      6144;
#endif
  constexpr int kCuteKAtomStride =
#if defined(TL_SM120_FULLTILE_RAW_UNPACKED_FP4_ACCESS_PTR)
      4096;
#elif defined(TL_SM120_FULLTILE_COMPACT_UNPACKED_FP4_SHARED)
      2048;
#else
      4096;
#endif
  char const *const a_cute_m0 =
      static_cast<char const *>(a_smem_0) - kCuteRowstartBackstep * m_group;
  char const *const b_cute_n0 =
      static_cast<char const *>(b_smem_0) - kCuteRowstartBackstep * n_group;
#endif

#if defined(TL_SM120_FULLTILE_SINGLE_OMMA)
#if !defined(TL_SM120_FULLTILE_SINGLE_KI)
#define TL_SM120_FULLTILE_SINGLE_KI 0
#endif
#if !defined(TL_SM120_FULLTILE_SINGLE_I)
#define TL_SM120_FULLTILE_SINGLE_I 0
#endif
#if !defined(TL_SM120_FULLTILE_SINGLE_J)
#define TL_SM120_FULLTILE_SINGLE_J 0
#endif
#if !defined(TL_SM120_FULLTILE_SINGLE_HALF)
#define TL_SM120_FULLTILE_SINGLE_HALF 0
#endif
  if (k_block_idx != TL_SM120_FULLTILE_SINGLE_KI) {
    return;
  }
  {
    uint32_t single_a0, single_a1, single_a2, single_a3;
    uint32_t single_b0, single_b1, single_b2, single_b3;
#if defined(TL_SM120_FULLTILE_SINGLE_SCRATCH_ONES)
#if TL_SM120_FULLTILE_SINGLE_I != 0 || TL_SM120_FULLTILE_SINGLE_J != 0
#error                                                                         \
    "TL_SM120_FULLTILE_SINGLE_SCRATCH_ONES currently supports site I=0,J=0 only"
#endif
#if !defined(TL_SM120_FULLTILE_SINGLE_SCRATCH_BYTE)
#define TL_SM120_FULLTILE_SINGLE_SCRATCH_BYTE 0x22
#endif
    __shared__ __align__(128) unsigned char tl_sm120_single_a_scratch[4 * 512];
    __shared__ __align__(128) unsigned char tl_sm120_single_b_scratch[4 * 512];
    int const single_warp = (int(threadIdx.x) & 127) >> 5;
    int const lane = int(threadIdx.x) & 31;
    unsigned char *const a_scratch =
        tl_sm120_single_a_scratch + single_warp * 512;
    unsigned char *const b_scratch =
        tl_sm120_single_b_scratch + single_warp * 512;
#pragma unroll
    for (int scratch_i = 0; scratch_i < 16; ++scratch_i) {
      int const byte_i = lane + scratch_i * 32;
      a_scratch[byte_i] = TL_SM120_FULLTILE_SINGLE_SCRATCH_BYTE;
      b_scratch[byte_i] = TL_SM120_FULLTILE_SINGLE_SCRATCH_BYTE;
    }
    __syncwarp();
    detail::sm120_ldmatrix_x4_fp4_u32(a_scratch + lane * 16, single_a0,
                                      single_a1, single_a2, single_a3);
    detail::sm120_ldmatrix_x4_fp4_u32(b_scratch + lane * 16, single_b0,
                                      single_b1, single_b2, single_b3);
#elif defined(TL_SM120_FULLTILE_SINGLE_REAL_SCRATCH)
#if TL_SM120_FULLTILE_SINGLE_I != 0 || TL_SM120_FULLTILE_SINGLE_J != 0
#error                                                                         \
    "TL_SM120_FULLTILE_SINGLE_REAL_SCRATCH currently supports site I=0,J=0 only"
#endif
    __shared__ __align__(
        128) unsigned char tl_sm120_single_a_real_scratch[4 * 512];
    __shared__ __align__(
        128) unsigned char tl_sm120_single_b_real_scratch[4 * 512];
    int const single_warp = (int(threadIdx.x) & 127) >> 5;
    int const lane = int(threadIdx.x) & 31;
    unsigned char *const a_scratch =
        tl_sm120_single_a_real_scratch + single_warp * 512;
    unsigned char *const b_scratch =
        tl_sm120_single_b_real_scratch + single_warp * 512;
    unsigned char const *const a_src =
        static_cast<unsigned char const *>(a_smem_0);
    unsigned char const *const b_src =
        static_cast<unsigned char const *>(b_smem_0);
#pragma unroll
    for (int byte_i = 0; byte_i < 8; ++byte_i) {
      a_scratch[lane * 16 + byte_i] = a_src[byte_i];
      b_scratch[lane * 16 + byte_i] = b_src[byte_i];
      a_scratch[lane * 16 + byte_i + 8] = 0;
      b_scratch[lane * 16 + byte_i + 8] = 0;
    }
    __syncwarp();
    detail::sm120_ldmatrix_x4_fp4_u32(a_scratch + lane * 16, single_a0,
                                      single_a1, single_a2, single_a3);
    detail::sm120_ldmatrix_x4_fp4_u32(b_scratch + lane * 16, single_b0,
                                      single_b1, single_b2, single_b3);
#else
#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_A)
    detail::sm120_ldmatrix_x4_fp4_u32(
        a_cute_m0 + kCuteKAtomStride * TL_SM120_FULLTILE_SINGLE_I, single_a0,
        single_a1, single_a2, single_a3);
#else
#if TL_SM120_FULLTILE_SINGLE_I == 0
    detail::sm120_ldmatrix_x4_blockscaled_operand(
        a_smem_0, single_a0, single_a1, single_a2, single_a3);
#elif TL_SM120_FULLTILE_SINGLE_I == 1
    detail::sm120_ldmatrix_x4_blockscaled_operand(
        a_smem_1, single_a0, single_a1, single_a2, single_a3);
#elif TL_SM120_FULLTILE_SINGLE_I == 2
    detail::sm120_ldmatrix_x4_blockscaled_operand(
        a_smem_2, single_a0, single_a1, single_a2, single_a3);
#elif TL_SM120_FULLTILE_SINGLE_I == 3
    detail::sm120_ldmatrix_x4_blockscaled_operand(
        a_smem_3, single_a0, single_a1, single_a2, single_a3);
#else
#error "TL_SM120_FULLTILE_SINGLE_I must be in [0, 3]"
#endif
#endif

#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_B)
    detail::sm120_ldmatrix_x4_fp4_u32(
        b_cute_n0 + kCuteKAtomStride * TL_SM120_FULLTILE_SINGLE_J, single_b0,
        single_b1, single_b2, single_b3);
#else
#if TL_SM120_FULLTILE_SINGLE_J == 0
    detail::sm120_ldmatrix_x4_blockscaled_operand(
        b_smem_0, single_b0, single_b1, single_b2, single_b3);
#elif TL_SM120_FULLTILE_SINGLE_J == 1
    detail::sm120_ldmatrix_x4_blockscaled_operand(
        b_smem_1, single_b0, single_b1, single_b2, single_b3);
#elif TL_SM120_FULLTILE_SINGLE_J == 2
    detail::sm120_ldmatrix_x4_blockscaled_operand(
        b_smem_2, single_b0, single_b1, single_b2, single_b3);
#elif TL_SM120_FULLTILE_SINGLE_J == 3
    detail::sm120_ldmatrix_x4_blockscaled_operand(
        b_smem_3, single_b0, single_b1, single_b2, single_b3);
#else
#error "TL_SM120_FULLTILE_SINGLE_J must be in [0, 3]"
#endif
#endif
#endif

#if TL_SM120_FULLTILE_SINGLE_I == 0
    uint32_t const single_sa = *sfa_0;
#elif TL_SM120_FULLTILE_SINGLE_I == 1
    uint32_t const single_sa = *sfa_1;
#elif TL_SM120_FULLTILE_SINGLE_I == 2
    uint32_t const single_sa = *sfa_2;
#elif TL_SM120_FULLTILE_SINGLE_I == 3
    uint32_t const single_sa = *sfa_3;
#else
#error "TL_SM120_FULLTILE_SINGLE_I must be in [0, 3]"
#endif

#if TL_SM120_FULLTILE_SINGLE_J == 0
#if TL_SM120_FULLTILE_SINGLE_HALF == 0
    uint32_t const single_sb = *sfb_0;
    uint32_t const single_b_lo = single_b0;
    uint32_t const single_b_hi = single_b1;
#else
    uint32_t const single_sb = *sfb_rep_0;
    uint32_t const single_b_lo = single_b2;
    uint32_t const single_b_hi = single_b3;
#endif
#elif TL_SM120_FULLTILE_SINGLE_J == 1
#if TL_SM120_FULLTILE_SINGLE_HALF == 0
    uint32_t const single_sb = *sfb_1;
    uint32_t const single_b_lo = single_b0;
    uint32_t const single_b_hi = single_b1;
#else
    uint32_t const single_sb = *sfb_rep_1;
    uint32_t const single_b_lo = single_b2;
    uint32_t const single_b_hi = single_b3;
#endif
#elif TL_SM120_FULLTILE_SINGLE_J == 2
#if TL_SM120_FULLTILE_SINGLE_HALF == 0
    uint32_t const single_sb = *sfb_2;
    uint32_t const single_b_lo = single_b0;
    uint32_t const single_b_hi = single_b1;
#else
    uint32_t const single_sb = *sfb_rep_2;
    uint32_t const single_b_lo = single_b2;
    uint32_t const single_b_hi = single_b3;
#endif
#elif TL_SM120_FULLTILE_SINGLE_J == 3
#if TL_SM120_FULLTILE_SINGLE_HALF == 0
    uint32_t const single_sb = *sfb_3;
    uint32_t const single_b_lo = single_b0;
    uint32_t const single_b_hi = single_b1;
#else
    uint32_t const single_sb = *sfb_rep_3;
    uint32_t const single_b_lo = single_b2;
    uint32_t const single_b_hi = single_b3;
#endif
#else
#error "TL_SM120_FULLTILE_SINGLE_J must be in [0, 3]"
#endif

#if defined(TL_SM120_FULLTILE_CUTE_ACCUM_LAYOUT)
    int const single_c_offset = TL_SM120_FULLTILE_SINGLE_I * 4 +
                                TL_SM120_FULLTILE_SINGLE_J * 32 +
                                TL_SM120_FULLTILE_SINGLE_HALF * 16;
#else
    int const single_c_offset = TL_SM120_FULLTILE_SINGLE_I * 32 +
                                TL_SM120_FULLTILE_SINGLE_J * 8 +
                                TL_SM120_FULLTILE_SINGLE_HALF * 4;
#endif
    float *d = c + single_c_offset;
#if defined(TL_SM120_FULLTILE_REG_DEBUG)
#if defined(TL_SM120_FULLTILE_REG_TAG_DEBUG)
    {
      uint32_t const debug_warp = (uint32_t(int(threadIdx.x) & 127) >> 5);
      uint32_t const debug_lane = uint32_t(int(threadIdx.x) & 31);
      uint32_t const debug_base =
          0xa5000000u | (debug_warp << 16) | (debug_lane << 8);
#define TL_SM120_REG_DEBUG_TAG(SLOT)                                           \
  d[SLOT] = __uint_as_float(debug_base | uint32_t(SLOT))
      TL_SM120_REG_DEBUG_TAG(0);
      TL_SM120_REG_DEBUG_TAG(1);
      TL_SM120_REG_DEBUG_TAG(2);
      TL_SM120_REG_DEBUG_TAG(3);
      TL_SM120_REG_DEBUG_TAG(4);
      TL_SM120_REG_DEBUG_TAG(5);
      TL_SM120_REG_DEBUG_TAG(6);
      TL_SM120_REG_DEBUG_TAG(7);
      TL_SM120_REG_DEBUG_TAG(8);
      TL_SM120_REG_DEBUG_TAG(9);
      TL_SM120_REG_DEBUG_TAG(10);
      TL_SM120_REG_DEBUG_TAG(11);
#undef TL_SM120_REG_DEBUG_TAG
    }
#else
    d[0] = __uint_as_float(single_a0);
    d[1] = __uint_as_float(single_a1);
    d[2] = __uint_as_float(single_a2);
    d[3] = __uint_as_float(single_a3);
    d[4] = __uint_as_float(single_b0);
    d[5] = __uint_as_float(single_b1);
    d[6] = __uint_as_float(single_b2);
    d[7] = __uint_as_float(single_b3);
    d[8] = __uint_as_float(single_b_lo);
    d[9] = __uint_as_float(single_b_hi);
    d[10] = __uint_as_float(single_sa);
    d[11] = __uint_as_float(single_sb);
#endif
    return;
#endif
    detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs(
        d, single_a0, single_a1, single_a2, single_a3, single_b_lo, single_b_hi,
        d, single_sa, single_sb);
    return;
  }
#endif

#if !defined(TL_SM120_FULLTILE_BOUNDED_2X2_PACKAGE) &&                         \
    !defined(TL_SM120_FULLTILE_AFULL_B_PANEL_STREAM)
#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_B)
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + 0, b00, b01, b02, b03);
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + kCuteKAtomStride, b10, b11, b12,
                                    b13);
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + 2 * kCuteKAtomStride, b20, b21,
                                    b22, b23);
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + 3 * kCuteKAtomStride, b30, b31,
                                    b32, b33);
#else
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_0, b00, b01, b02, b03);
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_1, b10, b11, b12, b13);
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_2, b20, b21, b22, b23);
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_3, b30, b31, b32, b33);
#endif

#if defined(TL_SM120_FULLTILE_CUTLASS_SF_BASEPTR)
  uint32_t const scale_warp = (uint32_t(int(threadIdx.x) & 127) >> 5);
  uint32_t const scale_warp_m = scale_warp & 1u;
  uint32_t const scale_warp_n = scale_warp >> 1;
  uint32_t const scale_lane = uint32_t(int(threadIdx.x) & 31);
  uint32_t const scale_sfa_row = 8u * (scale_lane & 1u) + (scale_lane >> 2);
  uint32_t const scale_sfb_col = scale_lane >> 2;
  uint32_t const scale_k = uint32_t(k_block_idx);
  uint32_t const scale_m0 = scale_warp_m * 64u + scale_sfa_row;
  uint32_t const scale_m1 = scale_m0 + 16u;
  uint32_t const scale_m2 = scale_m0 + 32u;
  uint32_t const scale_m3 = scale_m0 + 48u;
  uint32_t const scale_n0 = scale_warp_n * 64u + scale_sfb_col;
  uint32_t const scale_n1 = scale_n0 + 16u;
  uint32_t const scale_n2 = scale_n0 + 32u;
  uint32_t const scale_n3 = scale_n0 + 48u;
  uint32_t const *sfa_base = sfa_0;
  uint32_t const *sfb_base = sfb_0;
  uint32_t const sa0 =
      sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m0, scale_k)];
  uint32_t const sa1 =
      sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m1, scale_k)];
  uint32_t const sa2 =
      sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m2, scale_k)];
  uint32_t const sa3 =
      sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m3, scale_k)];
  uint32_t const sb0 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n0, scale_k)];
  uint32_t const sb1 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n1, scale_k)];
  uint32_t const sb2 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n2, scale_k)];
  uint32_t const sb3 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n3, scale_k)];
  uint32_t const sbr0 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n0 + 8u, scale_k)];
  uint32_t const sbr1 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n1 + 8u, scale_k)];
  uint32_t const sbr2 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n2 + 8u, scale_k)];
  uint32_t const sbr3 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n3 + 8u, scale_k)];
#define TL_SM120_SFA0 sa0
#define TL_SM120_SFA1 sa1
#define TL_SM120_SFA2 sa2
#define TL_SM120_SFA3 sa3
#else
  uint32_t const sb0 = *sfb_0;
  uint32_t const sb1 = *sfb_1;
  uint32_t const sb2 = *sfb_2;
  uint32_t const sb3 = *sfb_3;
  uint32_t const sbr0 = *sfb_rep_0;
  uint32_t const sbr1 = *sfb_rep_1;
  uint32_t const sbr2 = *sfb_rep_2;
  uint32_t const sbr3 = *sfb_rep_3;
#define TL_SM120_SFA0 (*sfa_0)
#define TL_SM120_SFA1 (*sfa_1)
#define TL_SM120_SFA2 (*sfa_2)
#define TL_SM120_SFA3 (*sfa_3)
#endif
#endif

#if defined(TL_SM120_FULLTILE_CUTE_ACCUM_LAYOUT) &&                            \
    !defined(TL_SM120_FULLTILE_CUTE_ACCUM_DIRECT)
  float c_cute[128];
#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int const tl_offset = mi * 32 + nj * 8 + half * 4;
        int const cute_offset = mi * 4 + nj * 32 + half * 16;
        *reinterpret_cast<float4 *>(c_cute + cute_offset) =
            *reinterpret_cast<float4 *>(c + tl_offset);
      }
    }
  }
  float *c_mma = c_cute;
#else
  float *c_mma = c;
#endif

#if defined(TL_SM120_FULLTILE_CUTE_ACCUM_LAYOUT) ||                            \
    defined(TL_SM120_FULLTILE_CUTE_ACCUM_DIRECT)
#define TL_SM120_C_OFFSET(I, J, HALF) ((I) * 4 + (J) * 32 + (HALF) * 16)
#else
#define TL_SM120_C_OFFSET(I, J, HALF) ((I) * 32 + (J) * 8 + (HALF) * 4)
#endif

#define TL_SM120_MMA_N8(I, J, HALF, A0, A1, A2, A3, B0, B1, SA, SB)            \
  do {                                                                         \
    float *d = c_mma + TL_SM120_C_OFFSET(I, J, HALF);                          \
    detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs(d, A0, A1, A2, A3, B0,   \
                                                      B1, d, SA, SB);          \
  } while (0)

#define TL_SM120_MMA_ROW(I, A0, A1, A2, A3, SA)                                \
  do {                                                                         \
    TL_SM120_MMA_N8(I, 0, 0, A0, A1, A2, A3, b00, b01, SA, sb0);               \
    TL_SM120_MMA_N8(I, 0, 1, A0, A1, A2, A3, b02, b03, SA, sbr0);              \
    TL_SM120_MMA_N8(I, 1, 0, A0, A1, A2, A3, b10, b11, SA, sb1);               \
    TL_SM120_MMA_N8(I, 1, 1, A0, A1, A2, A3, b12, b13, SA, sbr1);              \
    TL_SM120_MMA_N8(I, 2, 0, A0, A1, A2, A3, b20, b21, SA, sb2);               \
    TL_SM120_MMA_N8(I, 2, 1, A0, A1, A2, A3, b22, b23, SA, sbr2);              \
    TL_SM120_MMA_N8(I, 3, 0, A0, A1, A2, A3, b30, b31, SA, sb3);               \
    TL_SM120_MMA_N8(I, 3, 1, A0, A1, A2, A3, b32, b33, SA, sbr3);              \
  } while (0)

  uint32_t a0, a1, a2, a3;
#if defined(TL_SM120_FULLTILE_AFULL_B_PANEL_STREAM)
  uint32_t a4, a5, a6, a7;
  uint32_t a8, a9, a10, a11;
  uint32_t a12, a13, a14, a15;
#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_A)
  detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 0, a0, a1, a2, a3);
  detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + kCuteKAtomStride, a4, a5, a6,
                                    a7);
  detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 2 * kCuteKAtomStride, a8, a9,
                                    a10, a11);
  detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 3 * kCuteKAtomStride, a12, a13,
                                    a14, a15);
#else
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_0, a0, a1, a2, a3);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_1, a4, a5, a6, a7);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_2, a8, a9, a10, a11);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_3, a12, a13, a14, a15);
#endif

#if defined(TL_SM120_FULLTILE_AFULL_SCALE_REGS)
  uint32_t const sa0 = *sfa_0;
  uint32_t const sa1 = *sfa_1;
  uint32_t const sa2 = *sfa_2;
  uint32_t const sa3 = *sfa_3;
  uint32_t const sb0 = *sfb_0;
  uint32_t const sb1 = *sfb_1;
  uint32_t const sb2 = *sfb_2;
  uint32_t const sb3 = *sfb_3;
  uint32_t const sbr0 = *sfb_rep_0;
  uint32_t const sbr1 = *sfb_rep_1;
  uint32_t const sbr2 = *sfb_rep_2;
  uint32_t const sbr3 = *sfb_rep_3;
#define TL_SM120_SFA0 sa0
#define TL_SM120_SFA1 sa1
#define TL_SM120_SFA2 sa2
#define TL_SM120_SFA3 sa3
#define TL_SM120_SFB0 sb0
#define TL_SM120_SFB1 sb1
#define TL_SM120_SFB2 sb2
#define TL_SM120_SFB3 sb3
#define TL_SM120_SFBR0 sbr0
#define TL_SM120_SFBR1 sbr1
#define TL_SM120_SFBR2 sbr2
#define TL_SM120_SFBR3 sbr3
#else
#define TL_SM120_SFA0 (*sfa_0)
#define TL_SM120_SFA1 (*sfa_1)
#define TL_SM120_SFA2 (*sfa_2)
#define TL_SM120_SFA3 (*sfa_3)
#define TL_SM120_SFB0 (*sfb_0)
#define TL_SM120_SFB1 (*sfb_1)
#define TL_SM120_SFB2 (*sfb_2)
#define TL_SM120_SFB3 (*sfb_3)
#define TL_SM120_SFBR0 (*sfb_rep_0)
#define TL_SM120_SFBR1 (*sfb_rep_1)
#define TL_SM120_SFBR2 (*sfb_rep_2)
#define TL_SM120_SFBR3 (*sfb_rep_3)
#endif

#if defined(TL_SM120_FULLTILE_AFULL_B_PANEL_ADDRS)
#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_B)
  uint32_t const b_panel_0_addr = detail::sm120_smem_addr(b_cute_n0 + 0);
  uint32_t const b_panel_1_addr =
      detail::sm120_smem_addr(b_cute_n0 + kCuteKAtomStride);
  uint32_t const b_panel_2_addr =
      detail::sm120_smem_addr(b_cute_n0 + 2 * kCuteKAtomStride);
  uint32_t const b_panel_3_addr =
      detail::sm120_smem_addr(b_cute_n0 + 3 * kCuteKAtomStride);
#else
  uint32_t const b_panel_0_addr = detail::sm120_smem_addr(b_smem_0);
  uint32_t const b_panel_1_addr = detail::sm120_smem_addr(b_smem_1);
  uint32_t const b_panel_2_addr = detail::sm120_smem_addr(b_smem_2);
  uint32_t const b_panel_3_addr = detail::sm120_smem_addr(b_smem_3);
#endif
#endif

#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_B)
#if defined(TL_SM120_FULLTILE_AFULL_B_PANEL_ADDRS)
#define TL_SM120_LOAD_B_PANEL_0()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand_addr(b_panel_0_addr, b00, b01, \
                                                     b02, b03)
#define TL_SM120_LOAD_B_PANEL_1()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand_addr(b_panel_1_addr, b00, b01, \
                                                     b02, b03)
#define TL_SM120_LOAD_B_PANEL_2()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand_addr(b_panel_2_addr, b00, b01, \
                                                     b02, b03)
#define TL_SM120_LOAD_B_PANEL_3()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand_addr(b_panel_3_addr, b00, b01, \
                                                     b02, b03)
#else
#define TL_SM120_LOAD_B_PANEL_0()                                              \
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + 0, b00, b01, b02, b03)
#define TL_SM120_LOAD_B_PANEL_1()                                              \
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + kCuteKAtomStride, b00, b01,    \
                                    b02, b03)
#define TL_SM120_LOAD_B_PANEL_2()                                              \
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + 2 * kCuteKAtomStride, b00,     \
                                    b01, b02, b03)
#define TL_SM120_LOAD_B_PANEL_3()                                              \
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + 3 * kCuteKAtomStride, b00,     \
                                    b01, b02, b03)
#endif
#else
#if defined(TL_SM120_FULLTILE_AFULL_B_PANEL_ADDRS)
#define TL_SM120_LOAD_B_PANEL_0()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand_addr(b_panel_0_addr, b00, b01, \
                                                     b02, b03)
#define TL_SM120_LOAD_B_PANEL_1()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand_addr(b_panel_1_addr, b00, b01, \
                                                     b02, b03)
#define TL_SM120_LOAD_B_PANEL_2()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand_addr(b_panel_2_addr, b00, b01, \
                                                     b02, b03)
#define TL_SM120_LOAD_B_PANEL_3()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand_addr(b_panel_3_addr, b00, b01, \
                                                     b02, b03)
#else
#define TL_SM120_LOAD_B_PANEL_0()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_0, b00, b01, b02, b03)
#define TL_SM120_LOAD_B_PANEL_1()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_1, b00, b01, b02, b03)
#define TL_SM120_LOAD_B_PANEL_2()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_2, b00, b01, b02, b03)
#define TL_SM120_LOAD_B_PANEL_3()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_3, b00, b01, b02, b03)
#endif
#endif

#define TL_SM120_MMA_B_PANEL(J, LOAD_B, SFB, SFB_REP)                          \
  do {                                                                         \
    LOAD_B();                                                                  \
    uint32_t const sb = (SFB);                                                 \
    uint32_t const sbr = (SFB_REP);                                            \
    TL_SM120_MMA_N8(0, J, 0, a0, a1, a2, a3, b00, b01, TL_SM120_SFA0, sb);     \
    TL_SM120_MMA_N8(1, J, 0, a4, a5, a6, a7, b00, b01, TL_SM120_SFA1, sb);     \
    TL_SM120_MMA_N8(2, J, 0, a8, a9, a10, a11, b00, b01, TL_SM120_SFA2, sb);   \
    TL_SM120_MMA_N8(3, J, 0, a12, a13, a14, a15, b00, b01, TL_SM120_SFA3, sb); \
    TL_SM120_MMA_N8(0, J, 1, a0, a1, a2, a3, b02, b03, TL_SM120_SFA0, sbr);    \
    TL_SM120_MMA_N8(1, J, 1, a4, a5, a6, a7, b02, b03, TL_SM120_SFA1, sbr);    \
    TL_SM120_MMA_N8(2, J, 1, a8, a9, a10, a11, b02, b03, TL_SM120_SFA2, sbr);  \
    TL_SM120_MMA_N8(3, J, 1, a12, a13, a14, a15, b02, b03, TL_SM120_SFA3,      \
                    sbr);                                                      \
  } while (0)

  TL_SM120_MMA_B_PANEL(0, TL_SM120_LOAD_B_PANEL_0, TL_SM120_SFB0,
                       TL_SM120_SFBR0);
  TL_SM120_MMA_B_PANEL(1, TL_SM120_LOAD_B_PANEL_1, TL_SM120_SFB1,
                       TL_SM120_SFBR1);
  TL_SM120_MMA_B_PANEL(2, TL_SM120_LOAD_B_PANEL_2, TL_SM120_SFB2,
                       TL_SM120_SFBR2);
  TL_SM120_MMA_B_PANEL(3, TL_SM120_LOAD_B_PANEL_3, TL_SM120_SFB3,
                       TL_SM120_SFBR3);

#undef TL_SM120_MMA_B_PANEL
#undef TL_SM120_LOAD_B_PANEL_3
#undef TL_SM120_LOAD_B_PANEL_2
#undef TL_SM120_LOAD_B_PANEL_1
#undef TL_SM120_LOAD_B_PANEL_0
#undef TL_SM120_SFBR3
#undef TL_SM120_SFBR2
#undef TL_SM120_SFBR1
#undef TL_SM120_SFBR0
#undef TL_SM120_SFB3
#undef TL_SM120_SFB2
#undef TL_SM120_SFB1
#undef TL_SM120_SFB0
#undef TL_SM120_SFA3
#undef TL_SM120_SFA2
#undef TL_SM120_SFA1
#undef TL_SM120_SFA0
#elif defined(TL_SM120_FULLTILE_PAIR_ASM_B)
  uint32_t a4, a5, a6, a7;
#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_A)
#define TL_SM120_LOAD_A_PAIR_01()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 0, a0, a1, a2, a3);          \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + kCuteKAtomStride, a4, a5,    \
                                      a6, a7);                                 \
  } while (0)
#define TL_SM120_LOAD_A_PAIR_23()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 2 * kCuteKAtomStride, a0,    \
                                      a1, a2, a3);                             \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 3 * kCuteKAtomStride, a4,    \
                                      a5, a6, a7);                             \
  } while (0)
#else
#define TL_SM120_LOAD_A_PAIR_01()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_0, a0, a1, a2, a3);   \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_1, a4, a5, a6, a7);   \
  } while (0)
#define TL_SM120_LOAD_A_PAIR_23()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_2, a0, a1, a2, a3);   \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_3, a4, a5, a6, a7);   \
  } while (0)
#endif

#define TL_SM120_MMA2_SAME_B(I0, I1, J, HALF, B0, B1, SA0, SA1, SB)            \
  do {                                                                         \
    float *d0 = c_mma + TL_SM120_C_OFFSET(I0, J, HALF);                        \
    float *d1 = c_mma + TL_SM120_C_OFFSET(I1, J, HALF);                        \
    detail::sm120_mma2_m16n8k64_mxf4nvf4_4x_ue4m3_same_b_regs(                 \
        d0, a0, a1, a2, a3, d0, SA0, d1, a4, a5, a6, a7, d1, SA1, B0, B1, SB); \
  } while (0)

#define TL_SM120_MMA2_PAIR_ROWS(I0, I1, SA0, SA1)                              \
  do {                                                                         \
    TL_SM120_MMA2_SAME_B(I0, I1, 0, 0, b00, b01, SA0, SA1, sb0);               \
    TL_SM120_MMA2_SAME_B(I0, I1, 0, 1, b02, b03, SA0, SA1, sbr0);              \
    TL_SM120_MMA2_SAME_B(I0, I1, 1, 0, b10, b11, SA0, SA1, sb1);               \
    TL_SM120_MMA2_SAME_B(I0, I1, 1, 1, b12, b13, SA0, SA1, sbr1);              \
    TL_SM120_MMA2_SAME_B(I0, I1, 2, 0, b20, b21, SA0, SA1, sb2);               \
    TL_SM120_MMA2_SAME_B(I0, I1, 2, 1, b22, b23, SA0, SA1, sbr2);              \
    TL_SM120_MMA2_SAME_B(I0, I1, 3, 0, b30, b31, SA0, SA1, sb3);               \
    TL_SM120_MMA2_SAME_B(I0, I1, 3, 1, b32, b33, SA0, SA1, sbr3);              \
  } while (0)

  TL_SM120_LOAD_A_PAIR_01();
  TL_SM120_MMA2_PAIR_ROWS(0, 1, *sfa_0, *sfa_1);
  TL_SM120_LOAD_A_PAIR_23();
  TL_SM120_MMA2_PAIR_ROWS(2, 3, *sfa_2, *sfa_3);

#undef TL_SM120_MMA2_PAIR_ROWS
#undef TL_SM120_MMA2_SAME_B
#undef TL_SM120_LOAD_A_PAIR_23
#undef TL_SM120_LOAD_A_PAIR_01
#elif defined(TL_SM120_FULLTILE_BOUNDED_2X2_PACKAGE)
  uint32_t a4, a5, a6, a7;
#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_A)
#define TL_SM120_LOAD_A_PAIR_01()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 0, a0, a1, a2, a3);          \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + kCuteKAtomStride, a4, a5,    \
                                      a6, a7);                                 \
  } while (0)
#define TL_SM120_LOAD_A_PAIR_23()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 2 * kCuteKAtomStride, a0,    \
                                      a1, a2, a3);                             \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 3 * kCuteKAtomStride, a4,    \
                                      a5, a6, a7);                             \
  } while (0)
#else
#define TL_SM120_LOAD_A_PAIR_01()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_0, a0, a1, a2, a3);   \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_1, a4, a5, a6, a7);   \
  } while (0)
#define TL_SM120_LOAD_A_PAIR_23()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_2, a0, a1, a2, a3);   \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_3, a4, a5, a6, a7);   \
  } while (0)
#endif

#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_B)
#define TL_SM120_LOAD_B_PANEL_0()                                              \
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + 0, b00, b01, b02, b03)
#define TL_SM120_LOAD_B_PANEL_1()                                              \
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + kCuteKAtomStride, b00, b01,    \
                                    b02, b03)
#define TL_SM120_LOAD_B_PANEL_2()                                              \
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + 2 * kCuteKAtomStride, b00,     \
                                    b01, b02, b03)
#define TL_SM120_LOAD_B_PANEL_3()                                              \
  detail::sm120_ldmatrix_x4_fp4_u32(b_cute_n0 + 3 * kCuteKAtomStride, b00,     \
                                    b01, b02, b03)
#else
#define TL_SM120_LOAD_B_PANEL_0()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_0, b00, b01, b02, b03)
#define TL_SM120_LOAD_B_PANEL_1()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_1, b00, b01, b02, b03)
#define TL_SM120_LOAD_B_PANEL_2()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_2, b00, b01, b02, b03)
#define TL_SM120_LOAD_B_PANEL_3()                                              \
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_3, b00, b01, b02, b03)
#endif

#define TL_SM120_MMA_PAIR_FWD(I0, I1, J, HALF, B0, B1, SA0, SA1, SB)           \
  do {                                                                         \
    TL_SM120_MMA_N8(I0, J, HALF, a0, a1, a2, a3, B0, B1, SA0, SB);             \
    TL_SM120_MMA_N8(I1, J, HALF, a4, a5, a6, a7, B0, B1, SA1, SB);             \
  } while (0)

#define TL_SM120_MMA_PAIR_REV(I0, I1, J, HALF, B0, B1, SA0, SA1, SB)           \
  do {                                                                         \
    TL_SM120_MMA_N8(I1, J, HALF, a4, a5, a6, a7, B0, B1, SA1, SB);             \
    TL_SM120_MMA_N8(I0, J, HALF, a0, a1, a2, a3, B0, B1, SA0, SB);             \
  } while (0)

#define TL_SM120_MMA_B_PANEL(I0, I1, J, LOAD_B, SFB, SFB_REP, SA0, SA1)        \
  do {                                                                         \
    LOAD_B();                                                                  \
    uint32_t const sb = *(SFB);                                                \
    uint32_t const sbr = *(SFB_REP);                                           \
    TL_SM120_MMA_PAIR_FWD(I0, I1, J, 0, b00, b01, SA0, SA1, sb);               \
    TL_SM120_MMA_PAIR_REV(I0, I1, J, 1, b02, b03, SA0, SA1, sbr);              \
  } while (0)

#define TL_SM120_MMA_A_PAIR(I0, I1, LOAD_A, SA0, SA1)                          \
  do {                                                                         \
    LOAD_A();                                                                  \
    TL_SM120_MMA_B_PANEL(I0, I1, 0, TL_SM120_LOAD_B_PANEL_0, sfb_0, sfb_rep_0, \
                         SA0, SA1);                                            \
    TL_SM120_MMA_B_PANEL(I0, I1, 1, TL_SM120_LOAD_B_PANEL_1, sfb_1, sfb_rep_1, \
                         SA0, SA1);                                            \
    TL_SM120_MMA_B_PANEL(I0, I1, 2, TL_SM120_LOAD_B_PANEL_2, sfb_2, sfb_rep_2, \
                         SA0, SA1);                                            \
    TL_SM120_MMA_B_PANEL(I0, I1, 3, TL_SM120_LOAD_B_PANEL_3, sfb_3, sfb_rep_3, \
                         SA0, SA1);                                            \
  } while (0)

  TL_SM120_MMA_A_PAIR(0, 1, TL_SM120_LOAD_A_PAIR_01, *sfa_0, *sfa_1);
  TL_SM120_MMA_A_PAIR(2, 3, TL_SM120_LOAD_A_PAIR_23, *sfa_2, *sfa_3);

#undef TL_SM120_MMA_A_PAIR
#undef TL_SM120_MMA_B_PANEL
#undef TL_SM120_MMA_PAIR_REV
#undef TL_SM120_MMA_PAIR_FWD
#undef TL_SM120_LOAD_B_PANEL_3
#undef TL_SM120_LOAD_B_PANEL_2
#undef TL_SM120_LOAD_B_PANEL_1
#undef TL_SM120_LOAD_B_PANEL_0
#undef TL_SM120_LOAD_A_PAIR_23
#undef TL_SM120_LOAD_A_PAIR_01
#elif defined(TL_SM120_FULLTILE_A2_BPAIR_ORDER)
  uint32_t a4, a5, a6, a7;
#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_A)
#define TL_SM120_LOAD_A_PAIR_01()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 0, a0, a1, a2, a3);          \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + kCuteKAtomStride, a4, a5,    \
                                      a6, a7);                                 \
  } while (0)
#define TL_SM120_LOAD_A_PAIR_23()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 2 * kCuteKAtomStride, a0,    \
                                      a1, a2, a3);                             \
    detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 3 * kCuteKAtomStride, a4,    \
                                      a5, a6, a7);                             \
  } while (0)
#else
#define TL_SM120_LOAD_A_PAIR_01()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_0, a0, a1, a2, a3);   \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_1, a4, a5, a6, a7);   \
  } while (0)
#define TL_SM120_LOAD_A_PAIR_23()                                              \
  do {                                                                         \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_2, a0, a1, a2, a3);   \
    detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_3, a4, a5, a6, a7);   \
  } while (0)
#endif

#define TL_SM120_MMA_PAIR_FWD(I0, I1, J, HALF, B0, B1, SA0, SA1, SB)           \
  do {                                                                         \
    TL_SM120_MMA_N8(I0, J, HALF, a0, a1, a2, a3, B0, B1, SA0, SB);             \
    TL_SM120_MMA_N8(I1, J, HALF, a4, a5, a6, a7, B0, B1, SA1, SB);             \
  } while (0)

#define TL_SM120_MMA_PAIR_REV(I0, I1, J, HALF, B0, B1, SA0, SA1, SB)           \
  do {                                                                         \
    TL_SM120_MMA_N8(I1, J, HALF, a4, a5, a6, a7, B0, B1, SA1, SB);             \
    TL_SM120_MMA_N8(I0, J, HALF, a0, a1, a2, a3, B0, B1, SA0, SB);             \
  } while (0)

#define TL_SM120_MMA_PAIR_ROWS(I0, I1, SA0, SA1)                               \
  do {                                                                         \
    TL_SM120_MMA_PAIR_FWD(I0, I1, 0, 0, b00, b01, SA0, SA1, sb0);              \
    TL_SM120_MMA_PAIR_REV(I0, I1, 0, 1, b02, b03, SA0, SA1, sbr0);             \
    TL_SM120_MMA_PAIR_FWD(I0, I1, 1, 0, b10, b11, SA0, SA1, sb1);              \
    TL_SM120_MMA_PAIR_REV(I0, I1, 1, 1, b12, b13, SA0, SA1, sbr1);             \
    TL_SM120_MMA_PAIR_FWD(I0, I1, 2, 0, b20, b21, SA0, SA1, sb2);              \
    TL_SM120_MMA_PAIR_REV(I0, I1, 2, 1, b22, b23, SA0, SA1, sbr2);             \
    TL_SM120_MMA_PAIR_FWD(I0, I1, 3, 0, b30, b31, SA0, SA1, sb3);              \
    TL_SM120_MMA_PAIR_REV(I0, I1, 3, 1, b32, b33, SA0, SA1, sbr3);             \
  } while (0)

  TL_SM120_LOAD_A_PAIR_01();
  TL_SM120_MMA_PAIR_ROWS(0, 1, TL_SM120_SFA0, TL_SM120_SFA1);
  TL_SM120_LOAD_A_PAIR_23();
  TL_SM120_MMA_PAIR_ROWS(2, 3, TL_SM120_SFA2, TL_SM120_SFA3);

#undef TL_SM120_MMA_PAIR_ROWS
#undef TL_SM120_MMA_PAIR_REV
#undef TL_SM120_MMA_PAIR_FWD
#undef TL_SM120_LOAD_A_PAIR_23
#undef TL_SM120_LOAD_A_PAIR_01
#elif defined(TL_SM120_FULLTILE_ROLLING_A_ORDER)
#if defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                                \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_A)
#define TL_SM120_LOAD_A_ROW(I)                                                 \
  detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + (I) * kCuteKAtomStride, a0,    \
                                    a1, a2, a3)
#else
#define TL_SM120_LOAD_A_ROW(I)                                                 \
  do {                                                                         \
    if ((I) == 0) {                                                            \
      detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_0, a0, a1, a2, a3); \
    } else if ((I) == 1) {                                                     \
      detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_1, a0, a1, a2, a3); \
    } else if ((I) == 2) {                                                     \
      detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_2, a0, a1, a2, a3); \
    } else {                                                                   \
      detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_3, a0, a1, a2, a3); \
    }                                                                          \
  } while (0)
#endif

#define TL_SM120_MMA_N8_ROLL(I, J, HALF, B0, B1, SA, SB)                       \
  do {                                                                         \
    TL_SM120_LOAD_A_ROW(I);                                                    \
    TL_SM120_MMA_N8(I, J, HALF, a0, a1, a2, a3, B0, B1, SA, SB);               \
  } while (0)

#define TL_SM120_MMA_N8_ROLL_FWD(J, HALF, B0, B1, SB)                          \
  do {                                                                         \
    TL_SM120_MMA_N8_ROLL(0, J, HALF, B0, B1, *sfa_0, SB);                      \
    TL_SM120_MMA_N8_ROLL(1, J, HALF, B0, B1, *sfa_1, SB);                      \
    TL_SM120_MMA_N8_ROLL(2, J, HALF, B0, B1, *sfa_2, SB);                      \
    TL_SM120_MMA_N8_ROLL(3, J, HALF, B0, B1, *sfa_3, SB);                      \
  } while (0)

#define TL_SM120_MMA_N8_ROLL_REV(J, HALF, B0, B1, SB)                          \
  do {                                                                         \
    TL_SM120_MMA_N8_ROLL(3, J, HALF, B0, B1, *sfa_3, SB);                      \
    TL_SM120_MMA_N8_ROLL(2, J, HALF, B0, B1, *sfa_2, SB);                      \
    TL_SM120_MMA_N8_ROLL(1, J, HALF, B0, B1, *sfa_1, SB);                      \
    TL_SM120_MMA_N8_ROLL(0, J, HALF, B0, B1, *sfa_0, SB);                      \
  } while (0)

  TL_SM120_MMA_N8_ROLL_FWD(0, 0, b00, b01, sb0);
  TL_SM120_MMA_N8_ROLL_REV(0, 1, b02, b03, sbr0);
  TL_SM120_MMA_N8_ROLL_FWD(1, 0, b10, b11, sb1);
  TL_SM120_MMA_N8_ROLL_REV(1, 1, b12, b13, sbr1);
  TL_SM120_MMA_N8_ROLL_FWD(2, 0, b20, b21, sb2);
  TL_SM120_MMA_N8_ROLL_REV(2, 1, b22, b23, sbr2);
  TL_SM120_MMA_N8_ROLL_FWD(3, 0, b30, b31, sb3);
  TL_SM120_MMA_N8_ROLL_REV(3, 1, b32, b33, sbr3);

#undef TL_SM120_MMA_N8_ROLL_REV
#undef TL_SM120_MMA_N8_ROLL_FWD
#undef TL_SM120_MMA_N8_ROLL
#undef TL_SM120_LOAD_A_ROW
#elif defined(TL_SM120_FULLTILE_CUTE_ROWSTART) ||                              \
    defined(TL_SM120_FULLTILE_CUTE_ROWSTART_A)
  detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 0, a0, a1, a2, a3);
  TL_SM120_MMA_ROW(0, a0, a1, a2, a3, TL_SM120_SFA0);
  detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + kCuteKAtomStride, a0, a1, a2,
                                    a3);
  TL_SM120_MMA_ROW(1, a0, a1, a2, a3, TL_SM120_SFA1);
  detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 2 * kCuteKAtomStride, a0, a1,
                                    a2, a3);
  TL_SM120_MMA_ROW(2, a0, a1, a2, a3, TL_SM120_SFA2);
  detail::sm120_ldmatrix_x4_fp4_u32(a_cute_m0 + 3 * kCuteKAtomStride, a0, a1,
                                    a2, a3);
  TL_SM120_MMA_ROW(3, a0, a1, a2, a3, TL_SM120_SFA3);
#else
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_0, a0, a1, a2, a3);
  TL_SM120_MMA_ROW(0, a0, a1, a2, a3, TL_SM120_SFA0);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_1, a0, a1, a2, a3);
  TL_SM120_MMA_ROW(1, a0, a1, a2, a3, TL_SM120_SFA1);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_2, a0, a1, a2, a3);
  TL_SM120_MMA_ROW(2, a0, a1, a2, a3, TL_SM120_SFA2);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_3, a0, a1, a2, a3);
  TL_SM120_MMA_ROW(3, a0, a1, a2, a3, TL_SM120_SFA3);
#endif

#if !defined(TL_SM120_FULLTILE_BOUNDED_2X2_PACKAGE) &&                         \
    !defined(TL_SM120_FULLTILE_AFULL_B_PANEL_STREAM)
#undef TL_SM120_SFA3
#undef TL_SM120_SFA2
#undef TL_SM120_SFA1
#undef TL_SM120_SFA0
#endif

#if defined(TL_SM120_FULLTILE_CUTE_ACCUM_LAYOUT) &&                            \
    !defined(TL_SM120_FULLTILE_CUTE_ACCUM_DIRECT)
#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int const tl_offset = mi * 32 + nj * 8 + half * 4;
        int const cute_offset = mi * 4 + nj * 32 + half * 16;
        *reinterpret_cast<float4 *>(c + tl_offset) =
            *reinterpret_cast<float4 *>(c_cute + cute_offset);
      }
    }
  }
#endif

#undef TL_SM120_MMA_ROW
#undef TL_SM120_MMA_N8
#undef TL_SM120_C_OFFSET
}

TL_DEVICE void sm120_mma_blockscaled_kblock_fulltile_ab_owner_wide(
    float *c, const void *a_smem_0, const void *a_smem_1, const void *a_smem_2,
    const void *a_smem_3, const void *b_smem_0, const void *b_smem_1,
    const void *b_smem_2, const void *b_smem_3, const uint32_t *sfa_base,
    const uint32_t *sfb_base, int k_block_idx) {
  uint32_t b00, b01, b02, b03;
  uint32_t b10, b11, b12, b13;
  uint32_t b20, b21, b22, b23;
  uint32_t b30, b31, b32, b33;

  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_0, b00, b01, b02, b03);
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_1, b10, b11, b12, b13);
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_2, b20, b21, b22, b23);
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_3, b30, b31, b32, b33);

  uint32_t const tx = uint32_t(int(threadIdx.x) & 127);
  uint32_t const lane = tx & 31u;
  uint32_t const qlane = lane & 3u;
  uint32_t const warp = tx >> 5;
  uint32_t const warp_m = warp & 1u;
  uint32_t const warp_n = warp >> 1;
  uint32_t const sfa_row = 8u * (lane & 1u) + (lane >> 2);
  uint32_t const sfb_col = lane >> 2;
  uint32_t const a_owner_in_pair = qlane >> 1;
  uint32_t const scale_k = uint32_t(k_block_idx);

  uint32_t const scale_m0 = warp_m * 64u + a_owner_in_pair * 16u + sfa_row;
  uint32_t const scale_m1 = scale_m0 + 32u;
  uint32_t const scale_n0 = warp_n * 64u + qlane * 8u + sfb_col;
  uint32_t const scale_n1 = scale_n0 + 32u;
  uint32_t const sa_owner0 =
      sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m0, scale_k)];
  uint32_t const sa_owner1 =
      sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m1, scale_k)];
  uint32_t const sb_owner0 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n0, scale_k)];
  uint32_t const sb_owner1 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n1, scale_k)];

#if defined(TL_SM120_FULLTILE_CUTE_ACCUM_LAYOUT) ||                            \
    defined(TL_SM120_FULLTILE_CUTE_ACCUM_DIRECT)
#define TL_SM120_OWNER_C_OFFSET(I, J, HALF) ((I) * 4 + (J) * 32 + (HALF) * 16)
#else
#define TL_SM120_OWNER_C_OFFSET(I, J, HALF) ((I) * 32 + (J) * 8 + (HALF) * 4)
#endif

#define TL_SM120_OWNER_MMA_N8(I, J, HALF, A0, A1, A2, A3, B0, B1, SA, SB,      \
                              SA_TID, SB_TID)                                  \
  do {                                                                         \
    float *d = c + TL_SM120_OWNER_C_OFFSET(I, J, HALF);                        \
    detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs(                         \
        d, A0, A1, A2, A3, B0, B1, d, SA, SB, 0, uint16_t(SA_TID), 0,          \
        uint16_t(SB_TID));                                                     \
  } while (0)

#define TL_SM120_OWNER_MMA_ROW(I, A0, A1, A2, A3, SA, SA_TID)                  \
  do {                                                                         \
    TL_SM120_OWNER_MMA_N8(I, 0, 0, A0, A1, A2, A3, b00, b01, SA, sb_owner0,    \
                          SA_TID, 0);                                          \
    TL_SM120_OWNER_MMA_N8(I, 0, 1, A0, A1, A2, A3, b02, b03, SA, sb_owner0,    \
                          SA_TID, 1);                                          \
    TL_SM120_OWNER_MMA_N8(I, 1, 0, A0, A1, A2, A3, b10, b11, SA, sb_owner0,    \
                          SA_TID, 2);                                          \
    TL_SM120_OWNER_MMA_N8(I, 1, 1, A0, A1, A2, A3, b12, b13, SA, sb_owner0,    \
                          SA_TID, 3);                                          \
    TL_SM120_OWNER_MMA_N8(I, 2, 0, A0, A1, A2, A3, b20, b21, SA, sb_owner1,    \
                          SA_TID, 0);                                          \
    TL_SM120_OWNER_MMA_N8(I, 2, 1, A0, A1, A2, A3, b22, b23, SA, sb_owner1,    \
                          SA_TID, 1);                                          \
    TL_SM120_OWNER_MMA_N8(I, 3, 0, A0, A1, A2, A3, b30, b31, SA, sb_owner1,    \
                          SA_TID, 2);                                          \
    TL_SM120_OWNER_MMA_N8(I, 3, 1, A0, A1, A2, A3, b32, b33, SA, sb_owner1,    \
                          SA_TID, 3);                                          \
  } while (0)

  uint32_t a0, a1, a2, a3;
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_0, a0, a1, a2, a3);
  TL_SM120_OWNER_MMA_ROW(0, a0, a1, a2, a3, sa_owner0, 0);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_1, a0, a1, a2, a3);
  TL_SM120_OWNER_MMA_ROW(1, a0, a1, a2, a3, sa_owner0, 1);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_2, a0, a1, a2, a3);
  TL_SM120_OWNER_MMA_ROW(2, a0, a1, a2, a3, sa_owner1, 0);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_3, a0, a1, a2, a3);
  TL_SM120_OWNER_MMA_ROW(3, a0, a1, a2, a3, sa_owner1, 1);

#undef TL_SM120_OWNER_MMA_ROW
#undef TL_SM120_OWNER_MMA_N8
#undef TL_SM120_OWNER_C_OFFSET
}

TL_DEVICE void sm120_mma_blockscaled_kblock_fulltile_afull_bpanel_owner_wide(
    float *c, const void *a_smem_0, const void *a_smem_1, const void *a_smem_2,
    const void *a_smem_3, const void *b_smem_0, const void *b_smem_1,
    const void *b_smem_2, const void *b_smem_3, const uint32_t *sfa_base,
    const uint32_t *sfb_base, int k_block_idx) {
  uint32_t b00, b01, b02, b03;
  uint32_t b10, b11, b12, b13;

  // Load B0 first so the following A/scale setup gives it real distance before
  // its first OMMA use.
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_0, b00, b01, b02, b03);

  uint32_t const tx = uint32_t(int(threadIdx.x) & 127);
  uint32_t const lane = tx & 31u;
  uint32_t const qlane = lane & 3u;
  uint32_t const warp = tx >> 5;
  uint32_t const warp_m = warp & 1u;
  uint32_t const warp_n = warp >> 1;
  uint32_t const sfa_row = 8u * (lane & 1u) + (lane >> 2);
  uint32_t const sfb_col = lane >> 2;
  uint32_t const a_owner_in_pair = qlane >> 1;
  uint32_t const scale_k = uint32_t(k_block_idx);

  uint32_t const scale_m0 = warp_m * 64u + a_owner_in_pair * 16u + sfa_row;
  uint32_t const scale_m1 = scale_m0 + 32u;
  uint32_t const scale_n0 = warp_n * 64u + qlane * 8u + sfb_col;
  uint32_t const scale_n1 = scale_n0 + 32u;
  uint32_t const sa_owner0 =
      sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m0, scale_k)];
  uint32_t const sa_owner1 =
      sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m1, scale_k)];
  uint32_t const sb_owner0 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n0, scale_k)];
  uint32_t const sb_owner1 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n1, scale_k)];

  uint32_t a00, a01, a02, a03;
  uint32_t a10, a11, a12, a13;
  uint32_t a20, a21, a22, a23;
  uint32_t a30, a31, a32, a33;
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_0, a00, a01, a02, a03);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_1, a10, a11, a12, a13);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_2, a20, a21, a22, a23);
  detail::sm120_ldmatrix_x4_blockscaled_operand(a_smem_3, a30, a31, a32, a33);

#if defined(TL_SM120_FULLTILE_CUTE_ACCUM_LAYOUT) ||                            \
    defined(TL_SM120_FULLTILE_CUTE_ACCUM_DIRECT)
#define TL_SM120_AFULL_C_OFFSET(I, J, HALF) ((I) * 4 + (J) * 32 + (HALF) * 16)
#else
#define TL_SM120_AFULL_C_OFFSET(I, J, HALF) ((I) * 32 + (J) * 8 + (HALF) * 4)
#endif

#define TL_SM120_AFULL_MMA_N8(I, J, HALF, A0, A1, A2, A3, B0, B1, SA, SB,      \
                              SA_TID, SB_TID)                                  \
  do {                                                                         \
    float *d = c + TL_SM120_AFULL_C_OFFSET(I, J, HALF);                        \
    detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs(                         \
        d, A0, A1, A2, A3, B0, B1, d, SA, SB, 0, uint16_t(SA_TID), 0,          \
        uint16_t(SB_TID));                                                     \
  } while (0)

#define TL_SM120_AFULL_MMA_PANEL_LO(J, B0, B1, SB, SB_TID)                     \
  do {                                                                         \
    TL_SM120_AFULL_MMA_N8(0, J, 0, a00, a01, a02, a03, B0, B1, sa_owner0, SB,  \
                          0, SB_TID);                                          \
    TL_SM120_AFULL_MMA_N8(1, J, 0, a10, a11, a12, a13, B0, B1, sa_owner0, SB,  \
                          1, SB_TID);                                          \
    TL_SM120_AFULL_MMA_N8(2, J, 0, a20, a21, a22, a23, B0, B1, sa_owner1, SB,  \
                          0, SB_TID);                                          \
    TL_SM120_AFULL_MMA_N8(3, J, 0, a30, a31, a32, a33, B0, B1, sa_owner1, SB,  \
                          1, SB_TID);                                          \
  } while (0)

#define TL_SM120_AFULL_MMA_PANEL_HI(J, B2, B3, SB, SB_TID)                     \
  do {                                                                         \
    TL_SM120_AFULL_MMA_N8(3, J, 1, a30, a31, a32, a33, B2, B3, sa_owner1, SB,  \
                          1, SB_TID);                                          \
    TL_SM120_AFULL_MMA_N8(2, J, 1, a20, a21, a22, a23, B2, B3, sa_owner1, SB,  \
                          0, SB_TID);                                          \
    TL_SM120_AFULL_MMA_N8(1, J, 1, a10, a11, a12, a13, B2, B3, sa_owner0, SB,  \
                          1, SB_TID);                                          \
    TL_SM120_AFULL_MMA_N8(0, J, 1, a00, a01, a02, a03, B2, B3, sa_owner0, SB,  \
                          0, SB_TID);                                          \
  } while (0)

  TL_SM120_AFULL_MMA_PANEL_LO(0, b00, b01, sb_owner0, 0);
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_1, b10, b11, b12, b13);
  TL_SM120_AFULL_MMA_PANEL_HI(0, b02, b03, sb_owner0, 1);
  TL_SM120_AFULL_MMA_PANEL_LO(1, b10, b11, sb_owner0, 2);
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_2, b00, b01, b02, b03);
  TL_SM120_AFULL_MMA_PANEL_HI(1, b12, b13, sb_owner0, 3);
  TL_SM120_AFULL_MMA_PANEL_LO(2, b00, b01, sb_owner1, 0);
  detail::sm120_ldmatrix_x4_blockscaled_operand(b_smem_3, b10, b11, b12, b13);
  TL_SM120_AFULL_MMA_PANEL_HI(2, b02, b03, sb_owner1, 1);
  TL_SM120_AFULL_MMA_PANEL_LO(3, b10, b11, sb_owner1, 2);
  TL_SM120_AFULL_MMA_PANEL_HI(3, b12, b13, sb_owner1, 3);

#undef TL_SM120_AFULL_MMA_PANEL_HI
#undef TL_SM120_AFULL_MMA_PANEL_LO
#undef TL_SM120_AFULL_MMA_N8
#undef TL_SM120_AFULL_C_OFFSET
}

struct SM120FulltileABOwnerWidePackage {
  uint32_t a00, a01, a02, a03;
  uint32_t a10, a11, a12, a13;
  uint32_t a20, a21, a22, a23;
  uint32_t a30, a31, a32, a33;
  uint32_t b00, b01, b02, b03;
  uint32_t b10, b11, b12, b13;
  uint32_t b20, b21, b22, b23;
  uint32_t b30, b31, b32, b33;
#if defined(TL_SM120_FULLTILE_SCALE_SLOT_PACKAGE) ||                           \
    defined(TL_SM120_FULLTILE_SCALE_SELECTOR0_RETAINED_PACKAGE)
  uint32_t sa0, sa1, sa2, sa3;
  uint32_t sb0, sb1, sb2, sb3, sb4, sb5, sb6, sb7;
#else
  uint32_t sa0, sa1;
  uint32_t sb0, sb1;
#endif
};

TL_DEVICE void sm120_copy_fulltile_ab_owner_wide_package(
    SM120FulltileABOwnerWidePackage &pkg, const char *a_base,
    const char *b_base, const uint32_t *sfa_base, const uint32_t *sfb_base,
    int k_block_idx) {
  uint32_t const tx = uint32_t(int(threadIdx.x) & 127);
  uint32_t const lane = tx & 31u;
  uint32_t const warp = tx >> 5;
  uint32_t const warp_m = warp & 1u;
  uint32_t const warp_n = warp >> 1;
  uint32_t const scale_k = uint32_t(k_block_idx);
#if !defined(TL_SM120_FULLTILE_SCALE_SLOT_PACKAGE)
  uint32_t const qlane = lane & 3u;
  uint32_t const sfa_row = 8u * (lane & 1u) + (lane >> 2);
  uint32_t const sfb_col = lane >> 2;
  uint32_t const a_owner_in_pair = qlane >> 1;
  uint32_t const scale_m0 = warp_m * 64u + a_owner_in_pair * 16u + sfa_row;
  uint32_t const scale_n0 = warp_n * 64u + qlane * 8u + sfb_col;
#endif
#if defined(TL_SM120_FULLTILE_SCALE_SELECTOR0_RETAINED_PACKAGE)
  uint32_t const scale_m1 = scale_m0 + 32u;
  uint32_t const scale_n1 = scale_n0 + 32u;
  uint32_t const retained_sa0 =
      sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m0, scale_k)];
  uint32_t const retained_sa1 =
      sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m1, scale_k)];
  uint32_t const retained_sb0 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n0, scale_k)];
  uint32_t const retained_sb1 =
      sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n1, scale_k)];
  pkg.sa0 = detail::sm120_broadcast_u8_from_u32(retained_sa0, 0);
  pkg.sa1 = detail::sm120_broadcast_u8_from_u32(retained_sa0, 1);
  pkg.sa2 = detail::sm120_broadcast_u8_from_u32(retained_sa1, 0);
  pkg.sa3 = detail::sm120_broadcast_u8_from_u32(retained_sa1, 1);
  pkg.sb0 = detail::sm120_broadcast_u8_from_u32(retained_sb0, 0);
  pkg.sb1 = detail::sm120_broadcast_u8_from_u32(retained_sb0, 1);
  pkg.sb2 = detail::sm120_broadcast_u8_from_u32(retained_sb0, 2);
  pkg.sb3 = detail::sm120_broadcast_u8_from_u32(retained_sb0, 3);
  pkg.sb4 = detail::sm120_broadcast_u8_from_u32(retained_sb1, 0);
  pkg.sb5 = detail::sm120_broadcast_u8_from_u32(retained_sb1, 1);
  pkg.sb6 = detail::sm120_broadcast_u8_from_u32(retained_sb1, 2);
  pkg.sb7 = detail::sm120_broadcast_u8_from_u32(retained_sb1, 3);
#elif defined(TL_SM120_FULLTILE_SCALE_SLOT_PACKAGE)
  uint32_t const sfa_slot_base =
      detail::sm120_cute_sfa_slot_word(lane, warp_m, scale_k, 0);
  detail::sm120_ldscale_v4_u32(sfa_base + sfa_slot_base, pkg.sa0, pkg.sa1,
                               pkg.sa2, pkg.sa3);

  uint32_t sfb_even0, sfb_even1, sfb_even2, sfb_even3;
  uint32_t sfb_odd0, sfb_odd1, sfb_odd2, sfb_odd3;
  uint32_t const sfb_slot_even =
      detail::sm120_cute_sfb_slot_word(lane, warp_n, scale_k, 0, 0);
  uint32_t const sfb_slot_odd =
      detail::sm120_cute_sfb_slot_word(lane, warp_n, scale_k, 0, 1);
  detail::sm120_ldscale_v4_u32(sfb_base + sfb_slot_even, sfb_even0, sfb_even1,
                               sfb_even2, sfb_even3);
  detail::sm120_ldscale_v4_u32(sfb_base + sfb_slot_odd, sfb_odd0, sfb_odd1,
                               sfb_odd2, sfb_odd3);
  pkg.sb0 = sfb_even0;
  pkg.sb1 = sfb_odd0;
  pkg.sb2 = sfb_even1;
  pkg.sb3 = sfb_odd1;
  pkg.sb4 = sfb_even2;
  pkg.sb5 = sfb_odd2;
  pkg.sb6 = sfb_even3;
  pkg.sb7 = sfb_odd3;
#elif defined(TL_SM120_FULLTILE_SCALE_VECTOR_PACKAGE)
  uint32_t sfa0, sfa1, sfa2, sfa3;
  uint32_t sfb0, sfb1, sfb2, sfb3;
  detail::sm120_ldscale_v4_u32(
      sfa_base + detail::sm120_cutlass_128x4_sf_word(scale_m0 & 31u, scale_k),
      sfa0, sfa1, sfa2, sfa3);
  detail::sm120_ldscale_v4_u32(
      sfb_base + detail::sm120_cutlass_128x4_sf_word(scale_n0 & 31u, scale_k),
      sfb0, sfb1, sfb2, sfb3);
  uint32_t const scale_m_group = scale_m0 >> 5;
  uint32_t const scale_n_group = scale_n0 >> 5;
  pkg.sa0 = detail::sm120_select4_u32(sfa0, sfa1, sfa2, sfa3, scale_m_group);
  pkg.sa1 =
      detail::sm120_select4_u32(sfa0, sfa1, sfa2, sfa3, scale_m_group + 1u);
  pkg.sb0 = detail::sm120_select4_u32(sfb0, sfb1, sfb2, sfb3, scale_n_group);
  pkg.sb1 =
      detail::sm120_select4_u32(sfb0, sfb1, sfb2, sfb3, scale_n_group + 1u);
#else
  uint32_t const scale_m1 = scale_m0 + 32u;
  uint32_t const scale_n1 = scale_n0 + 32u;
  pkg.sa0 = sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m0, scale_k)];
  pkg.sa1 = sfa_base[detail::sm120_cutlass_128x4_sf_word(scale_m1, scale_k)];
  pkg.sb0 = sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n0, scale_k)];
  pkg.sb1 = sfb_base[detail::sm120_cutlass_128x4_sf_word(scale_n1, scale_k)];
#endif

  detail::sm120_ldmatrix_x4_blockscaled_operand(
      a_base + detail::sm120_fulltile_package_a_offset(tx, k_block_idx, 0),
      pkg.a00, pkg.a01, pkg.a02, pkg.a03);
  detail::sm120_ldmatrix_x4_blockscaled_operand(
      a_base + detail::sm120_fulltile_package_a_offset(tx, k_block_idx, 1),
      pkg.a10, pkg.a11, pkg.a12, pkg.a13);
  detail::sm120_ldmatrix_x4_blockscaled_operand(
      a_base + detail::sm120_fulltile_package_a_offset(tx, k_block_idx, 2),
      pkg.a20, pkg.a21, pkg.a22, pkg.a23);
  detail::sm120_ldmatrix_x4_blockscaled_operand(
      a_base + detail::sm120_fulltile_package_a_offset(tx, k_block_idx, 3),
      pkg.a30, pkg.a31, pkg.a32, pkg.a33);

  detail::sm120_ldmatrix_x4_blockscaled_operand(
      b_base + detail::sm120_fulltile_package_b_offset(tx, k_block_idx, 0),
      pkg.b00, pkg.b01, pkg.b02, pkg.b03);
  detail::sm120_ldmatrix_x4_blockscaled_operand(
      b_base + detail::sm120_fulltile_package_b_offset(tx, k_block_idx, 1),
      pkg.b10, pkg.b11, pkg.b12, pkg.b13);
  detail::sm120_ldmatrix_x4_blockscaled_operand(
      b_base + detail::sm120_fulltile_package_b_offset(tx, k_block_idx, 2),
      pkg.b20, pkg.b21, pkg.b22, pkg.b23);
  detail::sm120_ldmatrix_x4_blockscaled_operand(
      b_base + detail::sm120_fulltile_package_b_offset(tx, k_block_idx, 3),
      pkg.b30, pkg.b31, pkg.b32, pkg.b33);
}

TL_DEVICE void sm120_gemm_fulltile_ab_owner_wide_package(
    float *c, const SM120FulltileABOwnerWidePackage &pkg) {
#if defined(TL_SM120_FULLTILE_CUTE_ACCUM_LAYOUT) ||                            \
    defined(TL_SM120_FULLTILE_CUTE_ACCUM_DIRECT)
#define TL_SM120_PKG_C_OFFSET(I, J, HALF) ((I) * 4 + (J) * 32 + (HALF) * 16)
#else
#define TL_SM120_PKG_C_OFFSET(I, J, HALF) ((I) * 32 + (J) * 8 + (HALF) * 4)
#endif

#define TL_SM120_PKG_MMA_N8(I, J, HALF, A0, A1, A2, A3, B0, B1, SA, SB,        \
                            SA_TID, SB_TID)                                    \
  do {                                                                         \
    float *d = c + TL_SM120_PKG_C_OFFSET(I, J, HALF);                          \
    detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs(                         \
        d, A0, A1, A2, A3, B0, B1, d, SA, SB, 0, uint16_t(SA_TID), 0,          \
        uint16_t(SB_TID));                                                     \
  } while (0)

#if defined(TL_SM120_FULLTILE_SCALE_SLOT_PACKAGE) ||                           \
    defined(TL_SM120_FULLTILE_SCALE_SELECTOR0_RETAINED_PACKAGE)
#define TL_SM120_PKG_MMA_ROW_SLOT(I, A0, A1, A2, A3, SA)                       \
  do {                                                                         \
    TL_SM120_PKG_MMA_N8(I, 0, 0, A0, A1, A2, A3, pkg.b00, pkg.b01, SA,         \
                        pkg.sb0, 0, 0);                                        \
    TL_SM120_PKG_MMA_N8(I, 0, 1, A0, A1, A2, A3, pkg.b02, pkg.b03, SA,         \
                        pkg.sb1, 0, 0);                                        \
    TL_SM120_PKG_MMA_N8(I, 1, 0, A0, A1, A2, A3, pkg.b10, pkg.b11, SA,         \
                        pkg.sb2, 0, 0);                                        \
    TL_SM120_PKG_MMA_N8(I, 1, 1, A0, A1, A2, A3, pkg.b12, pkg.b13, SA,         \
                        pkg.sb3, 0, 0);                                        \
    TL_SM120_PKG_MMA_N8(I, 2, 0, A0, A1, A2, A3, pkg.b20, pkg.b21, SA,         \
                        pkg.sb4, 0, 0);                                        \
    TL_SM120_PKG_MMA_N8(I, 2, 1, A0, A1, A2, A3, pkg.b22, pkg.b23, SA,         \
                        pkg.sb5, 0, 0);                                        \
    TL_SM120_PKG_MMA_N8(I, 3, 0, A0, A1, A2, A3, pkg.b30, pkg.b31, SA,         \
                        pkg.sb6, 0, 0);                                        \
    TL_SM120_PKG_MMA_N8(I, 3, 1, A0, A1, A2, A3, pkg.b32, pkg.b33, SA,         \
                        pkg.sb7, 0, 0);                                        \
  } while (0)

  TL_SM120_PKG_MMA_ROW_SLOT(0, pkg.a00, pkg.a01, pkg.a02, pkg.a03, pkg.sa0);
  TL_SM120_PKG_MMA_ROW_SLOT(1, pkg.a10, pkg.a11, pkg.a12, pkg.a13, pkg.sa1);
  TL_SM120_PKG_MMA_ROW_SLOT(2, pkg.a20, pkg.a21, pkg.a22, pkg.a23, pkg.sa2);
  TL_SM120_PKG_MMA_ROW_SLOT(3, pkg.a30, pkg.a31, pkg.a32, pkg.a33, pkg.sa3);

#undef TL_SM120_PKG_MMA_ROW_SLOT
#else
#define TL_SM120_PKG_MMA_ROW(I, A0, A1, A2, A3, SA, SA_TID)                    \
  do {                                                                         \
    TL_SM120_PKG_MMA_N8(I, 0, 0, A0, A1, A2, A3, pkg.b00, pkg.b01, SA,         \
                        pkg.sb0, SA_TID, 0);                                   \
    TL_SM120_PKG_MMA_N8(I, 0, 1, A0, A1, A2, A3, pkg.b02, pkg.b03, SA,         \
                        pkg.sb0, SA_TID, 1);                                   \
    TL_SM120_PKG_MMA_N8(I, 1, 0, A0, A1, A2, A3, pkg.b10, pkg.b11, SA,         \
                        pkg.sb0, SA_TID, 2);                                   \
    TL_SM120_PKG_MMA_N8(I, 1, 1, A0, A1, A2, A3, pkg.b12, pkg.b13, SA,         \
                        pkg.sb0, SA_TID, 3);                                   \
    TL_SM120_PKG_MMA_N8(I, 2, 0, A0, A1, A2, A3, pkg.b20, pkg.b21, SA,         \
                        pkg.sb1, SA_TID, 0);                                   \
    TL_SM120_PKG_MMA_N8(I, 2, 1, A0, A1, A2, A3, pkg.b22, pkg.b23, SA,         \
                        pkg.sb1, SA_TID, 1);                                   \
    TL_SM120_PKG_MMA_N8(I, 3, 0, A0, A1, A2, A3, pkg.b30, pkg.b31, SA,         \
                        pkg.sb1, SA_TID, 2);                                   \
    TL_SM120_PKG_MMA_N8(I, 3, 1, A0, A1, A2, A3, pkg.b32, pkg.b33, SA,         \
                        pkg.sb1, SA_TID, 3);                                   \
  } while (0)

  TL_SM120_PKG_MMA_ROW(0, pkg.a00, pkg.a01, pkg.a02, pkg.a03, pkg.sa0, 0);
  TL_SM120_PKG_MMA_ROW(1, pkg.a10, pkg.a11, pkg.a12, pkg.a13, pkg.sa0, 1);
  TL_SM120_PKG_MMA_ROW(2, pkg.a20, pkg.a21, pkg.a22, pkg.a23, pkg.sa1, 0);
  TL_SM120_PKG_MMA_ROW(3, pkg.a30, pkg.a31, pkg.a32, pkg.a33, pkg.sa1, 1);

#undef TL_SM120_PKG_MMA_ROW
#endif
#undef TL_SM120_PKG_MMA_N8
#undef TL_SM120_PKG_C_OFFSET
}

TL_DEVICE void sm120_mma_blockscaled_kblock_fulltile_package_pingpong(
    float *c, const void *a_smem_base, const void *b_smem_base,
    const uint32_t *sfa_smem_base, const uint32_t *sfb_smem_base) {
  const char *a_base = static_cast<const char *>(a_smem_base);
  const char *b_base = static_cast<const char *>(b_smem_base);
  SM120FulltileABOwnerWidePackage pkg0;
  SM120FulltileABOwnerWidePackage pkg1;

  sm120_copy_fulltile_ab_owner_wide_package(pkg0, a_base, b_base, sfa_smem_base,
                                            sfb_smem_base, 0);
  sm120_copy_fulltile_ab_owner_wide_package(pkg1, a_base, b_base, sfa_smem_base,
                                            sfb_smem_base, 1);
  sm120_gemm_fulltile_ab_owner_wide_package(c, pkg0);
  sm120_copy_fulltile_ab_owner_wide_package(pkg0, a_base, b_base, sfa_smem_base,
                                            sfb_smem_base, 2);
  sm120_gemm_fulltile_ab_owner_wide_package(c, pkg1);
  sm120_copy_fulltile_ab_owner_wide_package(pkg1, a_base, b_base, sfa_smem_base,
                                            sfb_smem_base, 3);
  sm120_gemm_fulltile_ab_owner_wide_package(c, pkg0);
  sm120_gemm_fulltile_ab_owner_wide_package(c, pkg1);
}

} // namespace tl
