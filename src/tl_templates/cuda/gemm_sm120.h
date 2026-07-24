#pragma once

#include "common.h"
#include "ldsm.h"

#ifndef __CUDACC_RTC__
#include <cstdint>
#endif

namespace tl {

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

// CUTLASS BlockScaledBasicChunk K-major scale layout, compressed to uint32
// words. The scale-byte offset for one 128x64 atom is:
//
//   (idx % 32) * 16 + (idx / 32) * 4 + k16
//
// where k16 is the scale byte inside the four K/16 groups. TileLang packs those
// four adjacent scale bytes in one uint32, so this helper returns the flattened
// uint32 word offset:
//
//   ki * 128 + (idx % 32) * 4 + (idx / 32)
//
// This is the byte offset divided by four, not the byte-offset formula with
// only the final +k16 term removed.
TL_DEVICE uint32_t sm120_blockscaled_chunk_kmajor_sf_word(uint32_t idx,
                                                          uint32_t ki) {
  return ki * 128u + (idx & 31u) * 4u + (idx >> 5);
}

// First-class scale package for the SM120 blockscaled TV copy-view work. The
// current implementation intentionally preserves the compact-selector semantic
// rows used by production. The important contract boundary is that scale
// loading is now isolated from the A/B operand package, so future
// get_layoutSFA_TV / get_layoutSFB_TV lowering can change the producer-lane
// copy view without rewriting the OMMA.SF issue loop.
struct SM120ScaleTVPackage {
  uint32_t sa0, sa1;
  uint32_t sb0, sb1;
};

template <class ScalePkg>
TL_DEVICE void sm120_load_scale_tv_package(ScalePkg &pkg,
                                           const uint32_t *sfa_base,
                                           const uint32_t *sfb_base,
                                           uint32_t lane, uint32_t warp_m,
                                           uint32_t warp_n, uint32_t scale_k) {
  uint32_t const qlane = lane & 3u;
  uint32_t const sfa_row = 8u * (lane & 1u) + (lane >> 2);
  uint32_t const sfb_col = lane >> 2;
  uint32_t const a_owner_in_pair = qlane >> 1;
  uint32_t const scale_m0 = warp_m * 64u + a_owner_in_pair * 16u + sfa_row;
  uint32_t const scale_n0 = warp_n * 64u + qlane * 8u + sfb_col;
  pkg.sa0 = sfa_base[sm120_blockscaled_chunk_kmajor_sf_word(scale_m0, scale_k)];
  pkg.sa1 =
      sfa_base[sm120_blockscaled_chunk_kmajor_sf_word(scale_m0 + 32u, scale_k)];
  pkg.sb0 = sfb_base[sm120_blockscaled_chunk_kmajor_sf_word(scale_n0, scale_k)];
  pkg.sb1 =
      sfb_base[sm120_blockscaled_chunk_kmajor_sf_word(scale_n0 + 32u, scale_k)];
}

TL_DEVICE void sm120_copy_scale_tv_package(SM120ScaleTVPackage &pkg,
                                           const uint32_t *sfa_base,
                                           const uint32_t *sfb_base,
                                           int k_block_idx) {
  uint32_t const tx = uint32_t(int(threadIdx.x) & 127);
  uint32_t const lane = tx & 31u;
  uint32_t const warp = tx >> 5;
  uint32_t const warp_m = warp & 1u;
  uint32_t const warp_n = warp >> 1;
  sm120_load_scale_tv_package(pkg, sfa_base, sfb_base, lane, warp_m, warp_n,
                              uint32_t(k_block_idx));
}

// SM120 blockscaled fulltile kernels precompute some shared-memory addresses
// and keep operand fragments in named scalar registers. The generic
// ptx_ldmatrix_x4(void*, void*) helper takes a C++ pointer and a contiguous
// local buffer, so keep this address-register/scalar-register form local to the
// SM120 operand-package helpers.
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

// Dense SM120 NVFP4/mxf4nvf4 uses packed U4 shared storage, so operands are
// fetched with the ordinary x4.m8n8.shared.b16 ldmatrix path above (no
// b4x16_p64 unpacked-FP4 variant and no post-load nibble shift).
TL_DEVICE void sm120_ldmatrix_x4_blockscaled_operand(void const *const smem_ptr,
                                                     uint32_t &d0, uint32_t &d1,
                                                     uint32_t &d2,
                                                     uint32_t &d3) {
  sm120_ldmatrix_x4_u32(smem_ptr, d0, d1, d2, d3);
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

TL_DEVICE void sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3(
    float *d, const uint32_t *a, const uint32_t *b, const float *c,
    uint32_t scale_a, uint32_t scale_b, uint16_t scale_a_byte_id = 0,
    uint16_t scale_a_thread_id = 0, uint16_t scale_b_byte_id = 0,
    uint16_t scale_b_thread_id = 0) {
  sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs(
      d, a[0], a[1], a[2], a[3], b[0], b[1], c, scale_a, scale_b,
      scale_a_byte_id, scale_a_thread_id, scale_b_byte_id, scale_b_thread_id);
}

TL_DEVICE uint32_t sm120_fulltile_k_swizzle_offset(uint32_t tx,
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

// A/B operands are staged into shared memory as row-major tiles (the layout
// produced by T.copy for a (block, block_K) packed-FP4 buffer).  Each offset
// below resolves one ldmatrix.x4 source address for the current thread.
TL_DEVICE uint32_t sm120_fulltile_package_a_offset(uint32_t tx, int k_block_idx,
                                                   int row_idx) {
  return (((tx & 63u) >> 5) * 8192u) + ((tx & 15u) * 128u) +
         sm120_fulltile_k_swizzle_offset(tx, k_block_idx) +
         (((((tx & 31u) >> 4) + (tx & 1u)) & 1u) * 16u) +
         uint32_t(row_idx) * 2048u;
}

TL_DEVICE uint32_t sm120_fulltile_package_b_offset(uint32_t tx, int k_block_idx,
                                                   int panel_idx) {
  return ((tx >> 6) * 8192u) + (((tx & 31u) >> 4) * 1024u) +
         ((tx & 7u) * 128u) + sm120_fulltile_k_swizzle_offset(tx, k_block_idx) +
         (((((tx & 15u) >> 3) + (tx & 1u)) & 1u) * 16u) +
         uint32_t(panel_idx) * 2048u;
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

struct SM120FulltileABOwnerWidePackage {
  uint32_t a00, a01, a02, a03;
  uint32_t a10, a11, a12, a13;
  uint32_t a20, a21, a22, a23;
  uint32_t a30, a31, a32, a33;
  uint32_t b00, b01, b02, b03;
  uint32_t b10, b11, b12, b13;
  uint32_t b20, b21, b22, b23;
  uint32_t b30, b31, b32, b33;
};

TL_DEVICE void
sm120_copy_fulltile_ab_owner_wide_package(SM120FulltileABOwnerWidePackage &pkg,
                                          const char *a_base,
                                          const char *b_base, int k_block_idx) {
  uint32_t const tx = uint32_t(int(threadIdx.x) & 127);

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
    float *c, const SM120FulltileABOwnerWidePackage &pkg,
    const detail::SM120ScaleTVPackage &scale_pkg) {
#define TL_SM120_PKG_C_OFFSET(I, J, HALF) ((I) * 32 + (J) * 8 + (HALF) * 4)

#define TL_SM120_PKG_MMA_N8(I, J, HALF, A0, A1, A2, A3, B0, B1, SA, SB,        \
                            SA_TID, SB_TID)                                    \
  do {                                                                         \
    float *d = c + TL_SM120_PKG_C_OFFSET(I, J, HALF);                          \
    detail::sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs(                         \
        d, A0, A1, A2, A3, B0, B1, d, SA, SB, 0, uint16_t(SA_TID), 0,          \
        uint16_t(SB_TID));                                                     \
  } while (0)

#define TL_SM120_PKG_MMA_ROW(I, A0, A1, A2, A3, SA, SA_TID)                    \
  do {                                                                         \
    TL_SM120_PKG_MMA_N8(I, 0, 0, A0, A1, A2, A3, pkg.b00, pkg.b01, SA,         \
                        scale_pkg.sb0, SA_TID, 0);                             \
    TL_SM120_PKG_MMA_N8(I, 0, 1, A0, A1, A2, A3, pkg.b02, pkg.b03, SA,         \
                        scale_pkg.sb0, SA_TID, 1);                             \
    TL_SM120_PKG_MMA_N8(I, 1, 0, A0, A1, A2, A3, pkg.b10, pkg.b11, SA,         \
                        scale_pkg.sb0, SA_TID, 2);                             \
    TL_SM120_PKG_MMA_N8(I, 1, 1, A0, A1, A2, A3, pkg.b12, pkg.b13, SA,         \
                        scale_pkg.sb0, SA_TID, 3);                             \
    TL_SM120_PKG_MMA_N8(I, 2, 0, A0, A1, A2, A3, pkg.b20, pkg.b21, SA,         \
                        scale_pkg.sb1, SA_TID, 0);                             \
    TL_SM120_PKG_MMA_N8(I, 2, 1, A0, A1, A2, A3, pkg.b22, pkg.b23, SA,         \
                        scale_pkg.sb1, SA_TID, 1);                             \
    TL_SM120_PKG_MMA_N8(I, 3, 0, A0, A1, A2, A3, pkg.b30, pkg.b31, SA,         \
                        scale_pkg.sb1, SA_TID, 2);                             \
    TL_SM120_PKG_MMA_N8(I, 3, 1, A0, A1, A2, A3, pkg.b32, pkg.b33, SA,         \
                        scale_pkg.sb1, SA_TID, 3);                             \
  } while (0)

  TL_SM120_PKG_MMA_ROW(0, pkg.a00, pkg.a01, pkg.a02, pkg.a03, scale_pkg.sa0, 0);
  TL_SM120_PKG_MMA_ROW(1, pkg.a10, pkg.a11, pkg.a12, pkg.a13, scale_pkg.sa0, 1);
  TL_SM120_PKG_MMA_ROW(2, pkg.a20, pkg.a21, pkg.a22, pkg.a23, scale_pkg.sa1, 0);
  TL_SM120_PKG_MMA_ROW(3, pkg.a30, pkg.a31, pkg.a32, pkg.a33, scale_pkg.sa1, 1);

#undef TL_SM120_PKG_MMA_ROW
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
  detail::SM120ScaleTVPackage scale_pkg0;
  detail::SM120ScaleTVPackage scale_pkg1;

  sm120_copy_fulltile_ab_owner_wide_package(pkg0, a_base, b_base, 0);
  detail::sm120_copy_scale_tv_package(scale_pkg0, sfa_smem_base, sfb_smem_base,
                                      0);
  sm120_copy_fulltile_ab_owner_wide_package(pkg1, a_base, b_base, 1);
  detail::sm120_copy_scale_tv_package(scale_pkg1, sfa_smem_base, sfb_smem_base,
                                      1);
  sm120_gemm_fulltile_ab_owner_wide_package(c, pkg0, scale_pkg0);
  sm120_copy_fulltile_ab_owner_wide_package(pkg0, a_base, b_base, 2);
  detail::sm120_copy_scale_tv_package(scale_pkg0, sfa_smem_base, sfb_smem_base,
                                      2);
  sm120_gemm_fulltile_ab_owner_wide_package(c, pkg1, scale_pkg1);
  sm120_copy_fulltile_ab_owner_wide_package(pkg1, a_base, b_base, 3);
  detail::sm120_copy_scale_tv_package(scale_pkg1, sfa_smem_base, sfb_smem_base,
                                      3);
  sm120_gemm_fulltile_ab_owner_wide_package(c, pkg0, scale_pkg0);
  sm120_gemm_fulltile_ab_owner_wide_package(c, pkg1, scale_pkg1);
}

} // namespace tl
