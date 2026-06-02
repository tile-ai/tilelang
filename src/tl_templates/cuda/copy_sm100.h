#pragma once

#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include "barrier.h"
#include "cluster.h"
#include "common.h"
#include "cuda_fp8.h"
#include "tcgen_05.h"
#include "tcgen_05_ld.h"
#include "tcgen_05_st.h"
#include "instruction/tcgen05mma.h"

namespace tl {

// 256-bit load specialization for ulonglong4
__device__ __forceinline__ void global_load_256(ulonglong4 &D, void const *ptr,
                                                bool pred_guard) {
#if (__CUDACC_VER_MAJOR__ > 12) ||                                             \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  setp.ne.b32 p, %5, 0;\n"
               "  mov.b64 %0, %6;\n"
               "  mov.b64 %1, %7;\n"
               "  mov.b64 %2, %8;\n"
               "  mov.b64 %3, %9;\n"
#if TL_ENABLE_L2_PREFETCH
               "  @p ld.global.L2::128B.v4.u64 {%0, %1, %2, %3}, [%4];\n"
#else
               "  @p ld.global.v4.u64 {%0, %1, %2, %3}, [%4];\n"
#endif
               "}\n"
               : "=l"(D.x), "=l"(D.y), "=l"(D.z), "=l"(D.w)
               : "l"(ptr), "r"((int)pred_guard), "l"(D.x), "l"(D.y), "l"(D.z),
                 "l"(D.w));
#else
  // CUDA < 12.9 fallback: two 128-bit loads (may have performance regression)
  uint4 *data = reinterpret_cast<uint4 *>(&D);
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  setp.ne.b32 p, %9, 0;\n"
               "  mov.b32 %0, %10;\n"
               "  mov.b32 %1, %11;\n"
               "  mov.b32 %2, %12;\n"
               "  mov.b32 %3, %13;\n"
               "  mov.b32 %4, %14;\n"
               "  mov.b32 %5, %15;\n"
               "  mov.b32 %6, %16;\n"
               "  mov.b32 %7, %17;\n"
#if TL_ENABLE_L2_PREFETCH
               "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%8];\n"
               "  @p ld.global.L2::128B.v4.u32 {%4, %5, %6, %7}, [%18];\n"
#else
               "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%8];\n"
               "  @p ld.global.v4.u32 {%4, %5, %6, %7}, [%18];\n"
#endif
               "}\n"
               : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z),
                 "=r"(data[0].w), "=r"(data[1].x), "=r"(data[1].y),
                 "=r"(data[1].z), "=r"(data[1].w)
               : "l"(ptr), "r"((int)pred_guard), "r"(data[0].x), "r"(data[0].y),
                 "r"(data[0].z), "r"(data[0].w), "r"(data[1].x), "r"(data[1].y),
                 "r"(data[1].z), "r"(data[1].w), "l"(((uint8_t *)ptr) + 16));
#endif
}

// Convenience wrapper functions
__device__ __forceinline__ longlong4 load_global_256(const longlong4 *ptr) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, true);
  return *reinterpret_cast<longlong4 *>(&ret);
}

__device__ __forceinline__ ulonglong4 load_global_256(const ulonglong4 *ptr) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, true);
  return ret;
}

// Predicated (conditional) versions
__device__ __forceinline__ longlong4
load_global_256_conditional(const longlong4 *ptr, bool pred) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, pred);
  return *reinterpret_cast<longlong4 *>(&ret);
}

__device__ __forceinline__ ulonglong4
load_global_256_conditional(const ulonglong4 *ptr, bool pred) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, pred);
  return ret;
}

// Generic 256-bit load for FP8 and other types (returns ulonglong4)
template <typename T>
__device__ __forceinline__ ulonglong4 load_global_256(const T *ptr) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, true);
  return ret;
}

template <typename T>
__device__ __forceinline__ ulonglong4 load_global_256_conditional(const T *ptr,
                                                                  bool pred) {
  ulonglong4 ret{};
  global_load_256(ret, ptr, pred);
  return ret;
}

// 256-bit store specialization for ulonglong4
__device__ __forceinline__ void global_store_256(ulonglong4 const &D, void *ptr,
                                                 bool pred_guard) {
#if (__CUDACC_VER_MAJOR__ > 12) ||                                             \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  setp.ne.b32 p, %5, 0;\n"
               "  @p st.global.v4.u64 [%0], {%1, %2, %3, %4};\n"
               "}\n"
               :
               : "l"(ptr), "l"(D.x), "l"(D.y), "l"(D.z), "l"(D.w),
                 "r"((int)pred_guard));
#else
  // CUDA < 12.9 fallback: two 128-bit stores (may have performance
  // regression)
  uint4 const *data = reinterpret_cast<uint4 const *>(&D);
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  setp.ne.b32 p, %5, 0;\n"
               "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
               "  @p st.global.v4.u32 [%6], {%7, %8, %9, %10};\n"
               "}\n"
               :
               : "l"(ptr), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z),
                 "r"(data[0].w), "r"((int)pred_guard),
                 "l"(((uint8_t *)ptr) + 16), "r"(data[1].x), "r"(data[1].y),
                 "r"(data[1].z), "r"(data[1].w));
#endif
}

// Convenience wrapper functions for 256-bit store
template <typename T>
__device__ __forceinline__ void store_global_256(void *ptr, const T &val) {
  ulonglong4 const &val_u64 = *reinterpret_cast<ulonglong4 const *>(&val);
  global_store_256(val_u64, ptr, true);
}

template <typename T>
__device__ __forceinline__ void
store_global_256_conditional(void *ptr, const T &val, bool pred) {
  ulonglong4 const &val_u64 = *reinterpret_cast<ulonglong4 const *>(&val);
  global_store_256(val_u64, ptr, pred);
}

__device__ __forceinline__ unsigned long long
pack_bfloat16x4(const bfloat16_t x, const bfloat16_t y, const bfloat16_t z,
                const bfloat16_t w) {
  unsigned long long v0 = *((unsigned short *)&x);
  unsigned long long v1 = *((unsigned short *)&y);
  unsigned long long v2 = *((unsigned short *)&z);
  unsigned long long v3 = *((unsigned short *)&w);
  return (v0 | (v1 << 16) | (v2 << 32) | (v3 << 48));
}

__device__ __forceinline__ unsigned long long
pack_float16x4(const half x, const half y, const half z, const half w) {
  unsigned long long v0 = *((unsigned short *)&x);
  unsigned long long v1 = *((unsigned short *)&y);
  unsigned long long v2 = *((unsigned short *)&z);
  unsigned long long v3 = *((unsigned short *)&w);
  return (v0 | (v1 << 16) | (v2 << 32) | (v3 << 48));
}

// Helper function to find the largest K that 2**K <= N
// Requires N > 0
template <int N, int K = 0>
__device__ __forceinline__ constexpr int get_floor_log2() {
  static_assert(N > 0);
  if constexpr ((1 << (K + 1)) > N)
    return K;
  else
    return get_floor_log2<N, K + 1>();
}

template <typename target_call_cls, int MAX_LOGN, int N, typename dst_t>
__device__ __forceinline__ void tcgen05_ld_core(uint32_t const &tmem_start_col,
                                                dst_t *dst_ptr) {
  static_assert(N > 0);
  constexpr int LOG_N = get_floor_log2<N>();
  constexpr int CUR_SEGMENT_LEN = 1 << (LOG_N > MAX_LOGN ? MAX_LOGN : LOG_N);
  target_call_cls::copy<CUR_SEGMENT_LEN>(tmem_start_col, (uint32_t *)dst_ptr);
  if constexpr (N - CUR_SEGMENT_LEN > 0) {
    tcgen05_ld_core<target_call_cls, MAX_LOGN, N - CUR_SEGMENT_LEN>(
        tmem_start_col + CUR_SEGMENT_LEN, dst_ptr + CUR_SEGMENT_LEN);
  }
}

template <int N, bool pack16, typename dst_t>
__device__ __forceinline__ void
tcgen05_ld_32dp32bNx(uint32_t const &tmem_start_col,
                     uint32_t const &tmem_col_offset, dst_t *dst_ptr) {
  tcgen05_ld_core<tl::tmem_ld_32dp32bNx<pack16>, 7, N>(
      tmem_start_col + tmem_col_offset, dst_ptr);
  tl::fence_view_async_tmem_load();
}

// x16-max variant: splits loads into x16 instructions for cross-WG visibility.
// Adds explicit per-warp row offset for correct cross-WG TMEM addressing.
template <int N, bool pack16, typename dst_t>
__device__ __forceinline__ void
tcgen05_ld_32dp32bNx_x16(uint32_t const &tmem_start_col,
                         uint32_t const &tmem_col_offset,
                         uint32_t const &tmem_row_offset, dst_t *dst_ptr) {
  tcgen05_ld_core<tl::tmem_ld_32dp32bNx<pack16>, 4, N>(
      tmem_start_col + tmem_col_offset + (tmem_row_offset << 16), dst_ptr);
  tl::fence_view_async_tmem_load();
}

template <int N, bool pack16, typename dst_t>
__device__ __forceinline__ void
tcgen05_ld_32dp64bNx(uint32_t const &tmem_start_col,
                     uint32_t const &tmem_col_offset, dst_t *dst_ptr) {
  tcgen05_ld_core<tl::tmem_ld_32dp64bNx<pack16>, 7, N>(
      tmem_start_col + tmem_col_offset, dst_ptr);
  tl::fence_view_async_tmem_load();
}

template <int N, bool pack16, typename dst_t>
__device__ __forceinline__ void
tcgen05_ld_32dp128bNx(uint32_t const &tmem_start_col,
                      uint32_t const &tmem_col_offset, dst_t *dst_ptr) {
  tcgen05_ld_core<tl::tmem_ld_32dp128bNx<pack16>, 6, N>(
      tmem_start_col + tmem_col_offset, dst_ptr);
  tl::fence_view_async_tmem_load();
}

template <int N, bool pack16, typename dst_t>
__device__ __forceinline__ void
tcgen05_ld_32dp256bNx(uint32_t const &tmem_start_col,
                      uint32_t const &tmem_col_offset, dst_t *dst_ptr) {
  tcgen05_ld_core<tl::tmem_ld_32dp256bNx<pack16>, 5, N>(
      tmem_start_col + tmem_col_offset, dst_ptr);
  tl::fence_view_async_tmem_load();
}

// NOTE: The column offset increment (CUR_SEGMENT_LEN) assumes each register
// maps to exactly one TMEM column (i.e. unpack::16b is NOT active). If
// unpack::16b were used, each register would expand to 2 columns, requiring
// an increment of 2*CUR_SEGMENT_LEN. Currently the codegen always passes
// unpack16=false for stores (see copy.cc use_pack_unpack_modifier), so this
// is correct. Do not enable unpack for stores without fixing this offset.
template <typename target_call_cls, int MAX_LOGN, int N, typename src_t>
__device__ __forceinline__ void tcgen05_st_core(uint32_t const &tmem_start_col,
                                                src_t const *src_ptr) {
  static_assert(N > 0);
  constexpr int LOG_N = get_floor_log2<N>();
  constexpr int CUR_SEGMENT_LEN = 1 << (LOG_N > MAX_LOGN ? MAX_LOGN : LOG_N);
  target_call_cls::template copy<CUR_SEGMENT_LEN>(tmem_start_col,
                                                  (uint32_t const *)src_ptr);
  if constexpr (N - CUR_SEGMENT_LEN > 0) {
    tcgen05_st_core<target_call_cls, MAX_LOGN, N - CUR_SEGMENT_LEN>(
        tmem_start_col + CUR_SEGMENT_LEN, src_ptr + CUR_SEGMENT_LEN);
  }
}

template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp32bNx(uint32_t const &tmem_start_col,
                     uint32_t const &tmem_col_offset, src_t const *src_ptr) {
  tcgen05_st_core<tl::tmem_st_32dp32bNx<unpack16>, 7, N>(
      tmem_start_col + tmem_col_offset, src_ptr);
  tl::fence_view_async_tmem_store();
}

// x16-max variant: splits stores into x16 instructions for cross-WG visibility.
// Does NOT emit per-store wait_st — caller is responsible for calling
// fence_view_async_tmem_store() once after all batched stores complete.
// Adds explicit per-warp row offset for correct cross-WG TMEM addressing.
template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp32bNx_x16(uint32_t const &tmem_start_col,
                         uint32_t const &tmem_col_offset,
                         uint32_t const &tmem_row_offset,
                         src_t const *src_ptr) {
  tcgen05_st_core<tl::tmem_st_32dp32bNx<unpack16>, 4, N>(
      tmem_start_col + tmem_col_offset + (tmem_row_offset << 16), src_ptr);
}

// ====================================================================
// SM100 scalar math and TMEM helper instructions.
// ====================================================================

__device__ __forceinline__ float tcgen05_exp2f_approx(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float tcgen05_rcp_approx_ftz(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float tcgen05_fmax3(float a, float b, float c) {
  float r;
  asm("max.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b),
      "f"(c));
  return r;
}

__device__ __forceinline__ float tcgen05_fmax2(float a, float b) {
  float r;
  asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ void tcgen05_fma_f32x2(
    float &r0, float &r1, float a0, float a1, float b0, float b1,
    float c0, float c1) {
  float o0, o1;
  asm("{                                                \n"
      ".reg .b64 _a, _b, _c, _r;                       \n"
      "mov.b64 _a, {%2, %3};                           \n"
      "mov.b64 _b, {%4, %5};                           \n"
      "mov.b64 _c, {%6, %7};                           \n"
      "fma.rn.ftz.f32x2 _r, _a, _b, _c;                \n"
      "mov.b64 {%0, %1}, _r;                           \n"
      "}"
      : "=f"(o0), "=f"(o1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1));
  r0 = o0;
  r1 = o1;
}

__device__ __forceinline__ void tcgen05_exp2_poly_2(float &r0, float &r1,
                                                     float in0, float in1) {
  asm("{\n\t"
      ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
      ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
      ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
      "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
      "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
      "mov.b64 l1, {f1, f2};\n\t"
      "mov.f32 f3, 0f4B400000;\n\t"
      "mov.b64 l2, {f3, f3};\n\t"
      "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
      "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
      "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
      "mov.f32 f7, 0f3D9DF09D;\n\t"
      "mov.b64 l6, {f7, f7};\n\t"
      "mov.f32 f6, 0f3E6906A4;\n\t"
      "mov.b64 l5, {f6, f6};\n\t"
      "mov.f32 f5, 0f3F31F519;\n\t"
      "mov.b64 l4, {f5, f5};\n\t"
      "mov.f32 f4, 0f3F800000;\n\t"
      "mov.b64 l3, {f4, f4};\n\t"
      "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
      "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
      "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
      "mov.b64 {r1, r2}, l7;\n\t"
      "mov.b64 {r3, r4}, l10;\n\t"
      "shl.b32 r5, r1, 23;\n\t"
      "add.s32 r7, r5, r3;\n\t"
      "shl.b32 r6, r2, 23;\n\t"
      "add.s32 r8, r6, r4;\n\t"
      "mov.b32 %0, r7;\n\t"
      "mov.b32 %1, r8;\n\t"
      "}\n"
      : "=f"(r0), "=f"(r1)
      : "f"(in0), "f"(in1));
}

__device__ __forceinline__ void
tcgen05_st_32x32b_x4(uint32_t tmem_addr, uint32_t v0, uint32_t v1,
                     uint32_t v2, uint32_t v3) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], "
               "{%1, %2, %3, %4};"
               :
               : "r"(tmem_addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3));
}

__device__ __forceinline__ uint32_t
pack_bf16_pair(float a, float b) {
  bfloat16_t ha(a);
  bfloat16_t hb(b);
  return __pack_nv_bfloat162(ha, hb);
}

template <int N>
__device__ __forceinline__ void
tcgen05_float22bfloat162_xN(__nv_bfloat162 *result, const float *inputs) {
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    result[i] = __float22bfloat162_rn({inputs[i * 2], inputs[i * 2 + 1]});
  }
}

__device__ __forceinline__ void
tcgen05_wait_barrier(void const *smem_mbar_ptr, uint32_t phase);


__device__ __forceinline__ void
tcgen05_mbarrier_arrive_lane0(void const *mbar_ptr) {
  if ((threadIdx.x & 31) == 0) {
    uint32_t p = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<void *>(mbar_ptr)));
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :
                 : "r"(p)
                 : "memory");
  }
}


// ====================================================================
// SM100 TCGEN05 descriptor, MMA, and mbarrier helpers.
// ====================================================================

constexpr uint32_t kTcgen05FastDescSBO = 1024;
constexpr uint32_t kTcgen05FastDescHi =
    ((kTcgen05FastDescSBO >> 4) & 0x3FFF) | (1u << 14) | (2u << 29);

__device__ __forceinline__ uint64_t
tcgen05_mk_fast_desc(uint32_t base_lo, uint32_t byte_off) {
  uint32_t lo = base_lo + (byte_off >> 4);
  return (uint64_t(kTcgen05FastDescHi) << 32) | lo;
}

__device__ __forceinline__ void
tcgen05_commit_2cta(void const *smem_mbar_ptr);

// 1SM SS-MMA: C[128x128] = A[128xK] @ B[128xK]^T.
// Shared/shared 128x128 tcgen05 MMA with raw descriptors.
__device__ __forceinline__ void
tcgen05_mma_1sm_128x128(uint32_t tmem_c, uint64_t desc_a,
                         uint64_t desc_b, uint32_t accumulate) {
  uint32_t idesc = (1U << 4)               // c_format = F32
                 | (1U << 7)               // a_format = BF16
                 | (1U << 10)              // b_format = BF16
                 | ((128U / 8) << 17)      // n_dim = 16
                 | ((128U / 16) << 24);    // m_dim = 8
  tl::tcgen05mma_ss_nomask<DataType::kFloat16, false, false>(
      desc_a, desc_b, tmem_c, accumulate, idesc);
}

// 2CTA SS-MMA: C[256x128] = A[256xK] @ B[128xK]^T.  Each CTA contributes
// 128 M rows and 64 K rows through cta_group::2.
__device__ __forceinline__ void
tcgen05_mma_2cta_256x128(uint32_t tmem_c, uint64_t desc_a,
                         uint64_t desc_b, uint32_t accumulate) {
  uint32_t idesc = (1U << 4)               // c_format = F32
                 | (1U << 7)               // a_format = BF16
                 | (1U << 10)              // b_format = BF16
                 | ((128U / 8) << 17)      // n_dim = 16
                 | ((256U / 16) << 24);    // m_dim = 16
  tl::tcgen05mma_ss_nomask<DataType::kFloat16, true, false>(
      desc_a, desc_b, tmem_c, accumulate, idesc);
}

// Shared/shared 128x128 MMA sequence: two 64-wide descriptor tiles, four
// 16-wide MMA issues per tile, then a plain .b64 tcgen05 commit.
__device__ __forceinline__ void
tcgen05_mma_1sm_ss_128x128_commit(void const *A_smem_ptr,
                                  void const *B_smem_ptr,
                                  uint32_t C_tmem_addr,
                                  void const *smem_mbar_ptr) {
  if (!cute::elect_one_sync()) return;
  constexpr int kBlockMCTA = 128;
  constexpr int kBlockN = 128;
  constexpr int kTileCols = 64;

  uint32_t a_lo = (uint32_t)((__cvta_generic_to_shared(
                                  const_cast<void *>(A_smem_ptr)) &
                              0x3FFFF) >>
                             4);
  uint32_t b_lo = (uint32_t)((__cvta_generic_to_shared(
                                   const_cast<void *>(B_smem_ptr)) &
                               0x3FFFF) >>
                              4);

  int first = 1;
  #pragma unroll
  for (int t = 0; t < 2; t++) {
    uint32_t q_off = t * kBlockMCTA * kTileCols * 2;
    uint32_t k_off = t * kBlockN * kTileCols * 2;
    #pragma unroll
    for (int j = 0; j < kTileCols; j += 16) {
      tl::tcgen05_mma_1sm_128x128(
          C_tmem_addr,
          tl::tcgen05_mk_fast_desc(a_lo, q_off + j * 2),
          tl::tcgen05_mk_fast_desc(b_lo, k_off + j * 2),
          first ? 0u : 1u);
      first = 0;
    }
  }

  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(
      const_cast<void *>(smem_mbar_ptr)));
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
      : : "r"(p));
}

// 128×64 BMN TS-MMA: C[128×64] += A_tmem[128×K] @ B_smem[64×K]^T
// A in TMEM (K-major), B in SMEM (MN-major = transposed), C in TMEM (F32)
__device__ __forceinline__ void
tcgen05_mma_1sm_ts_128x64_bmn(uint32_t tmem_c, uint32_t tmem_a,
                               uint64_t desc_b, uint32_t accumulate) {
  uint32_t idesc = (1U << 4)               // c_format = F32
                 | (1U << 7)               // a_format = BF16
                 | (1U << 10)              // b_format = BF16
                 | (0U << 15)              // a_major = K-major
                 | (1U << 16)              // b_major = MN-major
                 | ((64U / 8) << 17)       // n_dim = 8 (N=64)
                 | ((128U / 16) << 24);    // m_dim = 8 (M=128)
  tl::tcgen05mma_ts_nomask<DataType::kFloat16, false, false>(
      tmem_a, desc_b, tmem_c, accumulate, idesc);
}

// 2CTA TS-MMA: C[256x128] += A_tmem[256xK] @ B_smem[128xK]^T.
__device__ __forceinline__ void
tcgen05_mma_2cta_ts_256x128_bmn(uint32_t tmem_c, uint32_t tmem_a,
                                uint64_t desc_b, uint32_t accumulate) {
  uint32_t idesc = (1U << 4)               // c_format = F32
                 | (1U << 7)               // a_format = BF16
                 | (1U << 10)              // b_format = BF16
                 | (1U << 16)              // b_major = MN-major
                 | ((128U / 8) << 17)      // n_dim = 16
                 | ((256U / 16) << 24);    // m_dim = 16
  tl::tcgen05mma_ts_nomask<DataType::kFloat16, true, false>(
      tmem_a, desc_b, tmem_c, accumulate, idesc);
}

// 1SM tcgen05 commit using the plain .b64 mbarrier address form.
__device__ __forceinline__ void
tcgen05_commit_1sm(void const *smem_mbar_ptr) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(
      const_cast<void *>(smem_mbar_ptr)));
  if (cute::elect_one_sync()) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        : : "r"(p));
	  }
}

__device__ __forceinline__ void
tcgen05_commit_2cta(void const *smem_mbar_ptr) {
  uint32_t p = static_cast<uint32_t>(
      __cvta_generic_to_shared(const_cast<void *>(smem_mbar_ptr)));
  uint16_t mask = 3;
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
      "multicast::cluster.b64 [%0], %1;"
      : : "r"(p), "h"(mask) : "memory");
}

__device__ __forceinline__ void
tcgen05_mbarrier_arrive_cluster_lane0(void const *mbar_ptr) {
  uint32_t p = static_cast<uint32_t>(
      __cvta_generic_to_shared(const_cast<void *>(mbar_ptr))) & 0xFEFFFFFFu;
  asm volatile("{                                                            \n"
               ".reg .pred p_l0; .reg .u32 lane;                            \n"
               "mov.u32 lane, %%laneid;                                      \n"
               "setp.eq.u32 p_l0, lane, 0;                                   \n"
               "@p_l0 mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0]; \n"
               "}" : : "r"(p) : "memory");
}

__device__ __forceinline__ void
tcgen05_mbarrier_arrive_expect_tx_cluster_lane0(void const *mbar_ptr,
                                                uint32_t bytes) {
  uint32_t p = static_cast<uint32_t>(
      __cvta_generic_to_shared(const_cast<void *>(mbar_ptr))) & 0xFEFFFFFFu;
  asm volatile("{                                                                        \n"
               ".reg .pred p_l0; .reg .u32 lane;                                        \n"
               "mov.u32 lane, %%laneid;                                                  \n"
               "setp.eq.u32 p_l0, lane, 0;                                               \n"
               "@p_l0 mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1; \n"
               "}" : : "r"(p), "r"(bytes) : "memory");
}

// Two 128x64 BMN TS-MMA sequences. B_hi is issued before B_lo to match the
// target output-column order used by the caller.
// B_lo_smem_ptr/B_hi_smem_ptr: BF16 [128,64] shared tiles, row stride=128B.
// A_tmem_addr/C_tmem_addr: TMEM base addresses.
// accumulate: 0 for k==0 (clear), 1 for k>0 (accumulate into existing O)
// Uses elect_one_sync internally so only one lane issues the MMA sequence.
__device__ __forceinline__ void
tcgen05_mma_1sm_ts_128x64_bmn_x2(void const *B_lo_smem_ptr,
                                 void const *B_hi_smem_ptr,
                                 uint32_t A_tmem_addr,
                                 uint32_t C_tmem_addr,
                                 uint32_t accumulate) {
  if (!cute::elect_one_sync()) return;
  constexpr int kBlockN = 128;   // K-rows (seq block)
  constexpr int kTileCols = 64;  // descriptor tile width

  auto *b_lo = (bfloat16_t *)const_cast<void *>(B_lo_smem_ptr);
  auto *b_hi = (bfloat16_t *)const_cast<void *>(B_hi_smem_ptr);
  tl::Tcgen05SMemDescriptor desc_lo, desc_hi;
  tl::initialize_tcgen05_descriptor(
      desc_lo, b_lo, 0, 64, 0, 0, 2);
  tl::initialize_tcgen05_descriptor(
      desc_hi, b_hi, 0, 64, 0, 0, 2);

  constexpr int kStride = 2048;  // bytes per 16 K-rows: 16 * 64 * 2

  // D-tile 1 (HIGH, dim 64-127) first.
  tl::fence_proxy_async();
  #pragma unroll
  for (int j = 0; j < kBlockN / 16; j++) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        C_tmem_addr + 64, A_tmem_addr + j * 8,
        uint64_t(desc_hi + j * kStride),
        (j == 0) ? accumulate : 1u);
  }
  // D-tile 0 (LOW, dim 0-63) SECOND
  #pragma unroll
  for (int j = 0; j < kBlockN / 16; j++) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        C_tmem_addr, A_tmem_addr + j * 8,
        uint64_t(desc_lo + j * kStride),
        (j == 0) ? accumulate : 1u);
  }
}

// Variant for a contiguous B layout: low tile at base and high tile at
// base + 128*64 elements.
__device__ __forceinline__ void
tcgen05_mma_1sm_ts_128x64_bmn_x2_contig(void const *B_smem_ptr,
                                        uint32_t A_tmem_addr,
                                        uint32_t C_tmem_addr,
                                        uint32_t accumulate) {
  auto *b_base = (bfloat16_t *)const_cast<void *>(B_smem_ptr);
  tl::tcgen05_mma_1sm_ts_128x64_bmn_x2(
      (void const *)b_base,
      (void const *)(b_base + 128 * 64),
      A_tmem_addr, C_tmem_addr, accumulate);
}

__device__ __forceinline__ void
tcgen05_mma_1sm_ss_128x128_commit_lane0(void const *A_smem_ptr,
                                        void const *B_smem_ptr,
                                        uint32_t C_tmem_addr,
                                        void const *smem_mbar_ptr) {
  constexpr int kBlockMCTA = 128;
  constexpr int kBlockN = 128;
  constexpr int kTileCols = 64;

  uint32_t a_lo = (uint32_t)((__cvta_generic_to_shared(
                                  const_cast<void *>(A_smem_ptr)) &
                              0x3FFFF) >>
                             4);
  uint32_t b_lo = (uint32_t)((__cvta_generic_to_shared(
                                   const_cast<void *>(B_smem_ptr)) &
                               0x3FFFF) >>
                              4);

  int first = 1;
  #pragma unroll
  for (int t = 0; t < 2; t++) {
    uint32_t q_off = t * kBlockMCTA * kTileCols * 2;
    uint32_t k_off = t * kBlockN * kTileCols * 2;
    #pragma unroll
    for (int j = 0; j < kTileCols; j += 16) {
      tl::tcgen05_mma_1sm_128x128(
          C_tmem_addr,
          tl::tcgen05_mk_fast_desc(a_lo, q_off + j * 2),
          tl::tcgen05_mk_fast_desc(b_lo, k_off + j * 2),
          first ? 0u : 1u);
      first = 0;
    }
  }

  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(
      const_cast<void *>(smem_mbar_ptr)));
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
      : : "r"(p));
}

__device__ __forceinline__ void
tcgen05_mma_1sm_ss_128x128_commit_fast(uint32_t sA_lo, uint32_t sB_lo,
                                 uint32_t q_off_base,
                                 uint32_t kv_off_base,
                                 uint32_t C_tmem_addr,
                                 void const *smem_mbar_ptr) {
  constexpr int kBlockMCTA = 128;
  constexpr int kBlockN = 128;
  constexpr int kTileCols = 64;

  int first = 1;
  #pragma unroll
  for (int t = 0; t < 2; t++) {
    uint32_t q_off = q_off_base + t * kBlockMCTA * kTileCols * 2;
    uint32_t k_off = kv_off_base + t * kBlockN * kTileCols * 2;
    #pragma unroll
    for (int j = 0; j < kTileCols; j += 16) {
      tl::tcgen05_mma_1sm_128x128(
          C_tmem_addr,
          tl::tcgen05_mk_fast_desc(sA_lo, q_off + j * 2),
          tl::tcgen05_mk_fast_desc(sB_lo, k_off + j * 2),
          first ? 0u : 1u);
      first = 0;
    }
  }

  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(
      const_cast<void *>(smem_mbar_ptr)));
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
      : : "r"(p));
}

__device__ __forceinline__ void
tcgen05_mma_1sm_ts_128x64_bmn_x2_contig_lane0(void const *B_smem_ptr,
                                              uint32_t A_tmem_addr,
                                              uint32_t C_tmem_addr,
                                              uint32_t accumulate) {
  constexpr int kBlockN = 128;
  auto *b_base = (bfloat16_t *)const_cast<void *>(B_smem_ptr);
  tl::Tcgen05SMemDescriptor desc_lo, desc_hi;
  tl::initialize_tcgen05_descriptor(desc_lo, b_base, 0, 64, 0, 0, 2);
  tl::initialize_tcgen05_descriptor(desc_hi, b_base + 128 * 64, 0, 64, 0, 0, 2);

  constexpr int kStride = 2048;
  #pragma unroll
  for (int j = 0; j < kBlockN / 16; j++) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        C_tmem_addr + 64, A_tmem_addr + j * 8,
        uint64_t(desc_hi + j * kStride),
        (j == 0) ? accumulate : 1u);
  }
  #pragma unroll
  for (int j = 0; j < kBlockN / 16; j++) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        C_tmem_addr, A_tmem_addr + j * 8,
        uint64_t(desc_lo + j * kStride),
        (j == 0) ? accumulate : 1u);
  }
}

__device__ __forceinline__ void
tcgen05_commit_1sm_lane0(void const *smem_mbar_ptr) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(
      const_cast<void *>(smem_mbar_ptr)));
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
      : : "r"(p));
}

__device__ __forceinline__ void
tcgen05_mma_1sm_ts_128x64_bmn_x2_contig_fast(uint32_t sB_lo, uint32_t b_off_base,
                                uint32_t A_tmem_addr,
                                uint32_t C_tmem_addr,
                                uint32_t accumulate) {
  constexpr int kBlockN = 128;
  constexpr int kTileCols = 64;
  uint32_t b_hi = b_off_base + kBlockN * kTileCols * 2;

  #pragma unroll
  for (int j = 0; j < kBlockN; j += 16) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        C_tmem_addr + 64, A_tmem_addr + j / 2,
        tl::tcgen05_mk_fast_desc(sB_lo, b_hi + j * kTileCols * 2),
        (j == 0) ? accumulate : 1u);
  }
  #pragma unroll
  for (int j = 0; j < kBlockN; j += 16) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        C_tmem_addr, A_tmem_addr + j / 2,
        tl::tcgen05_mk_fast_desc(sB_lo, b_off_base + j * kTileCols * 2),
        (j == 0) ? accumulate : 1u);
  }
}

__device__ __forceinline__ void
tcgen05_wait_barrier(void const *smem_mbar_ptr, uint32_t phase) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(
      const_cast<void *>(smem_mbar_ptr)));
  asm volatile(
      "{ .reg .pred P; WAIT%=:"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, %2;"
      "@P bra.uni DONE%=; bra.uni WAIT%=; DONE%=: }"
      : : "r"(p), "r"(phase), "r"(10000000u) : "memory");
}

__device__ __forceinline__ void
tcgen05_arrive_expect_tx(void const *smem_mbar_ptr, uint32_t tx_bytes) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(
      const_cast<void *>(smem_mbar_ptr)));
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
      : : "r"(p), "r"(tx_bytes) : "memory");
}

// Generic three-way selectors for hand-scheduled staged pipelines.
__device__ __forceinline__ bfloat16_t *
select_stage_ptr(void *ptr0, void *ptr1, void *ptr2, int stage) {
  if (stage == 0) return reinterpret_cast<bfloat16_t *>(ptr0);
  if (stage == 1) return reinterpret_cast<bfloat16_t *>(ptr1);
  return reinterpret_cast<bfloat16_t *>(ptr2);
}

__device__ __forceinline__ Barrier &
select_barrier_ref(Barrier &mbar0, Barrier &mbar1, Barrier &mbar2,
                   int stage) {
  if (stage == 0) return mbar0;
  if (stage == 1) return mbar1;
  return mbar2;
}

__device__ __forceinline__ bfloat16_t *
tcgen05_smem_ptr_add_bf16(void *ptr, int offset) {
  return reinterpret_cast<bfloat16_t *>(ptr) + offset;
}

// ====================================================================
// SM100 2CTA cluster TMA helpers.
// ====================================================================

enum class CacheHintSm100 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};

constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFF;

template <CacheHintSm100 cache_hint = CacheHintSm100::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0);

template <CacheHintSm100 cache_hint = CacheHintSm100::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1);

template <CacheHintSm100 cache_hint = CacheHintSm100::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1,
                            int32_t const &crd2);

template <CacheHintSm100 cache_hint = CacheHintSm100::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1,
                            int32_t const &crd2, int32_t const &crd3);

template <CacheHintSm100 cache_hint = CacheHintSm100::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1,
                            int32_t const &crd2, int32_t const &crd3,
                            int32_t const &crd4);

__device__ __forceinline__ void
tma_load_2sm_raw(const CUtensorMap *descriptor, void const *const smem_ptr,
                 void const *mbar_ptr, int32_t const &crd0,
                 int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(descriptor);
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void *>(smem_ptr)));
  uint32_t smem_int_mbar =
      static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void *>(mbar_ptr))) &
      Sm100MmaPeerBitMask;
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global."
               "mbarrier::complete_tx::bytes.cta_group::2 "
               "[%0], [%1, {%3, %4}], [%2];"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1)
               : "memory");
}

__device__ __forceinline__ void
tma_store_2d_raw(const CUtensorMap *descriptor, void const *const smem_ptr,
                 int32_t const &crd0, int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(descriptor);
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void *>(smem_ptr)));
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
               "[%0, {%1, %2}], [%3];"
               :
               : "l"(gmem_int_desc), "r"(crd0), "r"(crd1),
                 "r"(smem_int_ptr)
               : "memory");
}

// ====================================================================
// SM100 2CTA cluster synchronization helpers used by UMA-style DSL kernels.
// ====================================================================

__device__ __forceinline__ void
tcgen05_mbarrier_arrive_cluster_all(void const *mbar_ptr) {
  uint32_t p = static_cast<uint32_t>(
      __cvta_generic_to_shared(const_cast<void *>(mbar_ptr))) & 0xFEFFFFFFu;
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
               : : "r"(p) : "memory");
}

__device__ __forceinline__ void
tcgen05_mbarrier_arrive_local_all(void const *mbar_ptr) {
  uint32_t p = static_cast<uint32_t>(
      __cvta_generic_to_shared(const_cast<void *>(mbar_ptr)));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
               : : "r"(p) : "memory");
}

__device__ __forceinline__ void tcgen05_bar_arrive(int bar, int count) {
  asm volatile("bar.arrive %0, %1;" : : "r"(bar), "r"(count));
}

__device__ __forceinline__ void tcgen05_bar_sync(int bar, int count) {
  asm volatile("bar.sync %0, %1;" : : "r"(bar), "r"(count));
}

template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp64bNx(uint32_t const &tmem_start_col,
                     uint32_t const &tmem_col_offset, src_t const *src_ptr) {
  tcgen05_st_core<tl::tmem_st_32dp64bNx<unpack16>, 7, N>(
      tmem_start_col + tmem_col_offset, src_ptr);
  tl::fence_view_async_tmem_store();
}

template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp128bNx(uint32_t const &tmem_start_col,
                      uint32_t const &tmem_col_offset, src_t const *src_ptr) {
  tcgen05_st_core<tl::tmem_st_32dp128bNx<unpack16>, 6, N>(
      tmem_start_col + tmem_col_offset, src_ptr);
  tl::fence_view_async_tmem_store();
}

template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp256bNx(uint32_t const &tmem_start_col,
                      uint32_t const &tmem_col_offset, src_t const *src_ptr) {
  tcgen05_st_core<tl::tmem_st_32dp256bNx<unpack16>, 5, N>(
      tmem_start_col + tmem_col_offset, src_ptr);
  tl::fence_view_async_tmem_store();
}

/* SM100 TMA 2SM load (cta_group::2) */

template <CacheHintSm100 cache_hint, typename BarrierType>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  // Executed by both CTAs. Set peer bit to 0 so that the
  // transaction bytes will update CTA0's barrier.
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  smem_int_mbar &= Sm100MmaPeerBitMask;
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3}], [%2], %4;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm100 cache_hint, typename BarrierType>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  // Executed by both CTAs. Set peer bit to 0 so that the
  // transaction bytes will update CTA0's barrier.
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  smem_int_mbar &= Sm100MmaPeerBitMask;
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4}], [%2], %5;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm100 cache_hint, typename BarrierType>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1,
                            int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  // Executed by both CTAs. Set peer bit to 0 so that the
  // transaction bytes will update CTA0's barrier.
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  smem_int_mbar &= Sm100MmaPeerBitMask;
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm100 cache_hint, typename BarrierType>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1,
                            int32_t const &crd2, int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  // Executed by both CTAs. Set peer bit to 0 so that the
  // transaction bytes will update CTA0's barrier.
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  smem_int_mbar &= Sm100MmaPeerBitMask;
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm100 cache_hint, typename BarrierType>
TL_DEVICE void tma_load_2sm(const CUtensorMap &descriptor,
                            BarrierType &smem_mbar, void const *const smem_ptr,
                            int32_t const &crd0, int32_t const &crd1,
                            int32_t const &crd2, int32_t const &crd3,
                            int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  // Executed by both CTAs. Set peer bit to 0 so that the
  // transaction bytes will update CTA0's barrier.
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  smem_int_mbar &= Sm100MmaPeerBitMask;
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4),
                 "l"(cache_hint)
               : "memory");
}

} // namespace tl
