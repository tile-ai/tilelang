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
                          uint32_t const &tmem_col_offset, dst_t *dst_ptr) {
  uint32_t row_off = (uint32_t)(((threadIdx.x / 32) % 4) * 32) << 16;
  tcgen05_ld_core<tl::tmem_ld_32dp32bNx<pack16>, 4, N>(
      tmem_start_col + tmem_col_offset + row_off, dst_ptr);
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
                          uint32_t const &tmem_col_offset, src_t const *src_ptr) {
  uint32_t row_off = (uint32_t)(((threadIdx.x / 32) % 4) * 32) << 16;
  tcgen05_st_core<tl::tmem_st_32dp32bNx<unpack16>, 4, N>(
      tmem_start_col + tmem_col_offset + row_off, src_ptr);
}

// Per-warp O correction with explicit row offsets (avo-style).
// Each warp in the correction WG handles 32 rows independently.
// tmem_base: base TMEM address for this O tile (e.g., O0_tmem[0])
// scale: per-thread rescale factor (one float per row per thread)
// warp_group_offset: threadIdx.x offset for the correction WG (e.g., 256)
template <int HeadDim>
__device__ __forceinline__ void
tmem_correction_x16(uint32_t tmem_base, float scale, int warp_group_offset) {
  int warp_in_group = ((threadIdx.x - warp_group_offset) / 32) % 4;
  uint32_t tr = (uint32_t)((warp_in_group * 32) << 16);
  uint32_t O_base = tmem_base + tr;
  constexpr int kChunks = HeadDim / 16;
  float buf[2][16];
  int cur = 0;

  // Software-pipelined x16 ld/mul/st matching avo's correction_warp_fn
  tl::tmem_ld_32dp32bNx<false>::copy<16>(O_base, (uint32_t *)buf[0]);
  for (int g = 0; g < kChunks; g++) {
    tl::fence_view_async_tmem_load();
    int nxt = cur ^ 1;
    if (g + 1 < kChunks)
      tl::tmem_ld_32dp32bNx<false>::copy<16>(O_base + (g + 1) * 16,
                                              (uint32_t *)buf[nxt]);
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
      float a0 = buf[cur][i] * scale;
      float a1 = buf[cur][i + 1] * scale;
      buf[cur][i] = a0;
      buf[cur][i + 1] = a1;
    }
    tl::tmem_st_32dp32bNx<false>::copy<16>(O_base + g * 16,
                                            (uint32_t const *)buf[cur]);
    cur = nxt;
  }
  tl::fence_view_async_tmem_store();
}

// Non-template wrapper for HeadDim=128 (callable as extern)
__device__ __forceinline__ void
tmem_correction_x16_d128(uint32_t tmem_base, float scale, int warp_group_offset) {
  tl::tmem_correction_x16<128>(tmem_base, scale, warp_group_offset);
}

// ====================================================================
// Avo-style softmax helpers for FA4 1SM attention
// ====================================================================

__device__ __forceinline__ float tcgen05_exp2f_approx(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
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

__device__ __forceinline__ void
tcgen05_softmax_rescale_update(float &scale_out, float &rmax_local,
                               float &rsum_local, float nm,
                               float softmax_scale_log2) {
  float rs_diff = (rmax_local - nm) * softmax_scale_log2;
  float rs_exp = tl::tcgen05_exp2f_approx(rs_diff);
  float rs;
  asm("{                                                  \n"
      ".reg .pred p_skip;                                 \n"
      "setp.ge.ftz.f32 p_skip, %2, 0fC1800000;            \n"
      "selp.f32 %0, 0f3F800000, %3, p_skip;               \n"
      "selp.f32 %1, %1, %4, p_skip;                       \n"
      "}"
      : "=f"(rs), "+f"(rmax_local)
      : "f"(rs_diff), "f"(rs_exp), "f"(nm));
  rsum_local *= rs;
  scale_out = rs;
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
  float x0 = fmaxf(in0, -127.0f);
  float x1 = fmaxf(in1, -127.0f);
  float xr0, xr1;
  asm("{                                                \n"
      ".reg .b64 _xp, _mp, _rp;                        \n"
      "mov.b64 _xp, {%2, %3};                          \n"
      "mov.b64 _mp, {%4, %5};                          \n"
      "add.rm.ftz.f32x2 _rp, _xp, _mp;                 \n"
      "mov.b64 {%0, %1}, _rp;                          \n"
      "}"
      : "=f"(xr0), "=f"(xr1)
      : "f"(x0), "f"(x1), "f"(12582912.0f), "f"(12582912.0f));
  float f0 = x0 - (xr0 - 12582912.0f);
  float f1 = x1 - (xr1 - 12582912.0f);
  float h0 = 0.077119089663028717041015625f;
  float h1 = 0.077119089663028717041015625f;
  tl::tcgen05_fma_f32x2(h0, h1, h0, h1, f0, f1,
                        0.227564394474029541015625f,
                        0.227564394474029541015625f);
  tl::tcgen05_fma_f32x2(h0, h1, h0, h1, f0, f1,
                        0.695146143436431884765625f,
                        0.695146143436431884765625f);
  tl::tcgen05_fma_f32x2(h0, h1, h0, h1, f0, f1, 1.0f, 1.0f);

  int xi0, pi0, ri0, xi1, pi1, ri1;
  asm("mov.b32 %0, %1;" : "=r"(xi0) : "f"(xr0));
  asm("mov.b32 %0, %1;" : "=r"(pi0) : "f"(h0));
  asm("shl.b32 %0, %1, 23;" : "=r"(xi0) : "r"(xi0));
  asm("add.s32 %0, %1, %2;" : "=r"(ri0) : "r"(xi0), "r"(pi0));
  asm("mov.b32 %0, %1;" : "=f"(r0) : "r"(ri0));

  asm("mov.b32 %0, %1;" : "=r"(xi1) : "f"(xr1));
  asm("mov.b32 %0, %1;" : "=r"(pi1) : "f"(h1));
  asm("shl.b32 %0, %1, 23;" : "=r"(xi1) : "r"(xi1));
  asm("add.s32 %0, %1, %2;" : "=r"(ri1) : "r"(xi1), "r"(pi1));
  asm("mov.b32 %0, %1;" : "=f"(r1) : "r"(ri1));
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

__device__ __forceinline__ void
tcgen05_wait_barrier(void const *smem_mbar_ptr, uint32_t phase);

// Avo-style one-iteration S->P online softmax for a 128-row Q stage.
// The caller has already waited on S readiness and issued
// tcgen05.fence::after_thread_sync. One thread owns one row; four warps cover
// the 128 rows. `rmax_state` and `rsum_state` persist across KV blocks.
__device__ __forceinline__ void
tcgen05_softmax_128x128(uint32_t S_tmem_addr, uint32_t P_tmem_addr,
                        float *scale_smem, float *logsum_smem,
                        float *rmax_state, float *rsum_state,
                        void const *mbar_scale_ptr, void const *mbar_p2_ptr,
                        void const *mbar_p_ptr,
                        int k, int loop_extent, float softmax_scale_log2,
                        int seq_len, int q_row_base, int kv_col_base,
                        int is_causal, int warp_group_offset) {
  constexpr int kBlockN = 128;
  constexpr int kEx2EmuFreq = 10;
  constexpr int kEx2EmuRes = 4;
  constexpr int kEx2EmuStartFrg = 1;
  constexpr int kEx2FragSize = 32;

  int row = int(threadIdx.x) - warp_group_offset;
  uint32_t tr = uint32_t((((row >> 5) & 3) * 32) << 16);
  float sv[128];
  float rmax_local = *rmax_state;
  float rsum_local = *rsum_state;
  if (k == 0) {
    rmax_local = -CUDART_INF_F;
    rsum_local = 0.0f;
  }
  float nm = rmax_local;

  #pragma unroll
  for (int cc = 0; cc < kBlockN; cc += 16) {
    tl::tmem_ld_32dp32bNx<false>::copy<16>(S_tmem_addr + tr + cc,
                                            (uint32_t *)&sv[cc]);
  }
  tl::fence_view_async_tmem_load();

  if (is_causal) {
    #pragma unroll
    for (int i = 0; i < kBlockN; i++) {
      bool valid = (kv_col_base + i) < seq_len &&
                   (kv_col_base + i) <= (q_row_base + row);
      if (!valid) {
        sv[i] = -CUDART_INF_F;
      }
    }
  } else {
    int remaining = seq_len - kv_col_base;
    if (remaining < kBlockN) {
      #pragma unroll
      for (int i = 0; i < kBlockN; i++) {
        if (i >= remaining) {
          sv[i] = -CUDART_INF_F;
        }
      }
    }
  }

  float m0 = tl::tcgen05_fmax3(nm, sv[0], sv[1]);
  float m1 = tl::tcgen05_fmax3(sv[2], sv[3], sv[4]);
  float m2 = tl::tcgen05_fmax3(sv[5], sv[6], sv[7]);
  #pragma unroll
  for (int i = 8; i < kBlockN; i += 8) {
    m0 = tl::tcgen05_fmax3(m0, sv[i + 0], sv[i + 1]);
    m1 = tl::tcgen05_fmax3(m1, sv[i + 2], sv[i + 3]);
    m2 = tl::tcgen05_fmax3(m2, sv[i + 4], sv[i + 5]);
    nm = tl::tcgen05_fmax3(nm, sv[i + 6], sv[i + 7]);
  }
  nm = tl::tcgen05_fmax3(tl::tcgen05_fmax2(m0, m1), m2, nm);

  float rs;
  tl::tcgen05_softmax_rescale_update(rs, rmax_local, rsum_local, nm,
                                     softmax_scale_log2);
  scale_smem[row] = rs;
  __syncwarp();
  if ((threadIdx.x & 31) == 0) {
    uint32_t p = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<void *>(mbar_scale_ptr)));
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :
                 : "r"(p)
                 : "memory");
  }

  float neg_max_scaled = -(rmax_local * softmax_scale_log2);
  #pragma unroll
  for (int cc = 0; cc < kBlockN; cc += 16) {
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
      tl::tcgen05_fma_f32x2(sv[cc + i], sv[cc + i + 1],
                            sv[cc + i], sv[cc + i + 1],
                            softmax_scale_log2, softmax_scale_log2,
                            neg_max_scaled, neg_max_scaled);
    }
  }

  float psa[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int cc = kBlockN - 16; cc >= 0; cc -= 16) {
    #pragma unroll
    for (int g = 8; g >= 0; g -= 8) {
      float pval[8];
      #pragma unroll
      for (int i = 0; i < 8; i += 2) {
        int elem = cc + g + i;
        int frag = elem / kEx2FragSize;
        int k_in_frag = elem % kEx2FragSize;
        if (kEx2EmuFreq > 0 && frag >= kEx2EmuStartFrg &&
            frag < (kBlockN / kEx2FragSize - 1) &&
            (k_in_frag % kEx2EmuFreq) >= (kEx2EmuFreq - kEx2EmuRes)) {
          tl::tcgen05_exp2_poly_2(pval[i], pval[i + 1], sv[elem],
                                  sv[elem + 1]);
        } else {
          pval[i] = tl::tcgen05_exp2f_approx(sv[elem]);
          pval[i + 1] = tl::tcgen05_exp2f_approx(sv[elem + 1]);
        }
        psa[i >> 1] += pval[i] + pval[i + 1];
      }
      tl::tcgen05_st_32x32b_x4(
          P_tmem_addr + tr + (cc + g) / 2,
          tl::pack_bf16_pair(pval[0], pval[1]),
          tl::pack_bf16_pair(pval[2], pval[3]),
          tl::pack_bf16_pair(pval[4], pval[5]),
          tl::pack_bf16_pair(pval[6], pval[7]));
    }
    if (cc == 32) {
      tl::fence_view_async_tmem_store();
      if ((threadIdx.x & 31) == 0) {
        uint32_t p = static_cast<uint32_t>(
            __cvta_generic_to_shared(const_cast<void *>(mbar_p2_ptr)));
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                     :
                     : "r"(p)
                     : "memory");
      }
    }
  }
  tl::fence_view_async_tmem_store();

  rsum_local += (psa[0] + psa[1]) + (psa[2] + psa[3]);
  *rmax_state = rmax_local;
  *rsum_state = rsum_local;
  if (k == loop_extent - 1) {
    logsum_smem[row] = rsum_local;
  }

  if ((threadIdx.x & 31) == 0) {
    uint32_t p = static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<void *>(mbar_p_ptr)));
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :
                 : "r"(p)
                 : "memory");
  }
}

__device__ __forceinline__ void
tcgen05_softmax_store_8(uint32_t P_tmem_addr, float const *sv,
                        float &psa0, float &psa1, float &psa2, float &psa3,
                        int elem_base) {
  constexpr int kBlockN = 128;
  constexpr int kEx2EmuFreq = 10;
  constexpr int kEx2EmuRes = 4;
  constexpr int kEx2EmuStartFrg = 1;
  constexpr int kEx2FragSize = 32;

  bfloat16_t h[8];
  #pragma unroll
  for (int i = 0; i < 8; i += 2) {
    int elem = elem_base + i;
    float p0, p1;
    int frag = elem / kEx2FragSize;
    int k_in_frag = elem % kEx2FragSize;
    if (kEx2EmuFreq > 0 && frag >= kEx2EmuStartFrg &&
        frag < (kBlockN / kEx2FragSize - 1) &&
        (k_in_frag % kEx2EmuFreq) >= (kEx2EmuFreq - kEx2EmuRes)) {
      tl::tcgen05_exp2_poly_2(p0, p1, sv[i], sv[i + 1]);
    } else {
      p0 = tl::tcgen05_exp2f_approx(sv[i]);
      p1 = tl::tcgen05_exp2f_approx(sv[i + 1]);
    }
    if (i == 0) {
      psa0 += p0 + p1;
    } else if (i == 2) {
      psa1 += p0 + p1;
    } else if (i == 4) {
      psa2 += p0 + p1;
    } else {
      psa3 += p0 + p1;
    }
    h[i] = bfloat16_t(p0);
    h[i + 1] = bfloat16_t(p1);
  }
  tl::tcgen05_st_32x32b_x4(P_tmem_addr, *reinterpret_cast<uint32_t *>(&h[0]),
                           *reinterpret_cast<uint32_t *>(&h[2]),
                           *reinterpret_cast<uint32_t *>(&h[4]),
                           *reinterpret_cast<uint32_t *>(&h[6]));
}

// Avo-shaped full softmax warp loop.  Unlike tcgen05_softmax_128x128, this
// keeps rmax/rsum in registers for the whole KV loop, matching avo's
// softmax_warp_fn and avoiding per-iteration pointer state traffic.
__device__ __noinline__ void
tcgen05_softmax_warp_1sm(uint32_t S_tmem_addr, uint32_t P_tmem_addr,
                         float *scale_smem, float *logsum_smem,
                         void const *mbar_s_ptr,
                         void const *mbar_scale_base_ptr,
                         void const *mbar_p2_ptr, void const *mbar_p_ptr,
                         int loop_extent, int tile_k_base,
                         float softmax_scale_log2, int seq_len,
                         int q_row_base, int is_causal,
                         int warp_group_offset) {
  constexpr int kBlockN = 128;
  constexpr int kEx2EmuFreq = 10;
  constexpr int kEx2EmuRes = 4;
  constexpr int kEx2EmuStartFrg = 1;
  constexpr int kEx2FragSize = 32;

  auto *mb_scale = reinterpret_cast<Barrier const *>(mbar_scale_base_ptr);
  int row = int(threadIdx.x) - warp_group_offset;
  uint32_t tr = uint32_t((((row >> 5) & 3) * 32) << 16);
  float rmax_local = -CUDART_INF_F;
  float rsum_local = 0.0f;

  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    int tk = tile_k_base + k;
    uint32_t phase = uint32_t(tk & 1);
    int kv_col_base = k * kBlockN;

    tl::tcgen05_wait_barrier(mbar_s_ptr, phase);
    tl::tcgen05_after_thread_sync();

    float sv[128];
    float nm = rmax_local;

    #pragma unroll
    for (int cc = 0; cc < kBlockN; cc += 16) {
      tl::tmem_ld_32dp32bNx<false>::copy<16>(S_tmem_addr + tr + cc,
                                              (uint32_t *)&sv[cc]);
    }
    tl::fence_view_async_tmem_load();

    int remaining = seq_len - kv_col_base;
    if (remaining < kBlockN) {
      #pragma unroll
      for (int i = 0; i < kBlockN; i++) {
        if (i >= remaining) {
          sv[i] = -CUDART_INF_F;
        }
      }
    }

    float m0 = tl::tcgen05_fmax3(nm, sv[0], sv[1]);
    float m1 = tl::tcgen05_fmax3(sv[2], sv[3], sv[4]);
    float m2 = tl::tcgen05_fmax3(sv[5], sv[6], sv[7]);
    #pragma unroll
    for (int i = 8; i < kBlockN; i += 8) {
      m0 = tl::tcgen05_fmax3(m0, sv[i + 0], sv[i + 1]);
      m1 = tl::tcgen05_fmax3(m1, sv[i + 2], sv[i + 3]);
      m2 = tl::tcgen05_fmax3(m2, sv[i + 4], sv[i + 5]);
      nm = tl::tcgen05_fmax3(nm, sv[i + 6], sv[i + 7]);
    }
    nm = tl::tcgen05_fmax3(tl::tcgen05_fmax2(m0, m1), m2, nm);

    float rs;
    tl::tcgen05_softmax_rescale_update(rs, rmax_local, rsum_local, nm,
                                       softmax_scale_log2);
    scale_smem[phase * kBlockN + row] = rs;
    __syncwarp();
    if ((threadIdx.x & 31) == 0) {
      uint32_t p = static_cast<uint32_t>(
          __cvta_generic_to_shared(const_cast<Barrier *>(&mb_scale[phase])));
      asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                   :
                   : "r"(p)
                   : "memory");
    }

    float neg_max_scaled = -(rmax_local * softmax_scale_log2);
    #pragma unroll
    for (int cc = 0; cc < kBlockN; cc += 16) {
      #pragma unroll
      for (int i = 0; i < 16; i += 2) {
        tl::tcgen05_fma_f32x2(sv[cc + i], sv[cc + i + 1],
                              sv[cc + i], sv[cc + i + 1],
                              softmax_scale_log2, softmax_scale_log2,
                              neg_max_scaled, neg_max_scaled);
      }
    }

    float psa[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int cc = kBlockN - 16; cc >= 0; cc -= 16) {
      #pragma unroll
      for (int g = 8; g >= 0; g -= 8) {
        bfloat16_t h[8];
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
          int elem = cc + g + i;
          float p0, p1;
          int frag = elem / kEx2FragSize;
          int k_in_frag = elem % kEx2FragSize;
          if (kEx2EmuFreq > 0 && frag >= kEx2EmuStartFrg &&
              frag < (kBlockN / kEx2FragSize - 1) &&
              (k_in_frag % kEx2EmuFreq) >= (kEx2EmuFreq - kEx2EmuRes)) {
            tl::tcgen05_exp2_poly_2(p0, p1, sv[elem], sv[elem + 1]);
          } else {
            p0 = tl::tcgen05_exp2f_approx(sv[elem]);
            p1 = tl::tcgen05_exp2f_approx(sv[elem + 1]);
          }
          psa[i >> 1] += p0 + p1;
          h[i] = bfloat16_t(p0);
          h[i + 1] = bfloat16_t(p1);
        }
        tl::tcgen05_st_32x32b_x4(
            P_tmem_addr + tr + (cc + g) / 2,
            *reinterpret_cast<uint32_t *>(&h[0]),
            *reinterpret_cast<uint32_t *>(&h[2]),
            *reinterpret_cast<uint32_t *>(&h[4]),
            *reinterpret_cast<uint32_t *>(&h[6]));
      }
      if (cc == 32) {
        tl::fence_view_async_tmem_store();
        if ((threadIdx.x & 31) == 0) {
          uint32_t p = static_cast<uint32_t>(
              __cvta_generic_to_shared(const_cast<void *>(mbar_p2_ptr)));
          asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                       :
                       : "r"(p)
                       : "memory");
        }
      }
    }
    rsum_local += (psa[0] + psa[1]) + (psa[2] + psa[3]);
    if (k == loop_extent - 1) {
      logsum_smem[row] = rsum_local;
    }

    tl::fence_view_async_tmem_store();
    if ((threadIdx.x & 31) == 0) {
      uint32_t p = static_cast<uint32_t>(
          __cvta_generic_to_shared(const_cast<void *>(mbar_p_ptr)));
      asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                   :
                   : "r"(p)
                   : "memory");
    }
  }
}

// Per-warp final epilogue for FA4-style split attention.
//
// Correction warps (usually tid 256..383) own rows 0..127 via tid-wg_offset.
// This helper reads one row from O TMEM, multiplies by the row inverse logsum,
// converts to bf16, and writes into the same swizzled SMEM layout that TileLang's
// TMA store lowering expects for a [128, 128] bf16 shared tile. It lets the
// correction warpgroup produce the epilogue staging buffer while a separate
// warp issues the TMA store, matching avo's role split.
template <int HeadDim>
__device__ __forceinline__ void
tmem_epilogue_store_x16(uint32_t tmem_base, bfloat16_t *smem_base,
                        float rsum, int warp_group_offset) {
  int row = threadIdx.x - warp_group_offset;
  uint32_t O_base = tmem_base + ((uint32_t)row << 16);
  constexpr int kChunks = HeadDim / 16;
  float inv_logsum;
  if (rsum > 0.0f) {
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_logsum) : "f"(rsum));
  } else {
    inv_logsum = 0.0f;
  }

  int lane = threadIdx.x & 31;
  int warp_in_corr = (row >> 5) & 3;
  constexpr int kEpiBlockCols = 64;
  constexpr int kEpiBlockBytes = kEpiBlockCols * 2;
  constexpr int kEpiBlockElems = 32 * kEpiBlockCols;
  char *epi_base =
      reinterpret_cast<char *>(smem_base + warp_in_corr * 32 * HeadDim);
  char *epi_blk0 = epi_base;
  char *epi_blk1 = epi_base + kEpiBlockElems * 2;
  int swiz = (lane & 7) << 4;
  int row_off = lane * kEpiBlockBytes;

  #pragma unroll
  for (int g = 0; g < kChunks; g++) {
    float buf[16];
    tl::tmem_ld_32dp32bNx<false>::copy<16>(O_base + g * 16,
                                            (uint32_t *)buf);
    tl::fence_view_async_tmem_load();
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
      tl::tcgen05_fma_f32x2(buf[i], buf[i + 1], buf[i], buf[i + 1],
                            inv_logsum, inv_logsum, 0.0f, 0.0f);
    }
    bfloat16_t b[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      b[i] = bfloat16_t(buf[i]);
    }

    int d = g * 16;
    int d_in_blk = d % kEpiBlockCols;
    char *blk = (d < kEpiBlockCols) ? epi_blk0 : epi_blk1;
    int col0 = d_in_blk * 2;
    int col1 = (d_in_blk + 8) * 2;
    *reinterpret_cast<uint4 *>(blk + row_off + (col0 ^ swiz)) =
        *reinterpret_cast<uint4 *>(&b[0]);
    *reinterpret_cast<uint4 *>(blk + row_off + (col1 ^ swiz)) =
        *reinterpret_cast<uint4 *>(&b[8]);
  }
}

template <int HeadDim>
__device__ __forceinline__ void
tmem_correction_x16_skip(uint32_t tmem_base, float scale,
                         void const *mbar_pv_ptr, uint32_t pv_phase,
                         int warp_group_offset) {
  unsigned int needs_rescale = __ballot_sync(0xFFFFFFFFu, scale < 1.0f);
  if (needs_rescale == 0) {
    return;
  }
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(
      const_cast<void *>(mbar_pv_ptr)));
  asm volatile(
      "{ .reg .pred P; WAIT%=:"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, %2;"
      "@P bra.uni DONE%=; bra.uni WAIT%=; DONE%=: }"
      : : "r"(p), "r"(pv_phase), "r"(10000000u) : "memory");
  tl::tcgen05_after_thread_sync();
  tl::tmem_correction_x16<HeadDim>(tmem_base, scale, warp_group_offset);
}

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

// Avo-shaped correction + epilogue-staging role for the 1SM split FA4 path.
// This keeps the large correction loop out of the main kernel while preserving
// the same per-warp x16 TMEM correction and 32-row epilogue staging protocol.
__device__ __noinline__ void
tcgen05_correction_epilogue_warp_1sm_skv(
    uint32_t O0_tmem_addr, uint32_t O1_tmem_addr, void *Q_base_ptr,
    float const *scale0_base, float const *scale1_base,
    float const *logsum0_base, float const *logsum1_base,
    void const *mbar_scale0_base, void const *mbar_scale1_base,
    void const *mbar_pv,
    void const *mbar_corr0, void const *mbar_corr1,
    void const *mbar_epi0, void const *mbar_epi1,
    int loop_extent, int tile_k_base) {
  constexpr int kBlockMCTA = 128;
  constexpr int kHeadDim = 128;
  constexpr int kQStageElems = kBlockMCTA * kHeadDim;

  int row = int(threadIdx.x) - 256;
  auto *q_base = reinterpret_cast<bfloat16_t *>(Q_base_ptr);
  auto *mb_scale0 = reinterpret_cast<Barrier const *>(mbar_scale0_base);
  auto *mb_scale1 = reinterpret_cast<Barrier const *>(mbar_scale1_base);

  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    int tk = tile_k_base + k;
    uint32_t scale_idx = uint32_t(tk & 1);
    uint32_t scale_phase = uint32_t((tk >> 1) & 1);

    if (k > 0) {
      tl::tcgen05_wait_barrier((void const *)&mb_scale0[scale_idx],
                               scale_phase);
      float scale = scale0_base[scale_idx * kBlockMCTA + row];
      tl::tmem_correction_x16_skip<kHeadDim>(
          O0_tmem_addr, scale, mbar_pv, uint32_t((tk - 1) & 1), 256);
      tl::tcgen05_before_thread_sync();
      tl::tcgen05_mbarrier_arrive_lane0(mbar_corr0);
    }

    if (k > 0) {
      tl::tcgen05_wait_barrier((void const *)&mb_scale1[scale_idx],
                               scale_phase);
      float scale = scale1_base[scale_idx * kBlockMCTA + row];
      tl::tmem_correction_x16_skip<kHeadDim>(
          O1_tmem_addr, scale, mbar_pv, uint32_t((tk - 1) & 1), 256);
      tl::tcgen05_before_thread_sync();
      tl::tcgen05_mbarrier_arrive_lane0(mbar_corr1);
    }
  }

  uint32_t final_phase = uint32_t((tile_k_base + loop_extent - 1) & 1);
  tl::tcgen05_wait_barrier(mbar_pv, final_phase);
  tl::tcgen05_after_thread_sync();
  tl::tmem_epilogue_store_x16<kHeadDim>(O0_tmem_addr, q_base,
                                        logsum0_base[row], 256);
  tl::tcgen05_before_thread_sync();
  __syncwarp();
  tl::fence_proxy_async();
  tl::tcgen05_mbarrier_arrive_lane0(mbar_epi0);

  tl::tcgen05_after_thread_sync();
  tl::tmem_epilogue_store_x16<kHeadDim>(O1_tmem_addr,
                                        q_base + kQStageElems,
                                        logsum1_base[row], 256);
  tl::tcgen05_before_thread_sync();
  __syncwarp();
  tl::fence_proxy_async();
  tl::tcgen05_mbarrier_arrive_lane0(mbar_epi1);
}

// ====================================================================
// Avo-exact PV MMA helpers (copied from avo/kernels/fmha_2cta_raw.cuh)
// ====================================================================

constexpr uint32_t kTcgen05Fa4SBO = 1024;
constexpr uint32_t kTcgen05Fa4DescHi =
    ((kTcgen05Fa4SBO >> 4) & 0x3FFF) | (1u << 14) | (2u << 29);

__device__ __forceinline__ uint64_t
tcgen05_mk_fast_desc(uint32_t base_lo, uint32_t byte_off) {
  uint32_t lo = base_lo + (byte_off >> 4);
  return (uint64_t(kTcgen05Fa4DescHi) << 32) | lo;
}

// 1SM SS-MMA: C[128x128] = A[128xK] @ B[128xK]^T.
// This is the avo instruction form used for QK, with raw descriptors.
__device__ __forceinline__ void
tcgen05_mma_1sm_128x128(uint32_t tmem_c, uint64_t desc_a,
                         uint64_t desc_b, uint32_t accumulate) {
  uint32_t idesc = (1U << 4)               // c_format = F32
                 | (1U << 7)               // a_format = BF16
                 | (1U << 10)              // b_format = BF16
                 | ((128U / 8) << 17)      // n_dim = 16
                 | ((128U / 16) << 24);    // m_dim = 8
  asm volatile("{                                                        \n"
               ".reg .pred p0;                                           \n"
               "setp.ne.b32 p0, %4, 0;                                   \n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p0; \n"
               "}"
               : : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
		                   "r"(accumulate));
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
  asm volatile("{                                                        \n"
               ".reg .pred p0;                                           \n"
               "setp.ne.b32 p0, %4, 0;                                   \n"
               "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p0; \n"
               "}"
               : : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
                   "r"(accumulate));
}

// Full QK MMA for one Q stage and one K/V stage. Matches avo's qk loop:
// two 64-wide descriptor tiles, four 16-wide MMA issues per tile, then a
// plain .b64 tcgen05 commit to the S-ready mbarrier.
__device__ __forceinline__ void
tcgen05_qk_mma_128x128_skv(void const *Q_stage_ptr,
                            void const *KV_stage_ptr,
                            uint32_t S_tmem_addr,
                            void const *smem_mbar_ptr) {
  if (!cute::elect_one_sync()) return;
  constexpr int kBlockMCTA = 128;
  constexpr int kBlockN = 128;
  constexpr int kTileCols = 64;

  uint32_t q_lo = (uint32_t)((__cvta_generic_to_shared(
                                  const_cast<void *>(Q_stage_ptr)) &
                              0x3FFFF) >>
                             4);
  uint32_t kv_lo = (uint32_t)((__cvta_generic_to_shared(
                                   const_cast<void *>(KV_stage_ptr)) &
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
          S_tmem_addr,
          tl::tcgen05_mk_fast_desc(q_lo, q_off + j * 2),
          tl::tcgen05_mk_fast_desc(kv_lo, k_off + j * 2),
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
  asm volatile("{                                                            \n"
               ".reg .pred p0;                                               \n"
               "setp.ne.b32 p0, %4, 0;                                       \n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p0;   \n"
               "}"
               : : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
	                   "r"(accumulate));
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
  asm volatile("{                                                            \n"
               ".reg .pred p0;                                               \n"
               "setp.ne.b32 p0, %4, 0;                                       \n"
               "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p0;   \n"
               "}"
               : : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
                   "r"(accumulate));
}

// Avo-exact commit: NO shared:: qualifier (just .b64 [%0])
// This is different from TileLang's shared::cluster variant!
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

// Full PV MMA for one qs: avo-exact 128×64 BMN, high D first, low D second.
// V_lo_ptr: pointer to V low-D buffer (bf16, [128, 64], row stride=128B)
// V_hi_ptr: pointer to V high-D buffer (bf16, [128, 64], row stride=128B)
// P_tmem_addr: P tile TMEM base address
// O_tmem_addr: O tile TMEM base address
// accumulate: 0 for k==0 (clear), 1 for k>0 (accumulate into existing O)
// NOTE: Must be called from warp 12 only (codegen adds the guard).
// Uses elect_one_sync internally — only lane 0 of warp 12 issues MMA.
__device__ __forceinline__ void
tcgen05_pv_mma_128x64_avo(void const *V_lo_ptr, void const *V_hi_ptr,
                           uint32_t P_tmem_addr,
                           uint32_t O_tmem_addr, uint32_t accumulate) {
  if (!cute::elect_one_sync()) return;
  constexpr int kBlockN = 128;   // K-rows (seq block)
  constexpr int kTileCols = 64;  // descriptor tile width

  auto *v_lo = (bfloat16_t *)const_cast<void *>(V_lo_ptr);
  auto *v_hi = (bfloat16_t *)const_cast<void *>(V_hi_ptr);
  tl::Tcgen05SMemDescriptor desc_lo, desc_hi;
  tl::initialize_tcgen05_descriptor(
      desc_lo, v_lo, 0, 64, 0, 0, 2);
  tl::initialize_tcgen05_descriptor(
      desc_hi, v_hi, 0, 64, 0, 0, 2);

  constexpr int kStride = 2048;  // bytes per 16 K-rows: 16 * 64 * 2

  // D-tile 1 (HIGH, dim 64-127) FIRST — avo processes high D before low D
  tl::fence_proxy_async();
  #pragma unroll
  for (int j = 0; j < kBlockN / 16; j++) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        O_tmem_addr + 64, P_tmem_addr + j * 8,
        uint64_t(desc_hi + j * kStride),
        (j == 0) ? accumulate : 1u);
  }
  // D-tile 0 (LOW, dim 0-63) SECOND
  #pragma unroll
  for (int j = 0; j < kBlockN / 16; j++) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        O_tmem_addr, P_tmem_addr + j * 8,
        uint64_t(desc_lo + j * kStride),
        (j == 0) ? accumulate : 1u);
  }
}

// Variant for avo's shared K/V stage layout. One stage is physically laid out as
// two 128x64 bf16 tiles: low-D at base and high-D at base + 128*64 elements.
__device__ __forceinline__ void
tcgen05_pv_mma_128x64_skv(void const *V_stage_ptr,
                           uint32_t P_tmem_addr,
                           uint32_t O_tmem_addr, uint32_t accumulate) {
  auto *v_base = (bfloat16_t *)const_cast<void *>(V_stage_ptr);
  tl::tcgen05_pv_mma_128x64_avo(
      (void const *)v_base,
      (void const *)(v_base + 128 * 64),
      P_tmem_addr, O_tmem_addr, accumulate);
}

__device__ __forceinline__ void
tcgen05_qk_mma_128x128_skv_lane0(void const *Q_stage_ptr,
                                  void const *KV_stage_ptr,
                                  uint32_t S_tmem_addr,
                                  void const *smem_mbar_ptr) {
  constexpr int kBlockMCTA = 128;
  constexpr int kBlockN = 128;
  constexpr int kTileCols = 64;

  uint32_t q_lo = (uint32_t)((__cvta_generic_to_shared(
                                  const_cast<void *>(Q_stage_ptr)) &
                              0x3FFFF) >>
                             4);
  uint32_t kv_lo = (uint32_t)((__cvta_generic_to_shared(
                                   const_cast<void *>(KV_stage_ptr)) &
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
          S_tmem_addr,
          tl::tcgen05_mk_fast_desc(q_lo, q_off + j * 2),
          tl::tcgen05_mk_fast_desc(kv_lo, k_off + j * 2),
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
tcgen05_qk_mma_128x128_skv_fast(uint32_t sQ_lo, uint32_t sKV_lo,
                                 uint32_t q_off_base,
                                 uint32_t kv_off_base,
                                 uint32_t S_tmem_addr,
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
          S_tmem_addr,
          tl::tcgen05_mk_fast_desc(sQ_lo, q_off + j * 2),
          tl::tcgen05_mk_fast_desc(sKV_lo, k_off + j * 2),
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
tcgen05_pv_mma_128x64_skv_lane0(void const *V_stage_ptr,
                                 uint32_t P_tmem_addr,
                                 uint32_t O_tmem_addr, uint32_t accumulate) {
  constexpr int kBlockN = 128;
  auto *v_base = (bfloat16_t *)const_cast<void *>(V_stage_ptr);
  tl::Tcgen05SMemDescriptor desc_lo, desc_hi;
  tl::initialize_tcgen05_descriptor(desc_lo, v_base, 0, 64, 0, 0, 2);
  tl::initialize_tcgen05_descriptor(desc_hi, v_base + 128 * 64, 0, 64, 0, 0, 2);

  constexpr int kStride = 2048;
  #pragma unroll
  for (int j = 0; j < kBlockN / 16; j++) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        O_tmem_addr + 64, P_tmem_addr + j * 8,
        uint64_t(desc_hi + j * kStride),
        (j == 0) ? accumulate : 1u);
  }
  #pragma unroll
  for (int j = 0; j < kBlockN / 16; j++) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        O_tmem_addr, P_tmem_addr + j * 8,
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
tcgen05_pv_mma_128x64_skv_fast(uint32_t sKV_lo, uint32_t kv_off_base,
                                uint32_t P_tmem_addr,
                                uint32_t O_tmem_addr,
                                uint32_t accumulate) {
  constexpr int kBlockN = 128;
  constexpr int kTileCols = 64;
  uint32_t v_hi = kv_off_base + kBlockN * kTileCols * 2;

  #pragma unroll
  for (int j = 0; j < kBlockN; j += 16) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        O_tmem_addr + 64, P_tmem_addr + j / 2,
        tl::tcgen05_mk_fast_desc(sKV_lo, v_hi + j * kTileCols * 2),
        (j == 0) ? accumulate : 1u);
  }
  #pragma unroll
  for (int j = 0; j < kBlockN; j += 16) {
    tl::tcgen05_mma_1sm_ts_128x64_bmn(
        O_tmem_addr, P_tmem_addr + j / 2,
        tl::tcgen05_mk_fast_desc(sKV_lo, kv_off_base + j * kTileCols * 2),
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

// Avo-shaped producer warp loop for TileLang's 4D TMA tensor maps.  The caller
// passes Q0 and KV0 SMEM bases; Q1/KV1/KV2 are addressed by fixed stage offsets
// because the split kernel allocates them contiguously to mirror avo's sQ/sKV.
__device__ __noinline__ void
tcgen05_producer_warp_1sm_skv(
    const CUtensorMap &Q_desc, const CUtensorMap &K_desc,
    const CUtensorMap &V_desc, void *Q_base_ptr, void *KV_base_ptr,
    void const *mbar_q0, void const *mbar_q1,
    void const *mbar_k_base, void const *mbar_v_base,
    void const *mbar_s1, void const *mbar_pv,
    int loop_extent, int tile_k_base, int q_row_base,
    int q_head, int kv_head, int batch) {
  if (threadIdx.x % 32 != 0) return;

  constexpr int kBlockMCTA = 128;
  constexpr int kBlockN = 128;
  constexpr int kBPerCTA = 64;
  constexpr int kHeadDim = 128;
  constexpr int kTileCols = 64;
  constexpr int kKVStages = 3;
  constexpr int kQStageElems = kBlockMCTA * kHeadDim;
  constexpr int kKVStageElems = kBlockN * kHeadDim;
  constexpr int kQStageBytes = 2 * kBlockMCTA * kTileCols * 2;
  constexpr int kKVTileBytes = kBPerCTA * kTileCols * 2;
  constexpr int kKVStageBytes = 4 * kKVTileBytes;

  auto *q_base = reinterpret_cast<bfloat16_t *>(Q_base_ptr);
  auto *kv_base = reinterpret_cast<bfloat16_t *>(KV_base_ptr);
  auto *mb_k = reinterpret_cast<Barrier const *>(mbar_k_base);
  auto *mb_v = reinterpret_cast<Barrier const *>(mbar_v_base);

  tl::tcgen05_arrive_expect_tx(mbar_q0, kQStageBytes);
  tl::tma_load(Q_desc, *const_cast<Barrier *>(reinterpret_cast<Barrier const *>(mbar_q0)),
               q_base, 0, q_head, q_row_base, batch);
  tl::tma_load(Q_desc, *const_cast<Barrier *>(reinterpret_cast<Barrier const *>(mbar_q0)),
               q_base + kBlockMCTA * kTileCols, 64, q_head, q_row_base, batch);

  tl::tcgen05_arrive_expect_tx(mbar_q1, kQStageBytes);
  tl::tma_load(Q_desc, *const_cast<Barrier *>(reinterpret_cast<Barrier const *>(mbar_q1)),
               q_base + kQStageElems, 0, q_head, q_row_base + kBlockMCTA,
               batch);
  tl::tma_load(Q_desc, *const_cast<Barrier *>(reinterpret_cast<Barrier const *>(mbar_q1)),
               q_base + kQStageElems + kBlockMCTA * kTileCols, 64, q_head,
               q_row_base + kBlockMCTA, batch);

  #pragma unroll
  for (int s = 0; s < kKVStages; ++s) {
    if (s < loop_extent) {
      int tk = tile_k_base + s;
      int stage = tk % kKVStages;
      auto *bar = &mb_k[stage];
      bfloat16_t *dst = kv_base + stage * kKVStageElems;
      int row = s * kBlockN;
      tl::tcgen05_arrive_expect_tx((void const *)bar, kKVStageBytes);
      #pragma unroll
      for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
          tl::tma_load(K_desc, *const_cast<Barrier *>(bar),
                       dst + t * kBlockN * kTileCols + r * kBPerCTA * kTileCols,
                       t * kTileCols, kv_head, row + r * kBPerCTA, batch);
        }
      }
    }
  }

  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    int tk = tile_k_base + k;
    int stage = tk % kKVStages;
    bfloat16_t *dst = kv_base + stage * kKVStageElems;

    tl::tcgen05_wait_barrier(mbar_s1, uint32_t(tk & 1));
    tl::tcgen05_arrive_expect_tx((void const *)&mb_v[stage], kKVStageBytes);
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      #pragma unroll
      for (int t = 0; t < 2; ++t) {
        tl::tma_load(V_desc, *const_cast<Barrier *>(&mb_v[stage]),
                     dst + r * kBlockN * kTileCols + t * kBPerCTA * kTileCols,
                     r * kBPerCTA, kv_head, k * kBlockN + t * kBPerCTA, batch);
      }
    }

    if (k + kKVStages < loop_extent) {
      tl::tcgen05_wait_barrier(mbar_pv, uint32_t(tk & 1));
      int next_k = k + kKVStages;
      tl::tcgen05_arrive_expect_tx((void const *)&mb_k[stage], kKVStageBytes);
      #pragma unroll
      for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
          tl::tma_load(K_desc, *const_cast<Barrier *>(&mb_k[stage]),
                       dst + t * kBlockN * kTileCols + r * kBPerCTA * kTileCols,
                       t * kTileCols, kv_head, next_k * kBlockN + r * kBPerCTA,
                       batch);
        }
      }
    }
  }
}

// Avo-shaped epilogue warp: wait for correction-staged SMEM tiles, issue the
// two 64-column TMA stores for both Q stages, then wait for completion.  Keeping
// this in a noinline helper removes the long TMA-store branch from the main
// kernel and matches avo's epilogue_warp_fn structure more closely.
__device__ __noinline__ void
tcgen05_epilogue_warp_1sm_skv(
    const CUtensorMap &Output_desc, void const *Q_base_ptr,
    void const *mbar_epi0, void const *mbar_epi1,
    int tile_iter, int q_row_base, int q_head, int batch) {
  if (threadIdx.x % 32 != 0) return;

  constexpr int kBlockMCTA = 128;
  constexpr int kHeadDim = 128;
  constexpr int kTileCols = 64;
  constexpr int kEpiRows = 32;
  constexpr int kQStageElems = kBlockMCTA * kHeadDim;
  constexpr int kEpiBlockElems = kEpiRows * kTileCols;
  auto *q_base = reinterpret_cast<bfloat16_t const *>(Q_base_ptr);
  uint32_t phase = uint32_t(tile_iter & 1);

  tl::tcgen05_wait_barrier(mbar_epi0, phase);
  tl::fence_proxy_async();
  #pragma unroll
  for (int cw = 0; cw < 4; ++cw) {
    bfloat16_t const *epi_base = q_base + cw * kEpiRows * kHeadDim;
    int row = q_row_base + cw * kEpiRows;
    tl::tma_store(Output_desc, epi_base, 0, q_head, row, batch);
    tl::tma_store(Output_desc, epi_base + kEpiBlockElems, 64, q_head, row,
                  batch);
  }
  tl::tma_store_arrive();

  tl::tcgen05_wait_barrier(mbar_epi1, phase);
  tl::fence_proxy_async();
  bfloat16_t const *q1_base = q_base + kQStageElems;
  #pragma unroll
  for (int cw = 0; cw < 4; ++cw) {
    bfloat16_t const *epi_base = q1_base + cw * kEpiRows * kHeadDim;
    int row = q_row_base + kBlockMCTA + cw * kEpiRows;
    tl::tma_store(Output_desc, epi_base, 0, q_head, row, batch);
    tl::tma_store(Output_desc, epi_base + kEpiBlockElems, 64, q_head, row,
                  batch);
  }
  tl::tma_store_arrive();

  tl::tma_store_wait<0>();
}

// Avo-shaped MMA warp loop for the current 1SM split FA4 TileLang layout:
// Q0/Q1 are contiguous [128,128] bf16 stages, and KV0/KV1/KV2 are contiguous
// [128,128] bf16 stages reused first for K and then V.  Keeping descriptor base
// computation and stage selection inside this single helper prevents the
// TileLang outlined MMA role from expanding into three branch copies per stage.
__device__ __noinline__ void
tcgen05_mma_warp_1sm_skv_fast(
    void const *Q_base_ptr, void const *KV_base_ptr,
    void const *mbar_q0, void const *mbar_q1,
    void const *mbar_k_base, void const *mbar_v_base,
    void const *mbar_s0, void const *mbar_s1,
    void const *mbar_p2_0, void const *mbar_p2_1,
    void const *mbar_p0, void const *mbar_p1,
    void const *mbar_corr0, void const *mbar_corr1,
    void const *mbar_pv,
    uint32_t S0_tmem_addr, uint32_t S1_tmem_addr,
    uint32_t P0_tmem_addr, uint32_t P1_tmem_addr,
    uint32_t O0_tmem_addr, uint32_t O1_tmem_addr,
    int loop_extent, int tile_k_base, int tile_iter) {
  if (threadIdx.x % 32 != 0) return;

  constexpr int kBlockMCTA = 128;
  constexpr int kBlockN = 128;
  constexpr int kHeadDim = 128;
  constexpr int kKVStages = 3;
  constexpr int kQStageBytes = kBlockMCTA * kHeadDim * 2;
  constexpr int kKVStageBytes = kBlockN * kHeadDim * 2;

  auto *mb_k = reinterpret_cast<Barrier const *>(mbar_k_base);
  auto *mb_v = reinterpret_cast<Barrier const *>(mbar_v_base);

  uint32_t sQ_lo = (uint32_t)((__cvta_generic_to_shared(
                                  const_cast<void *>(Q_base_ptr)) &
                              0x3FFFF) >>
                             4);
  uint32_t sKV_lo = (uint32_t)((__cvta_generic_to_shared(
                                   const_cast<void *>(KV_base_ptr)) &
                               0x3FFFF) >>
                              4);

  tl::tcgen05_wait_barrier(mbar_q0, uint32_t(tile_iter & 1));
  tl::tcgen05_wait_barrier(mbar_q1, uint32_t(tile_iter & 1));
  tl::tcgen05_after_thread_sync();

  int kv0 = tile_k_base % kKVStages;
  int kv0_phase = (tile_k_base / kKVStages) & 1;
  uint32_t kv0_off = uint32_t(kv0 * kKVStageBytes);
  tl::tcgen05_wait_barrier((void const *)&mb_k[kv0], uint32_t(kv0_phase));
  tl::tcgen05_after_thread_sync();
  #pragma unroll
  for (int qs = 0; qs < 2; ++qs) {
    tl::tcgen05_qk_mma_128x128_skv_fast(
        sQ_lo, sKV_lo, uint32_t(qs * kQStageBytes), kv0_off,
        qs == 0 ? S0_tmem_addr : S1_tmem_addr, qs == 0 ? mbar_s0 : mbar_s1);
  }

  int O_should_accumulate = 0;
  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    int tk = tile_k_base + k;
    int kv_stage = tk % kKVStages;
    int kv_phase = (tk / kKVStages) & 1;
    uint32_t kv_off = uint32_t(kv_stage * kKVStageBytes);
    uint32_t pv_phase = uint32_t(tk & 1);
    uint32_t pv_acc = O_should_accumulate ? 1u : 0u;

    #pragma unroll
    for (int qs = 0; qs < 2; ++qs) {
      tl::tcgen05_wait_barrier(qs == 0 ? mbar_p2_0 : mbar_p2_1,
                               pv_phase);
      tl::tcgen05_wait_barrier((void const *)&mb_v[kv_stage],
                               uint32_t(kv_phase));
      if (O_should_accumulate) {
        tl::tcgen05_wait_barrier(qs == 0 ? mbar_corr0 : mbar_corr1,
                                 pv_phase);
      }
      tl::tcgen05_after_thread_sync();
      tl::tcgen05_wait_barrier(qs == 0 ? mbar_p0 : mbar_p1, pv_phase);
      tl::tcgen05_after_thread_sync();

      tl::tcgen05_pv_mma_128x64_skv_fast(
          sKV_lo, kv_off, qs == 0 ? P0_tmem_addr : P1_tmem_addr,
          qs == 0 ? O0_tmem_addr : O1_tmem_addr, pv_acc);

      if (qs == 1) {
        tl::tcgen05_commit_1sm_lane0(mbar_pv);
      }

      if (k + 1 < loop_extent) {
        int ntk = tk + 1;
        int next_stage = ntk % kKVStages;
        if (qs == 0) {
          int next_phase = (ntk / kKVStages) & 1;
          tl::tcgen05_wait_barrier((void const *)&mb_k[next_stage],
                                   uint32_t(next_phase));
          tl::tcgen05_after_thread_sync();
        }
        uint32_t next_off = uint32_t(next_stage * kKVStageBytes);
        tl::tcgen05_qk_mma_128x128_skv_fast(
            sQ_lo, sKV_lo, uint32_t(qs * kQStageBytes), next_off,
            qs == 0 ? S0_tmem_addr : S1_tmem_addr,
            qs == 0 ? mbar_s0 : mbar_s1);
      }
    }
    O_should_accumulate = 1;
  }
}

// Avo-style full MMA warp loop for the 1SM split FA4 path.  Keeping this loop
// in a single helper prevents TileLang's outlined MMA role from expanding into
// many stage-specific branches, which otherwise makes ptxas spill heavily.
__device__ __noinline__ void
tcgen05_mma_warp_1sm_skv(
    void const *Q0_stage_ptr, void const *Q1_stage_ptr,
    void const *KV0_stage_ptr, void const *KV1_stage_ptr,
    void const *KV2_stage_ptr,
    void const *mbar_q0, void const *mbar_q1,
    void const *mbar_k0, void const *mbar_k1, void const *mbar_k2,
    void const *mbar_v0, void const *mbar_v1, void const *mbar_v2,
    void const *mbar_p2_0, void const *mbar_p2_1,
    void const *mbar_p0, void const *mbar_p1,
    void const *mbar_corr0, void const *mbar_corr1,
    void const *mbar_pv, void const *mbar_s0, void const *mbar_s1,
    uint32_t S0_tmem_addr, uint32_t S1_tmem_addr,
    uint32_t P0_tmem_addr, uint32_t P1_tmem_addr,
    uint32_t O0_tmem_addr, uint32_t O1_tmem_addr,
    int loop_extent, int tile_k_base, int tile_corr_base, int tile_iter) {
  if (threadIdx.x % 32 != 0) return;

  void const *kv_ptrs[3] = {KV0_stage_ptr, KV1_stage_ptr, KV2_stage_ptr};
  void const *mbar_k[3] = {mbar_k0, mbar_k1, mbar_k2};
  void const *mbar_v[3] = {mbar_v0, mbar_v1, mbar_v2};

  tl::tcgen05_wait_barrier(mbar_q0, uint32_t(tile_iter & 1));
  tl::tcgen05_wait_barrier(mbar_q1, uint32_t(tile_iter & 1));
  tl::tcgen05_after_thread_sync();

  int first_stage = tile_k_base % 3;
  int first_phase = (tile_k_base / 3) & 1;
  tl::tcgen05_wait_barrier(mbar_k[first_stage], uint32_t(first_phase));
  tl::tcgen05_after_thread_sync();
  tl::tcgen05_qk_mma_128x128_skv_lane0(Q0_stage_ptr, kv_ptrs[first_stage],
                                       S0_tmem_addr, mbar_s0);
  tl::tcgen05_qk_mma_128x128_skv_lane0(Q1_stage_ptr, kv_ptrs[first_stage],
                                       S1_tmem_addr, mbar_s1);

  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    int tk = tile_k_base + k;
    int stage = tk % 3;
    int phase = (tk / 3) & 1;
    uint32_t pv_phase = uint32_t(tk & 1);
    uint32_t accum = (k == 0) ? 0u : 1u;

    tl::tcgen05_wait_barrier(mbar_p2_0, pv_phase);
    tl::tcgen05_wait_barrier(mbar_v[stage], uint32_t(phase));
    if (k > 0) {
      uint32_t corr_phase = uint32_t((tile_corr_base + k - 1) & 1);
      tl::tcgen05_wait_barrier(mbar_corr0, corr_phase);
      tl::tcgen05_after_thread_sync();
    }
    tl::tcgen05_wait_barrier(mbar_p0, pv_phase);
    tl::tcgen05_after_thread_sync();

    tl::tcgen05_pv_mma_128x64_skv_lane0(kv_ptrs[stage], P0_tmem_addr,
                                        O0_tmem_addr, accum);

    if (k + 1 < loop_extent) {
      int ntk = tk + 1;
      int next_stage = ntk % 3;
      int next_phase = (ntk / 3) & 1;
      tl::tcgen05_wait_barrier(mbar_k[next_stage], uint32_t(next_phase));
      tl::tcgen05_after_thread_sync();
      tl::tcgen05_qk_mma_128x128_skv_lane0(Q0_stage_ptr, kv_ptrs[next_stage],
                                           S0_tmem_addr, mbar_s0);
    }

    tl::tcgen05_wait_barrier(mbar_p2_1, pv_phase);
    tl::tcgen05_wait_barrier(mbar_v[stage], uint32_t(phase));
    if (k > 0) {
      uint32_t corr_phase = uint32_t((tile_corr_base + k - 1) & 1);
      tl::tcgen05_wait_barrier(mbar_corr1, corr_phase);
      tl::tcgen05_after_thread_sync();
    }
    tl::tcgen05_wait_barrier(mbar_p1, pv_phase);
    tl::tcgen05_after_thread_sync();
    tl::tcgen05_pv_mma_128x64_skv_lane0(kv_ptrs[stage], P1_tmem_addr,
                                        O1_tmem_addr, accum);
    tl::tcgen05_commit_1sm_lane0(mbar_pv);

    if (k + 1 < loop_extent) {
      int ntk = tk + 1;
      int next_stage = ntk % 3;
      tl::tcgen05_qk_mma_128x128_skv_lane0(Q1_stage_ptr, kv_ptrs[next_stage],
                                           S1_tmem_addr, mbar_s1);
    }
  }
}

// 3-stage K/V reuse helpers for the current TileLang 1SM split layout.
// This keeps the existing split-path shared memory allocation but uses three
// logical K/V stages:
//   stage0 -> K0 buffer, stage1 -> K1 buffer, stage2 -> V0_lo/V0_hi area.
// The positive A/B result is that this recovers part of avo's long-sequence
// pipeline benefit without requiring a wholesale source replacement.
__device__ __forceinline__ bfloat16_t *
tcgen05_reuse3_stage_ptr(void *k0_ptr, void *k1_ptr, void *stage2_ptr,
                         int stage) {
  if (stage == 0) return reinterpret_cast<bfloat16_t *>(k0_ptr);
  if (stage == 1) return reinterpret_cast<bfloat16_t *>(k1_ptr);
  return reinterpret_cast<bfloat16_t *>(stage2_ptr);
}

__device__ __forceinline__ void const *
tcgen05_reuse3_kbar(void const *mbar_k0, void const *mbar_k1,
                    void const *mbar_k2, int stage) {
  if (stage == 0) return mbar_k0;
  if (stage == 1) return mbar_k1;
  return mbar_k2;
}

__device__ __forceinline__ void const *
tcgen05_reuse3_vbar(void const *mbar_v0, void const *mbar_v1,
                    void const *mbar_v2, int stage) {
  if (stage == 0) return mbar_v0;
  if (stage == 1) return mbar_v1;
  return mbar_v2;
}

__device__ __forceinline__ void
tcgen05_reuse3_load_k(const CUtensorMap &K_desc, void *k0_ptr, void *k1_ptr,
                      void *stage2_ptr, void const *mbar_k0,
                      void const *mbar_k1, void const *mbar_k2, int k,
                      int kv_head, int batch) {
  constexpr int kBlockN = 128;
  constexpr int kTileCols = 64;
  constexpr int kBytes = kBlockN * 128 * 2;
  int stage = k % 3;
  auto *dst = tcgen05_reuse3_stage_ptr(k0_ptr, k1_ptr, stage2_ptr, stage);
  auto *bar = reinterpret_cast<Barrier *>(
      const_cast<void *>(tcgen05_reuse3_kbar(mbar_k0, mbar_k1, mbar_k2, stage)));
  tl::tcgen05_arrive_expect_tx((void const *)bar, kBytes);
  tl::tma_load(K_desc, *bar, dst, 0, kv_head, k * kBlockN, batch);
  tl::tma_load(K_desc, *bar, dst + kBlockN * kTileCols, 64, kv_head,
               k * kBlockN, batch);
}

__device__ __forceinline__ void
tcgen05_reuse3_load_v(const CUtensorMap &V_desc, void *k0_ptr, void *k1_ptr,
                      void *stage2_ptr, void const *mbar_v0,
                      void const *mbar_v1, void const *mbar_v2, int k,
                      int kv_head, int batch) {
  constexpr int kBlockN = 128;
  constexpr int kTileCols = 64;
  constexpr int kBytes = kBlockN * 128 * 2;
  int stage = k % 3;
  auto *dst = tcgen05_reuse3_stage_ptr(k0_ptr, k1_ptr, stage2_ptr, stage);
  auto *bar = reinterpret_cast<Barrier *>(
      const_cast<void *>(tcgen05_reuse3_vbar(mbar_v0, mbar_v1, mbar_v2, stage)));
  tl::tcgen05_arrive_expect_tx((void const *)bar, kBytes);
  tl::tma_load(V_desc, *bar, dst, 0, kv_head, k * kBlockN, batch);
  tl::tma_load(V_desc, *bar, dst + kBlockN * kTileCols, 64, kv_head,
               k * kBlockN, batch);
}

__device__ __noinline__ void
tcgen05_producer_warp_1sm_reuse3(
    const CUtensorMap &Q_desc, const CUtensorMap &K_desc,
    const CUtensorMap &V_desc, void *Q0_stage_ptr, void *Q1_stage_ptr,
    void *K0_stage_ptr, void *K1_stage_ptr, void *KV2_stage_ptr,
    void const *mbar_q0, void const *mbar_q1, void const *mbar_k0,
    void const *mbar_k1, void const *mbar_k2, void const *mbar_v0,
    void const *mbar_v1, void const *mbar_v2, void const *mbar_s1,
    void const *mbar_pv, int loop_extent, int q_row_base, int q_head,
    int kv_head, int batch) {
  if (threadIdx.x % 32 != 0) return;

  constexpr int kBlockM = 128;
  constexpr int kBlockN = 128;
  constexpr int kTileCols = 64;
  constexpr int kQBytes = kBlockM * 128 * 2;

  auto *q0 = reinterpret_cast<bfloat16_t *>(Q0_stage_ptr);
  auto *q1 = reinterpret_cast<bfloat16_t *>(Q1_stage_ptr);

  auto *mb_q0 = reinterpret_cast<Barrier *>(const_cast<void *>(mbar_q0));
  auto *mb_q1 = reinterpret_cast<Barrier *>(const_cast<void *>(mbar_q1));
  tl::tcgen05_arrive_expect_tx(mbar_q0, kQBytes);
  tl::tma_load(Q_desc, *mb_q0, q0, 0, q_head, q_row_base, batch);
  tl::tma_load(Q_desc, *mb_q0, q0 + kBlockM * kTileCols, 64, q_head,
               q_row_base, batch);

  tl::tcgen05_arrive_expect_tx(mbar_q1, kQBytes);
  tl::tma_load(Q_desc, *mb_q1, q1, 0, q_head, q_row_base + kBlockM, batch);
  tl::tma_load(Q_desc, *mb_q1, q1 + kBlockM * kTileCols, 64, q_head,
               q_row_base + kBlockM, batch);

  if (loop_extent > 0) {
    tl::tcgen05_reuse3_load_k(K_desc, K0_stage_ptr, K1_stage_ptr,
                              KV2_stage_ptr, mbar_k0, mbar_k1, mbar_k2, 0,
                              kv_head, batch);
  }
  if (loop_extent > 1) {
    tl::tcgen05_reuse3_load_k(K_desc, K0_stage_ptr, K1_stage_ptr,
                              KV2_stage_ptr, mbar_k0, mbar_k1, mbar_k2, 1,
                              kv_head, batch);
  }
  if (loop_extent > 2) {
    tl::tcgen05_reuse3_load_k(K_desc, K0_stage_ptr, K1_stage_ptr,
                              KV2_stage_ptr, mbar_k0, mbar_k1, mbar_k2, 2,
                              kv_head, batch);
  }

  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    tl::tcgen05_wait_barrier(mbar_s1, uint32_t(k & 1));
    tl::tcgen05_after_thread_sync();
    tl::tcgen05_reuse3_load_v(V_desc, K0_stage_ptr, K1_stage_ptr,
                              KV2_stage_ptr, mbar_v0, mbar_v1, mbar_v2, k,
                              kv_head, batch);
    if (k + 3 < loop_extent) {
      tl::tcgen05_wait_barrier(mbar_pv, uint32_t(k & 1));
      tl::tcgen05_after_thread_sync();
      tl::tcgen05_reuse3_load_k(K_desc, K0_stage_ptr, K1_stage_ptr,
                                KV2_stage_ptr, mbar_k0, mbar_k1, mbar_k2,
                                k + 3, kv_head, batch);
    }
  }
}

__device__ __noinline__ void
tcgen05_mma_warp_1sm_reuse3(
    void const *Q0_stage_ptr, void const *Q1_stage_ptr,
    void *K0_stage_ptr, void *K1_stage_ptr, void *KV2_stage_ptr,
    void const *mbar_q0, void const *mbar_q1, void const *mbar_k0,
    void const *mbar_k1, void const *mbar_k2, void const *mbar_v0,
    void const *mbar_v1, void const *mbar_v2, void const *mbar_s0,
    void const *mbar_s1, void const *mbar_p0, void const *mbar_p1,
    void const *mbar_p2_0, void const *mbar_p2_1,
    void const *mbar_corr0, void const *mbar_corr1, void const *mbar_pv,
    uint32_t S0_tmem_addr, uint32_t S1_tmem_addr, uint32_t P0_tmem_addr,
    uint32_t P1_tmem_addr, uint32_t O0_tmem_addr, uint32_t O1_tmem_addr,
    int loop_extent) {
  if (threadIdx.x % 32 != 0) return;

  tl::tcgen05_wait_barrier(mbar_q0, 0);
  tl::tcgen05_wait_barrier(mbar_q1, 0);
  tl::tcgen05_after_thread_sync();

  if (loop_extent <= 0) return;
  {
    void const *kbar = tl::tcgen05_reuse3_kbar(mbar_k0, mbar_k1, mbar_k2, 0);
    tl::tcgen05_wait_barrier(kbar, 0);
    tl::tcgen05_after_thread_sync();
    auto *kptr =
        tl::tcgen05_reuse3_stage_ptr(K0_stage_ptr, K1_stage_ptr, KV2_stage_ptr, 0);
    tl::tcgen05_qk_mma_128x128_skv_lane0(Q0_stage_ptr, kptr, S0_tmem_addr,
                                         mbar_s0);
    tl::tcgen05_qk_mma_128x128_skv_lane0(Q1_stage_ptr, kptr, S1_tmem_addr,
                                         mbar_s1);
  }

  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    int stage = k % 3;
    int stage_phase = (k / 3) & 1;
    uint32_t pv_phase = uint32_t(k & 1);
    uint32_t accum = (k == 0) ? 0u : 1u;

    tl::tcgen05_wait_barrier(mbar_p2_0, pv_phase);
    void const *vbar =
        tl::tcgen05_reuse3_vbar(mbar_v0, mbar_v1, mbar_v2, stage);
    tl::tcgen05_wait_barrier(vbar, uint32_t(stage_phase));
    if (k > 0) {
      tl::tcgen05_wait_barrier(mbar_corr0, uint32_t((k - 1) & 1));
    }
    tl::tcgen05_wait_barrier(mbar_p0, pv_phase);
    tl::tcgen05_after_thread_sync();
    auto *vptr = tl::tcgen05_reuse3_stage_ptr(K0_stage_ptr, K1_stage_ptr,
                                              KV2_stage_ptr, stage);
    tl::tcgen05_pv_mma_128x64_avo(vptr, vptr + 128 * 64, P0_tmem_addr,
                                  O0_tmem_addr, accum);

    tl::tcgen05_wait_barrier(mbar_p2_1, pv_phase);
    if (k > 0) {
      tl::tcgen05_wait_barrier(mbar_corr1, uint32_t((k - 1) & 1));
    }
    tl::tcgen05_wait_barrier(mbar_p1, pv_phase);
    tl::tcgen05_after_thread_sync();
    tl::tcgen05_pv_mma_128x64_avo(vptr, vptr + 128 * 64, P1_tmem_addr,
                                  O1_tmem_addr, accum);
    tl::tcgen05_commit_1sm(mbar_pv);

    if (k + 1 < loop_extent) {
      int nk = k + 1;
      int nstage = nk % 3;
      int nphase = (nk / 3) & 1;
      void const *kbar =
          tl::tcgen05_reuse3_kbar(mbar_k0, mbar_k1, mbar_k2, nstage);
      tl::tcgen05_wait_barrier(kbar, uint32_t(nphase));
      tl::tcgen05_after_thread_sync();
      auto *kptr = tl::tcgen05_reuse3_stage_ptr(
          K0_stage_ptr, K1_stage_ptr, KV2_stage_ptr, nstage);
      tl::tcgen05_qk_mma_128x128_skv_lane0(Q0_stage_ptr, kptr, S0_tmem_addr,
                                           mbar_s0);
      tl::tcgen05_qk_mma_128x128_skv_lane0(Q1_stage_ptr, kptr, S1_tmem_addr,
                                           mbar_s1);
    }
  }
}

// ====================================================================
// Avo-shaped 2CTA FA4 helpers. These are the performance path for
// attention_kernel.cu parity: 512 threads, cluster_dims=2, cta_group::2 MMA,
// separate K/V SMEM, and four-stage K/V pipeline.
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
tma_load_2sm_avo(const CUtensorMap *descriptor, void const *const smem_ptr,
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

__device__ __noinline__ void
tcgen05_softmax_warp_2cta(uint32_t S_tmem_addr, uint32_t P_tmem_addr,
                          void const *Q_stage_ptr, void *Output_ptr,
                          float *rs_smem, void const *mbar_s_ptr,
                          void const *mbar_p_ptr, void const *mbar_p2_ptr,
                          void const *mbar_pv_base_ptr,
                          void const *mbar_rs_ptr, int loop_extent,
                          int tile_k_base, float softmax_scale_log2,
                          int seq_len, int q_row_base, int64_t output_offset) {
  constexpr int kBlockN = 128;
  constexpr int kHeadDim = 128;
  constexpr int kQStages = 2;
  constexpr int kBlockMCTA = 128;
  constexpr int kKVStages = 4;
  constexpr int kEx2EmuFreq = 10;
  constexpr int kEx2EmuRes = 4;
  constexpr int kEx2EmuStartFrg = 1;
  constexpr int kEx2FragSize = 32;

  auto *mb_pv = reinterpret_cast<Barrier const *>(mbar_pv_base_ptr);
  int cr = tl::block_rank_in_cluster();
  int w = int(threadIdx.x) >> 5;
  int tid = int(threadIdx.x);
  int qs = w >> 2;
  int row = tid - (qs == 1 ? 128 : 0);
  uint32_t tr = uint32_t(((cr * kBlockMCTA + (w & 3) * 32) << 16));
  float rmax_local = -CUDART_INF_F;
  float rsum_local = 0.0f;

  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    int tk = tile_k_base + k;
    uint32_t phase = uint32_t(tk & 1);

    tl::tcgen05_wait_barrier(mbar_s_ptr, phase);
    tl::tcgen05_after_thread_sync();

    float sv[kBlockN];
    float nm = rmax_local;
    #pragma unroll
    for (int cc = 0; cc < kBlockN; cc += 16) {
      tl::tmem_ld_32dp32bNx<false>::copy<16>(S_tmem_addr + tr + cc,
                                              (uint32_t *)&sv[cc]);
    }
    tl::fence_view_async_tmem_load();

    int remaining = seq_len - k * kBlockN;
    if (remaining < kBlockN) {
      #pragma unroll
      for (int i = 0; i < kBlockN; ++i) {
        if (i >= remaining) {
          sv[i] = -CUDART_INF_F;
        }
      }
    }

    float m0 = tl::tcgen05_fmax3(nm, sv[0], sv[1]);
    float m1 = tl::tcgen05_fmax3(sv[2], sv[3], sv[4]);
    float m2 = tl::tcgen05_fmax3(sv[5], sv[6], sv[7]);
    #pragma unroll
    for (int i = 8; i < kBlockN; i += 8) {
      m0 = tl::tcgen05_fmax3(m0, sv[i + 0], sv[i + 1]);
      m1 = tl::tcgen05_fmax3(m1, sv[i + 2], sv[i + 3]);
      m2 = tl::tcgen05_fmax3(m2, sv[i + 4], sv[i + 5]);
      nm = tl::tcgen05_fmax3(nm, sv[i + 6], sv[i + 7]);
    }
    nm = tl::tcgen05_fmax3(tl::tcgen05_fmax2(m0, m1), m2, nm);

    float rs_diff = (rmax_local - nm) * softmax_scale_log2;
    float rs_exp = tl::tcgen05_exp2f_approx(rs_diff);
    float rs;
    asm("{                                                  \n"
        ".reg .pred p_skip;                                 \n"
        "setp.ge.ftz.f32 p_skip, %2, 0fC1000000;            \n"
        "selp.f32 %0, 0f3F800000, %3, p_skip;               \n"
        "selp.f32 %1, %1, %4, p_skip;                       \n"
        "}"
        : "=f"(rs), "+f"(rmax_local)
        : "f"(rs_diff), "f"(rs_exp), "f"(nm));
    rsum_local *= rs;

    rs_smem[phase * kQStages * kBlockMCTA + qs * kBlockMCTA + row] = rs;
    __syncwarp();
    tl::tcgen05_mbarrier_arrive_lane0(mbar_rs_ptr);

    float neg_max_scaled = -(rmax_local * softmax_scale_log2);
    #pragma unroll
    for (int cc = 0; cc < kBlockN; cc += 16) {
      #pragma unroll
      for (int i = 0; i < 16; i += 2) {
        tl::tcgen05_fma_f32x2(sv[cc + i], sv[cc + i + 1],
                              sv[cc + i], sv[cc + i + 1],
                              softmax_scale_log2, softmax_scale_log2,
                              neg_max_scaled, neg_max_scaled);
      }
    }

    float psa[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int cc = kBlockN - 16; cc >= 0; cc -= 16) {
      #pragma unroll
      for (int g = 8; g >= 0; g -= 8) {
        bfloat16_t h[8];
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
          int elem = cc + g + i;
          float p0, p1;
          int frag = elem / kEx2FragSize;
          int k_in_frag = elem % kEx2FragSize;
          if (kEx2EmuFreq > 0 && frag >= kEx2EmuStartFrg &&
              frag < (kBlockN / kEx2FragSize - 1) &&
              (k_in_frag % kEx2EmuFreq) >= (kEx2EmuFreq - kEx2EmuRes)) {
            tl::tcgen05_exp2_poly_2(p0, p1, sv[elem], sv[elem + 1]);
          } else {
            p0 = tl::tcgen05_exp2f_approx(sv[elem]);
            p1 = tl::tcgen05_exp2f_approx(sv[elem + 1]);
          }
          psa[i >> 1] += p0 + p1;
          h[i] = bfloat16_t(p0);
          h[i + 1] = bfloat16_t(p1);
        }
        tl::tcgen05_st_32x32b_x4(
            P_tmem_addr + tr + (cc + g) / 2,
            *reinterpret_cast<uint32_t *>(&h[0]),
            *reinterpret_cast<uint32_t *>(&h[2]),
            *reinterpret_cast<uint32_t *>(&h[4]),
            *reinterpret_cast<uint32_t *>(&h[6]));
      }
      if (cc == 32) {
        tl::fence_view_async_tmem_store();
        tl::tcgen05_mbarrier_arrive_cluster_lane0(mbar_p2_ptr);
      }
    }
    tl::fence_view_async_tmem_store();
    tl::tcgen05_mbarrier_arrive_cluster_lane0(mbar_p_ptr);
    rsum_local += (psa[0] + psa[1]) + (psa[2] + psa[3]);
  }

  int last_tk = tile_k_base + loop_extent - 1;
  int last_v_stage = last_tk % kKVStages;
  int last_v_phase = (last_tk / kKVStages) & 1;
  tl::tcgen05_wait_barrier((void const *)&mb_pv[last_v_stage],
                           uint32_t(last_v_phase));
  tl::tcgen05_after_thread_sync();

  int global_row = q_row_base + cr * kBlockMCTA + qs * 256 + row;
  if (global_row < seq_len) {
    float inv = 0.0f;
    if (rsum_local > 0.0f) {
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(rsum_local));
    }
    auto *out = reinterpret_cast<bfloat16_t *>(Output_ptr) + output_offset +
                int64_t(global_row) * kHeadDim;
    #pragma unroll
    for (int d = 0; d < kHeadDim; d += 16) {
      float t[16];
      tl::tmem_ld_32dp32bNx<false>::copy<16>(
          (S_tmem_addr + 256) + tr + d, (uint32_t *)t);
      tl::fence_view_async_tmem_load();
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        t[i] *= inv;
      }
      bfloat16_t b[16];
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        b[i] = bfloat16_t(t[i]);
      }
      tl::global_store_256(*reinterpret_cast<ulonglong4 *>(&b[0]),
                           out + d, true);
    }
  }
}

__device__ __noinline__ void
tcgen05_correction_warp_2cta(float *rs_smem, void const *mbar_rs_base,
                             void const *mbar_corr_base,
                             void const *mbar_pv_base, uint32_t O0_tmem_addr,
                             int loop_extent, int tile_k_base, int cr) {
  constexpr int kQStages = 2;
  constexpr int kBlockMCTA = 128;
  constexpr int kKVStages = 4;
  constexpr int kHeadDim = 128;

  auto *mb_rs = reinterpret_cast<Barrier const *>(mbar_rs_base);
  auto *mb_corr = reinterpret_cast<Barrier const *>(mbar_corr_base);
  auto *mb_pv = reinterpret_cast<Barrier const *>(mbar_pv_base);
  int w = int(threadIdx.x) >> 5;
  int tid = int(threadIdx.x);
  int corr_tid = tid - 256;
  uint32_t tr = uint32_t(((cr * kBlockMCTA + (w & 3) * 32) << 16));

  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    int tk = tile_k_base + k;
    #pragma unroll
    for (int qs = 0; qs < kQStages; ++qs) {
      tl::tcgen05_wait_barrier((void const *)&mb_rs[qs], uint32_t(tk & 1));
      float rs = rs_smem[(tk & 1) * kQStages * kBlockMCTA +
                         qs * kBlockMCTA + corr_tid];
      if (k > 0) {
        unsigned int needs_rescale = __ballot_sync(0xFFFFFFFFu, rs < 1.0f);
        if (needs_rescale) {
          int prev = tk - 1;
          int pv_stage = prev % kKVStages;
          int pv_phase = (prev / kKVStages) & 1;
          tl::tcgen05_wait_barrier((void const *)&mb_pv[pv_stage],
                                   uint32_t(pv_phase));
          tl::tcgen05_after_thread_sync();
          uint32_t O_base = (O0_tmem_addr + qs * 128) + tr;
          float buf[2][16];
          int cur = 0;
          tl::tmem_ld_32dp32bNx<false>::copy<16>(O_base,
                                                  (uint32_t *)buf[cur]);
          #pragma unroll
          for (int g = 0; g < kHeadDim / 16; ++g) {
            tl::fence_view_async_tmem_load();
            int nxt = cur ^ 1;
            if (g + 1 < kHeadDim / 16) {
              tl::tmem_ld_32dp32bNx<false>::copy<16>(
                  O_base + (g + 1) * 16, (uint32_t *)buf[nxt]);
            }
            #pragma unroll
            for (int i = 0; i < 16; i += 2) {
              tl::tcgen05_fma_f32x2(buf[cur][i], buf[cur][i + 1],
                                    buf[cur][i], buf[cur][i + 1],
                                    rs, rs, 0.0f, 0.0f);
            }
            tl::tmem_st_32dp32bNx<false>::copy<16>(
                O_base + g * 16, (uint32_t *)buf[cur]);
            cur = nxt;
          }
          tl::fence_view_async_tmem_store();
        }
      }
      tl::tcgen05_mbarrier_arrive_cluster_lane0((void const *)&mb_corr[qs]);
    }
  }
}

__device__ __forceinline__ void
tcgen05_qk_mma_2cta_fast(uint32_t sQ_lo, uint32_t sK_lo,
                         uint32_t q_off_base, uint32_t k_off_base,
                         uint32_t S_tmem_addr, void const *mbar_s) {
  constexpr int kBlockMCTA = 128;
  constexpr int kBPerCTA = 64;
  constexpr int kTileCols = 64;
  int first = 1;
  #pragma unroll
  for (int t = 0; t < 2; ++t) {
    uint32_t q_off = q_off_base + t * kBlockMCTA * kTileCols * 2;
    uint32_t k_off = k_off_base + t * kBPerCTA * kTileCols * 2;
    #pragma unroll
    for (int j = 0; j < kTileCols; j += 16) {
      tl::tcgen05_mma_2cta_256x128(
          S_tmem_addr,
          tl::tcgen05_mk_fast_desc(sQ_lo, q_off + j * 2),
          tl::tcgen05_mk_fast_desc(sK_lo, k_off + j * 2),
          first ? 0u : 1u);
      first = 0;
    }
  }
  tl::tcgen05_commit_2cta(mbar_s);
}

__device__ __noinline__ void
tcgen05_mma_warp_2cta(void const *Q_base_ptr, void const *K_base_ptr,
                      void const *V_base_ptr, void const *mbar_q_base,
                      void const *mbar_k_base, void const *mbar_s_base,
                      void const *mbar_p_base, void const *mbar_p2_base,
                      void const *mbar_v_base, void const *mbar_pv_base,
                      void const *mbar_corr_base,
                      void const *mbar_k_rel_base,
                      void const *mbar_v_rel_base, uint32_t S0_tmem_addr,
                      int loop_extent, int q_phase, int tile_k_base, int cr,
                      uint32_t q_stage_bytes, uint32_t k_stage_bytes,
                      uint32_t v_stage_bytes, uint32_t k_stage_elems) {
  if (cr != 0 || (threadIdx.x & 31) != 0) return;
  constexpr int kQStages = 2;
  constexpr int kKVStages = 4;
  constexpr int kBlockMCTA = 128;
  constexpr int kBPerCTA = 64;
  constexpr int kTileCols = 64;
  auto *mb_q = reinterpret_cast<Barrier const *>(mbar_q_base);
  auto *mb_k = reinterpret_cast<Barrier const *>(mbar_k_base);
  auto *mb_s = reinterpret_cast<Barrier const *>(mbar_s_base);
  auto *mb_p = reinterpret_cast<Barrier const *>(mbar_p_base);
  auto *mb_p2 = reinterpret_cast<Barrier const *>(mbar_p2_base);
  auto *mb_v = reinterpret_cast<Barrier const *>(mbar_v_base);
  auto *mb_pv = reinterpret_cast<Barrier const *>(mbar_pv_base);
  auto *mb_corr = reinterpret_cast<Barrier const *>(mbar_corr_base);
  auto *mb_k_rel = reinterpret_cast<Barrier const *>(mbar_k_rel_base);
  auto *mb_v_rel = reinterpret_cast<Barrier const *>(mbar_v_rel_base);

  uint32_t sQ_lo = uint32_t((__cvta_generic_to_shared(
                                 const_cast<void *>(Q_base_ptr)) & 0x3FFFF) >> 4);
  uint32_t sK_lo = uint32_t((__cvta_generic_to_shared(
                                 const_cast<void *>(K_base_ptr)) & 0x3FFFF) >> 4);
  uint32_t sV_lo = uint32_t((__cvta_generic_to_shared(
                                 const_cast<void *>(V_base_ptr)) & 0x3FFFF) >> 4);

  #pragma unroll
  for (int qs = 0; qs < kQStages; ++qs) {
    tl::tcgen05_wait_barrier((void const *)&mb_q[qs], uint32_t(q_phase));
  }
  tl::tcgen05_after_thread_sync();

  int kv0_stage = tile_k_base % kKVStages;
  int kv0_phase = (tile_k_base / kKVStages) & 1;
  tl::tcgen05_wait_barrier((void const *)&mb_k[kv0_stage],
                           uint32_t(kv0_phase));
  tl::tcgen05_after_thread_sync();
  #pragma unroll
  for (int qs = 0; qs < kQStages; ++qs) {
    uint32_t q_off = qs * q_stage_bytes;
    uint32_t k_off = kv0_stage * k_stage_bytes;
    tl::tcgen05_qk_mma_2cta_fast(sQ_lo, sK_lo, q_off, k_off,
                                 S0_tmem_addr + qs * 128,
                                 (void const *)&mb_s[qs]);
  }
  tl::tcgen05_commit_2cta((void const *)&mb_k_rel[kv0_stage]);

  int O_should_accumulate = 0;
  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    int tk = tile_k_base + k;
    int v_stage = tk % kKVStages;
    int v_phase = (tk / kKVStages) & 1;
    #pragma unroll
    for (int qs = 0; qs < kQStages; ++qs) {
      uint32_t pv_acc = O_should_accumulate ? 1u : 0u;
      uint32_t O_addr = S0_tmem_addr + 256 + qs * 128;
      uint32_t P_base = S0_tmem_addr + 64 + qs * 128;
      uint32_t v_base = v_stage * v_stage_bytes;
      uint32_t v_hi = v_base + kBPerCTA * kTileCols * 2;

      tl::tcgen05_wait_barrier((void const *)&mb_p2[qs], uint32_t(tk & 1));
      tl::tcgen05_wait_barrier((void const *)&mb_v[v_stage],
                               uint32_t(v_phase));
      if (O_should_accumulate) {
        tl::tcgen05_wait_barrier((void const *)&mb_corr[qs], uint32_t(tk & 1));
      }
      tl::tcgen05_after_thread_sync();

      tl::tcgen05_mma_2cta_ts_256x128_bmn(
          O_addr, P_base + kTileCols / 2,
          tl::tcgen05_mk_fast_desc(sV_lo, v_hi), pv_acc);
      #pragma unroll
      for (int j = 16; j < kTileCols; j += 16) {
        tl::tcgen05_mma_2cta_ts_256x128_bmn(
            O_addr, P_base + kTileCols / 2 + j / 2,
            tl::tcgen05_mk_fast_desc(sV_lo, v_hi + j * kTileCols * 2), 1u);
      }
      #pragma unroll
      for (int j = 32; j < kTileCols; j += 16) {
        tl::tcgen05_mma_2cta_ts_256x128_bmn(
            O_addr, P_base + j / 2,
            tl::tcgen05_mk_fast_desc(sV_lo, v_base + j * kTileCols * 2), 1u);
      }
      tl::tcgen05_wait_barrier((void const *)&mb_p[qs], uint32_t(tk & 1));
      tl::tcgen05_after_thread_sync();
      tl::tcgen05_mma_2cta_ts_256x128_bmn(
          O_addr, P_base, tl::tcgen05_mk_fast_desc(sV_lo, v_base), 1u);
      tl::tcgen05_mma_2cta_ts_256x128_bmn(
          O_addr, P_base + 8,
          tl::tcgen05_mk_fast_desc(sV_lo, v_base + 16 * kTileCols * 2), 1u);

      if (qs == kQStages - 1) {
        tl::tcgen05_commit_2cta((void const *)&mb_pv[v_stage]);
        tl::tcgen05_commit_2cta((void const *)&mb_v_rel[v_stage]);
      }
      if (k + 1 < loop_extent) {
        int ntk = tk + 1;
        int next_stage = ntk % kKVStages;
        int next_phase = (ntk / kKVStages) & 1;
        if (qs == 0) {
          tl::tcgen05_wait_barrier((void const *)&mb_k[next_stage],
                                   uint32_t(next_phase));
          tl::tcgen05_after_thread_sync();
        }
        uint32_t nq_off = qs * q_stage_bytes;
        uint32_t nk_off = next_stage * k_stage_bytes;
        tl::tcgen05_qk_mma_2cta_fast(sQ_lo, sK_lo, nq_off, nk_off,
                                     S0_tmem_addr + qs * 128,
                                     (void const *)&mb_s[qs]);
        if (qs == kQStages - 1) {
          tl::tcgen05_commit_2cta((void const *)&mb_k_rel[next_stage]);
        }
      }
    }
    O_should_accumulate = 1;
  }
}

__device__ __noinline__ void
tcgen05_producer_warp_2cta(
    const CUtensorMap *Q_desc, const CUtensorMap *K_desc,
    const CUtensorMap *V_desc, void *Q_base_ptr, void *K_base_ptr,
    void *V_base_ptr, void const *mbar_q_base, void const *mbar_k_base,
    void const *mbar_v_base, void const *mbar_k_rel_base,
    void const *mbar_v_rel_base, int loop_extent, int tile_k_base,
    int q_row_base, int kv_row_base, int cr, uint32_t q_stage_bytes,
    uint32_t kv_tile_bytes, uint32_t q_stage_elems,
    uint32_t k_stage_elems, uint32_t v_stage_elems, int q_phase) {
  int lane = int(threadIdx.x) & 31;
  if (lane >= 4) return;
  constexpr int kQStages = 2;
  constexpr int kKVStages = 4;
  constexpr int kBlockMCTA = 128;
  constexpr int kBlockN = 128;
  constexpr int kBPerCTA = 64;
  constexpr int kTileCols = 64;
  constexpr int kPageRows = 32;
  auto *q_base = reinterpret_cast<bfloat16_t *>(Q_base_ptr);
  auto *k_base = reinterpret_cast<bfloat16_t *>(K_base_ptr);
  auto *v_base = reinterpret_cast<bfloat16_t *>(V_base_ptr);
  auto *mb_q = reinterpret_cast<Barrier const *>(mbar_q_base);
  auto *mb_k = reinterpret_cast<Barrier const *>(mbar_k_base);
  auto *mb_v = reinterpret_cast<Barrier const *>(mbar_v_base);
  auto *mb_k_rel = reinterpret_cast<Barrier const *>(mbar_k_rel_base);
  auto *mb_v_rel = reinterpret_cast<Barrier const *>(mbar_v_rel_base);

  #pragma unroll
  for (int qs = 0; qs < kQStages; ++qs) {
    tl::tcgen05_mbarrier_arrive_expect_tx_cluster_lane0(
        (void const *)&mb_q[qs], q_stage_bytes);
    int q_row = q_row_base + qs * 256;
    int q_chunk = lane;
    tl::tma_load_2sm_avo(
        Q_desc,
        q_base + qs * q_stage_elems + q_chunk * kPageRows * kTileCols,
        (void const *)&mb_q[qs], 0, q_row + q_chunk * kPageRows);
    tl::tma_load_2sm_avo(
        Q_desc,
        q_base + qs * q_stage_elems + kBlockMCTA * kTileCols +
            q_chunk * kPageRows * kTileCols,
        (void const *)&mb_q[qs], kTileCols, q_row + q_chunk * kPageRows);
  }

  #pragma unroll
  for (int s = 0; s < kKVStages; ++s) {
    if (s < loop_extent) {
      int tk = tile_k_base + s;
      int stage = tk % kKVStages;
      int k_row = kv_row_base + s * kBlockN + cr * kBPerCTA;
      tl::tcgen05_mbarrier_arrive_expect_tx_cluster_lane0(
          (void const *)&mb_k[stage], kv_tile_bytes);
      int t = lane >> 1;
      int c = lane & 1;
      tl::tma_load_2sm_avo(
          K_desc,
          k_base + stage * k_stage_elems + t * kBPerCTA * kTileCols +
              c * kPageRows * kTileCols,
          (void const *)&mb_k[stage], t * kTileCols, k_row + c * kPageRows);
    }
  }

  #pragma unroll
  for (int s = 0; s < kKVStages; ++s) {
    if (s < loop_extent) {
      int tk = tile_k_base + s;
      int stage = tk % kKVStages;
      int v_row = kv_row_base + s * kBlockN;
      tl::tcgen05_mbarrier_arrive_expect_tx_cluster_lane0(
          (void const *)&mb_v[stage], kv_tile_bytes);
      int t = lane >> 1;
      int c = lane & 1;
      tl::tma_load_2sm_avo(
          V_desc,
          v_base + stage * v_stage_elems + t * kBPerCTA * kTileCols +
              c * kPageRows * kTileCols,
          (void const *)&mb_v[stage], cr * kBPerCTA,
          v_row + t * kBPerCTA + c * kPageRows);
    }
  }

  #pragma unroll 1
  for (int k = 0; k < loop_extent; ++k) {
    int tk = tile_k_base + k;
    int stage = tk % kKVStages;
    int phase = (tk / kKVStages) & 1;
    if (k + kKVStages < loop_extent) {
      tl::tcgen05_wait_barrier((void const *)&mb_k_rel[stage],
                               uint32_t(phase));
      int next = k + kKVStages;
      int next_stage = (tk + kKVStages) % kKVStages;
      int k_row = kv_row_base + next * kBlockN + cr * kBPerCTA;
      tl::tcgen05_mbarrier_arrive_expect_tx_cluster_lane0(
          (void const *)&mb_k[next_stage], kv_tile_bytes);
      int t = lane >> 1;
      int c = lane & 1;
      tl::tma_load_2sm_avo(
          K_desc,
          k_base + next_stage * k_stage_elems + t * kBPerCTA * kTileCols +
              c * kPageRows * kTileCols,
          (void const *)&mb_k[next_stage], t * kTileCols,
          k_row + c * kPageRows);
    }
    if (k + kKVStages < loop_extent) {
      tl::tcgen05_wait_barrier((void const *)&mb_v_rel[stage],
                               uint32_t(phase));
      int next = k + kKVStages;
      int next_stage = (tk + kKVStages) % kKVStages;
      int v_row = kv_row_base + next * kBlockN;
      tl::tcgen05_mbarrier_arrive_expect_tx_cluster_lane0(
          (void const *)&mb_v[next_stage], kv_tile_bytes);
      int t = lane >> 1;
      int c = lane & 1;
      tl::tma_load_2sm_avo(
          V_desc,
          v_base + next_stage * v_stage_elems + t * kBPerCTA * kTileCols +
              c * kPageRows * kTileCols,
          (void const *)&mb_v[next_stage], cr * kBPerCTA,
          v_row + t * kBPerCTA + c * kPageRows);
    }
  }
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
