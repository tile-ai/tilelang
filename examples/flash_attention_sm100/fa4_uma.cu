// AVO Kernel - 2CTA + TMA + tcgen05 Flash Attention Forward
// 2-Q-stage interleaved pipeline (FA4-style)
//
// Architecture:
//   - Cluster of 2 CTAs (2SM), 512 threads each (16 warps)
//   - Warps 0-3 (WG0): Softmax0 for Q-tile 0 (176 regs)
//   - Warps 4-7 (WG1): Softmax1 for Q-tile 1 (176 regs)
//   - Warps 8-11 (WG2): Correction (O rescaling) (88 regs)
//   - Warp 12: MMA consumer (tcgen05.mma.cta_group::2) (72 regs)
//   - Warp 13: Producer (TMA loads) (72 regs)
//   - Warp 14: Epilogue (TMA store) (72 regs)
//   - Warp 15: Empty (donate registers) (72 regs)
//
// Register budget: 8×176 + 4×88 + 4×72 = 2048 regs/thread → 2048×32 = 65536/SM ✓
//
// TMEM layout (FA4):
//   S0: 0-127, P0: 64-127 (overlap), S1: 128-255, P1: 192-255 (overlap)
//   O0: 256-383, O1: 384-511

 #include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <random>
#include <vector>

#include "cuda_check.h"
#include "utils/common.cuh"
#include "utils/sm100.cuh"
#include <cuda_bf16.h>

// 2^x approximate on device (avoid host __exp2f); SFU-alternate path uses this too.
__device__ __forceinline__ float fast_exp2f(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float fmax3(float a, float b, float c) {
  float r;
  asm("max.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}

__device__ __forceinline__ float fmax2(float a, float b) {
  float r;
  asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float fast_inv(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// ========= Configuration =========
constexpr int kBlockM=256, kBlockMCTA=128, kBlockN=128, kHeadDim=128;
constexpr int kBPerCTA=64;
constexpr int kTileCols=64;
constexpr uint32_t kSBO=1024;
constexpr int kQStages=2;

// 512 threads (16 warps) — FA4 warp layout with register donation via setmaxnreg
constexpr int kThreads=512;
constexpr int kTidProd = 416;  // Warp 13, lane 0 (producer init thread)
constexpr int kWarpMMA=12;
constexpr int kWarpProd=13;
constexpr int kWarpEpilogue=14;
constexpr int kBarSoftmaxStatsBase=0;

// Register budgets (multiples of 8). User tuned this for our specific kernel;
// 168/96/80 outperforms FA4's 176/88/72 default by ~1.5% mean here (the extra 8
// regs to correction help the rescale/finalize loops where ptxas was spilling).
constexpr int kRegsSoftmax=176;
constexpr int kRegsCorrection=88;
constexpr int kRegsOther=72;

// SMEM layout — full FA4 alignment: separate sQ (64KB) + sO (64KB) + sK + sV.
// kv_stage=3 matches FA4's effective 3K+3V pipeline depth (FA4 uses 6 logical
// stages with K/V sharing same 96KB region; we use 3+3 separate which is the
// same memory footprint and the same in-flight tensor count).
//   sQ + sO + sK + sV + scratch = 64+64+48+48+3 = 227 KiB (at the cap).
// The separate sO unblocks producer's next-Q reload from epilogue's TMA store
// — see correction_finalize_o_fn / epilogue_warp_fn.
constexpr int kKVStages=3;
constexpr int kPerStageKV=kBPerCTA*kHeadDim;
constexpr int kSmemQ=kQStages*kBlockMCTA*kHeadDim*2;
constexpr int kSmemO=kQStages*kBlockMCTA*kHeadDim*2;  // separate epilogue staging buffer (FA4-aligned)
constexpr int kSmemK=kKVStages*kPerStageKV*2;
constexpr int kSmemV=kKVStages*kPerStageKV*2;
constexpr int kSmemKV=kSmemK+kSmemV;
constexpr int kQStageHalfs=kBlockMCTA*kHeadDim;

// exp2 interleaving (FA4 tuning: 2CTA noncausal hdim=128 SM100)
constexpr int kEx2EmuFreq = 10;
constexpr int kEx2EmuRes = 4;
constexpr int kEx2EmuStartFrg = 1;
constexpr int kEx2FragSize = 32;
constexpr bool kEnableEarlyPSplit = true;

// Rescale threshold (log2-units). 0.0f = always rescale when max grows
// (standard FA behavior, best P-tile precision). Higher values skip small
// rescales as an optimization at the cost of precision.
constexpr float kRescaleThreshold = 8.0f;

// ========= Helpers =========
__device__ __forceinline__ uint64_t mk(void *p) { return make_smem_desc(p, kSBO, 2); }

// Precomputed upper 32 bits of SMEM descriptor:
// [32:45] = SBO >> 4, bit 46 = version, [61:63] = swizzle mode.
constexpr uint32_t kDescHi = ((kSBO >> 4) & 0x3FFF) | (1u << 14) | (2u << 29);

__device__ __forceinline__ uint64_t mk_fast(uint32_t base_lo, uint32_t byte_off) {
  uint32_t lo = base_lo + (byte_off >> 4);
  return ((uint64_t)kDescHi << 32) | lo;
}

#define FMA_F32X2(r0, r1, a0, a1, b0, b1, c0, c1)                                               \
  asm("{                                                \n"                                      \
      ".reg .b64 _a, _b, _c, _r;                       \n"                                      \
      "mov.b64 _a, {%2, %3};                           \n"                                       \
      "mov.b64 _b, {%4, %5};                           \n"                                      \
      "mov.b64 _c, {%6, %7};                           \n"                                      \
      "fma.rn.ftz.f32x2 _r, _a, _b, _c;               \n"                                        \
      "mov.b64 {%0, %1}, _r;                           \n"                                      \
      "}"                                                                                       \
      : "=f"(r0), "=f"(r1)                                                                      \
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1))

__device__ __forceinline__ float2 e2e_asm2(float x, float y) {
  float o0, o1;
  asm(
      "{\n\t"
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
      "mov.f32 f7, 0f3D9DF09D;\n\t"  // 0f3D9DF09D=0.07711908966302872
      "mov.b64 l6, {f7, f7};\n\t"
      "mov.f32 f6, 0f3E6906A4;\n\t"  // 0f3E6906A4=0.22756439447402954
      "mov.b64 l5, {f6, f6};\n\t"
      "mov.f32 f5, 0f3F31F519;\n\t"  // 0f3F31F519=0.6951461434364319
      "mov.b64 l4, {f5, f5};\n\t"
      "mov.f32 f4, 0f3F800000;\n\t"  // 0f3F800000=1.0
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
      : "=f"(o0), "=f"(o1)
      : "f"(x), "f"(y));
  return make_float2(o0, o1);
}

// ========= Warp-role device functions (noinline for separate register allocation) =========

// Softmax warp role: FA4-style 3-pass softmax + SFU / packed poly exp2 interleaving
__device__ __noinline__ void softmax_warp_fn(
    int qs,
    int tid,
    int cr,
    int w,
    float *rs_smem,
    float *sum_smem,
    Mbarrier *mb_s,
    Mbarrier *mb_p,
    Mbarrier *mb_p2,
    uint32_t tbase,
    int nkb,
    int seqlen_k,
    float softmax_scale,
    int kb_base) {
  const uint32_t tr = (uint32_t)((cr * 128 + (w % 4) * 32) << 16);
  const int sm_tid = tid - (qs == 1 ? 128 : 0);
  const int sm_bar = kBarSoftmaxStatsBase + qs * 4 + (w & 3);
  const float sc_log2 = softmax_scale * 1.44269504089f;
  float rmax_local = -FLT_MAX, rsum_local = 0.f;

  for (int kb = 0; kb < nkb; kb++) {
    int tkb = kb_base + kb;
    mb_s[qs].wait(tkb & 1);
    tcgen05_after_thread_sync();
    int kr = min(kBlockN, seqlen_k - kb * kBlockN);

    float sv[128];

    // Pass 1: TMEM load + mask + max
    // tmem load(S)
#pragma unroll
    for (int cc = 0; cc < kBlockN; cc += 32) {
      tcgen05_ld_sync_32x32b_x32(&sv[cc], (tbase + qs * 128) + tr + cc);
    }
    tcgen05_wait_ld_sync();
    if (kr < kBlockN) {
#pragma unroll
      for (int i = 0; i < kBlockN; i++)
        if (i >= kr) sv[i] = -FLT_MAX;
    }

    float block_max;
    {
      float m0 = fmax3(sv[0], sv[1], sv[2]);
      float m1 = fmax3(sv[3], sv[4], sv[5]);
      float m2 = fmax3(sv[6], sv[7], sv[8]);
      float m3 = sv[9];
#pragma unroll
      for (int i = 10; i < kBlockN; i += 8) {
        m0 = fmax3(m0, sv[i + 0], sv[i + 1]);
        m1 = fmax3(m1, sv[i + 2], sv[i + 3]);
        m2 = fmax3(m2, sv[i + 4], sv[i + 5]);
        m3 = fmax3(m3, sv[i + 6], sv[i + 7]);
      }
      block_max = fmax2(fmax2(m0, m1), fmax2(m2, m3));
    }

    float new_max = fmax2(rmax_local, block_max);
    if (new_max == -FLT_MAX) new_max = 0.f;
    float rs = 1.0f;
    if (kb == 0) {
      rmax_local = new_max;
    } else {
      float acc_scale_log2 = (rmax_local - new_max) * sc_log2;
      if constexpr (kRescaleThreshold > 0.0f) {
        if (acc_scale_log2 < -kRescaleThreshold) {
          rs = fast_exp2f(acc_scale_log2);
          rmax_local = new_max;
        }
      } else {
        rs = fast_exp2f(acc_scale_log2);
        rmax_local = new_max;
      }
    }
    rsum_local *= rs;

    rs_smem[(tkb & 1) * kQStages * kBlockMCTA + qs * kBlockMCTA + sm_tid] = rs;
    bar_arrive(sm_bar, 64);

    // Pass 2: scale to log2 domain and subtract row max
    float neg_max_scaled = -(rmax_local * sc_log2);
#pragma unroll
    for (int cc = 0; cc < kBlockN; cc += 16) {
#pragma unroll
      for (int i = 0; i < 16; i += 2) {
        FMA_F32X2(sv[cc + i], sv[cc + i + 1], sv[cc + i], sv[cc + i + 1], sc_log2, sc_log2, neg_max_scaled, neg_max_scaled);
      }
    }

    // Pass 3: exp2 + bf16 + TMEM P
    float psa[4] = {0.f, 0.f, 0.f, 0.f};
    constexpr int kSplitReadyCc = 80;
#pragma unroll
    for (int cc = 0; cc < kBlockN; cc += 16) {
#pragma unroll
      for (int g = 0; g <= 8; g += 8) {
        __nv_bfloat162 h4[4];
        float f8[8];
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          int elem = cc + g + i;
          float p0, p1;
          int frag = elem / kEx2FragSize;
          int k_in_frag = elem % kEx2FragSize;
          if (kEx2EmuFreq > 0 && frag >= kEx2EmuStartFrg && frag < (kBlockN / kEx2FragSize - 1) &&
              (k_in_frag % kEx2EmuFreq) >= (kEx2EmuFreq - kEx2EmuRes)) {
            float2 p = e2e_asm2(sv[elem], sv[elem + 1]);
            p0 = p.x;
            p1 = p.y;
          } else {
            p0 = fast_exp2f(sv[elem]);
            p1 = fast_exp2f(sv[elem + 1]);
          }
          psa[i >> 1] += p0 + p1;
          f8[i] = p0;
          f8[i + 1] = p1;
        }
        float22bfloat162_xN<4>(h4, f8);
        tcgen05_st_32x32b_x4((tbase + 64 + qs * 128) + tr + (cc + g) / 2, *reinterpret_cast<uint32_t*>(&h4[0]),
                             *reinterpret_cast<uint32_t*>(&h4[1]), *reinterpret_cast<uint32_t*>(&h4[2]),
                             *reinterpret_cast<uint32_t*>(&h4[3]));
      }
      if constexpr (kEnableEarlyPSplit) {
        if (cc == kSplitReadyCc) {
          fence_view_async_tmem_store();
          mb_p[qs].arrive();
        }
      }
    }
    fence_view_async_tmem_store();
    if constexpr (kEnableEarlyPSplit) {
      mb_p2[qs].arrive();
    } else {
      mb_p[qs].arrive();
      mb_p2[qs].arrive();
    }
    rsum_local += (psa[0] + psa[1]) + (psa[2] + psa[3]);
  }

  sum_smem[qs * kBlockMCTA + sm_tid] = rsum_local;
  bar_arrive(sm_bar, 64);
}

// Correction: ballot-before-wait, 3-stage LD pipeline, packed f32x2 multiply
__device__ __forceinline__ void correction_rescale_o_fn(uint32_t O_base, float rs) {
  constexpr int kChunks = kHeadDim / 16;
  float buf[2][16];
  int cur = 0;
  tcgen05_ld_16(buf[cur], O_base);
  for (int g = 0; g < kChunks; g++) {
    tcgen05_wait_ld_sync();
    int nxt = cur ^ 1;
    if (g + 1 < kChunks) tcgen05_ld_16(buf[nxt], O_base + (g + 1) * 16);
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      FMA_F32X2(buf[cur][i], buf[cur][i + 1], buf[cur][i], buf[cur][i + 1], rs, rs, 0.0f, 0.0f);
    }
    tcgen05_st_16(buf[cur], O_base + g * 16);
    cur = nxt;
  }
  tcgen05_wait_st_sync();
}

__device__ __forceinline__ void correction_finalize_o_fn(char *epi_base, uint32_t O_base, int lane, float inv) {
  constexpr int kEpiBlockCols = 64;
  constexpr int kEpiBlockBytes = kEpiBlockCols * sizeof(__nv_bfloat16);
  constexpr int kEpiBlockSize = 32 * kEpiBlockCols;
  char *epi_blk0 = epi_base;
  char *epi_blk1 = epi_base + kEpiBlockSize * sizeof(__nv_bfloat16);
  int row_off = lane * kEpiBlockBytes;
  int swiz = (lane & 7) << 4;

  for (int d = 0; d < kHeadDim; d += 16) {
    float t[16];
    tcgen05_ld_16(t, O_base + d);
    tcgen05_wait_ld_sync();
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      FMA_F32X2(t[i], t[i + 1], t[i], t[i + 1], inv, inv, 0.0f, 0.0f);
    }

    __nv_bfloat16 b[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) b[i] = __float2bfloat16(t[i]);

    int d_in_blk = d % kEpiBlockCols;
    char *blk = d < kEpiBlockCols ? epi_blk0 : epi_blk1;
    int col0 = d_in_blk * sizeof(__nv_bfloat16);
    int col1 = (d_in_blk + 8) * sizeof(__nv_bfloat16);
    *reinterpret_cast<float4 *>(blk + row_off + (col0 ^ swiz)) = *reinterpret_cast<float4 *>(&b[0]);
    *reinterpret_cast<float4 *>(blk + row_off + (col1 ^ swiz)) = *reinterpret_cast<float4 *>(&b[8]);
  }
}

__device__ __noinline__ void correction_warp_fn(
    int tid, int cr, int w,
    half *sO,
    float *rs_smem, float *sum_smem,
    Mbarrier *mb_corr, Mbarrier *mb_pv, Mbarrier *mb_epi,
    Mbarrier *mb_o_rel, Mbarrier *mb_o_tmem_rel,
    uint32_t tbase, int nkb, int kb_base, int tile_phase) {
  const uint32_t tr = (uint32_t)((cr * 128 + (w % 4) * 32) << 16);
  int corr_tid = tid - 256;
  const int sm_bar_base = kBarSoftmaxStatsBase + (w & 3);
#pragma unroll
  for (int kb = 0; kb < nkb; kb++) {
    int tkb = kb_base + kb;
    for (int qs = 0; qs < kQStages; qs++) {
      bar_sync(sm_bar_base + qs * 4, 64);
      float rs = rs_smem[(tkb & 1) * kQStages * kBlockMCTA + qs * kBlockMCTA + corr_tid];
      if (kb > 0) {
        unsigned int needs_rescale = __ballot_sync(0xFFFFFFFF, rs < 1.0f);
        if (needs_rescale) {
          int prev = tkb - 1;
          int pv_stage = prev % kKVStages;
          int pv_wait = (prev / kKVStages) & 1;
          mb_pv[pv_stage].wait(pv_wait);
          tcgen05_after_thread_sync();
          correction_rescale_o_fn((tbase + 256 + qs * 128) + tr, rs);
          fence_view_async_tmem_store();
        }
      }
      mb_corr[qs].arrive();
    }
  }

  int last_tkb = kb_base + nkb - 1;
  int last_v_stage = last_tkb % kKVStages;
  int last_v_phase = (last_tkb / kKVStages) & 1;
  int lane = tid & 31;
  int warp_in_corr = w & 3;

  if (kb_base > 0) {
    mb_o_rel->wait(tile_phase ^ 1);
  }

  for (int qs = 0; qs < kQStages; qs++) {
    mb_pv[last_v_stage].wait(last_v_phase);
    tcgen05_after_thread_sync();
    bar_sync(sm_bar_base + qs * 4, 64);

    float inv = 0.f;
    float rsum = sum_smem[qs * kBlockMCTA + corr_tid];
    if (rsum > 0.f) inv = 1.0f / rsum;

    uint32_t O_base = (tbase + 256 + qs * 128) + tr;
    // Write finalized O into the dedicated sO staging buffer (FA4-aligned).
    // Decoupling from sQ lets producer reload Q for the next persistent tile
    // while epilogue's TMA store for the current tile is still in flight.
    char *epi_base = reinterpret_cast<char *>(sO + qs * kQStageHalfs + warp_in_corr * 32 * kHeadDim);
    correction_finalize_o_fn(epi_base, O_base, lane, inv);

    fence_proxy_async();
    mb_epi[qs].arrive_local();
  }

  mb_o_tmem_rel->arrive();
}

template <int kSeqLen, int kNumQHeads>
__device__ __noinline__ void epilogue_warp_fn(
    half *sO,
    const CUtensorMap *tmap_O,
    Mbarrier *mb_epi,
    Mbarrier *mb_o_rel,
    int ms, int bi, int hi, int tile_phase) {
  if (!elect_one_sync()) return;
  constexpr int kEpiBlockCols = 64;
  constexpr int kEpiBlockSize = 32 * kEpiBlockCols;

  // BSHD: row coord = b*S + s, col coord = hi*D + {0, 64}
  const int col_base = hi * kHeadDim;
#pragma unroll
  for (int qs = 0; qs < kQStages; qs++) {
    mb_epi[qs].wait(tile_phase);

    for (int cw = 0; cw < 4; cw++) {
      int row_base = ms + qs * kBlockM + cw * 32;
      if (row_base >= kSeqLen) break;

      char *epi_base = reinterpret_cast<char *>(sO + qs * kQStageHalfs + cw * 32 * kHeadDim);
      char *epi_blk0 = epi_base;
      char *epi_blk1 = epi_base + kEpiBlockSize * sizeof(__nv_bfloat16);
      int coord_y = bi * kSeqLen + row_base;
      cp_async_bulk_tensor_2d_global_shared(tmap_O, epi_blk0, col_base, coord_y);
      cp_async_bulk_tensor_2d_global_shared(tmap_O, epi_blk1, col_base + 64, coord_y);
    }
    cp_async_bulk_commit_group();
  }
  cp_async_bulk_wait_group<0>();
  mb_o_rel->arrive_local();
}

// MMA: FA4-style ownership handoff.
// `mb_p` is the early partial-P + prior-O-rescaled acquire point; `mb_p2` is full-P ready.
// K and V use separate SMEM; consumer release via tcgen05_commit (cluster-wide)
__device__ __noinline__ void mma_warp_fn(int cr, half *sQ, half *sK, half *sV,
                                         Mbarrier *mb_q, Mbarrier *mb_k, Mbarrier *mb_s,
                                         Mbarrier *mb_p, Mbarrier *mb_p2, Mbarrier *mb_v,
                                         Mbarrier *mb_pv, Mbarrier *mb_corr,
                                         Mbarrier *mb_k_rel, Mbarrier *mb_v_rel,
                                         Mbarrier *mb_q_rel, Mbarrier *mb_o_tmem_rel,
                                         uint32_t tbase, int nkb, int q_phase, int kb_base) {
  if (cr != 0 || elect_one_sync() == 0) return;

  const uint32_t sQ_lo = static_cast<uint32_t>((cvta_u64(sQ) & 0x3FFFF) >> 4);
  const uint32_t sK_lo = static_cast<uint32_t>((cvta_u64(sK) & 0x3FFFF) >> 4);
  const uint32_t sV_lo = static_cast<uint32_t>((cvta_u64(sV) & 0x3FFFF) >> 4);

  for (int qs = 0; qs < kQStages; qs++) mb_q[qs].wait(q_phase);
  tcgen05_after_thread_sync();

  // Pre-loop: QK GEMM with K[0]
  int kv0_stage = kb_base % kKVStages;
  int kv0_phase = (kb_base / kKVStages) & 1;
  mb_k[kv0_stage].wait(kv0_phase);
  tcgen05_after_thread_sync();
  for (int qs = 0; qs < kQStages; qs++) {
    int f = 1;
    uint32_t q_off_base = qs * kQStageHalfs * sizeof(half);
    uint32_t k_off_base = kv0_stage * kPerStageKV * sizeof(half);
    for (int t = 0; t < 2; t++) {
      uint32_t qo = q_off_base + t * kBlockMCTA * kTileCols * sizeof(half);
      uint32_t ko = k_off_base + t * kBPerCTA * kTileCols * sizeof(half);
      for (int j = 0; j < kTileCols; j += 16) {
        tcgen05_mma_256x128((tbase + qs * 128), mk_fast(sQ_lo, qo + j * sizeof(half)),
                            mk_fast(sK_lo, ko + j * sizeof(half)), f ? 0u : 1u);
        f = 0;
      }
    }
    tcgen05_commit(&mb_s[qs]);
  }
  // K[0] SMEM captured by tcgen05_commit → slot 0 free for producer
  tcgen05_commit(&mb_k_rel[kv0_stage]);

  if (nkb > 0) {
    if (kb_base > 0) {
      mb_o_tmem_rel->wait(q_phase ^ 1);
      tcgen05_after_thread_sync();
    }

    int tkb = kb_base;
    int v_stage = tkb % kKVStages;
    int v_phase = (tkb / kKVStages) & 1;

    for (int qs = 0; qs < kQStages; qs++) {
      mb_p[qs].wait(tkb & 1);
      mb_v[v_stage].wait(v_phase);
      tcgen05_after_thread_sync();

      uint32_t O_addr = tbase + 256 + qs * 128;
      uint32_t P_base = tbase + 64 + qs * 128;

      uint32_t v_base = v_stage * kPerStageKV * sizeof(half);
      uint32_t v_hi = v_base + kBPerCTA * kTileCols * sizeof(half);
      if constexpr (kEnableEarlyPSplit) {
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 0,  mk_fast(sV_lo, v_base +  0 * kTileCols * sizeof(half)), 0u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 8,  mk_fast(sV_lo, v_base + 16 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 16, mk_fast(sV_lo, v_base + 32 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 24, mk_fast(sV_lo, v_base + 48 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 32, mk_fast(sV_lo, v_hi +  0 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 40, mk_fast(sV_lo, v_hi + 16 * kTileCols * sizeof(half)), 1u);

        mb_p2[qs].wait(tkb & 1);
        tcgen05_after_thread_sync();
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 48, mk_fast(sV_lo, v_hi + 32 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 56, mk_fast(sV_lo, v_hi + 48 * kTileCols * sizeof(half)), 1u);
      } else {
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 0,  mk_fast(sV_lo, v_base +  0 * kTileCols * sizeof(half)), 0u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 8,  mk_fast(sV_lo, v_base + 16 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 16, mk_fast(sV_lo, v_base + 32 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 24, mk_fast(sV_lo, v_base + 48 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 32, mk_fast(sV_lo, v_hi +  0 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 40, mk_fast(sV_lo, v_hi + 16 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 48, mk_fast(sV_lo, v_hi + 32 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 56, mk_fast(sV_lo, v_hi + 48 * kTileCols * sizeof(half)), 1u);
      }

      if (qs == kQStages - 1) {
        tcgen05_commit(&mb_pv[v_stage]);
        tcgen05_commit(&mb_v_rel[v_stage]);
      }

      if (1 < nkb) {
        int next_k_stage = (tkb + 1) % kKVStages;
        int next_k_phase = ((tkb + 1) / kKVStages) & 1;
        if (qs == 0) {
          mb_k[next_k_stage].wait(next_k_phase);
          tcgen05_after_thread_sync();
        }
        uint32_t nq_off_base = qs * kQStageHalfs * sizeof(half);
        uint32_t nk_off_base = next_k_stage * kPerStageKV * sizeof(half);
        int f = 1;
        for (int t = 0; t < 2; t++) {
          uint32_t nqo = nq_off_base + t * kBlockMCTA * kTileCols * sizeof(half);
          uint32_t nko = nk_off_base + t * kBPerCTA * kTileCols * sizeof(half);
          for (int j = 0; j < kTileCols; j += 16) {
            tcgen05_mma_256x128((tbase + qs * 128), mk_fast(sQ_lo, nqo + j * sizeof(half)),
                                mk_fast(sK_lo, nko + j * sizeof(half)), f ? 0u : 1u);
            f = 0;
          }
        }
        tcgen05_commit(&mb_s[qs]);
        if (qs == kQStages - 1) {
          tcgen05_commit(&mb_k_rel[next_k_stage]);
        }
      }
    }
  }

  for (int kb = 1; kb < nkb; kb++) {
    int tkb = kb_base + kb;
    int v_stage = tkb % kKVStages;
    int v_phase = (tkb / kKVStages) & 1;

    for (int qs = 0; qs < kQStages; qs++) {
      mb_p[qs].wait(tkb & 1);
      mb_v[v_stage].wait(v_phase);
      mb_corr[qs].wait(tkb & 1);
      tcgen05_after_thread_sync();

      uint32_t O_addr = tbase + 256 + qs * 128;
      uint32_t P_base = tbase + 64 + qs * 128;

      uint32_t v_base = v_stage * kPerStageKV * sizeof(half);
      uint32_t v_hi = v_base + kBPerCTA * kTileCols * sizeof(half);
      if constexpr (kEnableEarlyPSplit) {
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 0,  mk_fast(sV_lo, v_base +  0 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 8,  mk_fast(sV_lo, v_base + 16 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 16, mk_fast(sV_lo, v_base + 32 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 24, mk_fast(sV_lo, v_base + 48 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 32, mk_fast(sV_lo, v_hi +  0 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 40, mk_fast(sV_lo, v_hi + 16 * kTileCols * sizeof(half)), 1u);

        mb_p2[qs].wait(tkb & 1);
        tcgen05_after_thread_sync();
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 48, mk_fast(sV_lo, v_hi + 32 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 56, mk_fast(sV_lo, v_hi + 48 * kTileCols * sizeof(half)), 1u);
      } else {
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 0,  mk_fast(sV_lo, v_base +  0 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 8,  mk_fast(sV_lo, v_base + 16 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 16, mk_fast(sV_lo, v_base + 32 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 24, mk_fast(sV_lo, v_base + 48 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 32, mk_fast(sV_lo, v_hi +  0 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 40, mk_fast(sV_lo, v_hi + 16 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 48, mk_fast(sV_lo, v_hi + 32 * kTileCols * sizeof(half)), 1u);
        tcgen05_mma_ts_256x128_bmn(O_addr, P_base + 56, mk_fast(sV_lo, v_hi + 48 * kTileCols * sizeof(half)), 1u);
      }

      if (qs == kQStages - 1) {
        tcgen05_commit(&mb_pv[v_stage]);
        tcgen05_commit(&mb_v_rel[v_stage]);
      }

      if (kb + 1 < nkb) {
        int next_k_stage = (tkb + 1) % kKVStages;
        int next_k_phase = ((tkb + 1) / kKVStages) & 1;
        if (qs == 0) {
          mb_k[next_k_stage].wait(next_k_phase);
          tcgen05_after_thread_sync();
        }
        uint32_t nq_off_base = qs * kQStageHalfs * sizeof(half);
        uint32_t nk_off_base = next_k_stage * kPerStageKV * sizeof(half);
        int f = 1;
        for (int t = 0; t < 2; t++) {
          uint32_t nqo = nq_off_base + t * kBlockMCTA * kTileCols * sizeof(half);
          uint32_t nko = nk_off_base + t * kBPerCTA * kTileCols * sizeof(half);
          for (int j = 0; j < kTileCols; j += 16) {
            tcgen05_mma_256x128((tbase + qs * 128), mk_fast(sQ_lo, nqo + j * sizeof(half)),
                                mk_fast(sK_lo, nko + j * sizeof(half)), f ? 0u : 1u);
            f = 0;
          }
        }
        tcgen05_commit(&mb_s[qs]);
        if (qs == kQStages - 1) {
          tcgen05_commit(&mb_k_rel[next_k_stage]);
        }
      }
    }
  }

  tcgen05_commit(mb_q_rel);
}

// Producer warp role: TMA loads for Q, K, V (K and V in separate SMEM, both preloaded)
template <int kSeqLen, int kNumQHeads, int kNumKVHeads>
__device__ __noinline__ void producer_warp_fn(
    int cr,
    half *sQ, half *sK, half *sV,
    const CUtensorMap *tmap_Q, const CUtensorMap *tmap_K, const CUtensorMap *tmap_VT,
    Mbarrier *mb_q, Mbarrier *mb_k, Mbarrier *mb_v,
    Mbarrier *mb_k_rel, Mbarrier *mb_v_rel,
    Mbarrier *mb_q_rel,
    int ms, int nkb, int bi, int hi, int kvh,
    int kb_base, int tile_phase)
{
    if(elect_one_sync() == 0) return;

    // BSHD layout: row coord = b*S + seq_pos, col coord = head_idx*D + d_offset
    // Load Q
    if (kb_base > 0) {
        mb_q_rel->wait(tile_phase ^ 1);
    }
    int q_bytes=2*kBlockMCTA*kTileCols*sizeof(half);
    int q_row_base=bi*kSeqLen+ms;
    int q_col_base=hi*kHeadDim;
    for(int qs=0;qs<kQStages;qs++){
        int q_row=q_row_base+qs*kBlockM;
        mb_q[qs].arrive_expect_tx(q_bytes);
        for(int t=0;t<2;t++)
            cp_async_bulk_tensor_2d_shared_global((void *)(sQ + qs * kQStageHalfs + t * kBlockMCTA * kTileCols), tmap_Q,
                                                    q_row, q_col_base + t * kTileCols, &mb_q[qs]);
    }

    int kv_bytes=2*kBPerCTA*kTileCols*sizeof(half);
    int kv_row_base=bi*kSeqLen;
    int kv_col_base=kvh*kHeadDim;
    int v_col=kv_col_base+cr*kBPerCTA;

    // Pre-load K[0],V[0],K[1],V[1],... (FA4-style: producer issues K and V interleaved).
    // The TMA engine processes them in parallel; this slightly shortens the time to
    // first-V-ready relative to issuing all K's then all V's.
    for(int s=0;s<min(nkb,kKVStages);s++){
        int tkb_s = kb_base + s;
        int kvs = tkb_s % kKVStages;
        if (tkb_s >= kKVStages) {
            int prev_phase = ((tkb_s - kKVStages) / kKVStages) & 1;
            mb_k_rel[kvs].wait(prev_phase);
            mb_v_rel[kvs].wait(prev_phase);
        }
        int k_row=kv_row_base+s*kBlockN+cr*kBPerCTA;
        mb_k[kvs].arrive_expect_tx(kv_bytes);
        for(int t=0;t<2;t++)
            cp_async_bulk_tensor_2d_shared_global((void *)(sK + kvs * kPerStageKV + t * kBPerCTA * kTileCols), tmap_K,
                                                    k_row, kv_col_base + t * kTileCols, &mb_k[kvs]);
        int v_row=kv_row_base+s*kBlockN;
        mb_v[kvs].arrive_expect_tx(kv_bytes);
        for(int t=0;t<2;t++)
            cp_async_bulk_tensor_2d_shared_global((void *)(sV + kvs * kPerStageKV + t * kBPerCTA * kTileCols), tmap_VT,
                                                    v_row + t * kBPerCTA, v_col, &mb_v[kvs]);
    }

    // Main loop: reload K and V using consumer release barriers
    for(int kb=0;kb<nkb;kb++){
        int tkb = kb_base + kb;
        int k_stage=tkb%kKVStages;
        int k_phase=(tkb/kKVStages)&1;
        int v_stage=tkb%kKVStages;
        int v_phase=(tkb/kKVStages)&1;

        // Reload K[kb+kKVStages] after MMA consumed K[kb]
        if(kb+kKVStages<nkb){
            mb_k_rel[k_stage].wait(k_phase);
            int next_k=kb+kKVStages;
            int next_kvs=(tkb+kKVStages)%kKVStages;
            int k_row=kv_row_base+next_k*kBlockN+cr*kBPerCTA;
            mb_k[next_kvs].arrive_expect_tx(kv_bytes);
            for(int t=0;t<2;t++)
                cp_async_bulk_tensor_2d_shared_global((void *)(sK + next_kvs * kPerStageKV + t * kBPerCTA * kTileCols), tmap_K,
                                                        k_row, kv_col_base + t * kTileCols, &mb_k[next_kvs]);
        }

        // Reload V[kb+kKVStages] after MMA consumed V[kb]
        if(kb+kKVStages<nkb){
            mb_v_rel[v_stage].wait(v_phase);
            int next_v=kb+kKVStages;
            int next_kvs=(tkb+kKVStages)%kKVStages;
            int v_row=kv_row_base+next_v*kBlockN;
            mb_v[next_kvs].arrive_expect_tx(kv_bytes);
            for(int t=0;t<2;t++)
                cp_async_bulk_tensor_2d_shared_global((void *)(sV + next_kvs * kPerStageKV + t * kBPerCTA * kTileCols), tmap_VT,
                                                        v_row + t * kBPerCTA, v_col, &mb_v[next_kvs]);
        }
    }
}

// ========= Kernel =========
template <
    int kBatchSize,
    int kSeqLen,
    int kNumQHeads,
    int kNumKVHeads,
    int kHeadDimParam,
    int kIsCausal>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(kThreads, 1)
void fmha_fa4_kernel(
    const __grid_constant__ CUtensorMap tmap_Q,
    const __grid_constant__ CUtensorMap tmap_K,
    const __grid_constant__ CUtensorMap tmap_VT,
    const __grid_constant__ CUtensorMap tmap_O)
{
    static_assert(kHeadDimParam == kHeadDim, "fa4_uma currently supports only kHeadDim");
    static_assert(kIsCausal == 0, "causal attention is not implemented in fa4_uma");
    constexpr int kNumMBlocks = (kSeqLen + kQStages * kBlockM - 1) / (kQStages * kBlockM);
    constexpr float kSoftmaxScale = 0.08838834764831845f;  // 1 / sqrt(128)

    extern __shared__ __align__(1024) uint8_t smem[];
    constexpr int kAlign=1024;
    constexpr int oQ=0;
    constexpr int oO=((oQ+kSmemQ+kAlign-1)/kAlign)*kAlign;
    constexpr int oKV=((oO+kSmemO+kAlign-1)/kAlign)*kAlign;
    half *sQ=(half*)(smem+oQ);
    half *sO=(half*)(smem+oO);
    half *sK=(half*)(smem+oKV);
    half *sV=(half*)(smem+oKV+kSmemK);
    // rs_smem [2*kQStages*kBlockMCTA floats = 2KB] and sum_smem [kQStages*kBlockMCTA = 1KB]
    // overlap: rs is used per-kb in the softmax/correction loops; once those finish,
    // softmax writes sum_smem to the FRONT of the rs region (size fits since sum <= rs).
    // Net SMEM saved: 1KB — needed to fit under the 227KiB per-block cap with separate sO.
    float *rs_smem=(float*)(smem+oKV+kSmemKV);
    float *sum_smem=rs_smem;
    uint8_t *it=smem+oKV+kSmemKV+2*kQStages*kBlockMCTA*sizeof(float);
    it = (uint8_t*)(((uintptr_t)it + 7) & ~7);
    Mbarrier *mb_q=(Mbarrier*)it;it+=8*kQStages;
    Mbarrier *mb_k=(Mbarrier*)it;it+=8*kKVStages;
    Mbarrier *mb_s=(Mbarrier*)it;it+=8*kQStages;
    Mbarrier *mb_p=(Mbarrier*)it;it+=8*kQStages;
    Mbarrier *mb_p2=(Mbarrier*)it;it+=8*kQStages;
    Mbarrier *mb_v=(Mbarrier*)it;it+=8*kKVStages;
    Mbarrier *mb_k_rel=(Mbarrier*)it;it+=8*kKVStages;
    Mbarrier *mb_v_rel=(Mbarrier*)it;it+=8*kKVStages;
    Mbarrier *mb_pv=(Mbarrier*)it;it+=8*kKVStages;
    Mbarrier *mb_corr=(Mbarrier*)it;it+=8*kQStages;
    Mbarrier *mb_epi=(Mbarrier*)it;it+=8*kQStages;
    Mbarrier *mb_q_rel=(Mbarrier*)it;it+=8;
    Mbarrier *mb_o_rel=(Mbarrier*)it;it+=8;
    Mbarrier *mb_o_tmem_rel=(Mbarrier*)it;it+=8;
    uint32_t *tptr=(uint32_t*)it;it+=4;

    const int cr=cluster_ctarank(),w=threadIdx.x/32,cl=clusterid_x(),tid=threadIdx.x;
    const int wg=w/4;

    if (tid == 0) {
        prefetch_tma_descriptor(&tmap_Q);
        prefetch_tma_descriptor(&tmap_K);
        prefetch_tma_descriptor(&tmap_VT);
        prefetch_tma_descriptor(&tmap_O);
    }

    // Mbarrier init
    if(tid==kTidProd){
        for(int s=0;s<kQStages;s++){
            mb_q[s].init(2); mb_s[s].init(1);
            mb_p[s].init(256); mb_p2[s].init(256);
            mb_corr[s].init(256); mb_epi[s].init(kBlockMCTA);
        }
        for(int i=0;i<kKVStages;i++){
            mb_k[i].init(2);
            mb_v[i].init(2);
            mb_k_rel[i].init(1);
            mb_v_rel[i].init(1);
            mb_pv[i].init(1);
        }
        mb_q_rel->init(1);
        mb_o_rel->init(1);
        mb_o_tmem_rel->init(256);
        fence_mbarrier_init();
    }
    if(w==kWarpMMA) tcgen05_alloc_sync(tptr,2048);
    barrier_cluster_sync();
    constexpr int total_tiles = kNumMBlocks * kNumQHeads * kBatchSize;
    const int grid_clusters = gridDim.x / 2;
    constexpr int nkb=(kSeqLen+kBlockN-1)/kBlockN;
    const uint32_t tbase=tptr[0];

    // ============ PERSISTENT TILE LOOP ============
    for(int iter=0;;iter++){
        int cluster_tile = iter * grid_clusters + cl;
        if(cluster_tile >= total_tiles) break;

        const int mb_idx = cluster_tile % kNumMBlocks;
        const int hi = (cluster_tile / kNumMBlocks) % kNumQHeads;
        const int bi = cluster_tile / (kNumMBlocks * kNumQHeads);
        const int ms = mb_idx * kQStages * kBlockM + cr * kBlockMCTA;
        const int kvh = hi * kNumKVHeads / kNumQHeads;
        const int q_phase = iter & 1;
        const int kb_base = iter * nkb;

        if((wg >> 1) == 0){
            setmaxnreg_inc<kRegsSoftmax>();
            softmax_warp_fn(wg, tid, cr, w, rs_smem, sum_smem,
                            mb_s, mb_p, mb_p2,
                            tbase, nkb, kSeqLen, kSoftmaxScale, kb_base);
        } else if(wg==2){
            setmaxnreg_dec<kRegsCorrection>();
            correction_warp_fn(tid, cr, w, sO, rs_smem, sum_smem,
                               mb_corr, mb_pv, mb_epi,
                               mb_o_rel, mb_o_tmem_rel,
                               tbase, nkb, kb_base, q_phase);
        } else if(w==kWarpMMA){
            setmaxnreg_dec<kRegsOther>();
            mma_warp_fn(cr, sQ, sK, sV,
                        mb_q, mb_k, mb_s, mb_p, mb_p2, mb_v,
                        mb_pv, mb_corr, mb_k_rel, mb_v_rel,
                        mb_q_rel, mb_o_tmem_rel,
                        tbase, nkb, q_phase, kb_base);
        } else if(w==kWarpProd){
            setmaxnreg_dec<kRegsOther>();
            producer_warp_fn<kSeqLen, kNumQHeads, kNumKVHeads>(
                             cr, sQ, sK, sV, &tmap_Q, &tmap_K, &tmap_VT,
                             mb_q, mb_k, mb_v,
                             mb_k_rel, mb_v_rel,
                             mb_q_rel,
                             ms, nkb, bi, hi, kvh,
                             kb_base, q_phase);
        } else if(w==kWarpEpilogue){
            setmaxnreg_dec<kRegsOther>();
            epilogue_warp_fn<kSeqLen, kNumQHeads>(sO, &tmap_O, mb_epi, mb_o_rel, ms, bi, hi, q_phase);
        } else {
            // Empty/donor warp 15: decrease register budget, do nothing
            setmaxnreg_dec<kRegsOther>();
        }

    }

    // barrier_cluster_sync();
    if(w==kWarpMMA) tcgen05_dealloc_sync(tptr[0], 2048);
}

// ========= TMA tensor map creation =========
static CUtensorMap create_tmap_bf16(const __nv_bfloat16 *ptr, int inner_dim, int outer_dim, int stride_elems,
                        int box_cols, int box_rows) {
    CUtensorMap tmap;
    uint64_t gdim[2] = {(uint64_t)inner_dim, (uint64_t)outer_dim};
    uint64_t gstr[1] = {(uint64_t)stride_elems * sizeof(__nv_bfloat16)};
    uint32_t bdim[2] = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t estr[2] = {1, 1};
    cuTensorMapEncodeTiled(&tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void*)ptr, gdim, gstr, bdim, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    return tmap;
}

// ========= Template Interface =========

template <
    int kBatchSize,
    int kSeqLen,
    int kNumQHeads,
    int kNumKVHeads,
    int kHeadDimParam,
    int kIsCausal>
void flash_attention_forward(
    const void* Q, const void* K, const void* V, void* O,
    cudaStream_t stream
) {
    static_assert(kHeadDimParam == kHeadDim, "fa4_uma currently supports only kHeadDim");
    static_assert(kIsCausal == 0, "causal attention is not implemented in fa4_uma");
    const __nv_bfloat16 *bf16_Q = (const __nv_bfloat16*)Q;
    const __nv_bfloat16 *bf16_K = (const __nv_bfloat16*)K;
    const __nv_bfloat16 *bf16_V = (const __nv_bfloat16*)V;

    constexpr int kEffectiveBlockM = kQStages * kBlockM;
    constexpr int kNumMBlocks = (kSeqLen + kEffectiveBlockM - 1) / kEffectiveBlockM;

    // BSHD layout: [B, S, H, D] flattened to 2D as [B*S rows, H*D cols]
    // inner_dim = num_heads * head_dim, outer_dim = batch * seqlen, stride = num_heads * head_dim
    constexpr int kQoInner = kNumQHeads * kHeadDimParam;
    constexpr int kKvInner = kNumKVHeads * kHeadDimParam;
    constexpr int kOuterDim = kBatchSize * kSeqLen;
    CUtensorMap tmap_Q = create_tmap_bf16(bf16_Q, kQoInner, kOuterDim, kQoInner, 64, kBlockMCTA);
    CUtensorMap tmap_K = create_tmap_bf16(bf16_K, kKvInner, kOuterDim, kKvInner, 64, kBPerCTA);
    CUtensorMap tmap_VT = create_tmap_bf16(bf16_V, kKvInner, kOuterDim, kKvInner, 64, kBPerCTA);
    CUtensorMap tmap_O = create_tmap_bf16((const __nv_bfloat16*)O, kQoInner, kOuterDim, kQoInner, 64, 32);

    constexpr int kA = 1024;
    constexpr int _oO = ((kSmemQ + kA - 1) / kA) * kA;
    constexpr int _oKV = ((_oO + kSmemO + kA - 1) / kA) * kA;
    int smem_size = _oKV + kSmemKV + 2*kQStages*kBlockMCTA*sizeof(float)
                  + sizeof(Mbarrier) * (8*kQStages + 5*kKVStages + 3) + 8 + 64;

    auto kernel = fmha_fa4_kernel<kBatchSize, kSeqLen, kNumQHeads, kNumKVHeads, kHeadDimParam, kIsCausal>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    constexpr int tiles = kNumMBlocks * kNumQHeads * kBatchSize;
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int grid = min(tiles * 2, sm_count);
    grid = (grid / 2) * 2;
    if (grid < 2) grid = 2;

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = {(unsigned)grid, 1u, 1u};
    cfg.blockDim = {(unsigned)kThreads, 1, 1};
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    cfg.numAttrs = 1;
    cfg.attrs = attrs;

    cudaLaunchKernelEx(&cfg, kernel, tmap_Q, tmap_K, tmap_VT, tmap_O);
}

namespace {

template <int kSeqLen, int kNumQHeads, int kNumKVHeads>
void flash_attention_forward_dispatch_batch(
    const void* Q, const void* K, const void* V, void* O,
    int batch_size, cudaStream_t stream) {
  switch (batch_size) {
    case 1:
      flash_attention_forward<1, kSeqLen, kNumQHeads, kNumKVHeads, 128, 0>(Q, K, V, O, stream);
      break;
    case 2:
      flash_attention_forward<2, kSeqLen, kNumQHeads, kNumKVHeads, 128, 0>(Q, K, V, O, stream);
      break;
    case 4:
      flash_attention_forward<4, kSeqLen, kNumQHeads, kNumKVHeads, 128, 0>(Q, K, V, O, stream);
      break;
    case 8:
      flash_attention_forward<8, kSeqLen, kNumQHeads, kNumKVHeads, 128, 0>(Q, K, V, O, stream);
      break;
    default:
      fprintf(stderr, "Unsupported batch size %d. Supported template instantiations: 1, 2, 4, 8\n", batch_size);
      exit(EXIT_FAILURE);
  }
}

template <int kSeqLen>
void flash_attention_forward_dispatch_heads(
    const void* Q, const void* K, const void* V, void* O,
    int batch_size, int num_q_heads, int num_kv_heads, cudaStream_t stream) {
  if (num_q_heads == 16 && num_kv_heads == 16) {
    flash_attention_forward_dispatch_batch<kSeqLen, 16, 16>(Q, K, V, O, batch_size, stream);
  } else if (num_q_heads == 40 && num_kv_heads == 8) {
    flash_attention_forward_dispatch_batch<kSeqLen, 40, 8>(Q, K, V, O, batch_size, stream);
  } else if (num_q_heads == 48 && num_kv_heads == 8) {
    flash_attention_forward_dispatch_batch<kSeqLen, 48, 8>(Q, K, V, O, batch_size, stream);
  } else {
    fprintf(stderr,
            "Unsupported (num_q_heads, num_kv_heads) = (%d, %d). "
            "Supported template instantiations: (16,16), (40,8), (48,8)\n",
            num_q_heads, num_kv_heads);
    exit(EXIT_FAILURE);
  }
}

void flash_attention_forward_dispatch(
    const void* Q, const void* K, const void* V, void* O,
    int batch_size, int seq_len, int num_q_heads, int num_kv_heads,
    cudaStream_t stream) {
  switch (seq_len) {
    case 512:
      flash_attention_forward_dispatch_heads<512>(Q, K, V, O, batch_size, num_q_heads, num_kv_heads, stream);
      break;
    case 1024:
      flash_attention_forward_dispatch_heads<1024>(Q, K, V, O, batch_size, num_q_heads, num_kv_heads, stream);
      break;
    case 2048:
      flash_attention_forward_dispatch_heads<2048>(Q, K, V, O, batch_size, num_q_heads, num_kv_heads, stream);
      break;
    case 4096:
      flash_attention_forward_dispatch_heads<4096>(Q, K, V, O, batch_size, num_q_heads, num_kv_heads, stream);
      break;
    case 8192:
      flash_attention_forward_dispatch_heads<8192>(Q, K, V, O, batch_size, num_q_heads, num_kv_heads, stream);
      break;
    case 16384:
      flash_attention_forward_dispatch_heads<16384>(Q, K, V, O, batch_size, num_q_heads, num_kv_heads, stream);
      break;
    case 32768:
      flash_attention_forward_dispatch_heads<32768>(Q, K, V, O, batch_size, num_q_heads, num_kv_heads, stream);
      break;
    default:
      fprintf(stderr, "Unsupported seq_len %d. Supported template instantiations: 512, 1024, 2048, 4096, 8192, 16384, 32768\n", seq_len);
      exit(EXIT_FAILURE);
  }
}

// BSHD layout: Q/K/V/O indexed as [b, s, h, d]
void cpu_attention(const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V, __nv_bfloat16 *O, int B,
                   int Sq, int Sk, int H, int Hk, int D, float scale) {
  auto Qat = [&](int b, int s, int h, int d) {
    return __bfloat162float(Q[((size_t)b * Sq + s) * H * D + h * D + d]);
  };
  auto Kat = [&](int b, int s, int h, int d) {
    return __bfloat162float(K[((size_t)b * Sk + s) * Hk * D + h * D + d]);
  };
  auto Vat = [&](int b, int s, int h, int d) {
    return __bfloat162float(V[((size_t)b * Sk + s) * Hk * D + h * D + d]);
  };
  for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++) {
      int hk = h * Hk / H;
      for (int m = 0; m < Sq; m++) {
        std::vector<float> sc(Sk);
        float mx = -FLT_MAX, sm = 0;
        for (int n = 0; n < Sk; n++) {
          float dot = 0;
          for (int d = 0; d < D; d++) dot += Qat(b, m, h, d) * Kat(b, n, hk, d);
          sc[n] = dot * scale;
          mx = fmaxf(mx, sc[n]);
        }
        for (int n = 0; n < Sk; n++) {
          sc[n] = expf(sc[n] - mx);
          sm += sc[n];
        }
        for (int n = 0; n < Sk; n++) sc[n] /= sm;
        for (int d = 0; d < D; d++) {
          float a = 0;
          for (int n = 0; n < Sk; n++) a += sc[n] * Vat(b, n, hk, d);
          O[((size_t)b * Sq + m) * H * D + h * D + d] = __float2bfloat16(a);
        }
      }
    }
}

int parse_device_arg(int argc, char **argv) {
  for (int i = 1; i < argc; ++i)
    if (strncmp(argv[i], "--device=", 9) == 0) return atoi(argv[i] + 9);
  return 0;
}

int parse_int_arg(int argc, char **argv, const char *prefix, int default_value) {
  const size_t n = strlen(prefix);
  for (int i = 1; i < argc; ++i)
    if (strncmp(argv[i], prefix, n) == 0) return atoi(argv[i] + n);
  return default_value;
}

struct BenchConfig {
  int batch;
  int seq;
};

std::vector<BenchConfig> parse_bench_configs_arg(int argc, char **argv, bool bench_only, int default_batch) {
  std::vector<int> seqs;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--seq=", 6) == 0) {
      seqs.push_back(atoi(argv[i] + 6));
      continue;
    }
    if (strncmp(argv[i], "--seqs=", 7) == 0) {
      const char *p = argv[i] + 7;
      while (*p) {
        char *end = nullptr;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        seqs.push_back((int)v);
        p = (*end == ',') ? end + 1 : end;
      }
    }
  }
  if (!seqs.empty()) {
    std::vector<BenchConfig> configs;
    configs.reserve(seqs.size());
    for (int seq : seqs) configs.push_back({default_batch, seq});
    return configs;
  }
  if (bench_only) return {{1, 32768}, {2, 16384}, {4, 8192}, {8, 4096}};
  return {{default_batch, 32768}};
}

bool all_bench_configs_valid(const std::vector<BenchConfig> &configs) {
  for (const BenchConfig &cfg : configs)
    if (cfg.seq % kBlockN != 0) return false;
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  int device = parse_device_arg(argc, argv);

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  printf("device=%s\n", prop.name);

  bool bench_only = (argc > 1 && strcmp(argv[1], "bench") == 0);
  int B = parse_int_arg(argc, argv, "--batch=", 1);
  std::vector<BenchConfig> bench_configs = parse_bench_configs_arg(argc, argv, bench_only, B);
  int H = 16, HK = 16, D = 128;
  int Sv = 512;

  // For now we apply such constraints for simplicity
  if (Sv % kBlockM != 0) {
    fprintf(stderr, "Error: Validation seqlen (%d) must be divisible by kBlockM (%d)\n", Sv, kBlockM);
    exit(EXIT_FAILURE);
  }
  if (!all_bench_configs_valid(bench_configs)) {
    fprintf(stderr, "Error: benchmark seqlen must be divisible by kBlockN (%d)\n", kBlockN);
    exit(EXIT_FAILURE);
  }

  if (!bench_only) {
    printf("=== Validation: B=%d H=%d S=%d D=%d ===\n", B, H, Sv, D);
    size_t sz = (size_t)B * H * Sv * D, szK = (size_t)B * HK * Sv * D;
    std::vector<__nv_bfloat16> hQ(sz), hK(szK), hV(szK), hO(sz), hR(sz);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t i = 0; i < sz; i++) hQ[i] = __float2bfloat16(dist(rng));
    for (size_t i = 0; i < szK; i++) {
      hK[i] = __float2bfloat16(dist(rng));
      hV[i] = __float2bfloat16(dist(rng));
    }
    float scale = 1.f / sqrtf((float)D);

    printf("CPU ref...\n");
    cpu_attention(hQ.data(), hK.data(), hV.data(), hR.data(), B, Sv, Sv, H, HK, D, scale);
    printf("done\n");

    __nv_bfloat16 *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dK, szK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dV, szK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dO, sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), sz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), szK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), szK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0, sz * sizeof(__nv_bfloat16)));

    flash_attention_forward_dispatch(dQ, dK, dV, dO, B, Sv, H, HK, 0);
    cudaError_t err = cudaDeviceSynchronize();
    printf("Kernel: %s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) {
      cudaFree(dQ);
      cudaFree(dK);
      cudaFree(dV);
      cudaFree(dO);
      return 1;
    }

    CUDA_CHECK(cudaMemcpy(hO.data(), dO, sz * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    float ma = 0;
    int bad = 0;
    for (size_t i = 0; i < sz; i++) {
      float r = __bfloat162float(hR[i]), g = __bfloat162float(hO[i]);
      float df = fabsf(r - g);
      ma = fmaxf(ma, df);
      if (df > 0.05f) bad++;
    }
    printf("MaxAbs=%.6f Bad=%d/%zu %s\n", ma, bad, sz, ma < 0.1f ? "PASS" : "FAIL");
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
  }

  // L2 flush buffer: sized to device L2, memset before each timed iteration
  // so every kernel run starts with a cold L2. Allocated once and reused.
  int l2_size = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device));
  void *d_l2_flush = nullptr;
  CUDA_CHECK(cudaMalloc(&d_l2_flush, (size_t)l2_size));

  for (const BenchConfig &cfg : bench_configs) {
    const int Bb = cfg.batch;
    const int Sb = cfg.seq;
    printf("=== Benchmark: B=%d H=%d S=%d D=%d ===\n", Bb, H, Sb, D);
    size_t sz = (size_t)Bb * H * Sb * D, szK = (size_t)Bb * HK * Sb * D;
    __nv_bfloat16 *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dK, szK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dV, szK * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dO, sz * sizeof(__nv_bfloat16)));
    std::vector<__nv_bfloat16> tmp(std::max(sz, szK));
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    for (size_t i = 0; i < sz; i++) tmp[i] = __float2bfloat16(dist(rng));
    CUDA_CHECK(cudaMemcpy(dQ, tmp.data(), sz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    for (size_t i = 0; i < szK; i++) tmp[i] = __float2bfloat16(dist(rng));
    CUDA_CHECK(cudaMemcpy(dK, tmp.data(), szK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    for (size_t i = 0; i < szK; i++) tmp[i] = __float2bfloat16(dist(rng));
    CUDA_CHECK(cudaMemcpy(dV, tmp.data(), szK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0, sz * sizeof(__nv_bfloat16)));

    int warmup = 5, iters = 20;
    for (int i = 0; i < warmup; i++) flash_attention_forward_dispatch(dQ, dK, dV, dO, Bb, Sb, H, HK, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Per-iteration events sandwich only the kernel; the L2 flush memset
    // runs on the same stream right before but is not included in elapsed time.
    std::vector<cudaEvent_t> ev0(iters), ev1(iters);
    for (int i = 0; i < iters; i++) {
      CUDA_CHECK(cudaEventCreate(&ev0[i]));
      CUDA_CHECK(cudaEventCreate(&ev1[i]));
    }
    for (int i = 0; i < iters; i++) {
      CUDA_CHECK(cudaMemsetAsync(d_l2_flush, 0, (size_t)l2_size, 0));
      CUDA_CHECK(cudaEventRecord(ev0[i], 0));
      flash_attention_forward_dispatch(dQ, dK, dV, dO, Bb, Sb, H, HK, 0);
      CUDA_CHECK(cudaEventRecord(ev1[i], 0));
    }
    CUDA_CHECK(cudaEventSynchronize(ev1[iters - 1]));

    float ms_sum = 0;
    for (int i = 0; i < iters; i++) {
      float ms_i = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms_i, ev0[i], ev1[i]));
      ms_sum += ms_i;
      CUDA_CHECK(cudaEventDestroy(ev0[i]));
      CUDA_CHECK(cudaEventDestroy(ev1[i]));
    }
    float ms = ms_sum / iters;
    double flops = 4.0 * Bb * H * Sb * (double)Sb * D;
    printf("Time: %.3f ms, TFLOPS: %.1f\n", ms, flops / (ms * 1e-3) / 1e12);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
  }

  cudaFree(d_l2_flush);

  return 0;
}
