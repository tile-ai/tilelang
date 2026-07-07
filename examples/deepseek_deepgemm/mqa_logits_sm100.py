import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.meta_path = [finder for finder in sys.meta_path if finder.__class__.__module__ != "_tilelang_editable"]

import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver

_WRELU_REDUCE_SRC = r"""
#include <tl_templates/cuda/tcgen_05.h>
#include <tl_templates/cuda/tcgen_05_ld.h>
#include <tl_templates/cuda/barrier.h>

extern "C" __device__ __forceinline__ void tl_mqa_mbarrier_wait_acquire(
    Barrier* __restrict__ barrier, int phase) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra DONE;\n\t"
      "bra LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}\n"
      :
      : "r"(smem_addr), "r"(phase), "r"(ticks)
      : "memory");
}

extern "C" __device__ __forceinline__ void tl_mqa_mbarrier_arrive_release(
    Barrier* __restrict__ barrier) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :
               : "r"(smem_addr)
               : "memory");
}

extern "C" __device__ __forceinline__ float4 tl_mqa_ld_shared_v4_f32(
    const float* __restrict__ ptr) {
  float4 ret;
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
               : "r"(smem_addr));
  return ret;
}

extern "C" __device__ __forceinline__ float tl_mqa_wrelu_reduce32(
    const float* __restrict__ x, const float* __restrict__ w, float scale) {
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; i += 4) {
    acc0 = fmaf(fmaxf(x[i + 0], 0.0f), w[i + 0], acc0);
    acc1 = fmaf(fmaxf(x[i + 1], 0.0f), w[i + 1], acc1);
    acc2 = fmaf(fmaxf(x[i + 2], 0.0f), w[i + 2], acc2);
    acc3 = fmaf(fmaxf(x[i + 3], 0.0f), w[i + 3], acc3);
  }
  return scale * ((acc0 + acc1) + (acc2 + acc3));
}

extern "C" __device__ __forceinline__ float tl_mqa_wrelu_reduce64(
    const float* __restrict__ x, const float* __restrict__ w, float scale) {
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
  for (int i = 0; i < 64; i += 4) {
    acc0 = fmaf(fmaxf(x[i + 0], 0.0f), w[i + 0], acc0);
    acc1 = fmaf(fmaxf(x[i + 1], 0.0f), w[i + 1], acc1);
    acc2 = fmaf(fmaxf(x[i + 2], 0.0f), w[i + 2], acc2);
    acc3 = fmaf(fmaxf(x[i + 3], 0.0f), w[i + 3], acc3);
  }
  return scale * ((acc0 + acc1) + (acc2 + acc3));
}

template <int N, bool UseV4Weights>
__device__ __forceinline__ float tl_mqa_wrelu_tmem_reduce_impl(
    uint32_t tmem_start_col, uint32_t tmem_col_offset,
    const float* __restrict__ w, float scale) {
  float2 sum0 = make_float2(0.0f, 0.0f);
  float2 sum1 = make_float2(0.0f, 0.0f);
#pragma unroll
  for (int base = 0; base < N; base += 16) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
    uint32_t r8, r9, r10, r11, r12, r13, r14, r15;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},"
        "[%16];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
          "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7),
          "=r"(r8), "=r"(r9), "=r"(r10), "=r"(r11),
          "=r"(r12), "=r"(r13), "=r"(r14), "=r"(r15)
        : "r"(tmem_start_col + tmem_col_offset + base));
    tl::fence_view_async_tmem_load();

#define TL_MQA_ACC4(offset, x0, x1, x2, x3)                                      \
    do {                                                                         \
      float2 a0 = make_float2(fmaxf(__uint_as_float(x0), 0.0f),                  \
                              fmaxf(__uint_as_float(x1), 0.0f));                 \
      float2 a1 = make_float2(fmaxf(__uint_as_float(x2), 0.0f),                  \
                              fmaxf(__uint_as_float(x3), 0.0f));                 \
      float2 b0;                                                                 \
      float2 b1;                                                                 \
      if constexpr (UseV4Weights) {                                              \
        float4 wv = tl_mqa_ld_shared_v4_f32(w + base + (offset));                \
        b0 = make_float2(wv.x, wv.y);                                            \
        b1 = make_float2(wv.z, wv.w);                                            \
      } else {                                                                   \
        b0 = make_float2(w[base + (offset) + 0], w[base + (offset) + 1]);         \
        b1 = make_float2(w[base + (offset) + 2], w[base + (offset) + 3]);         \
      }                                                                          \
      sum0 = __ffma2_rn(a0, b0, sum0);                                           \
      sum1 = __ffma2_rn(a1, b1, sum1);                                           \
    } while (0)

    TL_MQA_ACC4(0, r0, r1, r2, r3);
    TL_MQA_ACC4(4, r4, r5, r6, r7);
    TL_MQA_ACC4(8, r8, r9, r10, r11);
    TL_MQA_ACC4(12, r12, r13, r14, r15);
#undef TL_MQA_ACC4
  }
  float2 sum = __fadd2_rn(sum0, sum1);
  return scale * (sum.x + sum.y);
}

extern "C" __device__ __forceinline__ float tl_mqa_wrelu_tmem_reduce32(
    uint32_t tmem_start_col, uint32_t tmem_col_offset,
    const float* __restrict__ w, float scale) {
  return tl_mqa_wrelu_tmem_reduce_impl<32, false>(tmem_start_col, tmem_col_offset, w, scale);
}

extern "C" __device__ __forceinline__ float tl_mqa_wrelu_tmem_reduce64(
    uint32_t tmem_start_col, uint32_t tmem_col_offset,
    const float* __restrict__ w, float scale) {
  return tl_mqa_wrelu_tmem_reduce_impl<64, false>(tmem_start_col, tmem_col_offset, w, scale);
}

extern "C" __device__ __forceinline__ float tl_mqa_wrelu_tmem_reduce32_v4w(
    uint32_t tmem_start_col, uint32_t tmem_col_offset,
    const float* __restrict__ w, float scale) {
  return tl_mqa_wrelu_tmem_reduce_impl<32, true>(tmem_start_col, tmem_col_offset, w, scale);
}

extern "C" __device__ __forceinline__ float tl_mqa_wrelu_tmem_reduce64_v4w(
    uint32_t tmem_start_col, uint32_t tmem_col_offset,
    const float* __restrict__ w, float scale) {
  return tl_mqa_wrelu_tmem_reduce_impl<64, true>(tmem_start_col, tmem_col_offset, w, scale);
}

template <int N, bool UseV4Weights>
__device__ __forceinline__ void tl_mqa_wrelu_tmem_reduce_store_f32_impl(
    uint32_t tmem_start_col, uint32_t tmem_col_offset,
    const float* __restrict__ w, float scale,
    float* __restrict__ logits, int logits_offset) {
  logits[logits_offset] =
      tl_mqa_wrelu_tmem_reduce_impl<N, UseV4Weights>(tmem_start_col, tmem_col_offset, w, scale);
}

extern "C" __device__ __forceinline__ void tl_mqa_wrelu_tmem_reduce64_store_f32(
    uint32_t tmem_start_col, uint32_t tmem_col_offset,
    const float* __restrict__ w, float scale,
    float* __restrict__ logits, int logits_offset) {
  tl_mqa_wrelu_tmem_reduce_store_f32_impl<64, false>(
      tmem_start_col, tmem_col_offset, w, scale, logits, logits_offset);
}

extern "C" __device__ __forceinline__ void tl_mqa_wrelu_tmem_reduce64_v4w_store_f32(
    uint32_t tmem_start_col, uint32_t tmem_col_offset,
    const float* __restrict__ w, float scale,
    float* __restrict__ logits, int logits_offset) {
  tl_mqa_wrelu_tmem_reduce_store_f32_impl<64, true>(
      tmem_start_col, tmem_col_offset, w, scale, logits, logits_offset);
}

#define TL_MQA_DECL_W64(p)                                                       \
  float4 p##0, p##1, p##2, p##3, p##4, p##5, p##6, p##7;                        \
  float4 p##8, p##9, p##10, p##11, p##12, p##13, p##14, p##15

#define TL_MQA_LOAD_W64(p, ptr)                                                  \
  do {                                                                           \
    p##0 = tl_mqa_ld_shared_v4_f32((ptr) + 0);                                   \
    p##1 = tl_mqa_ld_shared_v4_f32((ptr) + 4);                                   \
    p##2 = tl_mqa_ld_shared_v4_f32((ptr) + 8);                                   \
    p##3 = tl_mqa_ld_shared_v4_f32((ptr) + 12);                                  \
    p##4 = tl_mqa_ld_shared_v4_f32((ptr) + 16);                                  \
    p##5 = tl_mqa_ld_shared_v4_f32((ptr) + 20);                                  \
    p##6 = tl_mqa_ld_shared_v4_f32((ptr) + 24);                                  \
    p##7 = tl_mqa_ld_shared_v4_f32((ptr) + 28);                                  \
    p##8 = tl_mqa_ld_shared_v4_f32((ptr) + 32);                                  \
    p##9 = tl_mqa_ld_shared_v4_f32((ptr) + 36);                                  \
    p##10 = tl_mqa_ld_shared_v4_f32((ptr) + 40);                                 \
    p##11 = tl_mqa_ld_shared_v4_f32((ptr) + 44);                                 \
    p##12 = tl_mqa_ld_shared_v4_f32((ptr) + 48);                                 \
    p##13 = tl_mqa_ld_shared_v4_f32((ptr) + 52);                                 \
    p##14 = tl_mqa_ld_shared_v4_f32((ptr) + 56);                                 \
    p##15 = tl_mqa_ld_shared_v4_f32((ptr) + 60);                                 \
  } while (0)

#define TL_MQA_ACC4_WV(wv, x0, x1, x2, x3)                                       \
  do {                                                                           \
    float2 a0 = make_float2(fmaxf(__uint_as_float(x0), 0.0f),                    \
                            fmaxf(__uint_as_float(x1), 0.0f));                   \
    float2 a1 = make_float2(fmaxf(__uint_as_float(x2), 0.0f),                    \
                            fmaxf(__uint_as_float(x3), 0.0f));                   \
    float2 b0 = make_float2((wv).x, (wv).y);                                     \
    float2 b1 = make_float2((wv).z, (wv).w);                                     \
    sum0 = __ffma2_rn(a0, b0, sum0);                                             \
    sum1 = __ffma2_rn(a1, b1, sum1);                                             \
  } while (0)

#define TL_MQA_REDUCE16_W4(p0, p1, p2, p3, addr_expr)                            \
  do {                                                                           \
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;                                     \
    uint32_t r8, r9, r10, r11, r12, r13, r14, r15;                               \
    asm volatile(                                                                \
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"                                \
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}," \
        "[%16];\n"                                                               \
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),                               \
          "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7),                               \
          "=r"(r8), "=r"(r9), "=r"(r10), "=r"(r11),                             \
          "=r"(r12), "=r"(r13), "=r"(r14), "=r"(r15)                            \
        : "r"(addr_expr));                                                       \
    tl::fence_view_async_tmem_load();                                            \
    TL_MQA_ACC4_WV(p0, r0, r1, r2, r3);                                          \
    TL_MQA_ACC4_WV(p1, r4, r5, r6, r7);                                          \
    TL_MQA_ACC4_WV(p2, r8, r9, r10, r11);                                        \
    TL_MQA_ACC4_WV(p3, r12, r13, r14, r15);                                      \
  } while (0)

#define TL_MQA_REDUCE64_REGS(out, p, addr_expr)                                  \
  do {                                                                           \
    float2 sum0 = make_float2(0.0f, 0.0f);                                       \
    float2 sum1 = make_float2(0.0f, 0.0f);                                       \
    TL_MQA_REDUCE16_W4(p##0, p##1, p##2, p##3, (addr_expr) + 0);                 \
    TL_MQA_REDUCE16_W4(p##4, p##5, p##6, p##7, (addr_expr) + 16);                \
    TL_MQA_REDUCE16_W4(p##8, p##9, p##10, p##11, (addr_expr) + 32);              \
    TL_MQA_REDUCE16_W4(p##12, p##13, p##14, p##15, (addr_expr) + 48);            \
    float2 sum = __fadd2_rn(sum0, sum1);                                         \
    (out) = sum.x + sum.y;                                                       \
  } while (0)

extern "C" __device__ __forceinline__ void tl_mqa_fp4_epilogue_half_cached_h64(
    uint32_t tmem_start_col,
    Barrier* __restrict__ q_loaded,
    Barrier* __restrict__ q_empty,
    Barrier* __restrict__ tmem_full,
    Barrier* __restrict__ tmem_empty,
    const int* __restrict__ ks,
    const int* __restrict__ ke,
    float* __restrict__ logits,
    const float* __restrict__ weights_shared,
    int block_id,
    int sm_num,
    int num_q_blocks,
    int logits_stride,
    int half_offset) {
  constexpr int kBlockQ = 2;
  constexpr int kHeads = 64;
  constexpr int kBlockKV = 256;
  constexpr int kNumQStages = 3;
  constexpr int kNumTmemStages = 3;

  const int tx = static_cast<int>(threadIdx.x);
  const int bn = tx - (half_offset == 0 ? 128 : 256);
  int q_block = block_id;
  int q_iter = 0;
  int tile_iter = 0;
  int tmem_slot = 0;

  while (q_block < num_q_blocks) {
    const int q_row = q_block * kBlockQ;
    const int q_stage = q_iter % kNumQStages;
    const int q_phase = (q_iter / kNumQStages) & 1;

    int tile_min_ks = ks[q_row];
    int tile_max_ke = ke[q_row];
#pragma unroll
    for (int qi = 1; qi < kBlockQ; ++qi) {
      tile_min_ks = min(tile_min_ks, ks[q_row + qi]);
      tile_max_ke = max(tile_max_ke, ke[q_row + qi]);
    }
    const int first_bkv = tile_min_ks / kBlockKV;
    const int last_bkv = (tile_max_ke + kBlockKV - 1) / kBlockKV;
    const int num_kv_blocks = max(last_bkv - first_bkv, 0);

    tl_mqa_mbarrier_wait_acquire(q_loaded + q_stage, q_phase);

    const float* w_src = weights_shared + q_stage * kBlockQ * kHeads;
    TL_MQA_DECL_W64(wq0_);
    TL_MQA_DECL_W64(wq1_);
    if (num_kv_blocks > 0) {
      TL_MQA_LOAD_W64(wq0_, w_src);
      TL_MQA_LOAD_W64(wq1_, w_src + kHeads);
    }
    for (int kv_iter = 0; kv_iter < num_kv_blocks; ++kv_iter) {
      const int kv_row = (first_bkv + kv_iter) * kBlockKV;
      int tmem_stage, tmem_phase;
      if (half_offset == 0) {
        if (tmem_slot == 0) {
          tmem_stage = 0;
          tmem_phase = 0;
        } else if (tmem_slot == 1) {
          tmem_stage = 2;
          tmem_phase = 0;
        } else {
          tmem_stage = 1;
          tmem_phase = 1;
        }
      } else {
        if (tmem_slot == 0) {
          tmem_stage = 1;
          tmem_phase = 0;
        } else if (tmem_slot == 1) {
          tmem_stage = 0;
          tmem_phase = 1;
        } else {
          tmem_stage = 2;
          tmem_phase = 1;
        }
      }
      const uint32_t tmem_col = static_cast<uint32_t>(tmem_stage * kBlockQ * kHeads);
      tl_mqa_mbarrier_wait_acquire(tmem_full + tmem_stage, tmem_phase);
      tl::tcgen05_after_thread_sync();

      float result_q0;
      TL_MQA_REDUCE64_REGS(result_q0, wq0_, tmem_start_col + tmem_col);
      logits[q_row * logits_stride + kv_row + half_offset + bn] = result_q0;
      __syncwarp();

      float result_q1;
      TL_MQA_REDUCE64_REGS(result_q1, wq1_, tmem_start_col + tmem_col + kHeads);
      logits[(q_row + 1) * logits_stride + kv_row + half_offset + bn] = result_q1;
      __syncwarp();

      tl::tcgen05_before_thread_sync();
      tl_mqa_mbarrier_arrive_release(tmem_empty + tmem_stage);
      ++tile_iter;
      ++tmem_slot;
      if (tmem_slot == kNumTmemStages)
        tmem_slot = 0;
    }

    tl_mqa_mbarrier_arrive_release(q_empty + q_stage);
    q_block += sm_num;
    ++q_iter;
  }
}
"""

def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def _torch_logits_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported logits dtype: {dtype}")


def _tilelang_logits_dtype(dtype: str):
    if dtype == "float32":
        return T.float32
    if dtype == "bfloat16":
        return T.bfloat16
    raise ValueError(f"unsupported logits dtype: {dtype}")


@dataclass(frozen=True)
class MQALogitsConfig:
    seq_len: int = 2048
    seq_len_kv: int = 4096
    num_heads: int = 64
    head_dim: int = 128
    logits_dtype: str = "float32"
    seed: int = 0

    @property
    def block_q(self) -> int:
        return 128 // self.num_heads

    def validate(self) -> None:
        if self.num_heads != 64:
            raise ValueError("SM100 MQA SOTA kernels currently require num_heads=64")
        if self.head_dim != 128:
            raise ValueError("SM100 MQA SOTA kernels currently require head_dim=128")
        if self.seq_len <= 0 or self.seq_len_kv <= 0:
            raise ValueError("sequence lengths must be positive")
        if self.seq_len > self.seq_len_kv:
            raise ValueError("seq_len must be <= seq_len_kv for the demo causal ranges")
        if self.seq_len_kv - self.seq_len < 128:
            raise ValueError("seq_len_kv must exceed seq_len by at least one 128-wide tile")
        if self.seq_len % self.block_q != 0:
            raise ValueError("seq_len must be divisible by block_q")
        if self.seq_len_kv % 128 != 0:
            raise ValueError("seq_len_kv must be divisible by 128")
        _torch_logits_dtype(self.logits_dtype)


def generate_ks_ke(config: MQALogitsConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    ks = torch.zeros(config.seq_len, dtype=torch.int32, device="cuda")
    ke = torch.arange(config.seq_len, dtype=torch.int32, device="cuda")
    ke = ke + (config.seq_len_kv - config.seq_len)
    return ks, ke


def ref_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    seq_len_kv = kv.shape[0]
    q_f32 = q.float()
    kv_f32 = kv.float()
    cols = torch.arange(seq_len_kv, device=q.device)
    logits = torch.empty((q.shape[0], seq_len_kv), device=q.device, dtype=torch.float32)
    chunk = 128
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        score = torch.einsum("mhd,nd->hmn", q_f32[start:end], kv_f32)
        part = (score.relu() * weights[start:end].unsqueeze(-1).transpose(0, 1)).sum(dim=0)
        mask = (cols[None, :] >= ks[start:end, None]) & (cols[None, :] < ke[start:end, None])
        logits[start:end] = part.masked_fill(~mask, float("-inf"))
    return logits

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
)
def mqa_logits_fp4_persistent_ws_kernel(
    Q,
    QScale,
    KV,
    KVScale,
    Weights,
    KS,
    KE,
    Logits,
    seq_len: int,
    seq_len_kv: int,
    heads: int = 64,
    head_dim: int = 128,
    logits_stride: int = 4096,
    compressed_logits: bool = False,
    logits_dtype=T.float32,
):
    block_q = 128 // heads
    block_kv = 256
    half_kv = 128
    num_q_stages = 3
    num_stages = 6
    num_tmem_stages = 3
    sf_granularity_k = 32
    sf_k_groups = T.ceildiv(T.ceildiv(head_dim, sf_granularity_k), 4)
    accum_dtype = T.float32
    num_q_blocks = T.ceildiv(seq_len, block_q)
    sm_num = driver.get_num_sms()

    Q: T.Tensor((seq_len * heads, head_dim), T.float4_e2m1fn)
    QScale: T.Tensor((sf_k_groups * seq_len * heads,), T.uint32)
    KV: T.Tensor((seq_len_kv, head_dim), T.float4_e2m1fn)
    KVScale: T.Tensor((sf_k_groups * seq_len_kv,), T.uint32)
    Weights: T.Tensor((seq_len, heads), accum_dtype)
    KS: T.Tensor((seq_len,), T.int32)
    KE: T.Tensor((seq_len,), T.int32)
    Logits: T.Tensor((seq_len, logits_stride), logits_dtype)

    with T.Kernel(sm_num, threads=384) as block_id:
        T.import_source(_WRELU_REDUCE_SRC)
        q_shared = T.alloc_shared((num_q_stages, block_q * heads, head_dim), T.float4_e2m1fn)
        sf_q_shared = T.alloc_shared((num_q_stages, block_q * heads), T.uint32)
        weights_shared = T.alloc_shared((num_q_stages, block_q, heads), accum_dtype)
        kv_shared_0 = T.alloc_shared((num_stages, half_kv, head_dim), T.float4_e2m1fn)
        kv_shared_1 = T.alloc_shared((num_stages, half_kv, head_dim), T.float4_e2m1fn)
        sf_kv_shared_0 = T.alloc_shared((num_stages, half_kv), T.uint32)
        sf_kv_shared_1 = T.alloc_shared((num_stages, half_kv), T.uint32)
        c_tmem = T.alloc_tmem((half_kv, 512), accum_dtype)
        sf_q_col_0 = block_q * heads * num_tmem_stages
        sf_q_col_1 = sf_q_col_0 + 4
        sf_kv_col_0 = sf_q_col_1 + 4
        sf_kv_col_1 = sf_kv_col_0 + 4
        q_loaded = T.alloc_barrier([32] * num_q_stages)
        q_empty = T.alloc_barrier([352] * num_q_stages)
        q_sf_full = T.alloc_barrier([32] * num_q_stages)
        kv_loaded = T.alloc_barrier([32] * num_stages)
        kv_sf_full_0 = T.alloc_barrier([32] * num_stages)
        kv_sf_full_1 = T.alloc_barrier([32] * num_stages)
        kv_empty = T.alloc_barrier([64] * num_stages)
        tmem_full = T.alloc_barrier([1] * num_tmem_stages)
        tmem_empty = T.alloc_barrier([128] * num_tmem_stages)

        tx = T.get_thread_binding()

        if tx < 32:
            T.dec_max_nreg(56)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            tile_iter = T.alloc_var(T.int32, init=0)
            if q_block < num_q_blocks:
                first_q_row = q_block * block_q
                T.tma_copy(
                    Q[first_q_row * heads : first_q_row * heads + block_q * heads, :],
                    q_shared[0, :, :],
                    barrier=q_loaded[0],
                )
                T.tma_copy(
                    QScale[first_q_row * heads : first_q_row * heads + block_q * heads],
                    sf_q_shared[0, :],
                    barrier=q_loaded[0],
                )
                T.tma_copy(
                    Weights[first_q_row : first_q_row + block_q, :],
                    weights_shared[0, :, :],
                    barrier=q_loaded[0],
                )
                T.mbarrier_arrive(q_loaded[0])
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset in T.unroll(block_q - 1):
                    qi = qi_offset + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                next_q_block = q_block + sm_num
                next_q_iter = q_iter + 1
                next_q_stage = next_q_iter % num_q_stages
                next_q_phase = (next_q_iter // num_q_stages) & 1
                if next_q_block < num_q_blocks:
                    T.mbarrier_wait_parity(q_empty[next_q_stage], next_q_phase ^ 1)
                    next_q_row = next_q_block * block_q
                    T.tma_copy(
                        Q[next_q_row * heads : next_q_row * heads + block_q * heads, :],
                        q_shared[next_q_stage, :, :],
                        barrier=q_loaded[next_q_stage],
                    )
                    T.tma_copy(
                        QScale[next_q_row * heads : next_q_row * heads + block_q * heads],
                        sf_q_shared[next_q_stage, :],
                        barrier=q_loaded[next_q_stage],
                    )
                    T.tma_copy(
                        Weights[next_q_row : next_q_row + block_q, :],
                        weights_shared[next_q_stage, :, :],
                        barrier=q_loaded[next_q_stage],
                    )
                    T.mbarrier_arrive(q_loaded[next_q_stage])

                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    kv_row = (first_bkv + kv_iter) * block_kv
                    stage = tile_iter % num_stages
                    parity = (tile_iter // num_stages) & 1
                    T.mbarrier_wait_parity(kv_empty[stage], parity ^ 1)
                    T.tma_copy(KV[kv_row : kv_row + half_kv, :], kv_shared_0[stage, :, :], barrier=kv_loaded[stage])
                    T.tma_copy(
                        KV[kv_row + half_kv : kv_row + block_kv, :],
                        kv_shared_1[stage, :, :],
                        barrier=kv_loaded[stage],
                    )
                    T.tma_copy(KVScale[kv_row : kv_row + half_kv], sf_kv_shared_0[stage, :], barrier=kv_loaded[stage])
                    T.tma_copy(
                        KVScale[kv_row + half_kv : kv_row + block_kv],
                        sf_kv_shared_1[stage, :],
                        barrier=kv_loaded[stage],
                    )
                    T.mbarrier_arrive(kv_loaded[stage])
                    tile_iter = tile_iter + 1
                    kv_iter = kv_iter + 1

                q_block = next_q_block
                q_iter = next_q_iter

        elif 64 <= tx < 96:
            T.dec_max_nreg(56)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            tile_iter = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset in T.unroll(block_q - 1):
                    qi = qi_offset + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                T.tcgen05_sf_warp_transpose(sf_q_shared[q_stage, :])
                T.fence_proxy_async()
                T.mbarrier_arrive(q_sf_full[q_stage])
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    stage = tile_iter % num_stages
                    parity = (tile_iter // num_stages) & 1
                    T.mbarrier_wait_parity(kv_loaded[stage], parity)
                    T.tcgen05_sf_warp_transpose(sf_kv_shared_0[stage, :])
                    T.fence_proxy_async()
                    T.mbarrier_arrive(kv_sf_full_0[stage])
                    T.tcgen05_sf_warp_transpose(sf_kv_shared_1[stage, :])
                    T.fence_proxy_async()
                    T.mbarrier_arrive(kv_sf_full_1[stage])
                    tile_iter = tile_iter + 1
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 32 <= tx < 64:
            T.dec_max_nreg(56)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            tile_iter = T.alloc_var(T.int32, init=0)
            tmem_slot = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset in T.unroll(block_q - 1):
                    qi = qi_offset + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                T.mbarrier_wait_parity(q_sf_full[q_stage], q_phase)
                T.tcgen05_cp_warpx4(sf_q_shared[q_stage, :], c_tmem, tmem_col_offset=sf_q_col_0)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    stage = tile_iter % num_stages
                    parity = (tile_iter // num_stages) & 1
                    tmem_stage = T.alloc_var(T.int32)
                    tmem_phase = T.alloc_var(T.int32)
                    if tmem_slot == 0:
                        tmem_stage = 0
                        tmem_phase = 0
                    elif tmem_slot == 1:
                        tmem_stage = 2
                        tmem_phase = 0
                    else:
                        tmem_stage = 1
                        tmem_phase = 1
                    tmem_col = tmem_stage * block_q * heads
                    T.mbarrier_wait_parity(tmem_empty[tmem_stage], tmem_phase ^ 1)
                    T.mbarrier_wait_parity(kv_loaded[stage], parity)
                    T.mbarrier_wait_parity(kv_sf_full_0[stage], parity)
                    T.tcgen05_cp_warpx4(sf_kv_shared_0[stage, :], c_tmem, tmem_col_offset=sf_kv_col_0)
                    T.tcgen05_gemm_blockscaled(
                        kv_shared_0[stage, :, :],
                        q_shared[q_stage, :, :],
                        c_tmem[:, tmem_col : tmem_col + block_q * heads],
                        c_tmem[:, sf_kv_col_0 : sf_kv_col_0 + 4],
                        c_tmem[:, sf_q_col_0 : sf_q_col_0 + 4],
                        transpose_B=True,
                        mbar=tmem_full[tmem_stage],
                        clear_accum=True,
                        k_start=0,
                        sf_a_granularity_k=sf_granularity_k,
                        sf_b_granularity_k=sf_granularity_k,
                        blockscale_format="mx",
                    )
                    T.mbarrier_arrive(kv_empty[stage])
                    tile_iter = tile_iter + 1
                    tmem_slot = tmem_slot + 1
                    if tmem_slot == 3:
                        tmem_slot = 0
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 96 <= tx < 128:
            T.dec_max_nreg(56)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            tile_iter = T.alloc_var(T.int32, init=0)
            tmem_slot = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset in T.unroll(block_q - 1):
                    qi = qi_offset + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                T.mbarrier_wait_parity(q_sf_full[q_stage], q_phase)
                T.tcgen05_cp_warpx4(sf_q_shared[q_stage, :], c_tmem, tmem_col_offset=sf_q_col_1)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    stage = tile_iter % num_stages
                    parity = (tile_iter // num_stages) & 1
                    tmem_stage = T.alloc_var(T.int32)
                    tmem_phase = T.alloc_var(T.int32)
                    if tmem_slot == 0:
                        tmem_stage = 1
                        tmem_phase = 0
                    elif tmem_slot == 1:
                        tmem_stage = 0
                        tmem_phase = 1
                    else:
                        tmem_stage = 2
                        tmem_phase = 1
                    tmem_col = tmem_stage * block_q * heads
                    T.mbarrier_wait_parity(tmem_empty[tmem_stage], tmem_phase ^ 1)
                    T.mbarrier_wait_parity(kv_loaded[stage], parity)
                    T.mbarrier_wait_parity(kv_sf_full_1[stage], parity)
                    T.tcgen05_cp_warpx4(sf_kv_shared_1[stage, :], c_tmem, tmem_col_offset=sf_kv_col_1)
                    T.tcgen05_gemm_blockscaled(
                        kv_shared_1[stage, :, :],
                        q_shared[q_stage, :, :],
                        c_tmem[:, tmem_col : tmem_col + block_q * heads],
                        c_tmem[:, sf_kv_col_1 : sf_kv_col_1 + 4],
                        c_tmem[:, sf_q_col_1 : sf_q_col_1 + 4],
                        transpose_B=True,
                        mbar=tmem_full[tmem_stage],
                        clear_accum=True,
                        k_start=0,
                        sf_a_granularity_k=sf_granularity_k,
                        sf_b_granularity_k=sf_granularity_k,
                        blockscale_format="mx",
                    )
                    T.mbarrier_arrive(kv_empty[stage])
                    tile_iter = tile_iter + 1
                    tmem_slot = tmem_slot + 1
                    if tmem_slot == 3:
                        tmem_slot = 0
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 128 <= tx < 256:
            T.inc_max_nreg(224)
            T.evaluate(T.call_extern(
                "handle",
                "tl_mqa_fp4_epilogue_half_cached_h64",
                c_tmem[0, 0],
                T.address_of(q_loaded[0]),
                T.address_of(q_empty[0]),
                T.address_of(tmem_full[0]),
                T.address_of(tmem_empty[0]),
                T.address_of(KS[0]),
                T.address_of(KE[0]),
                T.address_of(Logits[0, 0]),
                T.address_of(weights_shared[0, 0, 0]),
                block_id,
                sm_num,
                num_q_blocks,
                logits_stride,
                0,
            ))

        elif 256 <= tx < 384:
            T.inc_max_nreg(224)
            T.evaluate(T.call_extern(
                "handle",
                "tl_mqa_fp4_epilogue_half_cached_h64",
                c_tmem[0, 0],
                T.address_of(q_loaded[0]),
                T.address_of(q_empty[0]),
                T.address_of(tmem_full[0]),
                T.address_of(tmem_empty[0]),
                T.address_of(KS[0]),
                T.address_of(KE[0]),
                T.address_of(Logits[0, 0]),
                T.address_of(weights_shared[0, 0, 0]),
                block_id,
                sm_num,
                num_q_blocks,
                logits_stride,
                half_kv,
            ))

        T.sync_threads()

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
)
def mqa_logits_fp8_persistent_ws_kernel(
    Q,
    KV,
    KVScale,
    Weights,
    KS,
    KE,
    Logits,
    seq_len: int,
    seq_len_kv: int,
    heads: int = 64,
    head_dim: int = 128,
    logits_stride: int = 4096,
    compressed_logits: bool = False,
    logits_dtype=T.float32,
):
    block_q = 128 // heads
    block_kv = 256
    half_kv = 128
    num_q_stages = 3
    num_stages = 3
    num_tmem_stages = 3
    accum_dtype = T.float32
    num_q_blocks = T.ceildiv(seq_len, block_q)
    sm_num = driver.get_num_sms()

    Q: T.Tensor((seq_len * heads, head_dim), T.float8_e4m3fn)
    KV: T.Tensor((seq_len_kv, head_dim), T.float8_e4m3fn)
    KVScale: T.Tensor((seq_len_kv,), accum_dtype)
    Weights: T.Tensor((seq_len, heads), accum_dtype)
    KS: T.Tensor((seq_len,), T.int32)
    KE: T.Tensor((seq_len,), T.int32)
    Logits: T.Tensor((seq_len, logits_stride), logits_dtype)

    with T.Kernel(sm_num, threads=384) as block_id:
        T.import_source(_WRELU_REDUCE_SRC)
        q_shared = T.alloc_shared((num_q_stages, block_q * heads, head_dim), T.float8_e4m3fn)
        weights_shared = T.alloc_shared((num_q_stages, block_q, heads), accum_dtype)
        kv_shared = T.alloc_shared((num_stages, block_kv, head_dim), T.float8_e4m3fn)
        kv_scale_shared = T.alloc_shared((num_stages, block_kv), accum_dtype)
        c_tmem = T.alloc_tmem((half_kv, block_q * heads * num_tmem_stages), accum_dtype)
        q_loaded = T.alloc_barrier([32] * num_q_stages)
        q_empty = T.alloc_barrier([320] * num_q_stages)
        kv_loaded = T.alloc_barrier([32] * num_stages)
        kv_empty = T.alloc_barrier([256] * num_stages)
        tmem_full = T.alloc_barrier([1] * num_tmem_stages)
        tmem_empty = T.alloc_barrier([128] * num_tmem_stages)

        tx = T.get_thread_binding()

        # Keep setmaxnreg inside the role branches. A separate pre-branch
        # setmaxnreg if emits the instruction but ptxas does not extend the
        # high-register region into the epilogue, and x16 TMEM loads spill.
        if tx < 32:
            T.dec_max_nreg(40)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            gemm_iter = T.alloc_var(T.int32, init=0)
            if q_block < num_q_blocks:
                first_q_row = q_block * block_q
                T.tma_copy(
                    Q[first_q_row * heads : first_q_row * heads + block_q * heads, :],
                    q_shared[0, :, :],
                    barrier=q_loaded[0],
                )
                T.tma_copy(
                    Weights[first_q_row : first_q_row + block_q, :],
                    weights_shared[0, :, :],
                    barrier=q_loaded[0],
                )
                T.mbarrier_arrive(q_loaded[0])
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_tail in T.unroll(block_q - 1):
                    qi_scan_tail = qi_offset_tail + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_tail])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_tail])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                next_q_block = q_block + sm_num
                next_q_iter = q_iter + 1
                next_q_stage = next_q_iter % num_q_stages
                next_q_phase = (next_q_iter // num_q_stages) & 1
                if next_q_block < num_q_blocks:
                    T.mbarrier_wait_parity(q_empty[next_q_stage], next_q_phase ^ 1)
                    next_q_row = next_q_block * block_q
                    T.tma_copy(
                        Q[next_q_row * heads : next_q_row * heads + block_q * heads, :],
                        q_shared[next_q_stage, :, :],
                        barrier=q_loaded[next_q_stage],
                    )
                    T.tma_copy(
                        Weights[next_q_row : next_q_row + block_q, :],
                        weights_shared[next_q_stage, :, :],
                        barrier=q_loaded[next_q_stage],
                    )
                    T.mbarrier_arrive(q_loaded[next_q_stage])

                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    kv_row = (first_bkv + kv_iter) * block_kv
                    stage = gemm_iter % num_stages
                    parity = (gemm_iter // num_stages) & 1
                    T.mbarrier_wait_parity(kv_empty[stage], parity ^ 1)
                    T.tma_copy(KV[kv_row : kv_row + block_kv, :], kv_shared[stage, :, :], barrier=kv_loaded[stage])
                    T.tma_copy(KVScale[kv_row : kv_row + block_kv], kv_scale_shared[stage, :], barrier=kv_loaded[stage])
                    T.mbarrier_arrive(kv_loaded[stage])
                    gemm_iter = gemm_iter + 1
                    kv_iter = kv_iter + 1

                q_block = next_q_block
                q_iter = next_q_iter

        elif 32 <= tx < 64:
            T.dec_max_nreg(40)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            gemm_iter = T.alloc_var(T.int32, init=0)
            tmem_slot = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_tail in T.unroll(block_q - 1):
                    qi_scan_tail = qi_offset_tail + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_tail])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_tail])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    stage = gemm_iter % num_stages
                    parity = (gemm_iter // num_stages) & 1
                    tmem_stage = T.alloc_var(T.int32)
                    tmem_phase = T.alloc_var(T.int32)
                    if tmem_slot == 0:
                        tmem_stage = 0
                        tmem_phase = 0
                    elif tmem_slot == 1:
                        tmem_stage = 2
                        tmem_phase = 0
                    else:
                        tmem_stage = 1
                        tmem_phase = 1
                    tmem_col = tmem_stage * block_q * heads
                    T.mbarrier_wait_parity(tmem_empty[tmem_stage], tmem_phase ^ 1)
                    T.mbarrier_wait_parity(kv_loaded[stage], parity)
                    T.tcgen05_gemm(
                        kv_shared[stage, 0:half_kv, :],
                        q_shared[q_stage, :, :],
                        c_tmem[:, tmem_col : tmem_col + block_q * heads],
                        transpose_B=True,
                        mbar=tmem_full[tmem_stage],
                        clear_accum=True,
                        disable_ws=True,
                    )
                    gemm_iter = gemm_iter + 1
                    tmem_slot = tmem_slot + 1
                    if tmem_slot == 3:
                        tmem_slot = 0
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 64 <= tx < 96:
            T.dec_max_nreg(40)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            gemm_iter = T.alloc_var(T.int32, init=0)
            tmem_slot = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_tail1 in T.unroll(block_q - 1):
                    qi_scan_tail1 = qi_offset_tail1 + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_tail1])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_tail1])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    stage = gemm_iter % num_stages
                    parity = (gemm_iter // num_stages) & 1
                    tmem_stage = T.alloc_var(T.int32)
                    tmem_phase = T.alloc_var(T.int32)
                    if tmem_slot == 0:
                        tmem_stage = 1
                        tmem_phase = 0
                    elif tmem_slot == 1:
                        tmem_stage = 0
                        tmem_phase = 1
                    else:
                        tmem_stage = 2
                        tmem_phase = 1
                    tmem_col = tmem_stage * block_q * heads
                    T.mbarrier_wait_parity(tmem_empty[tmem_stage], tmem_phase ^ 1)
                    T.mbarrier_wait_parity(kv_loaded[stage], parity)
                    T.tcgen05_gemm(
                        kv_shared[stage, half_kv:block_kv, :],
                        q_shared[q_stage, :, :],
                        c_tmem[:, tmem_col : tmem_col + block_q * heads],
                        transpose_B=True,
                        mbar=tmem_full[tmem_stage],
                        clear_accum=True,
                        disable_ws=True,
                    )
                    gemm_iter = gemm_iter + 1
                    tmem_slot = tmem_slot + 1
                    if tmem_slot == 3:
                        tmem_slot = 0
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 96 <= tx < 128:
            T.dec_max_nreg(40)

        elif 128 <= tx < 256:
            T.inc_max_nreg(232)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            gemm_iter = T.alloc_var(T.int32, init=0)
            tmem_slot = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_epi0 in T.unroll(block_q - 1):
                    qi_scan_epi0 = qi_offset_epi0 + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_epi0])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_epi0])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    kv_row = (first_bkv + kv_iter) * block_kv
                    stage = gemm_iter % num_stages
                    tmem_stage = T.alloc_var(T.int32)
                    tmem_phase = T.alloc_var(T.int32)
                    if tmem_slot == 0:
                        tmem_stage = 0
                        tmem_phase = 0
                    elif tmem_slot == 1:
                        tmem_stage = 2
                        tmem_phase = 0
                    else:
                        tmem_stage = 1
                        tmem_phase = 1
                    tmem_col = tmem_stage * block_q * heads
                    T.mbarrier_wait_parity(tmem_full[tmem_stage], tmem_phase)
                    T.tcgen05_after_thread_sync()
                    bn_epi0 = tx - 128
                    if logits_dtype == T.float32:
                        T.evaluate(T.call_extern(
                            "handle",
                            "tl_mqa_wrelu_tmem_reduce64_v4w_store_f32",
                            c_tmem[0, tmem_col],
                            0 * heads,
                            T.address_of(weights_shared[q_stage, 0, 0]),
                            kv_scale_shared[stage, bn_epi0],
                            T.address_of(Logits[0, 0]),
                            (q_row + 0) * logits_stride + kv_row + bn_epi0,
                        ))
                        T.evaluate(T.call_extern(
                            "handle",
                            "tl_mqa_wrelu_tmem_reduce64_v4w_store_f32",
                            c_tmem[0, tmem_col],
                            1 * heads,
                            T.address_of(weights_shared[q_stage, 1, 0]),
                            kv_scale_shared[stage, bn_epi0],
                            T.address_of(Logits[0, 0]),
                            (q_row + 1) * logits_stride + kv_row + bn_epi0,
                        ))
                    else:
                        result_epi0_q0 = T.call_extern(
                            "float32",
                            "tl_mqa_wrelu_tmem_reduce64_v4w",
                            c_tmem[0, tmem_col],
                            0 * heads,
                            T.address_of(weights_shared[q_stage, 0, 0]),
                            kv_scale_shared[stage, bn_epi0],
                        )
                        Logits[q_row + 0, kv_row + bn_epi0] = T.cast(result_epi0_q0, logits_dtype)
                        result_epi0_q1 = T.call_extern(
                            "float32",
                            "tl_mqa_wrelu_tmem_reduce64_v4w",
                            c_tmem[0, tmem_col],
                            1 * heads,
                            T.address_of(weights_shared[q_stage, 1, 0]),
                            kv_scale_shared[stage, bn_epi0],
                        )
                        Logits[q_row + 1, kv_row + bn_epi0] = T.cast(result_epi0_q1, logits_dtype)
                    T.tcgen05_before_thread_sync()
                    T.mbarrier_arrive(tmem_empty[tmem_stage])
                    T.mbarrier_arrive(kv_empty[stage])
                    gemm_iter = gemm_iter + 1
                    tmem_slot = tmem_slot + 1
                    if tmem_slot == 3:
                        tmem_slot = 0
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        elif 256 <= tx < 384:
            T.inc_max_nreg(232)
            q_block = T.alloc_var(T.int32, init=block_id)
            q_iter = T.alloc_var(T.int32, init=0)
            gemm_iter = T.alloc_var(T.int32, init=0)
            tmem_slot = T.alloc_var(T.int32, init=0)
            while q_block < num_q_blocks:
                q_row = q_block * block_q
                q_stage = q_iter % num_q_stages
                q_phase = (q_iter // num_q_stages) & 1
                tile_min_ks = T.alloc_var(T.int32)
                tile_max_ke = T.alloc_var(T.int32)
                tile_min_ks = KS[q_row]
                tile_max_ke = KE[q_row]
                for qi_offset_epi1 in T.unroll(block_q - 1):
                    qi_scan_epi1 = qi_offset_epi1 + 1
                    tile_min_ks = T.min(tile_min_ks, KS[q_row + qi_scan_epi1])
                    tile_max_ke = T.max(tile_max_ke, KE[q_row + qi_scan_epi1])
                first_bkv = T.alloc_var(T.int32)
                last_bkv = T.alloc_var(T.int32)
                num_kv_blocks = T.alloc_var(T.int32)
                first_bkv = tile_min_ks // block_kv
                last_bkv = T.ceildiv(tile_max_ke, block_kv)
                num_kv_blocks = T.max(last_bkv - first_bkv, 0)

                T.mbarrier_wait_parity(q_loaded[q_stage], q_phase)
                kv_iter = T.alloc_var(T.int32, init=0)
                while kv_iter < num_kv_blocks:
                    kv_row = (first_bkv + kv_iter) * block_kv
                    stage = gemm_iter % num_stages
                    tmem_stage = T.alloc_var(T.int32)
                    tmem_phase = T.alloc_var(T.int32)
                    if tmem_slot == 0:
                        tmem_stage = 1
                        tmem_phase = 0
                    elif tmem_slot == 1:
                        tmem_stage = 0
                        tmem_phase = 1
                    else:
                        tmem_stage = 2
                        tmem_phase = 1
                    tmem_col = tmem_stage * block_q * heads
                    T.mbarrier_wait_parity(tmem_full[tmem_stage], tmem_phase)
                    T.tcgen05_after_thread_sync()
                    bn_epi1 = tx - 256
                    if logits_dtype == T.float32:
                        T.evaluate(T.call_extern(
                            "handle",
                            "tl_mqa_wrelu_tmem_reduce64_v4w_store_f32",
                            c_tmem[0, tmem_col],
                            0 * heads,
                            T.address_of(weights_shared[q_stage, 0, 0]),
                            kv_scale_shared[stage, half_kv + bn_epi1],
                            T.address_of(Logits[0, 0]),
                            (q_row + 0) * logits_stride + kv_row + half_kv + bn_epi1,
                        ))
                        T.evaluate(T.call_extern(
                            "handle",
                            "tl_mqa_wrelu_tmem_reduce64_v4w_store_f32",
                            c_tmem[0, tmem_col],
                            1 * heads,
                            T.address_of(weights_shared[q_stage, 1, 0]),
                            kv_scale_shared[stage, half_kv + bn_epi1],
                            T.address_of(Logits[0, 0]),
                            (q_row + 1) * logits_stride + kv_row + half_kv + bn_epi1,
                        ))
                    else:
                        result_epi1_q0 = T.call_extern(
                            "float32",
                            "tl_mqa_wrelu_tmem_reduce64_v4w",
                            c_tmem[0, tmem_col],
                            0 * heads,
                            T.address_of(weights_shared[q_stage, 0, 0]),
                            kv_scale_shared[stage, half_kv + bn_epi1],
                        )
                        Logits[q_row + 0, kv_row + half_kv + bn_epi1] = T.cast(result_epi1_q0, logits_dtype)
                        result_epi1_q1 = T.call_extern(
                            "float32",
                            "tl_mqa_wrelu_tmem_reduce64_v4w",
                            c_tmem[0, tmem_col],
                            1 * heads,
                            T.address_of(weights_shared[q_stage, 1, 0]),
                            kv_scale_shared[stage, half_kv + bn_epi1],
                        )
                        Logits[q_row + 1, kv_row + half_kv + bn_epi1] = T.cast(result_epi1_q1, logits_dtype)
                    T.tcgen05_before_thread_sync()
                    T.mbarrier_arrive(tmem_empty[tmem_stage])
                    T.mbarrier_arrive(kv_empty[stage])
                    gemm_iter = gemm_iter + 1
                    tmem_slot = tmem_slot + 1
                    if tmem_slot == 3:
                        tmem_slot = 0
                    kv_iter = kv_iter + 1

                T.mbarrier_arrive(q_empty[q_stage])
                q_block = q_block + sm_num
                q_iter = q_iter + 1

        T.sync_threads()

def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double().flatten()
    y = y.double().flatten()
    den = (x * x + y * y).sum()
    if den == 0:
        return 0.0
    return float((1 - 2 * (x * y).sum() / den).item())


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def pack_sf_u8_to_u32_1d(sf_u8: torch.Tensor) -> torch.Tensor:
    assert sf_u8.dtype == torch.uint8
    assert sf_u8.dim() == 2
    _, sf_k_padded = sf_u8.shape
    assert sf_k_padded % 4 == 0
    words = sf_u8.to(torch.int64)
    packed = (
        words[:, 0::4]
        | (words[:, 1::4] << 8)
        | (words[:, 2::4] << 16)
        | (words[:, 3::4] << 24)
    ).to(torch.uint32)
    return packed.T.contiguous().reshape(-1)


_FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def fp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(_FP4_E2M1_VALUES, device=device, dtype=torch.float32)


def quantize_float_to_fp4_packed(x: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    assert k % 2 == 0
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries).reshape(m, k).to(torch.uint8)
    idx = idx | (((x < 0) & (idx != 0)).to(torch.uint8) << 3)
    lo = idx[:, 0::2]
    hi = idx[:, 1::2]
    return (lo | (hi << 4)).to(torch.int8)


def quantize_mxfp4_with_packed_ue8m0(
    x: torch.Tensor, gran_k: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert x.size(1) % 2 == 0
    mn, k = x.shape
    padded_k = _align_up(k, gran_k)
    x_padded = torch.zeros((mn, padded_k), device=x.device, dtype=x.dtype)
    x_padded[:, :k] = x
    x_view = x_padded.view(mn, padded_k // gran_k, gran_k)
    amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = ceil_to_ue8m0(amax / 6.0)
    x_fp4 = quantize_float_to_fp4_packed((x_view * (1.0 / sf.unsqueeze(2))).reshape(mn, padded_k))[
        :, : k // 2
    ].contiguous()
    sf_u8 = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    sf_k_padded = _align_up(sf_u8.shape[1], 4)
    if sf_k_padded != sf_u8.shape[1]:
        sf_padded = torch.full((mn, sf_k_padded), 127, device=x.device, dtype=torch.uint8)
        sf_padded[:, : sf_u8.shape[1]] = sf_u8
    else:
        sf_padded = sf_u8
    return x_fp4, pack_sf_u8_to_u32_1d(sf_padded), sf_u8


def cast_back_from_mxfp4(
    x_fp4: torch.Tensor, sf_packed: torch.Tensor, logical_k: int, gran_k: int = 32
) -> torch.Tensor:
    u = x_fp4.contiguous().view(torch.uint8)
    lut = fp4_lut(u.device)
    lo = lut[(u & 0x0F).long()]
    hi = lut[((u >> 4) & 0x0F).long()]
    x = torch.empty((u.shape[0], logical_k), device=u.device, dtype=torch.float32)
    x[:, 0::2] = lo[:, : logical_k // 2]
    x[:, 1::2] = hi[:, : logical_k // 2]
    sf_k_blocks = _ceil_div(logical_k, gran_k)
    sf_groups = _ceil_div(sf_k_blocks, 4)
    packed = sf_packed.view(sf_groups, u.shape[0]).T.contiguous().to(torch.int64)
    sf_u8 = torch.empty((u.shape[0], sf_groups * 4), device=u.device, dtype=torch.uint8)
    for i in range(4):
        sf_u8[:, i::4] = ((packed >> (8 * i)) & 0xFF).to(torch.uint8)
    scales = torch.pow(2.0, sf_u8[:, :sf_k_blocks].to(torch.float32) - 127.0)
    for bi in range(sf_k_blocks):
        k0 = bi * gran_k
        k1 = min(k0 + gran_k, logical_k)
        x[:, k0:k1] *= scales[:, bi : bi + 1]
    return x


def local_per_custom_dims_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().float().amax(dim=1).clamp_min(1e-4)
    scale = (amax / 448.0).contiguous()
    return (x.float() * (1.0 / scale[:, None])).to(torch.float8_e4m3fn).contiguous(), scale


def prepare_mqa_data(config: MQALogitsConfig, dtype: str):
    config.validate()
    torch.manual_seed(config.seed)
    q = torch.randn(config.seq_len, config.num_heads, config.head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(config.seq_len_kv, config.head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(config.seq_len, config.num_heads, device="cuda", dtype=torch.float32)
    ks, ke = generate_ks_ke(config)

    if dtype == "fp8":
        q_in = q.to(torch.float8_e4m3fn).contiguous()
        kv_in = local_per_custom_dims_cast_to_fp8(kv)
        q_sim = q_in.to(torch.bfloat16)
        kv_sim = (kv_in[0].float() * kv_in[1].unsqueeze(1)).to(torch.bfloat16)
        return {
            "q": q_sim,
            "kv": kv_sim,
            "q_in": q_in,
            "kv_in": kv_in,
            "weights": weights,
            "ks": ks,
            "ke": ke,
        }

    if dtype != "fp4":
        raise ValueError(f"unsupported dtype: {dtype}")
    if config.logits_dtype != "float32":
        raise ValueError("the FP4 SOTA kernel currently stores float32 logits")
    if config.seq_len_kv % 256 != 0:
        raise ValueError("seq_len_kv must be divisible by 256 for the FP4 SOTA tile")

    q_fp4 = quantize_mxfp4_with_packed_ue8m0(q.view(-1, config.head_dim), gran_k=32)
    kv_fp4 = quantize_mxfp4_with_packed_ue8m0(kv.view(-1, config.head_dim), gran_k=32)
    q_sim = cast_back_from_mxfp4(q_fp4[0], q_fp4[1], config.head_dim, gran_k=32).view(
        config.seq_len, config.num_heads, config.head_dim
    )
    kv_sim = cast_back_from_mxfp4(kv_fp4[0], kv_fp4[1], config.head_dim, gran_k=32).view(
        config.seq_len_kv, config.head_dim
    )
    q_in = (
        q_fp4[0].view(config.seq_len, config.num_heads, config.head_dim // 2).contiguous(),
        q_fp4[1].view(config.seq_len, config.num_heads).contiguous(),
    )
    kv_in = (
        kv_fp4[0].view(config.seq_len_kv, config.head_dim // 2).contiguous(),
        kv_fp4[1].view(config.seq_len_kv).contiguous(),
    )
    return {
        "q": q_sim.to(torch.bfloat16),
        "kv": kv_sim.to(torch.bfloat16),
        "q_in": q_in,
        "kv_in": kv_in,
        "weights": weights,
        "ks": ks,
        "ke": ke,
    }


def run_fp8(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    logits_dtype: str = "float32",
) -> torch.Tensor:
    seq_len, heads, head_dim = q_fp8.shape
    seq_len_kv = kv_fp8.shape[0]
    MQALogitsConfig(seq_len, seq_len_kv, heads, head_dim, logits_dtype).validate()
    logits = torch.full(
        (seq_len, seq_len_kv),
        float("-inf"),
        device=q_fp8.device,
        dtype=_torch_logits_dtype(logits_dtype),
    )
    mqa_logits_fp8_persistent_ws_kernel(
        q_fp8.reshape(seq_len * heads, head_dim),
        kv_fp8,
        kv_scale,
        weights,
        ks,
        ke,
        logits,
        seq_len,
        seq_len_kv,
        heads=heads,
        head_dim=head_dim,
        logits_stride=seq_len_kv,
        compressed_logits=False,
        logits_dtype=_tilelang_logits_dtype(logits_dtype),
    )
    return logits


def run_fp4(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_fp4: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    logits_dtype: str = "float32",
) -> torch.Tensor:
    seq_len, heads, head_dim_packed = q_fp4.shape
    head_dim = head_dim_packed * 2
    seq_len_kv = kv_fp4.shape[0]
    MQALogitsConfig(seq_len, seq_len_kv, heads, head_dim, logits_dtype).validate()
    if logits_dtype != "float32":
        raise ValueError("the FP4 SOTA kernel currently stores float32 logits")
    if seq_len_kv % 256 != 0:
        raise ValueError("seq_len_kv must be divisible by 256 for the FP4 SOTA tile")
    logits = torch.full((seq_len, seq_len_kv), float("-inf"), device=q_fp4.device, dtype=torch.float32)
    mqa_logits_fp4_persistent_ws_kernel(
        q_fp4.reshape(seq_len * heads, head_dim_packed),
        q_scale.reshape(-1),
        kv_fp4,
        kv_scale.reshape(-1),
        weights,
        ks,
        ke,
        logits,
        seq_len,
        seq_len_kv,
        heads=heads,
        head_dim=head_dim,
        logits_stride=seq_len_kv,
        compressed_logits=False,
        logits_dtype=T.float32,
    )
    return logits


def run_example_case(config: MQALogitsConfig, dtype: str, check: bool = True) -> None:
    data = prepare_mqa_data(config, dtype)
    ref = ref_mqa_logits(data["q"], data["kv"], data["weights"], data["ks"], data["ke"])
    if dtype == "fp8":
        kv_fp8, kv_scale = data["kv_in"]
        out = run_fp8(data["q_in"], kv_fp8, kv_scale, data["weights"], data["ks"], data["ke"], config.logits_dtype)
    elif dtype == "fp4":
        q_fp4, q_scale = data["q_in"]
        kv_fp4, kv_scale = data["kv_in"]
        out = run_fp4(q_fp4, q_scale, kv_fp4, kv_scale, data["weights"], data["ks"], data["ke"], config.logits_dtype)
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    observed = out.float().masked_fill(ref == float("-inf"), 0)
    ref_cmp = ref.masked_fill(ref == float("-inf"), 0)
    diff = calc_diff(observed, ref_cmp)
    if check:
        threshold = 2e-3 if dtype == "fp4" else 1e-4
        assert diff < threshold, f"{dtype} diff {diff} >= {threshold}"
    print(
        f"{dtype} s{config.seq_len}_skv{config.seq_len_kv}_h{config.num_heads}_d{config.head_dim}_"
        f"{config.logits_dtype}: diff={diff:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone SM100 MQA logits SOTA kernels.")
    parser.add_argument("--dtype", choices=("fp8", "fp4", "both"), default="both")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seq-len-kv", type=int, default=4096)
    parser.add_argument("--logits-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    cfg = MQALogitsConfig(
        seq_len=args.seq_len,
        seq_len_kv=args.seq_len_kv,
        logits_dtype=args.logits_dtype,
        seed=args.seed,
    )
    dtypes = ("fp8", "fp4") if args.dtype == "both" else (args.dtype,)
    for dtype in dtypes:
        run_example_case(cfg, dtype, check=not args.no_check)


if __name__ == "__main__":
    main()
