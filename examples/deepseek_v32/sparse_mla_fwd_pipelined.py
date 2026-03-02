# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from tilelang.engine.callback import register_cuda_postproc_callback
import argparse

tilelang.disable_cache()


def post_proc(source, _):
    source = r"""
#include <tl_templates/cuda/instruction/wgmma.h>
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const int* __restrict__ Indices, const bfloat16_t* __restrict__ KV, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, const int* __restrict__ q_start_index_s);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(const int* __restrict__ Indices, const bfloat16_t* __restrict__ KV, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, const int* __restrict__ q_start_index_s) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t bar_q_mem[1];
  auto bar_q = reinterpret_cast<Barrier*>(bar_q_mem);
  __shared__ __align__(16) uint64_t bar_k_0_ready_mem[1];
  auto bar_k_0_ready = reinterpret_cast<Barrier*>(bar_k_0_ready_mem);
  __shared__ __align__(16) uint64_t bar_k_1_ready_mem[1];
  auto bar_k_1_ready = reinterpret_cast<Barrier*>(bar_k_1_ready_mem);
  __shared__ __align__(16) uint64_t bar_k_0_free_mem[1];
  auto bar_k_0_free = reinterpret_cast<Barrier*>(bar_k_0_free_mem);
  __shared__ __align__(16) uint64_t bar_k_1_free_mem[1];
  auto bar_k_1_free = reinterpret_cast<Barrier*>(bar_k_1_free_mem);
  __shared__ __align__(16) uint64_t bar_sScale_and_sS_ready_mem[1];
  auto bar_sScale_and_sS_ready = reinterpret_cast<Barrier*>(bar_sScale_and_sS_ready_mem);
  __shared__ __align__(16) uint64_t bar_sScale_and_sS_free_mem[1];
  auto bar_sScale_and_sS_free = reinterpret_cast<Barrier*>(bar_sScale_and_sS_free_mem);
  float sumexp[2];
  float m_i[2];
  float acc_o_l[128];
  __shared__ __align__(16) signed char is_kv_valid[64];
  float acc_s[32];
  float m_i_prev[2];
  float alpha_local[2];
  float sumexp_i[2];
  __shared__ __align__(16) float alpha_shared[64];
  float acc_o_r[128];
  int indices_local = 0;
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  tl::GmmaDescriptor desc_a_1;
  tl::GmmaDescriptor desc_b_1;
  tl::GmmaDescriptor desc_a_2;
  tl::GmmaDescriptor desc_b_2;
  float m_i_clear[2];
  tl::GmmaDescriptor desc_a_3;
  tl::GmmaDescriptor desc_b_3;
  tl::GmmaDescriptor desc_a_4;
  tl::GmmaDescriptor desc_b_4;
  tl::GmmaDescriptor desc_a_5;
  tl::GmmaDescriptor desc_b_5;
  tl::GmmaDescriptor desc_a_6;
  tl::GmmaDescriptor desc_b_6;
  float m_i_clear_1[2];
  tl::GmmaDescriptor desc_a_7;
  tl::GmmaDescriptor desc_b_7;
  tl::GmmaDescriptor desc_a_8;
  tl::GmmaDescriptor desc_b_8;
  tl::GmmaDescriptor desc_a_9;
  tl::GmmaDescriptor desc_b_9;
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    bar_q[0].init(384);
    bar_k_0_ready[0].init(128);
    bar_k_1_ready[0].init(128);
    bar_k_0_free[0].init(256);
    bar_k_1_free[0].init(256);
    bar_sScale_and_sS_ready[0].init(256);
    bar_sScale_and_sS_free[0].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  int q_i = ((((int)blockIdx.x) >> 1) + q_start_index_s[0]);
  if (((int)threadIdx.x) == 0) {
    bar_q[0].expect_transaction(32768);
    tl::tma_load(Q_desc, bar_q[0], (&(((bfloat16_t*)buf_dyn_shmem)[0])), 0, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
    tl::tma_load(Q_desc, bar_q[0], (&(((bfloat16_t*)buf_dyn_shmem)[4096])), 64, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
    tl::tma_load(Q_desc, bar_q[0], (&(((bfloat16_t*)buf_dyn_shmem)[8192])), 128, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
    tl::tma_load(Q_desc, bar_q[0], (&(((bfloat16_t*)buf_dyn_shmem)[12288])), 192, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
    bar_q[0].expect_transaction(32768);
    tl::tma_load(Q_desc, bar_q[0], (&(((bfloat16_t*)buf_dyn_shmem)[16384])), 256, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
    tl::tma_load(Q_desc, bar_q[0], (&(((bfloat16_t*)buf_dyn_shmem)[20480])), 320, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
    tl::tma_load(Q_desc, bar_q[0], (&(((bfloat16_t*)buf_dyn_shmem)[24576])), 384, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
    tl::tma_load(Q_desc, bar_q[0], (&(((bfloat16_t*)buf_dyn_shmem)[28672])), 448, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
    bar_q[0].expect_transaction(8192);
    tl::tma_load(Q_desc, bar_q[0], (&(((bfloat16_t*)buf_dyn_shmem)[32768])), 512, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
  }
  bar_q[0].arrive();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<240>();
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(sumexp + 0) = make_float2(broadcast_var, broadcast_var);
    float broadcast_var_1 = -0x1p+30f/*-1.073742e+09*/;
    *(float2*)(m_i + 0) = make_float2(broadcast_var_1, broadcast_var_1);
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o_l + (i * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    bar_q[0].wait(0);
    for (int i_i = 0; i_i < 16; ++i_i) {
      bar_k_0_ready[0].wait((i_i & 1));
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        for (int vec_s = 0; vec_s < 2; ++vec_s) {
          float condval;
          if (((bool)is_kv_valid[((((i_1 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + vec_s)])) {
            condval = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval = -CUDART_INF_F;
          }
          acc_s[((i_1 * 2) + vec_s)] = condval;
        }
      }
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)buf_dyn_shmem)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((bfloat16_t*)buf_dyn_shmem)[36864])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki = 0; ki < 16; ++ki) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((bfloat16_t*)buf_dyn_shmem)[16384])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((bfloat16_t*)buf_dyn_shmem)[53248])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 16; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), uint64_t(desc_b_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((bfloat16_t*)buf_dyn_shmem)[32768])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((bfloat16_t*)buf_dyn_shmem)[102400])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((ki_2 * 32) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 32) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::wait_wgmma<0>();
      if (0 < i_i) {
        bar_sScale_and_sS_free[0].arrive();
        bar_sScale_and_sS_free[0].wait((((i_i * 2) & 1) ^ 1));
      }
      *(float2*)(m_i_prev + 0) = *(float2*)(m_i + 0);
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        m_i_clear[i_2] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 16; ++rv) {
          m_i_clear[i_2] = max(m_i_clear[i_2], acc_s[((((rv & 7) * 4) + (i_2 * 2)) + (rv >> 3))]);
        }
        m_i_clear[i_2] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_i_clear[i_2]);
        m_i[i_2] = max(m_i[i_2], m_i_clear[i_2]);
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        m_i[i_3] = max(m_i[i_3], m_i_prev[i_3]);
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        alpha_local[i_4] = exp2f(((m_i_prev[i_4] - m_i[i_4]) * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        acc_s[i_5] = exp2f(((acc_s[i_5] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (m_i[((i_5 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        sumexp_i[i_6] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
          sumexp_i[i_6] = (sumexp_i[i_6] + acc_s[((((rv_1 & 7) * 4) + (i_6 * 2)) + (rv_1 >> 3))]);
        }
        sumexp_i[i_6] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(sumexp_i[i_6]);
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        sumexp[i_7] = ((sumexp[i_7] * alpha_local[i_7]) + sumexp_i[i_7]);
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 128; ++i_8) {
        acc_o_l[i_8] = (acc_o_l[i_8] * alpha_local[((i_8 & 3) >> 1)]);
      }
      tl::__sync_thread_partial<3, 128>();
      if ((((int)threadIdx.x) % 4) == 0) {
        #pragma unroll
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          alpha_shared[((((((int)threadIdx.x) >> 5) * 16) + (i_9 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = alpha_local[i_9];
        }
      }
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_10 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 110592)])), __pack_half2(((bfloat16_t)acc_s[(i_10 * 8)]), ((bfloat16_t)acc_s[((i_10 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_s[((i_10 * 8) + 2)]), ((bfloat16_t)acc_s[((i_10 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_s[((i_10 * 8) + 4)]), ((bfloat16_t)acc_s[((i_10 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_s[((i_10 * 8) + 6)]), ((bfloat16_t)acc_s[((i_10 * 8) + 7)])));
      }
      tl::__sync_thread_partial<3, 128>();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((bfloat16_t*)buf_dyn_shmem)[110592])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_3, (&(((bfloat16_t*)buf_dyn_shmem)[36864])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_l + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_3 + ((ki_3 * 32) >> 4)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), ((uint32_t*)(acc_o_l + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_l + 0), 128);
      bar_sScale_and_sS_ready[0].arrive();
      bar_k_0_free[0].arrive();
      bar_k_1_ready[0].wait((i_i & 1));
      #pragma unroll
      for (int i_11 = 0; i_11 < 16; ++i_11) {
        for (int vec_s_1 = 0; vec_s_1 < 2; ++vec_s_1) {
          float condval_1;
          if (((bool)is_kv_valid[((((i_11 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1)])) {
            condval_1 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_1 = -CUDART_INF_F;
          }
          acc_s[((i_11 * 2) + vec_s_1)] = condval_1;
        }
      }
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((bfloat16_t*)buf_dyn_shmem)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_4, (&(((bfloat16_t*)buf_dyn_shmem)[69632])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_4 = 0; ki_4 < 16; ++ki_4) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_4 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), uint64_t(desc_b_4 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((bfloat16_t*)buf_dyn_shmem)[16384])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_5, (&(((bfloat16_t*)buf_dyn_shmem)[86016])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_5 = 0; ki_5 < 16; ++ki_5) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_5 + ((((ki_5 >> 2) * 8192) + ((ki_5 & 3) * 32)) >> 4)), uint64_t(desc_b_5 + ((((ki_5 >> 2) * 8192) + ((ki_5 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((bfloat16_t*)buf_dyn_shmem)[32768])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_6, (&(((bfloat16_t*)buf_dyn_shmem)[106496])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_6 + ((ki_6 * 32) >> 4)), uint64_t(desc_b_6 + ((ki_6 * 32) >> 4)), ((uint32_t*)(acc_s + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 32);
      tl::wait_wgmma<0>();
      bar_sScale_and_sS_free[0].arrive();
      bar_sScale_and_sS_free[0].wait(((((i_i * 2) + 1) & 1) ^ 1));
      *(float2*)(m_i_prev + 0) = *(float2*)(m_i + 0);
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        m_i_clear_1[i_12] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
          m_i_clear_1[i_12] = max(m_i_clear_1[i_12], acc_s[((((rv_2 & 7) * 4) + (i_12 * 2)) + (rv_2 >> 3))]);
        }
        m_i_clear_1[i_12] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_i_clear_1[i_12]);
        m_i[i_12] = max(m_i[i_12], m_i_clear_1[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        m_i[i_13] = max(m_i[i_13], m_i_prev[i_13]);
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        alpha_local[i_14] = exp2f(((m_i_prev[i_14] - m_i[i_14]) * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/));
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 32; ++i_15) {
        acc_s[i_15] = exp2f(((acc_s[i_15] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (m_i[((i_15 & 3) >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        sumexp_i[i_16] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
          sumexp_i[i_16] = (sumexp_i[i_16] + acc_s[((((rv_3 & 7) * 4) + (i_16 * 2)) + (rv_3 >> 3))]);
        }
        sumexp_i[i_16] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(sumexp_i[i_16]);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        sumexp[i_17] = ((sumexp[i_17] * alpha_local[i_17]) + sumexp_i[i_17]);
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 128; ++i_18) {
        acc_o_l[i_18] = (acc_o_l[i_18] * alpha_local[((i_18 & 3) >> 1)]);
      }
      tl::__sync_thread_partial<3, 128>();
      if ((((int)threadIdx.x) % 4) == 0) {
        #pragma unroll
        for (int i_19 = 0; i_19 < 2; ++i_19) {
          alpha_shared[((((((int)threadIdx.x) >> 5) * 16) + (i_19 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = alpha_local[i_19];
        }
      }
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_20 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 110592)])), __pack_half2(((bfloat16_t)acc_s[(i_20 * 8)]), ((bfloat16_t)acc_s[((i_20 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_s[((i_20 * 8) + 2)]), ((bfloat16_t)acc_s[((i_20 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_s[((i_20 * 8) + 4)]), ((bfloat16_t)acc_s[((i_20 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_s[((i_20 * 8) + 6)]), ((bfloat16_t)acc_s[((i_20 * 8) + 7)])));
      }
      tl::__sync_thread_partial<3, 128>();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_7, (&(((bfloat16_t*)buf_dyn_shmem)[110592])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_7, (&(((bfloat16_t*)buf_dyn_shmem)[69632])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_l + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_7 + ((ki_7 * 32) >> 4)), uint64_t(desc_b_7 + ((ki_7 * 2048) >> 4)), ((uint32_t*)(acc_o_l + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_l + 0), 128);
      bar_sScale_and_sS_ready[0].arrive();
      bar_k_1_free[0].arrive();
    }
    tl::__sync_thread_partial<3, 128>();
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + (i_21 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 57344)] = sumexp[i_21];
      }
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 128; ++i_22) {
      acc_o_l[i_22] = (acc_o_l[i_22] / sumexp[((i_22 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      sumexp[i_23] = (log2f(sumexp[i_23]) + (m_i[i_23] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/));
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_24 = 0; i_24 < 16; ++i_24) {
      tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_24 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_24 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_24 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)acc_o_l[(i_24 * 8)]), ((bfloat16_t)acc_o_l[((i_24 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_o_l[((i_24 * 8) + 2)]), ((bfloat16_t)acc_o_l[((i_24 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_o_l[((i_24 * 8) + 4)]), ((bfloat16_t)acc_o_l[((i_24 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_o_l[((i_24 * 8) + 6)]), ((bfloat16_t)acc_o_l[((i_24 * 8) + 7)])));
    }
    tl::__sync_thread_partial<3, 128>();
    if (((int)threadIdx.x) == 0) {
      tl::fence_proxy_async();
      tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), 0, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
      tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[4096])), 64, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
      tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[8192])), 128, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
      tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[12288])), 192, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<168>();
      #pragma unroll
      for (int i_25 = 0; i_25 < 32; ++i_25) {
        float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_o_r + (i_25 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
      }
      for (int i_i_1 = 0; i_i_1 < 16; ++i_i_1) {
        bar_sScale_and_sS_ready[0].arrive();
        bar_sScale_and_sS_ready[0].wait(((i_i_1 * 2) & 1));
        #pragma unroll
        for (int i_26 = 0; i_26 < 64; ++i_26) {
          float2 __1;
            float2 v_ = *(float2*)(acc_o_r + (i_26 * 2));
            float2 v__1 = make_float2(alpha_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_26 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)], alpha_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_26 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)]);
            __1.x = (v_.x*v__1.x);
            __1.y = (v_.y*v__1.y);
          *(float2*)(acc_o_r + (i_26 * 2)) = __1;
        }
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_8, (&(((bfloat16_t*)buf_dyn_shmem)[110592])));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_8, (&(((bfloat16_t*)buf_dyn_shmem)[53248])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_r + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_8 = 0; ki_8 < 4; ++ki_8) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_8 + ((ki_8 * 32) >> 4)), uint64_t(desc_b_8 + ((ki_8 * 2048) >> 4)), ((uint32_t*)(acc_o_r + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_r + 0), 128);
        bar_k_0_free[0].arrive();
        bar_sScale_and_sS_free[0].arrive();
        bar_sScale_and_sS_ready[0].arrive();
        bar_sScale_and_sS_ready[0].wait((((i_i_1 * 2) + 1) & 1));
        #pragma unroll
        for (int i_27 = 0; i_27 < 64; ++i_27) {
          float2 __2;
            float2 v__2 = *(float2*)(acc_o_r + (i_27 * 2));
            float2 v__3 = make_float2(alpha_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)], alpha_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)]);
            __2.x = (v__2.x*v__3.x);
            __2.y = (v__2.y*v__3.y);
          *(float2*)(acc_o_r + (i_27 * 2)) = __2;
        }
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_9, (&(((bfloat16_t*)buf_dyn_shmem)[110592])));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_9, (&(((bfloat16_t*)buf_dyn_shmem)[86016])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_r + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 256, 16, false, true, 1, 1>(uint64_t(desc_a_9 + ((ki_9 * 32) >> 4)), uint64_t(desc_b_9 + ((ki_9 * 2048) >> 4)), ((uint32_t*)(acc_o_r + 0)), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_r + 0), 128);
        bar_k_1_free[0].arrive();
        if (i_i_1 < 15) {
          bar_sScale_and_sS_free[0].arrive();
        }
      }
      #pragma unroll
      for (int i_28 = 0; i_28 < 64; ++i_28) {
        float2 __3;
          float2 v__4 = *(float2*)(acc_o_r + (i_28 * 2));
          float2 v__5 = make_float2(((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_28 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 57280)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_28 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 57280)]);
          __3.x = (v__4.x/v__5.x);
          __3.y = (v__4.y/v__5.y);
        *(float2*)(acc_o_r + (i_28 * 2)) = __3;
      }
      tl::__sync_thread_partial<4, 128>();
      #pragma unroll
      for (int i_29 = 0; i_29 < 16; ++i_29) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((i_29 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_29 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_29 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), __pack_half2(((bfloat16_t)acc_o_r[(i_29 * 8)]), ((bfloat16_t)acc_o_r[((i_29 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_o_r[((i_29 * 8) + 2)]), ((bfloat16_t)acc_o_r[((i_29 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_o_r[((i_29 * 8) + 4)]), ((bfloat16_t)acc_o_r[((i_29 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_o_r[((i_29 * 8) + 6)]), ((bfloat16_t)acc_o_r[((i_29 * 8) + 7)])));
      }
      tl::__sync_thread_partial<4, 128>();
      if (((int)threadIdx.x) == 128) {
        tl::fence_proxy_async();
        tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[16384])), 256, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
        tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[20480])), 320, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
        tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[24576])), 384, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
        tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[28672])), 448, ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), 0);
        tl::tma_store_arrive();
        tl::tma_store_wait<0>();
      }
    } else {
      tl::warpgroup_reg_dealloc<80>();
      for (int i_i_2 = 0; i_i_2 < 16; ++i_i_2) {
        bar_k_0_free[0].wait(((i_i_2 & 1) ^ 1));
        tl::__sync_thread_partial<4, 128>();
        for (int r = 0; r < 4; ++r) {
          indices_local = Indices[((((((((int)blockIdx.x) >> 1) * 2048) + (i_i_2 * 128)) + (r * 16)) + (((int)threadIdx.x) >> 3)) - 32)];
          is_kv_valid[(((r * 16) + (((int)threadIdx.x) >> 3)) - 32)] = ((signed char)(indices_local <= q_i));
          tl::__sync_thread_partial<4, 128>();
          if ((bool)is_kv_valid[(((r * 16) + (((int)threadIdx.x) >> 3)) - 32)]) {
            for (int u = 0; u < 4; ++u) {
              tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((u * 4096) + (r * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 34816)])), (&(KV[(((indices_local * 576) + (u * 64)) + ((((int)threadIdx.x) & 7) * 8))])), ((0 <= indices_local) && (indices_local < 8192)));
              tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((u * 4096) + (r * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 51200)])), (&(KV[((((indices_local * 576) + (u * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 256)])), ((0 <= indices_local) && (indices_local < 8192)));
            }
            tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[((((((r * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 100352)])), (&(KV[(((indices_local * 576) + ((((int)threadIdx.x) & 7) * 8)) + 512)])), ((0 <= indices_local) && (indices_local < 8192)));
          }
        }
        // tl::cp_async_commit();
        // tl::cp_async_wait<0>();
        tl::__sync_thread_partial<4, 128>();
        tl::mbarrier_cp_async_arrive_noinc(bar_k_0_ready[0]);
        bar_k_1_free[0].wait(((i_i_2 & 1) ^ 1));
        for (int r_1 = 0; r_1 < 4; ++r_1) {
          indices_local = Indices[((((((((int)blockIdx.x) >> 1) * 2048) + (i_i_2 * 128)) + (r_1 * 16)) + (((int)threadIdx.x) >> 3)) + 32)];
          is_kv_valid[(((r_1 * 16) + (((int)threadIdx.x) >> 3)) - 32)] = ((signed char)(indices_local <= q_i));
          tl::__sync_thread_partial<4, 128>();
          if ((bool)is_kv_valid[(((r_1 * 16) + (((int)threadIdx.x) >> 3)) - 32)]) {
            for (int u_1 = 0; u_1 < 4; ++u_1) {
              tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((u_1 * 4096) + (r_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 67584)])), (&(KV[(((indices_local * 576) + (u_1 * 64)) + ((((int)threadIdx.x) & 7) * 8))])), ((0 <= indices_local) && (indices_local < 8192)));
              tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((u_1 * 4096) + (r_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 83968)])), (&(KV[((((indices_local * 576) + (u_1 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 256)])), ((0 <= indices_local) && (indices_local < 8192)));
            }
            tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[((((((r_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 104448)])), (&(KV[(((indices_local * 576) + ((((int)threadIdx.x) & 7) * 8)) + 512)])), ((0 <= indices_local) && (indices_local < 8192)));
          }
        }
        // tl::cp_async_commit();
        // tl::cp_async_wait<0>();
        tl::mbarrier_cp_async_arrive_noinc(bar_k_1_ready[0]);
        //
      }
    }
  }
}
    """
    return source


# tilelang.register_cuda_postproc(post_proc)


@tilelang.jit(
    out_idx=[-2, -1],
    compile_flags=[
        "-O3",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10",
        "-DNDEBUG",
    ],
)
def sparse_mla_fwd(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    tail_dim,
    topk,
    kv_stride,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=0,
    threads=384,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    assert NI % 2 == 0, "NI should be a multiple of 2"
    D = dim
    D_tail = tail_dim
    KV_stride = kv_stride
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor(1, indices_dtype),
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel((seq_len - kv_stride + 1 if CP0 else seq_len) * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
            K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
            K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
            O_shared_l = Q_shared_l
            O_shared_r = Q_shared_r
            is_kv_valid = T.alloc_shared([BI], "bool", scope="shared")

            acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
            acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sum_exp_shared = T.alloc_shared([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([H_per_block], accum_dtype, scope="shared")
            alpha_local = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            indices_local = T.alloc_var(indices_dtype)

            # TODO: Multi buffer
            bar_q = T.alloc_barrier(arrive_count=384)
            bar_k_0_ready = T.alloc_barrier(arrive_count=128)
            bar_k_1_ready = T.alloc_barrier(arrive_count=128)
            bar_k_0_free = T.alloc_barrier(arrive_count=256)
            bar_k_1_free = T.alloc_barrier(arrive_count=256)
            bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
            bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

            b_i, g_i = by, bz
            s_i = (bx + (KV_stride - 1 if CP0 else 0)) if REPLICATE_H == 1 else (bx // REPLICATE_H + (KV_stride - 1 if CP0 else 0))
            q_i = q_start_index_s[0] + s_i
            max_kv_i = (q_i + 1 - KV_stride) // KV_stride

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            tx = T.get_thread_binding()

            T.copy(Q[b_i, s_i, H0:H1, 0 : D // 2], Q_shared_l)
            T.copy(Q[b_i, s_i, H0:H1, D // 2 : D], Q_shared_r)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)
            T.barrier_arrive(bar_q)

            if tx < 128:
                T.set_max_nreg(240, 1)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
                T.fill(acc_o_l, 0)
                T.barrier_wait(bar_q, 0)

                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    if i_i != 0:
                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_arrive(bar_k_0_free[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    T.barrier_arrive(bar_sScale_and_sS_free)
                    T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_arrive(bar_k_1_free[0])

                # Rescale
                for h_i in T.Parallel(H_per_block):
                    sum_exp_shared[h_i] = sumexp[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_l[h_i, d_i] /= sumexp[h_i]
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                T.copy(acc_o_l, O_shared_l)
                T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0 : D // 2])

            elif tx >= 128 and tx < 256:
                T.set_max_nreg(168, 1)
                T.fill(acc_o_r, 0)
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                    T.barrier_arrive(bar_k_0_free[0])
                    T.barrier_arrive(bar_sScale_and_sS_free)

                    # Buffer 1
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                    T.barrier_arrive(bar_k_1_free[0])
                    if i_i != T.ceildiv(NI, 2) - 1:
                        T.barrier_arrive(bar_sScale_and_sS_free)

                # Rescale
                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                T.copy(acc_o_r, O_shared_r)
                T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D // 2 : D])
            elif tx >= 256:
                # producer
                T.set_max_nreg(80, 0)
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local = Indices[b_i, s_i, g_i, (i_i * 2) * BI + r * 16 + (tx - 256) // 8]
                        is_kv_valid[r * 16 + (tx - 256) // 8] = indices_local <= max_kv_i
                        if is_kv_valid[r * 16 + (tx - 256) // 8]:
                            # Manually issue cp.async copies for KV_left, KV_right, and K_tail.
                            for u in T.serial(4):
                                T.ptx_cp_async(
                                    T.access_ptr(KV_shared_0_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8], "w", 8),
                                    T.access_ptr(KV[b_i, indices_local, g_i, 64 * u + (tx - 256) % 8 * 8], "r", 8),
                                    16,
                                )
                                T.ptx_cp_async(
                                    T.access_ptr(KV_shared_0_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8], "w", 8),
                                    T.access_ptr(KV[b_i, indices_local, g_i, D // 2 + 64 * u + (tx - 256) % 8 * 8], "r", 8),
                                    16,
                                )
                            T.ptx_cp_async(
                                T.access_ptr(K_tail_shared_0[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8], "w", 8),
                                T.access_ptr(KV[b_i, indices_local, g_i, D + (tx - 256) % 8 * 8], "r", 8),
                                16,
                            )
                    T.cp_async_barrier_noinc(bar_k_0_ready[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local = Indices[b_i, s_i, g_i, (i_i * 2 + 1) * BI + r * 16 + (tx - 256) // 8]
                        is_kv_valid[r * 16 + (tx - 256) // 8] = indices_local <= max_kv_i
                        if is_kv_valid[r * 16 + (tx - 256) // 8]:
                            # Manually issue cp.async copies for KV_left, KV_right, and K_tail.
                            for u in T.serial(4):
                                T.ptx_cp_async(
                                    T.access_ptr(KV_shared_1_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8], "w", 8),
                                    T.access_ptr(KV[b_i, indices_local, g_i, 64 * u + (tx - 256) % 8 * 8], "r", 8),
                                    16,
                                )
                                T.ptx_cp_async(
                                    T.access_ptr(KV_shared_1_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8], "w", 8),
                                    T.access_ptr(KV[b_i, indices_local, g_i, D // 2 + 64 * u + (tx - 256) % 8 * 8], "r", 8),
                                    16,
                                )
                            T.ptx_cp_async(
                                T.access_ptr(K_tail_shared_1[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8], "w", 8),
                                T.access_ptr(KV[b_i, indices_local, g_i, D + (tx - 256) % 8 * 8], "r", 8),
                                16,
                            )
                    T.cp_async_barrier_noinc(bar_k_1_ready[0])

    return main


def sparse_mla_fwd_interface(
    q, kv, indices, q_start_index_s, kv_stride, sm_scale=None, is_casual=True, return_kernel=False, print_kernel=False
):
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = 512

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    if q_start_index_s != 0:
        assert q_start_index_s > kv_stride, (
            "If it is because each cp has too short length, you should fix the logic involving CP0 (cp_rank == 0), to make sure q with pos < KV_Stride - 1 is masked (or you may just ignore how this is handled if nan in these q's Out would not effect others, which is reported to be likely to happen by wangding)"
        )
    CP0 = q_start_index_s == 0

    kernel = sparse_mla_fwd(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride, kv_group, sm_scale, is_casual, CP0)
    if print_kernel:
        print(kernel.get_kernel_source())
    out, lse = kernel(q, kv, indices, torch.tensor([q_start_index_s], dtype=torch.int32, device="cuda"))
    if return_kernel:
        return kernel
    if q_start_index_s == 0 and kv_stride > 1:
        out[:, : kv_stride - 1, :, :] = 0
    return out, lse


def ref_sparse_mla_fwd_interface(q, kv, indices, q_start_index_s, kv_stride=4, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape
    if q_start_index_s is None:
        q_start_index_s = sk * kv_stride - sq

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    num_kv_per_index = 1
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(q_start_index_s, sq + q_start_index_s, dtype=torch.int32, device="cuda").view(
        -1, 1
    ) >= torch.arange(kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32, device="cuda").view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def test_sparse_mla_fwd_pipelined(
    B=1, S=4096, SKV=8192, H=128, HKV=1, DQK=576, DV=512, topk=2048, dtype=torch.bfloat16, q_start_s_index=1024, check_correctness=True
):
    KV_stride = 1

    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True) / 10
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True) / 10
    q_start_s_index_t = torch.tensor([q_start_s_index], dtype=torch.int32, device="cuda")

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(min(max(1, ((t + q_start_s_index) // KV_stride)), SKV))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    kernel = sparse_mla_fwd_interface(q, kv, indices, q_start_s_index, KV_stride, return_kernel=True, print_kernel=True)

    def fn():
        out, lse = kernel(q, kv, indices, q_start_s_index_t)
        if q_start_s_index == 0 and KV_stride > 1:
            out[:, : KV_stride - 1, :, :] = 0
        return out, lse

    tl_out, tl_lse = fn()
    ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, q_start_s_index, KV_stride)
    # print(f"tl_out: {tl_out}")
    # print(f"ref_out: {ref_out}")

    torch.testing.assert_close(tl_out, ref_out, rtol=1e-3, atol=1e-3)

    from tilelang.profiler import do_bench

    ms = do_bench(
        fn,
        rep=10,
        warmup=10,
    )
    print(f"Average time: {ms:.3f} ms")
    print(f"fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    print(f"fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)


def run_regression_perf(B=1, S=4096, SKV=8192, H=128, HKV=1, DQK=576, DV=512, topk=2048, dtype=torch.bfloat16, q_start_s_index=1024):
    KV_stride = 1

    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True) / 10
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True) / 10
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(min(max(1, ((t + q_start_s_index) // KV_stride)), SKV))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape
    dim = 512
    tail_dim = dim_plus_tail_dim - dim
    CP0 = q_start_s_index == 0
    kernel = sparse_mla_fwd(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, KV_stride, kv_group, None, True, CP0)

    def run_kernel_only():
        kernel(q, kv, indices, torch.tensor([q_start_s_index], dtype=torch.int32, device="cuda"))

    from tilelang.profiler import do_bench

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_correctness", action="store_true")
    args = parser.parse_args()
    if args.test_correctness:
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 1, 1024, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
    else:
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 1, 4096, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
    test_sparse_mla_fwd_pipelined(B, S, SKV, H, HKV, DQK, DV, topk, dtype, check_correctness=args.test_correctness)
