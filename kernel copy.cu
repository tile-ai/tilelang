#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_no_split_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc);
extern "C" __global__ void __launch_bounds__(256, 1) main_no_split_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o_l[128];
  float logsum_0[2];
  float acc_s_0[32];
  float scores_max_0[2];
  float scores_max_prev_0[2];
  float scores_sum_0[2];
  float acc_o_r[128];
  float logsum_1[2];
  float acc_s_1[32];
  float scores_max_prev_1[4];
  float scores_max_1[2];
  float scores_sum_1[2];
  __shared__ uint64_t _mbarrier[17];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(Q_pe_desc);
    tl::prefetch_tma_descriptor(KV_desc);
    tl::prefetch_tma_descriptor(K_pe_desc);
    tl::prefetch_tma_descriptor(Output_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 128);
    tl::mbarrier_init(_mbarrier[5], 128);
    tl::mbarrier_init(_mbarrier[6], 128);
    tl::mbarrier_init(_mbarrier[7], 128);
    tl::mbarrier_init(_mbarrier[8], 128);
    tl::mbarrier_init(_mbarrier[9], 128);
    tl::mbarrier_init(_mbarrier[10], 128);
    tl::mbarrier_init(_mbarrier[11], 128);
    tl::mbarrier_init(_mbarrier[12], 256);
    tl::mbarrier_init(_mbarrier[13], 128);
    tl::mbarrier_init(_mbarrier[14], 128);
    tl::mbarrier_init(_mbarrier[15], 128);
    tl::mbarrier_init(_mbarrier[16], 128);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    tl::mbarrier_expect_tx(_mbarrier[12], 32768);
    tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[47104])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[51200])), 64, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[55296])), 128, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[59392])), 192, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    tl::mbarrier_expect_tx(_mbarrier[12], 32768);
    tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[79872])), 256, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[83968])), 320, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[88064])), 384, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[92160])), 448, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    tl::mbarrier_expect_tx(_mbarrier[12], 8192);
    tl::tma_load(Q_pe_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[6144])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
  }
  tl::mbarrier_arrive(_mbarrier[12]);
  tl::mbarrier_wait(_mbarrier[12], 0);
  if (((int)threadIdx.x) < 16) {
    *(float4*)(((float*)buf_dyn_shmem) + (((int)threadIdx.x) * 4)) = make_float4(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(float2*)(acc_o_l + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      logsum_0[i_1] = 0.000000e+00f;
    }
    tl::fence_proxy_async();
    if (((int)threadIdx.x) == 0) {
      tl::mbarrier_expect_tx(_mbarrier[3], 32768);
      tl::tma_load(KV_desc, _mbarrier[3], (&(((half_t*)buf_dyn_shmem)[14336])), 0, 0, 64, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[3], (&(((half_t*)buf_dyn_shmem)[18432])), 64, 0, 64, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[3], (&(((half_t*)buf_dyn_shmem)[22528])), 128, 0, 64, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[3], (&(((half_t*)buf_dyn_shmem)[26624])), 192, 0, 64, ((int)blockIdx.y));
    }
    tl::mbarrier_arrive(_mbarrier[3]);
    if (((int)threadIdx.x) == 0) {
      tl::mbarrier_expect_tx(_mbarrier[4], 32768);
      tl::tma_load(KV_desc, _mbarrier[4], (&(((half_t*)buf_dyn_shmem)[30720])), 256, 0, 64, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[4], (&(((half_t*)buf_dyn_shmem)[34816])), 320, 0, 64, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[4], (&(((half_t*)buf_dyn_shmem)[38912])), 384, 0, 64, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[4], (&(((half_t*)buf_dyn_shmem)[43008])), 448, 0, 64, ((int)blockIdx.y));
    }
    tl::mbarrier_arrive(_mbarrier[4]);
    if (((int)threadIdx.x) == 0) {
      tl::mbarrier_expect_tx(_mbarrier[5], 8192);
      tl::tma_load(K_pe_desc, _mbarrier[5], (&(((half_t*)buf_dyn_shmem)[2048])), 0, 0, 64, ((int)blockIdx.y));
    }
    tl::mbarrier_arrive(_mbarrier[5]);
    for (int k = 0; k < 64; ++k) {
      tl::mbarrier_wait(_mbarrier[0], (k & 1));
      tl::gemm_ss<64, 64, 256, 4, 1, 0, 1, 1, true, -1>((&(((half_t*)buf_dyn_shmem)[47104])), (&(((half_t*)buf_dyn_shmem)[63488])), (&(acc_s_0[0])));
      tl::mbarrier_wait(_mbarrier[1], (k & 1));
      tl::gemm_ss<64, 64, 256, 4, 1, 0, 1, 0, true, -1>((&(((half_t*)buf_dyn_shmem)[79872])), (&(((half_t*)buf_dyn_shmem)[96256])), (&(acc_s_0[0])));
      tl::mbarrier_wait(_mbarrier[2], (k & 1));
      tl::gemm_ss<64, 64, 64, 4, 1, 0, 1, 0, true, -1>((&(((half_t*)buf_dyn_shmem)[6144])), (&(((half_t*)buf_dyn_shmem)[10240])), (&(acc_s_0[0])));
      tl::wait_wgmma<0>();
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        scores_max_0[i_2] = ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 5) * 16) + (i_2 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        scores_max_prev_0[i_3] = scores_max_0[i_3];
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        scores_max_0[i_4] = -CUDART_INF_F;
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        #pragma unroll
        for (int rv = 0; rv < 16; ++rv) {
          scores_max_0[i_5] = max(scores_max_0[i_5], acc_s_0[((((rv & 7) * 4) + (i_5 * 2)) + (rv >> 3))]);
        }
        scores_max_0[i_5] = tl::AllReduce<tl::MaxOp, 4, 1, 128>::run_hopper(scores_max_0[i_5]);
      }
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 5) * 16) + (i_6 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = scores_max_0[i_6];
      }
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_7 = 0; i_7 < 16; ++i_7) {
        float2 __1;
        float2 __2;
          float2 __3;
            float2 v_ = *(float2*)(acc_s_0 + (i_7 * 2));
            float2 v__1 = make_float2(6.011229e-02f, 6.011229e-02f);
            __3.x = (v_.x*v__1.x);
            __3.y = (v_.y*v__1.y);
          float2 v__2 = make_float2((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 5) * 16) + ((i_7 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * 6.011229e-02f), (((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 5) * 16) + ((i_7 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * 6.011229e-02f));
          __2.x = (__3.x-v__2.x);
          __2.y = (__3.y-v__2.y);
        __1.x = exp2f(__2.x);
        __1.y = exp2f(__2.y);
        *(float2*)(acc_s_0 + (i_7 * 2)) = __1;
      }
      if ((((int)threadIdx.x) % 4) == 0) {
        #pragma unroll
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + (i_8 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 256)] = exp2f(((scores_max_prev_0[i_8] * 6.011229e-02f) - (((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 5) * 16) + (i_8 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * 6.011229e-02f)));
        }
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        scores_sum_0[i_9] = 0.000000e+00f;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
          scores_sum_0[i_9] = (scores_sum_0[i_9] + acc_s_0[((((rv_1 & 7) * 4) + (i_9 * 2)) + (rv_1 >> 3))]);
        }
        scores_sum_0[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 128>::run_hopper(scores_sum_0[i_9]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_10 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 10240)])), __pack_half2(((half_t)acc_s_0[(i_10 * 8)]), ((half_t)acc_s_0[((i_10 * 8) + 1)])), __pack_half2(((half_t)acc_s_0[((i_10 * 8) + 2)]), ((half_t)acc_s_0[((i_10 * 8) + 3)])), __pack_half2(((half_t)acc_s_0[((i_10 * 8) + 4)]), ((half_t)acc_s_0[((i_10 * 8) + 5)])), __pack_half2(((half_t)acc_s_0[((i_10 * 8) + 6)]), ((half_t)acc_s_0[((i_10 * 8) + 7)])));
      }
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_11 = 0; i_11 < 64; ++i_11) {
        float2 __4;
          float2 v__3 = *(float2*)(acc_o_l + (i_11 * 2));
          float2 v__4 = make_float2(((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_11 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 256)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_11 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 256)]);
          __4.x = (v__3.x*v__4.x);
          __4.y = (v__3.y*v__4.y);
        *(float2*)(acc_o_l + (i_11 * 2)) = __4;
      }
      if ((((int)threadIdx.x) % 4) == 0) {
        #pragma unroll
        for (int i_12 = 0; i_12 < 2; ++i_12) {
          logsum_0[i_12] = ((logsum_0[i_12] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + (i_12 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 256)]) + scores_sum_0[i_12]);
        }
      }
      tl::fence_proxy_async();
      tl::gemm_ss<64, 256, 64, 4, 1, 0, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[10240])), (&(((half_t*)buf_dyn_shmem)[63488])), (&(acc_o_l[0])));
      tl::mbarrier_arrive(_mbarrier[6]);
      tl::mbarrier_wait(_mbarrier[7], (k & 1));
      tl::__sync_thread_partial<3, 128>();
      if (k < 63) {
        if (((int)threadIdx.x) == 0) {
          tl::mbarrier_expect_tx(_mbarrier[0], 32768);
          tl::tma_load(KV_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[63488])), 0, 0, ((k * 128) + 128), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[67584])), 64, 0, ((k * 128) + 128), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[71680])), 128, 0, ((k * 128) + 128), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[75776])), 192, 0, ((k * 128) + 128), ((int)blockIdx.y));
        }
        tl::mbarrier_arrive(_mbarrier[0]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 16; ++i_13) {
        uint1 __5;
        float2 __6;
          float2 v__5 = *(float2*)(acc_s_0 + (i_13 * 2));
          float2 v__6 = make_float2(((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 512)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 512)]);
          __6.x = (v__5.x*v__6.x);
          __6.y = (v__5.y*v__6.y);
        ((half2*)(&(__5.x)))->x = (half_t)(__6.x);
        ((half2*)(&(__5.x)))->y = (half_t)(__6.y);
        *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_13 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = __5;
      }
      tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[8]);
      #pragma unroll
      for (int i_14 = 0; i_14 < 64; ++i_14) {
        float2 __7;
          float2 v__7 = *(float2*)(acc_o_l + (i_14 * 2));
          float2 v__8 = make_float2(((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 512)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 512)]);
          __7.x = (v__7.x*v__8.x);
          __7.y = (v__7.y*v__8.y);
        *(float2*)(acc_o_l + (i_14 * 2)) = __7;
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        logsum_0[i_15] = (logsum_0[i_15] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + (i_15 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 512)]);
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[15], (k & 1));
      tl::gemm_ss<64, 256, 64, 4, 1, 0, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[10240])), (&(((half_t*)buf_dyn_shmem)[14336])), (&(acc_o_l[0])));
      tl::mbarrier_arrive(_mbarrier[14]);
      tl::mbarrier_arrive(_mbarrier[16]);
      tl::__sync_thread_partial<3, 128>();
      if (k < 63) {
        tl::mbarrier_wait(_mbarrier[16], (k & 1));
        if (((int)threadIdx.x) == 0) {
          tl::mbarrier_expect_tx(_mbarrier[3], 32768);
          tl::tma_load(KV_desc, _mbarrier[3], (&(((half_t*)buf_dyn_shmem)[14336])), 0, 0, ((k * 128) + 192), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[3], (&(((half_t*)buf_dyn_shmem)[18432])), 64, 0, ((k * 128) + 192), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[3], (&(((half_t*)buf_dyn_shmem)[22528])), 128, 0, ((k * 128) + 192), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[3], (&(((half_t*)buf_dyn_shmem)[26624])), 192, 0, ((k * 128) + 192), ((int)blockIdx.y));
        }
        tl::mbarrier_arrive(_mbarrier[3]);
        tl::mbarrier_wait(_mbarrier[13], (k & 1));
        if (((int)threadIdx.x) == 0) {
          tl::mbarrier_expect_tx(_mbarrier[5], 8192);
          tl::tma_load(K_pe_desc, _mbarrier[5], (&(((half_t*)buf_dyn_shmem)[2048])), 0, 0, ((k * 128) + 192), ((int)blockIdx.y));
        }
        tl::mbarrier_arrive(_mbarrier[5]);
      }
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + (i_16 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 768)] = logsum_0[i_16];
    }
    tl::fence_proxy_async();
    tl::mbarrier_arrive(_mbarrier[9]);
    tl::mbarrier_wait(_mbarrier[10], 0);
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_17 = 0; i_17 < 64; ++i_17) {
      float2 __8;
        float2 v__9 = *(float2*)(acc_o_l + (i_17 * 2));
        float2 v__10 = make_float2(((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_17 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 768)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_17 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 768)]);
        __8.x = (v__9.x/v__10.x);
        __8.y = (v__9.y/v__10.y);
      *(float2*)(acc_o_l + (i_17 * 2)) = __8;
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 16; ++i_18) {
      tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((i_18 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_18 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 47104)])), __pack_half2(((half_t)acc_o_l[(i_18 * 8)]), ((half_t)acc_o_l[((i_18 * 8) + 1)])), __pack_half2(((half_t)acc_o_l[((i_18 * 8) + 2)]), ((half_t)acc_o_l[((i_18 * 8) + 3)])), __pack_half2(((half_t)acc_o_l[((i_18 * 8) + 4)]), ((half_t)acc_o_l[((i_18 * 8) + 5)])), __pack_half2(((half_t)acc_o_l[((i_18 * 8) + 6)]), ((half_t)acc_o_l[((i_18 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    tl::__sync_thread_partial<3, 128>();
    if (((int)threadIdx.x) == 0) {
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[47104])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[51200])), 64, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[55296])), 128, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[59392])), 192, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  } else {
    #pragma unroll
    for (int i_19 = 0; i_19 < 64; ++i_19) {
      *(float2*)(acc_o_r + (i_19 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      logsum_1[i_20] = 0.000000e+00f;
    }
    tl::fence_proxy_async();
    if (((int)threadIdx.x) == 128) {
      tl::mbarrier_expect_tx(_mbarrier[0], 32768);
      tl::tma_load(KV_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[63488])), 0, 0, 0, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[67584])), 64, 0, 0, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[71680])), 128, 0, 0, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[75776])), 192, 0, 0, ((int)blockIdx.y));
    }
    tl::mbarrier_arrive(_mbarrier[0]);
    if (((int)threadIdx.x) == 128) {
      tl::mbarrier_expect_tx(_mbarrier[1], 32768);
      tl::tma_load(KV_desc, _mbarrier[1], (&(((half_t*)buf_dyn_shmem)[96256])), 256, 0, 0, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[1], (&(((half_t*)buf_dyn_shmem)[100352])), 320, 0, 0, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[1], (&(((half_t*)buf_dyn_shmem)[104448])), 384, 0, 0, ((int)blockIdx.y));
      tl::tma_load(KV_desc, _mbarrier[1], (&(((half_t*)buf_dyn_shmem)[108544])), 448, 0, 0, ((int)blockIdx.y));
    }
    tl::mbarrier_arrive(_mbarrier[1]);
    if (((int)threadIdx.x) == 128) {
      tl::mbarrier_expect_tx(_mbarrier[2], 8192);
      tl::tma_load(K_pe_desc, _mbarrier[2], (&(((half_t*)buf_dyn_shmem)[10240])), 0, 0, 0, ((int)blockIdx.y));
    }
    tl::mbarrier_arrive(_mbarrier[2]);
    for (int k_1 = 0; k_1 < 64; ++k_1) {
      tl::mbarrier_wait(_mbarrier[3], (k_1 & 1));
      tl::gemm_ss<64, 64, 256, 4, 1, 0, 1, 1, true, -1>((&(((half_t*)buf_dyn_shmem)[47104])), (&(((half_t*)buf_dyn_shmem)[14336])), (&(acc_s_1[0])));
      tl::mbarrier_wait(_mbarrier[4], (k_1 & 1));
      tl::gemm_ss<64, 64, 256, 4, 1, 0, 1, 0, true, -1>((&(((half_t*)buf_dyn_shmem)[79872])), (&(((half_t*)buf_dyn_shmem)[30720])), (&(acc_s_1[0])));
      tl::mbarrier_wait(_mbarrier[5], (k_1 & 1));
      tl::gemm_ss<64, 64, 64, 4, 1, 0, 1, 0, true, -1>((&(((half_t*)buf_dyn_shmem)[6144])), (&(((half_t*)buf_dyn_shmem)[2048])), (&(acc_s_1[0])));
      tl::wait_wgmma<0>();
      tl::mbarrier_wait(_mbarrier[6], (k_1 & 1));
      *(float4*)(scores_max_prev_1 + 0) = *(float4*)(((float*)buf_dyn_shmem) + ((((int)threadIdx.x) & 15) * 4));
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        scores_max_1[i_21] = -CUDART_INF_F;
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
          scores_max_1[i_22] = max(scores_max_1[i_22], acc_s_1[((((rv_2 & 7) * 4) + (i_22 * 2)) + (rv_2 >> 3))]);
        }
        scores_max_1[i_22] = tl::AllReduce<tl::MaxOp, 4, 1, 128>::run_hopper(scores_max_1[i_22]);
      }
      tl::__sync_thread_partial<4, 128>();
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + (i_23 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)] = scores_max_1[i_23];
      }
      tl::__sync_thread_partial<4, 128>();
      if ((((int)threadIdx.x) >> 4) == 8) {
        float4 __9;
        float4 __10;
          float4 __11;
            float4 v__11 = *(float4*)(scores_max_prev_1 + 0);
            float4 v__12 = make_float4(6.011229e-02f, 6.011229e-02f, 6.011229e-02f, 6.011229e-02f);
            __11.x = (v__11.x*v__12.x);
            __11.y = (v__11.y*v__12.y);
            __11.z = (v__11.z*v__12.z);
            __11.w = (v__11.w*v__12.w);
          float4 __12;
            float4 v__13 = *(float4*)(((float*)buf_dyn_shmem) + ((((int)threadIdx.x) & 15) * 4));
            float4 v__14 = make_float4(6.011229e-02f, 6.011229e-02f, 6.011229e-02f, 6.011229e-02f);
            __12.x = (v__13.x*v__14.x);
            __12.y = (v__13.y*v__14.y);
            __12.z = (v__13.z*v__14.z);
            __12.w = (v__13.w*v__14.w);
          __10.x = (__11.x-__12.x);
          __10.y = (__11.y-__12.y);
          __10.z = (__11.z-__12.z);
          __10.w = (__11.w-__12.w);
        __9.x = exp2f(__10.x);
        __9.y = exp2f(__10.y);
        __9.z = exp2f(__10.z);
        __9.w = exp2f(__10.w);
        *(float4*)(((float*)buf_dyn_shmem) + (((((int)threadIdx.x) & 15) * 4) + 512)) = __9;
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 16; ++i_24) {
        float2 __13;
        float2 __14;
          float2 __15;
            float2 v__15 = *(float2*)(acc_s_1 + (i_24 * 2));
            float2 v__16 = make_float2(6.011229e-02f, 6.011229e-02f);
            __15.x = (v__15.x*v__16.x);
            __15.y = (v__15.y*v__16.y);
          float2 v__17 = make_float2((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_24 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)] * 6.011229e-02f), (((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_24 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)] * 6.011229e-02f));
          __14.x = (__15.x-v__17.x);
          __14.y = (__15.y-v__17.y);
        __13.x = exp2f(__14.x);
        __13.y = exp2f(__14.y);
        *(float2*)(acc_s_1 + (i_24 * 2)) = __13;
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        scores_sum_1[i_25] = 0.000000e+00f;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
          scores_sum_1[i_25] = (scores_sum_1[i_25] + acc_s_1[((((rv_3 & 7) * 4) + (i_25 * 2)) + (rv_3 >> 3))]);
        }
        scores_sum_1[i_25] = tl::AllReduce<tl::SumOp, 4, 1, 128>::run_hopper(scores_sum_1[i_25]);
      }
      tl::__sync_thread_partial<4, 128>();
      #pragma unroll
      for (int i_26 = 0; i_26 < 64; ++i_26) {
        float2 __16;
          float2 v__18 = *(float2*)(acc_o_r + (i_26 * 2));
          float2 v__19 = make_float2((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_26 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 192)] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_26 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 448)]), (((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_26 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 192)] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_26 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 448)]));
          __16.x = (v__18.x*v__19.x);
          __16.y = (v__18.y*v__19.y);
        *(float2*)(acc_o_r + (i_26 * 2)) = __16;
      }
      if ((((int)threadIdx.x) % 4) == 0) {
        #pragma unroll
        for (int i_27 = 0; i_27 < 2; ++i_27) {
          logsum_1[i_27] = (((logsum_1[i_27] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + (i_27 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 448)]) * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + (i_27 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 192)]) + scores_sum_1[i_27]);
        }
      }
      tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[7]);
      #pragma unroll
      for (int i_28 = 0; i_28 < 4; ++i_28) {
        tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_28 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6144)])), __pack_half2(((half_t)acc_s_1[(i_28 * 8)]), ((half_t)acc_s_1[((i_28 * 8) + 1)])), __pack_half2(((half_t)acc_s_1[((i_28 * 8) + 2)]), ((half_t)acc_s_1[((i_28 * 8) + 3)])), __pack_half2(((half_t)acc_s_1[((i_28 * 8) + 4)]), ((half_t)acc_s_1[((i_28 * 8) + 5)])), __pack_half2(((half_t)acc_s_1[((i_28 * 8) + 6)]), ((half_t)acc_s_1[((i_28 * 8) + 7)])));
      }
      tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[15]);
      tl::__sync_thread_partial<4, 128>();
      tl::gemm_ss<64, 256, 64, 4, 1, 0, 0, 0, true, -1>((&(((half_t*)buf_dyn_shmem)[10240])), (&(((half_t*)buf_dyn_shmem)[30720])), (&(acc_o_r[0])));
      tl::__sync_thread_partial<4, 128>();
      if (k_1 < 63) {
        if (((int)threadIdx.x) == 128) {
          tl::mbarrier_expect_tx(_mbarrier[4], 32768);
          tl::tma_load(KV_desc, _mbarrier[4], (&(((half_t*)buf_dyn_shmem)[30720])), 256, 0, ((k_1 * 128) + 192), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[4], (&(((half_t*)buf_dyn_shmem)[34816])), 320, 0, ((k_1 * 128) + 192), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[4], (&(((half_t*)buf_dyn_shmem)[38912])), 384, 0, ((k_1 * 128) + 192), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[4], (&(((half_t*)buf_dyn_shmem)[43008])), 448, 0, ((k_1 * 128) + 192), ((int)blockIdx.y));
        }
        tl::mbarrier_arrive(_mbarrier[4]);
      }
      tl::mbarrier_wait(_mbarrier[8], (k_1 & 1));
      tl::gemm_ss<64, 256, 64, 4, 1, 0, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[2048])), (&(((half_t*)buf_dyn_shmem)[96256])), (&(acc_o_r[0])));
      tl::mbarrier_arrive(_mbarrier[13]);
      tl::__sync_thread_partial<4, 128>();
      if (k_1 < 63) {
        if (((int)threadIdx.x) == 128) {
          tl::mbarrier_expect_tx(_mbarrier[1], 32768);
          tl::tma_load(KV_desc, _mbarrier[1], (&(((half_t*)buf_dyn_shmem)[96256])), 256, 0, ((k_1 * 128) + 128), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[1], (&(((half_t*)buf_dyn_shmem)[100352])), 320, 0, ((k_1 * 128) + 128), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[1], (&(((half_t*)buf_dyn_shmem)[104448])), 384, 0, ((k_1 * 128) + 128), ((int)blockIdx.y));
          tl::tma_load(KV_desc, _mbarrier[1], (&(((half_t*)buf_dyn_shmem)[108544])), 448, 0, ((k_1 * 128) + 128), ((int)blockIdx.y));
        }
        tl::mbarrier_arrive(_mbarrier[1]);
        tl::mbarrier_wait(_mbarrier[14], (k_1 & 1));
        if (((int)threadIdx.x) == 128) {
          tl::mbarrier_expect_tx(_mbarrier[2], 8192);
          tl::tma_load(K_pe_desc, _mbarrier[2], (&(((half_t*)buf_dyn_shmem)[10240])), 0, 0, ((k_1 * 128) + 128), ((int)blockIdx.y));
        }
        tl::mbarrier_arrive(_mbarrier[2]);
      }
    }
    tl::mbarrier_wait(_mbarrier[9], 0);
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_29 = 0; i_29 < 2; ++i_29) {
        ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + (i_29 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 704)] = (((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + (i_29 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 704)] + logsum_1[i_29]);
      }
    }
    tl::fence_proxy_async();
    tl::mbarrier_arrive(_mbarrier[10]);
    tl::__sync_thread_partial<4, 128>();
    #pragma unroll
    for (int i_30 = 0; i_30 < 64; ++i_30) {
      float2 __17;
        float2 v__20 = *(float2*)(acc_o_r + (i_30 * 2));
        float2 v__21 = make_float2(((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_30 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 704)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_30 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 704)]);
        __17.x = (v__20.x/v__21.x);
        __17.y = (v__20.y/v__21.y);
      *(float2*)(acc_o_r + (i_30 * 2)) = __17;
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 16; ++i_31) {
      tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((i_31 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_31 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_31 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 75776)])), __pack_half2(((half_t)acc_o_r[(i_31 * 8)]), ((half_t)acc_o_r[((i_31 * 8) + 1)])), __pack_half2(((half_t)acc_o_r[((i_31 * 8) + 2)]), ((half_t)acc_o_r[((i_31 * 8) + 3)])), __pack_half2(((half_t)acc_o_r[((i_31 * 8) + 4)]), ((half_t)acc_o_r[((i_31 * 8) + 5)])), __pack_half2(((half_t)acc_o_r[((i_31 * 8) + 6)]), ((half_t)acc_o_r[((i_31 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    tl::__sync_thread_partial<4, 128>();
    if (((int)threadIdx.x) == 128) {
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[79872])), 256, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[83968])), 320, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[88064])), 384, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[92160])), 448, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_no_split_kernel = cudaFuncSetAttribute(main_no_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 225280);
    if (result_main_no_split_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 225280, cudaGetErrorString(result_main_no_split_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ Q, half_t* __restrict__ Q_pe, half_t* __restrict__ KV, half_t* __restrict__ K_pe, half_t* __restrict__ glse, half_t* __restrict__ Output_partial, half_t* __restrict__ Output, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap KV_desc;
	CUtensorMapDataType KV_desc_type= (CUtensorMapDataType)6;
	cuuint32_t KV_desc_tensorRank= 4;
	void *KV_desc_globalAddress= KV;
	cuuint64_t KV_desc_globalDim[4]= {512,1,8192,132};
	cuuint64_t KV_desc_globalStride[4]= {2,1024,1024,8388608};
	cuuint32_t KV_desc_boxDim[4]= {64,1,64,1};
	cuuint32_t KV_desc_elementStrides[4]= {1,1,1,1};
	CUtensorMapInterleave KV_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle KV_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion KV_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill KV_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult KV_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &KV_desc, KV_desc_type, KV_desc_tensorRank, KV_desc_globalAddress, KV_desc_globalDim, KV_desc_globalStride + 1, KV_desc_boxDim, KV_desc_elementStrides, KV_desc_interleave, KV_desc_swizzle, KV_desc_l2Promotion, KV_desc_oobFill);

	if (KV_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor KV_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap K_pe_desc;
	CUtensorMapDataType K_pe_desc_type= (CUtensorMapDataType)6;
	cuuint32_t K_pe_desc_tensorRank= 4;
	void *K_pe_desc_globalAddress= K_pe;
	cuuint64_t K_pe_desc_globalDim[4]= {64,1,8192,132};
	cuuint64_t K_pe_desc_globalStride[4]= {2,128,128,1048576};
	cuuint32_t K_pe_desc_boxDim[4]= {64,1,64,1};
	cuuint32_t K_pe_desc_elementStrides[4]= {1,1,1,1};
	CUtensorMapInterleave K_pe_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle K_pe_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion K_pe_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill K_pe_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult K_pe_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &K_pe_desc, K_pe_desc_type, K_pe_desc_tensorRank, K_pe_desc_globalAddress, K_pe_desc_globalDim, K_pe_desc_globalStride + 1, K_pe_desc_boxDim, K_pe_desc_elementStrides, K_pe_desc_interleave, K_pe_desc_swizzle, K_pe_desc_l2Promotion, K_pe_desc_oobFill);

	if (K_pe_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor K_pe_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap Output_desc;
	CUtensorMapDataType Output_desc_type= (CUtensorMapDataType)6;
	cuuint32_t Output_desc_tensorRank= 3;
	void *Output_desc_globalAddress= Output;
	cuuint64_t Output_desc_globalDim[3]= {512,128,132};
	cuuint64_t Output_desc_globalStride[3]= {2,1024,131072};
	cuuint32_t Output_desc_boxDim[3]= {64,64,1};
	cuuint32_t Output_desc_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave Output_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Output_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion Output_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Output_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Output_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Output_desc, Output_desc_type, Output_desc_tensorRank, Output_desc_globalAddress, Output_desc_globalDim, Output_desc_globalStride + 1, Output_desc_boxDim, Output_desc_elementStrides, Output_desc_interleave, Output_desc_swizzle, Output_desc_l2Promotion, Output_desc_oobFill);

	if (Output_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Output_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap Q_desc;
	CUtensorMapDataType Q_desc_type= (CUtensorMapDataType)6;
	cuuint32_t Q_desc_tensorRank= 3;
	void *Q_desc_globalAddress= Q;
	cuuint64_t Q_desc_globalDim[3]= {512,128,132};
	cuuint64_t Q_desc_globalStride[3]= {2,1024,131072};
	cuuint32_t Q_desc_boxDim[3]= {64,64,1};
	cuuint32_t Q_desc_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave Q_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Q_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion Q_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Q_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Q_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Q_desc, Q_desc_type, Q_desc_tensorRank, Q_desc_globalAddress, Q_desc_globalDim, Q_desc_globalStride + 1, Q_desc_boxDim, Q_desc_elementStrides, Q_desc_interleave, Q_desc_swizzle, Q_desc_l2Promotion, Q_desc_oobFill);

	if (Q_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Q_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap Q_pe_desc;
	CUtensorMapDataType Q_pe_desc_type= (CUtensorMapDataType)6;
	cuuint32_t Q_pe_desc_tensorRank= 3;
	void *Q_pe_desc_globalAddress= Q_pe;
	cuuint64_t Q_pe_desc_globalDim[3]= {64,128,132};
	cuuint64_t Q_pe_desc_globalStride[3]= {2,128,16384};
	cuuint32_t Q_pe_desc_boxDim[3]= {64,64,1};
	cuuint32_t Q_pe_desc_elementStrides[3]= {1,1,1};
	CUtensorMapInterleave Q_pe_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle Q_pe_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion Q_pe_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill Q_pe_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult Q_pe_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &Q_pe_desc, Q_pe_desc_type, Q_pe_desc_tensorRank, Q_pe_desc_globalAddress, Q_pe_desc_globalDim, Q_pe_desc_globalStride + 1, Q_pe_desc_boxDim, Q_pe_desc_elementStrides, Q_pe_desc_interleave, Q_pe_desc_swizzle, Q_pe_desc_l2Promotion, Q_pe_desc_oobFill);

	if (Q_pe_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor Q_pe_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	main_no_split_kernel<<<dim3(2, 132, 1), dim3(256, 1, 1), 225280, stream>>>(KV_desc, K_pe_desc, Output_desc, Q_desc, Q_pe_desc);
	TILELANG_CHECK_LAST_ERROR("main_no_split_kernel");

	return 0;
}

