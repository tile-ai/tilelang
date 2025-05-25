#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_kernel(float* __restrict__ Delta, int* __restrict__ Indices, bfloat16_t* __restrict__ KV, float* __restrict__ Lse, bfloat16_t* __restrict__ Q, float* __restrict__ dKV, bfloat16_t* __restrict__ dO, bfloat16_t* __restrict__ dQ);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(float* __restrict__ Delta, int* __restrict__ Indices, bfloat16_t* __restrict__ KV, float* __restrict__ Lse, bfloat16_t* __restrict__ Q, float* __restrict__ dKV, bfloat16_t* __restrict__ dO, bfloat16_t* __restrict__ dQ) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_dq[128];
  float acc_dq_tail[16];
  __shared__ signed char mask[32];
  float acc_p[8];
  float acc_dp[8];
  float acc_dkv[64];
  float acc_dkv_tail[8];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 3) * 4096) + (i * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 26624)) = *(uint4*)(Q + ((((((int)blockIdx.x) * 36864) + (i * 2304)) + ((((int)threadIdx.x) >> 6) * 576)) + ((((int)threadIdx.x) & 63) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6144)) = *(uint4*)(Q + (((((((int)blockIdx.x) * 36864) + (i_1 * 18432)) + ((((int)threadIdx.x) >> 3) * 576)) + ((((int)threadIdx.x) & 7) * 8)) + 512));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 3) * 4096) + (i_2 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 59392)) = *(uint4*)(dO + (((((int)blockIdx.x) * 32768) + (i_2 * 2048)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 64; ++i_3) {
    *(float2*)(acc_dq + (i_3 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    *(float2*)(acc_dq_tail + (i_4 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
  }
  for (int i_i = 0; i_i < 64; ++i_i) {
    __syncthreads();
    if (((int)threadIdx.x) < 8) {
      int __1;
      ushort4 __2;
        int4 v_ = *(int4*)(Indices + (((((int)blockIdx.x) * 2048) + (i_i * 32)) + (((int)threadIdx.x) * 4)));
        int4 v__1 = make_int4((((int)blockIdx.x) + 28672), (((int)blockIdx.x) + 28672), (((int)blockIdx.x) + 28672), (((int)blockIdx.x) + 28672));
        __2.x = (v_.x<=v__1.x);
        __2.y = (v_.y<=v__1.y);
        __2.z = (v_.z<=v__1.z);
        __2.w = (v_.w<=v__1.w);
      __1=((signed char)(__2.x) << 0);
      __1=__1 & ~(0x000000ff << 8) |((signed char)(__2.y) << 8);
      __1=__1 & ~(0x000000ff << 16) |((signed char)(__2.z) << 16);
      __1=__1 & ~(0x000000ff << 24) |((signed char)(__2.w) << 24);
      *(int*)(mask + (((int)threadIdx.x) * 4)) = __1;
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      for (int vec_s = 0; vec_s < 2; ++vec_s) {
        float condval;
        if (((bool)mask[(((((((int)threadIdx.x) >> 7) * 16) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s)])) {
          condval = 0.000000e+00f;
        } else {
          condval = -CUDART_INF_F;
        }
        acc_p[((i_5 * 2) + vec_s)] = condval;
      }
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      uint4 condval_1;
      if (((0 <= Indices[((((((int)blockIdx.x) * 2048) + (i_i * 32)) + (i_6 * 4)) + (((int)threadIdx.x) >> 6))]) && (Indices[((((((int)blockIdx.x) * 2048) + (i_i * 32)) + (i_6 * 4)) + (((int)threadIdx.x) >> 6))] < 32768))) {
        condval_1 = *(uint4*)(KV + ((((int64_t)Indices[((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + (((int64_t)i_i) * (int64_t)32)) + (((int64_t)i_6) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6))]) * (int64_t)576) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
      } else {
        condval_1 = make_uint4(__pack_nv_bfloat162(bfloat16_t(0.000000e+00f), bfloat16_t(0.000000e+00f)), __pack_nv_bfloat162(bfloat16_t(0.000000e+00f), bfloat16_t(0.000000e+00f)), __pack_nv_bfloat162(bfloat16_t(0.000000e+00f), bfloat16_t(0.000000e+00f)), __pack_nv_bfloat162(bfloat16_t(0.000000e+00f), bfloat16_t(0.000000e+00f)));
      }
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i_6 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_6 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 10240)) = condval_1;
    }
    tl::fence_proxy_async();
    __syncthreads();
    tl::gemm_ss<64, 32, 512, 4, 2, 0, 1, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[26624])), (&(((bfloat16_t*)buf_dyn_shmem)[10240])), (&(acc_p[0])));
    uint4 condval_2;
    if (((0 <= Indices[(((((int)blockIdx.x) * 2048) + (i_i * 32)) + (((int)threadIdx.x) >> 3))]) && (Indices[(((((int)blockIdx.x) * 2048) + (i_i * 32)) + (((int)threadIdx.x) >> 3))] < 32768))) {
      condval_2 = *(uint4*)(KV + (((((int64_t)Indices[(((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + (((int64_t)i_i) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3))]) * (int64_t)576) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) + (int64_t)512));
    } else {
      condval_2 = make_uint4(__pack_nv_bfloat162(bfloat16_t(0.000000e+00f), bfloat16_t(0.000000e+00f)), __pack_nv_bfloat162(bfloat16_t(0.000000e+00f), bfloat16_t(0.000000e+00f)), __pack_nv_bfloat162(bfloat16_t(0.000000e+00f), bfloat16_t(0.000000e+00f)), __pack_nv_bfloat162(bfloat16_t(0.000000e+00f), bfloat16_t(0.000000e+00f)));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 3) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)) = condval_2;
    tl::fence_proxy_async();
    __syncthreads();
    tl::gemm_ss<64, 32, 64, 4, 2, 0, 1, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[6144])), (&(((bfloat16_t*)buf_dyn_shmem)[4096])), (&(acc_p[0])));
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      float2 __3;
      float2 __4;
        float2 __5;
          float2 v__2 = *(float2*)(acc_p + (i_7 * 2));
          float2 v__3 = make_float2(6.011229e-02f, 6.011229e-02f);
          __5.x = (v__2.x*v__3.x);
          __5.y = (v__2.y*v__3.y);
        float2 v__4 = make_float2(Lse[((((((int)blockIdx.x) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_7 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], Lse[((((((int)blockIdx.x) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_7 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        __4.x = (__5.x-v__4.x);
        __4.y = (__5.y-v__4.y);
      __3.x = exp2f(__4.x);
      __3.y = exp2f(__4.y);
      *(float2*)(acc_p + (i_7 * 2)) = __3;
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 1; ++i_8) {
      tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 127) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)acc_p[0]), ((bfloat16_t)acc_p[1])), __pack_half2(((bfloat16_t)acc_p[2]), ((bfloat16_t)acc_p[3])), __pack_half2(((bfloat16_t)acc_p[4]), ((bfloat16_t)acc_p[5])), __pack_half2(((bfloat16_t)acc_p[6]), ((bfloat16_t)acc_p[7])));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      *(float2*)(acc_dp + (i_9 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    tl::fence_proxy_async();
    tl::gemm_ss<64, 32, 512, 4, 2, 0, 1, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[59392])), (&(((bfloat16_t*)buf_dyn_shmem)[10240])), (&(acc_dp[0])));
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      float2 __6;
        float2 __7;
          float2 v__5 = *(float2*)(acc_p + (i_10 * 2));
          float2 __8;
            float2 v__6 = *(float2*)(acc_dp + (i_10 * 2));
            float2 v__7 = make_float2(Delta[((((((int)blockIdx.x) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], Delta[((((((int)blockIdx.x) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            __8.x = (v__6.x-v__7.x);
            __8.y = (v__6.y-v__7.y);
          __7.x = (v__5.x*__8.x);
          __7.y = (v__5.y*__8.y);
        float2 v__8 = make_float2(4.166667e-02f, 4.166667e-02f);
        __6.x = (__7.x*v__8.x);
        __6.y = (__7.y*v__8.y);
      *(float2*)(acc_dp + (i_10 * 2)) = __6;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 1; ++i_11) {
      tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 127) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 2048)])), __pack_half2(((bfloat16_t)acc_dp[0]), ((bfloat16_t)acc_dp[1])), __pack_half2(((bfloat16_t)acc_dp[2]), ((bfloat16_t)acc_dp[3])), __pack_half2(((bfloat16_t)acc_dp[4]), ((bfloat16_t)acc_dp[5])), __pack_half2(((bfloat16_t)acc_dp[6]), ((bfloat16_t)acc_dp[7])));
    }
    tl::fence_proxy_async();
    __syncthreads();
    tl::gemm_ss<64, 512, 32, 4, 2, 0, 0, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[2048])), (&(((bfloat16_t*)buf_dyn_shmem)[10240])), (&(acc_dq[0])));
    tl::gemm_ss<64, 64, 32, 4, 2, 0, 0, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[2048])), (&(((bfloat16_t*)buf_dyn_shmem)[4096])), (&(acc_dq_tail[0])));
    #pragma unroll
    for (int i_12 = 0; i_12 < 32; ++i_12) {
      *(float2*)(acc_dkv + (i_12 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    tl::fence_proxy_async();
    tl::gemm_ss<32, 512, 64, 1, 8, 1, 0, 0, false>((&(((bfloat16_t*)buf_dyn_shmem)[2048])), (&(((bfloat16_t*)buf_dyn_shmem)[26624])), (&(acc_dkv[0])));
    tl::gemm_ss<32, 512, 64, 1, 8, 1, 0, 0, false>((&(((bfloat16_t*)buf_dyn_shmem)[0])), (&(((bfloat16_t*)buf_dyn_shmem)[59392])), (&(acc_dkv[0])));
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      *(float2*)(acc_dkv_tail + (i_13 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    tl::fence_proxy_async();
    tl::gemm_ss<32, 64, 64, 1, 8, 1, 0, 0, false>((&(((bfloat16_t*)buf_dyn_shmem)[2048])), (&(((bfloat16_t*)buf_dyn_shmem)[6144])), (&(acc_dkv_tail[0])));
    for (int s = 0; s < 2; ++s) {
      __syncthreads();
      #pragma unroll
      for (int i_14 = 0; i_14 < 32; ++i_14) {
        if (i_14 < 16) {
          *(float2*)(((float*)buf_dyn_shmem) + (((((((i_14 >> 3) * 4096) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + ((i_14 & 7) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 5120)) = *(float2*)(acc_dkv + ((((i_14 & 7) * 8) + (s * 4)) + ((i_14 >> 3) * 2)));
        }
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        if (i_15 < 2) {
          *(float2*)(((float*)buf_dyn_shmem) + (((((i_15 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(float2*)(acc_dkv_tail + ((s * 4) + (i_15 * 2)));
        }
      }
      tl::fence_proxy_async();
      __syncthreads();
      #pragma unroll
      for (int i_16 = 0; i_16 < 32; ++i_16) {
        if (0 <= Indices[((((((int)blockIdx.x) * 2048) + (i_i * 32)) + (s * 16)) + (i_16 >> 1))]) {
          if (Indices[((((((int)blockIdx.x) * 2048) + (i_i * 32)) + (s * 16)) + (i_16 >> 1))] < 32768) {
            AtomicAdd((&(dKV[(((Indices[((((((int)blockIdx.x) * 2048) + (i_i * 32)) + (s * 16)) + (i_16 >> 1))] * 576) + ((i_16 & 1) * 256)) + ((int)threadIdx.x))])), ((float*)buf_dyn_shmem)[(((i_16 * 256) + ((int)threadIdx.x)) + 5120)]);
          }
        }
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 4; ++i_17) {
        if (0 <= Indices[(((((((int)blockIdx.x) * 2048) + (i_i * 32)) + (s * 16)) + (i_17 * 4)) + (((int)threadIdx.x) >> 6))]) {
          if (Indices[(((((((int)blockIdx.x) * 2048) + (i_i * 32)) + (s * 16)) + (i_17 * 4)) + (((int)threadIdx.x) >> 6))] < 32768) {
            AtomicAdd((&(dKV[(((Indices[(((((((int)blockIdx.x) * 2048) + (i_i * 32)) + (s * 16)) + (i_17 * 4)) + (((int)threadIdx.x) >> 6))] * 576) + (((int)threadIdx.x) & 63)) + 512)])), ((float*)buf_dyn_shmem)[(((i_17 * 256) + ((int)threadIdx.x)) + 2048)]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 16; ++i_18) {
    tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) & 127) >> 5) * 8192) + ((((int)threadIdx.x) & 15) * 512)) + ((((int)threadIdx.x) >> 7) * 256)) + (i_18 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) + 59392)])), __pack_half2(((bfloat16_t)acc_dq[(i_18 * 8)]), ((bfloat16_t)acc_dq[((i_18 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_dq[((i_18 * 8) + 2)]), ((bfloat16_t)acc_dq[((i_18 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_dq[((i_18 * 8) + 4)]), ((bfloat16_t)acc_dq[((i_18 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_dq[((i_18 * 8) + 6)]), ((bfloat16_t)acc_dq[((i_18 * 8) + 7)])));
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 2; ++i_19) {
    tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((int)threadIdx.x) >> 7) * 32)) + (i_19 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) + 6144)])), __pack_half2(((bfloat16_t)acc_dq_tail[(i_19 * 8)]), ((bfloat16_t)acc_dq_tail[((i_19 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_dq_tail[((i_19 * 8) + 2)]), ((bfloat16_t)acc_dq_tail[((i_19 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_dq_tail[((i_19 * 8) + 4)]), ((bfloat16_t)acc_dq_tail[((i_19 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_dq_tail[((i_19 * 8) + 6)]), ((bfloat16_t)acc_dq_tail[((i_19 * 8) + 7)])));
  }
  __syncthreads();
  #pragma unroll
  for (int i_20 = 0; i_20 < 16; ++i_20) {
    *(uint4*)(dQ + ((((((int)blockIdx.x) * 36864) + (i_20 * 2304)) + ((((int)threadIdx.x) >> 6) * 576)) + ((((int)threadIdx.x) & 63) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_20 * 2048) + (((int)threadIdx.x) * 8)) + 59392));
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    *(uint4*)(dQ + (((((((int)blockIdx.x) * 36864) + (i_21 * 18432)) + ((((int)threadIdx.x) >> 3) * 576)) + ((((int)threadIdx.x) & 7) * 8)) + 512)) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_21 * 2048) + (((int)threadIdx.x) * 8)) + 6144));
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 184320);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 184320, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ KV, bfloat16_t* __restrict__ dO, int* __restrict__ Indices, float* __restrict__ Lse, float* __restrict__ Delta, bfloat16_t* __restrict__ dQ, float* __restrict__ dKV, cudaStream_t stream=cudaStreamDefault) {
	main_kernel<<<dim3(4096, 1, 1), dim3(256, 1, 1), 184320, stream>>>(Delta, Indices, KV, Lse, Q, dKV, dO, dQ);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}

