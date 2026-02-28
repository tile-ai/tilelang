#include <tl_templates/cuda/instruction/tcgen05mma.h>
#include <tl_templates/cuda/tcgen_05.h>
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

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint D_tmem[1];
  __shared__ __align__(16) uint S_tmem[1];
  __shared__ __align__(16) uint P_tmem[1];
  __shared__ __align__(16) uint64_t mbar_s_mem[1];
  auto mbar_s = reinterpret_cast<Barrier*>(mbar_s_mem);
  __shared__ __align__(16) uint64_t mbar_d_mem[1];
  auto mbar_d = reinterpret_cast<Barrier*>(mbar_d_mem);
  float O_reg[128];
  float logsum[1];
  float scores_max[1];
  float S_reg[128];
  float scores_max_prev[1];
  float scores_scale[1];
  float scores_sum[1];
  bfloat16_t P_cast[128];
  float D_reg[128];
  tl::Tcgen05SMemDescriptor desc_a;
  tl::Tcgen05SMemDescriptor desc_b;
  float scores_max_clear[1];
  tl::Tcgen05SMemDescriptor desc_b_1;
  float scores_max_clear_1[1];
  bfloat16_t O_shared_local_cast[8];
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(D_tmem[0])), 128);
    tl::tmem_allocate((&(S_tmem[0])), 128);
    tl::tmem_allocate((&(P_tmem[0])), 128);
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<0>()) {
    mbar_s[0].init(1);
    mbar_d[0].init(1);
  }
  tl::fence_barrier_init();
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.x) * 131072)) + (i * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(O_reg + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  logsum[0] = 0x0p+0f/*0.000000e+00*/;
  scores_max[0] = -CUDART_INF_F;
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(K[(((((((int)blockIdx.z) * 2097152) + (i_2 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])), (&(V[(((((((int)blockIdx.z) * 2097152) + (i_3 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 15; ++k) {
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a, (&(((bfloat16_t*)buf_dyn_shmem)[0])), 1, 64, 0, 0, 2);
      tl::initialize_tcgen05_descriptor(desc_b, (&(((bfloat16_t*)buf_dyn_shmem)[16384])), 1, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::tcgen05mma_ss<tl::DataType::kBFloat16>(uint64_t(desc_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))), uint64_t(desc_b + (((ki >> 2) * 16384) + ((ki & 3) * 32))), (*reinterpret_cast<uint32_t*>(S_tmem)) + 0, ((0 < ki) ? 1 : 0), static_cast<uint32_t>(136316048), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive((&(mbar_s[0])));
    }
    mbar_s[0].wait((k & 1));
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(K[(((((((((int)blockIdx.z) * 2097152) + (k * 131072)) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072)])));
    }
    tl::cp_async_commit();
    tl::tcgen05_ld_32dp32bNx<128, false>(S_tmem[0], 0, (&(S_reg[0])));
    scores_max_prev[0] = scores_max[0];
    scores_max[0] = -CUDART_INF_F;
    scores_max_clear[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 128; ++rv) {
      scores_max_clear[0] = max(scores_max_clear[0], S_reg[rv]);
    }
    scores_max[0] = max(scores_max[0], scores_max_clear[0]);
    scores_max[0] = max(scores_max[0], scores_max_prev[0]);
    scores_scale[0] = exp2f(((scores_max_prev[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    #pragma unroll
    for (int i_5 = 0; i_5 < 128; ++i_5) {
      S_reg[i_5] = exp2f(((S_reg[i_5] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    scores_sum[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 128; ++rv_1) {
      scores_sum[0] = (scores_sum[0] + S_reg[rv_1]);
    }
    logsum[0] = ((logsum[0] * scores_scale[0]) + scores_sum[0]);
    #pragma unroll
    for (int i_6 = 0; i_6 < 128; ++i_6) {
      O_reg[i_6] = (O_reg[i_6] * scores_scale[0]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 32; ++i_7) {
      uint2 __1;
      float4 v_ = *(float4*)(S_reg + (i_7 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
      *(uint2*)(P_cast + (i_7 * 4)) = __1;
    }
    tl::tcgen05_st_32dp32bNx<64, false>(P_tmem[0], 0, (&(P_cast[0])));
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_b_1, (&(((bfloat16_t*)buf_dyn_shmem)[32768])), 1024, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
        tl::tcgen05mma_ts<tl::DataType::kBFloat16>( (*reinterpret_cast<uint32_t*>(P_tmem)) + (ki_1 * 8), uint64_t(desc_b_1 + (ki_1 * 2048)), (*reinterpret_cast<uint32_t*>(D_tmem)) + 0, ((0 < ki_1) ? 1 : 0), static_cast<uint32_t>(136381584), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive((&(mbar_d[0])));
    }
    mbar_d[0].wait((k & 1));
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_8 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])), (&(V[(((((((((int)blockIdx.z) * 2097152) + (k * 131072)) + (i_8 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072)])));
    }
    tl::cp_async_commit();
    tl::tcgen05_ld_32dp32bNx<128, false>(D_tmem[0], 0, (&(D_reg[0])));
    #pragma unroll
    for (int i_9 = 0; i_9 < 128; ++i_9) {
      O_reg[i_9] = (O_reg[i_9] + D_reg[i_9]);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::initialize_tcgen05_descriptor(desc_a, (&(((bfloat16_t*)buf_dyn_shmem)[0])), 1, 64, 0, 0, 2);
    tl::initialize_tcgen05_descriptor(desc_b, (&(((bfloat16_t*)buf_dyn_shmem)[16384])), 1, 64, 0, 0, 2);
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::tcgen05mma_ss<tl::DataType::kBFloat16>(uint64_t(desc_a + (((ki_2 >> 2) * 16384) + ((ki_2 & 3) * 32))), uint64_t(desc_b + (((ki_2 >> 2) * 16384) + ((ki_2 & 3) * 32))), (*reinterpret_cast<uint32_t*>(S_tmem)) + 0, ((0 < ki_2) ? 1 : 0), static_cast<uint32_t>(136316048), 0, 0, 0, 0);
    }
    tl::tcgen05_mma_arrive((&(mbar_s[0])));
  }
  mbar_s[0].wait(1);
  tl::tcgen05_ld_32dp32bNx<128, false>(S_tmem[0], 0, (&(S_reg[0])));
  scores_max_prev[0] = scores_max[0];
  scores_max[0] = -CUDART_INF_F;
  scores_max_clear_1[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv_2 = 0; rv_2 < 128; ++rv_2) {
    scores_max_clear_1[0] = max(scores_max_clear_1[0], S_reg[rv_2]);
  }
  scores_max[0] = max(scores_max[0], scores_max_clear_1[0]);
  scores_max[0] = max(scores_max[0], scores_max_prev[0]);
  scores_scale[0] = exp2f(((scores_max_prev[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  #pragma unroll
  for (int i_10 = 0; i_10 < 128; ++i_10) {
    S_reg[i_10] = exp2f(((S_reg[i_10] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  scores_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_3 = 0; rv_3 < 128; ++rv_3) {
    scores_sum[0] = (scores_sum[0] + S_reg[rv_3]);
  }
  logsum[0] = ((logsum[0] * scores_scale[0]) + scores_sum[0]);
  #pragma unroll
  for (int i_11 = 0; i_11 < 128; ++i_11) {
    O_reg[i_11] = (O_reg[i_11] * scores_scale[0]);
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 32; ++i_12) {
    uint2 __2;
    float4 v__1 = *(float4*)(S_reg + (i_12 * 4));
    (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__2))[1] = __float22bfloat162_rn(((float2*)(&v__1))[1]);
    *(uint2*)(P_cast + (i_12 * 4)) = __2;
  }
  tl::tcgen05_st_32dp32bNx<64, false>(P_tmem[0], 0, (&(P_cast[0])));
  tl::cp_async_wait<0>();
  __syncthreads();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::initialize_tcgen05_descriptor(desc_b_1, (&(((bfloat16_t*)buf_dyn_shmem)[32768])), 1024, 64, 0, 0, 2);
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
      tl::tcgen05mma_ts<tl::DataType::kBFloat16>( (*reinterpret_cast<uint32_t*>(P_tmem)) + (ki_3 * 8), uint64_t(desc_b_1 + (ki_3 * 2048)), (*reinterpret_cast<uint32_t*>(D_tmem)) + 0, ((0 < ki_3) ? 1 : 0), static_cast<uint32_t>(136381584), 0, 0, 0, 0);
    }
    tl::tcgen05_mma_arrive((&(mbar_d[0])));
  }
  mbar_d[0].wait(1);
  tl::tcgen05_ld_32dp32bNx<128, false>(D_tmem[0], 0, (&(D_reg[0])));
  #pragma unroll
  for (int i_13 = 0; i_13 < 128; ++i_13) {
    O_reg[i_13] = (O_reg[i_13] + D_reg[i_13]);
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 128; ++i_14) {
    O_reg[i_14] = (O_reg[i_14] / logsum[0]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_15 = 0; i_15 < 16; ++i_15) {
    for (int vec = 0; vec < 2; ++vec) {
      uint2 __3;
      float4 v__2 = *(float4*)(O_reg + ((i_15 * 8) + (vec * 4)));
      (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
      *(uint2*)(O_shared_local_cast + (vec * 4)) = __3;
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 128) + (i_15 * 8))) = *(uint4*)(O_shared_local_cast + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_16 = 0; i_16 < 16; ++i_16) {
    *(uint4*)(Output + ((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.x) * 131072)) + (i_16 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((i_16 * 1024) + (((int)threadIdx.x) * 8)));
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(D_tmem[0])), 128);
    tl::tmem_deallocate((&(S_tmem[0])), 128);
    tl::tmem_deallocate((&(P_tmem[0])), 128);
  }
}
