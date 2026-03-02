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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint S_tmem[1];
  __shared__ __align__(16) uint D_tmem[1];
  __shared__ __align__(16) uint64_t mbar_k_loaded_mem[1];
  auto mbar_k_loaded = reinterpret_cast<Barrier*>(mbar_k_loaded_mem);
  __shared__ __align__(16) uint64_t mbar_k_consumed_mem[1];
  auto mbar_k_consumed = reinterpret_cast<Barrier*>(mbar_k_consumed_mem);
  __shared__ __align__(16) uint64_t mbar_v_loaded_mem[1];
  auto mbar_v_loaded = reinterpret_cast<Barrier*>(mbar_v_loaded_mem);
  __shared__ __align__(16) uint64_t mbar_v_consumed_mem[1];
  auto mbar_v_consumed = reinterpret_cast<Barrier*>(mbar_v_consumed_mem);
  __shared__ __align__(16) uint64_t mbar_s_full_mem[1];
  auto mbar_s_full = reinterpret_cast<Barrier*>(mbar_s_full_mem);
  __shared__ __align__(16) uint64_t mbar_softmax_done_mem[1];
  auto mbar_softmax_done = reinterpret_cast<Barrier*>(mbar_softmax_done_mem);
  __shared__ __align__(16) uint64_t mbar_d_full_mem[1];
  auto mbar_d_full = reinterpret_cast<Barrier*>(mbar_d_full_mem);
  float logsum[1];
  float O_reg[128];
  tl::Tcgen05SMemDescriptor desc_a;
  tl::Tcgen05SMemDescriptor desc_b;
  tl::Tcgen05SMemDescriptor desc_a_1;
  tl::Tcgen05SMemDescriptor desc_b_1;
  float S_reg[128];
  float scores_max[1];
  float scores_max_prev[1];
  float scores_max_clear[1];
  float scores_scale[1];
  float scores_sum[1];
  bfloat16_t P_cast[128];
  float D_reg[128];
  bfloat16_t O_shared_local_cast[8];
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(S_tmem[0])), 128);
    tl::tmem_allocate((&(D_tmem[0])), 128);
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<0>()) {
    mbar_k_loaded[0].init(32);
    mbar_k_consumed[0].init(1);
    mbar_v_loaded[0].init(32);
    mbar_v_consumed[0].init(1);
    mbar_s_full[0].init(1);
    mbar_softmax_done[0].init(128);
    mbar_d_full[0].init(1);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 49152)) = *(uint4*)(Q + ((((((((int)blockIdx.z) * 131072) + (((int)blockIdx.x) * 65536)) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    for (int k = 0; k < 2; ++k) {
      mbar_k_consumed[0].wait((k ^ 1));
      #pragma unroll
      for (int i_1 = 0; i_1 < 64; ++i_1) {
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((((int)blockIdx.z) * 131072) + (k * 65536)) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
      }
      mbar_k_loaded[0].arrive();
      mbar_v_consumed[0].wait((k ^ 1));
      #pragma unroll
      for (int i_2 = 0; i_2 < 64; ++i_2) {
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_2 * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 65536)) = *(uint4*)(V + ((((((((int)blockIdx.z) * 131072) + (k * 65536)) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
      }
      mbar_v_loaded[0].arrive();
    }
  } else {
    if (((int)threadIdx.x) < 64) {
      for (int k_1 = 0; k_1 < 2; ++k_1) {
        mbar_k_loaded[0].wait(k_1);
        tl::initialize_tcgen05_descriptor(desc_a, (&(((bfloat16_t*)buf_dyn_shmem)[49152])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b, (&(((bfloat16_t*)buf_dyn_shmem)[0])), 1, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::tcgen05mma_ss<tl::DataType::kBFloat16>(uint64_t(desc_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))), uint64_t(desc_b + (((ki >> 2) * 16384) + ((ki & 3) * 32))), (*reinterpret_cast<uint32_t*>(S_tmem)) + 0, ((0 < ki) ? 1 : 0), static_cast<uint32_t>(136316048), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(mbar_k_consumed[0])));
        tl::tcgen05_mma_arrive((&(mbar_s_full[0])));
        mbar_softmax_done[0].wait(k_1);
        mbar_v_loaded[0].wait(k_1);
        tl::initialize_tcgen05_descriptor(desc_a_1, (&(((bfloat16_t*)buf_dyn_shmem)[32768])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b_1, (&(((bfloat16_t*)buf_dyn_shmem)[65536])), 1024, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
          tl::tcgen05mma_ss<tl::DataType::kBFloat16>(uint64_t(desc_a_1 + (((ki_1 >> 2) * 16384) + ((ki_1 & 3) * 32))), uint64_t(desc_b_1 + (ki_1 * 2048)), (*reinterpret_cast<uint32_t*>(D_tmem)) + 0, ((0 < ki_1) ? 1 : 0), static_cast<uint32_t>(136381584), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(mbar_v_consumed[0])));
        tl::tcgen05_mma_arrive((&(mbar_d_full[0])));
      }
    } else {
      if (128 <= ((int)threadIdx.x)) {
        for (int k_2 = 0; k_2 < 2; ++k_2) {
          mbar_s_full[0].wait(k_2);
          tl::__sync_thread_partial<1, 128>();
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
          for (int i_3 = 0; i_3 < 128; ++i_3) {
            S_reg[i_3] = exp2f(((S_reg[i_3] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
          }
          scores_sum[0] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_1 = 0; rv_1 < 128; ++rv_1) {
            scores_sum[0] = (scores_sum[0] + S_reg[rv_1]);
          }
          logsum[0] = ((logsum[0] * scores_scale[0]) + scores_sum[0]);
          #pragma unroll
          for (int i_4 = 0; i_4 < 128; ++i_4) {
            O_reg[i_4] = (O_reg[i_4] * scores_scale[0]);
          }
          #pragma unroll
          for (int i_5 = 0; i_5 < 32; ++i_5) {
            uint2 __1;
            float4 v_ = *(float4*)(S_reg + (i_5 * 4));
            (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
            (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
            *(uint2*)(P_cast + (i_5 * 4)) = __1;
          }
          tl::__sync_thread_partial<3, 128>();
          #pragma unroll
          for (int i_6 = 0; i_6 < 16; ++i_6) {
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_6 >> 3) * 8192) + (((int)threadIdx.x) * 64)) + (((((i_6 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_6 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((i_6 & 1) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)) = *(uint4*)(P_cast + (i_6 * 8));
          }
          mbar_softmax_done[0].arrive();
          mbar_d_full[0].wait(k_2);
          tl::__sync_thread_partial<1, 128>();
          tl::tcgen05_ld_32dp32bNx<128, false>(D_tmem[0], 0, (&(D_reg[0])));
          #pragma unroll
          for (int i_7 = 0; i_7 < 128; ++i_7) {
            O_reg[i_7] = (O_reg[i_7] + D_reg[i_7]);
          }
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 128; ++i_8) {
          O_reg[i_8] = (O_reg[i_8] / logsum[0]);
        }
        #pragma unroll
        for (int i_9 = 0; i_9 < 16; ++i_9) {
          for (int vec = 0; vec < 2; ++vec) {
            uint2 __2;
            float4 v__1 = *(float4*)(O_reg + ((i_9 * 8) + (vec * 4)));
            (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
            (reinterpret_cast<__nv_bfloat162*>(&__2))[1] = __float22bfloat162_rn(((float2*)(&v__1))[1]);
            *(uint2*)(O_shared_local_cast + (vec * 4)) = __2;
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 128) + (i_9 * 8))) = *(uint4*)(O_shared_local_cast + 0);
        }
        tl::__sync_thread_partial<3, 128>();
        #pragma unroll
        for (int i_10 = 0; i_10 < 16; ++i_10) {
          *(uint4*)(Output + (((((((((int)blockIdx.z) * 131072) + (((int)blockIdx.x) * 65536)) + (i_10 * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) - 4096)) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_10 * 1024) + (((int)threadIdx.x) * 8)) + 15360));
        }
      }
    }
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(S_tmem[0])), 128);
    tl::tmem_deallocate((&(D_tmem[0])), 128);
  }
}

