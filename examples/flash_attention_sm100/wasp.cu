#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
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

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ K, __grid_constant__ const CUtensorMap Output_desc, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const bfloat16_t* __restrict__ K, __grid_constant__ const CUtensorMap Output_desc, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t mbar_dma1_empty_mem[2];
  auto mbar_dma1_empty = reinterpret_cast<Barrier*>(mbar_dma1_empty_mem);
  __shared__ __align__(16) uint64_t mbar_dma1_full_mem[2];
  auto mbar_dma1_full = reinterpret_cast<Barrier*>(mbar_dma1_full_mem);
  __shared__ __align__(16) uint64_t mbar_bmm1_empty_mem[2];
  auto mbar_bmm1_empty = reinterpret_cast<Barrier*>(mbar_bmm1_empty_mem);
  __shared__ __align__(16) uint64_t mbar_bmm1_full_mem[2];
  auto mbar_bmm1_full = reinterpret_cast<Barrier*>(mbar_bmm1_full_mem);
  __shared__ __align__(16) uint64_t mbar_dma2_empty_mem[2];
  auto mbar_dma2_empty = reinterpret_cast<Barrier*>(mbar_dma2_empty_mem);
  __shared__ __align__(16) uint64_t mbar_dma2_full_mem[2];
  auto mbar_dma2_full = reinterpret_cast<Barrier*>(mbar_dma2_full_mem);
  __shared__ __align__(16) uint64_t mbar_bmm2_full_mem[2];
  auto mbar_bmm2_full = reinterpret_cast<Barrier*>(mbar_bmm2_full_mem);
  __shared__ __align__(16) uint64_t mbar_softmax_empty_mem[2];
  auto mbar_softmax_empty = reinterpret_cast<Barrier*>(mbar_softmax_empty_mem);
  __shared__ __align__(16) uint64_t mbar_softmax_full_mem[2];
  auto mbar_softmax_full = reinterpret_cast<Barrier*>(mbar_softmax_full_mem);
  __shared__ __align__(16) uint64_t mbar_correction_full_mem[2];
  auto mbar_correction_full = reinterpret_cast<Barrier*>(mbar_correction_full_mem);
  __shared__ __align__(16) uint S_tmem[1];
  __shared__ __align__(16) uint P_tmem[1];
  __shared__ __align__(16) uint O_tmem[1];
  float O_reg[64];
  float logsum[1];
  float scores_max[1];
  float S_reg[64];
  float scores_max_prev[1];
  float scores_rescale[1];
  float scores_sum[1];
  bfloat16_t P_cast[64];
  float scores_max_clear[1];
  bfloat16_t O_shared_local_cast[8];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbar_dma1_empty[0].init(32);
    mbar_dma1_empty[1].init(32);
    mbar_dma1_full[0].init(32);
    mbar_dma1_full[1].init(32);
    mbar_bmm1_empty[0].init(128);
    mbar_bmm1_empty[1].init(128);
    mbar_bmm1_full[0].init(1);
    mbar_bmm1_full[1].init(1);
    mbar_dma2_empty[0].init(32);
    mbar_dma2_empty[1].init(32);
    mbar_dma2_full[0].init(32);
    mbar_dma2_full[1].init(32);
    mbar_bmm2_full[0].init(1);
    mbar_bmm2_full[1].init(1);
    mbar_softmax_empty[0].init(32);
    mbar_softmax_empty[1].init(32);
    mbar_softmax_full[0].init(128);
    mbar_softmax_full[1].init(128);
    mbar_correction_full[0].init(32);
    mbar_correction_full[1].init(32);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(S_tmem[0])), 64);
    tl::tmem_allocate((&(P_tmem[0])), 64);
    tl::tmem_allocate((&(O_tmem[0])), 64);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if (((int)threadIdx.x) < 128) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(O_reg + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    logsum[0] = 0x0p+0f/*0.000000e+00*/;
    scores_max[0] = -CUDART_INF_F;
    tl::tcgen05_st_32dp32bNx<64, false>(O_tmem[0], 0, (&(O_reg[0])));
  }
  for (int k = 0; k < 32; ++k) {
    bool is_clear_accum = (k == 0);
    if ((128 <= ((int)threadIdx.x)) && (((int)threadIdx.x) < 160)) {
      mbar_dma1_empty[(k & 1)].wait((((k >> 1) & 1) ^ 1));
      if (k == 0) {
        tl::tcgen05_before_thread_sync();
        tl::__sync_thread_partial<3, 32>();
        tl::tcgen05_after_thread_sync();
        #pragma unroll
        for (int i_1 = 0; i_1 < 32; ++i_1) {
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + ((i_1 >> 2) * 512)) + ((((i_1 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_1) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 73728)) = *(uint4*)(Q + (((((((((int)blockIdx.z) * 16777216) + (((int)blockIdx.x) * 262144)) + (i_1 * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) - 32768));
        }
      }
      if ((k % 2) == 0) {
        tl::tcgen05_before_thread_sync();
        tl::__sync_thread_partial<3, 32>();
        tl::tcgen05_after_thread_sync();
        #pragma unroll
        for (int i_2 = 0; i_2 < 64; ++i_2) {
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + ((i_2 >> 2) * 512)) + ((((i_2 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_2) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_2) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((((int)blockIdx.z) * 16777216) + (k * 524288)) + (i_2 * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) - 32768));
        }
      } else {
        tl::tcgen05_before_thread_sync();
        tl::__sync_thread_partial<3, 32>();
        tl::tcgen05_after_thread_sync();
        #pragma unroll
        for (int i_3 = 0; i_3 < 64; ++i_3) {
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + ((i_3 >> 2) * 512)) + ((((i_3 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_3) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_3) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = *(uint4*)(K + (((((((((int)blockIdx.z) * 16777216) + (k * 524288)) + (i_3 * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) - 32768));
        }
      }
      mbar_dma1_full[(k & 1)].arrive();
      mbar_dma2_empty[(k & 1)].wait((((k >> 1) & 1) ^ 1));
      if ((k % 2) == 0) {
        tl::tcgen05_before_thread_sync();
        tl::__sync_thread_partial<3, 32>();
        tl::tcgen05_after_thread_sync();
        #pragma unroll
        for (int i_4 = 0; i_4 < 64; ++i_4) {
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + ((i_4 >> 2) * 512)) + ((((i_4 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_4) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_4) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)) = *(uint4*)(V + (((((((((int)blockIdx.z) * 16777216) + (k * 524288)) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) - 32768));
        }
      } else {
        tl::tcgen05_before_thread_sync();
        tl::__sync_thread_partial<3, 32>();
        tl::tcgen05_after_thread_sync();
        #pragma unroll
        for (int i_5 = 0; i_5 < 64; ++i_5) {
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + ((i_5 >> 2) * 512)) + ((((i_5 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_5) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_5) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 49152)) = *(uint4*)(V + (((((((((int)blockIdx.z) * 16777216) + (k * 524288)) + (i_5 * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) - 32768));
        }
      }
      mbar_dma2_full[(k & 1)].arrive();
    } else {
      if ((160 <= ((int)threadIdx.x)) && (((int)threadIdx.x) < 192)) {
        mbar_dma1_full[(k & 1)].wait(((k >> 1) & 1));
        mbar_bmm1_empty[(k & 1)].wait((((k >> 1) & 1) ^ 1));
        tl::tcgen05_after_thread_sync();
        if ((k % 2) == 0) {
          {
            tl::Tcgen05SMemDescriptor desc_a;
            tl::Tcgen05SMemDescriptor desc_b;
            tl::initialize_tcgen05_descriptor(desc_a, (&(((bfloat16_t*)buf_dyn_shmem)[73728])), 1, 64, 0, 0, 2);
            tl::initialize_tcgen05_descriptor(desc_b, (&(((bfloat16_t*)buf_dyn_shmem)[0])), 1, 64, 0, 0, 2);
            tl::fence_proxy_async();
            #pragma unroll
            for (int ki = 0; ki < 8; ++ki) {
              tl::tcgen05mma_ws_ss<tl::DataType::kBFloat16>(uint64_t(desc_a + (((ki >> 2) * 8192) + ((ki & 3) * 32))), uint64_t(desc_b + (((ki >> 2) * 16384) + ((ki & 3) * 32))), (*reinterpret_cast<uint32_t*>(S_tmem)) + 0, ((0 < ki) ? 1 : 0), static_cast<uint32_t>(69207184), 0, 0, 0, 0);
            }
            tl::tcgen05_mma_arrive((&(mbar_bmm1_full[0])));
          }
        } else {
          {
            tl::Tcgen05SMemDescriptor desc_a_1;
            tl::Tcgen05SMemDescriptor desc_b_1;
            tl::initialize_tcgen05_descriptor(desc_a_1, (&(((bfloat16_t*)buf_dyn_shmem)[73728])), 1, 64, 0, 0, 2);
            tl::initialize_tcgen05_descriptor(desc_b_1, (&(((bfloat16_t*)buf_dyn_shmem)[16384])), 1, 64, 0, 0, 2);
            tl::fence_proxy_async();
            #pragma unroll
            for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
              tl::tcgen05mma_ws_ss<tl::DataType::kBFloat16>(uint64_t(desc_a_1 + (((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32))), uint64_t(desc_b_1 + (((ki_1 >> 2) * 16384) + ((ki_1 & 3) * 32))), (*reinterpret_cast<uint32_t*>(S_tmem)) + 0, ((0 < ki_1) ? 1 : 0), static_cast<uint32_t>(69207184), 0, 0, 0, 0);
            }
            tl::tcgen05_mma_arrive((&(mbar_bmm1_full[(k & 1)])));
          }
        }
        tl::tcgen05_before_thread_sync();
        mbar_dma1_empty[(k & 1)].arrive();
        mbar_softmax_full[(k & 1)].wait(((k >> 1) & 1));
        mbar_dma2_full[(k & 1)].wait(((k >> 1) & 1));
        tl::tcgen05_after_thread_sync();
        if ((k % 2) == 0) {
          {
            tl::Tcgen05SMemDescriptor desc_b_2;
            tl::initialize_tcgen05_descriptor(desc_b_2, (&(((bfloat16_t*)buf_dyn_shmem)[32768])), 1024, 64, 0, 0, 2);
            tl::fence_proxy_async();
            #pragma unroll
            for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
              tl::tcgen05mma_ts<tl::DataType::kBFloat16, false>( (*reinterpret_cast<uint32_t*>(P_tmem)) + (ki_2 * 4), uint64_t(desc_b_2 + (ki_2 * 2048)), (*reinterpret_cast<uint32_t*>(O_tmem)) + 0, ((0 < ki_2) ? 1 : (is_clear_accum ? 0 : 1)), static_cast<uint32_t>(69272720), 0, 0, 0, 0);
            }
            tl::tcgen05_mma_arrive((&(mbar_bmm2_full[0])));
          }
        } else {
          {
            tl::Tcgen05SMemDescriptor desc_b_3;
            tl::initialize_tcgen05_descriptor(desc_b_3, (&(((bfloat16_t*)buf_dyn_shmem)[49152])), 1024, 64, 0, 0, 2);
            tl::fence_proxy_async();
            #pragma unroll
            for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
              tl::tcgen05mma_ts<tl::DataType::kBFloat16, false>( (*reinterpret_cast<uint32_t*>(P_tmem)) + (ki_3 * 4), uint64_t(desc_b_3 + (ki_3 * 2048)), (*reinterpret_cast<uint32_t*>(O_tmem)) + 0, ((0 < ki_3) ? 1 : (is_clear_accum ? 0 : 1)), static_cast<uint32_t>(69272720), 0, 0, 0, 0);
            }
            tl::tcgen05_mma_arrive((&(mbar_bmm2_full[(k & 1)])));
          }
        }
        tl::tcgen05_before_thread_sync();
        mbar_softmax_empty[(k & 1)].arrive();
        mbar_dma2_empty[(k & 1)].arrive();
        if (k == 31) {
          mbar_correction_full[0].arrive();
        }
      } else {
        if (((int)threadIdx.x) < 128) {
          mbar_softmax_empty[(k & 1)].wait((((k >> 1) & 1) ^ 1));
          mbar_bmm1_full[(k & 1)].wait(((k >> 1) & 1));
          tl::tcgen05_after_thread_sync();
          if (0 < k) {
            mbar_bmm2_full[((k + 1) & 1)].wait((((k - 1) >> 1) & 1));
          }
          tl::tcgen05_ld_32dp32bNx<64, false>(O_tmem[0], 0, (&(O_reg[0])));
          tl::tcgen05_ld_32dp32bNx<64, false>(S_tmem[0], 0, (&(S_reg[0])));
          scores_max_prev[0] = scores_max[0];
          scores_max[0] = -CUDART_INF_F;
          scores_max_clear[0] = -CUDART_INF_F;
          #pragma unroll
          for (int rv = 0; rv < 64; ++rv) {
            scores_max_clear[0] = max(scores_max_clear[0], S_reg[rv]);
          }
          scores_max_clear[0] = tl::AllReduce<tl::MaxOp, 128, 64, 0, tl::NamedBarrier<128>>::run(scores_max_clear[0], (&(((float*)buf_dyn_shmem)[41088])));
          scores_max[0] = max(scores_max[0], scores_max_clear[0]);
          scores_max[0] = max(scores_max[0], scores_max_prev[0]);
          scores_rescale[0] = exp2f(((scores_max_prev[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
          #pragma unroll
          for (int i_6 = 0; i_6 < 64; ++i_6) {
            S_reg[i_6] = exp2f(((S_reg[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
          }
          scores_sum[0] = 0x0p+0f/*0.000000e+00*/;
          #pragma unroll
          for (int rv_1 = 0; rv_1 < 64; ++rv_1) {
            scores_sum[0] = (scores_sum[0] + S_reg[rv_1]);
          }
          scores_sum[0] = tl::AllReduce<tl::SumOp, 128, 64, 0, tl::NamedBarrier<128>>::run(scores_sum[0], (&(((float*)buf_dyn_shmem)[40960])));
          logsum[0] = ((logsum[0] * scores_rescale[0]) + scores_sum[0]);
          #pragma unroll
          for (int i_7 = 0; i_7 < 64; ++i_7) {
            O_reg[i_7] = (O_reg[i_7] * scores_rescale[0]);
          }
          #pragma unroll
          for (int i_8 = 0; i_8 < 16; ++i_8) {
            uint2 __1;
            float4 v_ = *(float4*)(S_reg + (i_8 * 4));
            (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
            (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
            *(uint2*)(P_cast + (i_8 * 4)) = __1;
          }
          tl::tcgen05_st_32dp32bNx<32, false>(P_tmem[0], 0, (&(P_cast[0])));
          tl::tcgen05_st_32dp32bNx<64, false>(O_tmem[0], 0, (&(O_reg[0])));
          tl::tcgen05_before_thread_sync();
          mbar_softmax_full[(k & 1)].arrive();
          mbar_bmm1_empty[(k & 1)].arrive();
          if (k == 31) {
            mbar_correction_full[0].wait(0);
            mbar_bmm2_full[(k & 1)].wait(((k >> 1) & 1));
            tl::tcgen05_after_thread_sync();
            tl::tcgen05_ld_32dp32bNx<64, false>(O_tmem[0], 0, (&(O_reg[0])));
            #pragma unroll
            for (int i_9 = 0; i_9 < 64; ++i_9) {
              O_reg[i_9] = (O_reg[i_9] / logsum[0]);
            }
            tl::tcgen05_before_thread_sync();
            tl::__sync_thread_partial<4, 128>();
            tl::tcgen05_after_thread_sync();
            #pragma unroll
            for (int i_10 = 0; i_10 < 8; ++i_10) {
              for (int vec = 0; vec < 2; ++vec) {
                uint2 __2;
                float4 v__1 = *(float4*)(O_reg + ((i_10 * 8) + (vec * 4)));
                (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
                (reinterpret_cast<__nv_bfloat162*>(&__2))[1] = __float22bfloat162_rn(((float2*)(&v__1))[1]);
                *(uint2*)(O_shared_local_cast + (vec * 4)) = __2;
              }
              *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((int)threadIdx.x) * 64) + ((((i_10 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_10 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((i_10 & 1) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 65536)) = *(uint4*)(O_shared_local_cast + 0);
            }
            tl::tcgen05_before_thread_sync();
            tl::__sync_thread_partial<4, 128>();
            tl::tcgen05_after_thread_sync();
            if (tl::tl_shuffle_elect<128>()) {
              tl::fence_proxy_async();
              tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[65536])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 64), ((int)blockIdx.z));
              tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[69632])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 64), ((int)blockIdx.z));
              tl::tma_store_arrive();
              tl::tma_store_wait<0>();
            }
          }
        }
      }
    }
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(S_tmem[0])), 64);
    tl::tmem_deallocate((&(P_tmem[0])), 64);
    tl::tmem_deallocate((&(O_tmem[0])), 64);
  }
}