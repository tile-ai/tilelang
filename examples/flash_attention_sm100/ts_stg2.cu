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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t mbar_s_mem[2];
  auto mbar_s = reinterpret_cast<Barrier*>(mbar_s_mem);
  __shared__ __align__(16) uint64_t mbar_d_mem[2];
  auto mbar_d = reinterpret_cast<Barrier*>(mbar_d_mem);
  __shared__ __align__(16) uint64_t mbarrier_mem[9];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  __shared__ __align__(16) uint S_tmem[1];
  __shared__ __align__(16) uint D_tmem[1];
  __shared__ __align__(16) uint P_operand[1];
  float O_reg[64];
  float logsum[1];
  float scores_max[1];
  float S_reg[64];
  float scores_max_prev[1];
  float scores_max_clear[1];
  float scores_scale[1];
  float scores_sum[1];
  bfloat16_t P_cast[64];
  float D_reg[64];
  bfloat16_t O_shared_local_cast[8];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(Output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbar_s[0].init(1);
    mbar_s[1].init(1);
    mbar_d[0].init(1);
    mbar_d[1].init(1);
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
    mbarrier[4].init(256);
    mbarrier[5].init(256);
    mbarrier[6].init(256);
    mbarrier[7].init(256);
    mbarrier[8].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(S_tmem[0])), 128);
    tl::tmem_allocate((&(D_tmem[0])), 128);
    tl::tmem_allocate((&(P_operand[0])), 128);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if (tl::tl_shuffle_elect<384>()) {
    mbarrier[8].arrive_and_expect_tx(32768);
    tl::tma_load(Q_desc, mbarrier[8], (&(((bfloat16_t*)buf_dyn_shmem)[0])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 128), ((int)blockIdx.z));
    tl::tma_load(Q_desc, mbarrier[8], (&(((bfloat16_t*)buf_dyn_shmem)[8192])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 128), ((int)blockIdx.z));
  }
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 32; ++k) {
      mbarrier[((k & 1) + 4)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k & 1)].arrive_and_expect_tx(32768);
        tl::tma_load(K_desc, mbarrier[(k & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 16384)])), 0, ((int)blockIdx.y), (k * 128), ((int)blockIdx.z));
        tl::tma_load(K_desc, mbarrier[(k & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 24576)])), 64, ((int)blockIdx.y), (k * 128), ((int)blockIdx.z));
      }
      mbarrier[((k & 1) + 6)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((k & 1) + 2)].arrive_and_expect_tx(32768);
        tl::tma_load(V_desc, mbarrier[((k & 1) + 2)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 49152)])), 0, ((int)blockIdx.y), (k * 128), ((int)blockIdx.z));
        tl::tma_load(V_desc, mbarrier[((k & 1) + 2)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 57344)])), 64, ((int)blockIdx.y), (k * 128), ((int)blockIdx.z));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(O_reg + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    logsum[0] = 0x0p+0f/*0.000000e+00*/;
    scores_max[0] = -CUDART_INF_F;
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      if (k_1 == 0) {
        mbarrier[8].wait(0);
      }
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      tl::tcgen05_after_thread_sync();
      {
        tl::Tcgen05SMemDescriptor desc_a;
        tl::Tcgen05SMemDescriptor desc_b;
        tl::tcgen05_before_thread_sync();
        tl::__sync_thread_partial<3, 256>();
        tl::tcgen05_after_thread_sync();
        if ((((int)threadIdx.x) >> 5) == 0) {
          tl::initialize_tcgen05_descriptor(desc_a, (&(((bfloat16_t*)buf_dyn_shmem)[0])), 1, 64, 0, 0, 2);
          tl::initialize_tcgen05_descriptor(desc_b, (&(((bfloat16_t*)buf_dyn_shmem)[(((k_1 & 1) * 16384) + 16384)])), 1, 64, 0, 0, 2);
          tl::fence_proxy_async();
          #pragma unroll
          for (int ki = 0; ki < 8; ++ki) {
            tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(uint64_t(desc_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))), uint64_t(desc_b + (((ki >> 2) * 16384) + ((ki & 3) * 32))), (*reinterpret_cast<uint32_t*>(S_tmem)) + 0, ((0 < ki) ? 1 : 0), static_cast<uint32_t>(136316048), 0, 0, 0, 0);
          }
          tl::tcgen05_mma_arrive((&(mbar_s[(k_1 & 1)])));
        }
      }
      tl::tcgen05_before_thread_sync();
      mbarrier[((k_1 & 1) + 4)].arrive();
      mbar_s[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      tl::tcgen05_after_thread_sync();
      tl::tcgen05_ld_32dp32bNx<64, false>(S_tmem[0], ((((int)threadIdx.x) >> 7) * 64), (&(S_reg[0])));
      scores_max_prev[0] = scores_max[0];
      scores_max[0] = -CUDART_INF_F;
      scores_max_clear[0] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 64; ++rv) {
        scores_max_clear[0] = max(scores_max_clear[0], S_reg[rv]);
      }
      tl::tcgen05_before_thread_sync();
      tl::__sync_thread_partial<3, 256>();
      tl::tcgen05_after_thread_sync();
      scores_max_clear[0] = tl::AllReduce<tl::MaxOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_max_clear[0], (&(((float*)buf_dyn_shmem)[41216])));
      scores_max[0] = max(scores_max[0], scores_max_clear[0]);
      scores_max[0] = max(scores_max[0], scores_max_prev[0]);
      scores_scale[0] = exp2f(((scores_max_prev[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      #pragma unroll
      for (int i_1 = 0; i_1 < 64; ++i_1) {
        S_reg[i_1] = exp2f(((S_reg[i_1] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[0] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
      }
      scores_sum[0] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 64; ++rv_1) {
        scores_sum[0] = (scores_sum[0] + S_reg[rv_1]);
      }
      scores_sum[0] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(scores_sum[0], (&(((float*)buf_dyn_shmem)[40960])));
      logsum[0] = ((logsum[0] * scores_scale[0]) + scores_sum[0]);
      #pragma unroll
      for (int i_2 = 0; i_2 < 64; ++i_2) {
        O_reg[i_2] = (O_reg[i_2] * scores_scale[0]);
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        uint2 __1;
        float4 v_ = *(float4*)(S_reg + (i_3 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
        *(uint2*)(P_cast + (i_3 * 4)) = __1;
      }
      tl::tcgen05_st_32dp32bNx<32, false>(P_operand[0], ((((int)threadIdx.x) >> 7) * 32), (&(P_cast[0])));
      mbarrier[((k_1 & 1) + 2)].wait(((k_1 & 3) >> 1));
      tl::tcgen05_after_thread_sync();
      {
        tl::Tcgen05SMemDescriptor desc_b_1;
        if ((((int)threadIdx.x) >> 5) == 0) {
          tl::initialize_tcgen05_descriptor(desc_b_1, (&(((bfloat16_t*)buf_dyn_shmem)[(((k_1 & 1) * 16384) + 49152)])), 1024, 64, 0, 0, 2);
          tl::fence_proxy_async();
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
            tl::tcgen05mma_ts<tl::DataType::kBFloat16, false>( (*reinterpret_cast<uint32_t*>(P_operand)) + (ki_1 * 8), uint64_t(desc_b_1 + (ki_1 * 2048)), (*reinterpret_cast<uint32_t*>(D_tmem)) + 0, ((0 < ki_1) ? 1 : 0), static_cast<uint32_t>(136381584), 0, 0, 0, 0);
          }
          tl::tcgen05_mma_arrive((&(mbar_d[(k_1 & 1)])));
        }
      }
      tl::tcgen05_before_thread_sync();
      mbarrier[((k_1 & 1) + 6)].arrive();
      mbar_d[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      tl::tcgen05_after_thread_sync();
      tl::tcgen05_ld_32dp32bNx<64, false>(D_tmem[0], ((((int)threadIdx.x) >> 7) * 64), (&(D_reg[0])));
      #pragma unroll
      for (int i_4 = 0; i_4 < 64; ++i_4) {
        O_reg[i_4] = (O_reg[i_4] + D_reg[i_4]);
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 64; ++i_5) {
      O_reg[i_5] = (O_reg[i_5] / logsum[0]);
    }
    tl::tcgen05_before_thread_sync();
    tl::__sync_thread_partial<3, 256>();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      for (int vec = 0; vec < 2; ++vec) {
        uint2 __2;
        float4 v__1 = *(float4*)(O_reg + ((i_6 * 8) + (vec * 4)));
        (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__2))[1] = __float22bfloat162_rn(((float2*)(&v__1))[1]);
        *(uint2*)(O_shared_local_cast + (vec * 4)) = __2;
      }
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((int)threadIdx.x) * 64) + ((((i_6 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_6 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((i_6 & 1) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(O_shared_local_cast + 0);
    }
    tl::tcgen05_before_thread_sync();
    tl::__sync_thread_partial<3, 256>();
    tl::tcgen05_after_thread_sync();
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 128), ((int)blockIdx.z));
      tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[8192])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 128), ((int)blockIdx.z));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(S_tmem[0])), 128);
    tl::tmem_deallocate((&(D_tmem[0])), 128);
    tl::tmem_deallocate((&(P_operand[0])), 128);
  }
}