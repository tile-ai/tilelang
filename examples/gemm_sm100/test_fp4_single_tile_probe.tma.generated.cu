#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/tcgen05mma.h>
#include <tl_templates/cuda/tcgen_05.h>
#include <tl_templates/cuda/cuda_fp4.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t load_mbar_a_mem[1];
  auto load_mbar_a = reinterpret_cast<Barrier*>(load_mbar_a_mem);
  __shared__ __align__(16) uint64_t load_mbar_b_mem[1];
  auto load_mbar_b = reinterpret_cast<Barrier*>(load_mbar_b_mem);
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint C_tmem[1];
  float C_local[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    load_mbar_a[0].init(128);
    load_mbar_b[0].init(128);
    mbar[0].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(C_tmem[0])), 64);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if (tl::tl_shuffle_elect<128>()) {
    load_mbar_a[0].expect_transaction(16256);
    tl::tma_load(A_desc, load_mbar_a[0], (&(((fp4_e2_t*)buf_dyn_shmem)[0])), 0, 0);
    load_mbar_b[0].expect_transaction(8128);
    tl::tma_load(B_desc, load_mbar_b[0], (&(((fp4_e2_t*)buf_dyn_shmem)[32512])), 0, 0);
  }
  load_mbar_a[0].arrive();
  load_mbar_b[0].arrive();
  load_mbar_a[0].wait(0);
  load_mbar_b[0].wait(0);
  tl::tcgen05_after_thread_sync();
  tl::fence_proxy_async();
  {
    tl::Tcgen05SMemDescriptor desc_a;
    tl::Tcgen05SMemDescriptor desc_b;
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a, (&(((fp4_e2_t*)buf_dyn_shmem)[0])), 1, 64, 0, 0, 2);
      tl::initialize_tcgen05_descriptor(desc_b, (&(((fp4_e2_t*)buf_dyn_shmem)[32512])), 1, 64, 0, 0, 2);
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::tcgen05mma_ws_ss<tl::DataType::kFloat4_e2m1fn>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 32)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, ((0 < ki) ? 1 : 0), static_cast<uint32_t>(135272080), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
  }
  mbar[0].wait(0);
  tl::tcgen05_after_thread_sync();
  tl::tcgen05_ld_32dp32bNx<64, false>(C_tmem[0], 0, (&(C_local[0])));
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float4*)(((float*)buf_dyn_shmem) + ((((int)threadIdx.x) * 64) + (i * 4))) = *(float4*)(C_local + (i * 4));
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(C[0])), (&(((float*)buf_dyn_shmem)[0])), 32768);
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(C_tmem[0])), 64);
  }
}

