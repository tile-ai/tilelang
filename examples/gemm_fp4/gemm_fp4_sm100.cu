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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t mbar_mem[2];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  __shared__ __align__(16) uint C_tmem[1];
  tl::Tcgen05SMemDescriptor desc_a;
  tl::Tcgen05SMemDescriptor desc_b;
  float C_local[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(C_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
    mbar[1].init(1);
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(C_tmem[0])), 128);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    mbarrier[0].arrive_and_expect_tx(16256);
    tl::tma_load(A_desc, mbarrier[0], (&(((fp4_e2_t*)buf_dyn_shmem)[0])), 0, (((int)blockIdx.y) * 128));
    mbarrier[1].arrive_and_expect_tx(16256);
    tl::tma_load(B_desc, mbarrier[1], (&(((fp4_e2_t*)buf_dyn_shmem)[130048])), 0, (((int)blockIdx.x) * 128));
    mbarrier[2].arrive_and_expect_tx(16256);
    tl::tma_load(A_desc, mbarrier[2], (&(((fp4_e2_t*)buf_dyn_shmem)[97536])), 128, (((int)blockIdx.y) * 128));
    mbarrier[3].arrive_and_expect_tx(16256);
    tl::tma_load(B_desc, mbarrier[3], (&(((fp4_e2_t*)buf_dyn_shmem)[227584])), 128, (((int)blockIdx.x) * 128));
  }
  mbarrier[0].wait(0);
  mbarrier[1].wait(0);
  __syncthreads();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::initialize_tcgen05_descriptor(desc_a, (&(((fp4_e2_t*)buf_dyn_shmem)[0])), 1, 64, 0, 0, 2);
    tl::initialize_tcgen05_descriptor(desc_b, (&(((fp4_e2_t*)buf_dyn_shmem)[130048])), 1, 64, 0, 0, 2);
    #pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      tl::tcgen05mma_ss<tl::DataType::kFloat4_e2m1fn>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 32)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, ((0 < ki) ? 1 : 0), static_cast<uint32_t>(136320656), 0, 0, 0, 0);
    }
    tl::tcgen05_mma_arrive((&(mbar[0])));
  }
  mbar[0].wait(0);
  mbarrier[2].wait(0);
  mbarrier[3].wait(0);
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::initialize_tcgen05_descriptor(desc_a, (&(((fp4_e2_t*)buf_dyn_shmem)[97536])), 1, 64, 0, 0, 2);
    tl::initialize_tcgen05_descriptor(desc_b, (&(((fp4_e2_t*)buf_dyn_shmem)[227584])), 1, 64, 0, 0, 2);
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      tl::tcgen05mma_ss<tl::DataType::kFloat4_e2m1fn>(uint64_t(desc_a + (ki_1 * 32)), uint64_t(desc_b + (ki_1 * 32)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, 1, static_cast<uint32_t>(136320656), 0, 0, 0, 0);
    }
    tl::tcgen05_mma_arrive((&(mbar[1])));
  }
  mbar[1].wait(0);
  tl::tcgen05_ld_32dp32bNx<64, false>(C_tmem[0], ((((int)threadIdx.x) >> 7) * 64), (&(C_local[0])));
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float4*)(((float*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 7) * 8192) + ((i >> 3) * 4096)) + ((((int)threadIdx.x) & 127) * 32)) + (((((i & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((i & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((i & 1) + (((int)threadIdx.x) & 1)) & 1) * 4))) = *(float4*)(C_local + (i * 4));
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    tl::fence_proxy_async();
    tl::tma_store(C_desc, (&(((float*)buf_dyn_shmem)[0])), (((int)blockIdx.x) * 128), (((int)blockIdx.y) * 128));
    tl::tma_store(C_desc, (&(((float*)buf_dyn_shmem)[4096])), ((((int)blockIdx.x) * 128) + 32), (((int)blockIdx.y) * 128));
    tl::tma_store(C_desc, (&(((float*)buf_dyn_shmem)[8192])), ((((int)blockIdx.x) * 128) + 64), (((int)blockIdx.y) * 128));
    tl::tma_store(C_desc, (&(((float*)buf_dyn_shmem)[12288])), ((((int)blockIdx.x) * 128) + 96), (((int)blockIdx.y) * 128));
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(C_tmem[0])), 128);
  }
}

