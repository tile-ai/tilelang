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
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const fp4_e2_t* __restrict__ A_bytes, const fp4_e2_t* __restrict__ B_bytes, float* __restrict__ C, const uint* __restrict__ SFA, const uint* __restrict__ SFB);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const fp4_e2_t* __restrict__ A_bytes, const fp4_e2_t* __restrict__ B_bytes, float* __restrict__ C, const uint* __restrict__ SFA, const uint* __restrict__ SFB) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* C_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  void* SFA_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* SFB_shared = ((void*)((char*)buf_dyn_shmem + 16896));
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint C_tmem[1];
  __shared__ __align__(16) uint sfb_data[1];
  __shared__ __align__(16) uint sfa_data[1];
  float C_local[128];
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(C_tmem[0])), 128);
    tl::tmem_allocate((&(sfb_data[0])), 32);
    tl::tmem_allocate((&(sfa_data[0])), 32);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(((uchar*)A_shared) + ((((i * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(uint4*)(((uchar*)A_bytes) + ((i * 2048) + (((int)threadIdx.x) * 16)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(uint4*)(((uchar*)B_shared) + ((((i_1 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(uint4*)(((uchar*)B_bytes) + ((i_1 * 2048) + (((int)threadIdx.x) * 16)));
  }
  ((uint*)SFA_shared)[((int)threadIdx.x)] = SFA[((int)threadIdx.x)];
  ((uint*)SFB_shared)[((int)threadIdx.x)] = SFB[((int)threadIdx.x)];
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if (((int)threadIdx.x) < 32) {
    void* chunk_ptr = (&(((uint*)SFA_shared)[0]));
    tl::tcgen05_sf_warp_transpose(reinterpret_cast<uint32_t*>(chunk_ptr));
    void* chunk_ptr_1 = (&(((uint*)SFB_shared)[0]));
    tl::tcgen05_sf_warp_transpose(reinterpret_cast<uint32_t*>(chunk_ptr_1));
    tl::fence_proxy_async();
    tl::tcgen05_before_thread_sync();
    tl::__sync_thread_partial(3, 32);
    tl::tcgen05_after_thread_sync();
    void* chunk_ptr_2 = (&(((uint*)SFA_shared)[0]));
    tl::tcgen05_cp<false>(tl::make_sf_smem_desc(reinterpret_cast<void*>(chunk_ptr_2)), (*reinterpret_cast<uint32_t*>(sfa_data)) + 0);
    void* chunk_ptr_3 = (&(((uint*)SFB_shared)[0]));
    tl::tcgen05_cp<false>(tl::make_sf_smem_desc(reinterpret_cast<void*>(chunk_ptr_3)), (*reinterpret_cast<uint32_t*>(sfb_data)) + 0);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((32 <= ((int)threadIdx.x)) && (((int)threadIdx.x) < 64)) {
    {
      tl::Tcgen05SMemDescriptor v;
      tl::Tcgen05SMemDescriptor v_1;
      tl::initialize_tcgen05_descriptor(v, (&(((uchar*)A_shared)[0])), 1, 32, 0, 0, 4);
      tl::initialize_tcgen05_descriptor(v_1, (&(((uchar*)B_shared)[0])), 1, 32, 0, 0, 4);
      #pragma unroll
      for (int ki = 0; ki < 2; ++ki) {
        tl::tcgen05mma_mxf4nvf4_blockscaled_ss<false>(uint64_t(v + (ki * 32)), uint64_t(v_1 + (ki * 32)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, ((0 < ki) ? 1 : 0), static_cast<uint32_t>(136316032), (*reinterpret_cast<uint32_t*>(sfa_data)) + 0, (*reinterpret_cast<uint32_t*>(sfb_data)) + 0);
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
  }
  mbar[0].wait(0);
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  tl::tcgen05_ld_32dp32bNx<128, false>(C_tmem[0], 0, (&(C_local[0])));
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    *(float4*)(((float*)C_shared) + ((((int)threadIdx.x) * 128) + (i_2 * 4))) = *(float4*)(C_local + (i_2 * 4));
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(C[0])), (&(((float*)C_shared)[0])), 65536);
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(sfb_data[0])), 32);
    tl::tmem_deallocate((&(sfa_data[0])), 32);
    tl::tmem_deallocate((&(C_tmem[0])), 128);
  }
}

