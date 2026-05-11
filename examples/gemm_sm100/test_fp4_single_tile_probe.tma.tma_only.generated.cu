#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
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

extern "C" __global__ void main_kernel(const fp4_e2_t* __restrict__ A, const fp4_e2_t* __restrict__ B);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const fp4_e2_t* __restrict__ A, const fp4_e2_t* __restrict__ B) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t load_mbar_a_mem[1];
  auto load_mbar_a = reinterpret_cast<Barrier*>(load_mbar_a_mem);
  __shared__ __align__(16) uint64_t load_mbar_b_mem[1];
  auto load_mbar_b = reinterpret_cast<Barrier*>(load_mbar_b_mem);
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint C_tmem[1];
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
    load_mbar_a[0].expect_transaction(8192);
    tl::tma_load((&(((fp4_e2_t*)buf_dyn_shmem)[0])), (&(A[0])), load_mbar_a[0], 8192);
    load_mbar_b[0].expect_transaction(4096);
    tl::tma_load((&(((fp4_e2_t*)buf_dyn_shmem)[16384])), (&(B[0])), load_mbar_b[0], 4096);
  }
  load_mbar_a[0].arrive();
  load_mbar_b[0].arrive();
  load_mbar_a[0].wait(0);
  load_mbar_b[0].wait(0);
  tl::tcgen05_after_thread_sync();
  tl::fence_proxy_async();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(C_tmem[0])), 64);
  }
}

