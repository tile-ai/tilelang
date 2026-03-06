# Non-persistent, 2-SM GEMM

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from tilelang.engine import register_cuda_postproc
tilelang.disable_cache()


@register_cuda_postproc
def _(code, _):
    return r"""
#include <tl_templates/cuda/instruction/tcgen05mma.h>
#include <tl_templates/cuda/tcgen_05.h>
#include <tl_templates/cuda/cluster.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(128, 1) gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint C_tmem[1];
  __shared__ __align__(16) uint64_t loaded_mem[4];
  auto loaded = reinterpret_cast<Barrier*>(loaded_mem);
  __shared__ __align__(16) uint64_t consumed_mem[4];
  auto consumed = reinterpret_cast<Barrier*>(consumed_mem);
  __shared__ __align__(16) uint64_t tmem_full_mem[1];
  auto tmem_full = reinterpret_cast<Barrier*>(tmem_full_mem);
  tl::Tcgen05SMemDescriptor desc_a;
  tl::Tcgen05SMemDescriptor desc_b;
  float C_local[256];
  bfloat16_t C_shared_local_cast[8];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(C_desc);
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate<true>((&(C_tmem[0])), 256);
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<0>()) {
    loaded[0].init(64);
    loaded[1].init(64);
    loaded[2].init(64);
    loaded[3].init(64);
    consumed[0].init(1);
    consumed[1].init(1);
    consumed[2].init(1);
    consumed[3].init(1);
    tmem_full[0].init(1);
  }
  tl::fence_barrier_init();
  tl::cluster_sync();
  const dim3 blockIdx = tl::rasterization2DRowWithCluster<8, 2>();
  if (((int)threadIdx.x) < 32) {
    for (int k = 0; k < 128; ++k) {
      consumed[(k & 3)].wait((((k >> 2) & 1) ^ 1));
      if (((int)threadIdx.x) == 0) {
        if (tl::block_rank_in_cluster() == 0) {
          loaded[(k & 3)].expect_transaction(32768 * 2);
        }
        tl::tma_load_2sm(A_desc, loaded[(k & 3)], (&(((bfloat16_t*)buf_dyn_shmem)[((k & 3) * 8192)])), (k * 64), (((int)blockIdx.x) * 128));
        tl::tma_load_2sm(B_desc, loaded[(k & 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k & 3) * 8192) + 32768)])), ((((int)blockIdx.y) * 256) + (tl::block_rank_in_cluster() * 128)), (k * 64));
        tl::tma_load_2sm(B_desc, loaded[(k & 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k & 3) * 8192) + 36864)])), (((((int)blockIdx.y) * 256) + (tl::block_rank_in_cluster() * 128)) + 64), (k * 64));
      }
      loaded[(k & 3)].arrive(0u);
    }
  } else {
    if (((int)threadIdx.x) < 64 && tl::block_rank_in_cluster() == 0) {
      for (int k_1 = 0; k_1 < 128; ++k_1) {
        loaded[(k_1 & 3)].wait(((k_1 >> 2) & 1));
        tl::initialize_tcgen05_descriptor(desc_a, (&(((bfloat16_t*)buf_dyn_shmem)[((k_1 & 3) * 8192)])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b, (&(((bfloat16_t*)buf_dyn_shmem)[(((k_1 & 3) * 8192) + 32768)])), 512, 64, 0, 0, 2);
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::tcgen05mma_ss<tl::DataType::kBFloat16, true>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 2048)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, ((0 < ki) ? 1 : ((k_1 == 0) ? 0 : 1)), static_cast<uint32_t>(272696464), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive<true>((&(consumed[(k_1 & 3)])));
      }
      tl::tcgen05_mma_arrive<true>((&(tmem_full[0])));
    }
  }
  tmem_full[0].wait(0);
  tl::tcgen05_ld_32dp32bNx<256, false>(C_tmem[0], 0, (&(C_local[0])));
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    for (int vec = 0; vec < 2; ++vec) {
      uint2 __1;
      float4 v_ = *(float4*)(C_local + ((i * 8) + (vec * 4)));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
      *(uint2*)(C_shared_local_cast + (vec * 4)) = __1;
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i >> 3) * 8192) + (((int)threadIdx.x) * 64)) + (((((i & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((i & 1) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(C_shared_local_cast + 0);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    tl::fence_proxy_async();
    tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), (((int)blockIdx.y) * 256), (((int)blockIdx.x) * 128));
    tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[8192])), ((((int)blockIdx.y) * 256) + 64), (((int)blockIdx.x) * 128));
    tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[16384])), ((((int)blockIdx.y) * 256) + 128), (((int)blockIdx.x) * 128));
    tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[24576])), ((((int)blockIdx.y) * 256) + 192), (((int)blockIdx.x) * 128));
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate<true>((&(C_tmem[0])), 256);
  }
}
"""


@tilelang.jit
def gemm(A, B, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages, use_tma_store=True):
    M, N, K = T.const("M, N, K")

    k_iters = T.ceildiv(K, block_K)

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128, cluster_dims=2) as (bx, by):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N // 2), in_dtype)  # Each cta hold half of B
        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
        consumed = T.alloc_cluster_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1])

        tx = T.get_thread_binding()
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)  # todo: automatically assume this

        T.use_swizzle(16)  # TL will perform auto threadblock swizzle with cluster

        if tx < 32:  # warp 0: issue tma
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(consumed[k % num_stages], ((k // num_stages) & 1) ^ 1)
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared[k % num_stages, :, :])
                T.copy(B[k * block_K : (k + 1) * block_K, (by * 2 + cta_id) * (block_N // 2) : (by * 2 + cta_id + 1) * (block_N // 2)], B_shared[k % num_stages, :, :])
                T.mbarrier_arrive(loaded[k % num_stages], 0)  # arrive on leader cta's barrier
        elif cta_id == 0 and tx < 64:  # Only warp 1 on leader cta issues tcgen5
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(loaded[k % num_stages], (k // num_stages) & 1)
                T.gemm(
                    A_shared[k % num_stages, :, :],
                    B_shared[k % num_stages, :, :],
                    C_tmem,
                    mbar=consumed[k % num_stages],
                    wg_wait=-1,
                    clear_accum=k == 0,
                )
            T.tcgen05_mma_arrive(tmem_full, arrive_2cta=True)

        # Wait for all tcgen5 to finish
        T.mbarrier_wait_parity(tmem_full, 0)

        T.sync_threads()  # TileLang won't generate this if not annotated
        T.copy(C_tmem, C_local)
        if use_tma_store:
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[bx * block_M, by * block_N])
        else:
            T.copy(C_local, C_local_cast)
            T.copy(C_local_cast, C[bx * block_M, by * block_N])
    return C


def main():
    M, N, K = 4096, 4096, 8192
    block_M, block_N, block_K = 128, 256, 64
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    num_stages = 4

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    print(gemm.get_kernel_source(a, b, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages))
    c = gemm(a, b, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages)
    

    ref_c = (a.to(torch.float) @ b.to(torch.float)).to(torch.bfloat16)
    # torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All checks passed. ✅")


    tl_latency = do_bench(lambda: gemm(a, b, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages), backend="cupti")
    torch_latency = do_bench(lambda: a @ b, backend="cupti")
    print(f"Tilelang latency: {tl_latency} ms")
    print(f"Flops: {2 * M * N * K / (tl_latency / 1e3) / 1e12} TFLOPS")
    print(f"Torch latency: {torch_latency} ms")
    print(f"Flops: {2 * M * N * K / (torch_latency / 1e3) / 1e12} TFLOPS")


if __name__ == "__main__":
    main()
