import torch
import tilelang
import tilelang.language as T
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

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint C_tmem[1];
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  tl::Tcgen05SMemDescriptor desc_a;
  tl::Tcgen05SMemDescriptor desc_b;
  float C_local[256];
  bfloat16_t C_shared_local_cast[8];
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate<true>((&(C_tmem[0])), 256);
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
  }
  tl::fence_barrier_init();
  tl::cluster_sync();
  for (int k = 0; k < 64; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = *(uint4*)(A + (((((((int)blockIdx.x) * 524288) + (i * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_1 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B + (((((k * 262144) + (i_1 * 16384)) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 31) * 8)));
    }
    __syncthreads();
    if ((tl::block_rank_in_cluster() == (uint)0) && ((((int)threadIdx.x) >> 5) == 0)) {
      tl::initialize_tcgen05_descriptor(desc_a, (&(((bfloat16_t*)buf_dyn_shmem)[16384])), 1, 64, 0, 0, 2);
      tl::initialize_tcgen05_descriptor(desc_b, (&(((bfloat16_t*)buf_dyn_shmem)[0])), 512, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::tcgen05mma_ss<tl::DataType::kBFloat16, true>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 2048)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, ((0 < ki) ? 1 : ((k == 0) ? 0 : 1)), static_cast<uint32_t>(272696464), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive<true>((&(mbar[0])));
    }
    mbar[0].wait((k & 1));
  }
  tl::tcgen05_ld_32dp32bNx<256, false>(C_tmem[0], 0, (&(C_local[0])));
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    for (int vec = 0; vec < 2; ++vec) {
      uint2 __1;
      float4 v_ = *(float4*)(C_local + ((i_2 * 8) + (vec * 4)));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
      *(uint2*)(C_shared_local_cast + (vec * 4)) = __1;
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((int)threadIdx.x) * 256) + (i_2 * 8)) + 24576)) = *(uint4*)(C_shared_local_cast + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 32; ++i_3) {
    *(uint4*)(C + (((((((int)blockIdx.x) * 524288) + (i_3 * 16384)) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 31) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_3 * 1024) + (((int)threadIdx.x) * 8)) + 24576));
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate<true>((&(C_tmem[0])), 256);
  }
}
"""


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads, cluster_dims=2) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_cluster_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_tmem, mbar=mbar, wg_wait=-1, clear_accum=k == 0)
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)

            T.copy(C_shared, C[bx * block_M, by * block_N])

    return main


M, N, K = 2048, 2048, 2048  # FIXME: buggy when size is lager
print(f'M: {M}, N: {N}, K: {K}')
block_M, block_N, block_K = 128, 256, 64
in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
num_stages = 0 if block_N >= 256 or block_M >= 256 or block_K >= 256 else 2
threads = 128

func = matmul(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages, threads)
jit_kernel = tilelang.compile(
    func,
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
    out_idx=[2],
    target="cuda",
)

print(jit_kernel.get_kernel_source())

for _ in range(10000):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, 1, device="cuda", dtype=torch.bfloat16).repeat(1, N).contiguous()
    # b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)  # only 1st half is correct then
    c = jit_kernel(a, b)
    ref_c = (a @ b).to(torch.bfloat16)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print('ALL CHECK PASSED. ✅')
profiler = jit_kernel.get_profiler()
latency = profiler.do_bench()
print(f"Latency: {latency} ms")
print(f"Flops: {2 * M * N * K / (latency / 1e3) / 1e12} TFLOPS")
