#include <tl_templates/cuda/instruction/mma.h>
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

extern "C" __global__ void main_kernel(const fp4_e2_t* __restrict__ A, const fp4_e2_t* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const fp4_e2_t* __restrict__ A, const fp4_e2_t* __restrict__ B, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  fp4_e2_2_t A_local_packed[32];
  fp4_e2_2_t B_local_packed[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(fp4_e2_32_t*)(((fp4_e2_t*)buf_dyn_shmem) + ((((i_1 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(fp4_e2_32_t*)(A + ((((((int)blockIdx.y) * 16384) + (i_1 * 4096)) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 16)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    *(fp4_e2_32_t*)(((fp4_e2_t*)buf_dyn_shmem) + (((((i_2 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384)) = *(fp4_e2_32_t*)(B + ((((((int)blockIdx.x) * 16384) + (i_2 * 4096)) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 16)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    *(fp4_e2_32_t*)(((fp4_e2_t*)buf_dyn_shmem) + (((((i_3 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192)) = *(fp4_e2_32_t*)(A + (((((((int)blockIdx.y) * 16384) + (i_3 * 4096)) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 16)) + 64));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    *(fp4_e2_32_t*)(((fp4_e2_t*)buf_dyn_shmem) + (((((i_4 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576)) = *(fp4_e2_32_t*)(B + (((((((int)blockIdx.x) * 16384) + (i_4 * 4096)) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 16)) + 64));
  }
  __syncthreads();
  for (int ki = 0; ki < 4; ++ki) {
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::ptx_ldmatrix_x4((&(((fp4_e2_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 8192) + (i_5 * 2048)) + (((((int)threadIdx.x) & 15) >> 3) * 1024)) + ((((((((int)threadIdx.x) & 15) * 128) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((int)threadIdx.x) & 31) >> 4) * 16)) & 1023)) / 2)])) + 0, A_local_packed + (i_5 * 16));
    }
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::ptx_ldmatrix_x4((&(((fp4_e2_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) >> 6) * 4096) + (i_6 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 16384)])) + 0, B_local_packed + (i_6 * 16));
    }
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      for (int j = 0; j < 4; ++j) {
        tl::mma_sync<tl::DataType::kFloat4_e2m1fn, tl::DataType::kFloat4_e2m1fn, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_local + ((i_7 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local_packed + (i_7 * 16)), reinterpret_cast<const unsigned*>(B_local_packed + (j * 16)));
        tl::mma_sync<tl::DataType::kFloat4_e2m1fn, tl::DataType::kFloat4_e2m1fn, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_local + (((i_7 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_packed + (i_7 * 16)), reinterpret_cast<const unsigned*>(B_local_packed + ((j * 16) + 8)));
      }
    }
  }
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      tl::ptx_ldmatrix_x4((&(((fp4_e2_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) & 63) >> 5) * 8192) + (i_8 * 2048)) + (((((int)threadIdx.x) & 15) >> 3) * 1024)) + ((((((((int)threadIdx.x) & 15) * 128) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 32)) + (((((int)threadIdx.x) & 31) >> 4) * 16)) & 1023)) + 16384) / 2)])) + 0, A_local_packed + (i_8 * 16));
    }
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      tl::ptx_ldmatrix_x4((&(((fp4_e2_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) >> 6) * 4096) + (i_9 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 24576)])) + 0, B_local_packed + (i_9 * 16));
    }
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      for (int j_1 = 0; j_1 < 4; ++j_1) {
        tl::mma_sync<tl::DataType::kFloat4_e2m1fn, tl::DataType::kFloat4_e2m1fn, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_local + ((i_10 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_packed + (i_10 * 16)), reinterpret_cast<const unsigned*>(B_local_packed + (j_1 * 16)));
        tl::mma_sync<tl::DataType::kFloat4_e2m1fn, tl::DataType::kFloat4_e2m1fn, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_local + (((i_10 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_packed + (i_10 * 16)), reinterpret_cast<const unsigned*>(B_local_packed + ((j_1 * 16) + 8)));
      }
    }
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 64; ++i_11) {
    *(float2*)(C + (((((((((((int)blockIdx.y) * 32768) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + ((i_11 >> 4) * 4096)) + ((i_11 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_11 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(C_local + (i_11 * 2));
  }
}
