#include <tl_templates/cuda/cuda_fp8.h>
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
__device__ __managed__ unsigned __tvm_global_barrier_state = 0;

extern "C" __global__ void cast_to_fp8_e4m3_kernel(float* __restrict__ X_1d, fp8_e4_t* __restrict__ X_fp8_1d, float* __restrict__ scale, float* __restrict__ scale_inv);
extern "C" __global__ void __launch_bounds__(512, 1) cast_to_fp8_e4m3_kernel(float* __restrict__ X_1d, fp8_e4_t* __restrict__ X_fp8_1d, float* __restrict__ scale, float* __restrict__ scale_inv) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ unsigned __barrier_expect;
  if (threadIdx.x == 0) {
    __barrier_expect = 0;
  }
  float shared_max[1];
  float y_local[8];
  float local_max[1];
  float scale_local[4];
  float global_max[1];
  float rescale_factor[1];
  #pragma unroll
  for (int i = 0; i < 1; ++i) {
    shared_max[0] = 0.000000e+00f;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(float4*)(((float*)buf_dyn_shmem) + (((i_1 * 2048) + (((int)threadIdx.x) * 4)) + 512)) = *(float4*)(X_1d + (((((int)blockIdx.x) * 4096) + (i_1 * 2048)) + (((int)threadIdx.x) * 4)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(float4*)(y_local + (i_2 * 4)) = *(float4*)(((float*)buf_dyn_shmem) + (((i_2 * 2048) + (((int)threadIdx.x) * 4)) + 512));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 1; ++i_3) {
    local_max[0] = 0.000000e+00f;
    #pragma unroll
    for (int rv = 0; rv < 8; ++rv) {
      local_max[0] = max(max(local_max[0], y_local[(((rv & 1) * 4) + (rv >> 1))]), (0.000000e+00f - min(local_max[0], y_local[(((rv & 1) * 4) + (rv >> 1))])));
    }
    local_max[0] = tl::AllReduce<tl::MaxOp, 512, 1>::run_hopper(local_max[0], (&(((float*)buf_dyn_shmem)[0])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 1; ++i_4) {
    shared_max[0] = max(shared_max[0], local_max[0]);
  }
  scale[((int)blockIdx.x)] = shared_max[0];
  __threadfence_system();
  if ((((int)threadIdx.x) == 0)) {
    atomicAdd(&__tvm_global_barrier_state, 1);
    volatile unsigned* pf = &__tvm_global_barrier_state;
    __barrier_expect += 132;
    while (pf[0] < __barrier_expect);
  }
  __syncthreads();
  float condval;
  if ((((int)threadIdx.x) < 132)) {
    condval = scale[((int)threadIdx.x)];
  } else {
    condval = 0.000000e+00f;
  }
  ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = condval;
  __syncthreads();
  *(float4*)(scale_local + 0) = *(float4*)(((float*)buf_dyn_shmem) + ((((int)threadIdx.x) & 127) * 4));
  #pragma unroll
  for (int i_5 = 0; i_5 < 1; ++i_5) {
    global_max[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      global_max[0] = max(global_max[0], scale_local[rv_1]);
    }
    global_max[0] = tl::AllReduce<tl::MaxOp, 128, 1>::run_hopper(global_max[0], (&(((float*)buf_dyn_shmem)[0])));
  }
  __threadfence_system();
  if ((((int)threadIdx.x) == 0)) {
    atomicAdd(&__tvm_global_barrier_state, 1);
    volatile unsigned* pf_1 = &__tvm_global_barrier_state;
    __barrier_expect += 132;
    while (pf_1[0] < __barrier_expect);
  }
  __syncthreads();
  scale[0] = global_max[0];
  scale_inv[0] = (global_max[0] * 2.232143e-03f);
  __threadfence_system();
  if ((((int)threadIdx.x) == 0)) {
    atomicAdd(&__tvm_global_barrier_state, 1);
    volatile unsigned* pf_2 = &__tvm_global_barrier_state;
    __barrier_expect += 132;
    while (pf_2[0] < __barrier_expect);
  }
  __syncthreads();
  rescale_factor[0] = (4.480000e+02f / scale[0]);
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    *(float4*)(((float*)buf_dyn_shmem) + (((i_6 * 2048) + (((int)threadIdx.x) * 4)) + 512)) = *(float4*)(X_1d + (((((int)blockIdx.x) * 4096) + (i_6 * 2048)) + (((int)threadIdx.x) * 4)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    *(float4*)(y_local + (i_7 * 4)) = *(float4*)(((float*)buf_dyn_shmem) + (((i_7 * 2048) + (((int)threadIdx.x) * 4)) + 512));
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 2; ++i_8) {
    fp8_e4_4_t __1;
    float4 __2;
      float4 v_ = *(float4*)(y_local + (i_8 * 4));
      float4 v__1 = make_float4(rescale_factor[0], rescale_factor[0], rescale_factor[0], rescale_factor[0]);
      __2.x = (v_.x*v__1.x);
      __2.y = (v_.y*v__1.y);
      __2.z = (v_.z*v__1.z);
      __2.w = (v_.w*v__1.w);
    __1.x = (fp8_e4_t)(__2.x);
    __1.y = (fp8_e4_t)(__2.y);
    __1.z = (fp8_e4_t)(__2.z);
    __1.w = (fp8_e4_t)(__2.w);
    *(fp8_e4_4_t*)(((fp8_e4_t*)buf_dyn_shmem) + (((i_8 * 2048) + (((int)threadIdx.x) * 4)) + 2048)) = __1;
  }
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    *(fp8_e4_16_t*)(X_fp8_1d + ((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 16))) = *(fp8_e4_16_t*)(((fp8_e4_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 16) + 2048));
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_cast_to_fp8_e4m3_kernel = cudaFuncSetAttribute(cast_to_fp8_e4m3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 18432);
    if (result_cast_to_fp8_e4m3_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 18432, cudaGetErrorString(result_cast_to_fp8_e4m3_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(float* __restrict__ X_1d, fp8_e4_t* __restrict__ X_fp8_1d, float* __restrict__ scale_inv, float* __restrict__ scale, cudaStream_t stream=cudaStreamDefault) {
	cast_to_fp8_e4m3_kernel<<<dim3(132, 1, 1), dim3(512, 1, 1), 18432, stream>>>(X_1d, X_fp8_1d, scale, scale_inv);
	TILELANG_CHECK_LAST_ERROR("cast_to_fp8_e4m3_kernel");

	return 0;
}

