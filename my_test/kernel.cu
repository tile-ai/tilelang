#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void __launch_bounds__(64) main_kernel(unsigned char* __restrict__ A, unsigned char* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[4];
  unsigned char A_local[4];
  unsigned char B_local[4];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  *(float4*)(C_local + 0) = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  for (int ko = 0; ko < 4; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      ((unsigned char*)buf_dyn_shmem)[((((i * 64) + ((((int)threadIdx.x) >> 5) * 32)) + ((((((int)threadIdx.x) & 31) >> 3) ^ (i >> 1)) * 8)) + (((int)threadIdx.x) & 7))] = A[(((((((int)blockIdx.y) * 2048) + (i * 256)) + ((((int)threadIdx.x) >> 5) * 128)) + (ko * 32)) + (((int)threadIdx.x) & 31))];
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      ((unsigned char*)buf_dyn_shmem)[(((((i_1 * 64) + ((((int)threadIdx.x) >> 5) * 32)) + ((((((int)threadIdx.x) & 31) >> 3) ^ (i_1 >> 1)) * 8)) + (((int)threadIdx.x) & 7)) + 512)] = B[(((((((int)blockIdx.x) * 2048) + (i_1 * 256)) + ((((int)threadIdx.x) >> 5) * 128)) + (ko * 32)) + (((int)threadIdx.x) & 31))];
    }
    __syncthreads();
    for (int ki = 0; ki < 2; ++ki) {
      for (int local_id = 0; local_id < 4; ++local_id) {
        A_local[local_id] = ((unsigned char*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 32) + ((((ki * 2) + (((int)threadIdx.x) >> 5)) ^ ((((int)threadIdx.x) & 15) >> 2)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + local_id)];
      }
      for (int local_id_1 = 0; local_id_1 < 4; ++local_id_1) {
        B_local[local_id_1] = ((unsigned char*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 15) * 32) + ((((ki * 2) + (((int)threadIdx.x) >> 5)) ^ ((((int)threadIdx.x) & 15) >> 2)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + local_id_1) + 512)];
      }
      {
    *(((float32x4*)C_local) + 0) = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(*(((__hip_fp8_e4m3_fnuz*)B_local) + 0),
                  *(((__hip_fp8_e4m3_fnuz*)A_local) + 0),
                  *(((float32x4*)C_local) + 0), 0, 0, 0);
  }/*e4m3_float8x4 e4m3_float8x4 float32x4*/;
    }
  }
  for (int local_id_2 = 0; local_id_2 < 4; ++local_id_2) {
    C[(((((((int)blockIdx.y) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) >> 4) * 4)) + local_id_2)] = ((half_t)C_local[local_id_2]);
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    if (1024 > 65536) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size for main_kernel to %d", 1024);
        return -1;
    }
    return 0;

    return 0;
}

extern "C" int call(unsigned char* __restrict__ A, unsigned char* __restrict__ B, half_t* __restrict__ C, hipStream_t stream=hipStreamDefault) {
	main_kernel<<<dim3(8, 8, 1), dim3(64, 1, 1), 1024, stream>>>(A, B, C);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
