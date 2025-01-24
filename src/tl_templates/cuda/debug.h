#pragma once

#include "common.h"
#include <stdio.h>

// Template declaration for device-side debug printing (variable only)
template <typename T>
__device__ void debug_print_var(T var);

// Specialization for integer type
template <>
__device__ void debug_print_var<int>(int var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=int value=%d\n", 
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, var);
}

// Specialization for float type
template <>
__device__ void debug_print_var<float>(float var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=float value=%f\n", 
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, var);
}

// Specialization for half type
template <>
__device__ void debug_print_var<half>(half var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=half value=%f\n", 
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, (float)var);
}

// Specialization for half_t type
template <>
__device__ void debug_print_var<half_t>(half_t var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=half_t value=%f\n", 
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, (float)var);
}

// Specialization for bfloat16_t type
template <>
__device__ void debug_print_var<bfloat16_t>(bfloat16_t var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=bfloat16_t value=%f\n", 
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, (float)var);
}

// Specialization for double type
template <>
__device__ void debug_print_var<double>(double var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=double value=%lf\n", 
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, var);
}


#pragma once

#include "common.h"
#include <stdio.h>

// Template declaration for device-side debug printing (buffer only)
template <typename T>
__device__ void debug_print_buffer_value(char* buf_name, int index, T var);

// Specialization for integer type
template <>
__device__ void debug_print_buffer_value<int>(char* buf_name, int index, int var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, index=%d, dtype=int value=%d\n",
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, 
           buf_name, index, var);
}

// Specialization for float type
template <>
__device__ void debug_print_buffer_value<float>(char* buf_name, int index, float var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, index=%d, dtype=float value=%f\n",
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, 
           buf_name, index, var);
}

// Specialization for half type
template <>
__device__ void debug_print_buffer_value<half>(char* buf_name, int index, half var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, index=%d, dtype=half value=%f\n",
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, 
           buf_name, index, (float)var);
}

// Specialization for half_t type
template <>
__device__ void debug_print_buffer_value<half_t>(char* buf_name, int index, half_t var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, index=%d, dtype=half_t value=%f\n",
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, 
           buf_name, index, (float)var);
}

// Specialization for bfloat16_t type
template <>
__device__ void debug_print_buffer_value<bfloat16_t>(char* buf_name, int index, bfloat16_t var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, index=%d, dtype=bfloat16_t value=%f\n",
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, 
           buf_name, index, (float)var);
}

// Specialization for double type
template <>
__device__ void debug_print_buffer_value<double>(char* buf_name, int index, double var) {
    printf("BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, index=%d, dtype=double value=%lf\n",
           blockIdx.x, blockIdx.y, blockIdx.z, 
           threadIdx.x, threadIdx.y, threadIdx.z, 
           buf_name, index, var);
}
