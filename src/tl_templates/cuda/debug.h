#pragma once

#include "common.h"
#include <stdio.h>

// Template declaration for device-side debug printing (variable only)
template <typename T> __device__ void debug_print_var(char *msg, T var);

// Specialization for integer type
template <> __device__ void debug_print_var<int>(char *msg, int var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=int "
         "value=%d\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, var);
}

// Specialization for float type
template <> __device__ void debug_print_var<float>(char *msg, float var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=float "
         "value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, var);
}

// Specialization for half type
template <> __device__ void debug_print_var<half>(char *msg, half var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=half "
         "value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, (float)var);
}

// Specialization for half_t type
template <> __device__ void debug_print_var<half_t>(char *msg, half_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=half_t "
         "value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, (float)var);
}

// Specialization for bfloat16_t type
template <>
__device__ void debug_print_var<bfloat16_t>(char *msg, bfloat16_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
         "dtype=bfloat16_t value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, (float)var);
}

// Specialization for double type
template <> __device__ void debug_print_var<double>(char *msg, double var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=double "
         "value=%lf\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, var);
}

#pragma once

#include "common.h"
#include <stdio.h>

// Template declaration for device-side debug printing (buffer only)
template <typename T>
__device__ void debug_print_buffer_value(char *msg, char *buf_name, int index,
                                         T var);

// Specialization for integer type
template <>
__device__ void debug_print_buffer_value<int>(char *msg, char *buf_name,
                                              int index, int var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=int value=%d\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, var);
}

// Specialization for float type
template <>
__device__ void debug_print_buffer_value<float>(char *msg, char *buf_name,
                                                int index, float var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=float value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, var);
}

// Specialization for half type
template <>
__device__ void debug_print_buffer_value<half>(char *msg, char *buf_name,
                                               int index, half var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=half value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, (float)var);
}

// Specialization for half_t type
template <>
__device__ void debug_print_buffer_value<half_t>(char *msg, char *buf_name,
                                                 int index, half_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=half_t value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, (float)var);
}

// Specialization for bfloat16_t type
template <>
__device__ void debug_print_buffer_value<bfloat16_t>(char *msg, char *buf_name,
                                                     int index,
                                                     bfloat16_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=bfloat16_t value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, (float)var);
}

// Specialization for double type
template <>
__device__ void debug_print_buffer_value<double>(char *msg, char *buf_name,
                                                 int index, double var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=double value=%lf\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, var);
}
