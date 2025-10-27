#pragma once

#include "./cuda_fp8.h"
#include "common.h"

#ifndef __CUDACC_RTC__
#include <cstdio>
#endif

// Template declaration for device-side debug printing (variable only)
template <typename T> __device__ void debug_print_var(const char *msg, T var);

// Overload for pointer type (supports any cv-qualified T*)
template <typename T> __device__ void debug_print_var(const char *msg, T *var) {
  printf(
      "msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=pointer "
      "value=%p\n",
      msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
      threadIdx.z, var);
}

// Specialization for signed char type
template <>
__device__ void debug_print_var<signed char>(const char *msg, signed char var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=signed "
         "char "
         "value=%d\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, var);
}

// Specialization for unsigned char type
template <>
__device__ void debug_print_var<unsigned char>(const char *msg,
                                               unsigned char var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
         "dtype=unsigned char "
         "value=%d\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, var);
}

// Specialization for integer type
template <> __device__ void debug_print_var<int>(const char *msg, int var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=int "
         "value=%d\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, var);
}

// Specialization for unsigned integer type
template <>
__device__ void debug_print_var<unsigned int>(const char *msg,
                                              unsigned int var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=int "
         "value=%u\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, var);
}

// Specialization for float type
template <> __device__ void debug_print_var<float>(const char *msg, float var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=float "
         "value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, var);
}

// Specialization for half type
template <> __device__ void debug_print_var<half>(const char *msg, half var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=half "
         "value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, (float)var);
}

// Specialization for half_t type
template <>
__device__ void debug_print_var<half_t>(const char *msg, half_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=half_t "
         "value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, (float)var);
}

// Specialization for bfloat16_t type
template <>
__device__ void debug_print_var<bfloat16_t>(const char *msg, bfloat16_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
         "dtype=bfloat16_t value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, (float)var);
}

// Specialization for double type
template <>
__device__ void debug_print_var<double>(const char *msg, double var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=double "
         "value=%lf\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, var);
}

// Specialization for fp8_e4_t type
template <>
__device__ void debug_print_var<fp8_e4_t>(const char *msg, fp8_e4_t var) {
  printf(
      "msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=fp8_e4_t "
      "value=%f\n",
      msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
      threadIdx.z, (float)var);
}

// Specialization for fp8_e5_t type
template <>
__device__ void debug_print_var<fp8_e5_t>(const char *msg, fp8_e5_t var) {
  printf(
      "msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=fp8_e5_t "
      "value=%f\n",
      msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
      threadIdx.z, (float)var);
}

// Template declaration for device-side debug printing (buffer only)
template <typename T>
__device__ void debug_print_buffer_value(const char *msg, const char *buf_name,
                                         int index, T var);

// Specialization for signed char type
template <>
__device__ void
debug_print_buffer_value<signed char>(const char *msg, const char *buf_name,
                                      int index, signed char var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=signed char value=%d\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, var);
}

// Specialization for unsigned char type
template <>
__device__ void
debug_print_buffer_value<unsigned char>(const char *msg, const char *buf_name,
                                        int index, unsigned char var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=char value=%d\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, var);
}

// Specialization for integer type
template <>
__device__ void debug_print_buffer_value<int>(const char *msg,
                                              const char *buf_name, int index,
                                              int var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=int value=%d\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, var);
}

// Specialization for unsigned integer type
template <>
__device__ void
debug_print_buffer_value<unsigned int>(const char *msg, const char *buf_name,
                                       int index, unsigned int var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=int value=%u\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, var);
}

// Specialization for float type
template <>
__device__ void debug_print_buffer_value<float>(const char *msg,
                                                const char *buf_name, int index,
                                                float var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=float value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, var);
}

// Specialization for half type
template <>
__device__ void debug_print_buffer_value<half>(const char *msg,
                                               const char *buf_name, int index,
                                               half var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=half value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, (float)var);
}

// Specialization for half_t type
template <>
__device__ void debug_print_buffer_value<half_t>(const char *msg,
                                                 const char *buf_name,
                                                 int index, half_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=half_t value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, (float)var);
}

// Specialization for bfloat16_t type
template <>
__device__ void
debug_print_buffer_value<bfloat16_t>(const char *msg, const char *buf_name,
                                     int index, bfloat16_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=bfloat16_t value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, (float)var);
}

// Specialization for double type
template <>
__device__ void debug_print_buffer_value<double>(const char *msg,
                                                 const char *buf_name,
                                                 int index, double var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=double value=%lf\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, var);
}

// Specialization for fp8_e4_t type
template <>
__device__ void debug_print_buffer_value<fp8_e4_t>(const char *msg,
                                                   const char *buf_name,
                                                   int index, fp8_e4_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=fp8_e4_t value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, (float)var);
}

// Specialization for fp8_e5_t type
template <>
__device__ void debug_print_buffer_value<fp8_e5_t>(const char *msg,
                                                   const char *buf_name,
                                                   int index, fp8_e5_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=fp8_e5_t value=%f\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, (float)var);
}

// Specialization for int16 type
template <>
__device__ void debug_print_buffer_value<int16_t>(const char *msg,
                                                  const char *buf_name,
                                                  int index, int16_t var) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
         "index=%d, dtype=int16_t value=%d\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, (int32_t)var);
}

TL_DEVICE void device_assert(bool cond) { assert(cond); }

TL_DEVICE void device_assert_with_msg(bool cond, const char *msg) {
  if (!cond) {
    printf("Device assert failed: %s\n", msg);
    assert(0);
  }
}
