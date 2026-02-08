#pragma once
#include <hip/hip_runtime.h>

#include "hip_fp8.h"

template <typename T> struct PrintTraits {
  static __device__ void print_var(const char *msg, T val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
           "dtype=unknown value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, (const void *)&val);
  }

  static __device__ void print_buffer(const char *msg, const char *buf_name,
                                      int index, T val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
           "index=%d, dtype=unknown value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, buf_name, index, (const void *)&val);
  }
};

#define DEFINE_PRINT_TRAIT(TYPE, NAME, FORMAT, CAST_TYPE)                      \
  template <> struct PrintTraits<TYPE> {                                       \
    static __device__ void print_var(const char *msg, TYPE val) {              \
      printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "        \
             "dtype=" NAME " value=" FORMAT "\n",                              \
             msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,             \
             threadIdx.y, threadIdx.z, (CAST_TYPE)val);                        \
    }                                                                          \
    static __device__ void print_buffer(const char *msg, const char *buf_name, \
                                        int index, TYPE val) {                 \
      printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "        \
             "buffer=%s, index=%d, dtype=" NAME " value=" FORMAT "\n",         \
             msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,             \
             threadIdx.y, threadIdx.z, buf_name, index, (CAST_TYPE)val);       \
    }                                                                          \
  }

DEFINE_PRINT_TRAIT(char, "char", "%d", int);
DEFINE_PRINT_TRAIT(signed char, "signed char", "%d", int);
DEFINE_PRINT_TRAIT(unsigned char, "unsigned char", "%u", unsigned int);
DEFINE_PRINT_TRAIT(short, "short", "%d", int);
DEFINE_PRINT_TRAIT(unsigned short, "unsigned short", "%u", unsigned int);
DEFINE_PRINT_TRAIT(int, "int", "%d", int);
DEFINE_PRINT_TRAIT(unsigned int, "uint", "%u", unsigned int);
DEFINE_PRINT_TRAIT(long, "long", "%ld", long);
DEFINE_PRINT_TRAIT(unsigned long, "ulong", "%lu", unsigned long);
DEFINE_PRINT_TRAIT(long long, "long long", "%lld", long long);

DEFINE_PRINT_TRAIT(float, "float", "%f", float);
DEFINE_PRINT_TRAIT(double, "double", "%lf", double);
DEFINE_PRINT_TRAIT(half_t, "half_t", "%f", float);
DEFINE_PRINT_TRAIT(bfloat16_t, "bfloat16_t", "%f", float);

DEFINE_PRINT_TRAIT(fp8_e4_t, "fp8_e4_t", "%f", float);
DEFINE_PRINT_TRAIT(fp8_e5_t, "fp8_e5_t", "%f", float);

//
template <> struct PrintTraits<bool> {
  static __device__ void print_var(const char *msg, bool val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=bool "
           "value=%s\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, val ? "true" : "false");
  }
  static __device__ void print_buffer(const char *msg, const char *buf_name,
                                      int index, bool val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
           "index=%d, dtype=bool value=%s\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, buf_name, index, val ? "true" : "false");
  }
};

template <typename T> struct PrintTraits<T *> {
  static __device__ void print_var(const char *msg, T *val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
           "dtype=pointer value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, (void *)val);
  }
  static __device__ void print_buffer(const char *msg, const char *buf_name,
                                      int index, T *val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
           "index=%d, dtype=pointer value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, buf_name, index, (void *)val);
  }
};

template <typename T> __device__ void debug_print_var(const char *msg, T var) {
  PrintTraits<T>::print_var(msg, var);
}

// Template declaration for device-side debug printing (buffer only)
template <typename T>
__device__ void debug_print_buffer_value(const char *msg, const char *buf_name,
                                         int index, T var) {
  PrintTraits<T>::print_buffer(msg, buf_name, index, var);
}

// Specialization for msg-only debug print
__device__ void debug_print_msg(const char *msg) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d)\n", msg,
         blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z);
}
