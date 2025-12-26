#pragma once
#include <hip/hip_runtime.h>

#include "hip_fp8.h"

// // Base template declaration
// template <typename T> __device__ void debug_print_var(const char *msg, T var);

// // Specialization for signed char type
// template <>
// __device__ void debug_print_var<signed char>(const char *msg, signed char var) {
//   const char *safe_msg = msg;
//   int value = static_cast<int>(var);
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=signed "
//          "char value=%d\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, value);
// }

// // Specialization for unsigned char type
// template <>
// __device__ void debug_print_var<unsigned char>(const char *msg,
//                                                unsigned char var) {
//   const char *safe_msg = msg;
//   unsigned int value = static_cast<unsigned int>(var);
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
//          "dtype=unsigned char value=%u\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, value);
// }

// // Specialization for int type
// template <> __device__ void debug_print_var<int>(const char *msg, int var) {
//   const char *safe_msg = msg;
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=int "
//          "value=%d\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, var);
// }

// // Specialization for unsigned int type
// template <>
// __device__ void debug_print_var<unsigned int>(const char *msg,
//                                               unsigned int var) {
//   const char *safe_msg = msg;
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
//          "dtype=unsigned int value=%u\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, var);
// }

// // Specialization for unsigned short type
// template <>
// __device__ void debug_print_var<half_t>(const char *msg, half_t var) {
//   const char *safe_msg = msg;
//   float value = static_cast<float>(var);
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
//          "dtype=half_t value=%f\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, value);
// }

// // Specialization for float type
// template <> __device__ void debug_print_var<float>(const char *msg, float var) {
//   const char *safe_msg = msg;
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=float "
//          "value=%f\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, var);
// }

// // Specialization for double type
// template <>
// __device__ void debug_print_var<double>(const char *msg, double var) {
//   const char *safe_msg = msg;
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=double "
//          "value=%lf\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, var);
// }

// // Specialization for bool type
// template <> __device__ void debug_print_var<bool>(const char *msg, bool var) {
//   const char *safe_msg = msg;
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=bool "
//          "value=%s\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z,
//          var ? "true" : "false");
// }

// // Specialization for short type
// template <> __device__ void debug_print_var<short>(const char *msg, short var) {
//   const char *safe_msg = msg;
//   int value = static_cast<int>(var);
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=short "
//          "value=%d\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, value);
// }

// // Specialization for unsigned short type
// template <>
// __device__ void debug_print_var<unsigned short>(const char *msg,
//                                                 unsigned short var) {
//   const char *safe_msg = msg;
//   unsigned int value = static_cast<unsigned int>(var);
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
//          "dtype=unsigned short value=%u\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, value);
// }

// // Specialization for signed char type
// template <>
// __device__ void
// debug_print_buffer_value<signed char>(const char *msg, const char *buf_name,
//                                       int index, signed char var) {
//   const char *safe_msg = msg;
//   const char *safe_buf_name = buf_name;
//   int value = static_cast<int>(var);
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
//          "index=%d, dtype=signed char value=%d\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, safe_buf_name,
//          index, value);
// }

// // Specialization for unsigned char type
// template <>
// __device__ void
// debug_print_buffer_value<unsigned char>(const char *msg, const char *buf_name,
//                                         int index, unsigned char var) {
//   const char *safe_msg = msg;
//   const char *safe_buf_name = buf_name;
//   unsigned int value = static_cast<unsigned int>(var);
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
//          "index=%d, dtype=unsigned char value=%u\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, safe_buf_name,
//          index, value);
// }

// // Specialization for bool type
// template <>
// __device__ void debug_print_buffer_value<bool>(const char *msg,
//                                               const char *buf_name, int index,
//                                               bool var) {
//   const char *safe_msg = msg;
//   const char *safe_buf_name = buf_name;
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
//          "index=%d, dtype=bool value=%s\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, safe_buf_name,
//          index, var ? "true" : "false");
// }

// // Specialization for integer type
// template <>
// __device__ void debug_print_buffer_value<int>(const char *msg,
//                                               const char *buf_name, int index,
//                                               int var) {
//   const char *safe_msg = msg;
//   const char *safe_buf_name = buf_name;
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
//          "index=%d, dtype=int value=%d\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, safe_buf_name,
//          index, var);
// }

// // Specialization for float type
// template <>
// __device__ void debug_print_buffer_value<float>(const char *msg,
//                                                 const char *buf_name, int index,
//                                                 float var) {
//   const char *safe_msg = msg;
//   const char *safe_buf_name = buf_name;
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
//          "index=%d, dtype=float value=%f\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, safe_buf_name,
//          index, var);
// }

// // Specialization for half_t type
// template <>
// __device__ void debug_print_buffer_value<half_t>(const char *msg,
//                                                  const char *buf_name,
//                                                  int index, half_t var) {
//   const char *safe_msg = msg;
//   const char *safe_buf_name = buf_name;
//   float value = static_cast<float>(var);
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
//          "index=%d, dtype=half_t value=%f\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, safe_buf_name,
//          index, value);
// }

// // Specialization for double type
// template <>
// __device__ void debug_print_buffer_value<double>(const char *msg,
//                                                  const char *buf_name,
//                                                  int index, double var) {
//   const char *safe_msg = msg;
//   const char *safe_buf_name = buf_name;
//   printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
//          "index=%d, dtype=double value=%lf\n",
//          safe_msg, (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
//          (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z, safe_buf_name,
//          index, var);
// }

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
DEFINE_PRINT_TRAIT(unsigned long long, "ulong long", "%llu", unsigned long long);
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

