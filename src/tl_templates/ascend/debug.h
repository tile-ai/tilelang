#pragma once

#include <cstdint>

#include "kernel_operator.h"

// ---------------------------------------------------------------------------
// Ascend debug-print runtime.
// Defined in both __asc_aicore and __asc_simt_vf namespaces so that
// unqualified calls work from both __global__ __vector__ (aicore) and
// __simt_vf__ contexts – same mechanism as AscendC::printf.
// ---------------------------------------------------------------------------

#define TL_DEFINE_ASCEND_DEBUG(ATTR)                                           \
                                                                               \
  template <typename T> struct PrintTraits {                                   \
    static ATTR inline void print_var(__gm__ const char *msg, T val) {         \
      printf("msg='%s' BlockIdx=%d: dtype=unknown value=%p\n", msg,            \
             blockIdx.x, reinterpret_cast<const void *>(&val));                \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, T val) {                   \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=unknown "       \
             "value=%p\n",                                                     \
             msg, blockIdx.x, buf_name, index,                                 \
             reinterpret_cast<const void *>(&val));                            \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <> struct PrintTraits<char> {                                       \
    static ATTR inline void print_var(__gm__ const char *msg, char val) {      \
      printf("msg='%s' BlockIdx=%d: dtype=char value=%d\n", msg, blockIdx.x,   \
             (int)val);                                                        \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, char val) {                \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=char "          \
             "value=%d\n",                                                     \
             msg, blockIdx.x, buf_name, index, (int)val);                      \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<signed char> {                                \
    static ATTR inline void print_var(__gm__ const char *msg,                  \
                                      signed char val) {                       \
      printf("msg='%s' BlockIdx=%d: dtype=signed char value=%d\n", msg,        \
             blockIdx.x, (int)val);                                            \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, signed char val) {         \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=signed char "   \
             "value=%d\n",                                                     \
             msg, blockIdx.x, buf_name, index, (int)val);                      \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<unsigned char> {                              \
    static ATTR inline void print_var(__gm__ const char *msg,                  \
                                      unsigned char val) {                     \
      printf("msg='%s' BlockIdx=%d: dtype=unsigned char value=%u\n", msg,      \
             blockIdx.x, (unsigned int)val);                                   \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, unsigned char val) {       \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, "                     \
             "dtype=unsigned char value=%u\n",                                 \
             msg, blockIdx.x, buf_name, index, (unsigned int)val);             \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<short> {                                      \
    static ATTR inline void print_var(__gm__ const char *msg, short val) {     \
      printf("msg='%s' BlockIdx=%d: dtype=short value=%d\n", msg, blockIdx.x,  \
             (int)val);                                                        \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, short val) {               \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=short "         \
             "value=%d\n",                                                     \
             msg, blockIdx.x, buf_name, index, (int)val);                      \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<unsigned short> {                             \
    static ATTR inline void print_var(__gm__ const char *msg,                  \
                                      unsigned short val) {                    \
      printf("msg='%s' BlockIdx=%d: dtype=unsigned short value=%u\n", msg,     \
             blockIdx.x, (unsigned int)val);                                   \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, unsigned short val) {      \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, "                     \
             "dtype=unsigned short value=%u\n",                                \
             msg, blockIdx.x, buf_name, index, (unsigned int)val);             \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<int> {                                        \
    static ATTR inline void print_var(__gm__ const char *msg, int val) {       \
      printf("msg='%s' BlockIdx=%d: dtype=int value=%d\n", msg, blockIdx.x,    \
             val);                                                             \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, int val) {                 \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=int "           \
             "value=%d\n",                                                     \
             msg, blockIdx.x, buf_name, index, val);                           \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<unsigned int> {                               \
    static ATTR inline void print_var(__gm__ const char *msg,                  \
                                      unsigned int val) {                      \
      printf("msg='%s' BlockIdx=%d: dtype=uint value=%u\n", msg, blockIdx.x,   \
             val);                                                             \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, unsigned int val) {        \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=uint "          \
             "value=%u\n",                                                     \
             msg, blockIdx.x, buf_name, index, val);                           \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<long> {                                       \
    static ATTR inline void print_var(__gm__ const char *msg, long val) {      \
      printf("msg='%s' BlockIdx=%d: dtype=long value=%ld\n", msg, blockIdx.x,  \
             val);                                                             \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, long val) {                \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=long "          \
             "value=%ld\n",                                                    \
             msg, blockIdx.x, buf_name, index, val);                           \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<unsigned long> {                              \
    static ATTR inline void print_var(__gm__ const char *msg,                  \
                                      unsigned long val) {                     \
      printf("msg='%s' BlockIdx=%d: dtype=ulong value=%lu\n", msg, blockIdx.x, \
             val);                                                             \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, unsigned long val) {       \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=ulong "         \
             "value=%lu\n",                                                    \
             msg, blockIdx.x, buf_name, index, val);                           \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<long long> {                                  \
    static ATTR inline void print_var(__gm__ const char *msg, long long val) { \
      printf("msg='%s' BlockIdx=%d: dtype=long long value=%lld\n", msg,        \
             blockIdx.x, val);                                                 \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, long long val) {           \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=long long "     \
             "value=%lld\n",                                                   \
             msg, blockIdx.x, buf_name, index, val);                           \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<unsigned long long> {                         \
    static ATTR inline void print_var(__gm__ const char *msg,                  \
                                      unsigned long long val) {                \
      printf("msg='%s' BlockIdx=%d: dtype=unsigned long long value=%llu\n",    \
             msg, blockIdx.x, val);                                            \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, unsigned long long val) {  \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, "                     \
             "dtype=unsigned long long value=%llu\n",                          \
             msg, blockIdx.x, buf_name, index, val);                           \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<float> {                                      \
    static ATTR inline void print_var(__gm__ const char *msg, float val) {     \
      printf("msg='%s' BlockIdx=%d: dtype=float32 value=%f\n", msg,            \
             blockIdx.x, val);                                                 \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, float val) {               \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=float32 "       \
             "value=%f\n",                                                     \
             msg, blockIdx.x, buf_name, index, val);                           \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<double> {                                     \
    static ATTR inline void print_var(__gm__ const char *msg, double val) {    \
      printf("msg='%s' BlockIdx=%d: dtype=float64 value=%lf\n", msg,           \
             blockIdx.x, val);                                                 \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, double val) {              \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=float64 "       \
             "value=%lf\n",                                                    \
             msg, blockIdx.x, buf_name, index, val);                           \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<half> {                                       \
    static ATTR inline void print_var(__gm__ const char *msg, half val) {      \
      printf("msg='%s' BlockIdx=%d: dtype=float16 value=%f\n", msg,            \
             blockIdx.x, (float)val);                                          \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, half val) {                \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=float16 "       \
             "value=%f\n",                                                     \
             msg, blockIdx.x, buf_name, index, (float)val);                    \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<bfloat16_t> {                                 \
    static ATTR inline void print_var(__gm__ const char *msg,                  \
                                      bfloat16_t val) {                        \
      printf("msg='%s' BlockIdx=%d: dtype=bfloat16 value=%f\n", msg,           \
             blockIdx.x, (float)val);                                          \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, bfloat16_t val) {          \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=bfloat16 "      \
             "value=%f\n",                                                     \
             msg, blockIdx.x, buf_name, index, (float)val);                    \
    }                                                                          \
  };                                                                           \
  template <> struct PrintTraits<bool> {                                       \
    static ATTR inline void print_var(__gm__ const char *msg, bool val) {      \
      printf("msg='%s' BlockIdx=%d: dtype=bool value=%s\n", msg, blockIdx.x,   \
             val ? "true" : "false");                                          \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, bool val) {                \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=bool "          \
             "value=%s\n",                                                     \
             msg, blockIdx.x, buf_name, index, val ? "true" : "false");        \
    }                                                                          \
  };                                                                           \
  template <typename T> struct PrintTraits<T *> {                              \
    static ATTR inline void print_var(__gm__ const char *msg, T *val) {        \
      printf("msg='%s' BlockIdx=%d: dtype=pointer value=%p\n", msg,            \
             blockIdx.x, reinterpret_cast<void *>(val));                       \
    }                                                                          \
    static ATTR inline void print_buffer(__gm__ const char *msg,               \
                                         __gm__ const char *buf_name,          \
                                         int index, T *val) {                  \
      printf("msg='%s' BlockIdx=%d: buffer=%s, index=%d, dtype=pointer "       \
             "value=%p\n",                                                     \
             msg, blockIdx.x, buf_name, index, reinterpret_cast<void *>(val)); \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename T>                                                        \
  ATTR inline void debug_print_var(__gm__ const char *msg, T var) {            \
    PrintTraits<T>::print_var(msg, var);                                       \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  ATTR inline void debug_print_buffer_value(                                   \
      __gm__ const char *msg, __gm__ const char *buf_name, int index, T var) { \
    PrintTraits<T>::print_buffer(msg, buf_name, index, var);                   \
  }                                                                            \
                                                                               \
  ATTR inline void debug_print_msg(__gm__ const char *msg) {                   \
    printf("msg='%s' BlockIdx=%d\n", msg, blockIdx.x);                         \
  }                                                                            \
                                                                               \
  ATTR inline void device_assert(bool cond) { assert(cond); }                  \
                                                                               \
  ATTR inline void device_assert_with_msg(bool cond, __gm__ const char *msg) { \
    if (!cond) {                                                               \
      printf("Device assert failed: %s BlockIdx=%d\n", msg, blockIdx.x);       \
      assert(false);                                                           \
    }                                                                          \
  }

namespace __asc_aicore {
TL_DEFINE_ASCEND_DEBUG(__aicore__)
} // namespace __asc_aicore

namespace __asc_simt_vf {
TL_DEFINE_ASCEND_DEBUG(__simt_callee__)
} // namespace __asc_simt_vf

#undef TL_DEFINE_ASCEND_DEBUG
