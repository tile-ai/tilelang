#pragma once

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

#include "atomic.h"
#include <cute/arch/util.hpp>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>
#include <math_constants.h>

using cutlass::bfloat16_t;
using cutlass::half_t;
using cutlass::tfloat32_t;

using cute::cast_smem_ptr_to_uint;

using int4_t = int4;

#define hexp cutlass::fast_exp
#define hlog cutlass::fast_log
#define hsqrt cutlass::fast_sqrt
#define hsin cutlass::fast_sin
#define hcos cutlass::fast_cos
#define htanh cutlass::fast_tanh
#define hpow powf

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE __forceinline__ __device__
#define TL_DEVICE_NOINLINE __noinline__ __device__
#define TL_PATCH

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    cudaError_t __err = (stmt);                                                \
    if (__err != cudaSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: %s - %s", __FILE__,          \
               __LINE__, cudaGetErrorName(__err), cudaGetErrorString(__err));  \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, kernel_name ": %s - %s",             \
               cudaGetErrorName(__err), cudaGetErrorString(__err));            \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// using cutlass abs function for half_t
TL_PATCH TL_DEVICE half_t __habs(const half_t x) { return abs(x); }

// using cutlass abs function for bfloat_t
TL_PATCH TL_DEVICE bfloat16_t __habs(const bfloat16_t x) { return abs(x); }

// hrsqrt function for half_t
TL_PATCH TL_DEVICE half_t hrsqrt(const half_t x) {
  return half_t(hrsqrt(x.to_half()));
}

// Pack two half values.
TL_DEVICE unsigned __pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_half2(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_nv_bfloat162(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack four char values
TL_DEVICE int make_int(signed char x0, signed char x1, signed char x2,
                       signed char x3) {
  return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
}

// Pack sixteen char values.
TL_DEVICE int4_t make_int4(signed char x0, signed char x1, signed char x2,
                           signed char x3, signed char y0, signed char y1,
                           signed char y2, signed char y3, signed char z0,
                           signed char z1, signed char z2, signed char z3,
                           signed char w0, signed char w1, signed char w2,
                           signed char w3) {
  int4_t result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  result.z = make_int(z0, z1, z2, z3);
  result.w = make_int(w0, w1, w2, w3);
  return result;
}

// Helper to cast SMEM pointer to unsigned
TL_DEVICE uint32_t smem_ptr_to_uint(void const *const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

/**
 * Convert a shared-memory pointer to a 32-bit unsigned integer address.
 *
 * Casts the given pointer (expected to reference shared memory) into a 32-bit
 * unsigned integer using the device address-space conversion required for
 * shared-memory pointers.
 *
 * @param smem_ptr Pointer into shared memory.
 * @return 32-bit unsigned integer representation of the shared-memory address.
 *
 * @note The pointer must refer to shared memory; behavior is undefined for
 *       pointers in other address spaces.
 */
TL_DEVICE unsigned int cast_smem_ptr_to_int(const void *const smem_ptr) {
  unsigned int smem_int;
  asm volatile("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; "
               "cvt.u32.u64 %0, smem_int; }"
               : "=r"(smem_int)
               : "l"(smem_ptr));
  return smem_int;
}

// DP4A
template <typename InDatatype, typename OutDatatype>
TL_DEVICE /**
           * Compute a 4×8-bit dot-product-accumulate using the CUDA DP4A
           * intrinsic.
           *
           * Reads 32-bit packed values from `a` and `b` (each containing four
           * signed 8-bit lanes), applies the __dp4a operation (dot product of
           * the four lane pairs added to an accumulator), and stores the 32-bit
           * integer result through `c`.
           *
           * @param a Pointer to a 32-bit packed input containing four signed
           * 8-bit elements.
           * @param b Pointer to a 32-bit packed input containing four signed
           * 8-bit elements.
           * @param c Pointer to a 32-bit accumulator; its current value is used
           * as the initial accumulator and overwritten with the resulting int32
           * sum.
           */
    void
    DP4A(InDatatype *a, InDatatype *b, OutDatatype *c) {
  const int a_int = *((int *)a);
  const int b_int = *((int *)b);
  const int c_int = *((int *)c);
  *c = __dp4a(a_int, b_int, c_int);
}

namespace tl {
/*!
 * \brief PTX data type.
 * \note
 * PTX fundamental data types:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fundamental-types
 * PTX matrix data types:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-data-types
 */
enum class DataType : int {
  kInt4 = 0,
  kUInt4 = 1,
  kInt8 = 2,
  kUInt8 = 3,
  kInt16 = 4,
  kUInt16 = 5,
  kInt32 = 6,
  kUInt32 = 7,
  kInt64 = 8,
  kUInt64 = 9,
  kFloat8_e4m3 = 10,
  kFloat8_e5m2 = 11,
  kFloat16 = 12,
  kBFloat16 = 13,
  kFloat16x2 = 14,
  kFloat32 = 15,
  kTensorFloat32 = 16,
  kFloat64 = 17,
  kBit1 = 18,
  kBit8 = 19,
  kBit16 = 20,
  kBit32 = 21,
  kBit64 = 22
};

union GmmaDescriptor {
  CUTE_HOST_DEVICE constexpr GmmaDescriptor() noexcept : desc_(0) {}
  CUTE_HOST_DEVICE constexpr GmmaDescriptor(uint64_t desc) noexcept
      : desc_(desc) {}
  CUTE_HOST_DEVICE constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept
      : desc_(t.desc_) {}
  CUTE_HOST_DEVICE constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept
      : desc_(t.desc_) {}

  CUTE_HOST_DEVICE constexpr GmmaDescriptor &
  operator=(GmmaDescriptor const &t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  CUTE_HOST_DEVICE constexpr GmmaDescriptor &
  operator=(GmmaDescriptor &&t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2
    // brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1, base_offset_ : 3,
        : 4; // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2; // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // Decay to a uint64_t
  CUTE_HOST_DEVICE constexpr operator uint64_t() const noexcept {
    return desc_;
  }
  template <typename T>
  CUTE_HOST_DEVICE constexpr GmmaDescriptor operator+(const T &offset) const {
    GmmaDescriptor ret;
    ret.reg32_[0] = reg32_[0] + uint32_t(offset);
    ret.reg32_[1] = reg32_[1];
    return ret;
  }
};

// Any
template <typename T> TL_DEVICE bool Any(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i]) {
      return true;
    }
  }
  return false;
}

// All
template <typename T> TL_DEVICE bool All(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (!a[i]) {
      return false;
    }
  }
  return true;
}

// Pow of int
template <int y = 1, typename T> TL_DEVICE T pow_of_int(T x) {
  T result = x;
  for (int i = 1; i < y; i++) {
    result *= x;
  }
  return result;
}

// Thread partial barrier synchronization
// https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model
template <int barrier_id = 0, int thread_count = 0>
TL_DEVICE void __sync_thread_partial() {
  asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(thread_count));
}

template <int layout_type = 0, int leading_byte_offset = 0,
          int stride_byte_offset = 0, typename T>
TL_DEVICE void initialize_descriptor(GmmaDescriptor &descriptor,
                                     T *start_address) {
  descriptor.bitfield.start_address_ =
      cute::cast_smem_ptr_to_uint(start_address) >> 4;
  descriptor.bitfield.layout_type_ = layout_type;
  descriptor.bitfield.base_offset_ = 0;
  descriptor.bitfield.leading_byte_offset_ = leading_byte_offset;
  descriptor.bitfield.stride_byte_offset_ = stride_byte_offset;
}

template <typename T>
TL_DEVICE void increase_descriptor_offset(GmmaDescriptor &descriptor,
                                          T offset) {
  descriptor.reg32_[0] += (offset >> 4);
}

} // namespace tl

namespace cutlass {
TL_DEVICE
bfloat16_t fast_exp(bfloat16_t x) { return ::hexp(x); }
} // namespace cutlass
