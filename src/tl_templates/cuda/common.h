#pragma once

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

#include "atomic.h"
#include <cute/arch/util.hpp>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>
#include <math_constants.h>

#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

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

// Pack four char values.
TL_DEVICE int make_int(signed char x0, signed char x1, signed char x2,
                       signed char x3) {
  return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
}

// Pack eight char values.
TL_DEVICE int2 make_int2(signed char x0, signed char x1, signed char x2,
                         signed char x3, signed char y0, signed char y1,
                         signed char y2, signed char y3) {
  int2 result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  return result;
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

TL_DEVICE int4_t make_int4(short x0, short x1, short y0, short y1, short z0,
                           short z1, short w0, short w1) {
  int4_t result;
  *((short2 *)&result.x) = make_short2(x0, x1);
  *((short2 *)&result.y) = make_short2(y0, y1);
  *((short2 *)&result.z) = make_short2(z0, z1);
  *((short2 *)&result.w) = make_short2(w0, w1);
  return result;
}

// Pack eight int values.
TL_DEVICE longlong4 make_longlong4(int x0, int x1, int y0, int y1, int z0,
                                   int z1, int w0, int w1) {
  longlong4 result;
  *((int2 *)&result.x) = make_int2(x0, x1);
  *((int2 *)&result.y) = make_int2(y0, y1);
  *((int2 *)&result.z) = make_int2(z0, z1);
  *((int2 *)&result.w) = make_int2(w0, w1);
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

union Tcgen05SMemDescriptor {
  CUTE_HOST_DEVICE constexpr Tcgen05SMemDescriptor() noexcept : desc_(0) {}
  CUTE_HOST_DEVICE constexpr Tcgen05SMemDescriptor(uint64_t desc) noexcept
      : desc_(desc) {}
  CUTE_HOST_DEVICE constexpr Tcgen05SMemDescriptor(
      Tcgen05SMemDescriptor const &t) noexcept
      : desc_(t.desc_) {}
  CUTE_HOST_DEVICE constexpr Tcgen05SMemDescriptor(
      Tcgen05SMemDescriptor &&t) noexcept
      : desc_(t.desc_) {}

  CUTE_HOST_DEVICE constexpr Tcgen05SMemDescriptor &
  operator=(Tcgen05SMemDescriptor const &t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  CUTE_HOST_DEVICE constexpr Tcgen05SMemDescriptor &
  operator=(Tcgen05SMemDescriptor &&t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    uint16_t stride_byte_offset_ : 14,
        version_ : 2; // 14 bits [0,14), 2 bits [14,16)
    // base_offset, bit [49,52). leading_byte_offset_mode, bit [52,53).
    uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1,
        : 3; // 1 bit unused, 3 bits [1,4), 1 bit [4,5), 3 bits unused
    // layout type, bit [61,64), SWIZZLE_NONE matrix descriptor = 0,
    // SWIZZLE_128B matrix descriptor = 2, SWIZZLE_64B descriptor = 4,
    // SWIZZLE_32B descriptor = 6, SWIZZLE_128B_BASE32B = 1, N/A = 3, N/A = 5,
    // N/A = 7
    uint8_t : 5, layout_type_ : 3; // 6 bits unused, 3 bits [5,8)
  } bitfield;
  // Separate the field, as we may only update one part of desc
  struct {
    uint32_t lo;
    uint32_t hi;
  } words;

  CUTE_HOST_DEVICE constexpr operator uint64_t() const noexcept {
    return desc_;
  }
  template <typename T>
  CUTE_HOST_DEVICE constexpr Tcgen05SMemDescriptor
  operator+(const T &offset) const {
    Tcgen05SMemDescriptor ret;
    // Address addition is in units of 16 bytes (4 LSB not encoded)
    ret.reg32_[0] = reg32_[0] + (uint32_t(offset) >> 4);
    ret.reg32_[1] = reg32_[1];
    return ret;
  }
};

//
// Tcgen05 instruction descriptor (wraps cute::UMMA::InstrDescriptor layout)
//
union Tcgen05InstrDescriptor {
  CUTE_HOST_DEVICE constexpr Tcgen05InstrDescriptor() noexcept : desc_(0) {}
  CUTE_HOST_DEVICE constexpr Tcgen05InstrDescriptor(uint32_t desc) noexcept
      : desc_(desc) {}
  CUTE_HOST_DEVICE constexpr Tcgen05InstrDescriptor(
      Tcgen05InstrDescriptor const &t) noexcept
      : desc_(t.desc_) {}
  CUTE_HOST_DEVICE constexpr Tcgen05InstrDescriptor(
      Tcgen05InstrDescriptor &&t) noexcept
      : desc_(t.desc_) {}

  CUTE_HOST_DEVICE constexpr Tcgen05InstrDescriptor &
  operator=(Tcgen05InstrDescriptor const &t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  CUTE_HOST_DEVICE constexpr Tcgen05InstrDescriptor &
  operator=(Tcgen05InstrDescriptor &&t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint32_t desc_;
  uint16_t reg16_[2];

  // Bitfield implementation mirrors cute::UMMA::InstrDescriptor
  struct {
    // bit [ 0, 2) : Sparse meta data id2
    uint16_t sparse_id2_ : 2,
        // bit [ 2, 3) : 0 = dense. 1 = sparse. Only valid for
        // F32F16/S8/MXF8F6F4
        sparse_flag_ : 1,
        // bit [ 3, 4) : 0 = no saturate. 1 = saturate. Only valid for S8
        saturate_ : 1,
        // bit [ 4, 6) : 0 = F16. 1 = F32, 2 = S32
        c_format_ : 2,
        // padding
        : 1,
        // bit [ 7,10) : see UMMA format encoding
        a_format_ : 3,
        // bit [10,13) : see UMMA format encoding
        b_format_ : 3,
        // bit [13,14) : 0 = no negate. 1 = negate
        a_negate_ : 1,
        // bit [14,15) : 0 = no negate. 1 = negate
        b_negate_ : 1,
        // bit [15,16) : 0 = K-major. 1 = MN-major
        a_major_ : 1;

    // Upper 16 bits
    uint16_t b_major_ : 1, // bit [16,17)
        n_dim_ : 6,        // bit [17,23) : 3 LSBs not included
        : 1,               // padding
        m_dim_ : 5,        // bit [24,29) : 4 LSBs not included
        : 1,               // padding
        max_shift_ : 2;    // bit [30,32)
  } bitfield;

  // Decay to a uint32_t
  CUTE_HOST_DEVICE constexpr explicit operator uint32_t() const noexcept {
    return desc_;
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
TL_DEVICE void initialize_wgmma_descriptor(GmmaDescriptor &descriptor,
                                           T *start_address) {
  descriptor.bitfield.start_address_ =
      cute::cast_smem_ptr_to_uint(start_address) >> 4;
  descriptor.bitfield.layout_type_ = layout_type;
  descriptor.bitfield.base_offset_ = 0;
  descriptor.bitfield.leading_byte_offset_ = leading_byte_offset;
  descriptor.bitfield.stride_byte_offset_ = stride_byte_offset;
}

template <typename T>
TL_DEVICE void
initialize_tcgen05_descriptor(Tcgen05SMemDescriptor &descriptor,
                              T *start_address, int leading_byte_offset,
                              int stride_byte_offset, int base_offset,
                              bool leading_is_absolute, int swizzle_mode) {

  descriptor.bitfield.start_address_ =
      static_cast<uint16_t>(cast_smem_ptr_to_uint(start_address) >> 4);
  descriptor.bitfield.leading_byte_offset_ = leading_byte_offset;
  descriptor.bitfield.stride_byte_offset_ = stride_byte_offset;
  descriptor.bitfield.version_ = 1;
  descriptor.bitfield.base_offset_ = base_offset & 0x7;
  descriptor.bitfield.lbo_mode_ = leading_is_absolute ? 1 : 0;
  descriptor.bitfield.layout_type_ = swizzle_mode & 0x7;
}

template <typename T>
TL_DEVICE void increase_descriptor_offset(GmmaDescriptor &descriptor,
                                          T offset) {
  descriptor.reg32_[0] += (offset >> 4);
}

// and add the desired implicit conversion from bfloat16_t.
struct float_e4m3_t : public cute::float_e4m3_t {
  using cute::float_e4m3_t::float_e4m3_t;
  CUTLASS_HOST_DEVICE
  float_e4m3_t() = default;

  CUTLASS_HOST_DEVICE
  explicit float_e4m3_t(__nv_bfloat16 x)
      : float_e4m3_t(static_cast<float>(x)) {}
};

struct float_e5m2_t : public cute::float_e5m2_t {
  using cute::float_e5m2_t::float_e5m2_t;
  CUTLASS_HOST_DEVICE
  float_e5m2_t() = default;

  CUTLASS_HOST_DEVICE
  explicit float_e5m2_t(__nv_bfloat16 x)
      : float_e5m2_t(static_cast<float>(x)) {}
};

template <typename T> struct to_cute_type {
  using type = T;
};
template <> struct to_cute_type<tl::float_e4m3_t> {
  using type = cute::float_e4m3_t;
};
template <> struct to_cute_type<tl::float_e5m2_t> {
  using type = cute::float_e5m2_t;
};

} // namespace tl

namespace cutlass {
TL_DEVICE
bfloat16_t fast_exp(bfloat16_t x) { return ::hexp(x); }
} // namespace cutlass

//
// Type-safe warp shuffle helpers for 16-bit float types
// These wrappers avoid relying on implicit conversions that may be disallowed
// (e.g., converting float -> cutlass::bfloat16_t) by explicitly promoting to
// float for the shuffle and then down-converting.
//
namespace tl {

// Generic passthroughs
template <typename T>
TL_DEVICE T shfl_xor_sync(unsigned mask, T val, int laneMask) {
  return __shfl_xor_sync(mask, val, laneMask);
}

template <typename T>
TL_DEVICE T shfl_down_sync(unsigned mask, T val, int delta) {
  return __shfl_down_sync(mask, val, delta);
}

template <typename T>
TL_DEVICE T shfl_up_sync(unsigned mask, T val, int delta) {
  return __shfl_up_sync(mask, val, delta);
}

template <typename T> TL_DEVICE T shfl_sync(unsigned mask, T val, int srcLane) {
  return __shfl_sync(mask, val, srcLane);
}

// Specializations for cutlass::half_t
template <>
TL_DEVICE half_t shfl_xor_sync(unsigned mask, half_t val, int laneMask) {
  float f = static_cast<float>(val);
  float r = __shfl_xor_sync(mask, f, laneMask);
  return half_t(r);
}

template <>
TL_DEVICE half_t shfl_down_sync(unsigned mask, half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_down_sync(mask, f, delta);
  return half_t(r);
}

template <>
TL_DEVICE half_t shfl_up_sync(unsigned mask, half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_up_sync(mask, f, delta);
  return half_t(r);
}

template <> TL_DEVICE half_t shfl_sync(unsigned mask, half_t val, int srcLane) {
  float f = static_cast<float>(val);
  float r = __shfl_sync(mask, f, srcLane);
  return half_t(r);
}

// Specializations for cutlass::bfloat16_t
template <>
TL_DEVICE bfloat16_t shfl_xor_sync(unsigned mask, bfloat16_t val,
                                   int laneMask) {
  float f = static_cast<float>(val);
  float r = __shfl_xor_sync(mask, f, laneMask);
  return bfloat16_t(r);
}

template <>
TL_DEVICE bfloat16_t shfl_down_sync(unsigned mask, bfloat16_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_down_sync(mask, f, delta);
  return bfloat16_t(r);
}

template <>
TL_DEVICE bfloat16_t shfl_up_sync(unsigned mask, bfloat16_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_up_sync(mask, f, delta);
  return bfloat16_t(r);
}

template <>
TL_DEVICE bfloat16_t shfl_sync(unsigned mask, bfloat16_t val, int srcLane) {
  float f = static_cast<float>(val);
  float r = __shfl_sync(mask, f, srcLane);
  return bfloat16_t(r);
}

} // namespace tl
