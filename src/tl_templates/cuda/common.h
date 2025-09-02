#pragma once

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

#include <cuda/atomic>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>
#include <math_constants.h>

using cutlass::bfloat16_t;
using cutlass::half_t;
using cutlass::tfloat32_t;

using int4_t = int4;

#define hexp cutlass::fast_exp
#define hlog cutlass::fast_log
#define hsqrt cutlass::fast_sqrt
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

// abs function for bfloat_t and half_t since there is no implicit conversion
// method
TL_PATCH TL_DEVICE half_t __habs(const half_t x) {
  return half_t(__habs(x.to_half()));
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

template <typename T> struct normalize_atomic_type {
  using type = T;
};

template <> /**
             * Map the public half_t alias to the native `half` type for atomic
             * operations.
             *
             * Used by the atomic utilities to normalize externally exposed
             * typedefs (e.g., Cutlass half_t) to the compiler's native `half`
             * representation so correct atomic intrinsics or `cuda::atomic_ref`
             * specializations can be selected.
             */
struct normalize_atomic_type<half_t> {
  using type = half;
};

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 750))
template <> struct normalize_atomic_type<bfloat16_t> {
  using type = __nv_bfloat16;
};
#endif

template <typename T1, typename T2> TL_DEVICE T1 cuda_cast(T2 val) {
  return T1(val);
}

template <> TL_DEVICE half cuda_cast<half, float>(float val) {
  return __float2half(val);
}

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 750))
template <> TL_DEVICE __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val) {
  return __float2bfloat16(val);
}
#endif

template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 *address, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    atomicMax(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
  } else {
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    aref.fetch_max(cuda_cast<NT1>(val), cuda::memory_order(memory_order));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 *address, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    atomicMin(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
  } else {
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    aref.fetch_min(cuda_cast<NT1>(val), cuda::memory_order(memory_order));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
  } else {
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    aref.fetch_add(cuda_cast<NT1>(val), cuda::memory_order(memory_order));
  }
}

// AtomicAdd Functions for FP16x2
TL_DEVICE void AtomicAddx2(half_t *address, half_t *val) {
  atomicAdd(reinterpret_cast<half2 *>(address),
            static_cast<half2>(*reinterpret_cast<half2 *>(val)));
}

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 750))

// AtomicAdd Functions for BFLOAT16x2
TL_DEVICE void AtomicAddx2(bfloat16_t *address, bfloat16_t *val) {
  atomicAdd(
      reinterpret_cast<__nv_bfloat162 *>(address),
      static_cast<__nv_bfloat162>(*reinterpret_cast<__nv_bfloat162 *>(val)));
}
#endif

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
// AtomicAdd Functions for FLOAT16x2
TL_DEVICE void AtomicAddx2(float *address, float *val) {
  atomicAdd(reinterpret_cast<float2 *>(address),
            static_cast<float2>(*reinterpret_cast<float2 *>(val)));
}
// AtomicAdd Functions for FLOAT16x4
TL_DEVICE void AtomicAddx4(float *address, float *val) {
  atomicAdd(reinterpret_cast<float4 *>(address),
            static_cast<float4>(*reinterpret_cast<float4 *>(val)));
}
#endif

template <typename T> TL_DEVICE T AtomicLoad(T *address, int memory_order) {
  cuda::atomic_ref<T, cuda::thread_scope_device> aref(*address);
  return aref.load(cuda::memory_order(memory_order));
}

template <typename T1, typename T2>
TL_DEVICE /**
           * Atomically stores a value into the given address using the
           * specified memory ordering.
           *
           * The value is converted to the normalized atomic storage type for T1
           * before being stored (for example, vectorized or reduced-width types
           * such as FP16/BF16 are mapped to their underlying hardware
           * representation). `memory_order` must be an `int` representation of
           * a `cuda::memory_order` value (e.g.,
           * `int(cuda::memory_order_relaxed)`).
           *
           * @param address Pointer to the destination atomic object.
           * @param value Value to store; will be cast to the atomic storage
           * type.
           * @param memory_order Memory ordering for the atomic store (as an
           * `int`-cast `cuda::memory_order`).
           */
    void
    AtomicStore(T1 *address, T2 value, int memory_order) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
  aref.store(cuda_cast<NT1>(value), cuda::memory_order(memory_order));
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
} // namespace tl

namespace cutlass {
TL_DEVICE
bfloat16_t fast_exp(bfloat16_t x) { return ::hexp(x); }
} // namespace cutlass
