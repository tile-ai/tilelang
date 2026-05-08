#pragma once

#include "common.h"

#ifdef __CUDA_ARCH_LIST__
#if __CUDA_ARCH_LIST__ >= 900
#include "copy_sm90.h"
#endif
#if __CUDA_ARCH_LIST__ >= 1000
#include "copy_sm100.h"
#endif
#endif

namespace tl {

TL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> TL_DEVICE void cp_async_wait() {
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
}

template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr,
                           void const *global_ptr) {
  static_assert(N == 16 || N == 8 || N == 4);
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  if constexpr (N == 16) {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
        "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N));
  } else {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
#else
        "cp.async.ca.shared.global [%0], [%1], %2;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N));
  }
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *global_ptr, bool cond) {
  static_assert(N == 16 || N == 8 || N == 4);
  int bytes = cond ? N : 0;
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  if constexpr (N == 16) {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
        "cp.async.cg.shared.global [%0], [%1], %2, %3;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N), "r"(bytes));
  } else {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
        "cp.async.ca.shared.global [%0], [%1], %2, %3;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N), "r"(bytes));
  }
}

// Global memory load intrinsics with explicit vector widths
// Following CUTLASS style with template specialization

// Primary template declaration
template <typename AccessType, int LoadBytes> struct global_load;

// ldg32: Load 32 bits (4 bytes) from global memory
template <typename AccessType> struct global_load<AccessType, 4> {
  TL_DEVICE global_load(AccessType &D, void const *ptr, bool pred_guard) {
    unsigned &data = reinterpret_cast<unsigned &>(D);
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %2, 0;\n"
                 "  mov.b32 %0, %3;\n"
#if TL_ENABLE_L2_PREFETCH
                 "  @p ld.global.L2::128B.u32 %0, [%1];\n"
#else
                 "  @p ld.global.u32 %0, [%1];\n"
#endif
                 "}\n"
                 : "=r"(data)
                 : "l"(ptr), "r"((int)pred_guard), "r"(data));
  }
};

// ldg64: Load 64 bits (8 bytes) from global memory
template <typename AccessType> struct global_load<AccessType, 8> {
  TL_DEVICE global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint2 &data = reinterpret_cast<uint2 &>(D);
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %3, 0;\n"
                 "  mov.b32 %0, %4;\n"
                 "  mov.b32 %1, %5;\n"
#if TL_ENABLE_L2_PREFETCH
                 "  @p ld.global.L2::128B.v2.u32 {%0, %1}, [%2];\n"
#else
                 "  @p ld.global.v2.u32 {%0, %1}, [%2];\n"
#endif
                 "}\n"
                 : "=r"(data.x), "=r"(data.y)
                 : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y));
  }
};

// ldg128: Load 128 bits (16 bytes) from global memory
template <typename AccessType> struct global_load<AccessType, 16> {
  TL_DEVICE global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint4 &data = reinterpret_cast<uint4 &>(D);
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %5, 0;\n"
                 "  mov.b32 %0, %6;\n"
                 "  mov.b32 %1, %7;\n"
                 "  mov.b32 %2, %8;\n"
                 "  mov.b32 %3, %9;\n"
#if TL_ENABLE_L2_PREFETCH
                 "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
#else
                 "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
#endif
                 "}\n"
                 : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                 : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y),
                   "r"(data.z), "r"(data.w));
  }
};

// Convenience wrapper functions for direct use
// load_global_32: Load 32 bits, return uint32_t
TL_DEVICE uint32_t load_global_32(const void *ptr) {
  uint32_t ret{};
  global_load<uint32_t, 4>(ret, ptr, true);
  return ret;
}

// load_global_64: Load 64 bits, return uint64_t
TL_DEVICE uint2 load_global_64(const void *ptr) {
  uint2 ret{};
  global_load<uint2, 8>(ret, ptr, true);
  return ret;
}

// load_global_128: Load 128 bits, return uint4
TL_DEVICE uint4 load_global_128(const void *ptr) {
  uint4 ret{};
  global_load<uint4, 16>(ret, ptr, true);
  return ret;
}

// Predicated (conditional) versions
TL_DEVICE uint32_t load_global_32_conditional(const void *ptr, bool pred) {
  uint32_t ret{};
  global_load<uint32_t, 4>(ret, ptr, pred);
  return ret;
}

TL_DEVICE uint2 load_global_64_conditional(const void *ptr, bool pred) {
  uint2 ret{};
  global_load<uint2, 8>(ret, ptr, pred);
  return ret;
}

TL_DEVICE uint4 load_global_128_conditional(const void *ptr, bool pred) {
  uint4 ret{};
  global_load<uint4, 16>(ret, ptr, pred);
  return ret;
}

// Global memory store intrinsics with explicit vector widths
// Following CUTLASS style with template specialization

// Primary template declaration
template <typename AccessType, int StoreBytes> struct global_store;

// stg32: Store 32 bits (4 bytes) to global memory
template <typename AccessType> struct global_store<AccessType, 4> {
  TL_DEVICE global_store(void *ptr, AccessType const &D, bool pred_guard) {
    unsigned const &data = reinterpret_cast<unsigned const &>(D);
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %2, 0;\n"
                 "  @p st.global.u32 [%0], %1;\n"
                 "}\n"
                 :
                 : "l"(ptr), "r"(data), "r"((int)pred_guard));
  }
};

// stg64: Store 64 bits (8 bytes) to global memory
template <typename AccessType> struct global_store<AccessType, 8> {
  TL_DEVICE global_store(void *ptr, AccessType const &D, bool pred_guard) {
    uint2 const &data = reinterpret_cast<uint2 const &>(D);
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %3, 0;\n"
                 "  @p st.global.v2.u32 [%0], {%1, %2};\n"
                 "}\n"
                 :
                 : "l"(ptr), "r"(data.x), "r"(data.y), "r"((int)pred_guard));
  }
};

// stg128: Store 128 bits (16 bytes) to global memory
template <typename AccessType> struct global_store<AccessType, 16> {
  TL_DEVICE global_store(void *ptr, AccessType const &D, bool pred_guard) {
    uint4 const &data = reinterpret_cast<uint4 const &>(D);
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %5, 0;\n"
                 "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 "}\n"
                 :
                 : "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w),
                   "r"((int)pred_guard));
  }
};

// Convenience wrapper functions for direct use
// store_global_32: Store 32 bits
TL_DEVICE void store_global_32(void *ptr, uint32_t value) {
  global_store<uint32_t, 4>(ptr, value, true);
}

// store_global_64: Store 64 bits
TL_DEVICE void store_global_64(void *ptr, uint2 value) {
  global_store<uint2, 8>(ptr, value, true);
}

// store_global_128: Store 128 bits
TL_DEVICE void store_global_128(void *ptr, uint4 value) {
  global_store<uint4, 16>(ptr, value, true);
}

// Predicated (conditional) versions
TL_DEVICE void store_global_32_conditional(void *ptr, uint32_t value,
                                           bool pred) {
  global_store<uint32_t, 4>(ptr, value, pred);
}

TL_DEVICE void store_global_64_conditional(void *ptr, uint2 value, bool pred) {
  global_store<uint2, 8>(ptr, value, pred);
}

TL_DEVICE void store_global_128_conditional(void *ptr, uint4 value, bool pred) {
  global_store<uint4, 16>(ptr, value, pred);
}

} // namespace tl
