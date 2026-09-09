#pragma once

#include "common.h"

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

enum class LoadCachePolicy { kCA, kCG, kCS, kLU, kCV };
enum class StoreCachePolicy { kWB, kCG, kCS, kWT };

template <int Bytes> struct CachePolicyAccessType;
template <> struct CachePolicyAccessType<1> {
  using Type = unsigned char;
};
template <> struct CachePolicyAccessType<2> {
  using Type = unsigned short;
};
template <> struct CachePolicyAccessType<4> {
  using Type = unsigned int;
};
template <> struct CachePolicyAccessType<8> {
  using Type = unsigned long long;
};
template <> struct CachePolicyAccessType<16> {
  using Type = uint4;
};

template <LoadCachePolicy Policy, typename AccessType>
TL_DEVICE AccessType load_global_cache_native(const AccessType *ptr) {
  if constexpr (Policy == LoadCachePolicy::kCA) {
    return __ldca(ptr);
  } else if constexpr (Policy == LoadCachePolicy::kCG) {
    return __ldcg(ptr);
  } else if constexpr (Policy == LoadCachePolicy::kCS) {
    return __ldcs(ptr);
  } else if constexpr (Policy == LoadCachePolicy::kLU) {
    return __ldlu(ptr);
  } else {
    static_assert(Policy == LoadCachePolicy::kCV);
    return __ldcv(ptr);
  }
}

template <StoreCachePolicy Policy, typename AccessType>
TL_DEVICE void store_global_cache_native(AccessType *ptr, AccessType value) {
  if constexpr (Policy == StoreCachePolicy::kWB) {
    __stwb(ptr, value);
  } else if constexpr (Policy == StoreCachePolicy::kCG) {
    __stcg(ptr, value);
  } else if constexpr (Policy == StoreCachePolicy::kCS) {
    __stcs(ptr, value);
  } else {
    static_assert(Policy == StoreCachePolicy::kWT);
    __stwt(ptr, value);
  }
}

template <LoadCachePolicy Policy, typename T>
TL_DEVICE T load_global_cache(const T *ptr) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                sizeof(T) == 8 || sizeof(T) == 16 ||
                (sizeof(T) > 16 && sizeof(T) % 16 == 0));
  T result;
  if constexpr (sizeof(T) <= 16) {
    using AccessType = typename CachePolicyAccessType<sizeof(T)>::Type;
    *reinterpret_cast<AccessType *>(&result) = load_global_cache_native<Policy>(
        reinterpret_cast<const AccessType *>(ptr));
  } else {
    auto *result_chunks = reinterpret_cast<uint4 *>(&result);
    auto *ptr_chunks = reinterpret_cast<const uint4 *>(ptr);
#pragma unroll
    for (int i = 0; i < sizeof(T) / 16; ++i) {
      result_chunks[i] = load_global_cache_native<Policy>(ptr_chunks + i);
    }
  }
  return result;
}

template <StoreCachePolicy Policy, typename T>
TL_DEVICE void store_global_cache(T *ptr, const T &value) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                sizeof(T) == 8 || sizeof(T) == 16 ||
                (sizeof(T) > 16 && sizeof(T) % 16 == 0));
  if constexpr (sizeof(T) <= 16) {
    using AccessType = typename CachePolicyAccessType<sizeof(T)>::Type;
    store_global_cache_native<Policy>(
        reinterpret_cast<AccessType *>(ptr),
        *reinterpret_cast<const AccessType *>(&value));
  } else {
    auto *ptr_chunks = reinterpret_cast<uint4 *>(ptr);
    auto *value_chunks = reinterpret_cast<const uint4 *>(&value);
#pragma unroll
    for (int i = 0; i < sizeof(T) / 16; ++i) {
      store_global_cache_native<Policy>(ptr_chunks + i, value_chunks[i]);
    }
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

TL_DEVICE uint32_t load_shared_32(const void *ptr) {
  return *reinterpret_cast<const uint32_t *>(ptr);
}

TL_DEVICE uint2 load_shared_64(const void *ptr) {
  return *reinterpret_cast<const uint2 *>(ptr);
}

TL_DEVICE uint4 load_shared_128(const void *ptr) {
  return *reinterpret_cast<const uint4 *>(ptr);
}

TL_DEVICE void store_shared_32(void *ptr, uint32_t value) {
  *reinterpret_cast<uint32_t *>(ptr) = value;
}

TL_DEVICE void store_shared_64(void *ptr, uint2 value) {
  *reinterpret_cast<uint2 *>(ptr) = value;
}

TL_DEVICE void store_shared_128(void *ptr, uint4 value) {
  *reinterpret_cast<uint4 *>(ptr) = value;
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
