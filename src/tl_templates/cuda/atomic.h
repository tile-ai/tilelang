#pragma once

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

#include <cuda/atomic>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

using cutlass::bfloat16_t;
using cutlass::half_t;

#define TL_DEVICE __forceinline__ __device__
#define TL_NOT_IMPLEMENTED()                                                   \
  {                                                                            \
    printf("%s not implemented\n", __PRETTY_FUNCTION__);                       \
    asm volatile("brkpt;\n");                                                  \
  }
template <typename T> struct normalize_atomic_type {
  using type = T;
};

template <> struct normalize_atomic_type<half_t> {
  using type = half;
};

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 750))
template <> struct normalize_atomic_type<bfloat16_t> {
  using type = __nv_bfloat16;
};
#endif

template <> struct normalize_atomic_type<int64_t> {
  using type = unsigned long long;
};

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

// Helpers for atomic operations

namespace tl_atomic_detail {

TL_DEVICE bool IsRelaxedMemoryOrder(int memory_order) {
  return memory_order == int(cuda::memory_order_relaxed);
}

TL_DEVICE bool IsReleaseLikeMemoryOrder(int memory_order) {
  return memory_order == int(cuda::memory_order_release) ||
         memory_order == int(cuda::memory_order_consume);
}

TL_DEVICE bool IsAcquireMemoryOrder(int memory_order) {
  return memory_order == int(cuda::memory_order_acquire);
}

TL_DEVICE bool IsAcqRelLikeMemoryOrder(int memory_order) {
  return memory_order == int(cuda::memory_order_acq_rel) ||
         memory_order == int(cuda::memory_order_seq_cst);
}

template <typename T> TL_DEVICE unsigned short PackBits16(const T &val) {
  return *reinterpret_cast<const unsigned short *>(&val);
}

template <typename T> TL_DEVICE T UnpackBits16(unsigned short val) {
  return *reinterpret_cast<T *>(&val);
}

TL_DEVICE void tl_atomic_add_f16(unsigned short &ret, unsigned long long addr,
                                 unsigned short val, int memory_order) {
  if (IsReleaseLikeMemoryOrder(memory_order)) {
    asm volatile("atom.release.gpu.global.add.noftz.f16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  } else if (IsAcquireMemoryOrder(memory_order)) {
    asm volatile("atom.acquire.gpu.global.add.noftz.f16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  } else if (IsAcqRelLikeMemoryOrder(memory_order)) {
    asm volatile("atom.acq_rel.gpu.global.add.noftz.f16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  }
}

TL_DEVICE void tl_atomic_add_bf16(unsigned short &ret, unsigned long long addr,
                                  unsigned short val, int memory_order) {
  if (IsReleaseLikeMemoryOrder(memory_order)) {
    asm volatile("atom.release.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  } else if (IsAcquireMemoryOrder(memory_order)) {
    asm volatile("atom.acquire.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  } else if (IsAcqRelLikeMemoryOrder(memory_order)) {
    asm volatile("atom.acq_rel.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  }
}

TL_DEVICE void tl_atomic_add_v2_f16(unsigned short &ret_x,
                                    unsigned short &ret_y,
                                    unsigned long long addr,
                                    unsigned short val_x, unsigned short val_y,
                                    int memory_order) {
  if (IsReleaseLikeMemoryOrder(memory_order)) {
    asm volatile(
        "atom.release.gpu.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};"
        : "=h"(ret_x), "=h"(ret_y)
        : "l"(addr), "h"(val_x), "h"(val_y)
        : "memory");
  } else if (IsAcquireMemoryOrder(memory_order)) {
    asm volatile(
        "atom.acquire.gpu.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};"
        : "=h"(ret_x), "=h"(ret_y)
        : "l"(addr), "h"(val_x), "h"(val_y)
        : "memory");
  } else if (IsAcqRelLikeMemoryOrder(memory_order)) {
    asm volatile(
        "atom.acq_rel.gpu.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};"
        : "=h"(ret_x), "=h"(ret_y)
        : "l"(addr), "h"(val_x), "h"(val_y)
        : "memory");
  }
}

TL_DEVICE void tl_atomic_add_v2_bf16(unsigned short &ret_x,
                                     unsigned short &ret_y,
                                     unsigned long long addr,
                                     unsigned short val_x, unsigned short val_y,
                                     int memory_order) {
  if (IsReleaseLikeMemoryOrder(memory_order)) {
    asm volatile("atom.release.gpu.global.add.v2.bf16 {%0,%1}, [%2], {%3,%4};"
                 : "=h"(ret_x), "=h"(ret_y)
                 : "l"(addr), "h"(val_x), "h"(val_y)
                 : "memory");
  } else if (IsAcquireMemoryOrder(memory_order)) {
    asm volatile("atom.acquire.gpu.global.add.v2.bf16 {%0,%1}, [%2], {%3,%4};"
                 : "=h"(ret_x), "=h"(ret_y)
                 : "l"(addr), "h"(val_x), "h"(val_y)
                 : "memory");
  } else if (IsAcqRelLikeMemoryOrder(memory_order)) {
    asm volatile("atom.acq_rel.gpu.global.add.v2.bf16 {%0,%1}, [%2], {%3,%4};"
                 : "=h"(ret_x), "=h"(ret_y)
                 : "l"(addr), "h"(val_x), "h"(val_y)
                 : "memory");
  }
}

TL_DEVICE void tl_atomic_add_v2_f32(float &ret_x, float &ret_y,
                                    unsigned long long addr, float val_x,
                                    float val_y, int memory_order) {
  if (IsReleaseLikeMemoryOrder(memory_order)) {
    asm volatile("atom.release.gpu.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};"
                 : "=f"(ret_x), "=f"(ret_y)
                 : "l"(addr), "f"(val_x), "f"(val_y)
                 : "memory");
  } else if (IsAcquireMemoryOrder(memory_order)) {
    asm volatile("atom.acquire.gpu.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};"
                 : "=f"(ret_x), "=f"(ret_y)
                 : "l"(addr), "f"(val_x), "f"(val_y)
                 : "memory");
  } else if (IsAcqRelLikeMemoryOrder(memory_order)) {
    asm volatile("atom.acq_rel.gpu.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};"
                 : "=f"(ret_x), "=f"(ret_y)
                 : "l"(addr), "f"(val_x), "f"(val_y)
                 : "memory");
  }
}

TL_DEVICE void tl_atomic_add_v4_f32(float &ret_x, float &ret_y, float &ret_z,
                                    float &ret_w, unsigned long long addr,
                                    float val_x, float val_y, float val_z,
                                    float val_w, int memory_order) {
  if (IsReleaseLikeMemoryOrder(memory_order)) {
    asm volatile(
        "atom.release.gpu.global.add.v4.f32 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};"
        : "=f"(ret_x), "=f"(ret_y), "=f"(ret_z), "=f"(ret_w)
        : "l"(addr), "f"(val_x), "f"(val_y), "f"(val_z), "f"(val_w)
        : "memory");
  } else if (IsAcquireMemoryOrder(memory_order)) {
    asm volatile(
        "atom.acquire.gpu.global.add.v4.f32 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};"
        : "=f"(ret_x), "=f"(ret_y), "=f"(ret_z), "=f"(ret_w)
        : "l"(addr), "f"(val_x), "f"(val_y), "f"(val_z), "f"(val_w)
        : "memory");
  } else if (IsAcqRelLikeMemoryOrder(memory_order)) {
    asm volatile(
        "atom.acq_rel.gpu.global.add.v4.f32 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};"
        : "=f"(ret_x), "=f"(ret_y), "=f"(ret_z), "=f"(ret_w)
        : "l"(addr), "f"(val_x), "f"(val_y), "f"(val_z), "f"(val_w)
        : "memory");
  }
}

// Fallback implementations: do atomicAdd sequentially.

template <typename T> TL_DEVICE void AtomicAddx2Scalar(T *ref, T x, T y) {
  atomicAdd(ref + 0, x);
  atomicAdd(ref + 1, y);
}

template <typename T>
TL_DEVICE void AtomicAddx4Scalar(T *ref, T x, T y, T z, T w) {
  atomicAdd(ref + 0, x);
  atomicAdd(ref + 1, y);
  atomicAdd(ref + 2, z);
  atomicAdd(ref + 3, w);
}

TL_DEVICE float2 AtomicAddx2ScalarRet(float *ref, float2 add_val) {
  float2 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  return ret;
}

template <typename dst_dtype>
TL_DEVICE float4 AtomicAddx4ScalarRet(dst_dtype *ref, float4 add_val) {
  float4 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  ret.z = atomicAdd(ref + 2, add_val.z);
  ret.w = atomicAdd(ref + 3, add_val.w);
  return ret;
}

} // namespace tl_atomic_detail

template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 *ref, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    // There is no implementation of atomicMax for half and bf16 in cuda.
    // We simulate this process by atomicCAS loop.
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val > *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
  } else {
#if CUDART_VERSION >= 11080
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    aref.fetch_max(cuda_cast<NT1>(val), cuda::memory_order(memory_order));
#else
    TL_NOT_IMPLEMENTED();
#endif
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMaxRet(T1 *ref, T2 val,
                          int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val > *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
    return static_cast<T1>(*reinterpret_cast<T1 *>(&old_val_ushort));
  } else {
#if CUDART_VERSION >= 11080
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_max(cuda_cast<NT1>(val), cuda::memory_order(memory_order)));
#else
    TL_NOT_IMPLEMENTED();
#endif
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 *ref, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    // There is no implementation of atomicMin for half and bf16 in cuda.
    // We simulate this process by atomicCAS loop.
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val < *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
  } else {
#if CUDART_VERSION >= 11080
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    aref.fetch_min(cuda_cast<NT1>(val), cuda::memory_order(memory_order));
#else
    TL_NOT_IMPLEMENTED();
#endif
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMinRet(T1 *ref, T2 val,
                          int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val < *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
    return static_cast<T1>(*reinterpret_cast<T1 *>(&old_val_ushort));
  } else {
#if CUDART_VERSION >= 11080
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_min(cuda_cast<NT1>(val), cuda::memory_order(memory_order)));
#else
    TL_NOT_IMPLEMENTED();
#endif
  }
}

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 890))
template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
      atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
    } else {
      // Since atomic ref do not support memory order, we need to inline ptx
      // code here for each situation
      if constexpr (std::is_same_v<NT1, half>) {
        // fp16
        unsigned short ret_val_cast;
        unsigned long long ref_address =
            reinterpret_cast<unsigned long long>(address);
        unsigned short val_cast =
            tl_atomic_detail::PackBits16(cuda_cast<NT1>(val));
        tl_atomic_detail::tl_atomic_add_f16(ret_val_cast, ref_address, val_cast,
                                            memory_order);
      } else if constexpr (std::is_same_v<NT1, __nv_bfloat16>) {
        // bf16
        unsigned short ret_val_cast;
        unsigned long long ref_address =
            reinterpret_cast<unsigned long long>(address);
        unsigned short val_cast =
            tl_atomic_detail::PackBits16(cuda_cast<NT1>(val));
        tl_atomic_detail::tl_atomic_add_bf16(ret_val_cast, ref_address,
                                             val_cast, memory_order);
      }
    }
  } else {
    atomicAdd(reinterpret_cast<NT1 *>(address), cuda_cast<NT1>(val));
  }
}
#else
template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  (void)memory_order;
  atomicAdd(reinterpret_cast<NT1 *>(address), cuda_cast<NT1>(val));
}
#endif

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 *address, T2 val,
                          int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
      return static_cast<T1>(
          atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
    } else {
      if constexpr (std::is_same_v<NT1, half>) {
        // fp16
        unsigned short ret_val_cast;
        unsigned long long ref_address =
            reinterpret_cast<unsigned long long>(address);
        unsigned short val_cast =
            tl_atomic_detail::PackBits16(cuda_cast<NT1>(val));
        tl_atomic_detail::tl_atomic_add_f16(ret_val_cast, ref_address, val_cast,
                                            memory_order);
        return static_cast<T1>(
            tl_atomic_detail::UnpackBits16<__half>(ret_val_cast));
      } else if constexpr (std::is_same_v<NT1, __nv_bfloat16>) {
        // bf16
        unsigned short ret_val_cast;
        unsigned long long ref_address =
            reinterpret_cast<unsigned long long>(address);
        unsigned short val_cast =
            tl_atomic_detail::PackBits16(cuda_cast<NT1>(val));
        tl_atomic_detail::tl_atomic_add_bf16(ret_val_cast, ref_address,
                                             val_cast, memory_order);
        return static_cast<T1>(
            tl_atomic_detail::UnpackBits16<__nv_bfloat16>(ret_val_cast));
      }
    }
  } else {
#if CUDART_VERSION >= 11080
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_add(cuda_cast<NT1>(val), cuda::memory_order(memory_order)));
#else
    TL_NOT_IMPLEMENTED();
#endif
  }
}

// For vectorized AtomicAdd, we maintain two versions of interfaces:
// 1. AtomicAddxN(dst_type* ref, src_type *val) // Pass pointer
// 2. AtomicAddxN(dst_type* ref, src_type val) // Pass value
template <typename T> TL_DEVICE half2 ToHalf2(T *val) {
  return *reinterpret_cast<const half2 *>(val);
}

template <typename T> TL_DEVICE half2 ToHalf2(T val) {
  return static_cast<half2>(*reinterpret_cast<const half2 *>(&val));
}

TL_DEVICE half2 ToHalf2(half2 val) { return val; }

// Here ValType can be either value or value* (pointer)

template <typename ValType>
TL_DEVICE void AtomicAddx2(half_t *ref, ValType val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  half2 add_val = ToHalf2(val);
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    atomicAdd(reinterpret_cast<half2 *>(ref), add_val);
  } else {
    // Since atomicAdd does not support memory order, atomic_ref does not
    // support vectorized atomic operation we can only inline ptx code here
    // Note: Vectorized atomic operations only support global space
    // Note: for 16-bit value, we need to reinterpret_cast the value to unsigned
    // short and use "h" register in assembly
    unsigned short add_val_x_cast = tl_atomic_detail::PackBits16(add_val.x);
    unsigned short add_val_y_cast = tl_atomic_detail::PackBits16(add_val.y);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    unsigned short ret_val_x_cast;
    unsigned short ret_val_y_cast;
    tl_atomic_detail::tl_atomic_add_v2_f16(ret_val_x_cast, ret_val_y_cast,
                                           ref_addr, add_val_x_cast,
                                           add_val_y_cast, memory_order);
  }
}

template <typename ValType>
TL_DEVICE half2
AtomicAddx2Ret(half_t *ref, ValType val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  half2 add_val = ToHalf2(val);
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    return atomicAdd(reinterpret_cast<half2 *>(ref), add_val);
  } else {
    unsigned short add_val_x_cast = tl_atomic_detail::PackBits16(add_val.x);
    unsigned short add_val_y_cast = tl_atomic_detail::PackBits16(add_val.y);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    unsigned short ret_val_x_cast;
    unsigned short ret_val_y_cast;
    tl_atomic_detail::tl_atomic_add_v2_f16(ret_val_x_cast, ret_val_y_cast,
                                           ref_addr, add_val_x_cast,
                                           add_val_y_cast, memory_order);
    return half2(tl_atomic_detail::UnpackBits16<__half>(ret_val_x_cast),
                 tl_atomic_detail::UnpackBits16<__half>(ret_val_y_cast));
  }
}

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 750))
template <typename T> TL_DEVICE __nv_bfloat162 ToBfloat162(T *val) {
  return *reinterpret_cast<const __nv_bfloat162 *>(val);
}

template <typename T> TL_DEVICE __nv_bfloat162 ToBfloat162(T val) {
  return static_cast<__nv_bfloat162>(
      *reinterpret_cast<const __nv_bfloat162 *>(&val));
}

TL_DEVICE __nv_bfloat162 ToBfloat162(__nv_bfloat162 val) { return val; }

template <typename ValType>
TL_DEVICE void AtomicAddx2(bfloat16_t *ref, ValType val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  __nv_bfloat162 add_val = ToBfloat162(val);
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    atomicAdd(reinterpret_cast<__nv_bfloat162 *>(ref), add_val);
  } else {
    unsigned short add_val_x_cast = tl_atomic_detail::PackBits16(add_val.x);
    unsigned short add_val_y_cast = tl_atomic_detail::PackBits16(add_val.y);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    unsigned short ret_val_x_cast;
    unsigned short ret_val_y_cast;
    tl_atomic_detail::tl_atomic_add_v2_bf16(ret_val_x_cast, ret_val_y_cast,
                                            ref_addr, add_val_x_cast,
                                            add_val_y_cast, memory_order);
  }
}

template <typename src_type>
TL_DEVICE __nv_bfloat162
AtomicAddx2Ret(bfloat16_t *ref, src_type *val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    return atomicAdd(reinterpret_cast<__nv_bfloat162 *>(ref),
                     static_cast<__nv_bfloat162>(
                         *reinterpret_cast<const __nv_bfloat162 *>(val)));
  } else {
    __nv_bfloat162 add_val = *reinterpret_cast<const __nv_bfloat162 *>(val);
    unsigned short add_val_x_cast = tl_atomic_detail::PackBits16(add_val.x);
    unsigned short add_val_y_cast = tl_atomic_detail::PackBits16(add_val.y);
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    unsigned short ret_val_x_cast;
    unsigned short ret_val_y_cast;
    tl_atomic_detail::tl_atomic_add_v2_bf16(ret_val_x_cast, ret_val_y_cast,
                                            ref_addr, add_val_x_cast,
                                            add_val_y_cast, memory_order);
    return __nv_bfloat162(
        tl_atomic_detail::UnpackBits16<__nv_bfloat16>(ret_val_x_cast),
        tl_atomic_detail::UnpackBits16<__nv_bfloat16>(ret_val_y_cast));
  }
}
#endif

template <typename T> TL_DEVICE float2 ToFloat2(T *val) {
  return *reinterpret_cast<const float2 *>(val);
}

TL_DEVICE float2 ToFloat2(float2 val) { return val; }

template <typename T> TL_DEVICE float4 ToFloat4(T *val) {
  return *reinterpret_cast<const float4 *>(val);
}

TL_DEVICE float4 ToFloat4(float4 val) { return val; }

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
template <typename ValType>
TL_DEVICE void AtomicAddx2(float *ref, ValType val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  float2 add_val = ToFloat2(val);
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    atomicAdd(reinterpret_cast<float2 *>(ref), add_val);
  } else {
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    float2 ret_val;
    tl_atomic_detail::tl_atomic_add_v2_f32(ret_val.x, ret_val.y, ref_addr,
                                           add_val.x, add_val.y, memory_order);
  }
}

template <typename ValType>
TL_DEVICE float2
AtomicAddx2Ret(float *ref, ValType val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  float2 add_val = ToFloat2(val);
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    return atomicAdd(reinterpret_cast<float2 *>(ref), add_val);
  } else {
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    float2 ret_val;
    tl_atomic_detail::tl_atomic_add_v2_f32(ret_val.x, ret_val.y, ref_addr,
                                           add_val.x, add_val.y, memory_order);
    return ret_val;
  }
}

template <typename dst_dtype, typename ValType>
TL_DEVICE void AtomicAddx4(dst_dtype *ref, ValType val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  float4 add_val = ToFloat4(val);
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    atomicAdd(reinterpret_cast<float4 *>(ref), add_val);
  } else {
    // Since atomicAdd does not support memory order, atomic_ref does not
    // support vectorized atomic operation we can only inline ptx code here
    // Note: Vectorized atomic operations only support global space
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    float4 ret_val;
    tl_atomic_detail::tl_atomic_add_v4_f32(
        ret_val.x, ret_val.y, ret_val.z, ret_val.w, ref_addr, add_val.x,
        add_val.y, add_val.z, add_val.w, memory_order);
  }
}

template <typename dst_dtype, typename ValType>
TL_DEVICE float4
AtomicAddx4Ret(dst_dtype *ref, ValType val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  float4 add_val = ToFloat4(val);
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    return atomicAdd(reinterpret_cast<float4 *>(ref), add_val);
  } else {
    unsigned long long ref_addr = reinterpret_cast<unsigned long long>(ref);
    float4 ret_val;
    tl_atomic_detail::tl_atomic_add_v4_f32(
        ret_val.x, ret_val.y, ret_val.z, ret_val.w, ref_addr, add_val.x,
        add_val.y, add_val.z, add_val.w, memory_order);
    return ret_val;
  }
}
#else
template <typename ValType>
TL_DEVICE void AtomicAddx2(float *ref, ValType val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  (void)memory_order;
  float2 add_val = ToFloat2(val);
  tl_atomic_detail::AtomicAddx2Scalar(ref, add_val.x, add_val.y);
}

template <typename ValType>
TL_DEVICE float2
AtomicAddx2Ret(float *ref, ValType val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  (void)memory_order;
  float2 add_val = ToFloat2(val);
  return tl_atomic_detail::AtomicAddx2ScalarRet(ref, add_val);
}

template <typename dst_dtype, typename ValType>
TL_DEVICE void AtomicAddx4(dst_dtype *ref, ValType val,
                           int memory_order = int(cuda::memory_order_relaxed)) {
  (void)memory_order;
  float4 add_val = ToFloat4(val);
  tl_atomic_detail::AtomicAddx4Scalar(ref, add_val.x, add_val.y, add_val.z,
                                      add_val.w);
}

template <typename dst_dtype, typename ValType>
TL_DEVICE float4
AtomicAddx4Ret(dst_dtype *ref, ValType val,
               int memory_order = int(cuda::memory_order_relaxed)) {
  (void)memory_order;
  float4 add_val = ToFloat4(val);
  return tl_atomic_detail::AtomicAddx4ScalarRet(ref, add_val);
}
#endif

template <typename T> TL_DEVICE T AtomicLoad(T *ref, int memory_order) {
#if CUDART_VERSION >= 11080
  cuda::atomic_ref<T, cuda::thread_scope_device> aref(*ref);
  return aref.load(cuda::memory_order(memory_order));
#else
  TL_NOT_IMPLEMENTED();
#endif
}

template <typename T1, typename T2>
TL_DEVICE void AtomicStore(T1 *ref, T2 value, int memory_order) {
  using NT1 = typename normalize_atomic_type<T1>::type;
#if CUDART_VERSION >= 11080
  cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*ref);
  aref.store(cuda_cast<NT1>(value), cuda::memory_order(memory_order));
#else
  TL_NOT_IMPLEMENTED();
#endif
}
