#pragma once

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

#include <cuda/atomic>
#include <cutlass/numeric_types.h>

using cutlass::bfloat16_t;
using cutlass::half_t;

#define TL_DEVICE __forceinline__ __device__

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
TL_DEVICE void AtomicMax(T1 &ref, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    atomicMax(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
  } else {
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    aref.fetch_max(cuda_cast<NT1>(val), cuda::memory_order(memory_order));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMaxRet(T1 &ref, T2 val,
                          int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    return static_cast<T1>(
        atomicMax(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
  } else {
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_max(cuda_cast<NT1>(val), cuda::memory_order(memory_order)));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 &ref, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    atomicMin(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
  } else {
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    aref.fetch_min(cuda_cast<NT1>(val), cuda::memory_order(memory_order));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMinRet(T1 &ref, T2 val,
                          int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    return static_cast<T1>(
        atomicMin(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
  } else {
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_min(cuda_cast<NT1>(val), cuda::memory_order(memory_order)));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 &ref, T2 val,
                         int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
  } else {
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    aref.fetch_add(cuda_cast<NT1>(val), cuda::memory_order(memory_order));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 &ref, T2 val,
                          int memory_order = int(cuda::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __nv_bfloat16>) {
    return static_cast<T1>(
        atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
  } else {
    cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_add(cuda_cast<NT1>(val), cuda::memory_order(memory_order)));
  }
}

TL_DEVICE void AtomicAddx2(half_t *ref, half_t *val) {
  atomicAdd(reinterpret_cast<half2 *>(ref),
            static_cast<half2>(*reinterpret_cast<half2 *>(val)));
}

TL_DEVICE half2 AtomicAddx2Ret(half_t *ref, half_t *val) {
  return atomicAdd(reinterpret_cast<half2 *>(ref),
                   static_cast<half2>(*reinterpret_cast<half2 *>(val)));
}

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ > 750))
TL_DEVICE void AtomicAddx2(bfloat16_t *ref, bfloat16_t *val) {
  atomicAdd(
      reinterpret_cast<__nv_bfloat162 *>(ref),
      static_cast<__nv_bfloat162>(*reinterpret_cast<__nv_bfloat162 *>(val)));
}

TL_DEVICE __nv_bfloat162 AtomicAddx2Ret(bfloat16_t *ref, bfloat16_t *val) {
  return atomicAdd(
      reinterpret_cast<__nv_bfloat162 *>(ref),
      static_cast<__nv_bfloat162>(*reinterpret_cast<__nv_bfloat162 *>(val)));
}
#endif

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
TL_DEVICE void AtomicAddx2(float *ref, float *val) {
  atomicAdd(reinterpret_cast<float2 *>(ref),
            static_cast<float2>(*reinterpret_cast<float2 *>(val)));
}

TL_DEVICE float2 AtomicAddx2Ret(float *ref, float *val) {
  return atomicAdd(reinterpret_cast<float2 *>(ref),
                   static_cast<float2>(*reinterpret_cast<float2 *>(val)));
}

TL_DEVICE void AtomicAddx4(float *ref, float *val) {
  atomicAdd(reinterpret_cast<float4 *>(ref),
            static_cast<float4>(*reinterpret_cast<float4 *>(val)));
}

TL_DEVICE float4 AtomicAddx4Ret(float *ref, float *val) {
  return atomicAdd(reinterpret_cast<float4 *>(ref),
                   static_cast<float4>(*reinterpret_cast<float4 *>(val)));
}
#endif

template <typename T> TL_DEVICE T AtomicLoad(T &ref, int memory_order) {
  cuda::atomic_ref<T, cuda::thread_scope_device> aref(ref);
  return aref.load(cuda::memory_order(memory_order));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicStore(T1 &ref, T2 value, int memory_order) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  cuda::atomic_ref<NT1, cuda::thread_scope_device> aref(ref);
  aref.store(cuda_cast<NT1>(value), cuda::memory_order(memory_order));
}
