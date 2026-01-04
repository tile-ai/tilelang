#pragma once

#include <hip/hip_runtime.h>

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
template <typename T1, typename T2>
__forceinline__ __device__ void AtomicAdd(T1 *address, T2 val,
                                          int memory_order = 0) {
  atomicAdd(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
// Overload for when the first argument is a value instead of a pointer
template <typename T1, typename T2>
__forceinline__ __device__ void AtomicAdd(T1 &address, T2 val,
                                          int memory_order = 0) {
  atomicAdd(reinterpret_cast<T1 *>(&address), static_cast<T1>(val));
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
template <typename T1, typename T2>
__forceinline__ __device__ T1 AtomicAddRet(T1 *ref, T2 val,
                                           int memory_order = 0) {
  return atomicAdd(ref, static_cast<T1>(val));
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
template <typename T1, typename T2>
__forceinline__ __device__ void AtomicMax(T1 *address, T2 val,
                                          int memory_order = 0) {
  atomicMax(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
// Overload for when the first argument is a value instead of a pointer
template <typename T1, typename T2>
__forceinline__ __device__ void AtomicMax(T1 &address, T2 val,
                                          int memory_order = 0) {
  atomicMax(reinterpret_cast<T1 *>(&address), static_cast<T1>(val));
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
template <typename T1, typename T2>
__forceinline__ __device__ void AtomicMin(T1 *address, T2 val,
                                          int memory_order = 0) {
  atomicMin(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
// Overload for when the first argument is a value instead of a pointer
template <typename T1, typename T2>
__forceinline__ __device__ void AtomicMin(T1 &address, T2 val,
                                          int memory_order = 0) {
  atomicMin(reinterpret_cast<T1 *>(&address), static_cast<T1>(val));
}

__forceinline__ __device__ void AtomicAddx2(float *ref, float *val,
                                            int memory_order = 0) {
  float2 add_val = *reinterpret_cast<float2 *>(val);
  atomicAdd(ref + 0, add_val.x);
  atomicAdd(ref + 1, add_val.y);
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
__forceinline__ __device__ float2 AtomicAddx2Ret(float *ref, float *val,
                                                 int memory_order = 0) {
  float2 add_val = *reinterpret_cast<float2 *>(val);
  float2 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  return ret;
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
__forceinline__ __device__ void AtomicAddx4(float *ref, float *val,
                                            int memory_order = 0) {
  float4 add_val = *reinterpret_cast<float4 *>(val);
  atomicAdd(ref + 0, add_val.x);
  atomicAdd(ref + 1, add_val.y);
  atomicAdd(ref + 2, add_val.z);
  atomicAdd(ref + 3, add_val.w);
}

// Add an extra unused input to accommodate the additional 'memory_order'
// argument during lowering.
__forceinline__ __device__ float4 AtomicAddx4Ret(float *ref, float *val,
                                                 int memory_order = 0) {
  float4 add_val = *reinterpret_cast<float4 *>(val);
  float4 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  ret.z = atomicAdd(ref + 2, add_val.z);
  ret.w = atomicAdd(ref + 3, add_val.w);
  return ret;
}
