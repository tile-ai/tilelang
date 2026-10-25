#pragma once

#include <curand_kernel.h>
#ifdef __CUDACC_RTC__
#include <cuda/std/cstdint>
#else
#include <cstdint>
#endif

namespace tl {

template <typename State>
__device__ __forceinline__ void
random_init(uint64_t *storage, unsigned long long seed,
            unsigned long long sequence, unsigned long long offset) {
  static_assert(sizeof(State) <= 128);
  static_assert(alignof(State) <= 16);
  curand_init(seed, sequence, offset, reinterpret_cast<State *>(storage));
}

template <typename State>
__device__ __forceinline__ unsigned int random_uint(uint64_t *storage) {
  return curand(reinterpret_cast<State *>(storage));
}

template <typename State>
__device__ __forceinline__ float random_uniform(uint64_t *storage) {
  return curand_uniform(reinterpret_cast<State *>(storage));
}

template <typename State>
__device__ __forceinline__ double random_uniform_double(uint64_t *storage) {
  return curand_uniform_double(reinterpret_cast<State *>(storage));
}

template <typename State>
__device__ __forceinline__ float random_normal(uint64_t *storage) {
  return curand_normal(reinterpret_cast<State *>(storage));
}

template <typename State>
__device__ __forceinline__ double random_normal_double(uint64_t *storage) {
  return curand_normal_double(reinterpret_cast<State *>(storage));
}

} // namespace tl
