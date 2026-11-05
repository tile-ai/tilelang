#pragma once

#include "copy_sm100.h"
#ifndef __CUDACC_RTC__
// NVRTC supplies device types without the host C standard library headers.
#include <stddef.h>
#include <stdint.h>
#endif

namespace tl {

namespace detail {

// Use explicit byte accesses in the unaligned branch.  NVCC can combine an
// aligned vector access and a memcpy fallback into an unconditional vector
// access; volatile bytes prevent that transformation and bound every access.
// Volatility here is an access constraint, not a claim of atomicity.
template <typename T, int Bytes>
TL_DEVICE T load_vector_bytes(const volatile void *pointer) {
  static_assert(sizeof(T) == Bytes, "vector carrier must match storage bytes");
  T value;
  auto *target = reinterpret_cast<unsigned char *>(&value);
  auto *source = static_cast<const volatile unsigned char *>(pointer);
#pragma unroll
  for (int i = 0; i < Bytes; ++i) {
    target[i] = source[i];
  }
  return value;
}

template <typename T, int Bytes>
TL_DEVICE void store_vector_bytes(volatile void *pointer, const T &value) {
  static_assert(sizeof(T) == Bytes, "vector carrier must match storage bytes");
  auto *target = static_cast<volatile unsigned char *>(pointer);
  auto *source = reinterpret_cast<const unsigned char *>(&value);
#pragma unroll
  for (int i = 0; i < Bytes; ++i) {
    target[i] = source[i];
  }
}

} // namespace detail

// The address is passed before any vector pointer cast or dereference.  Global
// 256-bit instructions require 32-byte alignment even on CUDA versions whose
// ulonglong4 type has a smaller natural alignment.
template <typename T, int Bytes, bool Global256 = false>
TL_DEVICE T load_vector(const void *pointer) {
  static_assert(sizeof(T) == Bytes, "vector carrier must match storage bytes");
  static_assert(!Global256 || Bytes == 32,
                "global 256-bit access needs 32 bytes");
  constexpr int alignment = Global256 ? 32 : alignof(T);
  if ((reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0) {
    if constexpr (Global256) {
      return load_global_256(static_cast<const T *>(pointer));
    } else {
      return *static_cast<const T *>(pointer);
    }
  }
  return detail::load_vector_bytes<T, Bytes>(pointer);
}

template <typename T, int Bytes, bool Global256 = false>
TL_DEVICE void store_vector(void *pointer, const T &value) {
  static_assert(sizeof(T) == Bytes, "vector carrier must match storage bytes");
  static_assert(!Global256 || Bytes == 32,
                "global 256-bit access needs 32 bytes");
  constexpr int alignment = Global256 ? 32 : alignof(T);
  if ((reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0) {
    if constexpr (Global256) {
      store_global_256(pointer, value);
    } else {
      *static_cast<T *>(pointer) = value;
    }
    return;
  }
  detail::store_vector_bytes<T, Bytes>(pointer, value);
}

// CUDA vector structs generally cannot be copied through a volatile-qualified
// struct reference.  Keep each physical byte access volatile instead of
// casting away the qualifier or invoking non-volatile global-load helpers.
template <typename T, int Bytes, bool Global256 = false>
TL_DEVICE T load_vector(const volatile void *pointer) {
  return detail::load_vector_bytes<T, Bytes>(pointer);
}

template <typename T, int Bytes, bool Global256 = false>
TL_DEVICE void store_vector(volatile void *pointer, const T &value) {
  detail::store_vector_bytes<T, Bytes>(pointer, value);
}

// A bool tensor stores one byte per lane, while the CUDA register carrier is
// ushortN.  It must be converted lane by lane, never copied as sizeof(ushortN).
template <typename T, int Lanes>
TL_DEVICE T load_bool_vector(const volatile void *pointer) {
  static_assert(Lanes >= 2 && Lanes <= 4, "CUDA bool carriers have 2-4 lanes");
  static_assert(sizeof(T) == Lanes * sizeof(unsigned short),
                "bool vector carrier must contain ushort lanes");
  T value;
  auto *source = static_cast<const volatile unsigned char *>(pointer);
  value.x = source[0] != 0;
  value.y = source[1] != 0;
  if constexpr (Lanes >= 3) {
    value.z = source[2] != 0;
  }
  if constexpr (Lanes == 4) {
    value.w = source[3] != 0;
  }
  return value;
}

template <typename T, int Lanes>
TL_DEVICE void store_bool_vector(volatile void *pointer, const T &value) {
  static_assert(Lanes >= 2 && Lanes <= 4, "CUDA bool carriers have 2-4 lanes");
  static_assert(sizeof(T) == Lanes * sizeof(unsigned short),
                "bool vector carrier must contain ushort lanes");
  auto *target = static_cast<volatile unsigned char *>(pointer);
  target[0] = value.x != 0;
  target[1] = value.y != 0;
  if constexpr (Lanes >= 3) {
    target[2] = value.z != 0;
  }
  if constexpr (Lanes == 4) {
    target[3] = value.w != 0;
  }
}

} // namespace tl
