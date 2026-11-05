#pragma once

#include "half.hpp"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <type_traits>

using half_float::half;

// Vector types for CPU codegen.
// The C codegen emits vector pointer casts like *(float4*)(ptr + offset)
// and constructs vectors from either one broadcast value or N lane values.
template <typename T, int N> struct vec_type {
  T data[N];
  vec_type() = default;
  explicit vec_type(T v) {
    for (int i = 0; i < N; i++)
      data[i] = v;
  }
  template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == N>>
  explicit vec_type(Args... values) : data{static_cast<T>(values)...} {}
  template <typename U> explicit vec_type(const vec_type<U, N> &other) {
    for (int i = 0; i < N; ++i)
      data[i] = static_cast<T>(other.data[i]);
  }
  T &operator[](int lane) { return data[lane]; }
  const T &operator[](int lane) const { return data[lane]; }
  vec_type operator+(const vec_type &o) const {
    vec_type r;
    for (int i = 0; i < N; i++)
      r.data[i] = data[i] + o.data[i];
    return r;
  }
  vec_type operator-(const vec_type &o) const {
    vec_type r;
    for (int i = 0; i < N; i++)
      r.data[i] = data[i] - o.data[i];
    return r;
  }
  vec_type operator*(const vec_type &o) const {
    vec_type r;
    for (int i = 0; i < N; i++)
      r.data[i] = data[i] * o.data[i];
    return r;
  }
  vec_type operator/(const vec_type &o) const {
    vec_type r;
    for (int i = 0; i < N; i++)
      r.data[i] = data[i] / o.data[i];
    return r;
  }
  // Compound assignment — avoids temporary vec_type objects
  vec_type &operator+=(const vec_type &o) {
    for (int i = 0; i < N; i++)
      data[i] += o.data[i];
    return *this;
  }
  vec_type &operator-=(const vec_type &o) {
    for (int i = 0; i < N; i++)
      data[i] -= o.data[i];
    return *this;
  }
  vec_type &operator*=(const vec_type &o) {
    for (int i = 0; i < N; i++)
      data[i] *= o.data[i];
    return *this;
  }
  vec_type &operator/=(const vec_type &o) {
    for (int i = 0; i < N; i++)
      data[i] /= o.data[i];
    return *this;
  }
};

#define TL_DEFINE_VEC(T)                                                       \
  using T##2 = vec_type<T, 2>;                                                 \
  using T##3 = vec_type<T, 3>;                                                 \
  using T##4 = vec_type<T, 4>;                                                 \
  using T##8 = vec_type<T, 8>;                                                 \
  using T##16 = vec_type<T, 16>;

TL_DEFINE_VEC(float)
TL_DEFINE_VEC(double)
TL_DEFINE_VEC(half)
TL_DEFINE_VEC(int8_t)
TL_DEFINE_VEC(int16_t)
TL_DEFINE_VEC(int32_t)
TL_DEFINE_VEC(int64_t)
TL_DEFINE_VEC(uint8_t)
TL_DEFINE_VEC(uint16_t)
TL_DEFINE_VEC(uint32_t)
TL_DEFINE_VEC(uint64_t)

#undef TL_DEFINE_VEC

namespace tl {

template <typename T> inline T clamp(T x, T lo, T hi) {
  if (x != x)
    return x;
  if (lo != lo)
    return lo;
  if (hi != hi)
    return hi;
  T lower = x < lo ? lo : x;
  return lower > hi ? hi : lower;
}

template <typename T, int N>
inline vec_type<T, N> clamp(vec_type<T, N> x, vec_type<T, N> lo,
                            vec_type<T, N> hi) {
  vec_type<T, N> result;
  for (int i = 0; i < N; ++i) {
    result.data[i] = clamp(x.data[i], lo.data[i], hi.data[i]);
  }
  return result;
}

} // namespace tl
