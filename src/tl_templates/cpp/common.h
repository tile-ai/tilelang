#pragma once

#include "half.hpp"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

using half_float::half;

// Vector types for CPU codegen.
// The C codegen emits vector pointer casts like *(float4*)(ptr + offset)
// and broadcast expressions like ((float4)(v, v, v, v)) where the comma
// expression evaluates to a single value, requiring a one-arg constructor.
template <typename T, int N> struct vec_type {
  T data[N];
  vec_type() = default;
  explicit vec_type(T v) {
    for (int i = 0; i < N; i++)
      data[i] = v;
  }
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
