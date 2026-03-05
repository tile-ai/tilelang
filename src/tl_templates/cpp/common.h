#pragma once

#include "half.hpp"

#include <math.h>
#include <stdbool.h>

using half_float::half;

// ============================================================================
// Vector types for CPU/C++ backend
// ============================================================================

#ifdef __cplusplus
// C++ version with constructor
struct float4 {
  float x, y, z, w;

  // Default constructor
  float4() : x(0), y(0), z(0), w(0) {}

  float4(float val) : x(val), y(val), z(val), w(val) {}

  // Constructor for C++ (used in generated code)
  float4(float x_val, float y_val, float z_val, float w_val)
      : x(x_val), y(y_val), z(z_val), w(w_val) {}
};
#else
// C version
typedef struct {
  float x, y, z, w;
} float4;
#endif

// Constructor for float4 (C and C++)
static inline float4 make_float4(float x, float y, float z, float w) {
  float4 result;
  result.x = x;
  result.y = y;
  result.z = z;
  result.w = w;
  return result;
}
