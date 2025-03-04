// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cuda_runtime.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>
#include <math_constants.h>

using cutlass::bfloat16_t;
using cutlass::half_t;
using cutlass::tfloat32_t;

using int4_t = int4;

#define hexp cutlass::fast_exp
#define hlog cutlass::fast_log
#define hsqrt cutlass::fast_sqrt
#define htanh cutlass::fast_tanh
#define hpow powf

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE __forceinline__ __device__

// Pack two half values.
TL_DEVICE unsigned __pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_half2(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_nv_bfloat162(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack four char values
TL_DEVICE int make_int(signed char x0, signed char x1, signed char x2,
                       signed char x3) {
  return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
}

// Pack sixteen char values.
TL_DEVICE int4_t make_int4(signed char x0, signed char x1, signed char x2,
                           signed char x3, signed char y0, signed char y1,
                           signed char y2, signed char y3, signed char z0,
                           signed char z1, signed char z2, signed char z3,
                           signed char w0, signed char w1, signed char w2,
                           signed char w3) {
  int4_t result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  result.z = make_int(z0, z1, z2, z3);
  result.w = make_int(w0, w1, w2, w3);
  return result;
}

// Helper to cast SMEM pointer to unsigned
TL_DEVICE uint32_t smem_ptr_to_uint(void const *const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Helper to cast SMEM pointer to unsigned
TL_DEVICE unsigned int cast_smem_ptr_to_int(const void *const smem_ptr) {
  unsigned int smem_int;
  asm volatile("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; "
               "cvt.u32.u64 %0, smem_int; }"
               : "=r"(smem_int)
               : "l"(smem_ptr));
  return smem_int;
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 val) {
  atomicAdd(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

// // AtomicAdd Functions for FP32
// TL_DEVICE void AtomicAdd(float *address, float val) {
//   atomicAdd(reinterpret_cast<float *>(address), val);
// }

// AtomicAdd Functions for FP16
template <> TL_DEVICE void AtomicAdd(half_t *address, half_t val) {
  // Use atomicCAS with built-in cuda_fp16 support
  atomicAdd(reinterpret_cast<half *>(address), static_cast<half>(val));
}

// AtomicAdd Functions for FP16
template <> TL_DEVICE void AtomicAdd(half_t *address, half_t *val) {
  atomicAdd(reinterpret_cast<half *>(address), static_cast<half>(*val));
}

// AtomicAdd Functions for FP16
template <> TL_DEVICE void AtomicAdd(half_t *address, float val) {
  // Use atomicCAS with built-in cuda_fp16 support
  atomicAdd(reinterpret_cast<half *>(address), __float2half(val));
}

// AtomicAdd Functions for BFLOAT16
template <> TL_DEVICE void AtomicAdd(bfloat16_t *address, bfloat16_t *val) {
  atomicAdd(reinterpret_cast<__nv_bfloat16 *>(address),
            static_cast<__nv_bfloat16>(*val));
}

// AtomicAdd Functions for BFLOAT16
template <> TL_DEVICE void AtomicAdd(bfloat16_t *address, float val) {
  atomicAdd(reinterpret_cast<__nv_bfloat16 *>(address), __float2bfloat16(val));
}

// AtomicAdd Functions for BFLOAT16
template <> TL_DEVICE void AtomicAdd(bfloat16_t *address, bfloat16_t val) {
  atomicAdd(reinterpret_cast<__nv_bfloat16 *>(address),
            static_cast<__nv_bfloat16>(val));
}

// AtomicAdd Functions for FP16x2
TL_DEVICE void AtomicAddx2(half_t *address, half_t *val) {
  atomicAdd(reinterpret_cast<half2 *>(address),
            static_cast<half2>(*reinterpret_cast<half2 *>(val)));
}

// AtomicAdd Functions for BFLOAT16x2
TL_DEVICE void AtomicAddx2(bfloat16_t *address, bfloat16_t *val) {
  atomicAdd(
      reinterpret_cast<__nv_bfloat162 *>(address),
      static_cast<__nv_bfloat162>(*reinterpret_cast<__nv_bfloat162 *>(val)));
}

// DP4A
template <typename InDatatype, typename OutDatatype>
TL_DEVICE void DP4A(InDatatype *a, InDatatype *b, OutDatatype *c) {
  const int a_int = *((int *)a);
  const int b_int = *((int *)b);
  const int c_int = *((int *)c);
  *c = __dp4a(a_int, b_int, c_int);
}
