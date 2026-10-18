#include "cuda/math.h"
#include <cstdint>
#include <cuda_runtime.h>

template <bool Native>
__device__ __forceinline__ float checked_clamp(float x, float lo, float hi) {
  if constexpr (Native) {
    return tl::clamp(x, lo, hi);
  } else {
    // Frozen pre-optimization helper, including operand-by-value semantics.
    if (isnan(x))
      return x;
    if (isnan(lo))
      return lo;
    if (isnan(hi))
      return hi;
    return fminf(fmaxf(x, lo), hi);
  }
}

template <bool Native>
__global__ void values_kernel(const float *x, const float *lo, const float *hi,
                              float *out, int n, bool constants) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = constants ? checked_clamp<Native>(x[i], -1.0f, 1.0f)
                       : checked_clamp<Native>(x[i], lo[i], hi[i]);
  }
}

template <bool Native, int Chains>
__global__ void chain_kernel(const float *input, float *out, int iterations,
                             float lo, float hi, float alpha, float bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x[Chains];
#pragma unroll
  for (int j = 0; j < Chains; ++j)
    x[j] = input[i * Chains + j];
#pragma unroll 1
  for (int k = 0; k < iterations; ++k) {
#pragma unroll
    for (int j = 0; j < Chains; ++j)
      x[j] = checked_clamp<Native>(fmaf(x[j], alpha, bias), lo, hi);
  }
#pragma unroll
  for (int j = 0; j < Chains; ++j)
    out[i * Chains + j] = x[j];
}

template <bool Native>
__global__ void fused_kernel(const float *a, const float *b, const float *lo,
                             const float *hi, float *out, int n, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = checked_clamp<Native>(fmaf(a[i], alpha, b[i]), lo[i], hi[i]);
}

extern "C" int values(int native, const float *x, const float *lo,
                      const float *hi, float *out, int n, int constants,
                      uintptr_t stream) {
  if (native)
    values_kernel<true><<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        x, lo, hi, out, n, constants);
  else
    values_kernel<false><<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        x, lo, hi, out, n, constants);
  return cudaGetLastError();
}

extern "C" int chains(int native, int count, const float *input, float *out,
                      int blocks, int iterations, float lo, float hi,
                      float alpha, float bias, uintptr_t stream) {
  if (native && count == 1)
    chain_kernel<true, 1><<<blocks, 256, 0, (cudaStream_t)stream>>>(
        input, out, iterations, lo, hi, alpha, bias);
  else if (native)
    chain_kernel<true, 8><<<blocks, 256, 0, (cudaStream_t)stream>>>(
        input, out, iterations, lo, hi, alpha, bias);
  else if (count == 1)
    chain_kernel<false, 1><<<blocks, 256, 0, (cudaStream_t)stream>>>(
        input, out, iterations, lo, hi, alpha, bias);
  else
    chain_kernel<false, 8><<<blocks, 256, 0, (cudaStream_t)stream>>>(
        input, out, iterations, lo, hi, alpha, bias);
  return cudaGetLastError();
}

extern "C" int fused(int native, const float *a, const float *b,
                     const float *lo, const float *hi, float *out, int n,
                     float alpha, uintptr_t stream) {
  if (native)
    fused_kernel<true><<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        a, b, lo, hi, out, n, alpha);
  else
    fused_kernel<false><<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        a, b, lo, hi, out, n, alpha);
  return cudaGetLastError();
}
