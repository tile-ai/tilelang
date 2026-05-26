#pragma once

#include "common.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tl {

struct ScanSumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }
};

struct ScanMaxOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return cutlass::fast_max(x, y);
  }

  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x, bfloat16_t const &y) {
    return bfloat16_t(__hmax(x.to_nv_bfloat16(), y.to_nv_bfloat16()));
  }

  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmax(x.to_half(), y.to_half()));
  }
};

template <class Reducer, bool reverse, typename T, int SEG = 32>
static TL_DEVICE void InclusiveScanLine(const T *__restrict__ src,
                                        T *__restrict__ dst, int extent,
                                        int stride) {
  if (extent <= 0)
    return;

  constexpr unsigned MASK = 0xffffffff;
  const int lane = threadIdx.x % SEG;
  T carry{};
  bool has_carry = false;
  const int num_segments = (extent + SEG - 1) / SEG;

  if constexpr (reverse) {
    for (int seg = num_segments - 1; seg >= 0; --seg) {
      const int base = seg * SEG;
      const int active = (extent - base < SEG) ? (extent - base) : SEG;
      T val = src[base * stride];
      if (lane < active)
        val = src[(base + lane) * stride];

#pragma unroll
      for (int off = 1; off < SEG; off <<= 1) {
        T n = tl::shfl_down_sync(MASK, val, off);
        if (lane + off < active)
          val = Reducer()(val, n);
      }

      if (has_carry && lane < active)
        val = Reducer()(val, carry);

      if (lane < active)
        dst[(base + lane) * stride] = val;

      T seg_result = tl::shfl_sync(MASK, val, 0);
      if (lane == 0)
        carry = seg_result;
      carry = tl::shfl_sync(MASK, carry, 0);
      has_carry = true;
    }
  } else {
    for (int seg = 0; seg < num_segments; ++seg) {
      const int base = seg * SEG;
      const int active = (extent - base < SEG) ? (extent - base) : SEG;
      T val = src[base * stride];
      if (lane < active)
        val = src[(base + lane) * stride];

#pragma unroll
      for (int off = 1; off < SEG; off <<= 1) {
        T n = tl::shfl_up_sync(MASK, val, off);
        if (lane >= off && lane < active)
          val = Reducer()(val, n);
      }

      if (has_carry && lane < active)
        val = Reducer()(val, carry);

      if (lane < active)
        dst[(base + lane) * stride] = val;

      const int last_lane = active - 1;
      T seg_result = tl::shfl_sync(MASK, val, last_lane);
      if (lane == last_lane)
        carry = seg_result;
      carry = tl::shfl_sync(MASK, carry, last_lane);
      has_carry = true;
    }
  }
}

template <class Reducer, int threads, bool reverse = false>
struct InclusiveScan1D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    if (threadIdx.x >= SEG)
      return;
    InclusiveScanLine<Reducer, reverse, T, SEG>(src, dst, N, 1);
  }
};

template <class Reducer, int threads, int Axis = 0, bool reverse = false>
struct InclusiveScan2D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  static_assert(Axis == 0 or Axis == 1);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W) {
    if (H <= 0 || W <= 0)
      return;

    constexpr int TILE = threads / SEG;
    const int item = threadIdx.x / SEG;

    if constexpr (Axis == 1) {
      const int num_blocks = (H + TILE - 1) / TILE;
      for (int b = 0; b < num_blocks; ++b) {
        const int row = b * TILE + item;
        if (row >= H)
          return;
        InclusiveScanLine<Reducer, reverse, T, SEG>(src + row * W,
                                                    dst + row * W, W, 1);
      }
    } else {
      const int num_blocks = (W + TILE - 1) / TILE;
      for (int b = 0; b < num_blocks; ++b) {
        const int col = b * TILE + item;
        if (col >= W)
          return;
        InclusiveScanLine<Reducer, reverse, T, SEG>(src + col, dst + col, H, W);
      }
    }
  }
};

template <int threads, bool reverse = false> struct CumSum1D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    InclusiveScan1D<ScanSumOp, threads, reverse>::template run<T, SEG>(src, dst,
                                                                       N);
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumSum2D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W) {
    InclusiveScan2D<ScanSumOp, threads, Axis, reverse>::template run<T, SEG>(
        src, dst, H, W);
  }
};

template <int threads, bool reverse = false> struct CumMax1D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    InclusiveScan1D<ScanMaxOp, threads, reverse>::template run<T, SEG>(src, dst,
                                                                       N);
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumMax2D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W) {
    InclusiveScan2D<ScanMaxOp, threads, Axis, reverse>::template run<T, SEG>(
        src, dst, H, W);
  }
};

} // namespace tl
