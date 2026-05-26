#pragma once

#include "common.h"

namespace tl {

struct ScanSumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }
};

struct ScanMaxOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return ck_tile::max(x, y);
  }
};

template <class Reducer, bool reverse, typename T, int SEG = 64>
static TL_DEVICE void InclusiveScanLine(const T *__restrict__ src,
                                        T *__restrict__ dst, int extent,
                                        int stride) {
  if (extent <= 0)
    return;

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
        T n = tl::shfl_down(val, off);
        if (lane + off < active)
          val = Reducer()(val, n);
      }

      if (has_carry && lane < active)
        val = Reducer()(val, carry);

      if (lane < active)
        dst[(base + lane) * stride] = val;

      T seg_result = tl::shfl(val, 0);
      if (lane == 0)
        carry = seg_result;
      carry = tl::shfl(carry, 0);
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
        T n = tl::shfl_up(val, off);
        if (lane >= off && lane < active)
          val = Reducer()(val, n);
      }

      if (has_carry && lane < active)
        val = Reducer()(val, carry);

      if (lane < active)
        dst[(base + lane) * stride] = val;

      const int last_lane = active - 1;
      T seg_result = tl::shfl(val, last_lane);
      if (lane == last_lane)
        carry = seg_result;
      carry = tl::shfl(carry, last_lane);
      has_carry = true;
    }
  }
}

template <class Reducer, int threads, bool reverse = false>
struct InclusiveScan1D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64);
  template <typename T, int SEG = 64>
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
                threads == 128 or threads == 64);
  static_assert(Axis == 0 or Axis == 1);
  template <typename T, int SEG = 64>
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
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    InclusiveScan1D<ScanSumOp, threads, reverse>::template run<T, SEG>(src, dst,
                                                                       N);
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumSum2D {
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W) {
    InclusiveScan2D<ScanSumOp, threads, Axis, reverse>::template run<T, SEG>(
        src, dst, H, W);
  }
};

template <int threads, bool reverse = false> struct CumMax1D {
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    InclusiveScan1D<ScanMaxOp, threads, reverse>::template run<T, SEG>(src, dst,
                                                                       N);
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumMax2D {
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W) {
    InclusiveScan2D<ScanMaxOp, threads, Axis, reverse>::template run<T, SEG>(
        src, dst, H, W);
  }
};

} // namespace tl
