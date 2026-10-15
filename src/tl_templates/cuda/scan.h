#pragma once

#include "common.h"
#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tl {

struct ScanSumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }

  template <typename T> TL_DEVICE static T identity() { return T(0); }
};

struct ScanMaxOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::fast_max(x, y);
  }

  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x, bfloat16_t const &y) {
    return bfloat16_t(__hmax(x.to_nv_bfloat16(), y.to_nv_bfloat16()));
  }

  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmax(x.to_half(), y.to_half()));
  }

  template <typename T> TL_DEVICE static T identity() {
    // cuda::std::numeric_limits covers the builtin types; cutlass extended
    // types (half_t, bfloat16_t, fp8) fall through to cutlass's limits.
    if constexpr (cuda::std::numeric_limits<T>::is_specialized) {
      if constexpr (cuda::std::numeric_limits<T>::has_infinity) {
        return -cuda::std::numeric_limits<T>::infinity();
      } else {
        return cuda::std::numeric_limits<T>::lowest();
      }
    } else if constexpr (cutlass::platform::numeric_limits<T>::has_infinity) {
      return -cutlass::platform::numeric_limits<T>::infinity();
    } else {
      return cutlass::platform::numeric_limits<T>::lowest();
    }
  }
};

// Out-of-range lanes are padded with the reducer's identity rather than
// masked with per-lane bound predicates: a predicate chain inside the shuffle
// loop (and the variable-lane carry broadcast it requires) blocks nvcc from
// software-pipelining adjacent segments, costing ~1.5x on multi-segment lines.
template <class Reducer, bool reverse, typename T, int SEG = 32>
static TL_DEVICE void InclusiveScanLine(const T *__restrict__ src,
                                        T *__restrict__ dst, int extent,
                                        int src_stride, int dst_stride) {
  if (extent <= 0)
    return;

  constexpr unsigned MASK = 0xffffffff;
  const int lane = threadIdx.x % SEG;
  T carry = Reducer::template identity<T>();
  const int num_segments = (extent + SEG - 1) / SEG;

  if constexpr (reverse) {
    for (int seg = num_segments - 1; seg >= 0; --seg) {
      const int idx = seg * SEG + lane;
      T val = (idx < extent) ? src[idx * src_stride]
                             : Reducer::template identity<T>();

#pragma unroll
      for (int off = 1; off < SEG; off <<= 1) {
        T n = tl::shfl_down_sync(MASK, val, off);
        if (lane < SEG - off)
          val = Reducer()(val, n);
      }

      val = Reducer()(val, carry);

      if (idx < extent)
        dst[idx * dst_stride] = val;

      carry = tl::shfl_sync(MASK, val, 0);
    }
  } else {
    for (int seg = 0; seg < num_segments; ++seg) {
      const int idx = seg * SEG + lane;
      T val = (idx < extent) ? src[idx * src_stride]
                             : Reducer::template identity<T>();

#pragma unroll
      for (int off = 1; off < SEG; off <<= 1) {
        T n = tl::shfl_up_sync(MASK, val, off);
        if (lane >= off)
          val = Reducer()(val, n);
      }

      val = Reducer()(val, carry);

      if (idx < extent)
        dst[idx * dst_stride] = val;

      carry = tl::shfl_sync(MASK, val, SEG - 1);
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
    InclusiveScanLine<Reducer, reverse, T, SEG>(src, dst, N, 1, 1);
  }
};

template <class Reducer, int threads, int Axis = 0, bool reverse = false>
struct InclusiveScan2D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  static_assert(Axis == 0 or Axis == 1);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W, int src_stride, int dst_stride) {
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
        InclusiveScanLine<Reducer, reverse, T, SEG>(
            src + row * src_stride, dst + row * dst_stride, W, 1, 1);
      }
    } else {
      const int num_blocks = (W + TILE - 1) / TILE;
      for (int b = 0; b < num_blocks; ++b) {
        const int col = b * TILE + item;
        if (col >= W)
          return;
        InclusiveScanLine<Reducer, reverse, T, SEG>(src + col, dst + col, H,
                                                    src_stride, dst_stride);
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
                            int H, int W, int src_stride, int dst_stride) {
    InclusiveScan2D<ScanSumOp, threads, Axis, reverse>::template run<T, SEG>(
        src, dst, H, W, src_stride, dst_stride);
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
                            int H, int W, int src_stride, int dst_stride) {
    InclusiveScan2D<ScanMaxOp, threads, Axis, reverse>::template run<T, SEG>(
        src, dst, H, W, src_stride, dst_stride);
  }
};

} // namespace tl
