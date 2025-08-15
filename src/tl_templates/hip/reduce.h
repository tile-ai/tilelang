#pragma once

#include "common.h"

/**
   * Binary sum reduction operator usable in device code.
   *
   * Returns the arithmetic sum of two values.
   *
   * @tparam T value type
   * @param x left operand
   * @param y right operand
   * @return x + y
   */
  
  /**
   * Binary maximum reduction operator usable in device code.
   *
   * Returns the larger of the two values using ck_tile::max.
   *
   * @tparam T value type
   * @param x left operand
   * @param y right operand
   * @return maximum of x and y
   */
  
  /**
   * Binary minimum reduction operator usable in device code.
   *
   * Returns the smaller of the two values using ck_tile::min.
   *
   * @tparam T value type
   * @param x left operand
   * @param y right operand
   * @return minimum of x and y
   */
  
  /**
   * Perform a hierarchical, compile-time-configured all-reduce across threads.
   *
   * Combines a per-thread value `x` across `threads` threads using the binary
   * `Reducer`. Reduction proceeds in stages: when the current half-width
   * (threads/2) is greater than or equal to the warp size (64), values are
   * exchanged via the provided shared-memory buffer `red_buf` with __syncthreads
   * barriers; otherwise a warp-level shuffle (__shfl_xor) is used. The routine
   * recurses, halving the active thread count each stage until the number of
   * threads equals `scale`.
   *
   * Compile-time requirements (enforced via static_assert):
   * - `threads` must be one of: 1024, 512, 256, 128, 64, 32, 16, 8, 4, or 2.
   * - `threads % scale == 0`.
   *
   * @tparam Reducer binary functor type providing T operator()(T const&, T const&)
   * @tparam threads initial number of threads participating in the reduction
   * @tparam scale final number of threads to stop at (reduction target)
   * @tparam thread_offset unused template parameter reserved for caller-specific offsets
   * @tparam T value type
   * @param x per-thread input value (accumulates partial reductions)
   * @param red_buf pointer to a shared-memory buffer used when stage width >= warp size;
   *                must be non-null when that path is taken
   * @return reduced value for this thread after completion (for the thread that ends
   *         up holding the combined result at the final stage)
   */
  namespace tl {

struct SumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }
};

struct MaxOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return ck_tile::max(x, y);
  }
};

struct MinOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return ck_tile::min(x, y);
  }
};

template <class Reducer, int threads, int scale, int thread_offset = 0>
struct AllReduce {
  static_assert(threads == 1024 || threads == 512 || threads == 256 ||
                threads == 128 || threads == 64 || threads == 32 ||
                threads == 16 || threads == 8 || threads == 4 || threads == 2);
  static_assert(threads % scale == 0);

  template <typename T> static __device__ T run(T x, T *red_buf = nullptr) {
    constexpr int offset = threads / 2;
    constexpr int warpSize = 64;

    if constexpr (offset >= warpSize) {
      __syncthreads();
      red_buf[threadIdx.x] = x;
      __syncthreads();
      x = Reducer()(x, red_buf[threadIdx.x ^ offset]);
    } else {
      x = Reducer()(x, __shfl_xor(x, offset));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale, thread_offset>::run(x, red_buf);
    }
  }
};

} // namespace tl
