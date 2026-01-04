#pragma once

#include "common.h"

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

struct BitAndOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x & y;
  }
};

struct BitOrOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x | y;
  }
};

struct BitXorOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x ^ y;
  }
};

template <class Reducer, int Threads, bool UseAbs, bool NeedAccumulate>
struct SharedReduceWarp {
  template <typename T>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int total_dest, int reduce_extent, int tail,
                            T init_value) {
    if (total_dest <= 0 || reduce_extent <= 0)
      return;
    constexpr int kWarpSize = 64;
    static_assert(Threads % kWarpSize == 0,
                  "SharedReduceWarp expects blockDim.x to be a multiple of "
                  "wave size on HIP.");
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int num_warps = Threads / kWarpSize;

    for (int dest_idx = warp_id; dest_idx < total_dest; dest_idx += num_warps) {
      const int prefix = tail == 1 ? dest_idx : dest_idx / tail;
      const int suffix = tail == 1 ? 0 : dest_idx % tail;
      const int src_base = (prefix * reduce_extent) * tail + suffix;
      const int dst_index = prefix * tail + suffix;

      T partial = init_value;
      for (int rv = lane; rv < reduce_extent; rv += kWarpSize) {
        T val = src[src_base + rv * tail];
        if constexpr (UseAbs) {
          val = val < T(0) ? -val : val;
        }
        partial = Reducer()(partial, val);
      }

      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        T other = tl::shfl_down(partial, offset, kWarpSize);
        partial = Reducer()(partial, other);
      }

      if (lane == 0) {
        if constexpr (NeedAccumulate) {
          partial = Reducer()(dst[dst_index], partial);
        }
        dst[dst_index] = partial;
      }
    }
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
      x = Reducer()(x, tl::shfl_xor(x, offset));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale, thread_offset>::run(x, red_buf);
    }
  }
};

template <int threads, bool reverse = false> struct CumSum1D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64);
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    if (N <= 0)
      return;

    const int tid = threadIdx.x;
    const int lane = tid % SEG;

    if (tid >= SEG)
      return;

    T carry = (T)0;

    if (reverse) {
      const int num_segments = (N + SEG - 1) / SEG;
      for (int seg = num_segments - 1; seg >= 0; --seg) {
        const int idx = seg * SEG + lane;
        T val = (idx < N) ? src[idx] : (T)0;

#pragma unroll
        for (int off = 1; off < SEG; off <<= 1) {
          T n = tl::shfl_down(val, off);
          if (lane < SEG - off)
            val += n;
        }

        val += carry;

        if (idx < N)
          dst[idx] = val;

        T segSum = tl::shfl(val, 0);
        if (lane == 0)
          carry = segSum;
        carry = tl::shfl(carry, 0);
      }
    } else {
      const int num_segments = (N + SEG - 1) / SEG;
      for (int seg = 0; seg < num_segments; ++seg) {
        const int idx = seg * SEG + lane;
        T val = (idx < N) ? src[idx] : (T)0;

#pragma unroll
        for (int off = 1; off < SEG; off <<= 1) {
          T n = tl::shfl_up(val, off);
          if (lane >= off)
            val += n;
        }

        val += carry;

        if (idx < N)
          dst[idx] = val;

        T segSum = tl::shfl(val, SEG - 1);
        if (lane == SEG - 1)
          carry = segSum;
        carry = tl::shfl(carry, SEG - 1);
      }
    }
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumSum2D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64);
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W) {

    constexpr int TILE_H = threads / SEG;
    const int num_blocks = (H + TILE_H - 1) / TILE_H;
    const int tid = threadIdx.x;
    const int lane = tid % SEG;
    const int row = tid / SEG;

    for (int b = 0; b < num_blocks; ++b) {
      const int gRow = b * TILE_H + row;
      if (gRow >= H)
        return;

      T carry = (T)0;

      if (reverse) {
        // Start from the last segment for reverse mode
        for (int seg = (W + SEG - 1) / SEG - 1; seg >= 0; --seg) {
          const int col = seg * SEG + lane;

          const int real_row = Axis == 1 ? gRow : col;
          const int real_col = Axis == 1 ? col : gRow;

          T val = (col < W) ? src[real_row * W + real_col] : (T)0;

#pragma unroll
          for (int off = 1; off < SEG; off <<= 1) {
            T n = tl::shfl_down(val, off);
            if (lane < SEG - off)
              val += n;
          }

          val += carry;

          if (real_col < W)
            dst[real_row * W + real_col] = val;

          T segSum = tl::shfl(val, 0);
          if (lane == 0)
            carry = segSum;
          carry = tl::shfl(carry, 0);
        }
      } else {
        for (int seg = 0; seg * SEG < W; ++seg) {
          const int col = seg * SEG + lane;

          const int real_row = Axis == 1 ? gRow : col;
          const int real_col = Axis == 1 ? col : gRow;

          T val = (col < W) ? src[real_row * W + real_col] : (T)0;

#pragma unroll
          for (int off = 1; off < SEG; off <<= 1) {
            T n = tl::shfl_up(val, off);
            if (lane >= off)
              val += n;
          }

          val += carry;

          if (real_col < W)
            dst[real_row * W + real_col] = val;

          T segSum = tl::shfl(val, SEG - 1);
          if (lane == SEG - 1)
            carry = segSum;
          carry = tl::shfl(carry, SEG - 1);
        }
      }
    }
  }
};

template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op) {
  value = op(value, __shfl_xor(value, 32));
  value = op(value, __shfl_xor(value, 16));
  value = op(value, __shfl_xor(value, 8));
  value = op(value, __shfl_xor(value, 4));
  value = op(value, __shfl_xor(value, 2));
  value = op(value, __shfl_xor(value, 1));
  return value;
}

template <typename T> TL_DEVICE T warp_reduce_sum(T value) {
  return warp_reduce<T>(value, SumOp());
}

template <typename T> TL_DEVICE T warp_reduce_max(T value) {
  return warp_reduce<T>(value, MaxOp());
}

template <typename T> TL_DEVICE T warp_reduce_min(T value) {
  return warp_reduce<T>(value, MinOp());
}

template <typename T> TL_DEVICE T warp_reduce_bitand(T value) {
  return warp_reduce<T>(value, BitAndOp());
}

template <typename T> TL_DEVICE T warp_reduce_bitor(T value) {
  return warp_reduce<T>(value, BitOrOp());
}

} // namespace tl
