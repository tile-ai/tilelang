// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
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
    return cutlass::fast_max(x, y);
  }
};

struct MinOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return cutlass::fast_min(x, y);
  }
};

template <class Reducer, int threads, int scale, int all_threads = threads>
struct AllReduce {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32 or
                threads == 16 or threads == 8 or threads == 4 or threads == 2);
  static_assert(threads % scale == 0);
  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      __syncthreads();
      red_buf[threadIdx.x] = x;
      __syncthreads();
      x = Reducer()(x, red_buf[threadIdx.x ^ offset]);
    } else {
      x = Reducer()(x, T(__shfl_xor_sync(uint32_t(-1), x, offset)));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale>::run(x, red_buf);
    }
  }

  template <typename T>
  static TL_DEVICE T run_hopper(T x, T *red_buf = nullptr) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(all_threads));
      red_buf[threadIdx.x] = x;
      asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(all_threads));
      x = Reducer()(x, red_buf[threadIdx.x ^ offset]);
    } else {
      x = Reducer()(x, T(__shfl_xor_sync(uint32_t(-1), x, offset)));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale, all_threads>::run_hopper(
          x, red_buf);
    }
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumSum2D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  template <typename T, int SEG = 32>
  static TL_DEVICE T run(const T *__restrict__ src, T *__restrict__ dst, int H,
                         int W) {

    constexpr int TILE_H = threads / SEG;
    constexpr unsigned MASK = 0xffffffff;
    const int num_blocks = (H + TILE_H - 1) / TILE_H;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int row = tid / 32;

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
            T n = (T)__shfl_down_sync(MASK, val, off);
            if (lane < SEG - off)
              val += n;
          }

          val += carry;

          if (real_col < W)
            dst[real_row * W + real_col] = val;

          T segSum = (T)__shfl_sync(MASK, val, (T)0);
          if (lane == 0)
            carry = segSum;
          carry = (T)__shfl_sync(MASK, carry, (T)0);
        }
      } else {
        for (int seg = 0; seg * SEG < W; ++seg) {
          const int col = seg * SEG + lane;

          const int real_row = Axis == 1 ? gRow : col;
          const int real_col = Axis == 1 ? col : gRow;

          T val = (col < W) ? src[real_row * W + real_col] : (T)0;

#pragma unroll
          for (int off = 1; off < SEG; off <<= 1) {
            T n = (T)__shfl_up_sync(MASK, val, off);
            if (lane >= off)
              val += n;
          }

          val += carry;

          if (real_col < W)
            dst[real_row * W + real_col] = val;

          T segSum = (T)__shfl_sync(MASK, val, SEG - 1);
          if (lane == SEG - 1)
            carry = segSum;
          carry = (T)__shfl_sync(MASK, carry, SEG - 1);
        }
      }
    }
  }
};

} // namespace tl
