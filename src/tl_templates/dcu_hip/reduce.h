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
// Detect half types
template <typename T>
struct is_half_type : std::false_type {};

template <>
struct is_half_type<__half> : std::true_type {};

template <>
struct is_half_type<_Float16> : std::true_type {};

template <typename T>
inline constexpr bool is_half_v = is_half_type<std::decay_t<T>>::value;

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
      if constexpr (is_half_v<T>) {
        unsigned short x_raw;
        if constexpr (std::is_same_v<std::decay_t<T>, __half>) {
          x_raw = __half_as_ushort(x);
        } else { // _Float16
          union { _Float16 f; unsigned short s; } u;
          u.f = x;
          x_raw = u.s;
        }

        unsigned short shuffled_raw = __shfl_xor(x_raw, offset);

        T shuffled_x;
        if constexpr (std::is_same_v<std::decay_t<T>, __half>) {
          shuffled_x = __ushort_as_half(shuffled_raw);
        } else { // _Float16
          union { unsigned short s; _Float16 f; } u;
          u.s = shuffled_raw;
          shuffled_x = u.f;
        }

        x = Reducer()(x, shuffled_x);
      } else {
        x = Reducer()(x, __shfl_xor(x, offset));
      }
    }

    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale, thread_offset>::run(x, red_buf);
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
    constexpr uint64_t MASK = 0xffffffffffffffffULL;
    const int num_blocks = (H + TILE_H - 1) / TILE_H;
    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int row = tid / 64;

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

