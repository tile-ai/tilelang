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
        T other = __shfl_down(partial, offset, kWarpSize);
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
