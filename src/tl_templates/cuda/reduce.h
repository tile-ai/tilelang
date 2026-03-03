#pragma once

#include "common.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef __CUDACC_RTC__
#include <cstdint>
#include <type_traits>
#endif

namespace tl {

template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op);

// Select a wider accumulator type for improved numerical accuracy.
// Default: accumulate in the same type. Specialize FP16/BF16 to float.
template <typename T> struct AccType {
  using type = T;
};
template <> struct AccType<half_t> {
  using type = float;
};
template <> struct AccType<bfloat16_t> {
  using type = float;
};

struct SumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }
};

struct MaxOp {
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

struct MinOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return cutlass::fast_min(x, y);
  }

  TL_DEVICE cutlass::bfloat16_t operator()(bfloat16_t const &x,
                                           bfloat16_t const &y) {
    return bfloat16_t(__hmin(x.to_nv_bfloat16(), y.to_nv_bfloat16()));
  }

  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmin(x.to_half(), y.to_half()));
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

// Barrier policy: wraps __syncthreads().
// The phase template parameter is ignored (all phases use the same barrier).
struct SyncThreadsBarrier {
  template <int phase = 0> static TL_DEVICE void sync() { __syncthreads(); }
};

// Barrier policy: wraps named barrier (bar.sync) with compile-time phase IDs.
// Used on Hopper and later architectures where __syncthreads() cannot be used
// in certain contexts.
template <int all_threads> struct NamedBarrier {
  template <int phase = 1> static TL_DEVICE void sync() {
    asm volatile("bar.sync %0, %1;" : : "r"(phase), "r"(all_threads));
  }
};

// AllReduce performs a cross-thread reduction over a group of `threads`
// threads.
//
// Template parameters:
//   Reducer       - binary reduction functor (e.g. SumOp, MaxOp).
//   threads       - number of threads that span the reduce dimension,
//                   equal to extent * scale.
//   scale         - stride of participating threads in the thread index space.
//                   When the thread-to-data mapping is normalized as
//                     threadIdx = source * scale + ...
//                   `scale` is the stride between consecutive logical
//                   participants in the reduce dimension.
//                   The recursion terminates when threads == scale, meaning
//                   each reduce group has been collapsed to a single thread.
//                   Uses a recursive XOR-butterfly pattern: at each level,
//                   offset >= 32 goes through shared memory + barrier,
//                   offset < 32 uses warp shuffle (shfl_xor_sync).
//   thread_offset - base thread index offset within the block.
//   Barrier       - barrier policy type (SyncThreadsBarrier or
//                   NamedBarrier<N>).
template <class Reducer, int threads, int scale, int thread_offset = 0,
          class Barrier = SyncThreadsBarrier>
struct AllReduce {
  static_assert(threads % scale == 0);
  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      // Recursion base case: each reduce group has exactly one thread left.
      return x;
    } else {
      return butterfly_reduce(x, red_buf);
    }
  }

private:
  template <typename T> static TL_DEVICE T butterfly_reduce(T x, T *red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      Barrier::template sync<1>();
      red_buf[threadIdx.x - thread_offset] = x;
      Barrier::template sync<2>();
      x = Reducer()(x, red_buf[(threadIdx.x - thread_offset) ^ offset]);
    } else {
      x = Reducer()(x, tl::shfl_xor_sync(uint32_t(-1), x, offset));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale, thread_offset, Barrier>::run(
          x, red_buf);
    }
  }
};

template <int threads, bool reverse = false> struct CumSum1D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    if (N <= 0)
      return;

    constexpr unsigned MASK = 0xffffffff;
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
          T n = (T)tl::shfl_down_sync(MASK, val, off);
          if (lane < SEG - off)
            val += n;
        }

        val += carry;

        if (idx < N)
          dst[idx] = val;

        T segSum = (T)__shfl_sync(MASK, val, 0);
        if (lane == 0)
          carry = segSum;
        carry = (T)__shfl_sync(MASK, carry, 0);
      }
    } else {
      const int num_segments = (N + SEG - 1) / SEG;
      for (int seg = 0; seg < num_segments; ++seg) {
        const int idx = seg * SEG + lane;
        T val = (idx < N) ? src[idx] : (T)0;

#pragma unroll
        for (int off = 1; off < SEG; off <<= 1) {
          T n = (T)__shfl_up_sync(MASK, val, off);
          if (lane >= off)
            val += n;
        }

        val += carry;

        if (idx < N)
          dst[idx] = val;

        T segSum = (T)__shfl_sync(MASK, val, SEG - 1);
        if (lane == SEG - 1)
          carry = segSum;
        carry = (T)__shfl_sync(MASK, carry, SEG - 1);
      }
    }
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumSum2D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W) {

    constexpr int TILE_H = threads / SEG;
    constexpr unsigned MASK = 0xffffffff;
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
            T n = tl::shfl_down_sync(MASK, val, off);
            if (lane < SEG - off)
              val += n;
          }

          val += carry;

          if (real_col < W)
            dst[real_row * W + real_col] = val;

          T segSum = tl::shfl_sync(MASK, val, 0);
          if (lane == 0)
            carry = segSum;
          carry = tl::shfl_sync(MASK, carry, 0);
        }
      } else {
        for (int seg = 0; seg * SEG < W; ++seg) {
          const int col = seg * SEG + lane;

          const int real_row = Axis == 1 ? gRow : col;
          const int real_col = Axis == 1 ? col : gRow;

          T val = (col < W) ? src[real_row * W + real_col] : (T)0;

#pragma unroll
          for (int off = 1; off < SEG; off <<= 1) {
            T n = tl::shfl_up_sync(MASK, val, off);
            if (lane >= off)
              val += n;
          }

          val += carry;

          if (real_col < W)
            dst[real_row * W + real_col] = val;

          T segSum = tl::shfl_sync(MASK, val, SEG - 1);
          if (lane == SEG - 1)
            carry = segSum;
          carry = tl::shfl_sync(MASK, carry, SEG - 1);
        }
      }
    }
  }
};

// Reference:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#reduction
template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op) {
  constexpr uint32_t mask = 0xffffffff;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) &&                       \
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM100_F))
  float value_cast = 0.0f;
  if constexpr (std::is_same_v<T, half_t>) {
    value_cast = __half2float(value);
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    value_cast = __bfloat162float(value);
  } else {
    value_cast = static_cast<float>(value);
  }
  if constexpr (std::is_same_v<ReduceOp, MaxOp> && !std::is_integral_v<T>) {
    float res;
    asm("redux.sync.max.f32 %0, %1, %2;"
        : "=f"(res)
        : "f"(value_cast), "r"(mask));
    return static_cast<T>(res);
  } else if constexpr (std::is_same_v<ReduceOp, MinOp> &&
                       !std::is_integral_v<T>) {
    float res;
    asm("redux.sync.min.f32 %0, %1, %2;"
        : "=f"(res)
        : "f"(value_cast), "r"(mask));
    return static_cast<T>(res);
  }
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  auto run_reduce_sync = [&]<typename T_cast>(T_cast val) {
    if constexpr (std::is_same_v<ReduceOp, SumOp>) {
      return __reduce_add_sync(mask, val);
    } else if constexpr (std::is_same_v<ReduceOp, MaxOp>) {
      return __reduce_max_sync(mask, val);
    } else if constexpr (std::is_same_v<ReduceOp, MinOp>) {
      return __reduce_min_sync(mask, val);
    } else if constexpr (std::is_same_v<ReduceOp, BitAndOp>) {
      return __reduce_and_sync(mask, val);
    } else if constexpr (std::is_same_v<ReduceOp, BitOrOp>) {
      return __reduce_or_sync(mask, val);
    } else if constexpr (std::is_same_v<ReduceOp, BitXorOp>) {
      return __reduce_xor_sync(mask, val);
    }
  };

  if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
    return run_reduce_sync(value);
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(run_reduce_sync(static_cast<int32_t>(value)));
  }
#endif
  value = op(value, tl::shfl_xor_sync(mask, value, 16));
  value = op(value, tl::shfl_xor_sync(mask, value, 8));
  value = op(value, tl::shfl_xor_sync(mask, value, 4));
  value = op(value, tl::shfl_xor_sync(mask, value, 2));
  value = op(value, tl::shfl_xor_sync(mask, value, 1));
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
