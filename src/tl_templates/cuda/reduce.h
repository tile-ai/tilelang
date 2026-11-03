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
    return tl::fast_max(x, y);
  }

  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x, bfloat16_t const &y) {
    return bfloat16_t(__hmax(x.to_nv_bfloat16(), y.to_nv_bfloat16()));
  }

  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmax(x.to_half(), y.to_half()));
  }
};
struct MaxOpNan {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::fast_max(x, y);
  }

  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x, bfloat16_t const &y) {
    return bfloat16_t(__hmax_nan(x.to_nv_bfloat16(), y.to_nv_bfloat16()));
  }

  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmax_nan(x.to_half(), y.to_half()));
  }
};

struct MinOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::fast_min(x, y);
  }

  TL_DEVICE cutlass::bfloat16_t operator()(bfloat16_t const &x,
                                           bfloat16_t const &y) {
    return bfloat16_t(__hmin(x.to_nv_bfloat16(), y.to_nv_bfloat16()));
  }

  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmin(x.to_half(), y.to_half()));
  }
};

struct MinOpNan {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::fast_min(x, y);
  }

  TL_DEVICE bfloat16_t operator()(bfloat16_t const &x, bfloat16_t const &y) {
    return bfloat16_t(__hmin_nan(x.to_nv_bfloat16(), y.to_nv_bfloat16()));
  }

  TL_DEVICE half_t operator()(half_t const &x, half_t const &y) {
    return half_t(__hmin_nan(x.to_half(), y.to_half()));
  }
};

struct SumOp_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::to_uint1(tl::add2(tl::from_uint1<__nv_bfloat162>(x),
                                 tl::from_uint1<__nv_bfloat162>(y)));
  }
};

struct MaxOp_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::to_uint1(tl::max2(tl::from_uint1<__nv_bfloat162>(x),
                                 tl::from_uint1<__nv_bfloat162>(y)));
  }
};

struct MinOp_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::to_uint1(tl::min2(tl::from_uint1<__nv_bfloat162>(x),
                                 tl::from_uint1<__nv_bfloat162>(y)));
  }
};

struct SumOp_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::to_uint1(
        tl::add2(tl::from_uint1<__half2>(x), tl::from_uint1<__half2>(y)));
  }
};

struct MaxOp_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::to_uint1(
        tl::max2(tl::from_uint1<__half2>(x), tl::from_uint1<__half2>(y)));
  }
};

struct MinOp_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::to_uint1(
        tl::min2(tl::from_uint1<__half2>(x), tl::from_uint1<__half2>(y)));
  }
};

struct MaxOpNan_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::to_uint1(tl::max2_nan(tl::from_uint1<__nv_bfloat162>(x),
                                     tl::from_uint1<__nv_bfloat162>(y)));
  }
};

struct MinOpNan_bf16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::to_uint1(tl::min2_nan(tl::from_uint1<__nv_bfloat162>(x),
                                     tl::from_uint1<__nv_bfloat162>(y)));
  }
};

struct MaxOpNan_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::to_uint1(
        tl::max2_nan(tl::from_uint1<__half2>(x), tl::from_uint1<__half2>(y)));
  }
};

struct MinOpNan_fp16x2 {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return tl::to_uint1(
        tl::min2_nan(tl::from_uint1<__half2>(x), tl::from_uint1<__half2>(y)));
  }
};

struct SumOp_f32x2 {
  TL_DEVICE float2 operator()(float2 const &x, float2 const &y) {
    return tl::add2(x, y);
  }
};

struct MaxOp_f32x2 {
  TL_DEVICE float2 operator()(float2 const &x, float2 const &y) {
    return tl::max2(x, y);
  }
};

struct MinOp_f32x2 {
  TL_DEVICE float2 operator()(float2 const &x, float2 const &y) {
    return tl::min2(x, y);
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
// Available on Ampere and later, where it permits partial-CTA synchronization.
template <int all_threads> struct NamedBarrier {
  template <int phase = 1> static TL_DEVICE void sync() {
    asm volatile("bar.sync %0, %1;" : : "r"(phase), "r"(all_threads));
  }
};

// AllReduce performs a cross-thread reduction over a group of `threads`
// threads.
//
// Template parameters:
//   Reducer         - binary reduction functor (e.g. SumOp, MaxOp).
//   threads         - number of threads that span the reduce dimension,
//                     equal to extent * scale.
//   scale           - stride of participating threads in the thread index
//                     space. When the thread-to-data mapping is normalized as
//                       threadIdx = source * scale + ...
//                     `scale` is the stride between consecutive logical
//                     participants in the reduce dimension.
//                     The recursion terminates when threads == scale, meaning
//                     each reduce group has been collapsed to a single thread.
//                     Uses a recursive XOR-butterfly pattern: at each level,
//                     offset >= 32 goes through shared memory + barrier,
//                     offset < 32 uses warp shuffle (shfl_xor_sync).
//   thread_offset   - base thread index offset within the block.
//   Barrier         - barrier policy type (SyncThreadsBarrier or
//                     NamedBarrier<N>).
//   batch_size      - number of independent values to reduce in parallel,
//                     sharing synchronization barriers across all values.
//                     Default 1 preserves the original scalar behaviour.
//   workspace_stride - stride between per-channel slices in the shared-memory
//                     workspace (typically total threads in the block).
//                     Only used when batch_size > 1.
template <class Reducer, int threads, int scale, int thread_offset = 0,
          class Barrier = SyncThreadsBarrier, int batch_size = 1,
          int workspace_stride = 0>
struct AllReduce {
  static_assert(threads > 0, "tl::AllReduce threads must be positive");
  static_assert(scale > 0, "tl::AllReduce scale must be positive");
  static_assert(threads % scale == 0,
                "tl::AllReduce threads must be divisible by scale");
  static_assert(((threads / scale) & (threads / scale - 1)) == 0,
                "AllReduce reduce width (threads / scale) must be a power of "
                "two");
  // t ^ (scale * k) only flips the reduce coordinate when the non-reduced
  // part of the thread index sits below `scale`, which needs scale to be a
  // power of two; otherwise the pairs straddle reduce groups.
  static_assert((scale & (scale - 1)) == 0,
                "AllReduce scale (thread stride between reduce participants) "
                "must be a power of two");

  // Scalar interface (backward-compatible).
  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return x;
    } else if constexpr (full_warp) {
      return warp_reduce(x, Reducer());
    } else {
      return butterfly_reduce_scalar(x, red_buf);
    }
  }

  // Batch interface (named run_batch to avoid overload-resolution ambiguity
  // with the scalar run(T x, T*) when a pointer is passed as the first arg).
  template <typename T>
  static TL_DEVICE void run_batch(T *x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return;
    } else if constexpr (full_warp) {
#pragma unroll
      for (int index = 0; index < batch_size; ++index) {
        x[index] = warp_reduce(x[index], Reducer());
      }
    } else {
      butterfly_reduce_batch(x, red_buf);
    }
  }

private:
  static constexpr bool full_warp =
      threads == 32 && scale == 1 && thread_offset % 32 == 0;
  static constexpr bool is_idempotent =
      std::is_same_v<Reducer, MaxOp> || std::is_same_v<Reducer, MinOp> ||
      std::is_same_v<Reducer, BitAndOp> || std::is_same_v<Reducer, BitOrOp>;
  static constexpr bool has_zero_identity =
      std::is_same_v<Reducer, SumOp> || std::is_same_v<Reducer, BitXorOp>;
  static constexpr int in_warp_scale = scale < 32 ? scale : 32;
  static constexpr int cross_scale = scale < 32 ? 32 : scale;
  static constexpr int cross_radix =
      threads > cross_scale ? (threads / cross_scale) : 1;
  static constexpr int in_warp_lanes = 32 / in_warp_scale;
  static constexpr int merge_lanes =
      cross_radix < in_warp_lanes ? cross_radix : in_warp_lanes;
  static constexpr int items_per_lane =
      cross_radix > merge_lanes ? (cross_radix / merge_lanes) : 1;

  using InWarp = AllReduce<Reducer, 32, in_warp_scale, thread_offset, Barrier,
                           batch_size, workspace_stride>;
  using CrossWarpInLane =
      AllReduce<Reducer, merge_lanes * in_warp_scale, in_warp_scale,
                thread_offset, Barrier, batch_size, workspace_stride>;
  using Next = AllReduce<Reducer, threads / 2, scale, thread_offset, Barrier,
                         batch_size, workspace_stride>;

  template <typename T>
  static TL_DEVICE T butterfly_reduce_scalar(T x, T *red_buf) {
    if constexpr (threads > 32) {
      const int rel_tid = threadIdx.x - thread_offset;
      Barrier::template sync<1>();
      if constexpr (scale < 32) {
        x = InWarp::run(x);
        red_buf[rel_tid] = x;
        Barrier::template sync<2>();
        const int lane_in_group = (rel_tid & 31) / in_warp_scale;
        const int warp_base =
            (lane_in_group & (merge_lanes - 1)) * items_per_lane;
        const int group_base = rel_tid & ~(threads - cross_scale);
        const int smem_base = group_base + warp_base * cross_scale;
        T acc = red_buf[smem_base];
#pragma unroll 4
        for (int k = 1; k < items_per_lane; ++k) {
          acc = Reducer()(acc, red_buf[smem_base + k * cross_scale]);
        }
        if constexpr (scale == 1 && thread_offset % 32 == 0 &&
                      sizeof(T) <= sizeof(float) &&
                      (is_idempotent ||
                       (has_zero_identity && std::is_integral_v<T>))) {
          if constexpr (!is_idempotent && merge_lanes < 32) {
            if (lane_in_group >= merge_lanes) {
              acc = static_cast<T>(0);
            }
          }
          return InWarp::run(acc);
        } else {
          return CrossWarpInLane::run(acc);
        }
      } else {
        red_buf[rel_tid] = x;
        Barrier::template sync<2>();
        const int base = rel_tid & ~(threads - cross_scale);
        T acc = red_buf[base];
#pragma unroll 4
        for (int k = 1; k < cross_radix; ++k) {
          acc = Reducer()(acc, red_buf[base + k * cross_scale]);
        }
        return acc;
      }
    } else {
      constexpr int offset = threads / 2;
      x = Reducer()(x, tl::shfl_xor_sync(uint32_t(-1), x, offset));
      if constexpr (offset == scale) {
        return x;
      } else {
        return Next::run(x, red_buf);
      }
    }
  }

  template <typename T>
  static TL_DEVICE void butterfly_reduce_batch(T *x, T *red_buf) {
    if constexpr (threads > 32) {
      const int rel_tid = threadIdx.x - thread_offset;
      Barrier::template sync<1>();
      if constexpr (scale < 32) {
        InWarp::run_batch(x);
#pragma unroll
        for (int i = 0; i < batch_size; i++) {
          red_buf[rel_tid + i * workspace_stride] = x[i];
        }
        Barrier::template sync<2>();
        const int lane_in_group = (rel_tid & 31) / in_warp_scale;
        const int warp_base =
            (lane_in_group & (merge_lanes - 1)) * items_per_lane;
        const int group_base = rel_tid & ~(threads - cross_scale);
        const int smem_base = group_base + warp_base * cross_scale;
#pragma unroll
        for (int i = 0; i < batch_size; i++) {
          T acc = red_buf[smem_base + i * workspace_stride];
#pragma unroll 4
          for (int k = 1; k < items_per_lane; ++k) {
            acc = Reducer()(
                acc,
                red_buf[smem_base + k * cross_scale + i * workspace_stride]);
          }
          if constexpr (scale == 1 && thread_offset % 32 == 0 &&
                        sizeof(T) <= sizeof(float) && !is_idempotent &&
                        has_zero_identity && std::is_integral_v<T> &&
                        merge_lanes < 32) {
            if (lane_in_group >= merge_lanes) {
              acc = static_cast<T>(0);
            }
          }
          x[i] = acc;
        }
        if constexpr (scale == 1 && thread_offset % 32 == 0 &&
                      sizeof(T) <= sizeof(float) &&
                      (is_idempotent ||
                       (has_zero_identity && std::is_integral_v<T>))) {
          InWarp::run_batch(x);
        } else {
          CrossWarpInLane::run_batch(x);
        }
      } else {
#pragma unroll
        for (int i = 0; i < batch_size; i++) {
          red_buf[rel_tid + i * workspace_stride] = x[i];
        }
        Barrier::template sync<2>();
        const int base = rel_tid & ~(threads - cross_scale);
#pragma unroll
        for (int i = 0; i < batch_size; i++) {
          T acc = red_buf[base + i * workspace_stride];
#pragma unroll 4
          for (int k = 1; k < cross_radix; ++k) {
            acc = Reducer()(
                acc, red_buf[base + k * cross_scale + i * workspace_stride]);
          }
          x[i] = acc;
        }
      }
    } else {
      constexpr int offset = threads / 2;
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        x[i] = Reducer()(x[i], tl::shfl_xor_sync(uint32_t(-1), x[i], offset));
      }
      if constexpr (offset == scale) {
        return;
      } else {
        Next::run_batch(x, red_buf);
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
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) ||                                    \
     defined(__CUDA_ARCH_FEAT_SM103_ALL) ||                                    \
     (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) &&                                \
      (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 ||                                \
       __CUDA_ARCH_FAMILY_SPECIFIC__ == 1030)))
  if constexpr (!std::is_integral_v<T> && sizeof(T) <= sizeof(float) &&
                (std::is_same_v<ReduceOp, MaxOp> ||
                 std::is_same_v<ReduceOp, MinOp>)) {
    float value_cast = static_cast<float>(value);
    float result;
    if constexpr (std::is_same_v<ReduceOp, MaxOp>) {
      asm("redux.sync.max.f32 %0, %1, %2;"
          : "=f"(result)
          : "f"(value_cast), "r"(mask));
    } else {
      asm("redux.sync.min.f32 %0, %1, %2;"
          : "=f"(result)
          : "f"(value_cast), "r"(mask));
    }
    return static_cast<T>(result);
  }
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  if constexpr (std::is_integral_v<T> && sizeof(T) <= 4) {
    using ReduceType =
        std::conditional_t<std::is_same_v<T, uint32_t>, uint32_t, int32_t>;
    ReduceType value_cast = static_cast<ReduceType>(value);
    if constexpr (std::is_same_v<ReduceOp, SumOp>) {
      return static_cast<T>(__reduce_add_sync(mask, value_cast));
    } else if constexpr (std::is_same_v<ReduceOp, MaxOp>) {
      return static_cast<T>(__reduce_max_sync(mask, value_cast));
    } else if constexpr (std::is_same_v<ReduceOp, MinOp>) {
      return static_cast<T>(__reduce_min_sync(mask, value_cast));
    } else if constexpr (std::is_same_v<ReduceOp, BitAndOp>) {
      return static_cast<T>(__reduce_and_sync(mask, value_cast));
    } else if constexpr (std::is_same_v<ReduceOp, BitOrOp>) {
      return static_cast<T>(__reduce_or_sync(mask, value_cast));
    } else if constexpr (std::is_same_v<ReduceOp, BitXorOp>) {
      return static_cast<T>(__reduce_xor_sync(mask, value_cast));
    }
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
