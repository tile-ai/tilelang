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
struct MaxOpNan {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return cutlass::fast_max(x, y);
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

struct MinOpNan {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return cutlass::fast_min(x, y);
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
  static_assert(threads % scale == 0);

  // Scalar interface (backward-compatible).
  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return x;
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
    } else {
      butterfly_reduce_batch(x, red_buf);
    }
  }

private:
  using Next = AllReduce<Reducer, threads / 2, scale, thread_offset, Barrier,
                         batch_size, workspace_stride>;

  template <typename T>
  static TL_DEVICE T butterfly_reduce_scalar(T x, T *red_buf) {
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
      return Next::run(x, red_buf);
    }
  }

  template <typename T>
  static TL_DEVICE void butterfly_reduce_batch(T *x, T *red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      Barrier::template sync<1>();
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        red_buf[(threadIdx.x - thread_offset) + i * workspace_stride] = x[i];
      }
      Barrier::template sync<2>();
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        x[i] =
            Reducer()(x[i], red_buf[((threadIdx.x - thread_offset) ^ offset) +
                                    i * workspace_stride]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        x[i] = Reducer()(x[i], tl::shfl_xor_sync(uint32_t(-1), x[i], offset));
      }
    }
    if constexpr (offset == scale) {
      return;
    } else {
      Next::run_batch(x, red_buf);
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
