#pragma once

#include "simt_api/asc_simt.h"

#ifndef HUGE_VALF
#define HUGE_VALF __builtin_huge_valf()
#endif

namespace tl {

// ─── Reduction operators ────────────────────────────────────────────────────

struct SumOp {
  template <typename T> __simt_callee__ T operator()(T x, T y) { return x + y; }
};
struct MaxOp {
  template <typename T> __simt_callee__ T operator()(T x, T y) {
    return x > y ? x : y;
  }
};
struct MinOp {
  template <typename T> __simt_callee__ T operator()(T x, T y) {
    return x < y ? x : y;
  }
};

// ─── ShflXor: warp shuffle intrinsic, defined only for natively supported T ─
// No primary template → substitution failure for unsupported types.

template <typename T, typename = void> struct ShflXor;

template <> struct ShflXor<float> {
  static __simt_callee__ float call(float v, int o) {
    return asc_shfl_xor(v, o);
  }
};
template <> struct ShflXor<int32_t> {
  static __simt_callee__ int32_t call(int32_t v, int o) {
    return asc_shfl_xor(v, o);
  }
};
template <> struct ShflXor<uint32_t> {
  static __simt_callee__ uint32_t call(uint32_t v, int o) {
    return asc_shfl_xor(v, o);
  }
};
template <> struct ShflXor<int64_t> {
  static __simt_callee__ int64_t call(int64_t v, int o) {
    return asc_shfl_xor(v, o);
  }
};
template <> struct ShflXor<uint64_t> {
  static __simt_callee__ uint64_t call(uint64_t v, int o) {
    return asc_shfl_xor(v, o);
  }
};

// ─── HwReduce: hardware warp-reduce, defined per (Reducer, T) ───────────────

template <class Reducer, typename T, typename = void> struct HwReduce;

template <> struct HwReduce<SumOp, float> {
  static __simt_callee__ float call(float v) { return asc_reduce_add(v); }
};
template <> struct HwReduce<SumOp, int32_t> {
  static __simt_callee__ int32_t call(int32_t v) { return asc_reduce_add(v); }
};
template <> struct HwReduce<SumOp, uint32_t> {
  static __simt_callee__ uint32_t call(uint32_t v) { return asc_reduce_add(v); }
};
template <> struct HwReduce<MaxOp, float> {
  static __simt_callee__ float call(float v) { return asc_reduce_max(v); }
};
template <> struct HwReduce<MaxOp, int32_t> {
  static __simt_callee__ int32_t call(int32_t v) { return asc_reduce_max(v); }
};
template <> struct HwReduce<MaxOp, uint32_t> {
  static __simt_callee__ uint32_t call(uint32_t v) { return asc_reduce_max(v); }
};
template <> struct HwReduce<MinOp, float> {
  static __simt_callee__ float call(float v) { return asc_reduce_min(v); }
};
template <> struct HwReduce<MinOp, int32_t> {
  static __simt_callee__ int32_t call(int32_t v) { return asc_reduce_min(v); }
};
template <> struct HwReduce<MinOp, uint32_t> {
  static __simt_callee__ uint32_t call(uint32_t v) { return asc_reduce_min(v); }
};

/*!
 * \brief Cross-thread reduction on Ascend.
 *
 * Type dispatch:
 *   - ShflXor<T>  and  HwReduce<Reducer,T>  are explicitly specialized for
 *     each supported (type, reducer) combination.  When no specialization
 *     exists the name is incomplete → substitution failure → the
 *     corresponding warp_reduce overload is silently dropped (SFINAE).
 *   - warp_reduce(U, int)   → requires HwReduce  (hw intrinsic path)
 *   - warp_reduce(U, long)  → requires ShflXor   (butterfly-only path)
 *   - warp_reduce(U, __ubuf__ U*) → UB fallback, always available
 *   - warp_or_ub        tries int (hw) else falls back to UB
 *   - cross_warp_or_ub  tries cross_warp (which itself needs hw/shfl) else UB
 *
 * Template parameters:
 *   Reducer       - binary reduction functor (SumOp, MaxOp, MinOp).
 *   threads       - number of thread positions that span the reduce dimension,
 *                   equal to extent * scale.
 *   scale         - stride of participating threads in the thread index space.
 *                   When the thread-to-data mapping is normalized as
 *                     threadIdx = source * scale + ...
 *                   `scale` is the stride between consecutive logical
 *                   participants in the reduce dimension.
 *   thread_offset - base thread index offset within the block.
 *
 * Two warp-internal strategies are used depending on the number of independent
 * reduce groups per warp:
 *
 *   1. hw_reduce (asc_reduce_add/max/min + identity masking):
 *      Used when extent >= 16 && scale == 1, giving <= 2 groups per warp.
 *      Cost: groups_per_warp rounds of a single hardware reduce instruction.
 *
 *   2. butterfly (asc_shfl_xor):
 *      Used otherwise.  Cost: log2(extent) rounds of shfl_xor + reducer op.
 *
 * For cross-warp reductions (threads > 32), each warp first performs an
 * intra-warp reduce, writes the result to UB (shared memory), synchronizes,
 * and then a single warp finalizes via hw_reduce.
 *
 * Example: for a 3D tensor (1,4,4) with 16 elements mapped to threads as
 *   thread = j * 4 + k, reducing along j (dim=1):
 *   extent=4, scale=4, threads=16.
 *   XOR offsets: 8, 4 → terminates at scale=4.
 *   This correctly reduces threads {0,4,8,12}, {1,5,9,13}, etc.
 */
template <class Reducer, int threads, int scale = 1, int thread_offset = 0>
struct AscendAllReduce {
  template <class, int, int, int> friend struct AscendAllReduce;

  static_assert(threads >= 1);
  static_assert(scale >= 1);
  static_assert(threads % scale == 0);

private:
  static constexpr bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }
  static constexpr int extent = threads / scale;

  // ─── Identity value (per Reducer, per T) ───────────────────────────────
  template <typename T> static __simt_callee__ constexpr T reduce_identity() {
    if constexpr (std::is_same_v<Reducer, SumOp>) {
      return T(0);
    } else if constexpr (std::is_same_v<Reducer, MaxOp>) {
      if constexpr (std::is_same_v<T, float>)
        return -HUGE_VALF;
      else if constexpr (std::is_same_v<T, int32_t>)
        return T(-2147483647 - 1); // INT32_MIN
      else
        return T(0);
    } else { // MinOp
      if constexpr (std::is_same_v<T, float>)
        return HUGE_VALF;
      else if constexpr (std::is_same_v<T, int32_t>)
        return T(2147483647); // INT32_MAX
      else
        return T(-1); // UINT32_MAX
    }
  }

  // ─── Butterfly: recursive XOR shuffle reduce ─────────────────────────
  template <int cur, int stop, typename T>
  static __simt_callee__ T butterfly(T x) {
    if constexpr (cur <= stop) {
      return x;
    } else {
      constexpr int offset = cur / 2;
      Reducer op;
      x = op(x, ShflXor<T>::call(x, offset));
      return butterfly<cur / 2, stop>(x);
    }
  }

  // ─── Hardware reduce + identity masking ──────────────────────────────
  // Precondition: scale == 1.  HwReduce<Reducer,T> must exist.
  template <typename T>
  static __simt_callee__ auto warp_hw_reduce(T x)
      -> decltype(HwReduce<Reducer, T>::call(x)) {
    constexpr T id = reduce_identity<T>();
    constexpr int groups = 32 / threads;

    if constexpr (groups == 1) {
      return HwReduce<Reducer, T>::call(x);
    } else {
      int tx = threadIdx.x - thread_offset;
      int my_group = (tx & 31) / threads;
      T result = x;
#pragma unroll
      for (int g = 0; g < groups; ++g) {
        T v = HwReduce<Reducer, T>::call((my_group == g) ? result : id);
        if (my_group == g)
          result = v;
      }
      return result;
    }
  }

  // ─── warp_reduce overload set (SFINAE-dispatched by type support) ─────

  // Priority 1: int tag → HwReduce must exist
  template <typename U>
  static __simt_callee__ auto warp_reduce(U x, int)
      -> decltype(HwReduce<Reducer, U>::call(x)) {
    if constexpr (extent <= 1)
      return x;
    else if constexpr (extent >= 16 && scale == 1)
      return warp_hw_reduce(x);
    else
      return butterfly<threads, scale>(x);
  }

  // Priority 2: long tag → ShflXor must exist (butterfly only, no hw)
  template <typename U>
  static __simt_callee__ auto warp_reduce(U x, long)
      -> decltype(ShflXor<U>::call(x, 0), U{}) {
    if constexpr (extent <= 1)
      return x;
    else
      return butterfly<threads, scale>(x);
  }

  // ─── UB fallback (always available) ───────────────────────────────────
  template <typename T>
  static __simt_callee__ T ub_reduce(T x, __ubuf__ T *red_buf) {
    int tx = threadIdx.x - thread_offset;
    int group = tx / threads;
    int lane = tx % threads;
    Reducer op;

    red_buf[tx] = x;
    asc_syncthreads();

    if (lane < scale) {
      T result = red_buf[group * threads + lane];
      for (int i = scale; i < threads; i += scale) {
        result = op(result, red_buf[group * threads + lane + i]);
      }
      red_buf[group * threads + lane] = result;
    }
    asc_syncthreads();
    T result = red_buf[group * threads + (lane % scale)];
    asc_syncthreads();
    return result;
  }

  // ─── Cross-warp reduction ─────────────────────────────────────────────
  // SFINAE: available only when warp_reduce(U,int) or warp_reduce(U,long)
  //         is viable (tested via decltype in the trailing return).
  template <typename U>
  static __simt_callee__ auto cross_warp_reduce(U x, __ubuf__ U *red_buf)
      -> decltype(warp_reduce(x, 0), U{}) {
    constexpr int num_warps = threads / 32;
    int tx = threadIdx.x - thread_offset;
    int wid = tx >> 5, lid = tx & 31;

    // Step 1: intra-warp reduce.
    U warp_val =
        AscendAllReduce<Reducer, 32, scale, thread_offset>::warp_reduce(x, 0);

    // Step 2: each warp writes result(s) to UB.
    if (lid < scale)
      red_buf[wid * scale + lid] = warp_val;
    asc_syncthreads();

    // Step 3: cross-warp reduce — single warp reads and reduces.
    U result;
    if constexpr (scale == 1) {
      if (tx < 32) {
        U v = (lid < num_warps) ? red_buf[lid] : reduce_identity<U>();
        result = HwReduce<Reducer, U>::call(v);
      }
    } else if constexpr (scale * num_warps <= 32) {
      if (tx < 32) {
        constexpr int total = scale * num_warps;
        U v = (lid < total) ? red_buf[lid] : reduce_identity<U>();
        result = AscendAllReduce<Reducer, total, scale, 0>::warp_reduce(v, 0);
      }
    } else {
      if (tx < 32) {
        Reducer op;
        result = reduce_identity<U>();
        int my_slot = lid % scale;
        for (int w = 0; w < num_warps; ++w) {
          int idx = w * scale + my_slot;
          result = (lid < scale) ? op(result, red_buf[idx]) : result;
        }
      }
    }
    asc_syncthreads();

    // Step 4: broadcast result back to all threads.
    if (tx < scale)
      red_buf[tx] = result;
    asc_syncthreads();
    result = red_buf[tx % scale];
    asc_syncthreads();
    return result;
  }

  // ─── SFINAE dispatch wrappers (try hw/shfl, fall back to ub) ──────────

  // warp_or_ub: tries register-based reduce, else UB
  template <typename U>
  static __simt_callee__ auto warp_or_ub(U x, __ubuf__ U *red_buf, int)
      -> decltype(warp_reduce(x, 0)) {
    return warp_reduce(x, 0);
  }
  template <typename U>
  static __simt_callee__ U warp_or_ub(U x, __ubuf__ U *red_buf, long) {
    return ub_reduce(x, red_buf);
  }

  // cross_warp_or_ub: tries cross-warp reduce, else UB
  template <typename U>
  static __simt_callee__ auto cross_warp_or_ub(U x, __ubuf__ U *red_buf, int)
      -> decltype(cross_warp_reduce(x, red_buf)) {
    return cross_warp_reduce(x, red_buf);
  }
  template <typename U>
  static __simt_callee__ U cross_warp_or_ub(U x, __ubuf__ U *red_buf, long) {
    return ub_reduce(x, red_buf);
  }

public:
  template <typename T>
  static __simt_callee__ T run(T x, __ubuf__ T *red_buf = nullptr) {
    if constexpr (threads <= scale) {
      return x;
    } else if constexpr (threads <= 32 && is_pow2(threads) && is_pow2(scale)) {
      return warp_or_ub(x, red_buf, 0);
    } else if constexpr (threads <= 32) {
      return ub_reduce(x, red_buf);
    } else if constexpr (threads > 32 && is_pow2(threads) && scale <= 32 &&
                         is_pow2(scale)) {
      return cross_warp_or_ub(x, red_buf, 0);
    } else {
      return ub_reduce(x, red_buf);
    }
  }
};

} // namespace tl
