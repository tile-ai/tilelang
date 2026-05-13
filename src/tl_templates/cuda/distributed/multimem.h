#pragma once

#include "../common.h"

// multimem instructions require SM 90+ (Hopper) and CUDA 12.0+
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDACC_VER_MAJOR__ >= 12

#ifndef TL_ALWAYS_FALSE_V_DEFINED
#define TL_ALWAYS_FALSE_V_DEFINED
template <class> inline constexpr bool always_false_v = false;
#endif

namespace tl {
namespace multimem {

enum class ReduceOp { ADD = 0, MIN = 1, MAX = 2 };

// === Per-instruction primitives (used by MultimemRewriter post-process) ===

// --- LdReduceV4: 128-bit load-reduce from multicast address ---

template <ReduceOp op, typename DType> struct LdReduceV4 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::LdReduceV4: unsupported dtype/op combination");
  }
};

template <> struct LdReduceV4<ReduceOp::ADD, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int4 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, "
        "[%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int4 *>(dst) = ret;
  }
};

template <> struct LdReduceV4<ReduceOp::MIN, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int4 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.min.v4.f32 {%0, %1, %2, %3}, "
        "[%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int4 *>(dst) = ret;
  }
};

template <> struct LdReduceV4<ReduceOp::MAX, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int4 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.max.v4.f32 {%0, %1, %2, %3}, "
        "[%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int4 *>(dst) = ret;
  }
};

template <> struct LdReduceV4<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t *dst_u32 = reinterpret_cast<uint32_t *>(dst);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      uint32_t ret;
      asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.f16x2 %0, [%1];"
                   : "=r"(ret)
                   : "l"(mc + i * 4)
                   : "memory");
      dst_u32[i] = ret;
    }
  }
};

template <> struct LdReduceV4<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t *dst_u32 = reinterpret_cast<uint32_t *>(dst);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      uint32_t ret;
      asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.bf16x2 %0, [%1];"
                   : "=r"(ret)
                   : "l"(mc + i * 4)
                   : "memory");
      dst_u32[i] = ret;
    }
  }
};

// --- StV4: 128-bit store to multicast address ---

template <typename DType> struct StV4 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::StV4: unsupported dtype");
  }
};

template <> struct StV4<float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    int4 val = *reinterpret_cast<const int4 *>(src);
    asm volatile("multimem.st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(mcast_ptr), "r"(val.x), "r"(val.y), "r"(val.z),
                   "r"(val.w)
                 : "memory");
  }
};

template <> struct StV4<half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      asm volatile("multimem.st.weak.global.f16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

template <> struct StV4<bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      asm volatile("multimem.st.weak.global.bf16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

// --- RedV4: 128-bit reduce into multicast address ---

template <ReduceOp op, typename DType> struct RedV4 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::RedV4: unsupported dtype/op combination");
  }
};

template <> struct RedV4<ReduceOp::ADD, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    int4 val = *reinterpret_cast<const int4 *>(src);
    asm volatile(
        "multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(mcast_ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
        : "memory");
  }
};

template <> struct RedV4<ReduceOp::MIN, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    // multimem.red min not directly available as v4; use scalar fallback
    const float *src_f = reinterpret_cast<const float *>(src);
    const char *mc_bytes = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      unsigned val = __float_as_uint(src_f[i]);
      asm volatile("multimem.red.relaxed.sys.global.min.f32 [%0], %1;"
                   :
                   : "l"(mc_bytes + i * 4), "r"(val)
                   : "memory");
    }
  }
};

template <> struct RedV4<ReduceOp::MAX, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const float *src_f = reinterpret_cast<const float *>(src);
    const char *mc_bytes = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      unsigned val = __float_as_uint(src_f[i]);
      asm volatile("multimem.red.relaxed.sys.global.max.f32 [%0], %1;"
                   :
                   : "l"(mc_bytes + i * 4), "r"(val)
                   : "memory");
    }
  }
};

template <> struct RedV4<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      asm volatile("multimem.red.release.sys.global.add.f16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

template <> struct RedV4<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      asm volatile("multimem.red.release.sys.global.add.bf16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

// === V2 variants (64-bit = 2×f32, implemented as 2 scalar ops) ===

template <ReduceOp op, typename DType> struct LdReduceV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::LdReduceV2: unsupported dtype/op");
  }
};

template <> struct LdReduceV2<ReduceOp::ADD, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int2 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v2.f32 {%0, %1}, [%2];"
        : "=r"(ret.x), "=r"(ret.y)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int2 *>(dst) = ret;
  }
};

template <> struct LdReduceV2<ReduceOp::MIN, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int2 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.min.v2.f32 {%0, %1}, [%2];"
        : "=r"(ret.x), "=r"(ret.y)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int2 *>(dst) = ret;
  }
};

template <> struct LdReduceV2<ReduceOp::MAX, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int2 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.max.v2.f32 {%0, %1}, [%2];"
        : "=r"(ret.x), "=r"(ret.y)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int2 *>(dst) = ret;
  }
};

template <> struct LdReduceV2<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t ret;
    asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.f16x2 %0, [%1];"
                 : "=r"(ret)
                 : "l"(mcast_ptr)
                 : "memory");
    *reinterpret_cast<uint32_t *>(dst) = ret;
  }
};

template <> struct LdReduceV2<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t ret;
    asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.bf16x2 %0, [%1];"
                 : "=r"(ret)
                 : "l"(mcast_ptr)
                 : "memory");
    *reinterpret_cast<uint32_t *>(dst) = ret;
  }
};

template <typename DType> struct StV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::StV2: unsupported dtype");
  }
};

template <> struct StV2<float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    int2 val = *reinterpret_cast<const int2 *>(src);
    asm volatile("multimem.st.relaxed.sys.global.v2.b32 [%0], {%1, %2};"
                 :
                 : "l"(mcast_ptr), "r"(val.x), "r"(val.y)
                 : "memory");
  }
};

template <> struct StV2<half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.st.weak.global.f16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

template <> struct StV2<bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.st.weak.global.bf16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

template <ReduceOp op, typename DType> struct RedV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::RedV2: unsupported dtype/op");
  }
};

template <> struct RedV2<ReduceOp::ADD, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const float *src_f = reinterpret_cast<const float *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      unsigned val = __float_as_uint(src_f[i]);
      asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(val)
                   : "memory");
    }
  }
};

template <> struct RedV2<ReduceOp::MIN, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const float *src_f = reinterpret_cast<const float *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      unsigned val = __float_as_uint(src_f[i]);
      asm volatile("multimem.red.relaxed.sys.global.min.f32 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(val)
                   : "memory");
    }
  }
};

template <> struct RedV2<ReduceOp::MAX, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const float *src_f = reinterpret_cast<const float *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      unsigned val = __float_as_uint(src_f[i]);
      asm volatile("multimem.red.relaxed.sys.global.max.f32 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(val)
                   : "memory");
    }
  }
};

template <> struct RedV2<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.red.release.sys.global.add.f16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

template <> struct RedV2<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.red.release.sys.global.add.bf16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

// === V8 variants (128-bit = 8×fp16/bf16, implemented as 4 packed x2 ops) ===

template <ReduceOp op, typename DType> struct LdReduceV8 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::LdReduceV8: unsupported dtype/op");
  }
};

template <> struct LdReduceV8<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t *dst_u32 = reinterpret_cast<uint32_t *>(dst);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t ret;
      asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.f16x2 %0, [%1];"
                   : "=r"(ret)
                   : "l"(mc + i * 4)
                   : "memory");
      dst_u32[i] = ret;
    }
  }
};

template <> struct LdReduceV8<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t *dst_u32 = reinterpret_cast<uint32_t *>(dst);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t ret;
      asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.bf16x2 %0, [%1];"
                   : "=r"(ret)
                   : "l"(mc + i * 4)
                   : "memory");
      dst_u32[i] = ret;
    }
  }
};

template <typename DType> struct StV8 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::StV8: unsupported dtype");
  }
};

template <> struct StV8<half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      asm volatile("multimem.st.weak.global.f16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

template <> struct StV8<bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      asm volatile("multimem.st.weak.global.bf16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

template <ReduceOp op, typename DType> struct RedV8 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::RedV8: unsupported dtype/op");
  }
};

template <> struct RedV8<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      asm volatile("multimem.red.release.sys.global.add.f16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

template <> struct RedV8<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      asm volatile("multimem.red.release.sys.global.add.bf16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

// === Thread-level signal write to multicast address ===

template <typename T> struct Signal {
  TL_DEVICE static void run(void *, T) {
    static_assert(always_false_v<T>, "tl::multimem::Signal: unsupported type");
  }
};
template <> struct Signal<uint32_t> {
  TL_DEVICE static void run(void *mcast_ptr, uint32_t val) {
    asm volatile("multimem.st.release.sys.global.u32 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};
template <> struct Signal<uint64_t> {
  TL_DEVICE static void run(void *mcast_ptr, uint64_t val) {
    asm volatile("multimem.st.release.sys.global.u64 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "l"(val)
                 : "memory");
  }
};

template <typename T> struct SignalAdd {
  TL_DEVICE static void run(void *, T) {
    static_assert(always_false_v<T>,
                  "tl::multimem::SignalAdd: unsupported type");
  }
};
template <> struct SignalAdd<uint32_t> {
  TL_DEVICE static void run(void *mcast_ptr, uint32_t val) {
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};
template <> struct SignalAdd<int32_t> {
  TL_DEVICE static void run(void *mcast_ptr, int32_t val) {
    asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

// === Bulk async TMA-to-multicast (SM100+ / PTX 9.1+ / CUDA 13.0+) ===
// Both: shared::cta → global(mcast), bulk_group completion

#if __CUDA_ARCH__ >= 1000 && __CUDACC_VER_MAJOR__ >= 13

TL_DEVICE void cp_async_bulk(void *mcast_global, void *smem, uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_add_f32(void *mcast_global, void *smem,
                                            uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
      "[%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_min_f32(void *mcast_global, void *smem,
                                            uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f32 "
      "[%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_max_f32(void *mcast_global, void *smem,
                                            uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.max.f32 "
      "[%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_add_f16(void *mcast_global, void *smem,
                                            uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f16x2 "
      "[%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_add_bf16(void *mcast_global, void *smem,
                                             uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.bf16x2 "
      "[%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

#else // PTX 9.1 not available — unconditional trap

TL_DEVICE void cp_async_bulk(void *mcast_global, void *smem, uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_add_f32(void *mcast_global, void *smem,
                                            uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_min_f32(void *mcast_global, void *smem,
                                            uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_max_f32(void *mcast_global, void *smem,
                                            uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_add_f16(void *mcast_global, void *smem,
                                            uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_add_bf16(void *mcast_global, void *smem,
                                             uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}

#endif // __CUDA_ARCH__ >= 1000 && __CUDACC_VER_MAJOR__ >= 13

} // namespace multimem
} // namespace tl

#endif // __CUDA_ARCH__ >= 900 && __CUDACC_VER_MAJOR__ >= 12
