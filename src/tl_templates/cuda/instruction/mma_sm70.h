#pragma once

#include "../common.h"

#ifndef __CUDACC_RTC__
#include <type_traits>
#include <utility>
#endif

namespace tl {

#ifndef TL_ALWAYS_FALSE_V_DEFINED
#define TL_ALWAYS_FALSE_V_DEFINED
template <class> inline constexpr bool always_false_v = false;
#endif

namespace detail {

// SM70 MMA Instruction Traits and Implementations
// SM70 supports m16n16k4 (m8n8k4 instruction at warp level) with FP16/FP32
// accumulation

// Base template for SM70 MMA implementation
template <DataType AType, DataType BType, DataType CType, bool TransA,
          bool TransB>
struct MmaSm70Impl {
  // Default: unsupported configuration
  static constexpr bool kSupported = false;

  static TL_DEVICE void exec(void *, const void *, const void *, const void *) {
    static_assert(always_false_v<std::integral_constant<bool, TransA>>,
                  "tl::mma_sync_sm70: unsupported configuration");
  }
};

// FP16 inputs, FP16 accumulation - col.col (TransA=true, TransB=true)
template <>
struct MmaSm70Impl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   true, true> {
  using DRegisters = unsigned[4];
  using ARegisters = unsigned[2];
  using BRegisters = unsigned[2];
  using CRegisters = unsigned[4];

  static constexpr bool kSupported = true;

  static TL_DEVICE void fma(unsigned &d0, unsigned &d1, unsigned &d2,
                            unsigned &d3, unsigned a0, unsigned a1, unsigned b0,
                            unsigned b1, unsigned c0, unsigned c1, unsigned c2,
                            unsigned c3) {
    asm volatile("mma.sync.aligned.m8n8k4.col.col.f16.f16.f16.f16 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1),
                   "r"(c2), "r"(c3));
  }
};

// FP16 inputs, FP16 accumulation - col.row (TransA=true, TransB=false)
template <>
struct MmaSm70Impl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   true, false> {
  using DRegisters = unsigned[4];
  using ARegisters = unsigned[2];
  using BRegisters = unsigned[2];
  using CRegisters = unsigned[4];

  static constexpr bool kSupported = true;

  static TL_DEVICE void fma(unsigned &d0, unsigned &d1, unsigned &d2,
                            unsigned &d3, unsigned a0, unsigned a1, unsigned b0,
                            unsigned b1, unsigned c0, unsigned c1, unsigned c2,
                            unsigned c3) {
    asm volatile("mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1),
                   "r"(c2), "r"(c3));
  }
};

// FP16 inputs, FP16 accumulation - row.col (TransA=false, TransB=true)
template <>
struct MmaSm70Impl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   false, true> {
  using DRegisters = unsigned[4];
  using ARegisters = unsigned[2];
  using BRegisters = unsigned[2];
  using CRegisters = unsigned[4];

  static constexpr bool kSupported = true;

  static TL_DEVICE void fma(unsigned &d0, unsigned &d1, unsigned &d2,
                            unsigned &d3, unsigned a0, unsigned a1, unsigned b0,
                            unsigned b1, unsigned c0, unsigned c1, unsigned c2,
                            unsigned c3) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1),
                   "r"(c2), "r"(c3));
  }
};

// FP16 inputs, FP16 accumulation - row.row (TransA=false, TransB=false)
template <>
struct MmaSm70Impl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                   false, false> {
  using DRegisters = unsigned[4];
  using ARegisters = unsigned[2];
  using BRegisters = unsigned[2];
  using CRegisters = unsigned[4];

  static constexpr bool kSupported = true;

  static TL_DEVICE void fma(unsigned &d0, unsigned &d1, unsigned &d2,
                            unsigned &d3, unsigned a0, unsigned a1, unsigned b0,
                            unsigned b1, unsigned c0, unsigned c1, unsigned c2,
                            unsigned c3) {
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1),
                   "r"(c2), "r"(c3));
  }
};

// FP16 inputs, FP32 accumulation - col.col (TransA=true, TransB=true)
template <>
struct MmaSm70Impl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat32,
                   true, true> {
  using DRegisters = float[8];
  using ARegisters = unsigned[2];
  using BRegisters = unsigned[2];
  using CRegisters = float[8];

  static constexpr bool kSupported = true;

  static TL_DEVICE void fma(float &d0, float &d1, float &d2, float &d3,
                            float &d4, float &d5, float &d6, float &d7,
                            unsigned a0, unsigned a1, unsigned b0, unsigned b1,
                            float c0, float c1, float c2, float c3, float c4,
                            float c5, float c6, float c7) {
    asm volatile("mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5),
                   "=f"(d6), "=f"(d7)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
                   "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
  }
};

// FP16 inputs, FP32 accumulation - col.row (TransA=true, TransB=false)
template <>
struct MmaSm70Impl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat32,
                   true, false> {
  using DRegisters = float[8];
  using ARegisters = unsigned[2];
  using BRegisters = unsigned[2];
  using CRegisters = float[8];

  static constexpr bool kSupported = true;

  static TL_DEVICE void fma(float &d0, float &d1, float &d2, float &d3,
                            float &d4, float &d5, float &d6, float &d7,
                            unsigned a0, unsigned a1, unsigned b0, unsigned b1,
                            float c0, float c1, float c2, float c3, float c4,
                            float c5, float c6, float c7) {
    asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5),
                   "=f"(d6), "=f"(d7)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
                   "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
  }
};

// FP16 inputs, FP32 accumulation - row.col (TransA=false, TransB=true)
template <>
struct MmaSm70Impl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat32,
                   false, true> {
  using DRegisters = float[8];
  using ARegisters = unsigned[2];
  using BRegisters = unsigned[2];
  using CRegisters = float[8];

  static constexpr bool kSupported = true;

  static TL_DEVICE void fma(float &d0, float &d1, float &d2, float &d3,
                            float &d4, float &d5, float &d6, float &d7,
                            unsigned a0, unsigned a1, unsigned b0, unsigned b1,
                            float c0, float c1, float c2, float c3, float c4,
                            float c5, float c6, float c7) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5),
                   "=f"(d6), "=f"(d7)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
                   "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
  }
};

// FP16 inputs, FP32 accumulation - row.row (TransA=false, TransB=false)
template <>
struct MmaSm70Impl<DataType::kFloat16, DataType::kFloat16, DataType::kFloat32,
                   false, false> {
  using DRegisters = float[8];
  using ARegisters = unsigned[2];
  using BRegisters = unsigned[2];
  using CRegisters = float[8];

  static constexpr bool kSupported = true;

  static TL_DEVICE void fma(float &d0, float &d1, float &d2, float &d3,
                            float &d4, float &d5, float &d6, float &d7,
                            unsigned a0, unsigned a1, unsigned b0, unsigned b1,
                            float c0, float c1, float c2, float c3, float c4,
                            float c5, float c6, float c7) {
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5),
                   "=f"(d6), "=f"(d7)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
                   "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
  }
};

// Helper to extract register types
template <class Impl> struct MmaSm70ImplTraits {
  using DReg = std::remove_extent_t<typename Impl::DRegisters>;
  using AReg = std::remove_extent_t<typename Impl::ARegisters>;
  using BReg = std::remove_extent_t<typename Impl::BRegisters>;
  using CReg = std::remove_extent_t<typename Impl::CRegisters>;

  static constexpr int kDRegs = std::extent_v<typename Impl::DRegisters>;
  static constexpr int kARegs = std::extent_v<typename Impl::ARegisters>;
  static constexpr int kBRegs = std::extent_v<typename Impl::BRegisters>;
  static constexpr int kCRegs = std::extent_v<typename Impl::CRegisters>;
};

// Dispatcher for SM70 MMA operations
template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB>
struct MmaSm70Dispatcher {
  using CRegType = void;
  using ARegType = void;
  using BRegType = void;

  static TL_DEVICE void exec(CRegType *, const ARegType *, const BRegType *,
                             const CRegType *) {
    static_assert(always_false_v<std::integral_constant<int, M>>,
                  "tl::mma_sync_sm70: unsupported configuration. "
                  "SM70 only supports m16n16k4 with FP16 inputs and FP16/FP32 "
                  "accumulation.");
  }
};

// Helper to call fma with unpacked register arrays
template <class Impl, size_t... DIdx, size_t... AIdx, size_t... BIdx,
          size_t... CIdx>
TL_DEVICE void
call_fma_impl_sm70(typename MmaSm70ImplTraits<Impl>::DReg *d,
                   const typename MmaSm70ImplTraits<Impl>::AReg *a,
                   const typename MmaSm70ImplTraits<Impl>::BReg *b,
                   const typename MmaSm70ImplTraits<Impl>::CReg *c,
                   std::index_sequence<DIdx...>, std::index_sequence<AIdx...>,
                   std::index_sequence<BIdx...>, std::index_sequence<CIdx...>) {
  Impl::fma(d[DIdx]..., a[AIdx]..., b[BIdx]..., c[CIdx]...);
}

template <class Impl>
TL_DEVICE void call_fma_sm70(typename MmaSm70ImplTraits<Impl>::DReg *d,
                             const typename MmaSm70ImplTraits<Impl>::AReg *a,
                             const typename MmaSm70ImplTraits<Impl>::BReg *b,
                             const typename MmaSm70ImplTraits<Impl>::CReg *c) {
  call_fma_impl_sm70<Impl>(
      d, a, b, c, std::make_index_sequence<MmaSm70ImplTraits<Impl>::kDRegs>{},
      std::make_index_sequence<MmaSm70ImplTraits<Impl>::kARegs>{},
      std::make_index_sequence<MmaSm70ImplTraits<Impl>::kBRegs>{},
      std::make_index_sequence<MmaSm70ImplTraits<Impl>::kCRegs>{});
}

// Define dispatchers for all supported SM70 configurations
// Note: m8n8k4 instruction computes m16n16k4 at warp level
#define TL_DEFINE_MMA_SM70_DISPATCHER(ATypeEnum, BTypeEnum, CTypeEnum,         \
                                      TransAValue, TransBValue)                \
  template <>                                                                  \
  struct MmaSm70Dispatcher<DataType::ATypeEnum, DataType::BTypeEnum,           \
                           DataType::CTypeEnum, 16, 16, 4, TransAValue,        \
                           TransBValue> {                                      \
    using Impl = MmaSm70Impl<DataType::ATypeEnum, DataType::BTypeEnum,         \
                             DataType::CTypeEnum, TransAValue, TransBValue>;   \
    using Traits = MmaSm70ImplTraits<Impl>;                                    \
    using CRegType = typename Traits::DReg;                                    \
    using ARegType = typename Traits::AReg;                                    \
    using BRegType = typename Traits::BReg;                                    \
    static_assert(                                                             \
        std::is_same_v<typename Traits::DReg, typename Traits::CReg>,          \
        "tl::mma_sync_sm70 requires matching accumulator/output regs");        \
    static TL_DEVICE void exec(CRegType *d, const ARegType *a,                 \
                               const BRegType *b, const CRegType *c) {         \
      call_fma_sm70<Impl>(d, a, b, c);                                         \
    }                                                                          \
  };

// FP16 inputs with FP16 accumulation (all layout combinations)
TL_DEFINE_MMA_SM70_DISPATCHER(kFloat16, kFloat16, kFloat16, true, true)
TL_DEFINE_MMA_SM70_DISPATCHER(kFloat16, kFloat16, kFloat16, true, false)
TL_DEFINE_MMA_SM70_DISPATCHER(kFloat16, kFloat16, kFloat16, false, true)
TL_DEFINE_MMA_SM70_DISPATCHER(kFloat16, kFloat16, kFloat16, false, false)

// FP16 inputs with FP32 accumulation (all layout combinations)
TL_DEFINE_MMA_SM70_DISPATCHER(kFloat16, kFloat16, kFloat32, true, true)
TL_DEFINE_MMA_SM70_DISPATCHER(kFloat16, kFloat16, kFloat32, true, false)
TL_DEFINE_MMA_SM70_DISPATCHER(kFloat16, kFloat16, kFloat32, false, true)
TL_DEFINE_MMA_SM70_DISPATCHER(kFloat16, kFloat16, kFloat32, false, false)

#undef TL_DEFINE_MMA_SM70_DISPATCHER

} // namespace detail

/// SM70 MMA synchronous instruction wrapper
/// Supports m16n16k4 shape (m8n8k4 instruction at warp level) with FP16 inputs
/// and FP16/FP32 accumulation
///
/// @tparam AType Input A data type (kFloat16)
/// @tparam BType Input B data type (kFloat16)
/// @tparam CType Accumulator/output data type (kFloat16 or kFloat32)
/// @tparam M Matrix M dimension (16)
/// @tparam N Matrix N dimension (16)
/// @tparam K Matrix K dimension (4)
/// @tparam TransA Whether A is transposed (false=row-major, true=col-major)
/// @tparam TransB Whether B is transposed (false=row-major, true=col-major)
template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB>
TL_DEVICE void mma_sync_sm70(
    typename detail::MmaSm70Dispatcher<AType, BType, CType, M, N, K, TransA,
                                       TransB>::CRegType *c,
    const typename detail::MmaSm70Dispatcher<AType, BType, CType, M, N, K,
                                             TransA, TransB>::ARegType *a,
    const typename detail::MmaSm70Dispatcher<AType, BType, CType, M, N, K,
                                             TransA, TransB>::BRegType *b) {
  using Dispatcher =
      detail::MmaSm70Dispatcher<AType, BType, CType, M, N, K, TransA, TransB>;
  static_assert(!std::is_void_v<typename Dispatcher::CRegType>,
                "tl::mma_sync_sm70: unsupported configuration. "
                "SM70 only supports m16n16k4 with FP16 inputs.");
  Dispatcher::exec(c, a, b, c);
}

} // namespace tl
