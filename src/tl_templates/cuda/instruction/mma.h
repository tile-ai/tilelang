#pragma once

#include "../common.h"
#include <cute/arch/mma_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm89.hpp>

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 1200)) ||           \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200))
#define TL_HAS_F8F6F4_MMA_DISPATCHER 1
#include <cute/arch/mma_sm120.hpp>
#else
#define TL_HAS_F8F6F4_MMA_DISPATCHER 0
#endif

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

template <class Impl> struct MmaImplTraits {
  using DReg = std::remove_extent_t<typename Impl::DRegisters>;
  using AReg = std::remove_extent_t<typename Impl::ARegisters>;
  using BReg = std::remove_extent_t<typename Impl::BRegisters>;
  using CReg = std::remove_extent_t<typename Impl::CRegisters>;

  static constexpr int kDRegs = std::extent_v<typename Impl::DRegisters>;
  static constexpr int kARegs = std::extent_v<typename Impl::ARegisters>;
  static constexpr int kBRegs = std::extent_v<typename Impl::BRegisters>;
  static constexpr int kCRegs = std::extent_v<typename Impl::CRegisters>;
};

template <class Impl, size_t... DIdx, size_t... AIdx, size_t... BIdx,
          size_t... CIdx>
TL_DEVICE void
call_fma_impl(typename MmaImplTraits<Impl>::DReg *d,
              const typename MmaImplTraits<Impl>::AReg *a,
              const typename MmaImplTraits<Impl>::BReg *b,
              const typename MmaImplTraits<Impl>::CReg *c,
              std::index_sequence<DIdx...>, std::index_sequence<AIdx...>,
              std::index_sequence<BIdx...>, std::index_sequence<CIdx...>) {
  Impl::fma(d[DIdx]..., a[AIdx]..., b[BIdx]..., c[CIdx]...);
}

template <class Impl>
TL_DEVICE void call_fma(typename MmaImplTraits<Impl>::DReg *d,
                        const typename MmaImplTraits<Impl>::AReg *a,
                        const typename MmaImplTraits<Impl>::BReg *b,
                        const typename MmaImplTraits<Impl>::CReg *c) {
  call_fma_impl<Impl>(d, a, b, c,
                      std::make_index_sequence<MmaImplTraits<Impl>::kDRegs>{},
                      std::make_index_sequence<MmaImplTraits<Impl>::kARegs>{},
                      std::make_index_sequence<MmaImplTraits<Impl>::kBRegs>{},
                      std::make_index_sequence<MmaImplTraits<Impl>::kCRegs>{});
}

struct SM75_8x8x32_S32S4S4S32_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1,
                                   uint32_t const &a0, uint32_t const &b0,
                                   uint32_t const &c0, uint32_t const &c1) {
#if defined(CUTE_ARCH_MMA_SM75_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32"
                 "{%0, %1},"
                 "{%2},"
                 "{%3},"
                 "{%4, %5};\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a0), "r"(b0), "r"(c0), "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH(
        "Attempting to use SM75_8x8x32_S32S4S4S32_TN without "
        "CUTE_ARCH_MMA_SM75_ENABLED");
#endif
  }
};

struct SM75_8x8x32_S32U4U4S32_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1,
                                   uint32_t const &a0, uint32_t const &b0,
                                   uint32_t const &c0, uint32_t const &c1) {
#if defined(CUTE_ARCH_MMA_SM75_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32"
                 "{%0, %1},"
                 "{%2},"
                 "{%3},"
                 "{%4, %5};\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a0), "r"(b0), "r"(c0), "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH(
        "Attempting to use SM75_8x8x32_S32U4U4S32_TN without "
        "CUTE_ARCH_MMA_SM75_ENABLED");
#endif
  }
};

#if TL_HAS_F8F6F4_MMA_DISPATCHER
template <class Impl, bool ShiftA, bool ShiftB> struct F8F6F4_16x8x32_TN {
  using DRegisters = typename Impl::DRegisters;
  using ARegisters = typename Impl::ARegisters;
  using BRegisters = typename Impl::BRegisters;
  using CRegisters = typename Impl::CRegisters;

  template <bool Shift>
  CUTE_HOST_DEVICE static uint32_t shift_fp4_mma_operand(uint32_t value) {
    if constexpr (Shift) {
      return value << 2;
    } else {
      return value;
    }
  }

  CUTE_HOST_DEVICE static void fma(float &d0, float &d1, float &d2, float &d3,
                                   uint32_t const &a0, uint32_t const &a1,
                                   uint32_t const &a2, uint32_t const &a3,
                                   uint32_t const &b0, uint32_t const &b1,
                                   float const &c0, float const &c1,
                                   float const &c2, float const &c3) {
    // F8F6F4 MMA consumes the e2m1 payload from bits 2..5 of each
    // byte-carrier lane, so FP4 operands are shifted immediately before fma.
    Impl::fma(
        d0, d1, d2, d3, shift_fp4_mma_operand<ShiftA>(a0),
        shift_fp4_mma_operand<ShiftA>(a1), shift_fp4_mma_operand<ShiftA>(a2),
        shift_fp4_mma_operand<ShiftA>(a3), shift_fp4_mma_operand<ShiftB>(b0),
        shift_fp4_mma_operand<ShiftB>(b1), c0, c1, c2, c3);
  }
};

using F8F6F4_FP4_FP4_F32_TN = F8F6F4_16x8x32_TN<
    cute::SM120_16x8x32_TN<cute::float_e2m1_t, cute::float_e2m1_t, float>, true,
    true>;
using F8F6F4_FP8_FP4_F32_TN = F8F6F4_16x8x32_TN<
    cute::SM120_16x8x32_TN<cute::float_e4m3_t, cute::float_e2m1_t, float>,
    false, true>;
using F8F6F4_FP4_FP8_F32_TN = F8F6F4_16x8x32_TN<
    cute::SM120_16x8x32_TN<cute::float_e2m1_t, cute::float_e4m3_t, float>, true,
    false>;
#endif

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB, bool Saturate>
struct MmaDispatcher {
  using CRegType = void;
  using ARegType = void;
  using BRegType = void;

  static TL_DEVICE void exec(CRegType *, const ARegType *, const BRegType *,
                             const CRegType *) {
    static_assert(always_false_v<std::integral_constant<int, M>>,
                  "tl::mma_sync: unsupported configuration");
  }
};

#define TL_DEFINE_MMA_DISPATCHER(ATypeEnum, BTypeEnum, CTypeEnum, MValue,      \
                                 NValue, KValue, TransAValue, TransBValue,     \
                                 SaturateValue, ImplType)                      \
  template <>                                                                  \
  struct MmaDispatcher<DataType::ATypeEnum, DataType::BTypeEnum,               \
                       DataType::CTypeEnum, MValue, NValue, KValue,            \
                       TransAValue, TransBValue, SaturateValue> {              \
    using Impl = ImplType;                                                     \
    using Traits = MmaImplTraits<Impl>;                                        \
    using CRegType = typename Traits::DReg;                                    \
    using ARegType = typename Traits::AReg;                                    \
    using BRegType = typename Traits::BReg;                                    \
    static_assert(                                                             \
        std::is_same_v<typename Traits::DReg, typename Traits::CReg>,          \
        "tl::mma_sync requires matching accumulator/output regs");             \
    static TL_DEVICE void exec(CRegType *d, const ARegType *a,                 \
                               const BRegType *b, const CRegType *c) {         \
      call_fma<Impl>(d, a, b, c);                                              \
    }                                                                          \
  };

// FP16 inputs (TN layout: A row-major, B column-major)
TL_DEFINE_MMA_DISPATCHER(kFloat16, kFloat16, kFloat16, 16, 8, 16, false, true,
                         false, cute::SM80_16x8x16_F16F16F16F16_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat16, kFloat16, kFloat32, 16, 8, 16, false, true,
                         false, cute::SM80_16x8x16_F32F16F16F32_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat16, kFloat16, kFloat32, 16, 8, 8, false, true,
                         false, cute::SM75_16x8x8_F32F16F16F32_TN)

// BF16 inputs
TL_DEFINE_MMA_DISPATCHER(kBFloat16, kBFloat16, kFloat32, 16, 8, 16, false, true,
                         false, cute::SM80_16x8x16_F32BF16BF16F32_TN)

// INT8 inputs (k32)
TL_DEFINE_MMA_DISPATCHER(kInt8, kInt8, kInt32, 16, 8, 32, false, true, false,
                         cute::SM80_16x8x32_S32S8S8S32_TN)
TL_DEFINE_MMA_DISPATCHER(kUInt8, kUInt8, kInt32, 16, 8, 32, false, true, false,
                         cute::SM80_16x8x32_S32U8U8S32_TN)

// INT8 inputs (k16) for SM75
TL_DEFINE_MMA_DISPATCHER(kInt8, kInt8, kInt32, 8, 8, 16, false, true, false,
                         cute::SM75_8x8x16_S32S8S8S32_TN)

// INT4 inputs (k32) for SM75
TL_DEFINE_MMA_DISPATCHER(kInt4, kInt4, kInt32, 8, 8, 32, false, true, false,
                         tl::detail::SM75_8x8x32_S32S4S4S32_TN)
TL_DEFINE_MMA_DISPATCHER(kUInt4, kUInt4, kInt32, 8, 8, 32, false, true, false,
                         tl::detail::SM75_8x8x32_S32U4U4S32_TN)

// INT4 inputs (k32, k64) for SM80+
TL_DEFINE_MMA_DISPATCHER(kInt4, kInt4, kInt32, 16, 8, 32, false, true, false,
                         cute::SM80_16x8x32_S32S4S4S32_TN)
TL_DEFINE_MMA_DISPATCHER(kInt4, kInt4, kInt32, 16, 8, 64, false, true, false,
                         cute::SM80_16x8x64_S32S4S4S32_TN)
TL_DEFINE_MMA_DISPATCHER(kUInt4, kUInt4, kInt32, 16, 8, 32, false, true, false,
                         cute::SM80_16x8x32_S32U4U4S32_TN)
TL_DEFINE_MMA_DISPATCHER(kUInt4, kUInt4, kInt32, 16, 8, 64, false, true, false,
                         cute::SM80_16x8x64_S32U4U4S32_TN)

#if TL_HAS_F8F6F4_MMA_DISPATCHER
// FP4/F8F6F4 inputs (k32)
TL_DEFINE_MMA_DISPATCHER(kFloat4_e2m1fn, kFloat4_e2m1fn, kFloat32, 16, 8, 32,
                         false, true, false, tl::detail::F8F6F4_FP4_FP4_F32_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat8_e4m3, kFloat4_e2m1fn, kFloat32, 16, 8, 32,
                         false, true, false, tl::detail::F8F6F4_FP8_FP4_F32_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat4_e2m1fn, kFloat8_e4m3, kFloat32, 16, 8, 32,
                         false, true, false, tl::detail::F8F6F4_FP4_FP8_F32_TN)
#endif

// FP8 inputs (k32)
TL_DEFINE_MMA_DISPATCHER(kFloat8_e4m3, kFloat8_e4m3, kFloat16, 16, 8, 32, false,
                         true, false, cute::SM89_16x8x32_F16E4M3E4M3F16_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat8_e4m3, kFloat8_e4m3, kFloat32, 16, 8, 32, false,
                         true, false, cute::SM89_16x8x32_F32E4M3E4M3F32_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat8_e4m3, kFloat8_e5m2, kFloat16, 16, 8, 32, false,
                         true, false, cute::SM89_16x8x32_F16E4M3E5M2F16_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat8_e4m3, kFloat8_e5m2, kFloat32, 16, 8, 32, false,
                         true, false, cute::SM89_16x8x32_F32E4M3E5M2F32_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat8_e5m2, kFloat8_e4m3, kFloat16, 16, 8, 32, false,
                         true, false, cute::SM89_16x8x32_F16E5M2E4M3F16_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat8_e5m2, kFloat8_e4m3, kFloat32, 16, 8, 32, false,
                         true, false, cute::SM89_16x8x32_F32E5M2E4M3F32_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat8_e5m2, kFloat8_e5m2, kFloat16, 16, 8, 32, false,
                         true, false, cute::SM89_16x8x32_F16E5M2E5M2F16_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat8_e5m2, kFloat8_e5m2, kFloat32, 16, 8, 32, false,
                         true, false, cute::SM89_16x8x32_F32E5M2E5M2F32_TN)

// TF32 inputs (FP32 math on Tensor Cores)
// Support both k=4 and k=8 variants on SM80
TL_DEFINE_MMA_DISPATCHER(kTensorFloat32, kTensorFloat32, kFloat32, 16, 8, 4,
                         false, true, false,
                         cute::SM80_16x8x4_F32TF32TF32F32_TN)
TL_DEFINE_MMA_DISPATCHER(kTensorFloat32, kTensorFloat32, kFloat32, 16, 8, 8,
                         false, true, false,
                         cute::SM80_16x8x8_F32TF32TF32F32_TN)

// FP64 inputs (DMMA: m8n8k4, TN layout)
TL_DEFINE_MMA_DISPATCHER(kFloat64, kFloat64, kFloat64, 8, 8, 4, false, true,
                         false, cute::SM80_8x8x4_F64F64F64F64_TN)

#undef TL_DEFINE_MMA_DISPATCHER
#undef TL_HAS_F8F6F4_MMA_DISPATCHER

} // namespace detail

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB, bool Saturate = false>
TL_DEVICE void mma_sync(
    typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA, TransB,
                                   Saturate>::CRegType *c,
    const typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                         TransB, Saturate>::ARegType *a,
    const typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                         TransB, Saturate>::BRegType *b) {
  using Dispatcher = detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                           TransB, Saturate>;
  static_assert(!std::is_void_v<typename Dispatcher::CRegType>,
                "tl::mma_sync: unsupported configuration");
  Dispatcher::exec(c, a, b, c);
}

} // namespace tl
