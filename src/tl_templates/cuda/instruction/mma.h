#pragma once

#include "../common.h"
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm89.hpp>

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

// BF16 inputs
TL_DEFINE_MMA_DISPATCHER(kBFloat16, kBFloat16, kFloat32, 16, 8, 16, false, true,
                         false, cute::SM80_16x8x16_F32BF16BF16F32_TN)

// INT8 inputs (k32)
TL_DEFINE_MMA_DISPATCHER(kInt8, kInt8, kInt32, 16, 8, 32, false, true, false,
                         cute::SM80_16x8x32_S32S8S8S32_TN)
TL_DEFINE_MMA_DISPATCHER(kUInt8, kUInt8, kInt32, 16, 8, 32, false, true, false,
                         cute::SM80_16x8x32_S32U8U8S32_TN)

// INT4 inputs (k32)
TL_DEFINE_MMA_DISPATCHER(kInt4, kInt4, kInt32, 16, 8, 32, false, true, false,
                         cute::SM80_16x8x32_S32S4S4S32_TN)
TL_DEFINE_MMA_DISPATCHER(kUInt4, kUInt4, kInt32, 16, 8, 32, false, true, false,
                         cute::SM80_16x8x32_S32U4U4S32_TN)

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
