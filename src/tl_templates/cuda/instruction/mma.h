#pragma once

#include "../common.h"
#include <cute/arch/mma_sm120.hpp>
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

template <class Impl> struct BlockScaledMmaImplTraits : MmaImplTraits<Impl> {
  using SFReg = std::remove_extent_t<typename Impl::SFARegisters>;

  static constexpr int kSFARegs = std::extent_v<typename Impl::SFARegisters>;
  static constexpr int kSFBRegs = std::extent_v<typename Impl::SFBRegisters>;
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

template <class Impl, size_t... DIdx, size_t... AIdx, size_t... BIdx,
          size_t... CIdx, size_t... SFAIdx, size_t... SFBIdx>
TL_DEVICE void call_blockscaled_fma_impl(
    typename BlockScaledMmaImplTraits<Impl>::DReg *d,
    const typename BlockScaledMmaImplTraits<Impl>::AReg *a,
    const typename BlockScaledMmaImplTraits<Impl>::BReg *b,
    const typename BlockScaledMmaImplTraits<Impl>::CReg *c,
    const typename BlockScaledMmaImplTraits<Impl>::SFReg *sfa,
    const typename BlockScaledMmaImplTraits<Impl>::SFReg *sfb,
    std::index_sequence<DIdx...>, std::index_sequence<AIdx...>,
    std::index_sequence<BIdx...>, std::index_sequence<CIdx...>,
    std::index_sequence<SFAIdx...>, std::index_sequence<SFBIdx...>) {
  Impl::fma(d[DIdx]..., a[AIdx]..., b[BIdx]..., c[CIdx]..., sfa[SFAIdx]...,
            sfb[SFBIdx]...);
}

template <class Impl>
TL_DEVICE void
call_blockscaled_fma(typename BlockScaledMmaImplTraits<Impl>::DReg *d,
                     const typename BlockScaledMmaImplTraits<Impl>::AReg *a,
                     const typename BlockScaledMmaImplTraits<Impl>::BReg *b,
                     const typename BlockScaledMmaImplTraits<Impl>::CReg *c,
                     const typename BlockScaledMmaImplTraits<Impl>::SFReg *sfa,
                     const typename BlockScaledMmaImplTraits<Impl>::SFReg *sfb) {
  call_blockscaled_fma_impl<Impl>(
      d, a, b, c, sfa, sfb,
      std::make_index_sequence<BlockScaledMmaImplTraits<Impl>::kDRegs>{},
      std::make_index_sequence<BlockScaledMmaImplTraits<Impl>::kARegs>{},
      std::make_index_sequence<BlockScaledMmaImplTraits<Impl>::kBRegs>{},
      std::make_index_sequence<BlockScaledMmaImplTraits<Impl>::kCRegs>{},
      std::make_index_sequence<BlockScaledMmaImplTraits<Impl>::kSFARegs>{},
      std::make_index_sequence<BlockScaledMmaImplTraits<Impl>::kSFBRegs>{});
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

template <DataType AType, DataType BType, DataType CType, DataType SFType,
          int M, int N, int K, bool TransA, bool TransB, int VS>
struct BlockScaledMmaDispatcher {
  using CRegType = void;
  using ARegType = void;
  using BRegType = void;
  using SFRegType = void;

  static TL_DEVICE void exec(CRegType *, const ARegType *, const BRegType *,
                             const CRegType *, const SFRegType *,
                             const SFRegType *) {
    static_assert(always_false_v<std::integral_constant<int, M>>,
                  "tl::mma_sync_blockscaled: unsupported configuration");
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

#define TL_DEFINE_BLOCKSCALED_MMA_DISPATCHER(                                  \
    ATypeEnum, BTypeEnum, CTypeEnum, SFTypeEnum, MValue, NValue, KValue,       \
    TransAValue, TransBValue, VSValue, ImplType)                                \
  template <>                                                                  \
  struct BlockScaledMmaDispatcher<                                             \
      DataType::ATypeEnum, DataType::BTypeEnum, DataType::CTypeEnum,           \
      DataType::SFTypeEnum, MValue, NValue, KValue, TransAValue, TransBValue,  \
      VSValue> {                                                               \
    using Impl = ImplType;                                                     \
    using Traits = BlockScaledMmaImplTraits<Impl>;                             \
    using CRegType = typename Traits::DReg;                                    \
    using ARegType = typename Traits::AReg;                                    \
    using BRegType = typename Traits::BReg;                                    \
    using SFRegType = typename Traits::SFReg;                                  \
    static_assert(                                                             \
        std::is_same_v<typename Traits::DReg, typename Traits::CReg>,          \
        "tl::mma_sync_blockscaled requires matching accumulator/output regs"); \
    static TL_DEVICE void exec(CRegType *d, const ARegType *a,                 \
                               const BRegType *b, const CRegType *c,           \
                               const SFRegType *sfa, const SFRegType *sfb) {   \
      call_blockscaled_fma<Impl>(d, a, b, c, sfa, sfb);                        \
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

// INT4 inputs (k32, k64)
TL_DEFINE_MMA_DISPATCHER(kInt4, kInt4, kInt32, 16, 8, 32, false, true, false,
                         cute::SM80_16x8x32_S32S4S4S32_TN)
TL_DEFINE_MMA_DISPATCHER(kInt4, kInt4, kInt32, 16, 8, 64, false, true, false,
                         cute::SM80_16x8x64_S32S4S4S32_TN)
TL_DEFINE_MMA_DISPATCHER(kUInt4, kUInt4, kInt32, 16, 8, 32, false, true, false,
                         cute::SM80_16x8x32_S32U4U4S32_TN)
TL_DEFINE_MMA_DISPATCHER(kUInt4, kUInt4, kInt32, 16, 8, 64, false, true, false,
                         cute::SM80_16x8x64_S32U4U4S32_TN)

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

// FP4 inputs (k32, SM120 kind::f8f6f4)
using SM120_FP4_FP4_F32_TN =
    cute::SM120_16x8x32_TN<cute::float_e2m1_t, cute::float_e2m1_t, float>;
TL_DEFINE_MMA_DISPATCHER(kFloat4_e2m1fn, kFloat4_e2m1fn, kFloat32, 16, 8, 32,
                         false, true, false, SM120_FP4_FP4_F32_TN)

// Mixed FP8 x FP4 and FP4 x FP8 (k32, SM120 kind::f8f6f4)
using SM120_FP8_FP4_F32_TN =
    cute::SM120_16x8x32_TN<cute::float_e4m3_t, cute::float_e2m1_t, float>;
using SM120_FP4_FP8_F32_TN =
    cute::SM120_16x8x32_TN<cute::float_e2m1_t, cute::float_e4m3_t, float>;
TL_DEFINE_MMA_DISPATCHER(kFloat8_e4m3, kFloat4_e2m1fn, kFloat32, 16, 8, 32,
                         false, true, false, SM120_FP8_FP4_F32_TN)
TL_DEFINE_MMA_DISPATCHER(kFloat4_e2m1fn, kFloat8_e4m3, kFloat32, 16, 8, 32,
                         false, true, false, SM120_FP4_FP8_F32_TN)

// NVFP4 block-scaled MMA (SM120 kind::mxf4nvf4.block_scale).
using SM120_NVFP4_NVFP4_F32_UE4M3_TN =
    cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
        cute::float_e2m1_t, cute::float_e2m1_t, float, cute::float_ue4m3_t, 32>;
using SM120_NVFP4_NVFP4_F32_UE4M3_VS16_TN =
    cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
        cute::float_e2m1_t, cute::float_e2m1_t, float, cute::float_ue4m3_t, 16>;
TL_DEFINE_BLOCKSCALED_MMA_DISPATCHER(
    kFloat4_e2m1fn, kFloat4_e2m1fn, kFloat32, kFloat8_e4m3, 16, 8, 64, false,
    true, 32, SM120_NVFP4_NVFP4_F32_UE4M3_TN)
TL_DEFINE_BLOCKSCALED_MMA_DISPATCHER(
    kFloat4_e2m1fn, kFloat4_e2m1fn, kFloat32, kFloat8_e4m3, 16, 8, 64, false,
    true, 16, SM120_NVFP4_NVFP4_F32_UE4M3_VS16_TN)

#undef TL_DEFINE_MMA_DISPATCHER
#undef TL_DEFINE_BLOCKSCALED_MMA_DISPATCHER

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
  if constexpr (AType == DataType::kFloat4_e2m1fn ||
                BType == DataType::kFloat4_e2m1fn) {
    using AReg = typename Dispatcher::ARegType;
    using BReg = typename Dispatcher::BRegType;
    constexpr int nA = detail::MmaImplTraits<typename Dispatcher::Impl>::kARegs;
    constexpr int nB = detail::MmaImplTraits<typename Dispatcher::Impl>::kBRegs;
    AReg as[nA];
    BReg bs[nB];
#pragma unroll
    for (int i = 0; i < nA; ++i)
      as[i] = (AType == DataType::kFloat4_e2m1fn) ? (a[i] << 2) : a[i];
#pragma unroll
    for (int i = 0; i < nB; ++i)
      bs[i] = (BType == DataType::kFloat4_e2m1fn) ? (b[i] << 2) : b[i];
    Dispatcher::exec(c, as, bs, c);
  } else {
    Dispatcher::exec(c, a, b, c);
  }
}

template <DataType AType, DataType BType, DataType CType, DataType SFType,
          int M, int N, int K, bool TransA, bool TransB, int VS>
TL_DEVICE void mma_sync_blockscaled(
    typename detail::BlockScaledMmaDispatcher<
        AType, BType, CType, SFType, M, N, K, TransA, TransB, VS>::CRegType *c,
    const typename detail::BlockScaledMmaDispatcher<
        AType, BType, CType, SFType, M, N, K, TransA, TransB, VS>::ARegType *a,
    const typename detail::BlockScaledMmaDispatcher<
        AType, BType, CType, SFType, M, N, K, TransA, TransB, VS>::BRegType *b,
    const typename detail::BlockScaledMmaDispatcher<
        AType, BType, CType, SFType, M, N, K, TransA, TransB, VS>::SFRegType
        *sfa,
    const typename detail::BlockScaledMmaDispatcher<
        AType, BType, CType, SFType, M, N, K, TransA, TransB, VS>::SFRegType
        *sfb) {
  using Dispatcher = detail::BlockScaledMmaDispatcher<
      AType, BType, CType, SFType, M, N, K, TransA, TransB, VS>;
  static_assert(!std::is_void_v<typename Dispatcher::CRegType>,
                "tl::mma_sync_blockscaled: unsupported configuration");

  Dispatcher::exec(c, a, b, c, sfa, sfb);
}

} // namespace tl
