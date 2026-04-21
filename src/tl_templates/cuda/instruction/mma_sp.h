#pragma once

#include "../common.h"
#include "cute_extension/mma_sm80_sparse.hpp"

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

template <class Impl> struct MmaSpImplTraits {
  using DReg = std::remove_extent_t<typename Impl::DRegisters>;
  using AReg = std::remove_extent_t<typename Impl::ARegisters>;
  using BReg = std::remove_extent_t<typename Impl::BRegisters>;
  using CReg = std::remove_extent_t<typename Impl::CRegisters>;
  using EReg = std::remove_extent_t<typename Impl::ERegisters>;

  static constexpr int kDRegs = std::extent_v<typename Impl::DRegisters>;
  static constexpr int kARegs = std::extent_v<typename Impl::ARegisters>;
  static constexpr int kBRegs = std::extent_v<typename Impl::BRegisters>;
  static constexpr int kCRegs = std::extent_v<typename Impl::CRegisters>;
  static constexpr int kERegs = std::extent_v<typename Impl::ERegisters>;
};

template <class Impl, size_t... DIdx, size_t... AIdx, size_t... BIdx,
          size_t... CIdx, size_t... EIdx>
TL_DEVICE void
call_fma_sp_impl(typename MmaSpImplTraits<Impl>::DReg *d,
                 const typename MmaSpImplTraits<Impl>::AReg *a,
                 const typename MmaSpImplTraits<Impl>::BReg *b,
                 const typename MmaSpImplTraits<Impl>::CReg *c,
                 const typename MmaSpImplTraits<Impl>::EReg *e,
                 std::index_sequence<DIdx...>, std::index_sequence<AIdx...>,
                 std::index_sequence<BIdx...>, std::index_sequence<CIdx...>,
                 std::index_sequence<EIdx...>) {
  Impl::fma(d[DIdx]..., a[AIdx]..., b[BIdx]..., c[CIdx]..., e[EIdx]...);
}

template <class Impl>
TL_DEVICE void call_fma_sp(typename MmaSpImplTraits<Impl>::DReg *d,
                           const typename MmaSpImplTraits<Impl>::AReg *a,
                           const typename MmaSpImplTraits<Impl>::BReg *b,
                           const typename MmaSpImplTraits<Impl>::CReg *c,
                           const typename MmaSpImplTraits<Impl>::EReg *e) {
  call_fma_sp_impl<Impl>(
      d, a, b, c, e, std::make_index_sequence<MmaSpImplTraits<Impl>::kDRegs>{},
      std::make_index_sequence<MmaSpImplTraits<Impl>::kARegs>{},
      std::make_index_sequence<MmaSpImplTraits<Impl>::kBRegs>{},
      std::make_index_sequence<MmaSpImplTraits<Impl>::kCRegs>{},
      std::make_index_sequence<MmaSpImplTraits<Impl>::kERegs>{});
}

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB,
          SM80::MMA::SparseSel spsel = SM80::MMA::SparseSel::Zero,
          typename MetaType = uint32_t>
struct MmaSpDispatcher {
  using CRegType = void;
  using ARegType = void;
  using BRegType = void;
  using ERegType = MetaType;

  static TL_DEVICE void exec(CRegType *, const ARegType *, const BRegType *,
                             const CRegType *, const ERegType *) {
    static_assert(always_false_v<std::integral_constant<int, M>>,
                  "tl::mma_sp_sync: unsupported configuration");
  }
};

#define TL_DEFINE_MMA_SP_DISPATCHER(ATypeEnum, BTypeEnum, CTypeEnum, MValue,   \
                                    NValue, KValue, TransAValue, TransBValue,  \
                                    ImplTemplate)                              \
  template <SM80::MMA::SparseSel spsel, typename MetaType>                     \
  struct MmaSpDispatcher<DataType::ATypeEnum, DataType::BTypeEnum,             \
                         DataType::CTypeEnum, MValue, NValue, KValue,          \
                         TransAValue, TransBValue, spsel, MetaType> {          \
    using Impl = ImplTemplate<spsel>;                                          \
    using Traits = MmaSpImplTraits<Impl>;                                      \
    using CRegType = typename Traits::DReg;                                    \
    using ARegType = typename Traits::AReg;                                    \
    using BRegType = typename Traits::BReg;                                    \
    using ERegType = MetaType;                                                 \
    static_assert(                                                             \
        std::is_same_v<typename Traits::DReg, typename Traits::CReg>,          \
        "tl::mma_sp_sync requires matching accumulator/output regs");          \
    static TL_DEVICE void exec(CRegType *d, const ARegType *a,                 \
                               const BRegType *b, const CRegType *c,           \
                               const MetaType *e) {                            \
      call_fma_sp<Impl>(d, a, b, c,                                            \
                        reinterpret_cast<const typename Traits::EReg *>(e));   \
    }                                                                          \
  };

// FP16 — logical K=16 (A holds K/2=8 actual elements, 2 regs)
TL_DEFINE_MMA_SP_DISPATCHER(kFloat16, kFloat16, kFloat16, 16, 8, 16, false,
                            true,
                            SM80::MMA::SPARSE::SM80_16x8x16_F16F16F16F16_TN)
TL_DEFINE_MMA_SP_DISPATCHER(kFloat16, kFloat16, kFloat32, 16, 8, 16, false,
                            true,
                            SM80::MMA::SPARSE::SM80_16x8x16_F32F16F16F32_TN)

// FP16 — logical K=32 (A holds K/2=16 actual elements, 4 regs)
TL_DEFINE_MMA_SP_DISPATCHER(kFloat16, kFloat16, kFloat16, 16, 8, 32, false,
                            true,
                            SM80::MMA::SPARSE::SM80_16x8x32_F16F16F16F16_TN)
TL_DEFINE_MMA_SP_DISPATCHER(kFloat16, kFloat16, kFloat32, 16, 8, 32, false,
                            true,
                            SM80::MMA::SPARSE::SM80_16x8x32_F32F16F16F32_TN)

// BF16 — logical K=16
TL_DEFINE_MMA_SP_DISPATCHER(kBFloat16, kBFloat16, kFloat32, 16, 8, 16, false,
                            true,
                            SM80::MMA::SPARSE::SM80_16x8x16_F32BF16BF16F32_TN)

// BF16 — logical K=32
TL_DEFINE_MMA_SP_DISPATCHER(kBFloat16, kBFloat16, kFloat32, 16, 8, 32, false,
                            true,
                            SM80::MMA::SPARSE::SM80_16x8x32_F32BF16BF16F32_TN)

// TF32 — logical K=8 (A holds K/2=4 actual elements, 2 regs)
TL_DEFINE_MMA_SP_DISPATCHER(kTensorFloat32, kTensorFloat32, kFloat32, 16, 8, 8,
                            false, true,
                            SM80::MMA::SPARSE::SM80_16x8x8_F32TF32TF32F32_TN)

// TF32 — logical K=16 (A holds K/2=8 actual elements, 4 regs)
TL_DEFINE_MMA_SP_DISPATCHER(kTensorFloat32, kTensorFloat32, kFloat32, 16, 8, 16,
                            false, true,
                            SM80::MMA::SPARSE::SM80_16x8x16_F32TF32TF32F32_TN)

// INT8 — logical K=32 (A holds K/2=16, 2 regs); SparseSel::One is invalid
TL_DEFINE_MMA_SP_DISPATCHER(kInt8, kInt8, kInt32, 16, 8, 32, false, true,
                            SM80::MMA::SPARSE::SM80_16x8x32_S32S8S8S32_TN)
TL_DEFINE_MMA_SP_DISPATCHER(kUInt8, kUInt8, kInt32, 16, 8, 32, false, true,
                            SM80::MMA::SPARSE::SM80_16x8x32_S32U8U8S32_TN)

// INT8 — logical K=64 (A holds K/2=32, 4 regs); SparseSel::One is invalid
TL_DEFINE_MMA_SP_DISPATCHER(kInt8, kInt8, kInt32, 16, 8, 64, false, true,
                            SM80::MMA::SPARSE::SM80_16x8x64_S32S8S8S32_TN)
TL_DEFINE_MMA_SP_DISPATCHER(kUInt8, kUInt8, kInt32, 16, 8, 64, false, true,
                            SM80::MMA::SPARSE::SM80_16x8x64_S32U8U8S32_TN)

#undef TL_DEFINE_MMA_SP_DISPATCHER

} // namespace detail

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB,
          SM80::MMA::SparseSel spsel = SM80::MMA::SparseSel::Zero,
          typename MetaType = uint32_t>
TL_DEVICE void mma_sp_sync(
    typename detail::MmaSpDispatcher<AType, BType, CType, M, N, K, TransA,
                                     TransB, spsel, MetaType>::CRegType *c,
    const typename detail::MmaSpDispatcher<AType, BType, CType, M, N, K, TransA,
                                           TransB, spsel, MetaType>::ARegType
        *a,
    const typename detail::MmaSpDispatcher<AType, BType, CType, M, N, K, TransA,
                                           TransB, spsel, MetaType>::BRegType
        *b,
    const MetaType *e) {
  using Dispatcher = detail::MmaSpDispatcher<AType, BType, CType, M, N, K,
                                             TransA, TransB, spsel, MetaType>;
  static_assert(!std::is_void_v<typename Dispatcher::CRegType>,
                "tl::mma_sp_sync: unsupported configuration");
  Dispatcher::exec(c, a, b, c, e);
}

} // namespace tl
