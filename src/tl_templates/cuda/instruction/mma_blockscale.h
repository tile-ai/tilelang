#pragma once

#include "../common.h"
#include <cute/arch/mma_sm120.hpp>
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

template <class Impl> struct MmaBlockScaleImplTraits {
  using DReg = std::remove_extent_t<typename Impl::DRegisters>;
  using AReg = std::remove_extent_t<typename Impl::ARegisters>;
  using BReg = std::remove_extent_t<typename Impl::BRegisters>;
  using CReg = std::remove_extent_t<typename Impl::CRegisters>;
  using SFAReg = std::remove_extent_t<typename Impl::SFARegisters>;
  using SFBReg = std::remove_extent_t<typename Impl::SFBRegisters>;

  static constexpr int kDRegs = std::extent_v<typename Impl::DRegisters>;
  static constexpr int kARegs = std::extent_v<typename Impl::ARegisters>;
  static constexpr int kBRegs = std::extent_v<typename Impl::BRegisters>;
  static constexpr int kCRegs = std::extent_v<typename Impl::CRegisters>;
  static constexpr int kSFARegs = std::extent_v<typename Impl::SFARegisters>;
  static constexpr int kSFBRegs = std::extent_v<typename Impl::SFBRegisters>;
};

template <class Impl, size_t... DIdx, size_t... AIdx, size_t... BIdx,
          size_t... CIdx, size_t... SFAIdx, size_t... SFBIdx>
TL_DEVICE void call_fma_blockscale_impl(
    typename MmaBlockScaleImplTraits<Impl>::DReg *d,
    const typename MmaBlockScaleImplTraits<Impl>::AReg *a,
    const typename MmaBlockScaleImplTraits<Impl>::BReg *b,
    const typename MmaBlockScaleImplTraits<Impl>::CReg *c,
    const typename MmaBlockScaleImplTraits<Impl>::SFAReg *sfa,
    const typename MmaBlockScaleImplTraits<Impl>::SFBReg *sfb, uint16_t id_a,
    uint16_t id_b, std::index_sequence<DIdx...>, std::index_sequence<AIdx...>,
    std::index_sequence<BIdx...>, std::index_sequence<CIdx...>,
    std::index_sequence<SFAIdx...>, std::index_sequence<SFBIdx...>) {
  Impl::fma(d[DIdx]..., a[AIdx]..., b[BIdx]..., c[CIdx]..., sfa[SFAIdx]...,
            sfb[SFBIdx]..., id_a, id_b);
}

template <class Impl>
TL_DEVICE void
call_fma_blockscale(typename MmaBlockScaleImplTraits<Impl>::DReg *d,
                    const typename MmaBlockScaleImplTraits<Impl>::AReg *a,
                    const typename MmaBlockScaleImplTraits<Impl>::BReg *b,
                    const typename MmaBlockScaleImplTraits<Impl>::CReg *c,
                    const typename MmaBlockScaleImplTraits<Impl>::SFAReg *sfa,
                    const typename MmaBlockScaleImplTraits<Impl>::SFBReg *sfb,
                    uint16_t id_a, uint16_t id_b) {
  call_fma_blockscale_impl<Impl>(
      d, a, b, c, sfa, sfb, id_a, id_b,
      std::make_index_sequence<MmaBlockScaleImplTraits<Impl>::kDRegs>{},
      std::make_index_sequence<MmaBlockScaleImplTraits<Impl>::kARegs>{},
      std::make_index_sequence<MmaBlockScaleImplTraits<Impl>::kBRegs>{},
      std::make_index_sequence<MmaBlockScaleImplTraits<Impl>::kCRegs>{},
      std::make_index_sequence<MmaBlockScaleImplTraits<Impl>::kSFARegs>{},
      std::make_index_sequence<MmaBlockScaleImplTraits<Impl>::kSFBRegs>{});
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define TL_SM120_BLOCKSCALE_ASM(KindAsm, ScaleVecAsm, ShapeAsm, AAsm, BAsm,    \
                                SFAsm, d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, \
                                c0, c1, c2, c3, sfa, sfb, id_a, id_b)          \
  asm volatile("mma.sync.aligned.kind::" KindAsm                               \
               ".block_scale.scale_vec::" ScaleVecAsm "." ShapeAsm             \
               ".row.col.f32." AAsm "." BAsm ".f32." SFAsm " "                 \
               "{%0, %1, %2, %3}, "                                            \
               "{%4, %5, %6, %7}, "                                            \
               "{%8, %9}, "                                                    \
               "{%10, %11, %12, %13}, "                                        \
               "{%14}, {%15, %16}, "                                           \
               "{%17}, {%18, %19};"                                            \
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                        \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),         \
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(uint32_t(sfa)),       \
                 "h"((short)0), "h"((short)id_a), "r"(uint32_t(sfb)),          \
                 "h"((short)0), "h"((short)id_b));
#else
#define TL_SM120_BLOCKSCALE_ASM(KindAsm, ScaleVecAsm, ShapeAsm, AAsm, BAsm,    \
                                SFAsm, d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, \
                                c0, c1, c2, c3, sfa, sfb, id_a, id_b)          \
  CUTE_INVALID_CONTROL_PATH(                                                   \
      "Attempting to use SM120 block-scaled MMA without SM120+");
#endif

#define TL_DEFINE_SM120_BLOCKSCALE_IMPL(                                       \
    ImplName, KindAsm, ScaleVecAsm, ShapeAsm, AAsm, BAsm, SFAsm, ScaleRegType) \
  struct ImplName {                                                            \
    using DRegisters = float[4];                                               \
    using ARegisters = uint32_t[4];                                            \
    using BRegisters = uint32_t[2];                                            \
    using CRegisters = float[4];                                               \
    using SFARegisters = ScaleRegType[1];                                      \
    using SFBRegisters = ScaleRegType[1];                                      \
                                                                               \
    static TL_DEVICE void                                                      \
    fma(float &d0, float &d1, float &d2, float &d3, uint32_t const &a0,        \
        uint32_t const &a1, uint32_t const &a2, uint32_t const &a3,            \
        uint32_t const &b0, uint32_t const &b1, float const &c0,               \
        float const &c1, float const &c2, float const &c3,                     \
        ScaleRegType const &sfa, ScaleRegType const &sfb, uint16_t id_a,       \
        uint16_t id_b) {                                                       \
      TL_SM120_BLOCKSCALE_ASM(KindAsm, ScaleVecAsm, ShapeAsm, AAsm, BAsm,      \
                              SFAsm, d0, d1, d2, d3, a0, a1, a2, a3, b0, b1,   \
                              c0, c1, c2, c3, sfa, sfb, id_a, id_b)            \
    }                                                                          \
  };

#define TL_SM120_MXF8F6F4_IMPL_NAME_(AName, BName)                             \
  SM120_16x8x32_F32##AName##BName##F32_UE8M0_TN

#define TL_SM120_MXF8F6F4_IMPL_NAME_EXPANDED(AName, BName)                     \
  TL_SM120_MXF8F6F4_IMPL_NAME_(AName, BName)

#define TL_SM120_MXF8F6F4_IMPL_NAME(ATypeEnum, BTypeEnum)                      \
  TL_SM120_MXF8F6F4_IMPL_NAME_EXPANDED(TL_BLOCKSCALE_DTYPE_ASM(ATypeEnum),     \
                                       TL_BLOCKSCALE_DTYPE_ASM(BTypeEnum))

#define TL_SM120_MXF4NVF4_IMPL_NAME_(ScaleName)                                \
  SM120_16x8x64_F32_e2m1_e2m1_F32_##ScaleName##_TN

#define TL_BLOCKSCALE_DTYPE_ASM_kFloat4_e2m1fn e2m1
#define TL_BLOCKSCALE_DTYPE_ASM_kFloat6_e2m3fn e2m3
#define TL_BLOCKSCALE_DTYPE_ASM_kFloat6_e3m2fn e3m2
#define TL_BLOCKSCALE_DTYPE_ASM_kFloat8_e4m3 e4m3
#define TL_BLOCKSCALE_DTYPE_ASM_kFloat8_e5m2 e5m2

#define TL_BLOCKSCALE_DTYPE_ASM_(TypeEnum) TL_BLOCKSCALE_DTYPE_ASM_##TypeEnum
#define TL_BLOCKSCALE_DTYPE_ASM(TypeEnum) TL_BLOCKSCALE_DTYPE_ASM_(TypeEnum)

#define TL_SM120_BLOCKSCALE_DTYPE_ROWS(M)                                      \
  M(kFloat4_e2m1fn)                                                            \
  M(kFloat6_e2m3fn)                                                            \
  M(kFloat6_e3m2fn)                                                            \
  M(kFloat8_e4m3)                                                              \
  M(kFloat8_e5m2)

#define TL_SM120_BLOCKSCALE_DTYPE_COLS(M, ATypeEnum)                           \
  M(ATypeEnum, kFloat4_e2m1fn)                                                 \
  M(ATypeEnum, kFloat6_e2m3fn)                                                 \
  M(ATypeEnum, kFloat6_e3m2fn)                                                 \
  M(ATypeEnum, kFloat8_e4m3)                                                   \
  M(ATypeEnum, kFloat8_e5m2)

#define TL_BLOCKSCALE_STRINGIZE_(X) #X
#define TL_BLOCKSCALE_STRINGIZE(X) TL_BLOCKSCALE_STRINGIZE_(X)

#define TL_DEFINE_SM120_MXF8F6F4_IMPL(ATypeEnum, BTypeEnum)                    \
  TL_DEFINE_SM120_BLOCKSCALE_IMPL(                                             \
      TL_SM120_MXF8F6F4_IMPL_NAME(ATypeEnum, BTypeEnum), "mxf8f6f4", "1X",     \
      "m16n8k32", TL_BLOCKSCALE_STRINGIZE(TL_BLOCKSCALE_DTYPE_ASM(ATypeEnum)), \
      TL_BLOCKSCALE_STRINGIZE(TL_BLOCKSCALE_DTYPE_ASM(BTypeEnum)), "ue8m0",    \
      uint8_t)                                                                 \
  TL_DEFINE_MMA_BLOCKSCALE_DISPATCHER(                                         \
      ATypeEnum, BTypeEnum, kFloat32, 16, 8, 32, false, true, 1,               \
      tl::detail::TL_SM120_MXF8F6F4_IMPL_NAME(ATypeEnum, BTypeEnum))

#define TL_DEFINE_SM120_MXF4NVF4_IMPL(ScaleVecValue, ScaleVecAsm, ScaleName)   \
  TL_DEFINE_SM120_BLOCKSCALE_IMPL(TL_SM120_MXF4NVF4_IMPL_NAME_(ScaleName),     \
                                  "mxf4nvf4", ScaleVecAsm, "m16n8k64", "e2m1", \
                                  "e2m1", TL_BLOCKSCALE_STRINGIZE(ScaleName),  \
                                  uint32_t)                                    \
  TL_DEFINE_MMA_BLOCKSCALE_DISPATCHER(                                         \
      kFloat4_e2m1fn, kFloat4_e2m1fn, kFloat32, 16, 8, 64, false, true,        \
      ScaleVecValue, tl::detail::TL_SM120_MXF4NVF4_IMPL_NAME_(ScaleName))

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB, int ScaleVec>
struct MmaBlockScaleDispatcher {
  using CRegType = void;
  using ARegType = void;
  using BRegType = void;
  using SFARegType = void;
  using SFBRegType = void;

  static TL_DEVICE void exec(CRegType *, const ARegType *, const BRegType *,
                             const CRegType *, const SFARegType *,
                             const SFBRegType *, uint16_t, uint16_t) {
    static_assert(always_false_v<std::integral_constant<int, M>>,
                  "tl::mma_sync_blockscale: unsupported configuration");
  }
};

#define TL_DEFINE_MMA_BLOCKSCALE_DISPATCHER(                                   \
    ATypeEnum, BTypeEnum, CTypeEnum, MValue, NValue, KValue, TransAValue,      \
    TransBValue, ScaleVecValue, ImplType)                                      \
  template <>                                                                  \
  struct MmaBlockScaleDispatcher<DataType::ATypeEnum, DataType::BTypeEnum,     \
                                 DataType::CTypeEnum, MValue, NValue, KValue,  \
                                 TransAValue, TransBValue, ScaleVecValue> {    \
    using Impl = ImplType;                                                     \
    using Traits = MmaBlockScaleImplTraits<Impl>;                              \
    using CRegType = typename Traits::DReg;                                    \
    using ARegType = typename Traits::AReg;                                    \
    using BRegType = typename Traits::BReg;                                    \
    using SFARegType = typename Traits::SFAReg;                                \
    using SFBRegType = typename Traits::SFBReg;                                \
    static_assert(                                                             \
        std::is_same_v<typename Traits::DReg, typename Traits::CReg>,          \
        "tl::mma_sync_blockscale requires matching accumulator/output regs");  \
    static TL_DEVICE void exec(CRegType *d, const ARegType *a,                 \
                               const BRegType *b, const CRegType *c,           \
                               const SFARegType *sfa, const SFBRegType *sfb,   \
                               uint16_t id_a, uint16_t id_b) {                 \
      call_fma_blockscale<Impl>(d, a, b, c, sfa, sfb, id_a, id_b);             \
    }                                                                          \
  };

#define TL_DEFINE_SM120_MXF8F6F4_ROW(ATypeEnum)                                \
  TL_SM120_BLOCKSCALE_DTYPE_COLS(TL_DEFINE_SM120_MXF8F6F4_IMPL, ATypeEnum)

// SM120 dense block-scaled MMA, mxf8f6f4 scale_vec::1X, ue8m0 scale factors.
TL_SM120_BLOCKSCALE_DTYPE_ROWS(TL_DEFINE_SM120_MXF8F6F4_ROW)

// SM120 dense block-scaled MMA, mxf4nvf4 e2m1 x e2m1.
TL_DEFINE_SM120_MXF4NVF4_IMPL(2, "2X", ue8m0)
TL_DEFINE_SM120_MXF4NVF4_IMPL(4, "4X", ue4m3)

#undef TL_DEFINE_MMA_BLOCKSCALE_DISPATCHER
#undef TL_DEFINE_SM120_MXF8F6F4_ROW
#undef TL_DEFINE_SM120_MXF8F6F4_IMPL
#undef TL_DEFINE_SM120_MXF4NVF4_IMPL
#undef TL_SM120_MXF4NVF4_IMPL_NAME_
#undef TL_SM120_MXF8F6F4_IMPL_NAME
#undef TL_SM120_MXF8F6F4_IMPL_NAME_EXPANDED
#undef TL_SM120_MXF8F6F4_IMPL_NAME_
#undef TL_BLOCKSCALE_STRINGIZE
#undef TL_BLOCKSCALE_STRINGIZE_
#undef TL_SM120_BLOCKSCALE_DTYPE_COLS
#undef TL_SM120_BLOCKSCALE_DTYPE_ROWS
#undef TL_BLOCKSCALE_DTYPE_ASM
#undef TL_BLOCKSCALE_DTYPE_ASM_
#undef TL_BLOCKSCALE_DTYPE_ASM_kFloat8_e5m2
#undef TL_BLOCKSCALE_DTYPE_ASM_kFloat8_e4m3
#undef TL_BLOCKSCALE_DTYPE_ASM_kFloat6_e3m2fn
#undef TL_BLOCKSCALE_DTYPE_ASM_kFloat6_e2m3fn
#undef TL_BLOCKSCALE_DTYPE_ASM_kFloat4_e2m1fn
#undef TL_DEFINE_SM120_BLOCKSCALE_IMPL
#undef TL_SM120_BLOCKSCALE_ASM

} // namespace detail

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB, int ScaleVec>
TL_DEVICE void mma_sync_blockscale(
    typename detail::MmaBlockScaleDispatcher<
        AType, BType, CType, M, N, K, TransA, TransB, ScaleVec>::CRegType *c,
    const typename detail::MmaBlockScaleDispatcher<
        AType, BType, CType, M, N, K, TransA, TransB, ScaleVec>::ARegType *a,
    const typename detail::MmaBlockScaleDispatcher<
        AType, BType, CType, M, N, K, TransA, TransB, ScaleVec>::BRegType *b,
    typename detail::MmaBlockScaleDispatcher<
        AType, BType, CType, M, N, K, TransA, TransB, ScaleVec>::SFARegType sfa,
    typename detail::MmaBlockScaleDispatcher<
        AType, BType, CType, M, N, K, TransA, TransB, ScaleVec>::SFBRegType sfb,
    uint16_t id_a, uint16_t id_b) {
  using Dispatcher =
      detail::MmaBlockScaleDispatcher<AType, BType, CType, M, N, K, TransA,
                                      TransB, ScaleVec>;
  static_assert(!std::is_void_v<typename Dispatcher::CRegType>,
                "tl::mma_sync_blockscale: unsupported configuration");
  Dispatcher::exec(c, a, b, c, &sfa, &sfb, id_a, id_b);
}

} // namespace tl
