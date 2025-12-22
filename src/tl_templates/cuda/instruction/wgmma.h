#pragma once

#include "../common.h"
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>

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

template <bool IsMnMajor> struct MajorValue {
  static constexpr auto value =
      IsMnMajor ? cute::SM90::GMMA::Major::MN : cute::SM90::GMMA::Major::K;
};

template <int Scale> struct ScaleInValue {
  static_assert(Scale == 1 || Scale == -1,
                "tl::wgmma requires scale factors of +1 or -1.");
  static constexpr auto value = Scale == 1 ? cute::SM90::GMMA::ScaleIn::One
                                           : cute::SM90::GMMA::ScaleIn::Neg;
};

template <int Scale>
inline constexpr bool IsValidScale = (Scale == 1 || Scale == -1);

template <class Impl> struct CallWgmmaSS {
  using CReg = std::remove_extent_t<typename Impl::CRegisters>;
  static constexpr int kCRegs = std::extent_v<typename Impl::CRegisters>;
  static_assert(sizeof(CReg) == sizeof(uint32_t),
                "tl::wgmma_ss expects 32-bit accumulator registers.");

  template <size_t... Idx>
  TL_DEVICE static void Run(uint64_t desc_a, uint64_t desc_b, CReg *c,
                            cute::SM90::GMMA::ScaleOut scale,
                            std::index_sequence<Idx...>) {
    Impl::fma(desc_a, desc_b, c[Idx]..., scale);
  }

  TL_DEVICE static void exec(uint64_t desc_a, uint64_t desc_b, uint32_t *c_raw,
                             bool scale_out) {
    auto scale = scale_out ? cute::SM90::GMMA::ScaleOut::One
                           : cute::SM90::GMMA::ScaleOut::Zero;
    auto c = reinterpret_cast<CReg *>(c_raw);
    Run(desc_a, desc_b, c, scale, std::make_index_sequence<kCRegs>{});
  }
};

template <class Impl> struct CallWgmmaRS {
  using AReg = std::remove_extent_t<typename Impl::ARegisters>;
  using CReg = std::remove_extent_t<typename Impl::CRegisters>;
  static constexpr int kARegs = std::extent_v<typename Impl::ARegisters>;
  static constexpr int kCRegs = std::extent_v<typename Impl::CRegisters>;
  static_assert(sizeof(AReg) == sizeof(uint32_t),
                "tl::wgmma_rs expects 32-bit register operands for A.");
  static_assert(sizeof(CReg) == sizeof(uint32_t) ||
                    sizeof(CReg) == sizeof(float),
                "tl::wgmma_rs expects 32-bit accumulator registers.");

  template <size_t... AIdx, size_t... CIdx>
  TL_DEVICE static void
  Run(const AReg *a, uint64_t desc_b, CReg *c, cute::SM90::GMMA::ScaleOut scale,
      std::index_sequence<AIdx...>, std::index_sequence<CIdx...>) {
    Impl::fma(a[AIdx]..., desc_b, c[CIdx]..., scale);
  }

  TL_DEVICE static void exec(const uint32_t *a_raw, uint64_t desc_b,
                             uint32_t *c_raw, bool scale_out) {
    auto scale = scale_out ? cute::SM90::GMMA::ScaleOut::One
                           : cute::SM90::GMMA::ScaleOut::Zero;
    auto a = reinterpret_cast<const AReg *>(a_raw);
    auto c = reinterpret_cast<CReg *>(c_raw);
    Run(a, desc_b, c, scale, std::make_index_sequence<kARegs>{},
        std::make_index_sequence<kCRegs>{});
  }
};

} // namespace detail

template <DataType A_type, DataType B_type, DataType C_type, int M, int N,
          int K, bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaSSImpl {
  static_assert(detail::IsValidScale<scaleA>, "tl::wgmma_ss: invalid scaleA");
  static_assert(detail::IsValidScale<scaleB>, "tl::wgmma_ss: invalid scaleB");
  TL_DEVICE static void execute(uint64_t, uint64_t, uint32_t *, bool) {
    static_assert(always_false_v<std::integral_constant<int, M>>,
                  "tl::wgmma_ss: unsupported configuration");
  }
};

template <DataType A_type, DataType B_type, DataType C_type, int M, int N,
          int K, bool tnspA, bool tnspB, int scaleA, int scaleB>
struct WgmmaRSImpl {
  static_assert(detail::IsValidScale<scaleA>, "tl::wgmma_rs: invalid scaleA");
  static_assert(detail::IsValidScale<scaleB>, "tl::wgmma_rs: invalid scaleB");
  TL_DEVICE static void execute(const uint32_t *, uint64_t, uint32_t *, bool) {
    static_assert(always_false_v<std::integral_constant<int, M>>,
                  "tl::wgmma_rs: unsupported configuration");
  }
};

#define TL_WGMMA_DEFINE_SS_GENERAL(AType, BType, CType, M, N, K, ImplName)     \
  template <bool tnspA, bool tnspB, int scaleA, int scaleB>                    \
  struct WgmmaSSImpl<DataType::AType, DataType::BType, DataType::CType, M, N,  \
                     K, tnspA, tnspB, scaleA, scaleB> {                        \
    static_assert(detail::IsValidScale<scaleA>,                                \
                  "tl::wgmma_ss: invalid scaleA");                             \
    static_assert(detail::IsValidScale<scaleB>,                                \
                  "tl::wgmma_ss: invalid scaleB");                             \
    using Impl =                                                               \
        cute::SM90::GMMA::ImplName<detail::MajorValue<tnspA>::value,           \
                                   detail::MajorValue<tnspB>::value,           \
                                   detail::ScaleInValue<scaleA>::value,        \
                                   detail::ScaleInValue<scaleB>::value>;       \
    TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b,            \
                                  uint32_t *c, bool scale_out) {               \
      detail::CallWgmmaSS<Impl>::exec(desc_a, desc_b, c, scale_out);           \
    }                                                                          \
  };

#define TL_WGMMA_DEFINE_SS_TN(AType, BType, CType, M, N, K, ImplName)          \
  template <int scaleA, int scaleB>                                            \
  struct WgmmaSSImpl<DataType::AType, DataType::BType, DataType::CType, M, N,  \
                     K, false, false, scaleA, scaleB> {                        \
    static_assert(detail::IsValidScale<scaleA>,                                \
                  "tl::wgmma_ss: invalid scaleA");                             \
    static_assert(detail::IsValidScale<scaleB>,                                \
                  "tl::wgmma_ss: invalid scaleB");                             \
    using Impl =                                                               \
        cute::SM90::GMMA::ImplName<detail::ScaleInValue<scaleA>::value,        \
                                   detail::ScaleInValue<scaleB>::value>;       \
    TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b,            \
                                  uint32_t *c, bool scale_out) {               \
      detail::CallWgmmaSS<Impl>::exec(desc_a, desc_b, c, scale_out);           \
    }                                                                          \
  };

#define TL_WGMMA_DEFINE_SS_TN_FIXED_SCALE(AType, BType, CType, M, N, K,        \
                                          ImplName)                            \
  template <int scaleA, int scaleB>                                            \
  struct WgmmaSSImpl<DataType::AType, DataType::BType, DataType::CType, M, N,  \
                     K, false, false, scaleA, scaleB> {                        \
    static_assert(detail::IsValidScale<scaleA>,                                \
                  "tl::wgmma_ss: invalid scaleA");                             \
    static_assert(detail::IsValidScale<scaleB>,                                \
                  "tl::wgmma_ss: invalid scaleB");                             \
    static_assert(scaleA == 1 && scaleB == 1,                                  \
                  "tl::wgmma_ss: only +1 scaling supported for this WGMMA");   \
    using Impl = cute::SM90::GMMA::ImplName;                                   \
    TL_DEVICE static void execute(uint64_t desc_a, uint64_t desc_b,            \
                                  uint32_t *c, bool scale_out) {               \
      detail::CallWgmmaSS<Impl>::exec(desc_a, desc_b, c, scale_out);           \
    }                                                                          \
  };

#define TL_WGMMA_DEFINE_RS_GENERAL(AType, BType, CType, M, N, K, ImplName)     \
  template <bool tnspA, bool tnspB, int scaleA, int scaleB>                    \
  struct WgmmaRSImpl<DataType::AType, DataType::BType, DataType::CType, M, N,  \
                     K, tnspA, tnspB, scaleA, scaleB> {                        \
    static_assert(!tnspA, "tl::wgmma_rs: operand A must be K-major");          \
    static_assert(detail::IsValidScale<scaleA>,                                \
                  "tl::wgmma_rs: invalid scaleA");                             \
    static_assert(detail::IsValidScale<scaleB>,                                \
                  "tl::wgmma_rs: invalid scaleB");                             \
    using Impl =                                                               \
        cute::SM90::GMMA::ImplName<detail::MajorValue<tnspA>::value,           \
                                   detail::MajorValue<tnspB>::value,           \
                                   detail::ScaleInValue<scaleA>::value,        \
                                   detail::ScaleInValue<scaleB>::value>;       \
    TL_DEVICE static void execute(const uint32_t *a, uint64_t desc_b,          \
                                  uint32_t *c, bool scale_out) {               \
      detail::CallWgmmaRS<Impl>::exec(a, desc_b, c, scale_out);                \
    }                                                                          \
  };

#define TL_WGMMA_DEFINE_RS_TN(AType, BType, CType, M, N, K, ImplName)          \
  template <int scaleA, int scaleB>                                            \
  struct WgmmaRSImpl<DataType::AType, DataType::BType, DataType::CType, M, N,  \
                     K, false, false, scaleA, scaleB> {                        \
    static_assert(detail::IsValidScale<scaleA>,                                \
                  "tl::wgmma_rs: invalid scaleA");                             \
    static_assert(detail::IsValidScale<scaleB>,                                \
                  "tl::wgmma_rs: invalid scaleB");                             \
    using Impl =                                                               \
        cute::SM90::GMMA::ImplName<detail::ScaleInValue<scaleA>::value,        \
                                   detail::ScaleInValue<scaleB>::value>;       \
    TL_DEVICE static void execute(const uint32_t *a, uint64_t desc_b,          \
                                  uint32_t *c, bool scale_out) {               \
      detail::CallWgmmaRS<Impl>::exec(a, desc_b, c, scale_out);                \
    }                                                                          \
  };

#define TL_WGMMA_DEFINE_RS_TN_FIXED_SCALE(AType, BType, CType, M, N, K,        \
                                          ImplName)                            \
  template <int scaleA, int scaleB>                                            \
  struct WgmmaRSImpl<DataType::AType, DataType::BType, DataType::CType, M, N,  \
                     K, false, false, scaleA, scaleB> {                        \
    static_assert(detail::IsValidScale<scaleA>,                                \
                  "tl::wgmma_rs: invalid scaleA");                             \
    static_assert(detail::IsValidScale<scaleB>,                                \
                  "tl::wgmma_rs: invalid scaleB");                             \
    static_assert(scaleA == 1 && scaleB == 1,                                  \
                  "tl::wgmma_rs: only +1 scaling supported for this WGMMA");   \
    using Impl = cute::SM90::GMMA::ImplName;                                   \
    TL_DEVICE static void execute(const uint32_t *a, uint64_t desc_b,          \
                                  uint32_t *c, bool scale_out) {               \
      detail::CallWgmmaRS<Impl>::exec(a, desc_b, c, scale_out);                \
    }                                                                          \
  };

#define TL_WGMMA_FOREACH_N_FLOAT_MUL8(OP)                                      \
  OP(8)                                                                        \
  OP(16)                                                                       \
  OP(24)                                                                       \
  OP(32)                                                                       \
  OP(40)                                                                       \
  OP(48)                                                                       \
  OP(56)                                                                       \
  OP(64)                                                                       \
  OP(72)                                                                       \
  OP(80)                                                                       \
  OP(88)                                                                       \
  OP(96)                                                                       \
  OP(104)                                                                      \
  OP(112)                                                                      \
  OP(120)                                                                      \
  OP(128)                                                                      \
  OP(136)                                                                      \
  OP(144)                                                                      \
  OP(152)                                                                      \
  OP(160)                                                                      \
  OP(168)                                                                      \
  OP(176)                                                                      \
  OP(184)                                                                      \
  OP(192)                                                                      \
  OP(200)                                                                      \
  OP(208)                                                                      \
  OP(216)                                                                      \
  OP(224)                                                                      \
  OP(232)                                                                      \
  OP(240)                                                                      \
  OP(248)                                                                      \
  OP(256)

#define TL_WGMMA_FOREACH_N_INT32_MUL8(OP)                                      \
  OP(8)                                                                        \
  OP(16)                                                                       \
  OP(24)                                                                       \
  OP(32)                                                                       \
  OP(48)                                                                       \
  OP(64)                                                                       \
  OP(80)                                                                       \
  OP(96)                                                                       \
  OP(112)                                                                      \
  OP(128)                                                                      \
  OP(144)                                                                      \
  OP(160)                                                                      \
  OP(176)                                                                      \
  OP(192)                                                                      \
  OP(208)                                                                      \
  OP(224)                                                                      \
  OP(240)                                                                      \
  OP(256)

#define TL_WGMMA_DEFINE_F16_F16_F16_SS(N)                                      \
  TL_WGMMA_DEFINE_SS_GENERAL(kFloat16, kFloat16, kFloat16, 64, N, 16,          \
                             MMA_64x##N##x16_F16F16F16_SS)
#define TL_WGMMA_DEFINE_F16_F16_F32_SS(N)                                      \
  TL_WGMMA_DEFINE_SS_GENERAL(kFloat16, kFloat16, kFloat32, 64, N, 16,          \
                             MMA_64x##N##x16_F32F16F16_SS)
#define TL_WGMMA_DEFINE_BF16_BF16_F32_SS(N)                                    \
  TL_WGMMA_DEFINE_SS_GENERAL(kBFloat16, kBFloat16, kFloat32, 64, N, 16,        \
                             MMA_64x##N##x16_F32BF16BF16_SS)

#define TL_WGMMA_DEFINE_F32_TF32_SS_TN(N)                                      \
  TL_WGMMA_DEFINE_SS_TN(kTensorFloat32, kTensorFloat32, kFloat32, 64, N, 8,    \
                        MMA_64x##N##x8_F32TF32TF32_SS_TN)

#define TL_WGMMA_DEFINE_S32_S8S8_SS_TN(N)                                      \
  TL_WGMMA_DEFINE_SS_TN_FIXED_SCALE(kInt8, kInt8, kInt32, 64, N, 32,           \
                                    MMA_64x##N##x32_S32S8S8_SS_TN)
#define TL_WGMMA_DEFINE_S32_S8U8_SS_TN(N)                                      \
  TL_WGMMA_DEFINE_SS_TN_FIXED_SCALE(kInt8, kUInt8, kInt32, 64, N, 32,          \
                                    MMA_64x##N##x32_S32S8U8_SS_TN)
#define TL_WGMMA_DEFINE_S32_U8S8_SS_TN(N)                                      \
  TL_WGMMA_DEFINE_SS_TN_FIXED_SCALE(kUInt8, kInt8, kInt32, 64, N, 32,          \
                                    MMA_64x##N##x32_S32U8S8_SS_TN)
#define TL_WGMMA_DEFINE_S32_U8U8_SS_TN(N)                                      \
  TL_WGMMA_DEFINE_SS_TN_FIXED_SCALE(kUInt8, kUInt8, kInt32, 64, N, 32,         \
                                    MMA_64x##N##x32_S32U8U8_SS_TN)

#define TL_WGMMA_DEFINE_F16_E4M3E4M3_SS_TN(N)                                  \
  TL_WGMMA_DEFINE_SS_TN(kFloat8_e4m3, kFloat8_e4m3, kFloat16, 64, N, 32,       \
                        MMA_64x##N##x32_F16E4M3E4M3_SS_TN)
#define TL_WGMMA_DEFINE_F32_E4M3E4M3_SS_TN(N)                                  \
  TL_WGMMA_DEFINE_SS_TN(kFloat8_e4m3, kFloat8_e4m3, kFloat32, 64, N, 32,       \
                        MMA_64x##N##x32_F32E4M3E4M3_SS_TN)
#define TL_WGMMA_DEFINE_F16_E4M3E5M2_SS_TN(N)                                  \
  TL_WGMMA_DEFINE_SS_TN(kFloat8_e4m3, kFloat8_e5m2, kFloat16, 64, N, 32,       \
                        MMA_64x##N##x32_F16E4M3E5M2_SS_TN)
#define TL_WGMMA_DEFINE_F32_E4M3E5M2_SS_TN(N)                                  \
  TL_WGMMA_DEFINE_SS_TN(kFloat8_e4m3, kFloat8_e5m2, kFloat32, 64, N, 32,       \
                        MMA_64x##N##x32_F32E4M3E5M2_SS_TN)
#define TL_WGMMA_DEFINE_F16_E5M2E4M3_SS_TN(N)                                  \
  TL_WGMMA_DEFINE_SS_TN(kFloat8_e5m2, kFloat8_e4m3, kFloat16, 64, N, 32,       \
                        MMA_64x##N##x32_F16E5M2E4M3_SS_TN)
#define TL_WGMMA_DEFINE_F32_E5M2E4M3_SS_TN(N)                                  \
  TL_WGMMA_DEFINE_SS_TN(kFloat8_e5m2, kFloat8_e4m3, kFloat32, 64, N, 32,       \
                        MMA_64x##N##x32_F32E5M2E4M3_SS_TN)
#define TL_WGMMA_DEFINE_F16_E5M2E5M2_SS_TN(N)                                  \
  TL_WGMMA_DEFINE_SS_TN(kFloat8_e5m2, kFloat8_e5m2, kFloat16, 64, N, 32,       \
                        MMA_64x##N##x32_F16E5M2E5M2_SS_TN)
#define TL_WGMMA_DEFINE_F32_E5M2E5M2_SS_TN(N)                                  \
  TL_WGMMA_DEFINE_SS_TN(kFloat8_e5m2, kFloat8_e5m2, kFloat32, 64, N, 32,       \
                        MMA_64x##N##x32_F32E5M2E5M2_SS_TN)

TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_F16_F16_SS);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_F16_F32_SS);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_BF16_BF16_F32_SS);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F32_TF32_SS_TN);

TL_WGMMA_FOREACH_N_INT32_MUL8(TL_WGMMA_DEFINE_S32_S8S8_SS_TN);
TL_WGMMA_FOREACH_N_INT32_MUL8(TL_WGMMA_DEFINE_S32_S8U8_SS_TN);
TL_WGMMA_FOREACH_N_INT32_MUL8(TL_WGMMA_DEFINE_S32_U8S8_SS_TN);
TL_WGMMA_FOREACH_N_INT32_MUL8(TL_WGMMA_DEFINE_S32_U8U8_SS_TN);

TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_E4M3E4M3_SS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F32_E4M3E4M3_SS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_E4M3E5M2_SS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F32_E4M3E5M2_SS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_E5M2E4M3_SS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F32_E5M2E4M3_SS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_E5M2E5M2_SS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F32_E5M2E5M2_SS_TN);

#define TL_WGMMA_DEFINE_F16_F16_F16_RS(N)                                      \
  TL_WGMMA_DEFINE_RS_GENERAL(kFloat16, kFloat16, kFloat16, 64, N, 16,          \
                             MMA_64x##N##x16_F16F16F16_RS)
#define TL_WGMMA_DEFINE_F16_F16_F32_RS(N)                                      \
  TL_WGMMA_DEFINE_RS_GENERAL(kFloat16, kFloat16, kFloat32, 64, N, 16,          \
                             MMA_64x##N##x16_F32F16F16_RS)
#define TL_WGMMA_DEFINE_BF16_BF16_F32_RS(N)                                    \
  TL_WGMMA_DEFINE_RS_GENERAL(kBFloat16, kBFloat16, kFloat32, 64, N, 16,        \
                             MMA_64x##N##x16_F32BF16BF16_RS)

#define TL_WGMMA_DEFINE_F32_TF32_RS_TN(N)                                      \
  TL_WGMMA_DEFINE_RS_TN(kTensorFloat32, kTensorFloat32, kFloat32, 64, N, 8,    \
                        MMA_64x##N##x8_F32TF32TF32_RS_TN)

#define TL_WGMMA_DEFINE_S32_S8S8_RS_TN(N)                                      \
  TL_WGMMA_DEFINE_RS_TN_FIXED_SCALE(kInt8, kInt8, kInt32, 64, N, 32,           \
                                    MMA_64x##N##x32_S32S8S8_RS_TN)
#define TL_WGMMA_DEFINE_S32_S8U8_RS_TN(N)                                      \
  TL_WGMMA_DEFINE_RS_TN_FIXED_SCALE(kInt8, kUInt8, kInt32, 64, N, 32,          \
                                    MMA_64x##N##x32_S32S8U8_RS_TN)
#define TL_WGMMA_DEFINE_S32_U8S8_RS_TN(N)                                      \
  TL_WGMMA_DEFINE_RS_TN_FIXED_SCALE(kUInt8, kInt8, kInt32, 64, N, 32,          \
                                    MMA_64x##N##x32_S32U8S8_RS_TN)
#define TL_WGMMA_DEFINE_S32_U8U8_RS_TN(N)                                      \
  TL_WGMMA_DEFINE_RS_TN_FIXED_SCALE(kUInt8, kUInt8, kInt32, 64, N, 32,         \
                                    MMA_64x##N##x32_S32U8U8_RS_TN)

#define TL_WGMMA_DEFINE_F16_E4M3E4M3_RS_TN(N)                                  \
  TL_WGMMA_DEFINE_RS_TN(kFloat8_e4m3, kFloat8_e4m3, kFloat16, 64, N, 32,       \
                        MMA_64x##N##x32_F16E4M3E4M3_RS_TN)
#define TL_WGMMA_DEFINE_F32_E4M3E4M3_RS_TN(N)                                  \
  TL_WGMMA_DEFINE_RS_TN(kFloat8_e4m3, kFloat8_e4m3, kFloat32, 64, N, 32,       \
                        MMA_64x##N##x32_F32E4M3E4M3_RS_TN)
#define TL_WGMMA_DEFINE_F16_E4M3E5M2_RS_TN(N)                                  \
  TL_WGMMA_DEFINE_RS_TN(kFloat8_e4m3, kFloat8_e5m2, kFloat16, 64, N, 32,       \
                        MMA_64x##N##x32_F16E4M3E5M2_RS_TN)
#define TL_WGMMA_DEFINE_F32_E4M3E5M2_RS_TN(N)                                  \
  TL_WGMMA_DEFINE_RS_TN(kFloat8_e4m3, kFloat8_e5m2, kFloat32, 64, N, 32,       \
                        MMA_64x##N##x32_F32E4M3E5M2_RS_TN)
#define TL_WGMMA_DEFINE_F16_E5M2E4M3_RS_TN(N)                                  \
  TL_WGMMA_DEFINE_RS_TN(kFloat8_e5m2, kFloat8_e4m3, kFloat16, 64, N, 32,       \
                        MMA_64x##N##x32_F16E5M2E4M3_RS_TN)
#define TL_WGMMA_DEFINE_F32_E5M2E4M3_RS_TN(N)                                  \
  TL_WGMMA_DEFINE_RS_TN(kFloat8_e5m2, kFloat8_e4m3, kFloat32, 64, N, 32,       \
                        MMA_64x##N##x32_F32E5M2E4M3_RS_TN)
#define TL_WGMMA_DEFINE_F16_E5M2E5M2_RS_TN(N)                                  \
  TL_WGMMA_DEFINE_RS_TN(kFloat8_e5m2, kFloat8_e5m2, kFloat16, 64, N, 32,       \
                        MMA_64x##N##x32_F16E5M2E5M2_RS_TN)
#define TL_WGMMA_DEFINE_F32_E5M2E5M2_RS_TN(N)                                  \
  TL_WGMMA_DEFINE_RS_TN(kFloat8_e5m2, kFloat8_e5m2, kFloat32, 64, N, 32,       \
                        MMA_64x##N##x32_F32E5M2E5M2_RS_TN)

TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_F16_F16_RS);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_F16_F32_RS);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_BF16_BF16_F32_RS);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F32_TF32_RS_TN);

TL_WGMMA_FOREACH_N_INT32_MUL8(TL_WGMMA_DEFINE_S32_S8S8_RS_TN);
TL_WGMMA_FOREACH_N_INT32_MUL8(TL_WGMMA_DEFINE_S32_S8U8_RS_TN);
TL_WGMMA_FOREACH_N_INT32_MUL8(TL_WGMMA_DEFINE_S32_U8S8_RS_TN);
TL_WGMMA_FOREACH_N_INT32_MUL8(TL_WGMMA_DEFINE_S32_U8U8_RS_TN);

TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_E4M3E4M3_RS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F32_E4M3E4M3_RS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_E4M3E5M2_RS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F32_E4M3E5M2_RS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_E5M2E4M3_RS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F32_E5M2E4M3_RS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F16_E5M2E5M2_RS_TN);
TL_WGMMA_FOREACH_N_FLOAT_MUL8(TL_WGMMA_DEFINE_F32_E5M2E5M2_RS_TN);

#undef TL_WGMMA_DEFINE_F16_F16_F16_SS
#undef TL_WGMMA_DEFINE_F16_F16_F32_SS
#undef TL_WGMMA_DEFINE_BF16_BF16_F32_SS
#undef TL_WGMMA_DEFINE_F32_TF32_SS_TN
#undef TL_WGMMA_DEFINE_S32_S8S8_SS_TN
#undef TL_WGMMA_DEFINE_S32_S8U8_SS_TN
#undef TL_WGMMA_DEFINE_S32_U8S8_SS_TN
#undef TL_WGMMA_DEFINE_S32_U8U8_SS_TN
#undef TL_WGMMA_DEFINE_F16_E4M3E4M3_SS_TN
#undef TL_WGMMA_DEFINE_F32_E4M3E4M3_SS_TN
#undef TL_WGMMA_DEFINE_F16_E4M3E5M2_SS_TN
#undef TL_WGMMA_DEFINE_F32_E4M3E5M2_SS_TN
#undef TL_WGMMA_DEFINE_F16_E5M2E4M3_SS_TN
#undef TL_WGMMA_DEFINE_F32_E5M2E4M3_SS_TN
#undef TL_WGMMA_DEFINE_F16_E5M2E5M2_SS_TN
#undef TL_WGMMA_DEFINE_F32_E5M2E5M2_SS_TN
#undef TL_WGMMA_DEFINE_F16_F16_F16_RS
#undef TL_WGMMA_DEFINE_F16_F16_F32_RS
#undef TL_WGMMA_DEFINE_BF16_BF16_F32_RS
#undef TL_WGMMA_DEFINE_F32_TF32_RS_TN
#undef TL_WGMMA_DEFINE_S32_S8S8_RS_TN
#undef TL_WGMMA_DEFINE_S32_S8U8_RS_TN
#undef TL_WGMMA_DEFINE_S32_U8S8_RS_TN
#undef TL_WGMMA_DEFINE_S32_U8U8_RS_TN
#undef TL_WGMMA_DEFINE_F16_E4M3E4M3_RS_TN
#undef TL_WGMMA_DEFINE_F32_E4M3E4M3_RS_TN
#undef TL_WGMMA_DEFINE_F16_E4M3E5M2_RS_TN
#undef TL_WGMMA_DEFINE_F32_E4M3E5M2_RS_TN
#undef TL_WGMMA_DEFINE_F16_E5M2E4M3_RS_TN
#undef TL_WGMMA_DEFINE_F32_E5M2E4M3_RS_TN
#undef TL_WGMMA_DEFINE_F16_E5M2E5M2_RS_TN
#undef TL_WGMMA_DEFINE_F32_E5M2E5M2_RS_TN
#undef TL_WGMMA_FOREACH_N_FLOAT_MUL8
#undef TL_WGMMA_FOREACH_N_INT32_MUL8
#undef TL_WGMMA_DEFINE_SS_TN_FIXED_SCALE
#undef TL_WGMMA_DEFINE_SS_GENERAL
#undef TL_WGMMA_DEFINE_SS_TN
#undef TL_WGMMA_DEFINE_RS_TN_FIXED_SCALE
#undef TL_WGMMA_DEFINE_RS_GENERAL
#undef TL_WGMMA_DEFINE_RS_TN

template <DataType A_type, DataType B_type, DataType C_type, int M, int N,
          int K, bool tnspA, bool tnspB, int scaleA = 1, int scaleB = 1>
TL_DEVICE void wgmma_ss(uint64_t desc_a, uint64_t desc_b, uint32_t *c,
                        bool scale_out) {
  WgmmaSSImpl<A_type, B_type, C_type, M, N, K, tnspA, tnspB, scaleA,
              scaleB>::execute(desc_a, desc_b, c, scale_out);
}

template <DataType A_type, DataType B_type, DataType C_type, int M, int N,
          int K, bool tnspA, bool tnspB, int scaleA = 1, int scaleB = 1>
TL_DEVICE void wgmma_rs(const uint32_t *a, uint64_t desc_b, uint32_t *c,
                        bool scale_out) {
  WgmmaRSImpl<A_type, B_type, C_type, M, N, K, tnspA, tnspB, scaleA,
              scaleB>::execute(a, desc_b, c, scale_out);
}

} // namespace tl
