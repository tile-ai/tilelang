// Licensed under the MIT License.
#pragma once

#include "common.h"
#include "gemm_mma.h"
#include "intrin.h"

#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>

namespace cute {

// Extensions to CuTe
// CuTe don't support TCGEN5MMA with .ws, so we add it here
// About why we need .ws, plz refer to comments in tl_tcgen5mma::GemmTensorOp

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_SS {
  static_assert(M == 32 || M == 64 || M == 128,
                "SM100_MMA_F16BF16 (with .ws) M-mode size should be 32, 64 or "
                "128 for 1 CTA cluster MMA.");
  static_assert(
      N == 64 || N == 128 || N == 256,
      "SM100_MMA_F16BF16 (with .ws) N-mode size should be 32, 64 or 128");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
      uint32_t const &scaleC, uint64_t const &idescE) {
    if (cute::elect_one_sync()) {
      asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)),
            "r"(scaleC));
    }
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_WS_SS<a_type, b_type, c_type, M, N, a_major,
                                          b_major, a_neg, b_neg>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                    cute::sizeof_bits_v<b_type> == 16,
                "SM100_MMA_F16BF16_WS_SS supports 16bit types");

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;

  // Logical shape-K is always 256bits, transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout =
      Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout =
      Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout =
      Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                            a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void
  mma_unpack(MMA_Traits const &traits, Tensor<TD, DLayout> &D,
             Tensor<TA, ALayout> const &A, Tensor<TB, BLayout> const &B,
             Tensor<TC, CLayout> const &C) {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F16BF16_WS_SS<a_type, b_type, c_type, M, N, a_major, b_major,
                            a_neg, b_neg>::fma(desc_a, desc_b, tmem_c,
                                               uint32_t(traits.accumulate_),
                                               idesc);
  }
};

struct SM100_MMA_F8F6F4_WS_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
      uint32_t const &scaleC, uint64_t const &idescE) {
    if (cute::elect_one_sync()) {
      asm volatile("{\n\t"
                   ".reg .pred p;\n\t"
                   "setp.ne.b32 p, %4, 0;\n\t"
                   "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, "
                   "p, 0; \n\t"
                   "}\n"
                   :
                   : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
                     "r"(uint32_t(idescE >> 32)), "r"(scaleC));
    }
  }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F8F6F4_WS_SS, a_type, b_type, c_type, cute::C<M>,
                  cute::C<N>, cute::integral_constant<UMMA::Major, a_major>,
                  cute::integral_constant<UMMA::Major, b_major>,
                  cute::integral_constant<UMMA::ScaleIn, a_neg>,
                  cute::integral_constant<UMMA::ScaleIn, b_neg>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 &&
                    cute::sizeof_bits_v<b_type> <= 8,
                "SM100_MMA_F8F6F4_WS_SS supports types with leq 8bit types");
  static_assert(M == 32 || M == 64 || M == 128,
                "SM100_MMA_F8F6F4_WS_SS M-mode size should be 32, 64 or 128 "
                "for 1 CTA cluster MMA.");
  static_assert(
      N == 64 || N == 128 || N == 256,
      "SM100_MMA_F8F6F4_WS_SS (with .ws) N-mode size should be 32, 64 or 128");
  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;

  static_assert(sizeof_bits_v<ValTypeA> <= sizeof_bits_v<uint8_t> &&
                sizeof_bits_v<ValTypeB> <= sizeof_bits_v<uint8_t>);

  // Logical shape-K is always 256bits, transform to units of elements
  constexpr static int K = 32;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout =
      Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout =
      Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout =
      Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                            a_neg, b_neg>();

  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void
  mma_unpack(MMA_Traits const &traits, Tensor<TD, DLayout> &D,
             Tensor<TA, ALayout> const &A, Tensor<TB, BLayout> const &B,
             Tensor<TC, CLayout> const &C) {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TA>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value,
                  "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

    SM100_MMA_F8F6F4_WS_SS::fma(desc_a, desc_b, tmem_c,
                                uint32_t(traits.accumulate_), idesc);
  }
};

// MMA_Traits specialization for FP4 (float_e2m1_t) with SM100_MMA_F8F6F4_SS.
// CUTLASS's to_UMMAFormat<float_e2m1_t> returns MXF4Format::E2M1(=1), but
// kind::f8f6f4 requires MXF8F6F4Format::E2M1(=5). This more-specialized
// partial specialization provides the correct descriptor encoding.
template <class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F8F6F4_SS,
                  cute::float_e2m1_t, cute::float_e2m1_t, c_type,
                  cute::C<M>, cute::C<N>,
                  cute::integral_constant<UMMA::Major, a_major>,
                  cute::integral_constant<UMMA::Major, b_major>,
                  cute::integral_constant<UMMA::ScaleIn, a_neg>,
                  cute::integral_constant<UMMA::ScaleIn, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = cute::float_e2m1_t;
  using ValTypeB = cute::float_e2m1_t;
  using ValTypeC = c_type;

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  constexpr static int K = 32;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  static constexpr UMMA::InstrDescriptor make_fp4_desc() {
    UMMA::InstrDescriptor d = {};
    d.c_format_ = uint8_t(UMMA::to_CFormat<c_type>());
    d.a_format_ = 5;   // MXF8F6F4Format::E2M1
    d.b_format_ = 5;
    d.a_negate_ = uint8_t(a_neg);
    d.b_negate_ = uint8_t(b_neg);
    d.a_major_  = uint8_t(a_major);
    d.b_major_  = uint8_t(b_major);
    d.n_dim_    = N >> 3;
    d.m_dim_    = M >> 4;
    return d;
  }

  UMMA::InstrDescriptor idesc_ = make_fp4_desc();
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void
  mma_unpack(MMA_Traits const &traits, Tensor<TD, DLayout> &D,
             Tensor<TA, ALayout> const &A, Tensor<TB, BLayout> const &B,
             Tensor<TC, CLayout> const &C) {
    static_assert(is_tmem<TD>::value, "Expected tmem");
    static_assert(is_rmem<TA>::value, "Expected desc registers");
    static_assert(is_rmem<TB>::value, "Expected desc registers");
    static_assert(is_tmem<TC>::value, "Expected tmem");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
    SM100_MMA_F8F6F4_SS::fma(desc_a, desc_b, tmem_c,
                             uint32_t(traits.accumulate_), idesc);
  }
};

// Same fix for SM100_MMA_F8F6F4_WS_SS (weight-stationary .ws variant).
template <class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F8F6F4_WS_SS,
                  cute::float_e2m1_t, cute::float_e2m1_t, c_type,
                  cute::C<M>, cute::C<N>,
                  cute::integral_constant<UMMA::Major, a_major>,
                  cute::integral_constant<UMMA::Major, b_major>,
                  cute::integral_constant<UMMA::ScaleIn, a_neg>,
                  cute::integral_constant<UMMA::ScaleIn, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = cute::float_e2m1_t;
  using ValTypeB = cute::float_e2m1_t;
  using ValTypeC = c_type;

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;

  constexpr static int K = 32;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  static constexpr UMMA::InstrDescriptor make_fp4_desc() {
    UMMA::InstrDescriptor d = {};
    d.c_format_ = uint8_t(UMMA::to_CFormat<c_type>());
    d.a_format_ = 5;   // MXF8F6F4Format::E2M1
    d.b_format_ = 5;
    d.a_negate_ = uint8_t(a_neg);
    d.b_negate_ = uint8_t(b_neg);
    d.a_major_  = uint8_t(a_major);
    d.b_major_  = uint8_t(b_major);
    d.n_dim_    = N >> 3;
    d.m_dim_    = M >> 4;
    return d;
  }

  UMMA::InstrDescriptor idesc_ = make_fp4_desc();
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void
  mma_unpack(MMA_Traits const &traits, Tensor<TD, DLayout> &D,
             Tensor<TA, ALayout> const &A, Tensor<TB, BLayout> const &B,
             Tensor<TC, CLayout> const &C) {
    static_assert(is_tmem<TD>::value, "Expected tmem");
    static_assert(is_rmem<TA>::value, "Expected desc registers");
    static_assert(is_rmem<TB>::value, "Expected desc registers");
    static_assert(is_tmem<TC>::value, "Expected tmem");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
    SM100_MMA_F8F6F4_WS_SS::fma(desc_a, desc_b, tmem_c,
                                uint32_t(traits.accumulate_), idesc);
  }
};

// MMA_Traits specialization for FP4 (float_e2m1_t) with SM100_MMA_F8F6F4_TS.
// A operand comes from TMEM, B from SMEM. Same descriptor encoding fix as SS.
// 你的理解接近正确，但还需要澄清一点：
// 使用时其实不用再对 template 里面的参数“再次特化”，而是——
// 1. 外层 template <...> 这部分负责声明参数列表，即让这个偏特化对所有可能的 <c_type, M, N, ...> 组合都有效。
// 2. 内层 struct MMA_Traits<SM100_MMA_F8F6F4_TS<...>> 说明：只要模板参数匹配到 SM100_MMA_F8F6F4_TS 那一整组，
//    就会自动选择这个偏特化的实现，不需要“再特化”。
// 换句话说，你写一个 MMA_Traits<X>，如果 X 恰好能匹配 SM100_MMA_F8F6F4_TS<...> 那一组参数，
// 这个特化版本就会被用到。比如：
//     using Traits = MMA_Traits<SM100_MMA_F8F6F4_TS<float_e2m1_t, float_e2m1_t, float, 64, 32, ...>>;
// Traits的内容就是匹配到此特化，无需再继续特化template参数。
// 总结：外层 template 支持泛型匹配，struct MMA_Traits<...> 实现具体特化，实际用时只需传入对应参数即可自动选中。

template <class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg,
          UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_F8F6F4_TS<cute::float_e2m1_t, cute::float_e2m1_t, c_type,
                                       M, N, a_major, b_major,
                                       a_neg, b_neg, c_sat>>
{
  using ValTypeD = c_type;
  using ValTypeA = cute::float_e2m1_t;
  using ValTypeB = cute::float_e2m1_t;
  using ValTypeC = c_type;

  using FrgTypeA = UMMA::tmem_frg_1sm<cute::float_e2m1_t, cute::float_e2m1_t,
                                       UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t,
                                       UMMA::TmemAllocMode::NonInterleaved>;

  constexpr static int K = 32;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  static constexpr UMMA::InstrDescriptor make_fp4_desc() {
    UMMA::InstrDescriptor d = {};
    d.c_format_ = uint8_t(UMMA::to_CFormat<c_type>());
    d.a_format_ = 5;   // MXF8F6F4Format::E2M1
    d.b_format_ = 5;
    d.a_negate_ = uint8_t(a_neg);
    d.b_negate_ = uint8_t(b_neg);
    d.a_major_  = uint8_t(a_major);
    d.b_major_  = uint8_t(b_major);
    d.n_dim_    = N >> 3;
    d.m_dim_    = M >> 4;
    return d;
  }

  UMMA::InstrDescriptor idesc_ = make_fp4_desc();
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void
  mma_unpack(MMA_Traits const &traits, Tensor<TD, DLayout> &D,
             Tensor<TA, ALayout> const &A, Tensor<TB, BLayout> const &B,
             Tensor<TC, CLayout> const &C) {
    static_assert(is_tmem<TD>::value, "Expected tmem");
    static_assert(is_tmem<TA>::value, "Expected tmem");
    static_assert(is_rmem<TB>::value, "Expected desc registers");
    static_assert(is_tmem<TC>::value, "Expected tmem");

    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
    SM100_MMA_F8F6F4_TS<cute::float_e2m1_t, cute::float_e2m1_t, c_type,
                         M, N, a_major, b_major,
                         a_neg, b_neg, c_sat>::fma(
        tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

// Mixed-precision MMA_Traits: A = float_e4m3_t (FP8), B = float_e2m1_t (FP4).
// Descriptor: a_format = 0 (MXF8F6F4Format::E4M3), b_format = 5 (MXF8F6F4Format::E2M1).
// SS variant (both operands from shared memory).
template <class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F8F6F4_SS,
                  cute::float_e4m3_t, cute::float_e2m1_t, c_type,
                  cute::C<M>, cute::C<N>,
                  cute::integral_constant<UMMA::Major, a_major>,
                  cute::integral_constant<UMMA::Major, b_major>,
                  cute::integral_constant<UMMA::ScaleIn, a_neg>,
                  cute::integral_constant<UMMA::ScaleIn, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = cute::float_e4m3_t;
  using ValTypeB = cute::float_e2m1_t;
  using ValTypeC = c_type;

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

  constexpr static int K = 32;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  static constexpr UMMA::InstrDescriptor make_a8w4_desc() {
    UMMA::InstrDescriptor d = {};
    d.c_format_ = uint8_t(UMMA::to_CFormat<c_type>());
    d.a_format_ = 0;   // MXF8F6F4Format::E4M3
    d.b_format_ = 5;   // MXF8F6F4Format::E2M1
    d.a_negate_ = uint8_t(a_neg);
    d.b_negate_ = uint8_t(b_neg);
    d.a_major_  = uint8_t(a_major);
    d.b_major_  = uint8_t(b_major);
    d.n_dim_    = N >> 3;
    d.m_dim_    = M >> 4;
    return d;
  }

  UMMA::InstrDescriptor idesc_ = make_a8w4_desc();
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void
  mma_unpack(MMA_Traits const &traits, Tensor<TD, DLayout> &D,
             Tensor<TA, ALayout> const &A, Tensor<TB, BLayout> const &B,
             Tensor<TC, CLayout> const &C) {
    static_assert(is_tmem<TD>::value, "Expected tmem");
    static_assert(is_rmem<TA>::value, "Expected desc registers");
    static_assert(is_rmem<TB>::value, "Expected desc registers");
    static_assert(is_tmem<TC>::value, "Expected tmem");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
    SM100_MMA_F8F6F4_SS::fma(desc_a, desc_b, tmem_c,
                             uint32_t(traits.accumulate_), idesc);
  }
};

// Mixed A8W4 — WS_SS variant (weight-stationary, M=64 or M=32).
template <class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F8F6F4_WS_SS,
                  cute::float_e4m3_t, cute::float_e2m1_t, c_type,
                  cute::C<M>, cute::C<N>,
                  cute::integral_constant<UMMA::Major, a_major>,
                  cute::integral_constant<UMMA::Major, b_major>,
                  cute::integral_constant<UMMA::ScaleIn, a_neg>,
                  cute::integral_constant<UMMA::ScaleIn, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = cute::float_e4m3_t;
  using ValTypeB = cute::float_e2m1_t;
  using ValTypeC = c_type;

  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;

  constexpr static int K = 32;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  static constexpr UMMA::InstrDescriptor make_a8w4_desc() {
    UMMA::InstrDescriptor d = {};
    d.c_format_ = uint8_t(UMMA::to_CFormat<c_type>());
    d.a_format_ = 0;   // MXF8F6F4Format::E4M3
    d.b_format_ = 5;   // MXF8F6F4Format::E2M1
    d.a_negate_ = uint8_t(a_neg);
    d.b_negate_ = uint8_t(b_neg);
    d.a_major_  = uint8_t(a_major);
    d.b_major_  = uint8_t(b_major);
    d.n_dim_    = N >> 3;
    d.m_dim_    = M >> 4;
    return d;
  }

  UMMA::InstrDescriptor idesc_ = make_a8w4_desc();
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void
  mma_unpack(MMA_Traits const &traits, Tensor<TD, DLayout> &D,
             Tensor<TA, ALayout> const &A, Tensor<TB, BLayout> const &B,
             Tensor<TC, CLayout> const &C) {
    static_assert(is_tmem<TD>::value, "Expected tmem");
    static_assert(is_rmem<TA>::value, "Expected desc registers");
    static_assert(is_rmem<TB>::value, "Expected desc registers");
    static_assert(is_tmem<TC>::value, "Expected tmem");

    uint64_t desc_a = A[0];
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
    SM100_MMA_F8F6F4_WS_SS::fma(desc_a, desc_b, tmem_c,
                                uint32_t(traits.accumulate_), idesc);
  }
};

namespace tl_tcgen5mma {

using cutlass::gemm::collective::detail::sm100_smem_selector;

template <typename A_type, typename B_type, typename C_type, int M, int N,
          int K, UMMA::Major a_major, UMMA::Major b_major,
          typename Enable = void>
struct DispatchInstruction;

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<bfloat16_t, bfloat16_t, float, M, N, K, a_major,
                           b_major, std::enable_if_t<M == 128 && K == 16>> {
  using MMA = SM100_MMA_F16BF16_SS<bfloat16_t, bfloat16_t, float, M, N, a_major,
                                   b_major>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<bfloat16_t, bfloat16_t, float, M, N, K, a_major,
                           b_major,
                           std::enable_if_t<(M == 64 || M == 32) && K == 16>> {
  using MMA = SM100_MMA_F16BF16_WS_SS<bfloat16_t, bfloat16_t, float, M, N,
                                      a_major, b_major>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<half_t, half_t, float, M, N, K, a_major, b_major,
                           std::enable_if_t<M == 128 && K == 16>> {
  using MMA =
      SM100_MMA_F16BF16_SS<half_t, half_t, float, M, N, a_major, b_major>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<half_t, half_t, float, M, N, K, a_major, b_major,
                           std::enable_if_t<(M == 64 || M == 32) && K == 16>> {
  using MMA =
      SM100_MMA_F16BF16_WS_SS<half_t, half_t, float, M, N, a_major, b_major>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e4m3_t, cute::float_e4m3_t, float, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<M == 128 && K == 32>> {
  using MMA =
      MMA_Traits<SM100_MMA_F8F6F4_SS, cute::float_e4m3_t, cute::float_e4m3_t,
                 float, Int<M>, Int<N>, integral_constant<UMMA::Major, a_major>,
                 integral_constant<UMMA::Major, b_major>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e4m3_t, cute::float_e4m3_t, float, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<(M == 64 || M == 32) && K == 32>> {
  using MMA =
      MMA_Traits<SM100_MMA_F8F6F4_WS_SS, cute::float_e4m3_t, cute::float_e4m3_t,
                 float, Int<M>, Int<N>, integral_constant<UMMA::Major, a_major>,
                 integral_constant<UMMA::Major, b_major>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e4m3_t, cute::float_e4m3_t, half_t, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<M == 128 && K == 32>> {
  using MMA = MMA_Traits<SM100_MMA_F8F6F4_SS, cute::float_e4m3_t,
                         cute::float_e4m3_t, half_t, Int<M>, Int<N>,
                         integral_constant<UMMA::Major, a_major>,
                         integral_constant<UMMA::Major, b_major>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};
template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e4m3_t, cute::float_e4m3_t, half_t, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<(M == 64 || M == 32) && K == 32>> {
  using MMA = MMA_Traits<SM100_MMA_F8F6F4_WS_SS, cute::float_e4m3_t,
                         cute::float_e4m3_t, half_t, Int<M>, Int<N>,
                         integral_constant<UMMA::Major, a_major>,
                         integral_constant<UMMA::Major, b_major>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e5m2_t, cute::float_e5m2_t, float, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<M == 128 && K == 32>> {
  using MMA =
      MMA_Traits<SM100_MMA_F8F6F4_SS, cute::float_e5m2_t, cute::float_e5m2_t,
                 float, Int<M>, Int<N>, integral_constant<UMMA::Major, a_major>,
                 integral_constant<UMMA::Major, b_major>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e5m2_t, cute::float_e5m2_t, float, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<(M == 64 || M == 32) && K == 32>> {
  using MMA =
      MMA_Traits<SM100_MMA_F8F6F4_WS_SS, cute::float_e5m2_t, cute::float_e5m2_t,
                 float, Int<M>, Int<N>, integral_constant<UMMA::Major, a_major>,
                 integral_constant<UMMA::Major, b_major>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e5m2_t, cute::float_e5m2_t, half_t, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<M == 128 && K == 32>> {
  using MMA = MMA_Traits<SM100_MMA_F8F6F4_SS, cute::float_e5m2_t,
                         cute::float_e5m2_t, half_t, Int<M>, Int<N>,
                         integral_constant<UMMA::Major, a_major>,
                         integral_constant<UMMA::Major, b_major>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};
template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e5m2_t, cute::float_e5m2_t, half_t, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<(M == 64 || M == 32) && K == 32>> {
  using MMA = MMA_Traits<SM100_MMA_F8F6F4_WS_SS, cute::float_e5m2_t,
                         cute::float_e5m2_t, half_t, Int<M>, Int<N>,
                         integral_constant<UMMA::Major, a_major>,
                         integral_constant<UMMA::Major, b_major>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

// FP4 (float_e2m1_t) — C = float
template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e2m1_t, cute::float_e2m1_t, float, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<M == 128 && K == 32>> {
  using MMA =
      MMA_Traits<SM100_MMA_F8F6F4_SS, cute::float_e2m1_t, cute::float_e2m1_t,
                 float, Int<M>, Int<N>, integral_constant<UMMA::Major, a_major>,
                 integral_constant<UMMA::Major, b_major>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e2m1_t, cute::float_e2m1_t, float, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<(M == 64 || M == 32) && K == 32>> {
  using MMA =
      MMA_Traits<SM100_MMA_F8F6F4_WS_SS, cute::float_e2m1_t, cute::float_e2m1_t,
                 float, Int<M>, Int<N>, integral_constant<UMMA::Major, a_major>,
                 integral_constant<UMMA::Major, b_major>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

// FP4 (float_e2m1_t) — C = half_t
template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e2m1_t, cute::float_e2m1_t, half_t, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<M == 128 && K == 32>> {
  using MMA = MMA_Traits<SM100_MMA_F8F6F4_SS, cute::float_e2m1_t,
                         cute::float_e2m1_t, half_t, Int<M>, Int<N>,
                         integral_constant<UMMA::Major, a_major>,
                         integral_constant<UMMA::Major, b_major>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};
template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e2m1_t, cute::float_e2m1_t, half_t, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<(M == 64 || M == 32) && K == 32>> {
  using MMA = MMA_Traits<SM100_MMA_F8F6F4_WS_SS, cute::float_e2m1_t,
                         cute::float_e2m1_t, half_t, Int<M>, Int<N>,
                         integral_constant<UMMA::Major, a_major>,
                         integral_constant<UMMA::Major, b_major>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

// Mixed A8W4 (float_e4m3_t x float_e2m1_t) — C = float
template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e4m3_t, cute::float_e2m1_t, float, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<M == 128 && K == 32>> {
  using MMA =
      MMA_Traits<SM100_MMA_F8F6F4_SS, cute::float_e4m3_t, cute::float_e2m1_t,
                 float, Int<M>, Int<N>, integral_constant<UMMA::Major, a_major>,
                 integral_constant<UMMA::Major, b_major>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e4m3_t, cute::float_e2m1_t, float, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<(M == 64 || M == 32) && K == 32>> {
  using MMA =
      MMA_Traits<SM100_MMA_F8F6F4_WS_SS, cute::float_e4m3_t, cute::float_e2m1_t,
                 float, Int<M>, Int<N>, integral_constant<UMMA::Major, a_major>,
                 integral_constant<UMMA::Major, b_major>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

// Mixed A8W4 (float_e4m3_t x float_e2m1_t) — C = half_t
template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e4m3_t, cute::float_e2m1_t, half_t, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<M == 128 && K == 32>> {
  using MMA = MMA_Traits<SM100_MMA_F8F6F4_SS, cute::float_e4m3_t,
                         cute::float_e2m1_t, half_t, Int<M>, Int<N>,
                         integral_constant<UMMA::Major, a_major>,
                         integral_constant<UMMA::Major, b_major>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};
template <int M, int N, int K, UMMA::Major a_major, UMMA::Major b_major>
struct DispatchInstruction<cute::float_e4m3_t, cute::float_e2m1_t, half_t, M, N,
                           K, a_major, b_major,
                           std::enable_if_t<(M == 64 || M == 32) && K == 32>> {
  using MMA = MMA_Traits<SM100_MMA_F8F6F4_WS_SS, cute::float_e4m3_t,
                         cute::float_e2m1_t, half_t, Int<M>, Int<N>,
                         integral_constant<UMMA::Major, a_major>,
                         integral_constant<UMMA::Major, b_major>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                         integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>;
};

template <int M, int N, int K, int AtomM, int AtomN, int AtomK, bool trans_A,
          bool trans_B, typename A_type_raw, typename B_type_raw,
          typename C_type_raw>
class GemmTensorOp {
public:
  using A_type_cute = typename tl::to_cute_type<A_type_raw>::type;
  using B_type_cute = typename tl::to_cute_type<B_type_raw>::type;
  using A_type =
      typename std::conditional<std::is_same<A_type_cute, float>::value,
                                tfloat32_t, A_type_cute>::type;
  using B_type =
      typename std::conditional<std::is_same<B_type_cute, float>::value,
                                tfloat32_t, B_type_cute>::type;
  using C_type = C_type_raw;

  static_assert(AtomM == 128 || AtomM == 64 || AtomM == 32);

  static constexpr UMMA::Major UmmaMajorA =
      trans_A ? UMMA::Major::MN : UMMA::Major::K;
  static constexpr UMMA::Major UmmaMajorB =
      trans_B ? UMMA::Major::K : UMMA::Major::MN;

  using SmemLayoutAtomA =
      decltype(sm100_smem_selector<UmmaMajorA, A_type, Int<M>, Int<K>>());
  using SmemLayoutAtomB =
      decltype(sm100_smem_selector<UmmaMajorB, B_type, Int<N>, Int<K>>());

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{}, Shape<Int<M>, Int<K>>{},
      conditional_t<trans_A, Step<_2, _1>, Step<_1, _2>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{}, Shape<Int<N>, Int<K>>{},
      conditional_t<trans_B, Step<_1, _2>, Step<_2, _1>>{}));

  static CUTE_DEVICE void body_ss(A_type_raw *pA, B_type_raw *pB, uint32_t pC,
                                  uint64_t *umma_bar_ptr, bool clear_accum) {
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                            SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                            SmemLayoutB{});

    // TODO (lei): Normal TCGEN5MMA (the one w/o ws) don't saturate all 128
    // lanes when M == 64
    // (see layout F in
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-f)
    // So we use the .ws variant here
    using MmaAtom =
        typename DispatchInstruction<A_type, B_type, C_type, AtomM, AtomN,
                                     AtomK, UmmaMajorA, UmmaMajorB>::MMA;
    auto tiled_mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_1>>{},
                                    Tile<Int<M>, Int<N>, Int<K>>{});
    auto thr_mma = tiled_mma.get_slice(_0{});
    tiled_mma.accumulate_ =
        clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    Tensor acc = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
    acc.data() = pC;

    Tensor sA_frag = thr_mma.partition_fragment_A(sA);
    Tensor sB_frag = thr_mma.partition_fragment_B(sB);
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(sA_frag); ++k_block) {
      cute::gemm(tiled_mma, sA_frag(_, _, k_block), sB_frag(_, _, k_block),
                 acc);
      tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }

    cutlass::arch::umma_arrive(umma_bar_ptr);
  }
};

} // namespace tl_tcgen5mma

} // namespace cute

namespace tl {

using tl_mma::gemm_rs;
using tl_mma::gemm_sr;
using tl_mma::gemm_ss;

// TODO (lei): Implement gemm_ts
// template <int M, int N, int K, int warp_m, int warp_n, bool trans_A, bool
// trans_B, bool clear_accum, typename A_type, typename B_type, typename C_type>
// TL_DEVICE void gemm_ts(A_type *pA, B_type *pB, C_type *accum, uint64_t
// *umma_bar_ptr) {
// }

template <int M, int N, int K, int AtomM, int AtomN, int AtomK, bool trans_A,
          bool trans_B, typename C_type, typename A_type, typename B_type,
          typename Barrier_type>
TL_DEVICE void tcgen5mma_gemm_ss(A_type *pA, B_type *pB, uint32_t accum,
                                 Barrier_type *umma_bar_ptr, bool clear_accum) {
  using MMA =
      cute::tl_tcgen5mma::GemmTensorOp<M, N, K, AtomM, AtomN, AtomK, trans_A,
                                       trans_B, A_type, B_type, C_type>;
  MMA::body_ss(pA, pB, accum, reinterpret_cast<uint64_t *>(umma_bar_ptr),
               clear_accum);
}

} // namespace tl
