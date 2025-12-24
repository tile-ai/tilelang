#pragma once

#include <cute/algorithm/clear.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/underscore.hpp>

#include "common.h"
#include "intrin.h"

namespace cute::tl_mma {

template <typename A_type, typename B_type, typename C_type, int num_warp_m,
          int num_warp_n, int N>
struct DispatchInstruction;

using _X = Underscore;

} // namespace cute::tl_mma

#define TL_DISPATCH_MMA(A_type, B_type, C_type, MMA_instr)                     \
  namespace cute::tl_mma {                                                     \
  template <int num_warp_m, int num_warp_n, int N>                             \
  struct DispatchInstruction<A_type, B_type, C_type, num_warp_m, num_warp_n,   \
                             N> {                                              \
    using MMA = MMA_Atom<MMA_instr>;                                           \
    using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _X>;         \
  };                                                                           \
  }
#define TL_DISPATCH_MMA_TEMPLATE(A_type, B_type, C_type, MMA_instr)            \
  namespace cute::tl_mma {                                                     \
  template <int num_warp_m, int num_warp_n, int N>                             \
  struct DispatchInstruction<A_type, B_type, C_type, num_warp_m, num_warp_n,   \
                             N> {                                              \
    using MMA = MMA_Atom<MMA_instr<A_type, B_type, C_type>>;                   \
    using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _X>;         \
  };                                                                           \
  }

#ifdef __CUDA_ARCH_LIST__
#if __CUDA_ARCH_LIST__ >= 1200
#include "cuda_fp8.h"
#include <cute/arch/mma_sm120.hpp>
#include <cute/arch/mma_sm80.hpp>
TL_DISPATCH_MMA_TEMPLATE(fp8_e4_t, fp8_e4_t, float, SM120_16x8x32_TN)
TL_DISPATCH_MMA_TEMPLATE(fp8_e5_t, fp8_e5_t, float, SM120_16x8x32_TN)
TL_DISPATCH_MMA(half_t, half_t, half_t, SM80_16x8x16_F16F16F16F16_TN)
TL_DISPATCH_MMA(half_t, half_t, float, SM80_16x8x16_F32F16F16F32_TN)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, SM80_16x8x16_F32BF16BF16F32_TN)
TL_DISPATCH_MMA(tfloat32_t, tfloat32_t, float, SM80_16x8x8_F32TF32TF32F32_TN)
TL_DISPATCH_MMA(int8_t, int8_t, int, SM80_16x8x32_S32S8S8S32_TN)
TL_DISPATCH_MMA(double, double, double, SM80_8x8x4_F64F64F64F64_TN)
#elif __CUDA_ARCH_LIST__ >= 1000
#include "cuda_fp8.h"
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm89.hpp>
TL_DISPATCH_MMA(fp8_e4_t, fp8_e4_t, float, SM89_16x8x32_F32E4M3E4M3F32_TN)
TL_DISPATCH_MMA(fp8_e5_t, fp8_e5_t, float, SM89_16x8x32_F32E5M2E5M2F32_TN)
TL_DISPATCH_MMA(half_t, half_t, half_t, SM80_16x8x16_F16F16F16F16_TN)
TL_DISPATCH_MMA(half_t, half_t, float, SM80_16x8x16_F32F16F16F32_TN)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, SM80_16x8x16_F32BF16BF16F32_TN)
TL_DISPATCH_MMA(tfloat32_t, tfloat32_t, float, SM80_16x8x8_F32TF32TF32F32_TN)
TL_DISPATCH_MMA(int8_t, int8_t, int, SM80_16x8x32_S32S8S8S32_TN)
TL_DISPATCH_MMA(double, double, double, SM80_8x8x4_F64F64F64F64_TN)
#elif __CUDA_ARCH_LIST__ >= 900
#include "cuda_fp8.h"
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm89.hpp>
TL_DISPATCH_MMA(fp8_e4_t, fp8_e4_t, float, SM89_16x8x32_F32E4M3E4M3F32_TN)
TL_DISPATCH_MMA(fp8_e5_t, fp8_e5_t, float, SM89_16x8x32_F32E5M2E5M2F32_TN)
TL_DISPATCH_MMA(half_t, half_t, half_t, SM80_16x8x16_F16F16F16F16_TN)
TL_DISPATCH_MMA(half_t, half_t, float, SM80_16x8x16_F32F16F16F32_TN)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, SM80_16x8x16_F32BF16BF16F32_TN)
TL_DISPATCH_MMA(tfloat32_t, tfloat32_t, float, SM80_16x8x8_F32TF32TF32F32_TN)
TL_DISPATCH_MMA(int8_t, int8_t, int, SM80_16x8x32_S32S8S8S32_TN)
TL_DISPATCH_MMA(double, double, double, SM80_8x8x4_F64F64F64F64_TN)
#elif __CUDA_ARCH_LIST__ >= 890
#include "cuda_fp8.h"
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm89.hpp>
TL_DISPATCH_MMA(fp8_e4_t, fp8_e4_t, float, SM89_16x8x32_F32E4M3E4M3F32_TN)
TL_DISPATCH_MMA(fp8_e5_t, fp8_e5_t, float, SM89_16x8x32_F32E5M2E5M2F32_TN)
TL_DISPATCH_MMA(half_t, half_t, half_t, SM80_16x8x16_F16F16F16F16_TN)
TL_DISPATCH_MMA(half_t, half_t, float, SM80_16x8x16_F32F16F16F32_TN)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, SM80_16x8x16_F32BF16BF16F32_TN)
TL_DISPATCH_MMA(tfloat32_t, tfloat32_t, float, SM80_16x8x8_F32TF32TF32F32_TN)
TL_DISPATCH_MMA(int8_t, int8_t, int, SM80_16x8x32_S32S8S8S32_TN)
TL_DISPATCH_MMA(double, double, double, SM80_8x8x4_F64F64F64F64_TN)
#elif __CUDA_ARCH_LIST__ >= 800
#include <cute/arch/mma_sm80.hpp>
TL_DISPATCH_MMA(half_t, half_t, half_t, SM80_16x8x16_F16F16F16F16_TN)
TL_DISPATCH_MMA(half_t, half_t, float, SM80_16x8x16_F32F16F16F32_TN)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, SM80_16x8x16_F32BF16BF16F32_TN)
TL_DISPATCH_MMA(tfloat32_t, tfloat32_t, float, SM80_16x8x8_F32TF32TF32F32_TN)
TL_DISPATCH_MMA(int8_t, int8_t, int, SM80_16x8x32_S32S8S8S32_TN)
TL_DISPATCH_MMA(double, double, double, SM80_8x8x4_F64F64F64F64_TN)
#elif __CUDA_ARCH_LIST__ >= 750
TL_DISPATCH_MMA(half_t, half_t, float, SM75_16x8x8_F32F16F16F32_TN)
#endif
#endif
#undef TL_DISPATCH_MMA
#undef TL_DISPATCH_MMA_TEMPLATE

namespace cute::tl_mma {

template <int N, int num_warp_n, bool transpose> struct SelectCopy {
  static constexpr int remainder = (N / num_warp_n) % 16;
  using type = std::conditional_t<
      remainder == 4 || remainder == 8 || remainder == 0,
      std::conditional_t<
          transpose,
          std::conditional_t<
              remainder == 4, SM75_U32x1_LDSM_N,
              std::conditional_t<remainder == 8, SM75_U32x2_LDSM_N,
                                 SM75_U32x4_LDSM_N>>,
          std::conditional_t<
              remainder == 4, SM75_U16x2_LDSM_T,
              std::conditional_t<remainder == 8, SM75_U16x4_LDSM_T,
                                 SM75_U16x8_LDSM_T>>>,
      DefaultCopy>;
};

template <int Bits, int N, int K, bool K_inner, int num_warp_n, int leading_dim,
          typename Enable = void>
struct OperandTraits {
  // Primary template, use padded layout and default copy
  static constexpr int stride = leading_dim;
  static constexpr int padded =
      stride % (256 / Bits) == 0 ? stride + 128 / Bits : stride;
  using Layout = typename std::conditional<
      K_inner, Layout<Shape<Int<N>, Int<leading_dim>>, Shape<Int<padded>, _1>>,
      Layout<Shape<Int<leading_dim>, Int<K>>, Shape<_1, Int<padded>>>>::type;
  using Copy = DefaultCopy;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<16, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 64 == 32>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<16, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 64 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<16, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 64 == 32>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 3, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = typename SelectCopy<N, num_warp_n, false>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<16, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 64 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = typename SelectCopy<N, num_warp_n, false>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<32, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 32 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 2, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<32, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 32 == 16>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 2, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<32, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 32 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 2, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = UniversalCopy<tfloat32_t>;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<32, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 32 == 16>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 2, 3>{}, Layout<Shape<_16, _8>, Stride<_1, _16>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = UniversalCopy<tfloat32_t>;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<8, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 128 == 64>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 4, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<8, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 128 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 4, 3>{}, Layout<Shape<_8, _128>, Stride<_128, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<64, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 16 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 0, 4>{}, Layout<Shape<_4, _16>, Stride<_16, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = DefaultCopy;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<64, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 16 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 2, 2>{}, Layout<Shape<_16, _4>, Stride<_1, _16>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = DefaultCopy;
};

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, typename A_type_raw, typename B_type_raw,
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

  using Instruction = DispatchInstruction<A_type_raw, B_type_raw, C_type_raw,
                                          num_warp_m, num_warp_n, N>;

  using OperandATraits = OperandTraits<sizeof_bits<A_type>::value, M, K,
                                       !trans_A, num_warp_m, lda>;
  using OperandBTraits =
      OperandTraits<sizeof_bits<B_type>::value, N, K, trans_B, num_warp_n, ldb>;

  using SmemLayoutA = typename OperandATraits::Layout;
  using SmemLayoutB = typename OperandBTraits::Layout;
  using SmemCopyA = Copy_Atom<typename OperandATraits::Copy, A_type>;
  using SmemCopyB = Copy_Atom<typename OperandBTraits::Copy, B_type>;

  using TileMma = TiledMMA<typename Instruction::MMA,
                           Layout<Shape<Int<num_warp_m>, Int<num_warp_n>, _1>>,
                           typename Instruction::MMA_Group>;

  template <class... Args>
  static CUTE_DEVICE auto remove_swizzle(Layout<Args...> const &layout) {
    return layout;
  }
  // In fp16, when layout is KxN and n_warp is 1 and N % 64 == 0
  // the original layout fail to compile, currently using this as a workaround
  template <class... Args>
  static CUTE_DEVICE auto
  remove_swizzle(ComposedLayout<Args...> const &layout) {
    if constexpr (sizeof(A_type) == 2)
      return layout.layout_b();
    else
      return layout;
  }

  template <int offset, int NN, int KK, bool trans, int lddim, typename Engine0,
            typename Layout0>
  static CUTE_DEVICE auto get_region_tensor(Tensor<Engine0, Layout0> &sa) {
    if constexpr (offset == 0) {
      return composition(
          sa,
          Layout<Shape<Int<NN>, Int<KK>>,
                 Stride<_1, typename std::conditional<trans, Int<NN>,
                                                      Int<lddim>>::type>>{});
    } else {
      if constexpr (trans) {
        static_assert(offset % KK == 0, "Offset must be a multiple of K");
        constexpr int offset_n = offset / KK;
        return flat_divide(sa, Shape<Int<NN>, Int<KK>>{})(_, _, _0{},
                                                          Int<offset_n>{});
      } else {
        static_assert(offset % NN == 0, "Offset must be a multiple of N");
        constexpr int offset_n = offset / NN;
        return flat_divide(sa, Shape<Int<NN>, Int<KK>>{})(_, _, Int<offset_n>{},
                                                          _0{});
      }
    }
  }

  static CUTE_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sA_all = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                                SmemLayoutA{});
    Tensor sB_all = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                                SmemLayoutB{});
    Tensor sA = get_region_tensor<offset_a, M, K, !trans_A, lda>(sA_all);
    Tensor sB = get_region_tensor<offset_b, N, K, trans_B, ldb>(sB_all);
    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_A = make_tiled_copy_A(SmemCopyA{}, tiled_mma);
    auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
    auto thr_copy_A = tiled_copy_A.get_thread_slice(tid);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCsA = thr_copy_A.partition_S(sA);
    Tensor tCsB = thr_copy_B.partition_S(sB);

    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    // when layout is KxN and n_warp is 1, there seem to be a bug, use this as a
    // workaround
    auto tCrA_view = make_tensor(tCrA.data(), remove_swizzle(tCrA.layout()));
    auto tCrB_view = make_tensor(tCrB.data(), remove_swizzle(tCrB.layout()));
    if constexpr (clear_accum) {
      clear(acc);
    }
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      copy(tiled_copy_A, tCsA(_, _, k), tCrA_copy_view(_, _, k));
      copy(tiled_copy_B, tCsB(_, _, k), tCrB_copy_view(_, _, k));
      gemm(tiled_mma, tCrA_view(_, _, k), tCrB_view(_, _, k), acc);
    }
  }

  static CUTE_DEVICE void body_rs(A_type_raw *pA, B_type_raw *pB,
                                  C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sB_all = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                                SmemLayoutB{});
    Tensor sB = get_region_tensor<offset_b, N, K, trans_B, ldb>(sB_all);
    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCsB = thr_copy_B.partition_S(sB);

    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));
    Tensor tCrA =
        make_tensor(make_rmem_ptr(reinterpret_cast<A_type *>(pA)),
                    partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));
    auto tCrB_view = make_tensor(tCrB.data(), remove_swizzle(tCrB.layout()));
    if constexpr (clear_accum) {
      clear(acc);
    }
    copy(tiled_copy_B, tCsB(_, _, 0), tCrB_copy_view(_, _, 0));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      if (k < size<2>(tCrA) - 1) {
        copy(tiled_copy_B, tCsB(_, _, k + 1), tCrB_copy_view(_, _, k + 1));
      }
      gemm(tiled_mma, tCrA(_, _, k), tCrB_view(_, _, k), acc);
    }
  }

  static CUTE_DEVICE void body_sr(A_type_raw *pA, B_type_raw *pB,
                                  C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sA_all = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                                SmemLayoutA{});
    Tensor sA = get_region_tensor<offset_a, M, K, !trans_A, lda>(sA_all);
    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_A = make_tiled_copy_A(SmemCopyA{}, tiled_mma);
    auto thr_copy_A = tiled_copy_A.get_thread_slice(tid);

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCsA = thr_copy_A.partition_S(sA);

    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));
    Tensor tCrB =
        make_tensor(make_rmem_ptr(reinterpret_cast<B_type *>(pB)),
                    partition_shape_B(tiled_mma, Shape<Int<N>, Int<K>>{}));
    auto tCrA_view = make_tensor(tCrA.data(), remove_swizzle(tCrA.layout()));
    if constexpr (clear_accum) {
      clear(acc);
    }
    copy(tiled_copy_A, tCsA(_, _, 0), tCrA_copy_view(_, _, 0));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      if (k < size<2>(tCrA) - 1) {
        copy(tiled_copy_A, tCsA(_, _, k + 1), tCrA_copy_view(_, _, k + 1));
      }
      gemm(tiled_mma, tCrA_view(_, _, k), tCrB(_, _, k), acc);
    }
  }
};

} // namespace cute::tl_mma

namespace tl::tl_mma {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {
  using MMA =
      cute::tl_mma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                 trans_B, clear_accum, lda, ldb, offset_a,
                                 offset_b, A_type, B_type, C_type>;
  MMA::body(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_rs(A_type *pA, B_type *pB, C_type *accum) {
  using MMA =
      cute::tl_mma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                 trans_B, clear_accum, lda, ldb, offset_a,
                                 offset_b, A_type, B_type, C_type>;
  MMA::body_rs(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, typename A_type, typename B_type, typename C_type>
CUTLASS_DEVICE void gemm_sr(A_type *pA, B_type *pB, C_type *accum) {
  using MMA =
      cute::tl_mma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                 trans_B, clear_accum, lda, ldb, offset_a,
                                 offset_b, A_type, B_type, C_type>;
  MMA::body_sr(pA, pB, accum);
}

} // namespace tl::tl_mma
