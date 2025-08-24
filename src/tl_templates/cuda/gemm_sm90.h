#pragma once

#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>

#include "common.h"

namespace cute {

using namespace SM90;

namespace tl_wgmma {

using namespace cutlass::gemm::collective::detail; // ss_smem_selector

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
public:
  using A_type = conditional_t<std::is_same<A_type_raw, float>::value,
                               tfloat32_t, A_type_raw>;
  using B_type = conditional_t<std::is_same<B_type_raw, float>::value,
                               tfloat32_t, B_type_raw>;
  using C_type = C_type_raw;

  static constexpr GMMA::Major GmmaMajorA =
      trans_A ? GMMA::Major::MN : GMMA::Major::K;
  static constexpr GMMA::Major GmmaMajorB =
      trans_B ? GMMA::Major::K : GMMA::Major::MN;

  using SmemLayoutAtomA =
      decltype(ss_smem_selector<GmmaMajorA, A_type, Int<M>, Int<K>>());
  using SmemLayoutAtomB =
      decltype(ss_smem_selector<GmmaMajorB, B_type, Int<N>, Int<K>>());

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{}, Shape<Int<M>, Int<K>>{},
      conditional_t<trans_A, Step<_2, _1>, Step<_1, _2>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{}, Shape<Int<N>, Int<K>>{},
      conditional_t<trans_B, Step<_1, _2>, Step<_2, _1>>{}));

  static_assert(num_warp_m % 4 == 0,
                "num_warp_m must be a multiple of 4 for hopper wgmma");

  template <int wg_wait = 0>
  static CUTE_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                            SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                            SmemLayoutB{});
    auto tiled_mma = make_tiled_mma(
        GMMA::ss_op_selector<
            A_type, B_type, C_type,
            Shape<Int<4 * M / num_warp_m>, Int<N / num_warp_n>, Int<K>>,
            GmmaMajorA, GmmaMajorB>(),
        Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>{});
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // Allocate registers for pipelining
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_M,MMA_N,PIPE)

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    if constexpr (clear_accum) {
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // warpgroup_arrive();
      // (V,M) x (V,N) => (V,M,N)
      gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }

    warpgroup_commit_batch();
    if constexpr (wg_wait >= 0) {
      warpgroup_wait<wg_wait>();
    }
    warpgroup_fence_operand(acc);
  }

  template <int wg_wait = 0>
  static CUTE_DEVICE void body_rs(A_type_raw *pA, B_type_raw *pB,
                                  C_type_raw *pC) {
    // TODO: Move bar.sync out of body_rs
    // asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(num_warp_m * num_warp_n *
    // 32));
    const int tid = threadIdx.x;
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                            SmemLayoutB{});
    auto tiled_mma = make_tiled_mma(
        GMMA::rs_op_selector<
            A_type, B_type, C_type,
            Shape<Int<M / (num_warp_m / 4)>, Int<N / num_warp_n>, Int<K>>,
            GmmaMajorA, GmmaMajorB>(),
        Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>{});
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // Allocate registers for pipelining
    Tensor tCsB = thr_mma.partition_B(sB);       // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_M,MMA_N,PIPE)
    Tensor tCrA =
        make_tensor(make_rmem_ptr(reinterpret_cast<A_type *>(pA)),
                    partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));
    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    warpgroup_fence_operand(tCrA);
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    if constexpr (clear_accum) {
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // warpgroup_arrive();
      // (V,M) x (V,N) => (V,M,N)
      gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    if constexpr (wg_wait >= 0) {
      warpgroup_wait<wg_wait>();
    }
    warpgroup_fence_operand(acc);
    warpgroup_fence_operand(tCrA);
  }
};

} // namespace tl_wgmma

namespace tl_mma {

template <typename A_type, typename B_type, typename C_type, int num_warp_m,
          int num_warp_n, int N>
struct DispatchInstruction;

using _X = Underscore;

template <int num_warp_m, int num_warp_n, int N>
struct DispatchInstruction<fp8_e4_t, fp8_e4_t, float, num_warp_m, num_warp_n,
                           N> {
  using MMA = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;
  using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _X>;
};
template <int num_warp_m, int num_warp_n, int N>
struct DispatchInstruction<fp8_e5_t, fp8_e5_t, float, num_warp_m, num_warp_n,
                           N> {
  using MMA = MMA_Atom<SM89_16x8x32_F32E5M2E5M2F32_TN>;
  using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _X>;
};

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 800))
template <int num_warp_m, int num_warp_n, int N>
struct DispatchInstruction<half_t, half_t, half_t, num_warp_m, num_warp_n, N> {
  using MMA = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
  using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _X>;
};
template <int num_warp_m, int num_warp_n, int N>
struct DispatchInstruction<half_t, half_t, float, num_warp_m, num_warp_n, N> {
  using MMA = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _X>;
};
template <int num_warp_m, int num_warp_n, int N>
struct DispatchInstruction<bfloat16_t, bfloat16_t, float, num_warp_m,
                           num_warp_n, N> {
  using MMA = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _X>;
};
template <int num_warp_m, int num_warp_n, int N>
struct DispatchInstruction<tfloat32_t, tfloat32_t, float, num_warp_m,
                           num_warp_n, N> {
  using MMA = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>;
  using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _X>;
};
template <int num_warp_m, int num_warp_n, int N>
struct DispatchInstruction<int8_t, int8_t, int, num_warp_m, num_warp_n, N> {
  using MMA = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
  using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _X>;
};
template <int num_warp_m, int num_warp_n, int N>
struct DispatchInstruction<double, double, double, num_warp_m, num_warp_n, N> {
  using MMA = MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>;
  using MMA_Group = Tile<Int<num_warp_m * 16>, Int<num_warp_n * 16>, _X>;
};
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 750))
template <int num_warp_m, int num_warp_n, int N>
struct DispatchInstruction<half_t, half_t, float, num_warp_m, num_warp_n, N> {
  using MMA = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
  using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _16>;
};
#endif

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
  using Copy = typename std::conditional<N == 8 * num_warp_n, SM75_U32x2_LDSM_N,
                                         SM75_U32x4_LDSM_N>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<8, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 128 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 4, 3>{}, Layout<Shape<_8, _128>, Stride<_128, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename std::conditional<N == 8 * num_warp_n, SM75_U32x2_LDSM_N,
                                         SM75_U32x4_LDSM_N>::type;
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
  using A_type =
      typename std::conditional<std::is_same<A_type_raw, float>::value,
                                tfloat32_t, A_type_raw>::type;
  using B_type =
      typename std::conditional<std::is_same<B_type_raw, float>::value,
                                tfloat32_t, A_type_raw>::type;
  using C_type = C_type_raw;

  using Instruction =
      DispatchInstruction<A_type, B_type, C_type, num_warp_m, num_warp_n, N>;

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

} // namespace tl_mma

} // namespace cute
/**
 * Execute a tiled GEMM where A is read from global memory and B is staged in
 * shared memory.
 *
 * Dispatches to tl_mma::GemmTensorOp<M,N,K,...>::body_rs to perform the
 * computation.
 *
 * @param pA Pointer to the A tile region (device memory).
 * @param pB Pointer to the B tile region (device memory).
 * @param accum Pointer to the accumulator/output tile region (device memory).
 */
/**
 * Execute a tiled GEMM where A is staged in shared memory and B is read from
 * global memory.
 *
 * Dispatches to tl_mma::GemmTensorOp<M,N,K,...>::body_sr to perform the
 * computation.
 *
 * @param pA Pointer to the A tile region (device memory).
 * @param pB Pointer to the B tile region (device memory).
 * @param accum Pointer to the accumulator/output tile region (device memory).
 */
/**
 * Perform a tiled GEMM (both operands in shared memory or selected backend) and
 * write to accum.
 *
 * If use_wgmma is true, validates wgmma constraints (strides and offsets) and
 * dispatches to the Hopper wgmma implementation; otherwise dispatches to the
 * tl_mma implementation.
 *
 * @param pA Pointer to the A tile region (device memory).
 * @param pB Pointer to the B tile region (device memory).
 * @param accum Pointer to the accumulator/output tile region (device memory).
 */
/**
 * Perform a tiled GEMM with A in global memory and B in shared memory (or
 * selected backend).
 *
 * If use_wgmma is true, validates wgmma constraints (strides and offsets) and
 * dispatches to the Hopper wgmma read-share implementation; otherwise
 * dispatches to the tl_mma read-share.
 *
 * @param pA Pointer to the A tile region (device memory).
 * @param pB Pointer to the B tile region (device memory).
 * @param accum Pointer to the accumulator/output tile region (device memory).
 */
/**
 * Perform a tiled GEMM with A staged in shared memory and B in global memory
 * (tl_mma only).
 *
 * wgmma does not support this variant; caller must set use_wgmma == false.
 * Dispatches to tl_mma::GemmTensorOp<M,N,K,...>::body_sr.
 *
 * @param pA Pointer to the A tile region (device memory).
 * @param pB Pointer to the B tile region (device memory).
 * @param accum Pointer to the accumulator/output tile region (device memory).
 */
/**
 * Wait for a warp-group of WMMA/MMA warps to complete.
 *
 * Wrapper around cute::warpgroup_wait for the specified number of MMA warps.
 */
/**
 * Synchronize a named barrier across NumMmaThreads MMA threads.
 *
 * Calls cutlass::arch::NamedBarrier::sync with the canonical warp-group id.
 */
/**
 * Arrive at a named barrier for NumMmaThreads MMA threads using
 * architecture-aware mapping.
 *
 * Supported NumMmaThreads values: 256 or 384. The function issues one or two
 * barrier arrives depending on the thread-group topology to ensure proper
 * rendezvous ordering.
 */
/**
 * Initialize named-barrier state for multi-warp MMA execution.
 *
 * For NumMmaThreads == 256 or 384, performs the required initial barrier
 * arrivals for non-zero canonical warp-group indices to set up subsequent
 * barrier synchronization.
 */

namespace tl {

namespace tl_mma {

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

} // namespace tl_mma

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum = false, int lda = 0, int ldb = 0,
          int offset_a = 0, int offset_b = 0, bool use_wgmma = true,
          int wg_wait = 0, typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {
  if constexpr (use_wgmma) {
    static_assert((trans_A && lda == M) || (!trans_A && lda == K),
                  "Hopper wgmma doesn't support custom stride for A");
    static_assert((trans_B && ldb == K) || (!trans_B && ldb == N),
                  "Hopper wgmma doesn't support custom stride for B");
    static_assert(offset_a == 0 && offset_b == 0,
                  "offset_a and offset_b must be zero for wgmma");
    using MMA = cute::tl_wgmma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n,
                                             trans_A, trans_B, clear_accum,
                                             A_type, B_type, C_type>;
    MMA::body<wg_wait>(pA, pB, accum);
  } else {
    using MMA =
        cute::tl_mma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                   trans_B, clear_accum, lda, ldb, offset_a,
                                   offset_b, A_type, B_type, C_type>;
    MMA::body(pA, pB, accum);
  }
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum = false, int lda = 0, int ldb = 0,
          int offset_a = 0, int offset_b = 0, bool use_wgmma = true,
          int wg_wait = 0, typename A_type, typename B_type, typename C_type>
TL_DEVICE /**
           * Perform a read-share (B in shared memory, A in global) tiled GEMM
           * and accumulate into `accum`.
           *
           * Dispatches at compile time to either the Hopper wgmma
           * implementation or the fallback MMA implementation depending on
           * `use_wgmma`. The selected GemmTensorOp::body_rs performs the
           * region-tiled GEMM loop and updates the accumulator in-place.
           *
           * When `use_wgmma == true`, this function enforces wgmma constraints
           * at compile time:
           * - A's leading dimension must equal (trans_A ? M : K)
           * - B's leading dimension must equal (trans_B ? K : N)
           * - offset_a and offset_b must be zero
           *
           * @param pA Pointer to operand A (global memory). Layout/stride
           * expectations depend on template parameters.
           * @param pB Pointer to operand B (base for shared-memory staging).
           * Layout/stride expectations depend on template parameters.
           * @param accum Pointer to the accumulator/output C buffer updated
           * in-place.
           */
    void
    gemm_rs(A_type *pA, B_type *pB, C_type *accum) {
  if constexpr (use_wgmma) {
    static_assert((trans_A && lda == M) || (!trans_A && lda == K),
                  "Hopper wgmma doesn't support custom stride for A");
    static_assert((trans_B && ldb == K) || (!trans_B && ldb == N),
                  "Hopper wgmma doesn't support custom stride for B");
    static_assert(offset_a == 0 && offset_b == 0,
                  "offset_a and offset_b must be zero for wgmma");
    using MMA = cute::tl_wgmma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n,
                                             trans_A, trans_B, clear_accum,
                                             A_type, B_type, C_type>;
    MMA::body_rs<wg_wait>(pA, pB, accum);
  } else {
    using MMA =
        cute::tl_mma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                   trans_B, clear_accum, lda, ldb, offset_a,
                                   offset_b, A_type, B_type, C_type>;
    MMA::body_rs(pA, pB, accum);
  }
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum = false, int lda = 0, int ldb = 0,
          int offset_a = 0, int offset_b = 0, bool use_wgmma = true,
          int wg_wait = 0, typename A_type, typename B_type, typename C_type>
TL_DEVICE /**
           * Perform a non-wgmma tiled GEMM where A regions are staged into
           * shared memory and B is read directly from global memory,
           * accumulating into `accum`.
           *
           * This overload dispatches to the tl_mma::GemmTensorOp::body_sr
           * implementation. Must be instantiated with `use_wgmma = false`
           * (enforced via static_assert).
           *
           * @param pA Pointer to the A operand in global memory (source that
           * will be staged to shared memory).
           * @param pB Pointer to the B operand in global memory (read
           * directly).
           * @param accum Pointer to the output accumulator matrix in global
           * memory.
           */
    void
    gemm_sr(A_type *pA, B_type *pB, C_type *accum) {
  static_assert(!use_wgmma, "wgmma doesn't support gemm_sr");
  using MMA =
      cute::tl_mma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                 trans_B, clear_accum, lda, ldb, offset_a,
                                 offset_b, A_type, B_type, C_type>;
  MMA::body_sr(pA, pB, accum);
}

template <int num_mma>
TL_DEVICE /**
           * Wait for all WMMA/MMA warps in the current warp-group to
           * synchronize.
           *
           * Blocks until the warp-group-wide rendezvous for `num_mma` MMA lanes
           * completes, ensuring all participating warps have arrived before
           * proceeding.
           */
    void
    wait_wgmma() {
  cute::warpgroup_wait<num_mma>();
}

template <int NumMmaThreads> TL_DEVICE void warp_scheduler_barrier_sync() {
  cutlass::arch::NamedBarrier::sync(NumMmaThreads,
                                    cutlass::canonical_warp_group_idx() /*id*/);
}

template <int NumMmaThreads> TL_DEVICE void warp_scheduler_barrier_arrive() {
  static_assert(NumMmaThreads == 256 || NumMmaThreads == 384);
  if constexpr (NumMmaThreads == 256) {
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreads, (1 - cutlass::canonical_warp_group_idx()) /*id*/);
  } else {
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreads,
        (cutlass::canonical_warp_group_idx() <= 1
             ? cutlass::canonical_warp_group_idx() + 1
             : cutlass::canonical_warp_group_idx() + 1 - 3) /*id*/);
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreads,
        (cutlass::canonical_warp_group_idx() <= 0
             ? cutlass::canonical_warp_group_idx() + 2
             : cutlass::canonical_warp_group_idx() + 2 - 3) /*id*/);
  }
}

template <int NumMmaThreads> TL_DEVICE void mma_init() {
  static_assert(NumMmaThreads == 256 || NumMmaThreads == 384);
  if (cutlass::canonical_warp_group_idx() > 0) {
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads, 0);
  }
  if constexpr (NumMmaThreads == 384) {
    if (cutlass::canonical_warp_group_idx() > 1) {
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads, 1 /*id*/);
    }
  }
}
} // namespace tl
