// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <cute/arch/mma_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>

#include "common.h"

namespace tl_cute { // DispatchInstruction

using namespace cute;

template <typename A_type, typename B_type, typename C_type, int num_warp_m,
          int num_warp_n>
struct DispatchInstruction;

using _X = Underscore;

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 800))
template <int num_warp_m, int num_warp_n>
struct DispatchInstruction<half_t, half_t, half_t, num_warp_m, num_warp_n> {
  using Op = SM80_16x8x16_F16F16F16F16_TN;
  using Permutations = Tile<_X, Int<num_warp_n * 16>, _X>;
};
template <int num_warp_m, int num_warp_n>
struct DispatchInstruction<half_t, half_t, float, num_warp_m, num_warp_n> {
  using Op = SM80_16x8x16_F32F16F16F32_TN;
  using Permutations = Tile<_X, Int<num_warp_n * 16>, _X>;
};
template <int num_warp_m, int num_warp_n>
struct DispatchInstruction<bfloat16_t, bfloat16_t, float, num_warp_m,
                           num_warp_n> {
  using Op = SM80_16x8x16_F32BF16BF16F32_TN;
  using Permutations = Tile<_X, Int<num_warp_n * 16>, _X>;
};
template <int num_warp_m, int num_warp_n>
struct DispatchInstruction<tfloat32_t, tfloat32_t, float, num_warp_m,
                           num_warp_n> {
  using Op = SM80_16x8x8_F32TF32TF32F32_TN;
  using Permutations = Tile<_X, Int<num_warp_n * 16>, _X>;
};
template <int num_warp_m, int num_warp_n>
struct DispatchInstruction<int8_t, int8_t, int, num_warp_m, num_warp_n> {
  using Op = SM80_16x8x32_S32S8S8S32_TN;
  using Permutations = Tile<_X, Int<num_warp_n * 16>, _X>;
};
template <int num_warp_m, int num_warp_n>
struct DispatchInstruction<double, double, double, num_warp_m, num_warp_n> {
  using Op = SM80_8x8x4_F64F64F64F64_TN;
  using Permutations = Tile<Int<num_warp_m * 16>, Int<num_warp_n * 16>, _X>;
};
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 750))
template <int num_warp_m, int num_warp_n>
struct DispatchInstruction<half_t, half_t, float, num_warp_m, num_warp_n> {
  using Op = SM75_16x8x8_F32F16F16F32_TN;
  using Permutations = Tile<_X, Int<num_warp_n * 16>, _16>;
};
#endif

} // namespace tl_cute

namespace tl_cute { // OperandTraits

using namespace cute;

template <int Bits, int N, int K, bool K_inner, typename Enable = void>
struct OperandTraits {
  // Primary template, use padded layout and default copy
  static constexpr int stride = K_inner ? K : N;
  static constexpr int padded =
      stride % (256 / Bits) == 0 ? stride + 128 / Bits : stride;
  using Layout = typename std::conditional<
      K_inner, Layout<Shape<Int<N>, Int<K>>, Shape<Int<padded>, _1>>,
      Layout<Shape<Int<N>, Int<K>>, Shape<_1, Int<padded>>>>::type;
  using CopyOp = DefaultCopy;
};

template <int N, int K>
struct OperandTraits<16, N, K, true,
                     typename std::enable_if<K % 64 == 32>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using CopyOp = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<16, N, K, true,
                     typename std::enable_if<K % 64 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using CopyOp = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<16, N, K, false,
                     typename std::enable_if<N % 64 == 32>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 3, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{},
                                        Step<_2, _1>{}));
  using CopyOp = SM75_U16x8_LDSM_T;
};

template <int N, int K>
struct OperandTraits<16, N, K, false,
                     typename std::enable_if<N % 64 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{},
                                        Step<_2, _1>{}));
  using CopyOp = SM75_U16x8_LDSM_T;
};

template <int N, int K>
struct OperandTraits<32, N, K, true,
                     typename std::enable_if<K % 32 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 2, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using CopyOp = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<32, N, K, true,
                     typename std::enable_if<K % 32 == 16>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 2, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using CopyOp = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<32, N, K, false,
                     typename std::enable_if<N % 32 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 2, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{},
                                        Step<_2, _1>{}));
  using CopyOp = UniversalCopy<tfloat32_t>;
};

template <int N, int K>
struct OperandTraits<32, N, K, false,
                     typename std::enable_if<N % 32 == 16>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 2, 3>{}, Layout<Shape<_16, _8>, Stride<_1, _16>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{},
                                        Step<_2, _1>{}));
  using CopyOp = UniversalCopy<tfloat32_t>;
};

template <int N, int K>
struct OperandTraits<8, N, K, true,
                     typename std::enable_if<K % 128 == 64>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 4, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using CopyOp = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<8, N, K, true,
                     typename std::enable_if<K % 128 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 4, 3>{}, Layout<Shape<_8, _128>, Stride<_128, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using CopyOp = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<64, N, K, true,
                     typename std::enable_if<K % 16 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 0, 4>{}, Layout<Shape<_4, _16>, Stride<_16, _1>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
  using CopyOp = DefaultCopy;
};

template <int N, int K>
struct OperandTraits<64, N, K, false,
                     typename std::enable_if<N % 16 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 2, 2>{}, Layout<Shape<_16, _4>, Stride<_1, _16>>{}));
  using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{},
                                        Step<_2, _1>{}));
  using CopyOp = DefaultCopy;
};

} // namespace tl_cute

namespace tl_cute { // main body

using namespace cute;

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, typename A_type_raw, typename B_type_raw,
          typename C_type_raw>
class GemmOp {
public:
  using A_type =
      typename std::conditional<std::is_same<A_type_raw, float>::value,
                                tfloat32_t, A_type_raw>::type;
  using B_type =
      typename std::conditional<std::is_same<B_type_raw, float>::value,
                                tfloat32_t, B_type_raw>::type;
  using C_type = C_type_raw;

  using Instruction =
      DispatchInstruction<A_type, B_type, C_type, num_warp_m, num_warp_n>;

  // only use TN MMA Operations
  using MMA = TiledMMA<MMA_Atom<typename Instruction::Op>,
                       Layout<Shape<Int<num_warp_m>, Int<num_warp_n>, _1>>,
                       typename Instruction::Permutations>;

  using OperandATraits =
      OperandTraits<sizeof_bits<A_type>::value, M, K, !trans_A>;
  using OperandBTraits =
      OperandTraits<sizeof_bits<B_type>::value, N, K, trans_B>;

  using SmemLayoutA = typename OperandATraits::Layout;
  using SmemLayoutB = typename OperandBTraits::Layout;
  using RegLayoutA =
      decltype(partition_shape_A(MMA{}, Shape<Int<M>, Int<K>>{}));
  using RegLayoutB =
      decltype(partition_shape_B(MMA{}, Shape<Int<N>, Int<K>>{}));
  using RegLayoutC =
      decltype(partition_shape_C(MMA{}, Shape<Int<M>, Int<N>>{}));

  using TiledCopyA = decltype(make_tiled_copy_A(
      Copy_Atom<typename OperandATraits::CopyOp, A_type>{}, MMA{}));
  using TiledCopyB = decltype(make_tiled_copy_B(
      Copy_Atom<typename OperandBTraits::CopyOp, B_type>{}, MMA{}));

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
};

template <typename GemmOp, typename A_type_raw, typename B_type_raw,
          typename C_type_raw>
TL_DEVICE void gemm_ss(A_type_raw *pA, B_type_raw *pB, C_type_raw *accum) {
  const int tid = threadIdx.x;

  Tensor sA = make_tensor(
      make_smem_ptr(reinterpret_cast<typename GemmOp::A_type *>(pA)),
      typename GemmOp::SmemLayoutA{});
  Tensor sB = make_tensor(
      make_smem_ptr(reinterpret_cast<typename GemmOp::B_type *>(pB)),
      typename GemmOp::SmemLayoutB{});
  Tensor rAcc = make_tensor(
      make_rmem_ptr(reinterpret_cast<typename GemmOp::C_type *>(accum)),
      typename GemmOp::RegLayoutC{});

  typename GemmOp::MMA tiled_mma;
  typename GemmOp::TiledCopyA tiled_copyA;
  typename GemmOp::TiledCopyB tiled_copyB;

  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto thr_copy_A = tiled_copyA.get_thread_slice(tid);
  auto thr_copy_B = tiled_copyB.get_thread_slice(tid);

  Tensor tCrA = thr_mma.partition_fragment_A(sA);
  Tensor tCsA = thr_copy_A.partition_S(sA);
  Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);

  Tensor tCrB = thr_mma.partition_fragment_B(sB);
  Tensor tCsB = thr_copy_B.partition_S(sB);
  Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

  // when layout is KxN and n_warp is 1, there seem to be a bug, use this as a
  // workaround
  auto tCrA_mma_view =
      make_tensor(tCrA.data(), GemmOp::remove_swizzle(tCrA.layout()));
  auto tCrB_mma_view =
      make_tensor(tCrB.data(), GemmOp::remove_swizzle(tCrB.layout()));

  CUTE_UNROLL
  for (int k = 0; k < size<2>(tCrA); ++k) {
    copy(tiled_copyA, tCsA(_, _, k), tCrA_copy_view(_, _, k));
    copy(tiled_copyB, tCsB(_, _, k), tCrB_copy_view(_, _, k));
    gemm(tiled_mma, tCrA_mma_view(_, _, k), tCrB_mma_view(_, _, k), rAcc);
  }
}

template <typename GemmOp, typename A_type_raw, typename B_type_raw,
          typename C_type_raw>
TL_DEVICE void gemm_rs(A_type_raw *pA, B_type_raw *pB, C_type_raw *accum) {
  const int tid = threadIdx.x;

  Tensor rA = make_tensor(
      make_rmem_ptr(reinterpret_cast<typename GemmOp::A_type *>(pA)),
      typename GemmOp::RegLayoutA{});
  Tensor sB = make_tensor(
      make_smem_ptr(reinterpret_cast<typename GemmOp::B_type *>(pB)),
      typename GemmOp::SmemLayoutB{});
  Tensor rAcc = make_tensor(
      make_rmem_ptr(reinterpret_cast<typename GemmOp::C_type *>(accum)),
      typename GemmOp::RegLayoutC{});

  typename GemmOp::MMA tiled_mma;
  typename GemmOp::TiledCopyB tiled_copyB;

  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto thr_copy_B = tiled_copyB.get_thread_slice(tid);

  Tensor tCrB = thr_mma.partition_fragment_B(sB);
  Tensor tCsB = thr_copy_B.partition_S(sB);
  Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

  auto tCrB_mma_view =
      make_tensor(tCrB.data(), GemmOp::remove_swizzle(tCrB.layout()));

  copy(tiled_copyB, tCsB(_, _, 0), tCrB_copy_view(_, _, 0));
  CUTE_UNROLL
  for (int k = 0; k < size<2>(rA); ++k) {
    if (k < size<2>(rA) - 1) {
      copy(tiled_copyB, tCsB(_, _, k + 1), tCrB_copy_view(_, _, k + 1));
    }
    gemm(tiled_mma, rA(_, _, k), tCrB_mma_view(_, _, k), rAcc);
  }
}

template <typename GemmOp, typename A_type_raw, typename B_type_raw,
          typename C_type_raw>
TL_DEVICE void gemm_sr(A_type_raw *pA, B_type_raw *pB, C_type_raw *accum) {
  const int tid = threadIdx.x;

  Tensor sA = make_tensor(
      make_smem_ptr(reinterpret_cast<typename GemmOp::A_type *>(pA)),
      typename GemmOp::SmemLayoutA{});
  Tensor rB = make_tensor(
      make_rmem_ptr(reinterpret_cast<typename GemmOp::B_type *>(pB)),
      typename GemmOp::RegLayoutB{});
  Tensor rAcc = make_tensor(
      make_rmem_ptr(reinterpret_cast<typename GemmOp::C_type *>(accum)),
      typename GemmOp::RegLayoutC{});

  typename GemmOp::MMA tiled_mma;
  typename GemmOp::TiledCopyA tiled_copyA;

  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto thr_copy_A = tiled_copyA.get_thread_slice(tid);

  Tensor tCrA = thr_mma.partition_fragment_A(sA);
  Tensor tCsA = thr_copy_A.partition_S(sA);
  Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);

  auto tCrA_mma_view =
      make_tensor(tCrA.data(), GemmOp::remove_swizzle(tCrA.layout()));

  copy(tiled_copyA, tCsA(_, _, 0), tCrA_copy_view(_, _, 0));
  CUTE_UNROLL
  for (int k = 0; k < size<2>(tCrA); ++k) {
    if (k < size<2>(tCrA) - 1) {
      copy(tiled_copyA, tCsA(_, _, k + 1), tCrA_copy_view(_, _, k + 1));
    }
    gemm(tiled_mma, tCrA_mma_view(_, _, k), rB(_, _, k), rAcc);
  }
}

} // namespace tl_cute

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool from_reg_A, bool from_reg_B, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
TL_DEVICE void gemm_cute(A_type_raw *pA, B_type_raw *pB, C_type_raw *accum) {
  using GemmOp =
      typename tl_cute::GemmOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                               trans_B, A_type_raw, B_type_raw, C_type_raw>;
  if (!from_reg_A && !from_reg_B) {
    tl_cute::gemm_ss<GemmOp>(pA, pB, accum);
  }
  if (from_reg_A && !from_reg_B) {
    tl_cute::gemm_rs<GemmOp>(pA, pB, accum);
  }
  if (!from_reg_A && from_reg_B) {
    tl_cute::gemm_sr<GemmOp>(pA, pB, accum);
  }
  static_assert(!(from_reg_A && from_reg_B), "Not supported");
}

} // namespace tl
