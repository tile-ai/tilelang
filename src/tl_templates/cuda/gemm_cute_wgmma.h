// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_atom.hpp>

#include "common.h"

namespace tl_cute_wgmma { // main body

using namespace cute;
using namespace cute::SM90;

template <GMMA::Major major, class ElementType, class BLK_MN, class BLK_K>
CUTE_HOST_DEVICE constexpr auto ss_smem_selector() {
  auto BLK_MN0 = size<0>(BLK_MN{});
  auto BLK_K0 = size<0>(BLK_K{});

  static_assert(BLK_MN0 % 8 == 0, "BLK_MN0 must be a multiple of 8.");
  static_assert(BLK_K0 % 8 == 0, "BLK_K0 must be a multiple of 8.");

  if constexpr (major == GMMA::Major::MN) {
    if constexpr (BLK_MN0 %
                      size<0>(GMMA::Layout_MN_SW128_Atom<ElementType>{}) ==
                  0) {
      return GMMA::Layout_MN_SW128_Atom<ElementType>{};
    } else if constexpr (BLK_MN0 %
                             size<0>(
                                 GMMA::Layout_MN_SW64_Atom<ElementType>{}) ==
                         0) {
      return GMMA::Layout_MN_SW64_Atom<ElementType>{};
    } else if constexpr (BLK_MN0 %
                             size<0>(
                                 GMMA::Layout_MN_SW32_Atom<ElementType>{}) ==
                         0) {
      return GMMA::Layout_MN_SW32_Atom<ElementType>{};
    } else if constexpr (BLK_MN0 %
                             size<0>(
                                 GMMA::Layout_MN_INTER_Atom<ElementType>{}) ==
                         0) {
      return GMMA::Layout_MN_INTER_Atom<ElementType>{};
    } else {
      static_assert(
          BLK_MN0 % size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0,
          "BLK_MN0 must be a multiple of "
          "size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{})");
    }
  } else if constexpr (major == GMMA::Major::K) {
    if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW128_Atom<ElementType>{}) ==
                  0) {
      return GMMA::Layout_K_SW128_Atom<ElementType>{};
    } else if constexpr (BLK_K0 %
                             size<1>(GMMA::Layout_K_SW64_Atom<ElementType>{}) ==
                         0) {
      return GMMA::Layout_K_SW64_Atom<ElementType>{};
    } else if constexpr (BLK_K0 %
                             size<1>(GMMA::Layout_K_SW32_Atom<ElementType>{}) ==
                         0) {
      return GMMA::Layout_K_SW32_Atom<ElementType>{};
    } else if constexpr (BLK_K0 %
                             size<1>(
                                 GMMA::Layout_K_INTER_Atom<ElementType>{}) ==
                         0) {
      return GMMA::Layout_K_INTER_Atom<ElementType>{};
    } else {
      static_assert(
          BLK_K0 % size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{}) == 0,
          "BLK_K0 must be a multiple of "
          "size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{})");
    }
  }
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool is_rs, typename A_type_raw, typename B_type_raw,
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

  static constexpr auto GmmaMajorA = trans_A ? GMMA::Major::MN : GMMA::Major::K;
  static constexpr auto GmmaMajorB = trans_B ? GMMA::Major::K : GMMA::Major::MN;

  using TileShape = Shape<Int<M>, Int<N / num_warp_n>, Int<K>>;
  using ThrLayout = Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>;
  using MMA = typename std::conditional<
      is_rs,
      decltype(make_tiled_mma(
          GMMA::rs_op_selector<A_type, B_type, C_type, TileShape, GmmaMajorA,
                               GmmaMajorB>(),
          ThrLayout{})),
      decltype(make_tiled_mma(
          GMMA::ss_op_selector<A_type, B_type, C_type, TileShape, GmmaMajorA,
                               GmmaMajorB>(),
          ThrLayout{}))>::type;

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
  using RegLayoutA =
      decltype(partition_shape_A(MMA{}, Shape<Int<M>, Int<K>>{}));
  using RegLayoutC =
      decltype(partition_shape_C(MMA{}, Shape<Int<M>, Int<N>>{}));

  static_assert(num_warp_m % 4 == 0, "num_warp_m must be a multiple of 4");
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
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);

  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  warpgroup_fence_operand(rAcc);
  warpgroup_arrive();
  CUTE_UNROLL
  for (int k = 0; k < size<2>(tCrA); ++k) {
    gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), rAcc);
  }
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(rAcc);
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
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  warpgroup_fence_operand(rA);
  warpgroup_fence_operand(rAcc);
  warpgroup_arrive();
  CUTE_UNROLL
  for (int k = 0; k < size<2>(rA); ++k) {
    gemm(tiled_mma, rA(_, _, k), tCrB(_, _, k), rAcc);
  }
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(rAcc);
  warpgroup_fence_operand(rA);
}

} // namespace tl_cute_wgmma

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool from_reg_A, bool from_reg_B, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
TL_DEVICE void gemm_cute_wgmma(A_type_raw *pA, B_type_raw *pB,
                               C_type_raw *accum) {
  using GemmOp =
      typename tl_cute_wgmma::GemmOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                     trans_B, from_reg_A, A_type_raw,
                                     B_type_raw, C_type_raw>;
  if constexpr (!from_reg_A && !from_reg_B) {
    tl_cute_wgmma::gemm_ss<GemmOp>(pA, pB, accum);
  }
  if constexpr (from_reg_A && !from_reg_B) {
    tl_cute_wgmma::gemm_rs<GemmOp>(pA, pB, accum);
  }
  static_assert((!from_reg_A && !from_reg_B) || (from_reg_A && !from_reg_B),
                "Not supported");
}

} // namespace tl
