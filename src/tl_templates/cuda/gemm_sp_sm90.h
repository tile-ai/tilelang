#pragma once

#include <cute/arch/mma_sm90_gmma_sparse.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/gemm/collective/builders/sm90_sparse_config.inl>

namespace cute {
namespace tl_wgmma_sp {
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
public:
  static_assert(num_warp_m % 4 == 0, "num_warp_m must be a multiple of 4");

  using A_type_cute = typename tl::to_cute_type<A_type_raw>::type;
  using B_type_cute = typename tl::to_cute_type<B_type_raw>::type;
  using A_type = conditional_t<std::is_same<A_type_cute, float>::value,
                               tfloat32_t, A_type_cute>;
  using B_type = conditional_t<std::is_same<B_type_cute, float>::value,
                               tfloat32_t, B_type_cute>;
  using C_type = C_type_raw;

  static constexpr bool need_tfloat32_cast =
      std::is_same<A_type_raw, float>::value &&
      std::is_same<B_type_raw, float>::value;

  static constexpr GMMA::Major GmmaMajorA =
      trans_A ? GMMA::Major::MN : GMMA::Major::K;
  static constexpr GMMA::Major GmmaMajorB =
      trans_B ? GMMA::Major::K : GMMA::Major::MN;

  using TiledMma = decltype(make_tiled_mma(
      GMMA::ss_op_selector_sparse<
          A_type, B_type, C_type,
          Shape<Int<M / (num_warp_m / 4)>, Int<N / num_warp_n>, Int<K>>,
          GmmaMajorA, GmmaMajorB>(),
      Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>{}));

  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementAMmaSparsity = Int<ElementAMma::sparsity>;
  using ElementBMma = typename TiledMma::ValTypeB;
  using ElementEMma = typename TiledMma::ValTypeE;
  using ElementEMmaSparsity = Int<ElementEMma::sparsity>;
  using E_type_raw = typename ElementEMma::raw_type;

  using SparseConfig =
      cutlass::Sm90GemmSparseConfig<ElementAMma, GmmaMajorA, ElementEMma,
                                    decltype(min(Int<K>{}, _128{}))>;

  using LayoutA = decltype(SparseConfig::deduce_layoutA());
  using LayoutE = decltype(SparseConfig::deduce_layoutE());

  using SmemLayoutAtomA =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector_sparse<
               GmmaMajorA, A_type, Int<M>, Int<K>, ElementAMmaSparsity>());
  using SmemLayoutAtomB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GmmaMajorB, B_type, Int<N>, Int<K>>());

  using SmemLayoutAtomE_ = typename SparseConfig::TensorEAtom;
  using SmemLayoutAtomE =
      ComposedLayout<Swizzle<0, 4, 3>,
                     smem_sparse_ptr_flag_bits<ElementEMmaSparsity::value,
                                               sizeof_bits_v<E_type_raw>>,
                     SmemLayoutAtomE_>;

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{}, Shape<Int<M>, Int<K>>{},
      conditional_t<trans_A, Step<_2, _1>, Step<_1, _2>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{}, Shape<Int<N>, Int<K>>{},
      conditional_t<trans_B, Step<_1, _2>, Step<_2, _1>>{}));
  using SmemLayoutE = decltype(tile_to_shape(
      SmemLayoutAtomE{}, Shape<Int<M>, Int<K>>{},
      conditional_t<trans_A, Step<_2, _1>, Step<_1, _2>>{}));

  using SmemCopyAtomE = AutoVectorizingCopy;

  template <int wg_wait = 0>
  static CUTE_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC,
                               E_type_raw *pE) {
    const int tid = threadIdx.x;
    Tensor sA =
        make_tensor(make_smem_ptr(recast_ptr<ElementAMma>(pA)), SmemLayoutA{});
    Tensor sB =
        make_tensor(make_smem_ptr(recast_ptr<ElementBMma>(pB)), SmemLayoutB{});
    Tensor sE = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(recast_ptr<ElementEMma>(pE)), SmemLayoutE{}));

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCsE = partition_E(thr_mma, sE(_, _));

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrE = make_fragment_like<ElementEMma>(tCsE);

    auto copy_atom_E = Copy_Atom<SmemCopyAtomE, uint32_t>{};
    auto smem_tiled_copy_E = make_tiled_copy_E(copy_atom_E, tiled_mma);
    auto smem_thr_copy_E = smem_tiled_copy_E.get_thread_slice(tid);
    Tensor tEsE = smem_thr_copy_E.partition_S(sE);
    Tensor tErE = smem_thr_copy_E.retile_D(tCrE);

    Tensor acc =
        make_tensor(make_rmem_ptr(pC),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    if constexpr (clear_accum) {
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    }
    copy(smem_tiled_copy_E, tEsE, tErE);

    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // warpgroup_arrive();
      // (V,M) x (V,N) => (V,M,N)
      gemm(tiled_mma, make_zip_tensor(tCrA(_, _, k_block), tCrE(_, _, k_block)),
           tCrB(_, _, k_block), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }

    warpgroup_commit_batch();
    if constexpr (wg_wait >= 0) {
      warpgroup_wait<wg_wait>();
    }
    warpgroup_fence_operand(acc);
  }

  template <class MMA_Atom, class AtomLayoutMNK, class PermutationMNK,
            class ETensor>
  CUTE_HOST_DEVICE static constexpr auto
  thrfrg_E(TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK> const &mma,
           ETensor &&etensor) {
    using TiledMma = TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK>;

    CUTE_STATIC_ASSERT_V(rank(etensor) >= Int<2>{});

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(PermutationMNK{}), get<2>(PermutationMNK{}));
    auto t_tensor = logical_divide(etensor, t_tile); // (PermM,PermK)

    // Tile the tensor for the Atom
    auto e_tile =
        make_tile(make_layout(size<0>(typename TiledMma::AtomShape_MNK{})),
                  make_layout(size<2>(typename TiledMma::AtomShape_MNK{})));
    auto e_tensor =
        zipped_divide(t_tensor, e_tile); // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    using AtomLayoutE_TV = typename TiledMma::Atom::Traits::ELayout;
    auto tv_tensor =
        e_tensor.compose(AtomLayoutE_TV{}, _); // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile =
        make_tile(_, make_tile(make_layout(size<1>(mma.thr_layout_vmnk_)),
                               make_layout(size<3>(mma.thr_layout_vmnk_))));
    auto thr_tensor = zipped_divide(
        tv_tensor, thr_tile); // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  template <class... MArgs>
  CUTE_HOST_DEVICE static constexpr auto
  get_layoutE_TV(TiledMMA<MArgs...> const &mma) {
    // (M,K) -> (M,K)
    auto ref_E = make_layout(make_shape(tile_size<0>(mma), tile_size<2>(mma)));
    // (ethrid,val) -> (M,K)
    auto layoutE_TV = thrfrg_E(mma, ref_E);

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto etile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(mma.thr_layout_vmnk_),
                                            size<2>(mma.thr_layout_vmnk_)),
                                 make_stride(Int<1>{}, Int<0>{})),
                     _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(mma.thr_layout_vmnk_);

    // (thr_idx,val) -> (M,K)
    return layoutE_TV.compose(etile, _).compose(thridx_2_thrid, _);
  }

  template <class... MArgs, class ETensor>
  CUTE_HOST_DEVICE static constexpr auto
  partition_E(ThrMMA<MArgs...> const &thr_mma, ETensor &&etensor) {
    auto thr_tensor = make_tensor(static_cast<ETensor &&>(etensor).data(),
                                  thrfrg_E(thr_mma, etensor.layout()));

    auto thr_vmk = make_coord(
        get<0>(thr_mma.thr_vmnk_),
        make_coord(get<1>(thr_mma.thr_vmnk_), get<3>(thr_mma.thr_vmnk_)));
    return thr_tensor(thr_vmk,
                      make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  }

  template <class... CArgs, class... MArgs>
  CUTE_HOST_DEVICE static constexpr auto
  make_tiled_copy_E(Copy_Atom<CArgs...> const &copy_atom,
                    TiledMMA<MArgs...> const &mma) {
    return make_tiled_copy_impl(
        copy_atom, get_layoutE_TV(mma),
        make_shape(tile_size<0>(mma), tile_size<2>(mma)));
  }
};

} // namespace tl_wgmma_sp
} // namespace cute

namespace tl {
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum = false, bool use_wgmma = true,
          int wg_wait = 0, typename A_type, typename B_type, typename C_type,
          typename GMMA = cute::tl_wgmma_sp::GemmTensorOp<
              M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, clear_accum,
              A_type, B_type, C_type>,
          typename E_type = typename GMMA::ElementEMma::raw_type>
TL_DEVICE void gemm_sp_ss(A_type *pA, B_type *pB, C_type *accum, E_type *pE) {
  static_assert(use_wgmma, "only wgmma is supported for now");
  if constexpr (use_wgmma) {
    GMMA::body<wg_wait>(pA, pB, accum, pE);
  } else {
    CUTE_GCC_UNREACHABLE;
  }
}
} // namespace tl
