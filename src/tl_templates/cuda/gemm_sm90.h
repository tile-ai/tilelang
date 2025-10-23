#pragma once

#include "common.h"
#include "gemm_mma.h"
#include "intrin.h"

#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>

namespace cute {

using namespace SM90;

namespace tl_wgmma {

using namespace cutlass::gemm::collective::detail; // ss_smem_selector

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
public:
  using A_type_cute = typename tl::to_cute_type<A_type_raw>::type;
  using B_type_cute = typename tl::to_cute_type<B_type_raw>::type;
  using A_type = conditional_t<std::is_same<A_type_cute, float>::value,
                               tfloat32_t, A_type_cute>;
  using B_type = conditional_t<std::is_same<B_type_cute, float>::value,
                               tfloat32_t, A_type_cute>;
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
