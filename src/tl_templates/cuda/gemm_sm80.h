#pragma once

#include "gemm_mma.h"

namespace tl {

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

} // namespace tl
