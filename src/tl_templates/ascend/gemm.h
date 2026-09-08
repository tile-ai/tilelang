#pragma once

#include "kernel_operator.h"

/*!
 * \brief Ascend GEMM template for L1-scoped inputs with double-buffered L0.
 *
 * Encapsulates the sub-K pipeline: double-buffered L0A/L0B allocation,
 * L1->L0 DMA copies with SetFlag/WaitFlag synchronization, and mad calls.
 *
 * ## Event ID contract
 *
 * This template uses **M_MTE1 event IDs 0 and 1** internally for L0
 * double-buffer synchronization (sk & 1).  The caller MUST:
 *
 *   1. Pre-set these flags exactly once before the first call:
 *        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
 *        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
 *
 *   2. Drain them exactly once after the last call:
 *        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
 *        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
 *
 *   3. NOT use IDs 0/1 for any other M_MTE1 flags in the surrounding
 *      code.  If the caller also needs M_MTE1 flags for its own L1
 *      double-buffer (e.g. an explicit sub-K loop), use IDs >= 2.
 *
 * The flags are self-sustaining across consecutive calls — no per-call
 * re-initialization is needed.
 *
 * \tparam M           Tile M dimension
 * \tparam K           Full K dimension of the L1 tile
 * \tparam N           Tile N dimension
 * \tparam TILE_K_SUB  Sub-K chunk size that fits in L0A/L0B
 * \tparam TRANS_B     Whether B is transposed (must be true on dav-3510;
 *                     for NN matmul use dn2nz to transpose B during GM→L1)
 * \tparam InT         Input element type (bfloat16_t, half_t, etc.)
 * \tparam AccumT      Accumulator element type (deduced from cc_ptr)
 */
template <int M, int K, int N, int TILE_K_SUB, bool TRANS_B, typename InT,
          typename AccumT>
__aicore__ inline void
ascend_gemm_l1(__cc__ AccumT *cc_ptr, __cbuf__ InT *cbuf_a_ptr,
               __cbuf__ InT *cbuf_b_ptr, int clear_accum, int buf_offset = 0,
               int unit_flag_ctrl = 0) {
  // C0 = 32B / sizeof(InT): 16 for bf16/fp16, 8 for fp32, 32 for int8.
  // FRAC_N = 16: row-block size in the n-direction, dtype-independent.
  constexpr int C0 = 32 / sizeof(InT);
  constexpr int FRAC_N = 16;

  static_assert(K % TILE_K_SUB == 0, "K must be divisible by TILE_K_SUB");
  static_assert(TILE_K_SUB % C0 == 0, "TILE_K_SUB must be a multiple of C0");
  static_assert(M % FRAC_N == 0, "M must be a multiple of 16");
  static_assert(N % FRAC_N == 0, "N must be a multiple of 16");

  constexpr int SUB_K = K / TILE_K_SUB;
  constexpr int mStepA = M / FRAC_N;
  constexpr int mStepB = N / FRAC_N;
  constexpr int kStep = TILE_K_SUB / C0;
  constexpr int L0A_STAGE = M * TILE_K_SUB;
  constexpr int L0B_STAGE = N * TILE_K_SUB;

  __ca__ InT *l0a = (__ca__ InT *)0;
  __cb__ InT *l0b = (__cb__ InT *)0;

  for (int sk = 0; sk < SUB_K; ++sk) {
    get_buf(PIPE_MTE1, buf_offset + (sk & 1), false);
    load_cbuf_to_ca((__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)), cbuf_a_ptr, 0,
                    (sk * kStep), mStepA, kStep, mStepA, mStepA, 0);
    if constexpr (TRANS_B) {
      // NT case: W[N,K] in L1, cb transpose=false (ZZ)
      // rows=N, cols=K → sub-K along cols (kStart)
      // mStep=N/16 (nBlk), kStep=TILE_K_SUB/C0 (nGroups in K)
      load_cbuf_to_cb((__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), cbuf_b_ptr,
                      0, (sk * kStep), mStepB, kStep, mStepB, mStepB, 0);
    } else {
      // NN case: B[K,N] in L1, cb transpose=true (ZN).
      // ⚠️ transpose=true in load_cbuf_to_cb only works for b16 types;
      //    fp32 NN matmul MUST use the NT path (transpose during GM→L1).
      // rows=K, cols=N → sub-K along rows (mStart)
      constexpr int srcStrideNN = K / C0; // nBlk in L1
      load_cbuf_to_cb((__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), cbuf_b_ptr,
                      (sk * kStep), // mStartPosition: sub-K row offset
                      0,            // kStartPosition: start from col 0
                      kStep,        // mStep: TILE_K_SUB/C0 K-row fractals
                      mStepB,       // kStep: N/FRAC_N N-col fractals
                      srcStrideNN,  // srcStride: K/C0 (nBlk in L1)
                      kStep,        // dstStride: mStep
                      1);
    }
    rls_buf(PIPE_MTE1, buf_offset + (sk & 1), false);
    get_buf(PIPE_M, buf_offset + (sk & 1), false);

    bool is_first = (sk == 0);
    bool is_last = (sk == SUB_K - 1);
    uint32_t uf_ctrl = unit_flag_ctrl;
    if (uf_ctrl == 3 && !is_last) {
      uf_ctrl = 2;
    }

    mad(cc_ptr, (__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)),
        (__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), M, TILE_K_SUB, N, uf_ctrl,
        true, false, clear_accum && is_first);
    rls_buf(PIPE_M, buf_offset + (sk & 1), false);
  }
}

/*!
 * \brief Ascend block-scaled GEMM template for L1-scoped inputs with
 *        double-buffered L0 and scale factor pipeline.
 *
 * Data flow:
 *   1. L1→L0A/L0B  data:  load_cbuf_to_ca  / load_cbuf_to_cb
 *   2. L1→L0A/L0B  scale: load_cbuf_to_ca_mx / load_cbuf_to_cb_mx
 *   3. Compute:            mad_mx (reads data + scale from L0A/L0B)
 *
 * SF layout conventions:
 *   - kSFDivisor = 64:  one SF pair (int16_t = 2×float8_e8m0) per 64 K-elements
 *   - kSFPack    = 2:   two SF values packed in each int16_t
 *   - kSFAddrDiv = 16:  hardware divisor for L0A/L0B SF destination address
 *   - kAddrUnit  = 32:  L1 address unit for NZ stride parameters (bytes)
 *
 * \tparam M           Tile M dimension
 * \tparam K           Full K dimension of the L1 tile
 * \tparam N           Tile N dimension
 * \tparam TILE_K_SUB  Sub-K chunk size that fits in L0A/L0B
 * \tparam TRANS_B     Whether B is transposed
 * \tparam SF_NZ_STRIDE  NZ stride in SF L1 buffer (int16 element count in
 * \tparam InT         Input element type (float8_e4m3_t, etc.)
 * \tparam SFT         Scale factor L1 storage type (int16_t: 2× float8_e8m0
 *                     packed per element, one pair per 64 K-elements)
 * \tparam AccumT      Accumulator element type (deduced from cc_ptr)
 */
template <int M, int K, int N, int TILE_K_SUB, bool TRANS_B, int SF_NZ_STRIDE,
          typename InT, typename SFT, typename AccumT>
__aicore__ inline void ascend_blockscaled_gemm_l1(
    __cc__ AccumT *cc_ptr, __cbuf__ InT *cbuf_a_ptr, __cbuf__ InT *cbuf_b_ptr,
    __cbuf__ SFT *cbuf_sfa_ptr, __cbuf__ SFT *cbuf_sfb_ptr, int clear_accum,
    int buf_offset = 0, int sf_k_offset = 0, int unit_flag_ctrl = 0) {
  constexpr bool is_fp4 = std::is_same_v<InT, float4_e2m1x2_t>;
  constexpr int C0 = is_fp4 ? 64 : 32 / sizeof(InT);
  constexpr int FRAC_N = 16;

  static_assert(K % TILE_K_SUB == 0, "K must be divisible by TILE_K_SUB");
  static_assert(TILE_K_SUB % C0 == 0, "TILE_K_SUB must be a multiple of C0");
  static_assert(TILE_K_SUB % 64 == 0,
                "MXFP8 requires TILE_K_SUB to be a multiple of 64 "
                "(one SF pair covers 64 K-elements)");
  static_assert(M % FRAC_N == 0, "M must be a multiple of 16");
  static_assert(N % FRAC_N == 0, "N must be a multiple of 16");

  constexpr int SUB_K = K / TILE_K_SUB;
  constexpr int mStepA = M / FRAC_N;
  constexpr int mStepB = N / FRAC_N;
  constexpr int kStep = TILE_K_SUB / C0;

  constexpr int kPerSlot = is_fp4 ? 2 : 1;
  constexpr int L0A_STAGE = M * TILE_K_SUB / kPerSlot;
  constexpr int L0B_STAGE = N * TILE_K_SUB / kPerSlot;

  static constexpr int kSFDivisor = 64;
  static constexpr int kSFPack = 2;
  static constexpr int kSFAddrDiv = 16;

  static constexpr int kSFPairsPerInner = TILE_K_SUB / kSFDivisor;

  __ca__ InT *l0a = (__ca__ InT *)0;
  __cb__ InT *l0b = (__cb__ InT *)0;

  for (int sk = 0; sk < SUB_K; ++sk) {
    get_buf(PIPE_MTE1, buf_offset + (sk & 1), false);

    if constexpr (is_fp4) {
      load_cbuf_to_ca_s4((__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)),
                         cbuf_a_ptr, 0, (sk * kStep), mStepA, kStep, mStepA,
                         mStepA, 0);

      if constexpr (TRANS_B) {
        load_cbuf_to_cb_s4((__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)),
                           cbuf_b_ptr, 0, (sk * kStep), mStepB, kStep, mStepB,
                           mStepB, 0);
      } else {
        constexpr int srcStrideNN = K / C0;
        load_cbuf_to_cb_s4((__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)),
                           cbuf_b_ptr, (sk * kStep), 0, kStep, mStepB,
                           srcStrideNN, kStep, 1);
      }
    } else {
      load_cbuf_to_ca((__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)), cbuf_a_ptr,
                      0, (sk * kStep), mStepA, kStep, mStepA, mStepA, 0);

      if constexpr (TRANS_B) {
        load_cbuf_to_cb((__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)),
                        cbuf_b_ptr, 0, (sk * kStep), mStepB, kStep, mStepB,
                        mStepB, 0);
      } else {
        constexpr int srcStrideNN = K / C0;
        load_cbuf_to_cb((__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)),
                        cbuf_b_ptr, (sk * kStep), 0, kStep, mStepB, srcStrideNN,
                        kStep, 1);
      }
    }

    {
      uint64_t dst_a =
          (uint64_t)(uintptr_t)((__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE))) /
          kSFAddrDiv;
      uint64_t dst_b =
          (uint64_t)(uintptr_t)((__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE))) /
          kSFAddrDiv;

      uint16_t sf_y = sf_k_offset + sk * kSFPairsPerInner;

      load_cbuf_to_ca_mx(dst_a, (__cbuf__ fp8_e4_t *)cbuf_sfa_ptr, 0, sf_y,
                         mStepA, kSFPairsPerInner, SF_NZ_STRIDE,
                         kSFPairsPerInner);

      load_cbuf_to_cb_mx(dst_b, (__cbuf__ fp8_e4_t *)cbuf_sfb_ptr, 0, sf_y,
                         mStepB, kSFPairsPerInner, SF_NZ_STRIDE,
                         kSFPairsPerInner);
    }

    rls_buf(PIPE_MTE1, buf_offset + (sk & 1), false);

    get_buf(PIPE_M, buf_offset + (sk & 1), false);

    bool is_first = (sk == 0);
    bool is_last = (sk == SUB_K - 1);
    uint32_t uf_ctrl = unit_flag_ctrl;
    if (uf_ctrl == 3 && !is_last) {
      uf_ctrl = 2;
    }

    mad_mx(cc_ptr, (__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)),
           (__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), M, TILE_K_SUB, N,
           uf_ctrl, true, false, clear_accum && is_first);

    rls_buf(PIPE_M, buf_offset + (sk & 1), false);
  }
}
