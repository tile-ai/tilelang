"""Legacy non-split TileLang port of avo/kernels/attention_kernel_1sm.cu.

This is the first slice toward full parity with the FA4-style 1SM kernels in
~/avo/kernels/{attention_kernel_1sm.cu, attention_kernel_1sm_d256.cu}. It is
intentionally architecturally simpler than the reference .cu and uses only
existing tilelang primitives.

What this slice keeps from the reference .cu:
  * Warp-specialized (producer / MMA / math) layout, distinct mbarriers per
    pipeline edge, no synchronous T.gemm.
  * Single-buffered S/P/D TMEM accumulators (TMEM cols are precious).
  * Multi-stage K/V SMEM ring buffers gated by per-stage barriers.
  * tcgen05_gemm for both Q@K^T (ss) and P@V (ts) GEMMs, matching the .cu.

What this slice intentionally diverges from the .cu (to be fixed by later
slices, in order):
  Slice 2 (P0 __maxnreg__): no register donation here. ptxas picks a single
    static count for all warps; spills expected, especially for D=256.
  Slice 3 (P1 TMEM col-slice + lane-0 arrive): the running O accumulator
    lives in fragment registers in this slice (not TMEM). With the column
    slice landed, O will move into D_tmem and the math WG will read/multiply/
    write back in 16-col chunks (FA4 "chunked rescale").
  Slice 4 (P2 TMEM aliasing + warp_ballot): no kQStages=2 interleaving. With
    aliasing, S+P will overlap and we can fit two Q-tiles in TMEM concurrently.
  Slice 5 (P3 max3 + FTZ + exp2 poly): scalar T.max / T.exp2 today; replaced
    with PTX-explicit packed variants later.

Expected Slice 1 perf vs reference .cu (which hits ~900 TFLOPS on H100/B100):
  ~500-600 TFLOPS — softmax warps spill, no chunked rescale, single Q-stage.
"""

import argparse
from typing import Optional

import torch
import tilelang
import tilelang.layout
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Pass config matches the SM100 wasp examples already in this directory.
# TL_DISABLE_WARP_SPECIALIZED=False lets tilelang auto-detect the if-on-tx
# branches and emit per-role setmaxnreg (currently runtime PTX only; Slice 2
# adds the __maxnreg__ launch attribute and JIT setter).
# --------------------------------------------------------------------------- #
PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    "tl.disable_thread_storage_sync": True,
    # Outline each warp-role branch of the kq_stages=2 for-loop body into a
    # separate __device__ __noinline__ function. Each device fn gets its own
    # register-allocation budget from ptxas; without this, all 384 threads
    # share one register pool. (kq_stages=1 also outlines via the same
    # codegen path; it currently runs only one branch per role.)
    "tl.outline_warp_spec_branches": True,
}


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG, target="cuda -arch=sm_100")
def attention_kernel_1sm(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    num_kv_heads: Optional[int] = None,
    is_causal: bool = False,
    block_M: int = 128,
    block_N: int = 128,
    kv_stages: int = 2,
    threads: int = 256,
    p_storage: str = "auto",
    kq_stages: int = 1,
    split_correction: bool = False,
    mma_dsl: bool = True,
):
    """Build the attention prim_func.

    Role map at threads=256 (Slice 4a v2: 3-role WS, O in TMEM):

        tx in [  0,  32):  producer  -- TMA loads K[k], V[k]
        tx in [ 32,  64):  mma issue -- tcgen05_gemm Q@K^T -> S, P@V -> O
        tx in [128, 256):  math WG   -- reads S_tmem, online softmax, writes
                                         P_tmem; reads O_tmem, rescales by
                                         scores_scale, writes back; at loop
                                         end, normalizes O by 1/logsum and
                                         TMA-stores Output.

    When split_correction=True and kq_stages=2, the dual-Q path uses a
    separate correction WG. The math WGs double-buffer scores_scale in SMEM
    and mbarriers so the correction WG never consumes the previous KV tile's
    scale signal with the current tile's scale address.
    """
    if num_kv_heads is None:
        num_kv_heads = heads
    if heads % num_kv_heads != 0:
        raise ValueError(f"heads={heads} must be divisible by num_kv_heads={num_kv_heads}")
    groups = heads // num_kv_heads

    # Where to stage the bf16-cast P matrix between softmax and P@V.
    #   "tmem"   : P_tmem path — fast, requires dim == block_N (else the
    #              tcgen05 atom validation in tcgen05_macro_generator.py:482
    #              rejects the shape — same root cause as Slice 3 P1).
    #   "shared" : P_shared path — works for any dim. Costs an extra round-
    #              trip through SMEM.
    #   "auto"   : "tmem" when dim == block_N, else "shared".
    if p_storage == "auto":
        p_storage = "tmem" if dim == block_N else "shared"
    if p_storage not in ("tmem", "shared"):
        raise ValueError(f"p_storage must be 'auto'|'tmem'|'shared', got {p_storage!r}")
    p_in_tmem = (p_storage == "tmem")

    # softmax scale baked with log2(e) so exp2 can be used directly
    scale = (1.0 / dim) ** 0.5 * 1.44269504

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, num_kv_heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    # The fa4 pattern uses pairs of SMEM buffers for K/V (not a 3D ring
    # buffer). Hard-code that here too — kv_stages above is kept as a kwarg
    # for backward compat but has no effect on the new shape.
    num_stages = 2

    if kq_stages == 1:
        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(seq_len, block_M),
                heads,
                batch,
                threads=threads,
            ) as (bx, by, bz):
                # ---------- SMEM ----------
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                # Paired K/V SMEM buffers (fa4 pattern). The mma_corr_load WG
                # alternates between them per iter based on stage_id = k % 2.
                K_shared_0 = T.alloc_shared([block_N, dim], dtype)
                K_shared_1 = T.alloc_shared([block_N, dim], dtype)
                V_shared_0 = T.alloc_shared([block_N, dim], dtype)
                V_shared_1 = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                scale_shared = T.alloc_shared([block_M], accum_dtype)
                logsum_shared = T.alloc_shared([block_M], accum_dtype)

                # ---------- TMEM ----------
                # S (fp32 [block_M, block_N]) needs block_N cols, but P (bf16
                # [block_M, block_N]) only needs block_N/2 cols (packed 2-per-col).
                # P can alias S's upper half: P[i, j] lives in the SAME column as
                # S[i, j + block_N/2]'s high 16 bits. S is dead before P is
                # written so the lifetime is safe.
                S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
                O_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
                if p_in_tmem:
                    P_tmem = T.alloc_tmem(
                        [block_M, block_N], dtype,
                        alias=S_tmem, col_offset=block_N // 2,
                    )
                else:
                    P_shared = T.alloc_shared([block_M, block_N], dtype)

                # ---------- Mbarriers ----------
                # fa4 pattern: per-stage barriers, double-buffered.
                #   mbar_s_ready[i]  : 1   — Q@K^T → S_tmem done for stage i
                #   mbar_scale_ready : 128 — softmax wrote scale_shared
                #   mbar_p_ready     : 128 — softmax wrote P_tmem
                #   mbar_gemm2       : 1   — P@V → O_tmem done for stage i
                mbar_s_ready = T.alloc_barrier([1] * num_stages)
                mbar_scale_ready = T.alloc_barrier([128] * num_stages)
                mbar_p_ready = T.alloc_barrier([128] * num_stages)
                mbar_gemm2 = T.alloc_barrier([1] * num_stages)
                # math signals rescale done (single, mma_load waits per iter
                # parity = k & 1 since one arrive per iter).
                mbar_rescale = T.alloc_barrier(128)
                # TMA-load completion barriers (one per stage so iter K+1 can
                # be in-flight while iter K is still consuming). Paired with
                # K_shared_{0,1} / V_shared_{0,1}.
                mbar_q_load = T.alloc_barrier(1)
                mbar_k_load = T.alloc_barrier([1] * num_stages)
                mbar_v_load = T.alloc_barrier([1] * num_stages)

                # ---------- Fragments ----------
                # Softmax-WG-private fragments (tid < 128).
                S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
                P_cast = T.alloc_fragment([block_M, block_N], dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                # MMA+corr+load WG-private fragments (tid >= 128).
                # Distinct allocations from the softmax-side scores_scale / logsum
                # — if both WGs reference the same fragment, tilelang's DCE pass
                # collapses the cross-WG SMEM round-trip and the rescale / divide
                # ends up using uninitialized registers.
                O_reg = T.alloc_fragment([block_M, dim], accum_dtype)
                corr_scale = T.alloc_fragment([block_M], accum_dtype)
                corr_logsum = T.alloc_fragment([block_M], accum_dtype)

                tid = T.get_thread_binding()

                # Initial state per role. We tried T.set_max_nreg here but
                # both WGs need similar reg counts (~168 each) and there's no
                # net donor, so explicit donation hurt perf (357 vs 400 TFLOPS).
                if tid < 128:
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.fill(logsum, 0)
                elif tid >= 128:
                    T.fill(O_reg, 0)
                    T.copy(O_reg, O_tmem)
                    T.tma_copy(
                        Q[bz, bx * block_M : (bx + 1) * block_M, by, :],
                        Q_shared, barrier=mbar_q_load,
                        annotations={"emit_arrive": 1},
                    )
                    # Prefetch K[0]+V[0] into stage 0 so iter-0 sees them
                    # already loaded (subsequent iters prefetch the next).
                    T.tma_copy(
                        K[bz, 0 : block_N, by // groups, :], K_shared_0,
                        barrier=mbar_k_load[0],
                        annotations={"emit_arrive": 1},
                    )
                    T.tma_copy(
                        V[bz, 0 : block_N, by // groups, :], V_shared_0,
                        barrier=mbar_v_load[0],
                        annotations={"emit_arrive": 1},
                    )
                    T.mbarrier_wait_parity(mbar_q_load, 0)

                loop_range = (
                    T.min(
                        T.ceildiv(seq_len, block_N),
                        T.ceildiv((bx + 1) * block_M, block_N),
                    )
                    if is_causal
                    else T.ceildiv(seq_len, block_N)
                )

                for k in T.serial(loop_range):
                    stage_id = k % num_stages
                    parity = (k // num_stages) & 1

                    # ============================================================
                    # MMA + correction + TMA load  --  tid in [128, 256)
                    # ============================================================
                    if tid >= 128:
                        next_stage = (k + 1) % num_stages
                        # 1. Prefetch K[k+1]+V[k+1] into the OTHER stage so
                        # they're in-flight while iter K's compute runs.
                        if k + 1 < loop_range:
                            if next_stage == 0:
                                T.tma_copy(
                                    K[bz, (k + 1) * block_N : (k + 2) * block_N, by // groups, :],
                                    K_shared_0, barrier=mbar_k_load[0],
                                    annotations={"emit_arrive": 1},
                                )
                                T.tma_copy(
                                    V[bz, (k + 1) * block_N : (k + 2) * block_N, by // groups, :],
                                    V_shared_0, barrier=mbar_v_load[0],
                                    annotations={"emit_arrive": 1},
                                )
                            else:
                                T.tma_copy(
                                    K[bz, (k + 1) * block_N : (k + 2) * block_N, by // groups, :],
                                    K_shared_1, barrier=mbar_k_load[1],
                                    annotations={"emit_arrive": 1},
                                )
                                T.tma_copy(
                                    V[bz, (k + 1) * block_N : (k + 2) * block_N, by // groups, :],
                                    V_shared_1, barrier=mbar_v_load[1],
                                    annotations={"emit_arrive": 1},
                                )

                        # 2. Wait for K (needed for QK), issue Q@K^T -> S_tmem
                        T.mbarrier_wait_parity(mbar_k_load[stage_id], parity)
                        if stage_id == 0:
                            T.tcgen05_gemm(
                                Q_shared, K_shared_0, S_tmem,
                                transpose_B=True, mbar=mbar_s_ready[stage_id], clear_accum=True,
                            )
                        else:
                            T.tcgen05_gemm(
                                Q_shared, K_shared_1, S_tmem,
                                transpose_B=True, mbar=mbar_s_ready[stage_id], clear_accum=True,
                            )

                        # 3. Wait for softmax to publish P_tmem.
                        T.mbarrier_wait_parity(mbar_p_ready[stage_id], parity)
                        # 4. Wait for math to finish rescaling O_tmem (k>0).
                        # Math does rescale-in-math now. At k=0, no rescale.
                        if k > 0:
                            T.mbarrier_wait_parity(mbar_rescale, (k - 1) & 1)

                        # 5. Wait for V (now needed by PV) and issue P @ V -> O_tmem.
                        T.mbarrier_wait_parity(mbar_v_load[stage_id], parity)
                        if stage_id == 0:
                            T.tcgen05_gemm(
                                P_tmem if p_in_tmem else P_shared,
                                V_shared_0,
                                O_tmem,
                                mbar=mbar_gemm2[stage_id],
                                clear_accum=(k == 0),
                            )
                        else:
                            T.tcgen05_gemm(
                                P_tmem if p_in_tmem else P_shared,
                                V_shared_1,
                                O_tmem,
                                mbar=mbar_gemm2[stage_id],
                                clear_accum=(k == 0),
                            )
                        # No wait_O here — deferred. Final wait in epilogue.

                    # ============================================================
                    # Softmax  --  tid in [0, 128)
                    # ============================================================
                    elif tid < 128:
                        T.mbarrier_wait_parity(mbar_s_ready[stage_id], parity)
                        T.copy(S_tmem, S_reg)

                        if is_causal:
                            for i, j in T.Parallel(block_M, block_N):
                                S_reg[i, j] = T.if_then_else(
                                    bx * block_M + i >= k * block_N + j,
                                    S_reg[i, j],
                                    -T.infinity(accum_dtype),
                                )
                        else:
                            for i, j in T.Parallel(block_M, block_N):
                                S_reg[i, j] = T.if_then_else(
                                    k * block_N + j >= seq_len,
                                    -T.infinity(accum_dtype),
                                    S_reg[i, j],
                                )

                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(S_reg, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        for i in T.Parallel(block_M):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_M, block_N):
                            S_reg[i, j] = T.exp2(S_reg[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(S_reg, scores_sum, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                        # Cast S -> P and publish to TMEM (P_tmem if dim==block_N).
                        T.copy(S_reg, P_cast)
                        if p_in_tmem:
                            T.copy(P_cast, P_tmem)
                        else:
                            T.copy(P_cast, P_shared)
                        T.mbarrier_arrive(mbar_p_ready[stage_id])

                        # Math WG does its own O rescale (free mma_load).
                        if k > 0:
                            prev_stage = (k - 1) % num_stages
                            prev_parity = ((k - 1) // num_stages) & 1
                            T.mbarrier_wait_parity(mbar_gemm2[prev_stage], prev_parity)
                            T.copy(O_tmem, S_reg)
                            for i, j in T.Parallel(block_M, dim):
                                S_reg[i, j] *= scores_scale[i]
                            T.copy(S_reg, O_tmem)
                            T.mbarrier_arrive(mbar_rescale)

                # ---- Epilogue: softmax → SMEM logsum, mma_corr_load → normalize + store ----
                if tid < 128:
                    T.copy(logsum, logsum_shared)
                elif tid >= 128:
                    # Final wait for last iter's PV (we deferred per-iter wait_O).
                    final_stage = (loop_range - 1) % num_stages
                    final_parity = ((loop_range - 1) // num_stages) & 1
                    T.mbarrier_wait_parity(mbar_gemm2[final_stage], final_parity)
                    T.copy(logsum_shared, corr_logsum)
                    T.copy(O_tmem, O_reg)
                    for i, j in T.Parallel(block_M, dim):
                        O_reg[i, j] /= corr_logsum[i]
                    T.copy(O_reg, O_shared)
                    T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

        return main

    # kq_stages == 2: dual-Q-tile FA4 pattern (uses TMEM aliasing)
    else:

        # ============================================================
        # kq_stages = 2: dual-Q-tile FA4 pattern
        # ============================================================
        # Block layout (threads=384):
        #   tid [  0, 128): math WG 0 — softmax + rescale O0_tmem for Q-tile 0
        #   tid [128, 256): math WG 1 — softmax + rescale O1_tmem for Q-tile 1
        #   tid [256, 384): mma_load  — TMA K/V + 4 GEMMs/iter + epilogue store
        #
        # TMEM layout (uses alloc_tmem alias=, col_offset=):
        #   S0 fp32 [128, block_N]    @ cols  0..127
        #   P0 bf16 [128, block_N]    alias S0, col_offset=block_N/2 (overlaps S0)
        #   S1 fp32 [128, block_N]    @ cols 128..255
        #   P1 bf16 [128, block_N]    alias S1, col_offset=block_N/2
        #   O0 fp32 [128, dim]        @ cols 256..383
        #   O1 fp32 [128, dim]        @ cols 384..511      (total: 512 cols, exact)
        #
        # SMEM (target ≤228 KB on B200):
        #   Q0, Q1 (block_M*dim bf16)            : 32K + 32K
        #   K_shared, V_shared (block_N*dim bf16): 32K + 32K   (single-buffered)
        #   O0_shared, O1_shared                 : 32K + 32K
        #   scale_*, logsum_* (block_M fp32)     : small
        #   Total ≈ 192 KB. Fits without P_shared (requires p_storage="tmem").
        #
        # The 2x perf comes from running math0 and math1 in parallel — both Q-tiles
        # do their softmax concurrently while mma_load issues the 4 GEMMs/iter.

        if not p_in_tmem:
            raise ValueError(
                f"kq_stages=2 needs p_storage='tmem' (requires dim == block_N); got "
                f"dim={dim}, block_N={block_N}. Use kq_stages=1 for d>block_N."
            )

        # 512 still blocked: math's tcgen05_ld_32dp32bNx<128> for S_reg
        # pins 128 regs at the load instruction (unspillable). With the
        # surrounding live state ptxas needs 146+ — over the 128 cap at
        # 512 threads. Fitting requires chunking the S load too (4 partial
        # loads + per-chunk partial reductions). Not yet wired up.
        kq2_threads = 512
        # Each block covers 2 Q-tiles = 2*block_M rows.
        kq2_block_M = block_M  # per-tile
        kq2_total_M = 2 * kq2_block_M

        @T.prim_func
        def main_kq2(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(seq_len, kq2_total_M),
                heads,
                batch,
                threads=kq2_threads,
            ) as (bx, by, bz):
                # SMEM (double-buffered K and V).
                Q0_shared = T.alloc_shared([kq2_block_M, dim], dtype)
                Q1_shared = T.alloc_shared([kq2_block_M, dim], dtype)
                K_shared_0 = T.alloc_shared([block_N, dim], dtype)
                K_shared_1 = T.alloc_shared([block_N, dim], dtype)
                V_shared_0 = T.alloc_shared([block_N, dim], dtype)
                V_shared_1 = T.alloc_shared([block_N, dim], dtype)
                O0_shared = T.alloc_shared([kq2_block_M, dim], dtype)
                O1_shared = T.alloc_shared([kq2_block_M, dim], dtype)
                logsum0_shared = T.alloc_shared([kq2_block_M], accum_dtype)
                logsum1_shared = T.alloc_shared([kq2_block_M], accum_dtype)

                # TMEM: S0/P0 aliased, S1/P1 aliased, O0, O1
                S0_tmem = T.alloc_tmem([kq2_block_M, block_N], accum_dtype)
                P0_tmem = T.alloc_tmem(
                    [kq2_block_M, block_N], dtype,
                    alias=S0_tmem, col_offset=block_N // 2,
                )
                S1_tmem = T.alloc_tmem([kq2_block_M, block_N], accum_dtype)
                P1_tmem = T.alloc_tmem(
                    [kq2_block_M, block_N], dtype,
                    alias=S1_tmem, col_offset=block_N // 2,
                )
                O0_tmem = T.alloc_tmem([kq2_block_M, dim], accum_dtype)
                O1_tmem = T.alloc_tmem([kq2_block_M, dim], accum_dtype)

                # Barriers
                mbar_s0 = T.alloc_barrier(1)
                mbar_s1 = T.alloc_barrier(1)
                # math signals P ready.
                mbar_p0 = T.alloc_barrier(128)
                mbar_p1 = T.alloc_barrier(128)
                mbar_o0 = T.alloc_barrier(1)
                mbar_o1 = T.alloc_barrier(1)
                # Math → mma_load: scores_scale is written to SMEM.
                mbar_scale0 = T.alloc_barrier(128)
                mbar_scale1 = T.alloc_barrier(128)
                # TMA-load completion barriers. Both K and V double-buffered.
                mbar_q0_load = T.alloc_barrier(1)
                mbar_q1_load = T.alloc_barrier(1)
                mbar_k_load = T.alloc_barrier([1, 1])
                mbar_v_load = T.alloc_barrier([1, 1])

                # SMEM for math→mma_load scores_scale handoff
                scores_scale0_shared = T.alloc_shared([kq2_block_M], accum_dtype)
                scores_scale1_shared = T.alloc_shared([kq2_block_M], accum_dtype)

                # Math-WG-0 fragments (tid < 128)
                # S_chunk / P_chunk are 32-col slices for chunked 2-pass
                # softmax — the full S_reg[128] tcgen05_ld<128> pins 128
                # unspillable regs; ptxas can't fit that under the 128-reg
                # cap at 512 threads.
                S0_chunk = T.alloc_fragment([kq2_block_M, 32], accum_dtype)
                P0_chunk = T.alloc_fragment([kq2_block_M, 32], dtype)
                scores_max0 = T.alloc_fragment([kq2_block_M], accum_dtype)
                scores_max_prev0 = T.alloc_fragment([kq2_block_M], accum_dtype)
                scores_scale0 = T.alloc_fragment([kq2_block_M], accum_dtype)
                scores_sum0 = T.alloc_fragment([kq2_block_M], accum_dtype)
                logsum0 = T.alloc_fragment([kq2_block_M], accum_dtype)

                # Math-WG-1 fragments (tid in [128, 256))
                S1_chunk = T.alloc_fragment([kq2_block_M, 32], accum_dtype)
                P1_chunk = T.alloc_fragment([kq2_block_M, 32], dtype)
                scores_max1 = T.alloc_fragment([kq2_block_M], accum_dtype)
                scores_max_prev1 = T.alloc_fragment([kq2_block_M], accum_dtype)
                scores_scale1 = T.alloc_fragment([kq2_block_M], accum_dtype)
                scores_sum1 = T.alloc_fragment([kq2_block_M], accum_dtype)
                logsum1 = T.alloc_fragment([kq2_block_M], accum_dtype)

                # mma_load WG fragments (tid in [256, 384))
                # O_chunk[128, 32]: shared 32-col scratch used by both the
                # in-loop chunked rescale AND the post-loop normalize+store
                # epilogue. 32 regs/thread (vs 128 for a full-tile fragment)
                # keeps both outlined fns below the 128-reg cap at 512
                # threads. Also need a small init scratch — reuse O_chunk
                # for that too (zero-fill 32 cols, write to TMEM, repeat 8x).
                O_chunk = T.alloc_fragment([kq2_block_M, 32], accum_dtype)
                mma_logsum0 = T.alloc_fragment([kq2_block_M], accum_dtype)
                mma_logsum1 = T.alloc_fragment([kq2_block_M], accum_dtype)
                # mma_load uses these for O rescale (between QK and PV)
                mma_scale0 = T.alloc_fragment([kq2_block_M], accum_dtype)
                mma_scale1 = T.alloc_fragment([kq2_block_M], accum_dtype)

                tid = T.get_thread_binding()
                kq2_loop_range = (
                    T.min(
                        T.ceildiv(seq_len, block_N),
                        T.ceildiv((bx + 1) * kq2_total_M, block_N),
                    )
                    if is_causal
                    else T.ceildiv(seq_len, block_N)
                )


                # Outlining via `tl.outline_warp_spec_branches=True` only sees
                # the for-loop body; each warp's per-iter branch becomes a
                # __device__ __noinline__ function with its own register set.
                # That means any locals initialized *outside* the for-loop in
                # the kernel stay in the kernel's registers and are invisible
                # to the outlined fn (which gets fresh `= {}` zero-init).
                #
                # To keep the math warps' `scores_max=-INF` / `logsum=0` init
                # reaching the outlined fn, we fold init and the math
                # epilogue into the loop body itself: init runs on k==0,
                # math's logsum->SMEM publish runs on k==loop_range-1. Only
                # the mma_load epilogue (which reads from TMEM/SMEM, not
                # private registers) lives outside the loop.

                # Register donation. ptxas verbose showed the mma_load WG
                # spilling 328 bytes / iter at the launch_bounds 128-reg
                # cap. Maximize mma_load's headroom by donating the idle
                # WG down to 24:
                #   8*128 + 4*232 + 4*24 = 1024+928+96 = 2048 = launch
                #   bound (512*128 = 65536 regs/SM).
                # Bumping math beyond 128 fails ptxas — likely a kernel-
                # wide alloc heuristic that propagates the per-WG max.
                if tid >= 256 and tid < 384:
                    T.set_max_nreg(232, 1)
                elif tid >= 384:
                    T.set_max_nreg(24, 0)

                for k in T.serial(kq2_loop_range):
                    # ============================================================
                    # mma_load WG  --  tid in [256, 384)
                    # ============================================================
                    if tid >= 256 and tid < 384:
                        stage = k & 1
                        next_stage = (k + 1) & 1

                        # k==0 init: zero TMEM, TMA load Q + first K/V.
                        if k == 0:
                            # Chunked TMEM zero-init via O_chunk (32 regs/
                            # thread) instead of full O0_reg (128 regs/thread)
                            # — keeps mma_load below the 128-reg cap.
                            T.fill(O_chunk, 0)
                            T.copy(O_chunk, O0_tmem[:, 0:32])
                            T.copy(O_chunk, O0_tmem[:, 32:64])
                            T.copy(O_chunk, O0_tmem[:, 64:96])
                            T.copy(O_chunk, O0_tmem[:, 96:128])
                            T.copy(O_chunk, O1_tmem[:, 0:32])
                            T.copy(O_chunk, O1_tmem[:, 32:64])
                            T.copy(O_chunk, O1_tmem[:, 64:96])
                            T.copy(O_chunk, O1_tmem[:, 96:128])
                            T.tma_copy(
                                Q[bz, bx * kq2_total_M : bx * kq2_total_M + kq2_block_M, by, :],
                                Q0_shared,
                                barrier=mbar_q0_load,
                                annotations={"emit_arrive": 1},
                            )
                            T.tma_copy(
                                Q[bz, bx * kq2_total_M + kq2_block_M : (bx + 1) * kq2_total_M, by, :],
                                Q1_shared,
                                barrier=mbar_q1_load,
                                annotations={"emit_arrive": 1},
                            )
                            T.tma_copy(
                                K[bz, 0 : block_N, by // groups, :], K_shared_0,
                                barrier=mbar_k_load[0],
                                annotations={"emit_arrive": 1},
                            )
                            T.tma_copy(
                                V[bz, 0 : block_N, by // groups, :], V_shared_0,
                                barrier=mbar_v_load[0],
                                annotations={"emit_arrive": 1},
                            )
                            T.mbarrier_wait_parity(mbar_q0_load, 0)
                            T.mbarrier_wait_parity(mbar_q1_load, 0)

                        # Prefetch K[k+1] and V[k+1] into the OTHER stage,
                        # parallel with this iter's compute.
                        if k + 1 < kq2_loop_range:
                            if next_stage == 0:
                                T.tma_copy(
                                    K[bz, (k + 1) * block_N : (k + 2) * block_N, by // groups, :],
                                    K_shared_0, barrier=mbar_k_load[0],
                                    annotations={"emit_arrive": 1},
                                )
                                T.tma_copy(
                                    V[bz, (k + 1) * block_N : (k + 2) * block_N, by // groups, :],
                                    V_shared_0, barrier=mbar_v_load[0],
                                    annotations={"emit_arrive": 1},
                                )
                            else:
                                T.tma_copy(
                                    K[bz, (k + 1) * block_N : (k + 2) * block_N, by // groups, :],
                                    K_shared_1, barrier=mbar_k_load[1],
                                    annotations={"emit_arrive": 1},
                                )
                                T.tma_copy(
                                    V[bz, (k + 1) * block_N : (k + 2) * block_N, by // groups, :],
                                    V_shared_1, barrier=mbar_v_load[1],
                                    annotations={"emit_arrive": 1},
                                )

                        T.mbarrier_wait_parity(mbar_k_load[stage], (k >> 1) & 1)
                        if stage == 0:
                            T.tcgen05_gemm(
                                Q0_shared, K_shared_0, S0_tmem,
                                transpose_B=True, mbar=mbar_s0, clear_accum=True,
                            )
                            T.tcgen05_gemm(
                                Q1_shared, K_shared_0, S1_tmem,
                                transpose_B=True, mbar=mbar_s1, clear_accum=True,
                            )
                        else:
                            T.tcgen05_gemm(
                                Q0_shared, K_shared_1, S0_tmem,
                                transpose_B=True, mbar=mbar_s0, clear_accum=True,
                            )
                            T.tcgen05_gemm(
                                Q1_shared, K_shared_1, S1_tmem,
                                transpose_B=True, mbar=mbar_s1, clear_accum=True,
                            )

                        # --- O rescale (chunked, mirrors avo correction_warp_fn) ---
                        # 16-col TMEM slices via direct slice expressions on
                        # the parent O*_tmem (its layout comes from the
                        # tcgen05_gemm calls above). Manually unrolled — the
                        # prim_func parser rejects Python loops with range().
                        if k > 0:
                            T.mbarrier_wait_parity(mbar_scale0, k & 1)
                            T.copy(scores_scale0_shared, mma_scale0)
                            T.mbarrier_wait_parity(mbar_o0, (k - 1) & 1)
                            T.copy(O0_tmem[:, 0:32], O_chunk)
                            for i, j in T.Parallel(kq2_block_M, 32):
                                O_chunk[i, j] *= mma_scale0[i]
                            T.copy(O_chunk, O0_tmem[:, 0:32])
                            T.copy(O0_tmem[:, 32:64], O_chunk)
                            for i, j in T.Parallel(kq2_block_M, 32):
                                O_chunk[i, j] *= mma_scale0[i]
                            T.copy(O_chunk, O0_tmem[:, 32:64])
                            T.copy(O0_tmem[:, 64:96], O_chunk)
                            for i, j in T.Parallel(kq2_block_M, 32):
                                O_chunk[i, j] *= mma_scale0[i]
                            T.copy(O_chunk, O0_tmem[:, 64:96])
                            T.copy(O0_tmem[:, 96:128], O_chunk)
                            for i, j in T.Parallel(kq2_block_M, 32):
                                O_chunk[i, j] *= mma_scale0[i]
                            T.copy(O_chunk, O0_tmem[:, 96:128])

                            T.mbarrier_wait_parity(mbar_scale1, k & 1)
                            T.copy(scores_scale1_shared, mma_scale1)
                            T.mbarrier_wait_parity(mbar_o1, (k - 1) & 1)
                            T.copy(O1_tmem[:, 0:32], O_chunk)
                            for i, j in T.Parallel(kq2_block_M, 32):
                                O_chunk[i, j] *= mma_scale1[i]
                            T.copy(O_chunk, O1_tmem[:, 0:32])
                            T.copy(O1_tmem[:, 32:64], O_chunk)
                            for i, j in T.Parallel(kq2_block_M, 32):
                                O_chunk[i, j] *= mma_scale1[i]
                            T.copy(O_chunk, O1_tmem[:, 32:64])
                            T.copy(O1_tmem[:, 64:96], O_chunk)
                            for i, j in T.Parallel(kq2_block_M, 32):
                                O_chunk[i, j] *= mma_scale1[i]
                            T.copy(O_chunk, O1_tmem[:, 64:96])
                            T.copy(O1_tmem[:, 96:128], O_chunk)
                            for i, j in T.Parallel(kq2_block_M, 32):
                                O_chunk[i, j] *= mma_scale1[i]
                            T.copy(O_chunk, O1_tmem[:, 96:128])

                        T.mbarrier_wait_parity(mbar_p0, k & 1)
                        T.mbarrier_wait_parity(mbar_p1, k & 1)
                        T.mbarrier_wait_parity(mbar_v_load[stage], (k >> 1) & 1)
                        if stage == 0:
                            T.tcgen05_gemm(
                                P0_tmem, V_shared_0, O0_tmem,
                                mbar=mbar_o0, clear_accum=(k == 0),
                            )
                            T.tcgen05_gemm(
                                P1_tmem, V_shared_0, O1_tmem,
                                mbar=mbar_o1, clear_accum=(k == 0),
                            )
                        else:
                            T.tcgen05_gemm(
                                P0_tmem, V_shared_1, O0_tmem,
                                mbar=mbar_o0, clear_accum=(k == 0),
                            )
                            T.tcgen05_gemm(
                                P1_tmem, V_shared_1, O1_tmem,
                                mbar=mbar_o1, clear_accum=(k == 0),
                            )

                    # ============================================================
                    # Math WG 0  --  tid in [0, 128)
                    # ============================================================
                    elif tid < 128:
                        if k == 0:
                            T.fill(scores_max0, -T.infinity(accum_dtype))
                            T.fill(logsum0, 0)

                        T.mbarrier_wait_parity(mbar_s0, k & 1)

                        # 2-pass chunked softmax. Pass 1 = chunked max,
                        # pass 2 = chunked exp+sum+cast+write P.
                        # IMPORTANT: pass 2 chunks MUST run in REVERSE order
                        # (3→2→1→0). P_tmem aliases S_tmem at col_offset=64
                        # (S cells [64,128) overlap P cells [0,64)); in
                        # forward order, the cast+store of chunk 0 would
                        # overwrite S cells that chunks 2 and 3 still need
                        # to read. Reverse order keeps every read ahead of
                        # the writes that touch its cells.
                        T.copy(scores_max0, scores_max_prev0)
                        T.fill(scores_max0, -T.infinity(accum_dtype))
                        # ---- Pass 1: chunked max ----
                        T.copy(S0_tmem[:, 0:32], S0_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + i >= k * block_N + j,
                                    S0_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    k * block_N + j >= seq_len,
                                    -T.infinity(accum_dtype), S0_chunk[i, j])
                        T.reduce_max(S0_chunk, scores_max0, dim=1, clear=False)
                        T.copy(S0_tmem[:, 32:64], S0_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + i >= k * block_N + 32 + j,
                                    S0_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    k * block_N + 32 + j >= seq_len,
                                    -T.infinity(accum_dtype), S0_chunk[i, j])
                        T.reduce_max(S0_chunk, scores_max0, dim=1, clear=False)
                        T.copy(S0_tmem[:, 64:96], S0_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + i >= k * block_N + 64 + j,
                                    S0_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    k * block_N + 64 + j >= seq_len,
                                    -T.infinity(accum_dtype), S0_chunk[i, j])
                        T.reduce_max(S0_chunk, scores_max0, dim=1, clear=False)
                        T.copy(S0_tmem[:, 96:128], S0_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + i >= k * block_N + 96 + j,
                                    S0_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    k * block_N + 96 + j >= seq_len,
                                    -T.infinity(accum_dtype), S0_chunk[i, j])
                        T.reduce_max(S0_chunk, scores_max0, dim=1, clear=False)

                        for i in T.Parallel(kq2_block_M):
                            scores_max0[i] = T.max(scores_max0[i], scores_max_prev0[i])
                        for i in T.Parallel(kq2_block_M):
                            scores_scale0[i] = T.exp2(
                                scores_max_prev0[i] * scale - scores_max0[i] * scale
                            )
                        T.copy(scores_scale0, scores_scale0_shared)
                        T.mbarrier_arrive(mbar_scale0)

                        # ---- Pass 2: chunked exp+sum+cast+write P (REVERSE order) ----
                        T.fill(scores_sum0, 0)
                        # Chunk 3 (cols 96-127, writes P cells [48,64) = S cells [112,128))
                        T.copy(S0_tmem[:, 96:128], S0_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + i >= k * block_N + 96 + j,
                                    S0_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    k * block_N + 96 + j >= seq_len,
                                    -T.infinity(accum_dtype), S0_chunk[i, j])
                        for i, j in T.Parallel(kq2_block_M, 32):
                            S0_chunk[i, j] = T.exp2(
                                S0_chunk[i, j] * scale - scores_max0[i] * scale)
                        T.reduce_sum(S0_chunk, scores_sum0, dim=1, clear=False)
                        for i, j in T.Parallel(kq2_block_M, 32):
                            P0_chunk[i, j] = T.cast(S0_chunk[i, j], dtype)
                        T.copy(P0_chunk, P0_tmem[:, 96:128])
                        # Chunk 2 (cols 64-95, writes P cells [32,48) = S cells [96,112))
                        T.copy(S0_tmem[:, 64:96], S0_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + i >= k * block_N + 64 + j,
                                    S0_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    k * block_N + 64 + j >= seq_len,
                                    -T.infinity(accum_dtype), S0_chunk[i, j])
                        for i, j in T.Parallel(kq2_block_M, 32):
                            S0_chunk[i, j] = T.exp2(
                                S0_chunk[i, j] * scale - scores_max0[i] * scale)
                        T.reduce_sum(S0_chunk, scores_sum0, dim=1, clear=False)
                        for i, j in T.Parallel(kq2_block_M, 32):
                            P0_chunk[i, j] = T.cast(S0_chunk[i, j], dtype)
                        T.copy(P0_chunk, P0_tmem[:, 64:96])
                        # Chunk 1 (cols 32-63, writes P cells [16,32) = S cells [80,96))
                        T.copy(S0_tmem[:, 32:64], S0_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + i >= k * block_N + 32 + j,
                                    S0_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    k * block_N + 32 + j >= seq_len,
                                    -T.infinity(accum_dtype), S0_chunk[i, j])
                        for i, j in T.Parallel(kq2_block_M, 32):
                            S0_chunk[i, j] = T.exp2(
                                S0_chunk[i, j] * scale - scores_max0[i] * scale)
                        T.reduce_sum(S0_chunk, scores_sum0, dim=1, clear=False)
                        for i, j in T.Parallel(kq2_block_M, 32):
                            P0_chunk[i, j] = T.cast(S0_chunk[i, j], dtype)
                        T.copy(P0_chunk, P0_tmem[:, 32:64])
                        # Chunk 0 (cols 0-31, writes P cells [0,16) = S cells [64,80))
                        T.copy(S0_tmem[:, 0:32], S0_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + i >= k * block_N + j,
                                    S0_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S0_chunk[i, j] = T.if_then_else(
                                    k * block_N + j >= seq_len,
                                    -T.infinity(accum_dtype), S0_chunk[i, j])
                        for i, j in T.Parallel(kq2_block_M, 32):
                            S0_chunk[i, j] = T.exp2(
                                S0_chunk[i, j] * scale - scores_max0[i] * scale)
                        T.reduce_sum(S0_chunk, scores_sum0, dim=1, clear=False)
                        for i, j in T.Parallel(kq2_block_M, 32):
                            P0_chunk[i, j] = T.cast(S0_chunk[i, j], dtype)
                        T.copy(P0_chunk, P0_tmem[:, 0:32])

                        for i in T.Parallel(kq2_block_M):
                            logsum0[i] = logsum0[i] * scores_scale0[i] + scores_sum0[i]

                        T.mbarrier_arrive(mbar_p0)

                        if k == kq2_loop_range - 1:
                            T.copy(logsum0, logsum0_shared)

                    # ============================================================
                    # Math WG 1  --  tid in [128, 256)
                    # ============================================================
                    elif tid >= 128 and tid < 256:
                        if k == 0:
                            T.fill(scores_max1, -T.infinity(accum_dtype))
                            T.fill(logsum1, 0)

                        T.mbarrier_wait_parity(mbar_s1, k & 1)

                        T.copy(scores_max1, scores_max_prev1)
                        T.fill(scores_max1, -T.infinity(accum_dtype))
                        # ---- Pass 1: chunked max ----
                        T.copy(S1_tmem[:, 0:32], S1_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + kq2_block_M + i >= k * block_N + j,
                                    S1_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    k * block_N + j >= seq_len,
                                    -T.infinity(accum_dtype), S1_chunk[i, j])
                        T.reduce_max(S1_chunk, scores_max1, dim=1, clear=False)
                        T.copy(S1_tmem[:, 32:64], S1_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + kq2_block_M + i >= k * block_N + 32 + j,
                                    S1_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    k * block_N + 32 + j >= seq_len,
                                    -T.infinity(accum_dtype), S1_chunk[i, j])
                        T.reduce_max(S1_chunk, scores_max1, dim=1, clear=False)
                        T.copy(S1_tmem[:, 64:96], S1_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + kq2_block_M + i >= k * block_N + 64 + j,
                                    S1_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    k * block_N + 64 + j >= seq_len,
                                    -T.infinity(accum_dtype), S1_chunk[i, j])
                        T.reduce_max(S1_chunk, scores_max1, dim=1, clear=False)
                        T.copy(S1_tmem[:, 96:128], S1_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + kq2_block_M + i >= k * block_N + 96 + j,
                                    S1_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    k * block_N + 96 + j >= seq_len,
                                    -T.infinity(accum_dtype), S1_chunk[i, j])
                        T.reduce_max(S1_chunk, scores_max1, dim=1, clear=False)

                        for i in T.Parallel(kq2_block_M):
                            scores_max1[i] = T.max(scores_max1[i], scores_max_prev1[i])
                        for i in T.Parallel(kq2_block_M):
                            scores_scale1[i] = T.exp2(
                                scores_max_prev1[i] * scale - scores_max1[i] * scale
                            )
                        T.copy(scores_scale1, scores_scale1_shared)
                        T.mbarrier_arrive(mbar_scale1)

                        # ---- Pass 2: REVERSE order to avoid P→S overlap ----
                        T.fill(scores_sum1, 0)
                        T.copy(S1_tmem[:, 96:128], S1_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + kq2_block_M + i >= k * block_N + 96 + j,
                                    S1_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    k * block_N + 96 + j >= seq_len,
                                    -T.infinity(accum_dtype), S1_chunk[i, j])
                        for i, j in T.Parallel(kq2_block_M, 32):
                            S1_chunk[i, j] = T.exp2(
                                S1_chunk[i, j] * scale - scores_max1[i] * scale)
                        T.reduce_sum(S1_chunk, scores_sum1, dim=1, clear=False)
                        for i, j in T.Parallel(kq2_block_M, 32):
                            P1_chunk[i, j] = T.cast(S1_chunk[i, j], dtype)
                        T.copy(P1_chunk, P1_tmem[:, 96:128])
                        T.copy(S1_tmem[:, 64:96], S1_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + kq2_block_M + i >= k * block_N + 64 + j,
                                    S1_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    k * block_N + 64 + j >= seq_len,
                                    -T.infinity(accum_dtype), S1_chunk[i, j])
                        for i, j in T.Parallel(kq2_block_M, 32):
                            S1_chunk[i, j] = T.exp2(
                                S1_chunk[i, j] * scale - scores_max1[i] * scale)
                        T.reduce_sum(S1_chunk, scores_sum1, dim=1, clear=False)
                        for i, j in T.Parallel(kq2_block_M, 32):
                            P1_chunk[i, j] = T.cast(S1_chunk[i, j], dtype)
                        T.copy(P1_chunk, P1_tmem[:, 64:96])
                        T.copy(S1_tmem[:, 32:64], S1_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + kq2_block_M + i >= k * block_N + 32 + j,
                                    S1_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    k * block_N + 32 + j >= seq_len,
                                    -T.infinity(accum_dtype), S1_chunk[i, j])
                        for i, j in T.Parallel(kq2_block_M, 32):
                            S1_chunk[i, j] = T.exp2(
                                S1_chunk[i, j] * scale - scores_max1[i] * scale)
                        T.reduce_sum(S1_chunk, scores_sum1, dim=1, clear=False)
                        for i, j in T.Parallel(kq2_block_M, 32):
                            P1_chunk[i, j] = T.cast(S1_chunk[i, j], dtype)
                        T.copy(P1_chunk, P1_tmem[:, 32:64])
                        T.copy(S1_tmem[:, 0:32], S1_chunk)
                        if is_causal:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    bx * kq2_total_M + kq2_block_M + i >= k * block_N + j,
                                    S1_chunk[i, j], -T.infinity(accum_dtype))
                        else:
                            for i, j in T.Parallel(kq2_block_M, 32):
                                S1_chunk[i, j] = T.if_then_else(
                                    k * block_N + j >= seq_len,
                                    -T.infinity(accum_dtype), S1_chunk[i, j])
                        for i, j in T.Parallel(kq2_block_M, 32):
                            S1_chunk[i, j] = T.exp2(
                                S1_chunk[i, j] * scale - scores_max1[i] * scale)
                        T.reduce_sum(S1_chunk, scores_sum1, dim=1, clear=False)
                        for i, j in T.Parallel(kq2_block_M, 32):
                            P1_chunk[i, j] = T.cast(S1_chunk[i, j], dtype)
                        T.copy(P1_chunk, P1_tmem[:, 0:32])

                        for i in T.Parallel(kq2_block_M):
                            logsum1[i] = logsum1[i] * scores_scale1[i] + scores_sum1[i]

                        T.mbarrier_arrive(mbar_p1)

                        if k == kq2_loop_range - 1:
                            T.copy(logsum1, logsum1_shared)


                # ---- mma_load epilogue (outside the for-loop) ----
                # Chunked O normalize via O_chunk[128, 32] (32 regs/thread)
                # instead of full O_reg[128, 128] (128 regs/thread) — keeps
                # the epilogue outlined fn below the 128-reg cap at 512
                # threads. No O→S aliasing concern here (writing to SMEM,
                # not back to TMEM), so forward order is fine.
                if tid >= 256 and tid < 384:
                    T.mbarrier_wait_parity(mbar_o0, (kq2_loop_range - 1) & 1)
                    T.mbarrier_wait_parity(mbar_o1, (kq2_loop_range - 1) & 1)
                    T.copy(logsum0_shared, mma_logsum0)
                    T.copy(O0_tmem[:, 0:32], O_chunk)
                    for i, j in T.Parallel(kq2_block_M, 32):
                        O_chunk[i, j] /= mma_logsum0[i]
                    T.copy(O_chunk, O0_shared[:, 0:32])
                    T.copy(O0_tmem[:, 32:64], O_chunk)
                    for i, j in T.Parallel(kq2_block_M, 32):
                        O_chunk[i, j] /= mma_logsum0[i]
                    T.copy(O_chunk, O0_shared[:, 32:64])
                    T.copy(O0_tmem[:, 64:96], O_chunk)
                    for i, j in T.Parallel(kq2_block_M, 32):
                        O_chunk[i, j] /= mma_logsum0[i]
                    T.copy(O_chunk, O0_shared[:, 64:96])
                    T.copy(O0_tmem[:, 96:128], O_chunk)
                    for i, j in T.Parallel(kq2_block_M, 32):
                        O_chunk[i, j] /= mma_logsum0[i]
                    T.copy(O_chunk, O0_shared[:, 96:128])
                    T.copy(
                        O0_shared,
                        Output[bz, bx * kq2_total_M : bx * kq2_total_M + kq2_block_M, by, :],
                    )

                    T.copy(logsum1_shared, mma_logsum1)
                    T.copy(O1_tmem[:, 0:32], O_chunk)
                    for i, j in T.Parallel(kq2_block_M, 32):
                        O_chunk[i, j] /= mma_logsum1[i]
                    T.copy(O_chunk, O1_shared[:, 0:32])
                    T.copy(O1_tmem[:, 32:64], O_chunk)
                    for i, j in T.Parallel(kq2_block_M, 32):
                        O_chunk[i, j] /= mma_logsum1[i]
                    T.copy(O_chunk, O1_shared[:, 32:64])
                    T.copy(O1_tmem[:, 64:96], O_chunk)
                    for i, j in T.Parallel(kq2_block_M, 32):
                        O_chunk[i, j] /= mma_logsum1[i]
                    T.copy(O_chunk, O1_shared[:, 64:96])
                    T.copy(O1_tmem[:, 96:128], O_chunk)
                    for i, j in T.Parallel(kq2_block_M, 32):
                        O_chunk[i, j] /= mma_logsum1[i]
                    T.copy(O_chunk, O1_shared[:, 96:128])
                    T.copy(
                        O1_shared,
                        Output[bz, bx * kq2_total_M + kq2_block_M : (bx + 1) * kq2_total_M, by, :],
                    )

        if not split_correction:
            return main_kq2

        # ============================================================
        # 4-role split: correction WG2 separate from mma+TMA+epi WG3.
        # ============================================================
        # Same TMEM / SMEM layout as main_kq2, but redistributes work:
        #   tid [  0, 128): math WG 0 — softmax
        #   tid [128, 256): math WG 1 — softmax
        #   tid [256, 384): correction WG — only the chunked O rescale
        #   tid [384, 512): mma + TMA + epi WG — TMA loads, both gemms,
        #                                         post-loop TMA store
        #
        # New mbarriers:
        #   mbar_corr0, mbar_corr1 (count=4, lane0 arrives from
        #                           correction's 4 warps) — correction →
        #                           mma signal: O is rescaled, safe to
        #                           accumulate next PV into it.
        #
        # The mma -> correction handshake uses an avo-style PV commit barrier:
        # the MMA WG commits once after both O0/O1 PV MMAs, and correction waits
        # on that single completion before reading O TMEM.
        @T.macro
        def softmax_warp_dsl(
            S_tmem: T.Buffer,
            P_tmem: T.Buffer,
            scores_scale_shared: T.SharedBuffer([2, kq2_block_M], accum_dtype),
            logsum_shared: T.SharedBuffer([kq2_block_M], accum_dtype),
            mbar_s: T.Buffer,
            mbar_scale: T.Buffer,
            mbar_p2: T.Buffer,
            mbar_p: T.Buffer,
            loop_extent: T.int32,
            q_row_base: T.int32,
            warp_group_offset: T.int32,
            tid_arg: T.int32,
        ):
            row = tid_arg - warp_group_offset
            warp_row_base = (((row >> 5) & 3) * 32) << 16
            sv = T.alloc_local((128,), accum_dtype)
            psa = T.alloc_local((4,), accum_dtype)
            rmax_local = T.alloc_var(accum_dtype, -T.infinity(accum_dtype))
            rsum_local = T.alloc_var(accum_dtype, 0.0)
            nm = T.alloc_var(accum_dtype)
            rs = T.alloc_var(accum_dtype)
            neg_max_scaled = T.alloc_var(accum_dtype)
            m0 = T.alloc_var(accum_dtype)
            m1 = T.alloc_var(accum_dtype)
            m2 = T.alloc_var(accum_dtype)

            for k_soft in T.unroll(loop_extent, explicit=False, unroll_factor=1):
                phase = k_soft & 1
                kv_col_base = k_soft * block_N
                T.mbarrier_wait_parity(mbar_s, phase)
                T.tcgen05_after_thread_sync()

                nm = rmax_local
                for cc in T.unroll(0, 128, 16):
                    T.tcgen05_ld(
                        32,
                        16,
                        False,
                        S_tmem[0, 0],
                        warp_row_base + cc,
                        T.access_ptr(sv[cc], "w", 16),
                        emit_fence=False,
                    )
                T.tcgen05_fence_tmem_load()

                if is_causal:
                    for i in T.unroll(128):
                        sv[i] = T.if_then_else(
                            (kv_col_base + i < seq_len) & (kv_col_base + i <= q_row_base + row),
                            sv[i],
                            -T.infinity(accum_dtype),
                        )
                else:
                    remaining = seq_len - kv_col_base
                    if remaining < block_N:
                        for i in T.unroll(128):
                            if i >= remaining:
                                sv[i] = -T.infinity(accum_dtype)

                m0 = T.max3(nm, sv[0], sv[1])
                m1 = T.max3(sv[2], sv[3], sv[4])
                m2 = T.max3(sv[5], sv[6], sv[7])
                for i in T.unroll(8, 128, 8):
                    m0 = T.max3(m0, sv[i + 0], sv[i + 1])
                    m1 = T.max3(m1, sv[i + 2], sv[i + 3])
                    m2 = T.max3(m2, sv[i + 4], sv[i + 5])
                    nm = T.max3(nm, sv[i + 6], sv[i + 7])
                nm = T.max3(T.fmax2_ftz(m0, m1), m2, nm)

                T.tcgen05_softmax_rescale_update(rs, rmax_local, rsum_local, nm, scale)
                scores_scale_shared[phase, row] = rs
                T.sync_warp()
                T.mbarrier_arrive(T.mbarrier_at(mbar_scale, phase), lane0=True)

                neg_max_scaled = -(rmax_local * scale)
                for cc in T.unroll(0, 128, 16):
                    for i in T.unroll(0, 16, 2):
                        T.tcgen05_fma_f32x2(
                            sv[cc + i],
                            sv[cc + i + 1],
                            sv[cc + i],
                            sv[cc + i + 1],
                            scale,
                            scale,
                            neg_max_scaled,
                            neg_max_scaled,
                        )

                for i in T.unroll(4):
                    psa[i] = 0.0

                for chunk in T.unroll(8):
                    cc = 112 - chunk * 16
                    for g_iter in T.unroll(2):
                        g = 8 - g_iter * 8
                        elem_base = cc + g
                        T.tcgen05_softmax_store_8(
                            P_tmem[0, 0],
                            warp_row_base + elem_base // 2,
                            T.access_ptr(sv[elem_base], "r", 8),
                            psa[0],
                            psa[1],
                            psa[2],
                            psa[3],
                            elem_base,
                        )
                    if cc == 32:
                        T.tcgen05_fence_tmem_store()
                        T.mbarrier_arrive(mbar_p2, lane0=True)

                rsum_local += (psa[0] + psa[1]) + (psa[2] + psa[3])
                if k_soft == loop_extent - 1:
                    logsum_shared[row] = rsum_local

                T.tcgen05_fence_tmem_store()
                T.mbarrier_arrive(mbar_p, lane0=True)


        @T.macro
        def correction_epilogue_warp_dsl(
            O0_tmem: T.Buffer,
            O1_tmem: T.Buffer,
            O0_shared: T.SharedBuffer([kq2_block_M, dim], dtype),
            scores_scale0_shared: T.SharedBuffer([2, kq2_block_M], accum_dtype),
            scores_scale1_shared: T.SharedBuffer([2, kq2_block_M], accum_dtype),
            logsum0_shared: T.SharedBuffer([kq2_block_M], accum_dtype),
            logsum1_shared: T.SharedBuffer([kq2_block_M], accum_dtype),
            mbar_scale0: T.Buffer,
            mbar_scale1: T.Buffer,
            mbar_pv: T.Buffer,
            mbar_corr0: T.Buffer,
            mbar_corr1: T.Buffer,
            mbar_epi0: T.Buffer,
            mbar_epi1: T.Buffer,
            loop_extent: T.int32,
        ):
            row = T.get_thread_binding() - 256
            for k_corr in T.unroll(loop_extent, explicit=False, unroll_factor=1):
                if k_corr > 0:
                    scale_idx = k_corr & 1
                    scale_phase = (k_corr >> 1) & 1
                    pv_phase = (k_corr - 1) & 1

                    T.mbarrier_wait_parity(T.mbarrier_at(mbar_scale0, scale_idx), scale_phase)
                    T.tcgen05_correction_x16(
                        O0_tmem[0, 0],
                        scores_scale0_shared[scale_idx, row],
                        mbar_pv=mbar_pv,
                        pv_phase=pv_phase,
                        warp_group_offset=256,
                        head_dim=dim,
                    )
                    T.tcgen05_before_thread_sync()
                    T.mbarrier_arrive(mbar_corr0, lane0=True)

                    T.mbarrier_wait_parity(T.mbarrier_at(mbar_scale1, scale_idx), scale_phase)
                    T.tcgen05_correction_x16(
                        O1_tmem[0, 0],
                        scores_scale1_shared[scale_idx, row],
                        mbar_pv=mbar_pv,
                        pv_phase=pv_phase,
                        warp_group_offset=256,
                        head_dim=dim,
                    )
                    T.tcgen05_before_thread_sync()
                    T.mbarrier_arrive(mbar_corr1, lane0=True)

            final_phase = (loop_extent - 1) & 1
            T.mbarrier_wait_parity(mbar_pv, final_phase)
            T.tcgen05_after_thread_sync()
            T.tcgen05_epilogue_store_x16(
                O0_tmem[0, 0],
                T.access_ptr(O0_shared, "w"),
                logsum0_shared[row],
                warp_group_offset=256,
                head_dim=dim,
            )
            T.tcgen05_before_thread_sync()
            T.sync_warp()
            T.fence_proxy_async()
            T.mbarrier_arrive(mbar_epi0, lane0=True)

            T.tcgen05_after_thread_sync()
            T.tcgen05_epilogue_store_x16(
                O1_tmem[0, 0],
                T.access_ptr(O0_shared, "w", offset=kq2_block_M * dim),
                logsum1_shared[row],
                warp_group_offset=256,
                head_dim=dim,
            )
            T.tcgen05_before_thread_sync()
            T.sync_warp()
            T.fence_proxy_async()
            T.mbarrier_arrive(mbar_epi1, lane0=True)


        @T.macro
        def mma_qk_stage_dsl(
            Q0_stage: T.SharedBuffer([kq2_block_M, dim], dtype),
            Q1_stage: T.SharedBuffer([kq2_block_M, dim], dtype),
            K_stage: T.SharedBuffer([block_N, dim], dtype),
            mbar_k: T.Buffer,
            mbar_s0: T.Buffer,
            mbar_s1: T.Buffer,
            S0_tmem: T.Buffer,
            S1_tmem: T.Buffer,
            phase: T.int32,
        ):
            T.mbarrier_wait_parity(mbar_k, phase)
            T.tcgen05_after_thread_sync()
            T.tcgen05_mma_1sm_ss_128x128_commit_lane0(
                T.access_ptr(Q0_stage, "r"),
                T.access_ptr(K_stage, "r"),
                S0_tmem[0, 0],
                mbar_s0,
            )
            T.tcgen05_mma_1sm_ss_128x128_commit_lane0(
                T.access_ptr(Q1_stage, "r"),
                T.access_ptr(K_stage, "r"),
                S1_tmem[0, 0],
                mbar_s1,
            )


        @T.macro
        def mma_pv_stage_dsl(
            V_stage: T.SharedBuffer([block_N, dim], dtype),
            mbar_v: T.Buffer,
            mbar_p2: T.Buffer,
            mbar_p: T.Buffer,
            mbar_corr: T.Buffer,
            P_tmem: T.Buffer,
            O_tmem: T.Buffer,
            phase: T.int32,
            pv_phase: T.int32,
            corr_phase: T.int32,
            accum: T.int32,
            needs_corr: T.int32,
        ):
            T.mbarrier_wait_parity(mbar_p2, pv_phase)
            T.mbarrier_wait_parity(mbar_v, phase)
            if needs_corr != 0:
                T.mbarrier_wait_parity(mbar_corr, corr_phase)
            T.mbarrier_wait_parity(mbar_p, pv_phase)
            T.tcgen05_after_thread_sync()
            T.tcgen05_mma_1sm_ts_128x64_bmn_x2_contig_lane0(
                T.access_ptr(V_stage, "r"),
                P_tmem[0, 0],
                O_tmem[0, 0],
                accum,
            )


        @T.macro
        def mma_warp_reuse3_dsl(
            Q0_stage: T.SharedBuffer([kq2_block_M, dim], dtype),
            Q1_stage: T.SharedBuffer([kq2_block_M, dim], dtype),
            K0_stage: T.SharedBuffer([block_N, dim], dtype),
            K1_stage: T.SharedBuffer([block_N, dim], dtype),
            KV2_stage: T.SharedBuffer([block_N, dim], dtype),
            mbar_q0: T.Buffer,
            mbar_q1: T.Buffer,
            mbar_k0: T.Buffer,
            mbar_k1: T.Buffer,
            mbar_k2: T.Buffer,
            mbar_v0: T.Buffer,
            mbar_v1: T.Buffer,
            mbar_v2: T.Buffer,
            mbar_s0: T.Buffer,
            mbar_s1: T.Buffer,
            mbar_p0: T.Buffer,
            mbar_p1: T.Buffer,
            mbar_p2_0: T.Buffer,
            mbar_p2_1: T.Buffer,
            mbar_corr0: T.Buffer,
            mbar_corr1: T.Buffer,
            mbar_pv: T.Buffer,
            S0_tmem: T.Buffer,
            S1_tmem: T.Buffer,
            P0_tmem: T.Buffer,
            P1_tmem: T.Buffer,
            O0_tmem: T.Buffer,
            O1_tmem: T.Buffer,
            loop_extent: T.int32,
        ):
            tid_mma = T.get_thread_binding()
            if (tid_mma & 31) == 0:
                q0_ptr = T.access_ptr(Q0_stage, "r")
                q1_ptr = T.access_ptr(Q1_stage, "r")
                k0_ptr = T.access_ptr(K0_stage, "r")
                k1_ptr = T.access_ptr(K1_stage, "r")
                kv2_ptr = T.access_ptr(KV2_stage, "r")

                T.tcgen05_wait_barrier(mbar_q0, 0)
                T.tcgen05_wait_barrier(mbar_q1, 0)
                T.tcgen05_after_thread_sync()

                T.tcgen05_wait_barrier(mbar_k0, 0)
                T.tcgen05_after_thread_sync()
                T.tcgen05_mma_1sm_ss_128x128_commit_lane0(
                    q0_ptr,
                    k0_ptr,
                    S0_tmem[0, 0],
                    mbar_s0,
                )
                T.tcgen05_mma_1sm_ss_128x128_commit_lane0(
                    q1_ptr,
                    k0_ptr,
                    S1_tmem[0, 0],
                    mbar_s1,
                )

                for k_mma in T.unroll(loop_extent, explicit=False, unroll_factor=1):
                    stage = k_mma % 3
                    phase = (k_mma // 3) & 1
                    pv_phase = k_mma & 1
                    corr_phase = (k_mma - 1) & 1
                    accum = T.Select(k_mma == 0, T.uint32(0), T.uint32(1))
                    vbar = T.tcgen05_reuse3_barrier_ptr(mbar_v0, mbar_v1, mbar_v2, stage)

                    T.tcgen05_wait_barrier(mbar_p2_0, pv_phase)
                    T.tcgen05_wait_barrier_ptr(vbar, phase)
                    if k_mma > 0:
                        T.tcgen05_wait_barrier(mbar_corr0, corr_phase)
                    T.tcgen05_wait_barrier(mbar_p0, pv_phase)
                    T.tcgen05_after_thread_sync()
                    vptr = T.tcgen05_reuse3_stage_ptr(k0_ptr, k1_ptr, kv2_ptr, stage)
                    T.tcgen05_mma_1sm_ts_128x64_bmn_x2_contig_lane0(
                        vptr,
                        P0_tmem[0, 0],
                        O0_tmem[0, 0],
                        accum,
                    )

                    T.tcgen05_wait_barrier(mbar_p2_1, pv_phase)
                    if k_mma > 0:
                        T.tcgen05_wait_barrier(mbar_corr1, corr_phase)
                    T.tcgen05_wait_barrier(mbar_p1, pv_phase)
                    T.tcgen05_after_thread_sync()
                    T.tcgen05_mma_1sm_ts_128x64_bmn_x2_contig_lane0(
                        vptr,
                        P1_tmem[0, 0],
                        O1_tmem[0, 0],
                        accum,
                    )
                    T.tcgen05_commit_1sm_lane0(mbar_pv)

                    if k_mma + 1 < loop_extent:
                        next_stage = (k_mma + 1) % 3
                        next_phase = ((k_mma + 1) // 3) & 1
                        kbar = T.tcgen05_reuse3_barrier_ptr(mbar_k0, mbar_k1, mbar_k2, next_stage)
                        T.tcgen05_wait_barrier_ptr(kbar, next_phase)
                        T.tcgen05_after_thread_sync()
                        kptr = T.tcgen05_reuse3_stage_ptr(k0_ptr, k1_ptr, kv2_ptr, next_stage)
                        T.tcgen05_mma_1sm_ss_128x128_commit_lane0(
                            q0_ptr,
                            kptr,
                            S0_tmem[0, 0],
                            mbar_s0,
                        )
                        T.tcgen05_mma_1sm_ss_128x128_commit_lane0(
                            q1_ptr,
                            kptr,
                            S1_tmem[0, 0],
                            mbar_s1,
                        )


        @T.prim_func
        def main_kq2_split(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(seq_len, kq2_total_M),
                heads,
                batch,
                threads=kq2_threads,
                cluster_dims=1,
            ) as (bx, by, bz):
                T.annotate_min_blocks_per_sm(1)
                Q0_shared = T.alloc_shared([kq2_block_M, dim], dtype)
                Q1_shared = T.alloc_shared([kq2_block_M, dim], dtype)
                K_shared_0 = T.alloc_shared([block_N, dim], dtype)
                K_shared_1 = T.alloc_shared([block_N, dim], dtype)
                KV_shared_2 = T.alloc_shared([block_N, dim], dtype)
                O0_shared = T.alloc_shared([kq2_block_M, dim], dtype)
                O1_shared = T.alloc_shared([kq2_block_M, dim], dtype)
                logsum0_shared = T.alloc_shared([kq2_block_M], accum_dtype)
                logsum1_shared = T.alloc_shared([kq2_block_M], accum_dtype)

                S0_tmem = T.alloc_tmem([kq2_block_M, block_N], accum_dtype)
                P0_tmem = T.alloc_tmem(
                    [kq2_block_M, block_N], dtype,
                    alias=S0_tmem, col_offset=block_N // 2,
                )
                S1_tmem = T.alloc_tmem([kq2_block_M, block_N], accum_dtype)
                P1_tmem = T.alloc_tmem(
                    [kq2_block_M, block_N], dtype,
                    alias=S1_tmem, col_offset=block_N // 2,
                )
                O0_tmem = T.alloc_tmem([kq2_block_M, dim], accum_dtype)
                O1_tmem = T.alloc_tmem([kq2_block_M, dim], accum_dtype)
                pv_o_layout = T.Layout(
                    [kq2_block_M, dim],
                    lambda i, j: [i, j],
                )
                pv_p_layout = T.Layout(
                    [kq2_block_M, block_N],
                    lambda i, j: [i, j],
                )
                T.annotate_layout({
                    S0_tmem: pv_p_layout,
                    S1_tmem: pv_p_layout,
                    O0_tmem: pv_o_layout,
                    O1_tmem: pv_o_layout,
                    P0_tmem: pv_p_layout,
                    P1_tmem: pv_p_layout,
                    Q0_shared: tilelang.layout.make_full_bank_swizzled_layout(Q0_shared),
                    Q1_shared: tilelang.layout.make_full_bank_swizzled_layout(Q1_shared),
                    K_shared_0: tilelang.layout.make_full_bank_swizzled_layout(K_shared_0),
                    K_shared_1: tilelang.layout.make_full_bank_swizzled_layout(K_shared_1),
                    KV_shared_2: tilelang.layout.make_full_bank_swizzled_layout(KV_shared_2),
                })

                mbar_s0 = T.alloc_barrier(1)
                mbar_s1 = T.alloc_barrier(1)
                # Full-P and partial-P ready signals. Match avo's mb_p/mb_p2:
                # one arrive per softmax warp, issued by lane 0 after P TMEM
                # stores finish and after the halfway point respectively.
                mbar_p0 = T.alloc_barrier(4)
                mbar_p1 = T.alloc_barrier(4)
                mbar_p2_0 = T.alloc_barrier(4)
                mbar_p2_1 = T.alloc_barrier(4)
                mbar_pv = T.alloc_barrier(1)
                # Softmax -> correction scale handoff. Match avo's mb_rs:
                # each of the 4 softmax warps arrives once via lane 0 after
                # writing its 32 row scale values.
                mbar_scale0 = T.alloc_barrier([4, 4])
                mbar_scale1 = T.alloc_barrier([4, 4])
                # Correction -> mma WG. Match avo: one arrive per correction
                # warp, issued by lane 0, after the per-warp x16 TMEM stores.
                mbar_corr0 = T.alloc_barrier(4)
                mbar_corr1 = T.alloc_barrier(4)
                mbar_epi0 = T.alloc_barrier(4)
                mbar_epi1 = T.alloc_barrier(4)
                mbar_q0_load = T.alloc_barrier(1)
                mbar_q1_load = T.alloc_barrier(1)
                mbar_k_load = T.alloc_barrier([1, 1])
                mbar_v_lo_load = T.alloc_barrier([1, 1])
                mbar_v_hi_load = T.alloc_barrier([1, 1])

                scores_scale0_shared = T.alloc_shared([2, kq2_block_M], accum_dtype)
                scores_scale1_shared = T.alloc_shared([2, kq2_block_M], accum_dtype)

                tid = T.get_thread_binding()
                loop_range = (
                    T.min(
                        T.ceildiv(seq_len, block_N),
                        T.ceildiv((bx + 1) * kq2_total_M, block_N),
                    )
                    if is_causal
                    else T.ceildiv(seq_len, block_N)
                )

                # Avo-style register donation for the 4-role split:
                #   warps 0-7   softmax    184 regs
                #   warps 8-11  correction  64 regs
                #   warps 12-15 mma/prod/epi/idle 80 regs
                # Avo uses __launch_bounds__(512, 1). Empirically this still
                # preserves the per-role setmaxnreg donation while giving ptxas
                # a slightly better scheduling target than minBlocks=0.
                if tid < 256:
                    T.set_max_nreg(184, 1)
                elif tid < 384:
                    T.set_max_nreg(64, 0)
                else:
                    T.set_max_nreg(80, 0)

                # The DSL softmax body owns the full KV loop internally, like
                # avo's softmax_warp_fn, but exposes the readable loop/max/
                # rescale/store structure instead of hiding it in one helper.
                if tid < 128:
                    softmax_warp_dsl(
                        S0_tmem,
                        P0_tmem,
                        scores_scale0_shared,
                        logsum0_shared,
                        mbar_s0,
                        mbar_scale0,
                        mbar_p2_0,
                        mbar_p0,
                        loop_range,
                        bx * kq2_total_M,
                        0,
                        tid,
                    )
                elif tid < 256:
                    softmax_warp_dsl(
                        S1_tmem,
                        P1_tmem,
                        scores_scale1_shared,
                        logsum1_shared,
                        mbar_s1,
                        mbar_scale1,
                        mbar_p2_1,
                        mbar_p1,
                        loop_range,
                        bx * kq2_total_M + kq2_block_M,
                        128,
                        tid,
                    )

                for k in T.serial(loop_range):
                    # ============================================================
                    # Producer warp 13  --  tid in [416, 448)
                    # ============================================================
                    if tid >= 416 and tid < 448:
                        if k == 0:
                            q_desc = T.create_tma_descriptor(
                                9, 4, T.access_ptr(Q, "r"),
                                dim, heads, seq_len, batch,
                                2, dim * 2, heads * dim * 2, seq_len * heads * dim * 2,
                                64, 1, 128, 1,
                                1, 1, 1, 1,
                                0, 3, 2, 0,
                            )
                            k_desc = T.create_tma_descriptor(
                                9, 4, T.access_ptr(K, "r"),
                                dim, num_kv_heads, seq_len, batch,
                                2, dim * 2, num_kv_heads * dim * 2,
                                seq_len * num_kv_heads * dim * 2,
                                64, 1, 128, 1,
                                1, 1, 1, 1,
                                0, 3, 2, 0,
                            )
                            v_desc = T.create_tma_descriptor(
                                9, 4, T.access_ptr(V, "r"),
                                dim, num_kv_heads, seq_len, batch,
                                2, dim * 2, num_kv_heads * dim * 2,
                                seq_len * num_kv_heads * dim * 2,
                                64, 1, 128, 1,
                                1, 1, 1, 1,
                                0, 3, 2, 0,
                            )
                            T.tcgen05_producer_warp_1sm_reuse3(
                                q_desc, k_desc, v_desc,
                                T.access_ptr(Q0_shared, "w"),
                                T.access_ptr(Q1_shared, "w"),
                                T.access_ptr(K_shared_0, "w"),
                                T.access_ptr(K_shared_1, "w"),
                                T.access_ptr(KV_shared_2, "w"),
                                mbar_q0_load,
                                mbar_q1_load,
                                mbar_k_load[0],
                                mbar_k_load[1],
                                mbar_v_hi_load[0],
                                mbar_v_lo_load[0],
                                mbar_v_lo_load[1],
                                mbar_v_hi_load[1],
                                mbar_s1,
                                mbar_pv,
                                loop_range,
                                bx * kq2_total_M,
                                by,
                                by // groups,
                                bz,
                            )

                    # ============================================================
                    # MMA warp 12  --  tid in [384, 416)
                    # ============================================================
                    elif tid >= 384 and tid < 416:
                        if k == 0:
                            if not mma_dsl:
                                T.tcgen05_mma_warp_1sm_reuse3(
                                    T.access_ptr(Q0_shared, "r"),
                                    T.access_ptr(Q1_shared, "r"),
                                    T.access_ptr(K_shared_0, "r"),
                                    T.access_ptr(K_shared_1, "r"),
                                    T.access_ptr(KV_shared_2, "r"),
                                    mbar_q0_load,
                                    mbar_q1_load,
                                    mbar_k_load[0],
                                    mbar_k_load[1],
                                    mbar_v_hi_load[0],
                                    mbar_v_lo_load[0],
                                    mbar_v_lo_load[1],
                                    mbar_v_hi_load[1],
                                    mbar_s0,
                                    mbar_s1,
                                    mbar_p0,
                                    mbar_p1,
                                    mbar_p2_0,
                                    mbar_p2_1,
                                    mbar_corr0,
                                    mbar_corr1,
                                    mbar_pv,
                                    S0_tmem[0, 0],
                                    S1_tmem[0, 0],
                                    P0_tmem[0, 0],
                                    P1_tmem[0, 0],
                                    O0_tmem[0, 0],
                                    O1_tmem[0, 0],
                                    loop_range,
                                )
                            else:
                                with T.device_func():
                                    mma_warp_reuse3_dsl(
                                        Q0_shared,
                                        Q1_shared,
                                        K_shared_0,
                                        K_shared_1,
                                        KV_shared_2,
                                        mbar_q0_load,
                                        mbar_q1_load,
                                        mbar_k_load[0],
                                        mbar_k_load[1],
                                        mbar_v_hi_load[0],
                                        mbar_v_lo_load[0],
                                        mbar_v_lo_load[1],
                                        mbar_v_hi_load[1],
                                        mbar_s0,
                                        mbar_s1,
                                        mbar_p0,
                                        mbar_p1,
                                        mbar_p2_0,
                                        mbar_p2_1,
                                        mbar_corr0,
                                        mbar_corr1,
                                        mbar_pv,
                                        S0_tmem,
                                        S1_tmem,
                                        P0_tmem,
                                        P1_tmem,
                                        O0_tmem,
                                        O1_tmem,
                                        loop_range,
                                    )

                    # ============================================================
                    # Correction WG  --  tid in [256, 384)
                    # ============================================================
                    elif tid >= 256 and tid < 384:
                        if k == 0:
                            correction_epilogue_warp_dsl(
                                O0_tmem,
                                O1_tmem,
                                O0_shared,
                                scores_scale0_shared,
                                scores_scale1_shared,
                                logsum0_shared,
                                logsum1_shared,
                                mbar_scale0,
                                mbar_scale1,
                                mbar_pv,
                                mbar_corr0,
                                mbar_corr1,
                                mbar_epi0,
                                mbar_epi1,
                                loop_range,
                            )

                    else:
                        T.evaluate(0)

                # ---- Avo-style split epilogue TMA store ----
                if tid >= 448 and tid < 480:
                    output_desc = T.create_tma_descriptor(
                        9, 4, T.access_ptr(Output, "w"),
                        dim, heads, seq_len, batch,
                        2, dim * 2, heads * dim * 2, seq_len * heads * dim * 2,
                        64, 1, 32, 1,
                        1, 1, 1, 1,
                        0, 3, 2, 0,
                    )
                    T.tcgen05_epilogue_warp_1sm_skv(
                        output_desc,
                        T.access_ptr(O0_shared, "r"),
                        mbar_epi0,
                        mbar_epi1,
                        0,
                        bx * kq2_total_M,
                        by,
                        bz,
                    )

        return main_kq2_split


# --------------------------------------------------------------------------- #
# Reference + driver for smoke testing.
# --------------------------------------------------------------------------- #
def reference_attention(Q, K, V, is_causal=False):
    Q_f = Q.permute(0, 2, 1, 3).to(torch.float32)
    K_f = K.permute(0, 2, 1, 3).to(torch.float32)
    V_f = V.permute(0, 2, 1, 3).to(torch.float32)
    if Q_f.size(1) != K_f.size(1):
        if Q_f.size(1) % K_f.size(1) != 0:
            raise ValueError(
                f"q_heads={Q_f.size(1)} must be divisible by kv_heads={K_f.size(1)}"
            )
        groups = Q_f.size(1) // K_f.size(1)
        K_f = K_f.repeat_interleave(groups, dim=1)
        V_f = V_f.repeat_interleave(groups, dim=1)
    dim = Q.size(-1)
    seq_q = Q.size(1)
    seq_k = K.size(1)
    scores = (Q_f @ K_f.transpose(-1, -2)) * (1.0 / dim**0.5)
    if is_causal:
        mask = torch.tril(torch.ones(seq_q, seq_k, device=Q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    out = weights @ V_f
    return out.permute(0, 2, 1, 3).to(Q.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv_heads", type=int, default=None)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument(
        "--skip_ref", action="store_true",
        help="Skip PyTorch reference validation; useful for long-sequence benchmarking.",
    )
    ap.add_argument(
        "--kq_stages", type=int, default=2,
        help="1 = single Q-tile path; 2 = dual-Q-tile FA4 pattern (faster, default)",
    )
    ap.add_argument("--print_source", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    Q = torch.randn(args.batch, args.seq, args.heads, args.dim,
                    dtype=torch.bfloat16, device="cuda")
    kv_h = args.kv_heads or args.heads
    K = torch.randn(args.batch, args.seq, kv_h, args.dim,
                    dtype=torch.bfloat16, device="cuda")
    V = torch.randn(args.batch, args.seq, kv_h, args.dim,
                    dtype=torch.bfloat16, device="cuda")

    fn = attention_kernel_1sm(
        args.batch,
        args.heads,
        args.seq,
        args.dim,
        num_kv_heads=kv_h,
        is_causal=args.causal,
        kq_stages=args.kq_stages,
        split_correction=False,
        mma_dsl=True,
    )
    if args.print_source:
        print(fn.get_kernel_source())
    
    O = fn(Q, K, V)
    if args.skip_ref:
        print(f"shape={tuple(O.shape)}  correctness=skipped")
    else:
        O_ref = reference_attention(Q, K, V, is_causal=args.causal)
        err_abs = (O.to(torch.float32) - O_ref.to(torch.float32)).abs()
        print(
            f"shape={tuple(O.shape)}  "
            f"max_abs={err_abs.max().item():.4f}  "
            f"mean_abs={err_abs.mean().item():.4f}"
        )

    if args.bench:
        from tilelang.profiler import do_bench
        for _ in range(3):
            _ = fn(Q, K, V)
        torch.cuda.synchronize()
        lat = do_bench(lambda: fn(Q, K, V), warmup=25, rep=100)
        causal_factor = 0.5 if args.causal else 1.0
        flops = 2.0 * 2.0 * args.batch * args.heads * args.seq * args.seq * args.dim * causal_factor
        tflops = flops / lat * 1e-9
        print(f"latency={lat:.3f} ms  perf={tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
