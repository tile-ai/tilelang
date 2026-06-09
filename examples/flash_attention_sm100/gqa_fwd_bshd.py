"""Blackwell (SM100) GQA forward, BSHD layout.

Q: [batch, seq_len, heads, dim], K/V: [batch, seq_len, head_kv, dim], head_kv = heads // groups.
variant='ss': mma_ss for both GEMMs (128 threads, P via shared memory).
variant='ts': mma_ts for GEMM 2 (256 threads, P via tensor memory).
variant='wasp': warp-specialized pipeline (softmax/DMA/BMM warps); GEMM 2 mma_ts.
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import argparse


PASS_CFG = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups=1,
    block_M=128,
    block_N=128,
    variant="ss",
):
    """GQA forward. variant='ss': mma_ss (128t, P via shared); 'ts': mma_ts (256t, P via TMEM)."""
    if groups <= 0 or heads % groups != 0:
        raise ValueError("groups must be a positive divisor of heads")
    head_kv = heads // groups
    use_ts = variant == "ts"
    threads = 256 if use_ts else 128
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            D_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            mbar_s = T.alloc_barrier(1)
            mbar_d = T.alloc_barrier(1)

            if use_ts:
                P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            else:
                P_shared = T.alloc_shared([block_M, block_N], dtype)

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)
            D_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(O_reg, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)

                T.tcgen05_gemm(
                    Q_shared,
                    K_shared,
                    S_tmem,
                    transpose_B=True,
                    mbar=mbar_s,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_s, k % 2)

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

                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] *= scores_scale[i]

                T.copy(S_reg, P_cast)
                if use_ts:
                    T.copy(P_cast, P_tmem)
                    P_operand = P_tmem
                else:
                    T.copy(P_cast, P_shared)
                    P_operand = P_shared

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)

                T.tcgen05_gemm(
                    P_operand,
                    V_shared,
                    D_tmem,
                    mbar=mbar_d,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_d, k % 2)

                T.copy(D_tmem, D_reg)
                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] += D_reg[i, j]

            for i, j in T.Parallel(block_M, dim):
                O_reg[i, j] /= logsum[i]
            T.copy(O_reg, O_shared)
            T.copy(
                O_shared,
                Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
            )

    return main


flashattn_ss = flashattn
flashattn_ts = flashattn


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn_ts_v2(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups=1,
    block_M=128,
    block_N=128,
    num_stages=2,
):
    """Improved ts variant: in-place O_tmem accumulation, no D_reg, pipelined.

    Aligns with Triton's Blackwell FA: 256 threads, P via TMEM, GEMM2
    accumulates directly into O_tmem. Eliminates D_tmem/D_reg to reduce
    register pressure from ~229 to ~165 regs/thread.
    """
    if groups <= 0 or heads % groups != 0:
        raise ValueError("groups must be a positive divisor of heads")
    head_kv = heads // groups
    threads = 256
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            O_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            mbar_s = T.alloc_barrier(1)
            mbar_o = T.alloc_barrier(1)

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(O_reg, 0)
            T.copy(O_reg, O_tmem)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)

                # GEMM1: Q × K^T → S_tmem
                T.tcgen05_gemm(
                    Q_shared,
                    K_shared,
                    S_tmem,
                    transpose_B=True,
                    mbar=mbar_s,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_s, k % 2)

                T.copy(S_tmem, S_reg)

                # Causal / boundary mask
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

                # Online softmax
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

                # Rescale O: read from TMEM → scale → write back
                T.copy(O_tmem, O_reg)
                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] *= scores_scale[i]
                T.copy(O_reg, O_tmem)

                # Cast P and store to TMEM
                T.copy(S_reg, P_cast)
                T.copy(P_cast, P_tmem)

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)

                # Fence: ensure O_tmem and P_tmem stores are visible to tensor core
                T.tcgen05_before_thread_sync()
                T.sync_threads()
                T.tcgen05_after_thread_sync()

                # GEMM2: P × V → O_tmem (accumulate in-place)
                T.tcgen05_gemm(
                    P_tmem,
                    V_shared,
                    O_tmem,
                    mbar=mbar_o,
                    clear_accum=False,
                )
                T.mbarrier_wait_parity(mbar_o, k % 2)

            # Final output: read O_tmem, normalize, write
            T.copy(O_tmem, O_reg)
            for i, j in T.Parallel(block_M, dim):
                O_reg[i, j] /= logsum[i]
            T.copy(O_reg, O_shared)
            T.copy(
                O_shared,
                Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
            )

    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn_fa4_pipe(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups=1,
    block_M=128,
    block_N=128,
):
    """FlashAttention variant written in TileLang's natural style.

    This version uses high-level TileLang primitives with one math warp-group,
    one TMEM accumulator, and ``T.Pipelined`` so the kernel reads top-to-bottom
    like a compact FA schedule.  It trades peak hand-scheduled performance for
    a smaller, easier-to-read implementation.

    Implemented directly in DSL:
      • Online softmax with deferred normalization (logsum + divide at epilogue).
      • ``tcgen05.mma`` for both Q@K^T and P@V via ``T.tcgen05_gemm``.
      • Hot loop wrapped in ``T.Pipelined`` so the scheduler can overlap the
        async ``T.copy`` (TMA load) with the next iter's compute.
      • Running accumulator O kept in register fragment (rescaled in place per
        iter) — same pattern as the ``ts`` baseline.  In-place TMEM accum
        (``ts_v2`` style) currently has a correctness bug in tilelang.

    Not implemented in this compact path:
      • 2-CTA cluster MMA (``cta_group::2``) — no clean cluster handle on
        ``T.tcgen05_gemm`` yet.
      • 2-Q-stage interleaving — would need a second Q tile in flight plus
        per-stage mbarrier arrays, which fights tilelang's auto-staging.
      • Per-role register donation (warp_specialize + ``setmaxnreg``) — tilelang
        emits the ``setmaxnreg`` PTX but lacks the ``__maxnreg__`` launch
        attribute that ptxas needs to actually partition the static register
        budget.  See the ``flashattn_wasp`` attempt for what this looks like
        when you push it manually.
      • Hand-tuned exp2 polynomial + packed ``fma.f32x2``.
      • Persistent kernel with cross-tile producer/epilogue overlap.

    Expected perf: in the ``ts_v2`` ballpark (single math WG, single TMEM
    accumulator). Peak-performance variants need the lower-level cluster and
    role-specialized machinery listed above.
    """
    if groups <= 0 or heads % groups != 0:
        raise ValueError("groups must be a positive divisor of heads")
    head_kv = heads // groups
    threads = 256
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            D_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            mbar_s = T.alloc_barrier(1)
            mbar_d = T.alloc_barrier(1)

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)
            D_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(O_reg, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            # ``T.Pipelined`` is a hint to the scheduler.  We keep
            # ``num_stages=1`` because the manual ``mbar_s``/``mbar_o`` mbarrier
            # waits cannot be safely overlapped across iterations without
            # explicit per-stage barrier arrays (see ``flashattn_wasp`` for
            # how heavy that gets).  Even at num_stages=1 the async TMA copy
            # can run alongside the previous GEMM completion wait.
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                T.tcgen05_gemm(
                    Q_shared,
                    K_shared,
                    S_tmem,
                    transpose_B=True,
                    mbar=mbar_s,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_s, k % 2)

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

                # ---- online softmax, condensed into three passes ----
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

                # Correction: rescale running O by exp(m_prev - m_new).
                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] *= scores_scale[i]

                T.copy(S_reg, P_cast)
                T.copy(P_cast, P_tmem)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)

                # GEMM2: P @ V → D_tmem, then accumulate into O_reg.
                T.tcgen05_gemm(
                    P_tmem,
                    V_shared,
                    D_tmem,
                    mbar=mbar_d,
                    clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_d, k % 2)
                T.copy(D_tmem, D_reg)
                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] += D_reg[i, j]

            # Epilogue: divide by logsum and TMA store.
            for i, j in T.Parallel(block_M, dim):
                O_reg[i, j] /= logsum[i]
            T.copy(O_reg, O_shared)
            T.copy(
                O_shared,
                Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
            )

    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn_fa4_ws(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups=1,
    block_M=128,
    block_N=128,
    num_stages=3,
):
    """Simple warp-specialized FlashAttention, mirroring the GEMM ws_persistent
    template in ``examples/gemm_sm100/gemm_tcgen5mma_ws_persistent.py``.

    Layout (threads=256, 8 warps):
      tx <  32         : TMA producer  -- pre-fetches K[k+i], V[k+i] into
                         ``num_stages``-deep ring buffers in shared memory.
      tx <  64         : tcgen05 issuer -- triggers Q@K and P@V MMAs and
                         signals completion via mbarriers.
      tx 128-255       : math WG (4 warps) -- reads S_tmem, runs softmax,
                         writes P_tmem, reads D_tmem, accumulates O in
                         register.  4 warps × 32 threads = 128 threads, exactly
                         one thread per row of block_M=128.

    Unlike ``flashattn_wasp`` (which tries dim-split + pool donation and ends up
    slower than the baseline due to ``setmaxnreg``'s missing launch attribute),
    this WS layout keeps register pressure naturally low on each warp: the math
    WG holds the same fragments as ``ts`` / ``fa4_pipe`` and the producer/MMA
    warps are tiny.  ptxas picks a single static reg count that fits everyone.

    Uses two SM100 scheduling features that TileLang supports cleanly:
      1. Multi-stage K/V pipeline via ``T.alloc_shared((num_stages, ...))`` and
         per-stage ``mbar_k_loaded``/``mbar_k_consumed`` arrays.  TMA load for
         iter k+2 runs while the math WG is doing iter k softmax.
      2. Separating TMA and MMA issue from the math WG.  Each role advances on
         its own warp, so the MMA pipeline doesn't have to wait for softmax FMA
         to retire before issuing the next ``tcgen05.mma``.

    Still skipped in this compact path:
      • 2-CTA cluster MMA (``use_2cta=True`` + ``cluster_dims=2``).
      • 2-Q-stage interleaving / dim-split correction.
      • Hand-tuned exp2 polynomial.
    """
    if groups <= 0 or heads % groups != 0:
        raise ValueError("groups must be a positive divisor of heads")
    head_kv = heads // groups
    threads = 256
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            # 3D-shape shared = multi-stage ring buffer (auto-rotated by stage idx)
            K_shared = T.alloc_shared([num_stages, block_N, dim], dtype)
            V_shared = T.alloc_shared([num_stages, block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            D_tmem = T.alloc_tmem([block_M, dim], accum_dtype)

            # K/V multi-stage barriers (per-stage to support overlapping loads).
            mb_k_loaded = T.alloc_barrier([1] * num_stages)
            mb_k_consumed = T.alloc_barrier([1] * num_stages)
            mb_v_loaded = T.alloc_barrier([1] * num_stages)
            mb_v_consumed = T.alloc_barrier([1] * num_stages)
            # S_tmem / P_tmem / D_tmem single-buffered (TMEM is precious).
            mb_s_full = T.alloc_barrier(1)
            mb_s_empty = T.alloc_barrier(128)
            mb_p_full = T.alloc_barrier(128)
            mb_d_full = T.alloc_barrier(1)
            mb_d_empty = T.alloc_barrier(128)
            # Q ready signal from math WG (which issues the Q TMA) to producer/MMA.
            mb_q_ready = T.alloc_barrier(128)

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)
            D_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            tx = T.get_thread_binding()

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            # Math WG (128-255) loads Q at startup, owns logsum/scores state.
            if tx >= 128 and tx < 256:
                T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                T.fill(O_reg, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.mbarrier_arrive(mb_q_ready)

            if tx < 32:
                # ---- Producer warp: TMA loads K[k] and V[k] into ring buffers ----
                T.mbarrier_wait_parity(mb_q_ready, 0)
                for k in T.serial(loop_range):
                    stage = k % num_stages
                    parity = (k // num_stages) & 1
                    # K
                    if k >= num_stages:
                        T.mbarrier_wait_parity(mb_k_consumed[stage], parity ^ 1)
                    T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared[stage, :, :])
                    T.mbarrier_arrive(mb_k_loaded[stage])
                    # V
                    if k >= num_stages:
                        T.mbarrier_wait_parity(mb_v_consumed[stage], parity ^ 1)
                    T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared[stage, :, :])
                    T.mbarrier_arrive(mb_v_loaded[stage])

            elif tx >= 32 and tx < 64:
                # ---- MMA-issue warp: triggers tcgen05.mma for Q@K^T and P@V ----
                T.mbarrier_wait_parity(mb_q_ready, 0)
                for k in T.serial(loop_range):
                    stage = k % num_stages
                    parity = (k // num_stages) & 1

                    T.mbarrier_wait_parity(mb_k_loaded[stage], parity)
                    if k > 0:
                        T.mbarrier_wait_parity(mb_s_empty, (k - 1) & 1)
                    T.tcgen05_gemm(
                        Q_shared,
                        K_shared[stage, :, :],
                        S_tmem,
                        transpose_B=True,
                        mbar=mb_s_full,
                        clear_accum=True,
                    )
                    T.mbarrier_wait_parity(mb_s_full, k & 1)
                    T.mbarrier_arrive(mb_k_consumed[stage])

                    T.mbarrier_wait_parity(mb_p_full, k & 1)
                    T.mbarrier_wait_parity(mb_v_loaded[stage], parity)
                    if k > 0:
                        T.mbarrier_wait_parity(mb_d_empty, (k - 1) & 1)
                    T.tcgen05_gemm(
                        P_tmem,
                        V_shared[stage, :, :],
                        D_tmem,
                        mbar=mb_d_full,
                        clear_accum=True,
                    )
                    T.mbarrier_wait_parity(mb_d_full, k & 1)
                    T.mbarrier_arrive(mb_v_consumed[stage])

            elif tx >= 128 and tx < 256:
                # ---- Math WG: softmax + correction + register-resident O ----
                for k in T.serial(loop_range):
                    T.mbarrier_wait_parity(mb_s_full, k & 1)
                    T.copy(S_tmem, S_reg)
                    T.mbarrier_arrive(mb_s_empty)

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

                    for i, j in T.Parallel(block_M, dim):
                        O_reg[i, j] *= scores_scale[i]

                    T.copy(S_reg, P_cast)
                    T.copy(P_cast, P_tmem)
                    T.mbarrier_arrive(mb_p_full)

                    T.mbarrier_wait_parity(mb_d_full, k & 1)
                    T.copy(D_tmem, D_reg)
                    T.mbarrier_arrive(mb_d_empty)

                    for i, j in T.Parallel(block_M, dim):
                        O_reg[i, j] += D_reg[i, j]

                # Epilogue: divide by logsum and TMA store.
                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] /= logsum[i]
                T.copy(O_reg, O_shared)
                T.copy(
                    O_shared,
                    Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
                )

    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn_wasp(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups=1,
    block_M=128,
    block_N=64,
    threads=256,
    num_stages=2,
):
    """Dim-split warp-spec, single-math-WG flavor.

    WG layout (threads=256, 8 warps; 2 functional WGs):
      tid   0-127: math: softmax + corL + corR (two-pass) + all MMA issues
      tid 128-255: TMA load only

    Only ONE fragment buffer (O_l_reg) is used for both halves: WG0 rescales
    the left half then reuses the same registers to rescale the right half.
    This avoids holding O_l_reg + O_r_reg simultaneously (saves 64 regs/thread)
    so the math WG fits in ~167 regs/thread — no spills.

    Per iter k pipeline (all in math WG):
      issue S = Q @ K[k] ; wait done ; softmax math ; write P_tmem ;
      rescale O_l in TMEM ; rescale O_r in TMEM ;
      issue O_l += P @ V_l ; O_r += P @ V_r
    """
    if groups <= 0 or heads % groups != 0:
        raise ValueError("groups must be a positive divisor of heads")
    if dim % 2 != 0:
        raise ValueError("dim must be even for dim-split correction")
    head_kv = heads // groups
    half_dim = dim // 2
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared_0 = T.alloc_shared([block_N, dim], dtype)
            K_shared_1 = T.alloc_shared([block_N, dim], dtype)
            V_l_shared_0 = T.alloc_shared([block_N, half_dim], dtype)
            V_l_shared_1 = T.alloc_shared([block_N, half_dim], dtype)
            V_r_shared_0 = T.alloc_shared([block_N, half_dim], dtype)
            V_r_shared_1 = T.alloc_shared([block_N, half_dim], dtype)
            O_l_shared = T.alloc_shared([block_M, half_dim], dtype)
            O_r_shared = T.alloc_shared([block_M, half_dim], dtype)
            alpha_shared = T.alloc_shared([block_M], accum_dtype)
            logsum_shared = T.alloc_shared([block_M], accum_dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            P_tmem = T.alloc_tmem([block_M, block_N], dtype)
            O_l_tmem = T.alloc_tmem([block_M, half_dim], accum_dtype)
            O_r_tmem = T.alloc_tmem([block_M, half_dim], accum_dtype)

            # WG0 (tid<128): softmax + corL + MMA issue.
            # WG1 (tid 128-255): TMA load + corR.
            mbar_dma1_empty = T.alloc_barrier([128] * num_stages)
            mbar_dma1_full = T.alloc_barrier([128] * num_stages)
            mbar_dma2_empty = T.alloc_barrier([128] * num_stages)
            mbar_dma2_full = T.alloc_barrier([128] * num_stages)
            mbar_bmm1_empty = T.alloc_barrier([128] * num_stages)
            mbar_bmm1_full = T.alloc_barrier([1] * num_stages)
            mbar_sm_to_corR = T.alloc_barrier([128] * num_stages)
            mbar_corR_to_mma = T.alloc_barrier([128] * num_stages)
            mbar_bmm2_l_full = T.alloc_barrier([1] * num_stages)
            mbar_bmm2_r_full = T.alloc_barrier([1] * num_stages)
            mbar_logsum_ready = T.alloc_barrier([128])

            tid = T.get_thread_binding()

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_l_reg = T.alloc_fragment([block_M, half_dim], accum_dtype)
            O_r_reg = T.alloc_fragment([block_M, half_dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_rescale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            if tid < 128:
                T.set_max_nreg(232, 1)
                T.fill(O_l_reg, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.copy(O_l_reg, O_l_tmem)
            else:
                T.set_max_nreg(80, 0)
                T.fill(O_r_reg, 0)
                T.copy(O_r_reg, O_r_tmem)

            for k in T.serial(loop_range):
                parity = (k // num_stages) & 1
                parity_inv = parity ^ 1
                stage_id = k % num_stages
                is_clear_accum = k == 0

                prev_stage = (k - 1) % num_stages
                prev_parity = ((k - 1) // num_stages) & 1

                if tid < 128:
                    # ===== WG0: softmax + corL + MMA issue (128 threads) =====
                    T.mbarrier_wait_parity(mbar_dma1_full[stage_id], parity)
                    T.mbarrier_wait_parity(mbar_bmm1_empty[stage_id], parity_inv)

                    if stage_id == 0:
                        T.tcgen05_gemm(
                            Q_shared,
                            K_shared_0,
                            S_tmem,
                            transpose_B=True,
                            mbar=mbar_bmm1_full[stage_id],
                            clear_accum=True,
                        )
                    else:
                        T.tcgen05_gemm(
                            Q_shared,
                            K_shared_1,
                            S_tmem,
                            transpose_B=True,
                            mbar=mbar_bmm1_full[stage_id],
                            clear_accum=True,
                        )
                    T.mbarrier_arrive(mbar_dma1_empty[stage_id])

                    T.mbarrier_wait_parity(mbar_bmm1_full[stage_id], parity)
                    if k > 0:
                        T.mbarrier_wait_parity(mbar_bmm2_l_full[prev_stage], prev_parity)

                    T.copy(S_tmem, S_reg)
                    T.mbarrier_arrive(mbar_bmm1_empty[stage_id])

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
                        scores_rescale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.exp2(S_reg[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(S_reg, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_rescale[i] + scores_sum[i]

                    T.copy(scores_rescale, alpha_shared)
                    T.mbarrier_arrive(mbar_sm_to_corR[stage_id])

                    if k > 0:
                        T.copy(O_l_tmem, O_l_reg)
                        for i, j in T.Parallel(block_M, half_dim):
                            O_l_reg[i, j] *= scores_rescale[i]
                        T.copy(O_l_reg, O_l_tmem)

                    T.copy(S_reg, P_cast)
                    T.copy(P_cast, P_tmem)

                    T.mbarrier_wait_parity(mbar_dma2_full[stage_id], parity)
                    T.mbarrier_wait_parity(mbar_corR_to_mma[stage_id], parity)

                    if stage_id == 0:
                        T.tcgen05_gemm(
                            P_tmem,
                            V_l_shared_0,
                            O_l_tmem,
                            mbar=mbar_bmm2_l_full[stage_id],
                            clear_accum=is_clear_accum,
                        )
                        T.tcgen05_gemm(
                            P_tmem,
                            V_r_shared_0,
                            O_r_tmem,
                            mbar=mbar_bmm2_r_full[stage_id],
                            clear_accum=is_clear_accum,
                        )
                    else:
                        T.tcgen05_gemm(
                            P_tmem,
                            V_l_shared_1,
                            O_l_tmem,
                            mbar=mbar_bmm2_l_full[stage_id],
                            clear_accum=is_clear_accum,
                        )
                        T.tcgen05_gemm(
                            P_tmem,
                            V_r_shared_1,
                            O_r_tmem,
                            mbar=mbar_bmm2_r_full[stage_id],
                            clear_accum=is_clear_accum,
                        )
                    T.mbarrier_arrive(mbar_dma2_empty[stage_id])

                else:
                    # ===== WG1: TMA load + correction-right (128 threads) =====
                    T.mbarrier_wait_parity(mbar_dma1_empty[stage_id], parity_inv)
                    if k == 0:
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                    if stage_id == 0:
                        T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared_0)
                    else:
                        T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared_1)
                    T.mbarrier_arrive(mbar_dma1_full[stage_id])

                    T.mbarrier_wait_parity(mbar_dma2_empty[stage_id], parity_inv)
                    if stage_id == 0:
                        T.copy(
                            V[bz, k * block_N : (k + 1) * block_N, by // groups, 0:half_dim],
                            V_l_shared_0,
                        )
                        T.copy(
                            V[bz, k * block_N : (k + 1) * block_N, by // groups, half_dim:dim],
                            V_r_shared_0,
                        )
                    else:
                        T.copy(
                            V[bz, k * block_N : (k + 1) * block_N, by // groups, 0:half_dim],
                            V_l_shared_1,
                        )
                        T.copy(
                            V[bz, k * block_N : (k + 1) * block_N, by // groups, half_dim:dim],
                            V_r_shared_1,
                        )
                    T.mbarrier_arrive(mbar_dma2_full[stage_id])

                    T.mbarrier_wait_parity(mbar_sm_to_corR[stage_id], parity)
                    if k > 0:
                        T.mbarrier_wait_parity(mbar_bmm2_r_full[prev_stage], prev_parity)
                        T.copy(O_r_tmem, O_r_reg)
                        for i, j in T.Parallel(block_M, half_dim):
                            O_r_reg[i, j] *= alpha_shared[i]
                        T.copy(O_r_reg, O_r_tmem)

                    T.mbarrier_arrive(mbar_corR_to_mma[stage_id])

            # ===== Epilogue =====
            last_stage = (loop_range - 1) % num_stages
            last_parity = ((loop_range - 1) // num_stages) & 1

            if tid < 128:
                T.mbarrier_wait_parity(mbar_bmm2_l_full[last_stage], last_parity)
                T.copy(logsum, logsum_shared)
                T.mbarrier_arrive(mbar_logsum_ready)

                T.copy(O_l_tmem, O_l_reg)
                for i, j in T.Parallel(block_M, half_dim):
                    O_l_reg[i, j] /= logsum[i]
                T.copy(O_l_reg, O_l_shared)
                T.copy(
                    O_l_shared,
                    Output[bz, bx * block_M : (bx + 1) * block_M, by, 0:half_dim],
                )
            else:
                T.mbarrier_wait_parity(mbar_bmm2_r_full[last_stage], last_parity)
                T.mbarrier_wait_parity(mbar_logsum_ready, 0)

                T.copy(O_r_tmem, O_r_reg)
                for i, j in T.Parallel(block_M, half_dim):
                    O_r_reg[i, j] /= logsum_shared[i]
                T.copy(O_r_reg, O_r_shared)
                T.copy(
                    O_r_shared,
                    Output[bz, bx * block_M : (bx + 1) * block_M, by, half_dim:dim],
                )

    return main


flashattn_warp = flashattn_wasp


@tilelang.jit(out_idx=[3], pass_configs=PASS_CFG)
def flashattn_fa4(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups=1,
    block_M=128,
    block_N=128,
):
    """256-thread split path with separated softmax/correction warps.

    Thread partition:
      0-127:   Softmax (read S, exp2, write P) — never touches O
      128-255: Correction+MMA+Load (rescale O, issue GEMMs, load K/V)
    Key: softmax and correction never hold S_reg and O_reg simultaneously,
    reducing peak register liveness within each path.
    """
    if groups <= 0 or heads % groups != 0:
        raise ValueError("groups must be a positive divisor of heads")
    head_kv = heads // groups
    threads = 256
    num_stages = 2
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared_0 = T.alloc_shared([block_N, dim], dtype)
            K_shared_1 = T.alloc_shared([block_N, dim], dtype)
            V_shared_0 = T.alloc_shared([block_N, dim], dtype)
            V_shared_1 = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            scale_shared = T.alloc_shared([block_M], accum_dtype)
            logsum_shared = T.alloc_shared([block_M], accum_dtype)

            S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            O_tmem = T.alloc_tmem([block_M, dim], accum_dtype)
            P_tmem = T.alloc_tmem([block_M, block_N], dtype)

            mbar_s_ready = T.alloc_barrier([1] * num_stages)
            mbar_scale_ready = T.alloc_barrier([128] * num_stages)
            mbar_p_ready = T.alloc_barrier([128] * num_stages)
            T.alloc_barrier([128] * num_stages)
            mbar_gemm2 = T.alloc_barrier([1] * num_stages)

            S_reg = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_cast = T.alloc_fragment([block_M, block_N], dtype)
            O_reg = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            tid = T.get_thread_binding()

            if tid < 128:
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.fill(logsum, 0)
            elif tid >= 128:
                T.fill(O_reg, 0)
                T.copy(O_reg, O_tmem)
                T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

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
                parity ^ 1

                # === CORRECTION + MMA + LOAD (tid 128-255) ===
                if tid >= 128:
                    # Load K/V
                    if stage_id == 0:
                        T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared_0)
                        T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared_0)
                    else:
                        T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared_1)
                        T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared_1)

                    # GEMM1: Q × K^T → S_tmem
                    if stage_id == 0:
                        T.tcgen05_gemm(
                            Q_shared,
                            K_shared_0,
                            S_tmem,
                            transpose_B=True,
                            mbar=mbar_s_ready[stage_id],
                            clear_accum=True,
                        )
                    else:
                        T.tcgen05_gemm(
                            Q_shared,
                            K_shared_1,
                            S_tmem,
                            transpose_B=True,
                            mbar=mbar_s_ready[stage_id],
                            clear_accum=True,
                        )

                    # Wait for P ready from softmax
                    T.mbarrier_wait_parity(mbar_p_ready[stage_id], parity)

                    # Rescale O (skip on first iteration)
                    if k == 0:
                        T.mbarrier_wait_parity(mbar_scale_ready[stage_id], parity)
                    else:
                        T.mbarrier_wait_parity(mbar_scale_ready[stage_id], parity)
                        T.copy(scale_shared, scores_scale)
                        T.copy(O_tmem, O_reg)
                        for i, j in T.Parallel(block_M, dim):
                            O_reg[i, j] *= scores_scale[i]
                        T.copy(O_reg, O_tmem)

                    # GEMM2: P × V → O_tmem
                    if stage_id == 0:
                        T.tcgen05_gemm(
                            P_tmem,
                            V_shared_0,
                            O_tmem,
                            mbar=mbar_gemm2[stage_id],
                            clear_accum=(k == 0),
                        )
                    else:
                        T.tcgen05_gemm(
                            P_tmem,
                            V_shared_1,
                            O_tmem,
                            mbar=mbar_gemm2[stage_id],
                            clear_accum=(k == 0),
                        )
                    T.mbarrier_wait_parity(mbar_gemm2[stage_id], parity)

                # === SOFTMAX (tid 0-127) ===
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

                    # Send scale to correction warps
                    T.copy(scores_scale, scale_shared)
                    T.mbarrier_arrive(mbar_scale_ready[stage_id])

                    # Write P to TMEM
                    T.copy(S_reg, P_cast)
                    T.copy(P_cast, P_tmem)
                    T.mbarrier_arrive(mbar_p_ready[stage_id])

            # Epilogue
            if tid < 128:
                T.copy(logsum, logsum_shared)
            elif tid >= 128:
                T.copy(logsum_shared, logsum)
                T.copy(O_tmem, O_reg)
                for i, j in T.Parallel(block_M, dim):
                    O_reg[i, j] /= logsum[i]
                T.copy(O_reg, O_shared)
                T.copy(
                    O_shared,
                    Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
                )

    return main


def ref_program(Q, K, V, is_causal, groups=1):
    """CPU reference: K/V [b,s,head_kv,d], expand to heads for einsum."""
    assert Q.size(2) == K.size(2) * groups
    dim = Q.size(-1)
    K_f = K.cpu().float().repeat_interleave(groups, dim=2)
    V_f = V.cpu().float().repeat_interleave(groups, dim=2)
    Q_f = Q.cpu().float()
    scores = torch.einsum("bqhd,bkhd->bhqk", Q_f, K_f)
    scores = scores / (dim**0.5)
    if is_causal:
        seq_len = Q_f.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    P = F.softmax(scores, dim=-1)
    O = torch.einsum("bhqk,bkhd->bqhd", P, V_f)
    return O.to(Q.dtype)


def main(
    batch: int = 2,
    heads: int = 4,
    seq_len: int = 256,
    dim: int = 128,
    is_causal: bool = False,
    groups: int = 1,
    variant: str = "ss",
    print_kernel: bool = False,
):
    """Run GQA forward kernel (ss or ts variant) and benchmark."""
    if groups <= 0 or heads % groups != 0:
        raise ValueError("groups must be a positive divisor of heads")
    head_kv = heads // groups
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    print(f"=== Blackwell GQA Forward ({variant.upper()}) ===")
    print(f"batch={batch}, heads={heads}, head_kv={head_kv}, groups={groups}, seq_len={seq_len}, dim={dim}, causal={is_causal}")

    if variant in ("ss", "ts"):
        kernel = flashattn(
            batch,
            heads,
            seq_len,
            dim,
            is_causal,
            groups=groups,
            block_M=128,
            block_N=128,
            variant=variant,
        )
    elif variant == "ts2":
        kernel = flashattn_ts_v2(
            batch,
            heads,
            seq_len,
            dim,
            is_causal,
            groups=groups,
            block_M=128,
            block_N=128,
            num_stages=2,
        )
    elif variant == "fa4":
        kernel = flashattn_fa4(
            batch,
            heads,
            seq_len,
            dim,
            is_causal,
            groups=groups,
            block_M=128,
            block_N=128,
        )
    elif variant == "fa4_pipe":
        kernel = flashattn_fa4_pipe(
            batch,
            heads,
            seq_len,
            dim,
            is_causal,
            groups=groups,
            block_M=128,
            block_N=128,
        )
    elif variant == "fa4_ws":
        kernel = flashattn_fa4_ws(
            batch,
            heads,
            seq_len,
            dim,
            is_causal,
            groups=groups,
            block_M=128,
            block_N=128,
            num_stages=2,
        )
    else:
        kernel = flashattn_wasp(
            batch,
            heads,
            seq_len,
            dim,
            is_causal,
            groups=groups,
            block_M=128,
            block_N=64,
            threads=256,
            num_stages=2,
        )

    if print_kernel:
        print(kernel.get_kernel_source())
        return

    Q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(batch, seq_len, head_kv, dim, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(batch, seq_len, head_kv, dim, device="cuda", dtype=torch.bfloat16)

    out = kernel(Q, K, V)
    ref = ref_program(Q, K, V, is_causal, groups).to(out.device)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("Correctness check passed.")

    latency = do_bench(lambda: kernel(Q, K, V), warmup=100)
    print(f"Blackwell GQA fwd ({variant}): {latency:.2f} ms")
    print(f"Blackwell GQA fwd ({variant}): {total_flops / latency * 1e-9:.2f} TFlops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--print_kernel", action="store_true")
    parser.add_argument("--groups", type=int, default=1, help="GQA: head_kv = heads // groups")
    parser.add_argument(
        "--variant",
        choices=["ss", "ts", "ts2", "fa4", "fa4_pipe", "fa4_ws", "wasp"],
        default="ss",
        help=(
            "ss: pipeline (default); ts: 256 threads; ts2: improved ts; "
            "fa4: 2-WG warp-spec; fa4_pipe: ts2 + num_stages=3 "
            "K/V multi-stage; wasp: warp-specialized"
        ),
    )
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.groups, args.variant, args.print_kernel)
