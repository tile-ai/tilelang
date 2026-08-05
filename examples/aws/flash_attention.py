import argparse

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}


@T.macro
def get_num_waves(block_id, num_blocks, sm_num):
    return T.ceildiv(num_blocks - block_id, sm_num)


# The FA4 static persistent scheduler's tile order: the q block index is the
# minor dimension, then head, then batch (flash_attn/cute/tile_scheduler.py,
# StaticPersistentTileScheduler.get_current_work).
@T.macro
def get_attn_tile(block_id, wave_id, sm_num, q_blocks, heads):
    tile_id = block_id + wave_id * sm_num
    bx = tile_id % q_blocks
    hn = tile_id // q_blocks
    by = hn % heads
    bz = hn // heads
    return bx, by, bz


@tilelang.jit(pass_configs=PASS_CFG)
def flash_attention(
    Q,
    K,
    V,
    block_M,
    block_N,
    store_block_N,
    dim,
    dtype,
    accum_dtype,
    num_stages,
    num_persistent_stages,
):
    batch, seq_len, heads = T.const("batch, seq_len, heads")

    Q: T.Tensor((batch, seq_len, heads, dim), dtype)
    K: T.Tensor((batch, seq_len, heads, dim), dtype)
    V: T.Tensor((batch, seq_len, heads, dim), dtype)
    Output = T.empty((batch, seq_len, heads, dim), dtype)

    q_blocks = T.ceildiv(seq_len, block_M)
    loop_range = T.ceildiv(seq_len, block_N)
    num_blocks = q_blocks * heads * batch
    sm_num = driver.get_num_sms()
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    assert dim == 128
    assert block_M == 128
    assert block_N == 128
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    assert dim % store_block_N == 0

    with T.Kernel(sm_num, threads=128) as block_id:
        Q_shared = T.alloc_shared((block_M, dim), dtype)
        K_shared = T.alloc_shared((block_N, dim), dtype)
        V_shared = T.alloc_shared((block_N, dim), dtype)
        O_shared = T.alloc_shared((block_M, dim), dtype)
        scale_shared = T.alloc_shared((block_M,), accum_dtype)
        logsum_shared = T.alloc_shared((block_M,), accum_dtype)

        S_tmem = T.alloc_tmem((block_M, block_N), accum_dtype)
        P_tmem = T.alloc_tmem((block_M, block_N), dtype)
        O_tmem = T.alloc_tmem((block_M, dim), accum_dtype)

        tid = T.get_thread_binding()

        S_reg = T.alloc_fragment((block_M, block_N), accum_dtype)
        P_cast = T.alloc_fragment((block_M, block_N), dtype)
        scores_max = T.alloc_fragment((block_M,), accum_dtype)
        scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
        scores_rescale = T.alloc_fragment((block_M,), accum_dtype)
        scores_sum = T.alloc_fragment((block_M,), accum_dtype)
        logsum = T.alloc_fragment((block_M,), accum_dtype)
        O_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
        inv_sum = T.alloc_fragment((block_M,), accum_dtype)

        # The schedulable task is the tile op; instructions reference the
        # stable ids attached in the kernel body (a T.WSID entry in the
        # annotations= of tile ops and loops — every statement of a
        # scope carries its own id). The pass derives read/write sets
        # itself; the schedule carries no metadata.
        #
        # Four roles (an FA4-style split, minus 2-CTA):
        #  - Softmax owns the S -> P transform and the running statistics.
        #    It hands the per-row rescale factor to Correction through the
        #    "scale" pipeline immediately after row-max, and publishes P
        #    before the row-sum, so neither O correction nor the reduction
        #    sits on the tensor-core critical path.
        #  - Correction owns the O accumulator: the per-iteration rescale
        #    (skipped entirely on the stale-max fast path) and the final
        #    normalize + store, in store_block_N-column TMEM slices.
        #  - TMA feeds Q/K/V; MMA issues both GEMMs.
        #
        # The kernel is persistent, following FA4's static persistent
        # scheduler: one CTA per SM, each looping over its q tiles (block
        # minor, then head, then batch). The q pipeline has
        # num_persistent_stages versions so the next wave's Q load overlaps
        # the current wave.
        #
        # Sync stages express intra-role software pipelining: acquire/commit
        # (and wait/release) of a pipeline pair up at one stage, and the
        # instructions between them run at that stage's iteration offset.
        # MMA's PV-side spans sit at stage 1, one iteration behind its QK
        # span: the materialized MMA loop runs an extra iteration and issues
        # QK(i) before PV(i-1), so the tensor core computes S(i) while
        # Softmax is still turning S(i-1) into P(i-1).
        #
        # The acc pipeline cycles loop_range + 1 times per wave on both
        # sides: Correction's per-iteration rescales plus its final
        # normalize, matched by MMA's PV consumes plus one wave-level
        # consume pairing the normalize commit. Without the extra consume
        # the full/empty parity would diverge from the second wave on. Both
        # roles touch acc at two loop depths, so the pass tracks their acc
        # phases with runtime counters.
        T.annotate_ws_schedule(
            T.WSSchedule(
                num_warps=12,
                roles=[
                    T.WSRole("Softmax", warps_lo=0, warps_hi=4, max_nreg=224),
                    T.WSRole("Correction", warps_lo=4, warps_hi=8, max_nreg=80),
                    T.WSRole("TMA", warps_lo=8, warps_hi=9, max_nreg=40),
                    T.WSRole("MMA", warps_lo=9, warps_hi=10, max_nreg=40),
                    # Warps 10-11 are unassigned register donors.
                ],
                pipelines=[
                    T.WSPipeline("q", [Q_shared], depth=num_persistent_stages),
                    T.WSPipeline("k", [K_shared], depth=num_stages),
                    T.WSPipeline("v", [V_shared], depth=num_stages),
                    T.WSPipeline("score", [S_tmem], depth=1),
                    T.WSPipeline("prob", [P_tmem], depth=1),
                    T.WSPipeline("acc", [O_tmem], depth=1),
                    # Softmax -> Correction rescale handoff.
                    T.WSPipeline("scale", [scale_shared], depth=num_stages),
                    # Softmax -> Correction logsum handoff for the epilogue.
                    T.WSPipeline("stats", [logsum_shared], depth=1),
                ],
                scopes=[
                    T.WSScope(
                        "loop_kv",
                        {
                            "TMA": [
                                T.WSSync.producer_acquire("k"),
                                "copy_K_g2s",
                                T.WSSync.producer_commit("k"),
                                T.WSSync.producer_acquire("v"),
                                "copy_V_g2s",
                                T.WSSync.producer_commit("v"),
                            ],
                            "MMA": [
                                T.WSSync.consumer_wait("k", stage=0),
                                T.WSSync.producer_acquire("score", stage=0),
                                "gemm_QK",
                                T.WSSync.producer_commit("score", stage=0),
                                T.WSSync.consumer_release("k", stage=0),
                                # The PV spans run at stage 1: this loop
                                # iteration issues PV of the PREVIOUS kv
                                # step, after QK of the current one.
                                T.WSSync.consumer_wait("prob", stage=1),
                                T.WSSync.consumer_wait("v", stage=1),
                                T.WSSync.consumer_wait("acc", stage=1),
                                "gemm_PV",
                                T.WSSync.consumer_release("prob", stage=1),
                                T.WSSync.consumer_release("v", stage=1),
                                T.WSSync.consumer_release("acc", stage=1),
                            ],
                            "Softmax": [
                                T.WSSync.consumer_wait("score"),
                                "copy_S_t2r",
                                T.WSSync.consumer_release("score"),
                                "copy_max_prev",
                                "reduce_max",
                                "stale_max",
                                # Publish the rescale right after row-max so
                                # Correction overlaps with exp below.
                                T.WSSync.producer_acquire("scale"),
                                "copy_scale_s",
                                T.WSSync.producer_commit("scale"),
                                "exp_scale",
                                "softmax_exp",
                                # Publish P before row-sum: the reduction is
                                # not a PV dependency.
                                T.WSSync.producer_acquire("prob"),
                                "copy_P_cast",
                                "copy_P_r2t",
                                T.WSSync.producer_commit("prob"),
                                "reduce_sum",
                                "update_logsum",
                            ],
                            "Correction": [
                                T.WSSync.consumer_wait("scale"),
                                "rescale_vote",
                                # Correction produces the rescaled O that
                                # MMA's PV then accumulates onto.
                                T.WSSync.producer_acquire("acc"),
                                "rescale_O",
                                T.WSSync.producer_commit("acc"),
                                T.WSSync.consumer_release("scale"),
                            ],
                        },
                    ),
                    # The persistent loop over this CTA's q tiles.
                    T.WSScope(
                        "loop_wave_id",
                        {
                            "TMA": [
                                T.WSSync.producer_acquire("q"),
                                "copy_Q_g2s",
                                T.WSSync.producer_commit("q"),
                                "loop_kv",
                            ],
                            "MMA": [
                                T.WSSync.consumer_wait("q"),
                                "loop_kv",
                                T.WSSync.consumer_release("q"),
                                # Consume the version produced by
                                # Correction's normalize commit, keeping
                                # acc's per-wave cycle counts equal.
                                T.WSSync.consumer_wait("acc"),
                                T.WSSync.consumer_release("acc"),
                            ],
                            "Softmax": [
                                "init_max",
                                "init_logsum",
                                "loop_kv",
                                T.WSSync.producer_acquire("stats"),
                                "copy_logsum_s",
                                T.WSSync.producer_commit("stats"),
                            ],
                            "Correction": [
                                "loop_kv",
                                T.WSSync.consumer_wait("stats"),
                                "copy_inv_sum",
                                "recip_sum",
                                T.WSSync.producer_acquire("acc"),
                                "normalize_O",
                                T.WSSync.producer_commit("acc"),
                                T.WSSync.consumer_release("stats"),
                                "copy_O_s2g",
                            ],
                        },
                    ),
                    T.WSScope(
                        T.WSScope.ROOT,
                        {
                            "TMA": ["loop_wave_id"],
                            "MMA": ["loop_wave_id"],
                            "Softmax": ["loop_wave_id"],
                            "Correction": ["loop_wave_id"],
                        },
                    ),
                ],
            )
        )

        num_waves = get_num_waves(block_id, num_blocks, sm_num)

        for wave_id in T.Pipelined(
            num_waves,
            num_stages=num_persistent_stages,
            annotations={T.WSID: "loop_wave_id"},
        ):
            bx, by, bz = get_attn_tile(block_id, wave_id, sm_num, q_blocks, heads)

            T.copy(
                Q[bz, bx * block_M : (bx + 1) * block_M, by, :],
                Q_shared,
                annotations={T.WSID: "copy_Q_g2s"},
            )
            T.fill(scores_max, -T.infinity(accum_dtype), annotations={T.WSID: "init_max"})
            T.fill(logsum, 0, annotations={T.WSID: "init_logsum"})

            for k in T.Pipelined(loop_range, num_stages=1, annotations={T.WSID: "loop_kv"}):
                T.copy(
                    K[bz, k * block_N : (k + 1) * block_N, by, :],
                    K_shared,
                    annotations={T.WSID: "copy_K_g2s"},
                )
                T.copy(
                    V[bz, k * block_N : (k + 1) * block_N, by, :],
                    V_shared,
                    annotations={T.WSID: "copy_V_g2s"},
                )

                T.gemm(
                    Q_shared,
                    K_shared,
                    S_tmem,
                    transpose_B=True,
                    clear_accum=True,
                    annotations={T.WSID: "gemm_QK"},
                )

                T.copy(S_tmem, S_reg, annotations={T.WSID: "copy_S_t2r"})
                T.copy(scores_max, scores_max_prev, annotations={T.WSID: "copy_max_prev"})
                T.reduce_max(S_reg, scores_max, dim=1, clear=False, annotations={T.WSID: "reduce_max"})
                for i in T.Parallel(block_M, annotations={T.WSID: "stale_max"}):
                    # Stale-max fast path: when the max moves by less
                    # than 8 exponent steps, keep the previous max so
                    # the rescale factor is exactly 1 and Correction can
                    # skip the whole O read-modify-write.
                    scores_rescale[i] = T.if_then_else(
                        (scores_max_prev[i] - scores_max[i]) * scale >= -8.0,
                        1.0,
                        T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                    )
                    scores_max[i] = T.if_then_else(
                        (scores_max_prev[i] - scores_max[i]) * scale >= -8.0,
                        scores_max_prev[i],
                        scores_max[i],
                    )

                T.copy(scores_rescale, scale_shared, annotations={T.WSID: "copy_scale_s"})

                # Warp-local vote: each warp owns 32 rows of O, so it
                # can skip its own read-modify-write when all its
                # rescale factors are 1. At k == 0 the factor is 0 and
                # O_tmem is garbage, but PV(0) clears the accumulator
                # anyway. The vote binds a scalar consumed only by the
                # rescale guard; it guards the op body alone — the
                # schedule's acc/scale sync entries stay unconditional.
                with T.ws_op("rescale_vote"):
                    should_rescale = T.any_sync(scale_shared[tid % block_M] < 1.0)
                if should_rescale != 0:
                    for s in T.unroll(T.ceildiv(dim, store_block_N), annotations={T.WSID: "rescale_O"}):
                        T.copy(
                            O_tmem[:, s * store_block_N : (s + 1) * store_block_N],
                            O_local,
                        )
                        for i, j in T.Parallel(block_M, store_block_N):
                            O_local[i, j] *= scale_shared[i]
                        T.copy(
                            O_local,
                            O_tmem[:, s * store_block_N : (s + 1) * store_block_N],
                        )

                # Affine first (packed f32x2 FMA), exp2 separately.
                for i in T.Parallel(block_M, annotations={T.WSID: "exp_scale"}):
                    for j in T.vectorized(block_N):
                        S_reg[i, j] = S_reg[i, j] * scale + (-scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N, annotations={T.WSID: "softmax_exp"}):
                    S_reg[i, j] = T.exp2(S_reg[i, j])

                T.copy(S_reg, P_cast, annotations={T.WSID: "copy_P_cast"})
                T.copy(P_cast, P_tmem, annotations={T.WSID: "copy_P_r2t"})

                T.reduce_sum(S_reg, scores_sum, dim=1, annotations={T.WSID: "reduce_sum"})
                for i in T.Parallel(block_M, annotations={T.WSID: "update_logsum"}):
                    logsum[i] = logsum[i] * scores_rescale[i] + scores_sum[i]

                T.gemm(
                    P_tmem,
                    V_shared,
                    O_tmem,
                    clear_accum=k == 0,
                    annotations={T.WSID: "gemm_PV"},
                )

            T.copy(logsum, logsum_shared, annotations={T.WSID: "copy_logsum_s"})
            T.copy(logsum_shared, inv_sum, annotations={T.WSID: "copy_inv_sum"})
            # One reciprocal per row, reused across all output slices.
            for i in T.Parallel(block_M, annotations={T.WSID: "recip_sum"}):
                inv_sum[i] = 1.0 / inv_sum[i]
            for s in T.unroll(T.ceildiv(dim, store_block_N), annotations={T.WSID: "normalize_O"}):
                T.copy(
                    O_tmem[:, s * store_block_N : (s + 1) * store_block_N],
                    O_local,
                )
                for i, j in T.Parallel(block_M, store_block_N):
                    O_local[i, j] *= inv_sum[i]
                T.copy(
                    O_local,
                    O_shared[:, s * store_block_N : (s + 1) * store_block_N],
                )
            T.copy(
                O_shared,
                Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
                annotations={T.WSID: "copy_O_s2g"},
            )

    return Output


@tilelang.jit(pass_configs=PASS_CFG)
def flash_attention_ws(
    Q,
    K,
    V,
    block_M,
    block_N,
    store_block_N,
    dim,
    dtype,
    accum_dtype,
    num_stages,
    num_persistent_stages,
):
    batch, seq_len, heads = T.const("batch, seq_len, heads")

    Q: T.Tensor((batch, seq_len, heads, dim), dtype)
    K: T.Tensor((batch, seq_len, heads, dim), dtype)
    V: T.Tensor((batch, seq_len, heads, dim), dtype)
    Output = T.empty((batch, seq_len, heads, dim), dtype)

    q_blocks = T.ceildiv(seq_len, block_M)
    loop_range = T.ceildiv(seq_len, block_N)
    num_blocks = q_blocks * heads * batch
    sm_num = driver.get_num_sms()
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    assert dim == 128
    assert block_M == 128
    assert block_N == 128
    assert seq_len % block_M == 0
    assert seq_len % block_N == 0
    assert dim % store_block_N == 0

    with T.Kernel(sm_num, threads=384) as block_id:
        Q_shared = T.alloc_shared((num_persistent_stages, block_M, dim), dtype)
        K_shared = T.alloc_shared((num_stages, block_N, dim), dtype)
        V_shared = T.alloc_shared((num_stages, block_N, dim), dtype)
        O_shared = T.alloc_shared((block_M, dim), dtype)
        scale_shared = T.alloc_shared((num_stages, block_M), accum_dtype)
        logsum_shared = T.alloc_shared((block_M,), accum_dtype)

        S_tmem = T.alloc_tmem((block_M, block_N), accum_dtype)
        P_tmem = T.alloc_tmem((block_M, block_N), dtype)
        O_tmem = T.alloc_tmem((block_M, dim), accum_dtype)

        q_full = T.alloc_barrier([32] * num_persistent_stages)
        q_empty = T.alloc_barrier([1] * num_persistent_stages)
        k_full = T.alloc_barrier([32] * num_stages)
        k_empty = T.alloc_barrier([1] * num_stages)
        v_full = T.alloc_barrier([32] * num_stages)
        v_empty = T.alloc_barrier([1] * num_stages)
        s_full = T.alloc_barrier([1])
        s_empty = T.alloc_barrier([128])
        prob_full = T.alloc_barrier([128])
        prob_empty = T.alloc_barrier([1])
        acc_full = T.alloc_barrier([128])
        acc_empty = T.alloc_barrier([1])
        scale_full = T.alloc_barrier([128] * num_stages)
        scale_empty = T.alloc_barrier([128] * num_stages)
        stats_full = T.alloc_barrier([128])
        stats_empty = T.alloc_barrier([128])

        tid = T.get_thread_binding()

        S_reg = T.alloc_fragment((block_M, block_N), accum_dtype)
        P_cast = T.alloc_fragment((block_M, block_N), dtype)
        scores_max = T.alloc_fragment((block_M,), accum_dtype)
        scores_max_prev = T.alloc_fragment((block_M,), accum_dtype)
        scores_rescale = T.alloc_fragment((block_M,), accum_dtype)
        scores_sum = T.alloc_fragment((block_M,), accum_dtype)
        logsum = T.alloc_fragment((block_M,), accum_dtype)
        O_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
        inv_sum = T.alloc_fragment((block_M,), accum_dtype)

        if tid < 128:  # warps 0-3: online softmax
            T.set_max_nreg(224, 1)
            num_waves = get_num_waves(block_id, num_blocks, sm_num)
            for wave_id in T.serial(num_waves):
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.fill(logsum, 0)

                for k in T.serial(loop_range):
                    g = wave_id * loop_range + k

                    T.mbarrier_wait_parity(s_full[0], g & 1)
                    T.copy(S_tmem, S_reg)
                    T.mbarrier_arrive(s_empty[0])

                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(S_reg, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        # Stale-max fast path: keep the previous max when it
                        # moves by less than 8 exponent steps, so the
                        # rescale factor is exactly 1.
                        scores_rescale[i] = T.if_then_else(
                            (scores_max_prev[i] - scores_max[i]) * scale >= -8.0,
                            1.0,
                            T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                        )
                        scores_max[i] = T.if_then_else(
                            (scores_max_prev[i] - scores_max[i]) * scale >= -8.0,
                            scores_max_prev[i],
                            scores_max[i],
                        )

                    # Publish the rescale immediately after row-max so the
                    # correction warps overlap with exp / P production.
                    T.mbarrier_wait_parity(scale_empty[g % num_stages], ((g // num_stages) & 1) ^ 1)
                    T.copy(scores_rescale, scale_shared[g % num_stages, :])
                    T.mbarrier_arrive(scale_full[g % num_stages])

                    # Affine first (packed f32x2 FMA), exp2 separately.
                    for i in T.Parallel(block_M):
                        for j in T.vectorized(block_N):
                            S_reg[i, j] = S_reg[i, j] * scale + (-scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.exp2(S_reg[i, j])

                    # Publish P before row-sum: the reduction is not a PV
                    # dependency.
                    T.mbarrier_wait_parity(prob_empty[0], (g & 1) ^ 1)
                    T.copy(S_reg, P_cast)
                    T.copy(P_cast, P_tmem)
                    T.mbarrier_arrive(prob_full[0])

                    T.reduce_sum(S_reg, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_rescale[i] + scores_sum[i]

                T.mbarrier_wait_parity(stats_empty[0], (wave_id & 1) ^ 1)
                T.copy(logsum, logsum_shared)
                T.mbarrier_arrive(stats_full[0])

        elif tid < 256:  # warps 4-7: O correction + epilogue
            T.set_max_nreg(80, 0)
            num_waves = get_num_waves(block_id, num_blocks, sm_num)
            for wave_id in T.serial(num_waves):
                bx, by, bz = get_attn_tile(block_id, wave_id, sm_num, q_blocks, heads)

                for k in T.serial(loop_range):
                    g = wave_id * loop_range + k
                    # acc cycles loop_range + 1 times per wave (the final
                    # normalize is one more producer cycle).
                    c = wave_id * (loop_range + 1) + k

                    T.mbarrier_wait_parity(scale_full[g % num_stages], (g // num_stages) & 1)
                    T.mbarrier_wait_parity(acc_empty[0], (c & 1) ^ 1)
                    # Common case: every rescale factor is exactly 1
                    # (stale-max fast path), and each warp can skip the O
                    # read-modify-write for its own rows. At k == 0 the
                    # factor is 0 and O_tmem is garbage, but PV(0) clears
                    # the accumulator anyway.
                    should_rescale = T.any_sync(scale_shared[g % num_stages, tid % block_M] < 1.0)
                    if should_rescale != 0:
                        for s in T.unroll(T.ceildiv(dim, store_block_N)):
                            T.copy(
                                O_tmem[:, s * store_block_N : (s + 1) * store_block_N],
                                O_local,
                            )
                            for i, j in T.Parallel(block_M, store_block_N):
                                O_local[i, j] *= scale_shared[g % num_stages, i]
                            T.copy(
                                O_local,
                                O_tmem[:, s * store_block_N : (s + 1) * store_block_N],
                            )
                    T.mbarrier_arrive(scale_empty[g % num_stages])
                    T.mbarrier_arrive(acc_full[0])

                c_last = wave_id * (loop_range + 1) + loop_range
                T.mbarrier_wait_parity(stats_full[0], wave_id & 1)
                T.mbarrier_wait_parity(acc_empty[0], (c_last & 1) ^ 1)
                T.copy(logsum_shared, inv_sum)
                # One reciprocal per row, reused across all output slices.
                for i in T.Parallel(block_M):
                    inv_sum[i] = 1.0 / inv_sum[i]
                for s in T.unroll(T.ceildiv(dim, store_block_N)):
                    T.copy(
                        O_tmem[:, s * store_block_N : (s + 1) * store_block_N],
                        O_local,
                    )
                    for i, j in T.Parallel(block_M, store_block_N):
                        O_local[i, j] *= inv_sum[i]
                    T.copy(
                        O_local,
                        O_shared[:, s * store_block_N : (s + 1) * store_block_N],
                    )
                T.mbarrier_arrive(stats_empty[0])
                T.mbarrier_arrive(acc_full[0])
                T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

        elif tid < 288:  # warp 8: TMA
            T.set_max_nreg(40, 0)
            num_waves = get_num_waves(block_id, num_blocks, sm_num)
            for wave_id in T.serial(num_waves):
                bx, by, bz = get_attn_tile(block_id, wave_id, sm_num, q_blocks, heads)

                q_stage = wave_id % num_persistent_stages
                T.mbarrier_wait_parity(q_empty[q_stage], ((wave_id // num_persistent_stages) & 1) ^ 1)
                T.tma_copy(
                    Q[bz, bx * block_M : (bx + 1) * block_M, by, :],
                    Q_shared[q_stage, :, :],
                    barrier=q_full[q_stage],
                )
                T.mbarrier_arrive(q_full[q_stage])

                for k in T.serial(loop_range):
                    g = wave_id * loop_range + k
                    stage = g % num_stages
                    parity_inv = ((g // num_stages) & 1) ^ 1
                    T.mbarrier_wait_parity(k_empty[stage], parity_inv)
                    T.tma_copy(
                        K[bz, k * block_N : (k + 1) * block_N, by, :],
                        K_shared[stage, :, :],
                        barrier=k_full[stage],
                    )
                    T.mbarrier_arrive(k_full[stage])
                    T.mbarrier_wait_parity(v_empty[stage], parity_inv)
                    T.tma_copy(
                        V[bz, k * block_N : (k + 1) * block_N, by, :],
                        V_shared[stage, :, :],
                        barrier=v_full[stage],
                    )
                    T.mbarrier_arrive(v_full[stage])

        elif tid < 320:  # warp 9: tensor-core issue
            T.set_max_nreg(40, 0)
            num_waves = get_num_waves(block_id, num_blocks, sm_num)
            for wave_id in T.serial(num_waves):
                q_stage = wave_id % num_persistent_stages
                T.mbarrier_wait_parity(q_full[q_stage], (wave_id // num_persistent_stages) & 1)
                # PV runs one software-pipeline stage behind QK: QK(i) is
                # issued before PV(i-1), so the tensor core computes S(i)
                # while softmax is still turning S(i-1) into P(i-1).
                for i in T.serial(loop_range + 1):
                    if i < loop_range:
                        g = wave_id * loop_range + i
                        T.mbarrier_wait_parity(k_full[g % num_stages], (g // num_stages) & 1)
                        T.mbarrier_wait_parity(s_empty[0], (g & 1) ^ 1)
                        T.tcgen05_gemm(
                            Q_shared[q_stage, :, :],
                            K_shared[g % num_stages, :, :],
                            S_tmem,
                            transpose_B=True,
                            mbar=None,
                            clear_accum=True,
                        )
                        T.tcgen05_mma_arrive(s_full[0])
                        T.tcgen05_mma_arrive(k_empty[g % num_stages])
                    if i >= 1:
                        gp = wave_id * loop_range + i - 1
                        cp = wave_id * (loop_range + 1) + i - 1
                        T.mbarrier_wait_parity(prob_full[0], gp & 1)
                        T.mbarrier_wait_parity(v_full[gp % num_stages], (gp // num_stages) & 1)
                        T.mbarrier_wait_parity(acc_full[0], cp & 1)
                        T.tcgen05_gemm(
                            P_tmem,
                            V_shared[gp % num_stages, :, :],
                            O_tmem,
                            mbar=None,
                            clear_accum=(i - 1) == 0,
                        )
                        T.tcgen05_mma_arrive(prob_empty[0])
                        T.tcgen05_mma_arrive(v_empty[gp % num_stages])
                        T.tcgen05_mma_arrive(acc_empty[0])
                T.tcgen05_mma_arrive(q_empty[q_stage])
                # Consume the version produced by the normalize commit,
                # keeping acc's per-wave cycle counts equal on both sides.
                c_last = wave_id * (loop_range + 1) + loop_range
                T.mbarrier_wait_parity(acc_full[0], c_last & 1)
                T.tcgen05_mma_arrive(acc_empty[0])

        else:  # warps 10-11: idle register donors
            T.set_max_nreg(40, 0)

    return Output


KERNELS = {
    "flash_attention": flash_attention,
    "flash_attention_ws": flash_attention_ws,
}


def reference_attention(q, k, v):
    q_cpu = q.cpu().float()
    k_cpu = k.cpu().float()
    v_cpu = v.cpu().float()
    scores = torch.einsum("bqhd,bkhd->bhqk", q_cpu, k_cpu)
    scores /= q.shape[-1] ** 0.5
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, v_cpu).to(torch.bfloat16)


def main(
    kernel="flash_attention",
    batch=1,
    heads=16,
    seq_len=16384,
    dim=128,
    block_M=128,
    block_N=128,
    store_block_N=16,
    num_stages=2,
    num_persistent_stages=2,
):
    if dim != 128:
        raise ValueError("this example requires head dimension 128")

    dtype, accum_dtype = T.bfloat16, T.float32
    kernel_fn = KERNELS[kernel]
    kwargs = {
        "block_M": block_M,
        "block_N": block_N,
        "store_block_N": store_block_N,
        "dim": dim,
        "dtype": dtype,
        "accum_dtype": accum_dtype,
        "num_stages": num_stages,
        "num_persistent_stages": num_persistent_stages,
    }

    shape = (batch, seq_len, heads, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output = kernel_fn(q, k, v, **kwargs)

    expected = reference_attention(q, k, v).to(output.device)
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    print("All checks passed. ✅")

    tl_latency = do_bench(lambda: kernel_fn(q, k, v, **kwargs), backend="cupti")
    q4 = q.permute(0, 2, 1, 3)
    k4 = k.permute(0, 2, 1, 3)
    v4 = v.permute(0, 2, 1, 3)
    torch_latency = do_bench(lambda: F.scaled_dot_product_attention(q4, k4, v4), backend="cupti")
    total_flops = 4 * batch * heads * seq_len * seq_len * dim
    print(f"Tilelang latency: {tl_latency} ms")
    print(f"Flops: {total_flops / (tl_latency / 1e3) / 1e12} TFLOPS")
    print(f"Torch latency: {torch_latency} ms")
    print(f"Flops: {total_flops / (torch_latency / 1e3) / 1e12} TFLOPS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=sorted(KERNELS), default="flash_attention")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=16384)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--block_m", type=int, default=128)
    p.add_argument("--block_n", type=int, default=128)
    p.add_argument(
        "--store_block_n",
        type=int,
        default=16,
        help="epilogue store slice width",
    )
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--num_persistent_stages", type=int, default=2)
    args = p.parse_args()
    main(
        kernel=args.kernel,
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        dim=args.dim,
        block_M=args.block_m,
        block_N=args.block_n,
        store_block_N=args.store_block_n,
        num_stages=args.num_stages,
        num_persistent_stages=args.num_persistent_stages,
    )
