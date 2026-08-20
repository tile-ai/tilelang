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


@tilelang.jit(
    pass_configs={
        **PASS_CFG,
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: "role_based",
    }
)
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

        sched = T.PersistentTileScheduler(q_blocks, heads * batch, name="sched")
        sched.init(block_id)

        while sched.valid():
            bx = sched.m_idx
            hn = sched.n_idx
            by = hn % heads
            bz = hn // heads

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.fill(logsum, 0)

            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)

                T.gemm(Q_shared, K_shared, S_tmem, transpose_B=True, clear_accum=True)

                T.copy(S_tmem, S_reg)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(S_reg, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    # Stale-max fast path: a barely-moved max keeps a
                    # rescale factor of exactly 1.
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

                T.copy(scores_rescale, scale_shared)

                # Skip the O read-modify-write when every factor is 1.
                should_rescale = T.any_sync(scale_shared[tid % block_M] < 1.0)
                if should_rescale != 0:
                    for s in T.unroll(T.ceildiv(dim, store_block_N)):
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
                for i in T.Parallel(block_M):
                    for j in T.vectorized(block_N):
                        S_reg[i, j] = S_reg[i, j] * scale + (-scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    S_reg[i, j] = T.exp2(S_reg[i, j])

                T.copy(S_reg, P_cast)
                T.copy(P_cast, P_tmem)

                T.reduce_sum(S_reg, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_rescale[i] + scores_sum[i]

                T.gemm(P_tmem, V_shared, O_tmem, clear_accum=k == 0)

            T.copy(logsum, logsum_shared)
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
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

            sched.next_tile()

    return Output


@tilelang.jit(pass_configs=PASS_CFG)
def flash_attention_manual(
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

        # The persistent tile scheduler, in FA4's static tile order: the q
        # block index is the minor dimension, then head, then batch
        # (m = q block, n = head-batch). State is per thread; every role
        # runs its own copy of the init / advance ops.
        sched = T.PersistentTileScheduler(q_blocks, heads * batch, name="sched")

        # Five roles (an FA4-style split, minus 2-CTA):
        #  - Softmax owns the S -> P transform and the running
        #    statistics, handing the rescale factor to Correction right
        #    after row-max and publishing P before the row-sum.
        #  - Correction owns the O accumulator: the per-iteration rescale
        #    (skipped on the stale-max fast path) and the final normalize,
        #    staged into O_shared for the Epilogue.
        #  - TMA feeds Q/K/V; MMA issues both GEMMs; Epilogue stores O.
        #
        # MMA's PV spans sit at stage 1, one iteration behind QK: the
        # tensor core computes S(i) while Softmax turns S(i-1) into
        # P(i-1).
        #
        # O_tmem is bound to two nested pipelines (each pipeline
        # synchronizes exactly one scope): `acc` alternates Correction's
        # rescale with MMA's PV inside loop_kv, and `acc_wave` hands the
        # finished accumulator to the normalize once per wave — MMA
        # producer-brackets the whole kv loop. acc_wave's depth
        # double-buffers O_tmem across waves (nested bindings multiply
        # the version count), so wave n+1's QK/PV target the other slot
        # and never stall behind normalize(n); Correction's rescale,
        # which holds only the inner span, derives the wave slot from
        # its own acc_wave phase counter.
        #
        # Not reflected from flash_attention_ws (inexpressible here):
        # q_stage=2 with PV-first interleave needs P overlaying S in TMEM
        # (pipelines cannot share one storage through an overlay) to fit
        # 512 columns; the
        # merged S/P/O barrier has two signaling consumer roles; the
        # commit-only barriers rely on in-order tcgen05 completion; split_P
        # signals at sub-op granularity.
        T.annotate_ws_schedule(
            T.WSSchedule(
                num_warps=12,
                roles=[
                    T.WSRole("Softmax", warps_lo=0, warps_hi=4, max_nreg=224),
                    T.WSRole("Correction", warps_lo=4, warps_hi=8, max_nreg=80),
                    T.WSRole("TMA", warps_lo=8, warps_hi=9, max_nreg=40),
                    T.WSRole("MMA", warps_lo=9, warps_hi=10, max_nreg=40),
                    # FA4-style dedicated store warp: Correction hands the
                    # normalized O over in smem and moves on; the store's
                    # drain never blocks the next tile's rescale work.
                    T.WSRole("Epilogue", warps_lo=10, warps_hi=11, max_nreg=40),
                    # Warp 11 is an unassigned register donor.
                ],
                pipelines=[
                    T.WSPipeline("q", [Q_shared], depth=num_persistent_stages),
                    T.WSPipeline("k", [K_shared], depth=num_stages),
                    T.WSPipeline("v", [V_shared], depth=num_stages),
                    T.WSPipeline("score", [S_tmem], depth=1),
                    T.WSPipeline("prob", [P_tmem], depth=1),
                    # O_tmem's nested bindings; versions multiply, giving
                    # num_persistent_stages accumulator slots.
                    T.WSPipeline("acc", [O_tmem], depth=1),
                    T.WSPipeline("acc_wave", [O_tmem], depth=num_persistent_stages),
                    # Softmax -> Correction rescale handoff.
                    T.WSPipeline("scale", [scale_shared], depth=num_stages),
                    # Softmax -> Correction logsum handoff for the epilogue.
                    T.WSPipeline("stats", [logsum_shared], depth=1),
                    # Correction -> Epilogue staged-O handoff. Correction's
                    # generic O_shared writes are observed by the Epilogue's
                    # TMA store, so the materializer emits the writer-side
                    # proxy fence before the commit.
                    T.WSPipeline("o_epi", [O_shared], depth=1),
                ],
                scopes=[
                    # The unrolled O loops are scheduled scopes on
                    # Correction; the acc brackets stay in the enclosing scopes.
                    T.WSScope(
                        "rescale_O",
                        {"Correction": ["rescale_load", "rescale_mul", "rescale_store"]},
                    ),
                    T.WSScope(
                        "normalize_O",
                        {"Correction": ["normalize_load", "normalize_mul", "normalize_store"]},
                    ),
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
                                # Stage 1: PV of the PREVIOUS kv step,
                                # after QK of the current one.
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
                    # The persistent loop: a while scope. It has no
                    # iteration expression, so pipelines synced under it
                    # use runtime phase counters. tile_idx / sched_next
                    # touch no pipeline buffers, so several roles place
                    # them — each runs its own copy.
                    T.WSScope(
                        "loop_wave",
                        {
                            "TMA": [
                                "tile_idx",
                                T.WSSync.producer_acquire("q"),
                                "copy_Q_g2s",
                                T.WSSync.producer_commit("q"),
                                "loop_kv",
                                "sched_next",
                            ],
                            "MMA": [
                                T.WSSync.consumer_wait("q"),
                                # The producer bracket around the whole kv
                                # loop: the commit (a tcgen05 watermark)
                                # publishes the last PV; with the
                                # depth-num_persistent_stages ring the
                                # acquire pairs the normalize two waves
                                # back, so the next wave never stalls.
                                T.WSSync.producer_acquire("acc_wave"),
                                "loop_kv",
                                T.WSSync.producer_commit("acc_wave"),
                                T.WSSync.consumer_release("q"),
                                "sched_next",
                            ],
                            "Softmax": [
                                "init_max",
                                "init_logsum",
                                "loop_kv",
                                T.WSSync.producer_acquire("stats"),
                                "copy_logsum_s",
                                T.WSSync.producer_commit("stats"),
                                "sched_next",
                            ],
                            "Correction": [
                                "tile_idx",
                                "loop_kv",
                                T.WSSync.consumer_wait("stats"),
                                "copy_inv_sum",
                                "recip_sum",
                                T.WSSync.consumer_wait("acc_wave"),
                                T.WSSync.producer_acquire("o_epi"),
                                "normalize_O",
                                T.WSSync.producer_commit("o_epi"),
                                T.WSSync.consumer_release("acc_wave"),
                                T.WSSync.consumer_release("stats"),
                                "sched_next",
                            ],
                            "Epilogue": [
                                "tile_idx",
                                T.WSSync.consumer_wait("o_epi"),
                                "copy_O_s2g",
                                T.WSSync.consumer_release("o_epi"),
                                "sched_next",
                            ],
                        },
                    ),
                    T.WSScope(
                        T.WSScope.ROOT,
                        {
                            "TMA": ["sched_init", "loop_wave"],
                            "MMA": ["sched_init", "loop_wave"],
                            "Softmax": ["sched_init", "loop_wave"],
                            "Correction": ["sched_init", "loop_wave"],
                            "Epilogue": ["sched_init", "loop_wave"],
                        },
                    ),
                ],
            )
        )

        with T.ws_op("sched_init"):
            sched.init(block_id)

        with T.ws_op("loop_wave"):
            while sched.valid():
                with T.ws_op("tile_idx"):
                    bx = sched.m_idx
                    hn = sched.n_idx
                    by = hn % heads
                    bz = hn // heads

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

                    # Warp-local vote: each warp owns 32 rows of O and
                    # skips its read-modify-write when all its rescale
                    # factors are 1 (at k == 0 PV clears the accumulator
                    # anyway). The guard covers the scope body alone; sync
                    # entries stay unconditional.
                    with T.ws_op("rescale_vote"):
                        should_rescale = T.any_sync(scale_shared[tid % block_M] < 1.0)
                    if should_rescale != 0:
                        for s in T.unroll(T.ceildiv(dim, store_block_N), annotations={T.WSID: "rescale_O"}):
                            T.copy(
                                O_tmem[:, s * store_block_N : (s + 1) * store_block_N],
                                O_local,
                                annotations={T.WSID: "rescale_load"},
                            )
                            for i, j in T.Parallel(block_M, store_block_N, annotations={T.WSID: "rescale_mul"}):
                                O_local[i, j] *= scale_shared[i]
                            T.copy(
                                O_local,
                                O_tmem[:, s * store_block_N : (s + 1) * store_block_N],
                                annotations={T.WSID: "rescale_store"},
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
                        annotations={T.WSID: "normalize_load"},
                    )
                    for i, j in T.Parallel(block_M, store_block_N, annotations={T.WSID: "normalize_mul"}):
                        O_local[i, j] *= inv_sum[i]
                    T.copy(
                        O_local,
                        O_shared[:, s * store_block_N : (s + 1) * store_block_N],
                        annotations={T.WSID: "normalize_store"},
                    )
                T.copy(
                    O_shared,
                    Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
                    annotations={T.WSID: "copy_O_s2g"},
                )

                with T.ws_op("sched_next"):
                    sched.next_tile()

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

    # FA4 sm100 structure: each tile is 2*block_M query rows split into two
    # q stages that share the K/V stream; the MMA warp interleaves the
    # stages PV-first, so while one stage waits on its P the other's QK is
    # already in flight. num_persistent_stages is the q-stage count.
    assert num_persistent_stages == 2
    q_blocks = T.ceildiv(seq_len, 2 * block_M)
    loop_range = T.ceildiv(seq_len, block_N)
    sm_num = driver.get_num_sms()
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    assert dim == 128
    assert block_M == 128
    assert block_N == 128
    assert seq_len % (2 * block_M) == 0
    assert seq_len % block_N == 0
    assert dim % store_block_N == 0

    with T.Kernel(sm_num, threads=512) as block_id:
        Q_shared = T.alloc_shared((2, block_M, dim), dtype)
        K_shared = T.alloc_shared((num_stages, block_N, dim), dtype)
        V_shared = T.alloc_shared((num_stages, block_N, dim), dtype)
        O_shared = T.alloc_shared((block_M, dim), dtype)
        # Single-slot per-stage handoffs (FA4's sScale): softmax and
        # correction run in lockstep per iteration.
        scale_shared = T.alloc_shared((2, block_M), accum_dtype)
        logsum_shared = T.alloc_shared((2, block_M), accum_dtype)

        # TMEM (512 cols): S0 S1 | O0 O1. P_s overlays the upper half of
        # S_s as bf16 (FA4's tmem_s_to_p_offset): S is dead there once
        # softmax read it, and PV(i) is issued before QK(i+1) overwrites S.
        S_tmem = T.alloc_tmem((2, block_M, block_N), accum_dtype)
        O_tmem = T.alloc_tmem((2, block_M, dim), accum_dtype)
        P_tmem = T.view(S_tmem, shape=(2, block_M, 2 * block_N), dtype=dtype)

        q_full = T.alloc_barrier([32, 32])
        q_empty = T.alloc_barrier([1, 1])
        k_full = T.alloc_barrier([32] * num_stages)
        k_empty = T.alloc_barrier([1] * num_stages)
        v_full = T.alloc_barrier([32] * num_stages)
        v_empty = T.alloc_barrier([1] * num_stages)
        # FA4's merged S/P/O pipeline, one per q stage. Full side: "S ready",
        # one tcgen05 commit per QK. Empty side: "P written AND O rescaled",
        # 128 softmax + 128 correction arrives; the MMA warp acquires it
        # before each PV.
        spo_full = T.alloc_barrier([1, 1])
        spo_empty = T.alloc_barrier([256, 256])
        # Commit-only, once per tile after the last PV; correction waits
        # before the normalize read. The per-iteration O signal is elided:
        # scale(g) published means QK(g) completed, and PV(g-1) was issued
        # before QK(g), so it completed too.
        o_full = T.alloc_barrier([1, 1])
        # split_P: the spo arrive covers the first 3/4 of P; the last 1/4
        # commits here and the MMA warp waits for it between the two PV
        # halves (FA4's pipeline_p_lastsplit).
        p_last = T.alloc_barrier([128, 128])
        scale_full = T.alloc_barrier([128, 128])
        scale_empty = T.alloc_barrier([128, 128])
        stats_full = T.alloc_barrier([128, 128])
        stats_empty = T.alloc_barrier([128, 128])
        # Correction -> epilogue handoff (single O_shared slot): full =
        # normalized O staged in smem; empty = the epilogue warp's gmem TMA
        # store completed. Two handoffs per tile, completion #(wave*2 + s).
        o_epi_full = T.alloc_barrier([128])
        o_epi_empty = T.alloc_barrier([32])

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
        # Softmax stage 1 runs in a different warpgroup: fragment layouts
        # bind to threads, so it needs its own register set.
        S_reg_1 = T.alloc_fragment((block_M, block_N), accum_dtype)
        P_cast_1 = T.alloc_fragment((block_M, block_N), dtype)
        scores_max_1 = T.alloc_fragment((block_M,), accum_dtype)
        scores_max_prev_1 = T.alloc_fragment((block_M,), accum_dtype)
        scores_rescale_1 = T.alloc_fragment((block_M,), accum_dtype)
        scores_sum_1 = T.alloc_fragment((block_M,), accum_dtype)
        logsum_1 = T.alloc_fragment((block_M,), accum_dtype)

        # Scheduler state is per thread (FA4's static tile order: q block
        # minor, then head, then batch): init once, then every role runs
        # its own `while sched.valid()` loop at its own pace.
        sched = T.PersistentTileScheduler(q_blocks, heads * batch, name="sched")
        sched.init(block_id)

        if tid < 128:  # warps 0-3: softmax, q stage 0
            T.set_max_nreg(200, 1)
            while sched.valid():
                wave_id = sched.current_iter
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.fill(logsum, 0)

                for k in T.serial(loop_range):
                    g = wave_id * loop_range + k

                    T.mbarrier_wait_parity(spo_full[0], g & 1)
                    # Two x64 tmem loads: a single 32x32b.x128 load defines
                    # 146 registers at once, above the 128-register entry
                    # budget of a 512-thread kernel (region liveness may
                    # exceed it, a single instruction's operands may not).
                    for h in T.unroll(2):
                        T.copy(
                            S_tmem[0, :, h * (block_N // 2) : (h + 1) * (block_N // 2)],
                            S_reg[:, h * (block_N // 2) : (h + 1) * (block_N // 2)],
                        )

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

                    # Publish the rescale right after row-max so correction
                    # overlaps with exp / P production below.
                    T.mbarrier_wait_parity(scale_empty[0], (g & 1) ^ 1)
                    T.copy(scores_rescale, scale_shared[0, :])
                    T.mbarrier_arrive(scale_full[0])

                    # Affine first (packed f32x2 FMA), exp2 separately.
                    for i in T.Parallel(block_M):
                        for j in T.vectorized(block_N):
                            S_reg[i, j] = S_reg[i, j] * scale + (-scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg[i, j] = T.exp2(S_reg[i, j])

                    # Publish P before row-sum: the reduction is not a PV
                    # dependency. This arrive also hands S back — P and S
                    # share the tmem columns, so one signal covers both.
                    T.copy(S_reg, P_cast)
                    T.copy(P_cast[:, 0:96], P_tmem[0, :, block_N : block_N + 96])
                    T.mbarrier_arrive(spo_empty[0])
                    T.copy(P_cast[:, 96:128], P_tmem[0, :, block_N + 96 : 2 * block_N])
                    T.mbarrier_arrive(p_last[0])

                    T.reduce_sum(S_reg, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_rescale[i] + scores_sum[i]

                T.mbarrier_wait_parity(stats_empty[0], (wave_id & 1) ^ 1)
                T.copy(logsum, logsum_shared[0, :])
                T.mbarrier_arrive(stats_full[0])
                sched.next_tile()

        elif tid < 256:  # warps 4-7: softmax, q stage 1
            T.set_max_nreg(200, 1)
            while sched.valid():
                wave_id = sched.current_iter
                T.fill(scores_max_1, -T.infinity(accum_dtype))
                T.fill(logsum_1, 0)

                for k in T.serial(loop_range):
                    g = wave_id * loop_range + k

                    T.mbarrier_wait_parity(spo_full[1], g & 1)
                    # Two x64 tmem loads: a single 32x32b.x128 load defines
                    # 146 registers at once, above the 128-register entry
                    # budget of a 512-thread kernel (region liveness may
                    # exceed it, a single instruction's operands may not).
                    for h in T.unroll(2):
                        T.copy(
                            S_tmem[1, :, h * (block_N // 2) : (h + 1) * (block_N // 2)],
                            S_reg_1[:, h * (block_N // 2) : (h + 1) * (block_N // 2)],
                        )

                    T.copy(scores_max_1, scores_max_prev_1)
                    T.reduce_max(S_reg_1, scores_max_1, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        # Stale-max fast path: keep the previous max when it
                        # moves by less than 8 exponent steps, so the
                        # rescale factor is exactly 1.
                        scores_rescale_1[i] = T.if_then_else(
                            (scores_max_prev_1[i] - scores_max_1[i]) * scale >= -8.0,
                            1.0,
                            T.exp2(scores_max_prev_1[i] * scale - scores_max_1[i] * scale),
                        )
                        scores_max_1[i] = T.if_then_else(
                            (scores_max_prev_1[i] - scores_max_1[i]) * scale >= -8.0,
                            scores_max_prev_1[i],
                            scores_max_1[i],
                        )

                    # Publish the rescale right after row-max so correction
                    # overlaps with exp / P production below.
                    T.mbarrier_wait_parity(scale_empty[1], (g & 1) ^ 1)
                    T.copy(scores_rescale_1, scale_shared[1, :])
                    T.mbarrier_arrive(scale_full[1])

                    # Affine first (packed f32x2 FMA), exp2 separately.
                    for i in T.Parallel(block_M):
                        for j in T.vectorized(block_N):
                            S_reg_1[i, j] = S_reg_1[i, j] * scale + (-scores_max_1[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        S_reg_1[i, j] = T.exp2(S_reg_1[i, j])

                    # Publish P before row-sum: the reduction is not a PV
                    # dependency. This arrive also hands S back — P and S
                    # share the tmem columns, so one signal covers both.
                    T.copy(S_reg_1, P_cast_1)
                    T.copy(P_cast_1[:, 0:96], P_tmem[1, :, block_N : block_N + 96])
                    T.mbarrier_arrive(spo_empty[1])
                    T.copy(P_cast_1[:, 96:128], P_tmem[1, :, block_N + 96 : 2 * block_N])
                    T.mbarrier_arrive(p_last[1])

                    T.reduce_sum(S_reg_1, scores_sum_1, dim=1)
                    for i in T.Parallel(block_M):
                        logsum_1[i] = logsum_1[i] * scores_rescale_1[i] + scores_sum_1[i]

                T.mbarrier_wait_parity(stats_empty[1], (wave_id & 1) ^ 1)
                T.copy(logsum_1, logsum_shared[1, :])
                T.mbarrier_arrive(stats_full[1])
                sched.next_tile()

        elif tid < 384:  # warps 8-11: O correction, both stages
            T.set_max_nreg(64, 0)
            # One-time priming: the first PV of each stage's first tile
            # needs no correction, so pre-supply this side's arrive.
            T.mbarrier_arrive(spo_empty[0])
            T.mbarrier_arrive(spo_empty[1])
            while sched.valid():
                wave_id = sched.current_iter
                g0 = wave_id * loop_range

                # Each stage's first rescale factor covers an O that PV(0)
                # clears anyway: consume and discard it.
                for s in T.serial(2):
                    T.mbarrier_wait_parity(scale_full[s], g0 & 1)
                    T.mbarrier_arrive(scale_empty[s])

                for k in T.serial(loop_range - 1):
                    g = g0 + k + 1
                    for s in T.serial(2):
                        T.mbarrier_wait_parity(scale_full[s], g & 1)
                        # No O wait: scale(g) published means QK(s, g)
                        # completed, so PV(s, g-1) — issued before it — has
                        # completed as well. Common case: every rescale
                        # factor is exactly 1 (stale-max fast path), and
                        # each warp skips the O read-modify-write.
                        should_rescale = T.any_sync(scale_shared[s, tid % block_M] < 1.0)
                        if should_rescale != 0:
                            for t in T.unroll(T.ceildiv(dim, store_block_N)):
                                T.copy(
                                    O_tmem[s, :, t * store_block_N : (t + 1) * store_block_N],
                                    O_local,
                                )
                                for i, j in T.Parallel(block_M, store_block_N):
                                    O_local[i, j] *= scale_shared[s, i]
                                T.copy(
                                    O_local,
                                    O_tmem[s, :, t * store_block_N : (t + 1) * store_block_N],
                                )
                        T.mbarrier_arrive(spo_empty[s])
                        T.mbarrier_arrive(scale_empty[s])

                for s in T.serial(2):
                    T.mbarrier_wait_parity(stats_full[s], wave_id & 1)
                    T.mbarrier_wait_parity(o_full[s], wave_id & 1)
                    T.copy(logsum_shared[s, :], inv_sum)
                    # One reciprocal per row, reused across all output slices.
                    for i in T.Parallel(block_M):
                        inv_sum[i] = 1.0 / inv_sum[i]
                    # O_shared free once the epilogue's previous store is done.
                    T.mbarrier_wait_parity(o_epi_empty[0], ((wave_id * 2 + s) & 1) ^ 1)
                    for t in T.unroll(T.ceildiv(dim, store_block_N)):
                        T.copy(
                            O_tmem[s, :, t * store_block_N : (t + 1) * store_block_N],
                            O_local,
                        )
                        for i, j in T.Parallel(block_M, store_block_N):
                            O_local[i, j] *= inv_sum[i]
                        T.copy(
                            O_local,
                            O_shared[:, t * store_block_N : (t + 1) * store_block_N],
                        )
                    # O[s] read: the next tile's clearing PV may overwrite.
                    T.mbarrier_arrive(spo_empty[s])
                    T.mbarrier_arrive(stats_empty[s])
                    # Generic O_shared writes are observed by the epilogue's
                    # TMA store (async proxy): writer-side fence, then arrive.
                    T.fence_proxy_async()
                    T.mbarrier_arrive(o_epi_full[0])
                sched.next_tile()

        elif tid < 416:  # warp 12: TMA
            T.set_max_nreg(48, 0)
            while sched.valid():
                wave_id = sched.current_iter
                bx = sched.m_idx
                by = sched.n_idx % heads
                bz = sched.n_idx // heads

                for s in T.serial(2):
                    T.mbarrier_wait_parity(q_empty[s], (wave_id & 1) ^ 1)
                    T.tma_copy(
                        Q[bz, (bx * 2 + s) * block_M : (bx * 2 + s + 1) * block_M, by, :],
                        Q_shared[s, :, :],
                        barrier=q_full[s],
                    )
                    T.mbarrier_arrive(q_full[s])

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
                sched.next_tile()

        elif tid < 448:  # warp 13: tensor-core issue
            T.set_max_nreg(48, 0)
            while sched.valid():
                wave_id = sched.current_iter
                g0 = wave_id * loop_range

                # Prologue QKs need no S acquire: the previous tile's tail
                # PV acquired spo, so S was read (and reused as P) by then.
                T.mbarrier_wait_parity(q_full[0], wave_id & 1)
                T.mbarrier_wait_parity(k_full[g0 % num_stages], (g0 // num_stages) & 1)
                T.tcgen05_gemm(
                    Q_shared[0, :, :],
                    K_shared[g0 % num_stages, :, :],
                    S_tmem[0, :, :],
                    transpose_B=True,
                    mbar=None,
                    clear_accum=True,
                )
                T.tcgen05_mma_arrive(spo_full[0])
                T.mbarrier_wait_parity(q_full[1], wave_id & 1)
                T.tcgen05_gemm(
                    Q_shared[1, :, :],
                    K_shared[g0 % num_stages, :, :],
                    S_tmem[1, :, :],
                    transpose_B=True,
                    mbar=None,
                    clear_accum=True,
                )
                T.tcgen05_mma_arrive(spo_full[1])
                T.tcgen05_mma_arrive(k_empty[g0 % num_stages])

                # PV-first, stages interleaved: while a stage waits on its
                # P, the other stage's QK is already executing. PV(s, i)
                # before QK(s, i+1) is also what makes the P-in-S overlay
                # and correction's elided O wait sound.
                for i in T.serial(loop_range - 1):
                    gv = g0 + i
                    gk = gv + 1
                    T.mbarrier_wait_parity(v_full[gv % num_stages], (gv // num_stages) & 1)
                    T.mbarrier_wait_parity(spo_empty[0], gv & 1)
                    # split_P: the last K chunk waits for its p_last commit.
                    for h in T.unroll(4):
                        if h == 3:
                            T.mbarrier_wait_parity(p_last[0], gv & 1)
                        T.tcgen05_gemm(
                            P_tmem[0, :, block_N + 32 * h : block_N + 32 * (h + 1)],
                            V_shared[gv % num_stages, 32 * h : 32 * (h + 1), :],
                            O_tmem[0, :, :],
                            mbar=None,
                            clear_accum=(i == 0) & (h == 0),
                        )
                    T.mbarrier_wait_parity(k_full[gk % num_stages], (gk // num_stages) & 1)
                    T.tcgen05_gemm(
                        Q_shared[0, :, :],
                        K_shared[gk % num_stages, :, :],
                        S_tmem[0, :, :],
                        transpose_B=True,
                        mbar=None,
                        clear_accum=True,
                    )
                    T.tcgen05_mma_arrive(spo_full[0])
                    T.mbarrier_wait_parity(spo_empty[1], gv & 1)
                    for h in T.unroll(4):
                        if h == 3:
                            T.mbarrier_wait_parity(p_last[1], gv & 1)
                        T.tcgen05_gemm(
                            P_tmem[1, :, block_N + 32 * h : block_N + 32 * (h + 1)],
                            V_shared[gv % num_stages, 32 * h : 32 * (h + 1), :],
                            O_tmem[1, :, :],
                            mbar=None,
                            clear_accum=(i == 0) & (h == 0),
                        )
                    T.tcgen05_mma_arrive(v_empty[gv % num_stages])
                    T.tcgen05_gemm(
                        Q_shared[1, :, :],
                        K_shared[gk % num_stages, :, :],
                        S_tmem[1, :, :],
                        transpose_B=True,
                        mbar=None,
                        clear_accum=True,
                    )
                    T.tcgen05_mma_arrive(spo_full[1])
                    T.tcgen05_mma_arrive(k_empty[gk % num_stages])

                T.tcgen05_mma_arrive(q_empty[0])
                T.tcgen05_mma_arrive(q_empty[1])

                gl = g0 + loop_range - 1
                T.mbarrier_wait_parity(v_full[gl % num_stages], (gl // num_stages) & 1)
                T.mbarrier_wait_parity(spo_empty[0], gl & 1)
                for h in T.unroll(4):
                    if h == 3:
                        T.mbarrier_wait_parity(p_last[0], gl & 1)
                    T.tcgen05_gemm(
                        P_tmem[0, :, block_N + 32 * h : block_N + 32 * (h + 1)],
                        V_shared[gl % num_stages, 32 * h : 32 * (h + 1), :],
                        O_tmem[0, :, :],
                        mbar=None,
                        clear_accum=(loop_range == 1) & (h == 0),
                    )
                # The only O signal of the tile: softmax's last signal does
                # not imply the tail PV finished, so correction must wait.
                T.tcgen05_mma_arrive(o_full[0])
                T.mbarrier_wait_parity(spo_empty[1], gl & 1)
                for h in T.unroll(4):
                    if h == 3:
                        T.mbarrier_wait_parity(p_last[1], gl & 1)
                    T.tcgen05_gemm(
                        P_tmem[1, :, block_N + 32 * h : block_N + 32 * (h + 1)],
                        V_shared[gl % num_stages, 32 * h : 32 * (h + 1), :],
                        O_tmem[1, :, :],
                        mbar=None,
                        clear_accum=(loop_range == 1) & (h == 0),
                    )
                T.tcgen05_mma_arrive(o_full[1])
                T.tcgen05_mma_arrive(v_empty[gl % num_stages])
                sched.next_tile()

        elif tid < 480:  # warp 14: epilogue, O store to gmem
            T.set_max_nreg(48, 0)
            while sched.valid():
                wave_id = sched.current_iter
                bx = sched.m_idx
                by = sched.n_idx % heads
                bz = sched.n_idx // heads
                for s in T.serial(2):
                    T.mbarrier_wait_parity(o_epi_full[0], (wave_id * 2 + s) & 1)
                    T.copy(
                        O_shared,
                        Output[bz, (bx * 2 + s) * block_M : (bx * 2 + s + 1) * block_M, by, :],
                    )
                    T.mbarrier_arrive(o_epi_empty[0])
                sched.next_tile()

        else:  # warp 15: idle register donor
            T.set_max_nreg(48, 0)

    return Output


KERNELS = {
    "flash_attention": flash_attention,
    "flash_attention_manual": flash_attention_manual,
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
