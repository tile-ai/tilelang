# ruff: noqa
import torch
import tilelang
from tilelang import language as T
import argparse


@tilelang.jit(
    out_idx=[-2, -1],
    compile_flags=[
        "-O3",
        "--ptxas-options=-v,--register-usage-level=10",
        "-DNDEBUG",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ],
)
def sparse_mla_fwd(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    tail_dim,
    topk,
    kv_stride,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=0,
    threads=384,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you "
            "should handle Q copy and Output copy with your mask (when "
            "kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would "
            "be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    assert NI % 2 == 0, "NI should be a multiple of 2"
    D = dim
    D_tail = tail_dim
    KV_stride = kv_stride
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    # Increasing from 32->64 reduces the time spent reading kvcache. If num_query_head = 128
    # and num_kv_head = 1, the same kvcache originally needed to be read 4 times, but now only 2 times
    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor(1, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(
            # If CP0 is True (i.e., start of sequence), skip the first (KV_stride - 1)
            # queries that cannot see any KV. Also be careful that seq_len < kv_stride could cause negative grid size
            (max(0, seq_len - kv_stride + 1) if CP0 else seq_len) * REPLICATE_H,
            batch,
            kv_group,
            threads=threads,
        ) as (bx, by, bz):
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)

            KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
            K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
            K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)

            O_shared_l = Q_shared_l
            O_shared_r = Q_shared_r

            # Whether the kv in current BI is visible for this query
            # Producer alternates writing to buf0 and buf1 masks. To avoid the situation
            # where consumer0 is still reading buf0 mask when producer has already started
            # writing buf1 mask, we use two buf masks
            is_kv_valid = T.alloc_shared([2, BI], "bool", scope="shared")

            acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
            acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)

            # WG0 computes S0(BI_2*i), WG1 computes S1(BI_2*i+1), shared via shared memory

            # Reuse K_tail_shared for S_shared to save memory when dimensions match
            # Must reuse, otherwise H100 SM's shared mem is insufficient (> 228kb), this is shared mem bound
            S_shared_0 = K_tail_shared_0
            S_shared_1 = K_tail_shared_1

            # WG0 and WG1 exchange local max with each other, compare to compute global max, and rescale their O_L or O_R accordingly
            row_max_shared_0 = T.alloc_shared([H_per_block], accum_dtype)
            row_max_shared_1 = T.alloc_shared([H_per_block], accum_dtype)

            # Used to store sum of exps for even BI and odd BI respectively, which will be summed up for integration later
            row_sum_shared_0 = T.alloc_shared([H_per_block], accum_dtype)
            row_sum_shared_1 = T.alloc_shared([H_per_block], accum_dtype)

            # acc_s, sumexp, m_i each need to be allocated separately for consumer0 and consumer1
            acc_s_0 = T.alloc_fragment([H_per_block, BI], accum_dtype)
            acc_s_1 = T.alloc_fragment([H_per_block, BI], accum_dtype)

            sumexp_0 = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i_0 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_0 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev_0 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_peer_0 = T.alloc_fragment([H_per_block], accum_dtype)

            sumexp_1 = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i_1 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_1 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev_1 = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_peer_1 = T.alloc_fragment([H_per_block], accum_dtype)

            bar_q = T.alloc_barrier(arrive_count=384)

            # Producer -> Consumer Barriers
            bar_k_0_ready = T.alloc_barrier(arrive_count=128)  # Prod arrives
            bar_k_1_ready = T.alloc_barrier(arrive_count=128)  # Prod arrives

            # Consumer -> Producer Barriers (Both consumers must arrive)
            bar_k_0_free = T.alloc_barrier(arrive_count=256)
            bar_k_1_free = T.alloc_barrier(arrive_count=256)

            # Inter-Consumer Barriers (Seesaw Sync)
            bar_stats_0_ready = T.alloc_barrier(arrive_count=128)  # Cons 0 arrives
            bar_stats_1_ready = T.alloc_barrier(arrive_count=128)  # Cons 1 arrives

            bar_S_0_ready = T.alloc_barrier(arrive_count=128)  # Cons 0 arrives
            bar_S_1_ready = T.alloc_barrier(arrive_count=128)  # Cons 1 arrives

            b_i, g_i = by, bz
            # If it's the first chunk, start computing directly from the (kv_stride - 1)-th token
            s_i = (bx + (KV_stride - 1 if CP0 else 0)) if REPLICATE_H == 1 else (bx // REPLICATE_H + (KV_stride - 1 if CP0 else 0))
            q_i = q_start_index_s[0] + s_i
            # Sometimes to reduce kvcache size, we may not store KV for every token, but store
            # KV every KV_stride tokens (usually the last token in the stride window),
            # so the kv range visible to the current query should be [0:max_kv_i]
            max_kv_i = (q_i + 1 - KV_stride) // KV_stride

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            tx = T.get_thread_binding()

            T.copy(Q[b_i, s_i, H0:H1, 0 : D // 2], Q_shared_l)
            T.copy(Q[b_i, s_i, H0:H1, D // 2 : D], Q_shared_r)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            # Non-blockingly increment the barrier's internal counter, producer threads can start loading kv ahead of time
            T.barrier_arrive(bar_q)

            if tx >= 256:
                # producer: prefetch kvcache to shared mem
                T.set_max_nreg(72, 0)

                prefetch_indices_0 = T.alloc_fragment([4], indices_dtype)
                prefetch_indices_1 = T.alloc_fragment([4], indices_dtype)

                # Prime the Pump! Prefetch indices for iter_0
                for r in T.serial(4):
                    # This read will cause a long scoreboard stall, but it only happens once before the loop starts
                    prefetch_indices_0[r] = Indices[b_i, s_i, g_i, r * 16 + (tx - 256) // 8]
                    prefetch_indices_1[r] = Indices[b_i, s_i, g_i, BI + r * 16 + (tx - 256) // 8]

                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    # Wait for both KV_shared_0_l and KV_shared_0_r to be done being used

                    T.barrier_wait(bar_k_0_free[0], (i_i & 1))

                    # Block size `BI` is 64, loading is divided into 4 iterations, each processing 16 indices
                    # Producer has 128 threads total, 8 consecutive threads collaborate to load kv for one index
                    for r in T.serial(4):
                        # mitigate long scoreboard stall here
                        index = prefetch_indices_0[r]
                        is_kv_valid[0, r * 16 + (tx - 256) // 8] = index <= max_kv_i
                        if is_kv_valid[0, r * 16 + (tx - 256) // 8]:
                            # Here we assume dim = 512, tail_dim = 64
                            with T.attr("default", "async_scope", 1):
                                # 8 threads collaborate to load one row of KV_dim (length 512), divided into 4 iterations
                                # In each iteration, each thread loads 8 consecutive elements for both KV_shared_0_l
                                # and KV_shared_0_r, 8 threads load 64 elements total for each
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        # (tx - 256) // 8 determines which row the thread is responsible for,
                                        # (tx - 256) % 8 determines which part of the row the thread loads
                                        KV_shared_0_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                                            b_i, index, g_i, 64 * u + (tx - 256) % 8 * 8 + v
                                        ]
                                        KV_shared_0_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                                            b_i, index, g_i, D // 2 + 64 * u + (tx - 256) % 8 * 8 + v
                                        ]
                            with T.attr("default", "async_scope", 1):
                                # tail_dim (length 64) only needs 8 threads collaborating in one iteration to complete loading
                                for v in T.vectorized(8):
                                    K_tail_shared_0[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v] = KV[
                                        b_i, index, g_i, D + (tx - 256) % 8 * 8 + v
                                    ]
                    T.cp_async_barrier_noinc(bar_k_0_ready[0])

                    if i_i + 1 < T.ceildiv(NI, 2):
                        # Async prefetch indices needed for the next round of kv data loading, overlaps with current round to hide latency
                        for r in T.serial(4):
                            prefetch_indices_0[r] = Indices[b_i, s_i, g_i, ((i_i + 1) * 2) * BI + r * 16 + (tx - 256) // 8]

                    # Buffer 1
                    T.barrier_wait(bar_k_1_free[0], (i_i & 1))

                    for r in T.serial(4):
                        index = prefetch_indices_1[r]
                        is_kv_valid[1, r * 16 + (tx - 256) // 8] = index <= max_kv_i
                        if is_kv_valid[1, r * 16 + (tx - 256) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        KV_shared_1_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                                            b_i, index, g_i, 64 * u + (tx - 256) % 8 * 8 + v
                                        ]
                                        KV_shared_1_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                                            b_i, index, g_i, D // 2 + 64 * u + (tx - 256) % 8 * 8 + v
                                        ]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_1[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v] = KV[
                                        b_i, index, g_i, D + (tx - 256) % 8 * 8 + v
                                    ]
                    T.cp_async_barrier_noinc(bar_k_1_ready[0])

                    if i_i + 1 < T.ceildiv(NI, 2):
                        for r in T.serial(4):
                            prefetch_indices_1[r] = Indices[b_i, s_i, g_i, ((i_i + 1) * 2 + 1) * BI + r * 16 + (tx - 256) // 8]

            elif tx < 128:
                # Check if 384 threads have already arrived at bar_q (phase0 completed),
                # if not continue waiting, otherwise pass through directly
                T.barrier_wait(bar_q, 0)

                # pre-arrive free barriers to indicate buffers are initially free
                # At the beginning of phase0, tells producer it can load data into both buffers
                T.barrier_arrive(bar_k_0_free[0])
                T.barrier_arrive(bar_k_1_free[0])

                # Consumer 0 (WG0): Responsible for Even Blocks and O_L (Left Half)
                T.set_max_nreg(216, 1)
                T.fill(sumexp_0, 0)
                for h_i in T.Parallel(H_per_block):
                    m_i_0[h_i] = -5e4
                T.fill(acc_o_l, 0)

                # Each iteration, two consumers cooperate to compute two BIs
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # --- Step 1: Compute S0 = Q @ K0^T (Even Block) ---
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                    T.fill(acc_s_0, 0)
                    T.gemm(Q_shared_l, KV_shared_0_l, acc_s_0, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_0_r, acc_s_0, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_0, acc_s_0, transpose_B=True, wg_wait=-1)

                    T.copy(m_i_0, m_i_prev_0)
                    T.wait_wgmma(0)

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        if not is_kv_valid[0, bi_i]:
                            acc_s_0[h_i, bi_i] = -5e4
                    T.reduce_max(acc_s_0, m_i_0, dim=1, clear=False)

                    # --- Step 2: Local Softmax Stats & Exchange ---
                    T.copy(m_i_0, row_max_shared_0)
                    T.barrier_arrive(bar_stats_0_ready)
                    # If consumer0 has received the local max from consumer1 at iter_i, this also means
                    # consumer1 has finished using S_0 passed by consumer0 at iter_i-1,
                    # so we can write to it directly without blocking below
                    T.barrier_wait(bar_stats_1_ready, (i_i & 1))
                    T.copy(row_max_shared_1, m_i_peer_0)

                    # Update global max and scale O
                    for h_i in T.Parallel(H_per_block):
                        m_i_0[h_i] = T.max(m_i_0[h_i], m_i_peer_0[h_i])

                    # Scale O_L
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= T.exp2((m_i_prev_0[h_i] - m_i_0[h_i]) * sm_scale)

                    # Scale SumExp
                    for h_i in T.Parallel(H_per_block):
                        sumexp_0[h_i] *= T.exp2((m_i_prev_0[h_i] - m_i_0[h_i]) * sm_scale)

                    # Compute P0 = exp(S0 - m_new)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s_0[h_i, bi_i] = T.exp2(acc_s_0[h_i, bi_i] * sm_scale - m_i_0[h_i] * sm_scale)

                    # Update SumExp with P0
                    T.reduce_sum(acc_s_0, sumexp_i_0, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp_0[h_i] += sumexp_i_0[h_i]

                    # --- Step 3: O_L += P0 @ V0_L (Self-Attention) ---
                    # Wait for S0 buffer to be free (consumed by peer in prev iter)
                    # T.barrier_wait(bar_S_0_free, (i_i & 1))
                    T.copy(acc_s_0, S_shared_0)
                    T.barrier_arrive(bar_S_0_ready)

                    T.gemm(S_shared_0, KV_shared_0_l, acc_o_l, transpose_B=False, wg_wait=-1)

                    # --- Step 4: O_L += P1 @ V1_L (Cross-Attention) ---
                    # Wait for P1 (S1) from peer
                    T.barrier_wait(bar_S_1_ready, (i_i & 1))

                    T.gemm(S_shared_1, KV_shared_1_l, acc_o_l, transpose_B=False, wg_wait=-1)

                    # NOTE: However, k_0 and k_1 are used by both consumer0 and consumer1, so this doesn't bring much performance improvement
                    # Except for the most recent async gemm (i.e., S_shared_1 @ KV_shared_1_k), all others need to wait to finish
                    T.wait_wgmma(1)
                    T.barrier_arrive(bar_k_0_free[0])
                    # Wait for all async gemms to finish
                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_k_1_free[0])

                T.copy(sumexp_0, row_sum_shared_0)
                T.barrier_arrive(bar_stats_0_ready)  # Reuse barrier
                T.barrier_wait(bar_stats_1_ready, T.ceildiv(NI, 2) & 1)
                T.copy(row_sum_shared_1, sumexp_i_0)  # Reuse sumexp_i buffer

                for h_i in T.Parallel(H_per_block):
                    sumexp_0[h_i] += sumexp_i_0[h_i]

                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_l[h_i, d_i] /= sumexp_0[h_i]

                for h_i in T.Parallel(H_per_block):
                    sumexp_0[h_i] = T.log2(sumexp_0[h_i]) + m_i_0[h_i] * sm_scale

                T.copy(acc_o_l, O_shared_l)
                T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0 : D // 2])
                T.copy(sumexp_0, Lse[b_i, s_i, H0:H1])  # Write LSE

            elif tx >= 128 and tx < 256:
                T.barrier_wait(bar_q, 0)

                # pre-arrive free barriers to indicate buffers are initially free
                # At the beginning of phase0, tells producer it can load data into both buffers
                T.barrier_arrive(bar_k_0_free[0])
                T.barrier_arrive(bar_k_1_free[0])

                # Consumer 1 (WG1): Responsible for Odd Blocks and O_R (Right Half)
                # NOTE: 256 * 216 + 128 * 72 = 64,512 < 65536 (H100 SM RegFile Limit),
                # setting more registers will cause a hang, all values must be multiples of 8
                T.set_max_nreg(216, 1)
                T.fill(sumexp_1, 0)
                for h_i in T.Parallel(H_per_block):
                    m_i_1[h_i] = -5e4
                T.fill(acc_o_r, 0)

                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # --- Step 1: Compute S1 = Q @ K1^T (Odd Block) ---
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                    T.fill(acc_s_1, 0)
                    T.gemm(Q_shared_l, KV_shared_1_l, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_1_r, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_1, acc_s_1, transpose_B=True, wg_wait=-1)

                    # --- Step 2: Local Softmax Stats & Exchange ---
                    T.copy(m_i_1, m_i_prev_1)
                    T.wait_wgmma(0)

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        if not is_kv_valid[1, bi_i]:
                            acc_s_1[h_i, bi_i] = -5e4

                    T.reduce_max(acc_s_1, m_i_1, dim=1, clear=False)
                    T.copy(m_i_1, row_max_shared_1)
                    T.barrier_arrive(bar_stats_1_ready)
                    T.barrier_wait(bar_stats_0_ready, (i_i & 1))
                    T.copy(row_max_shared_0, m_i_peer_1)

                    for h_i in T.Parallel(H_per_block):
                        m_i_1[h_i] = T.max(m_i_1[h_i], m_i_peer_1[h_i])

                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= T.exp2((m_i_prev_1[h_i] - m_i_1[h_i]) * sm_scale)

                    for h_i in T.Parallel(H_per_block):
                        sumexp_1[h_i] *= T.exp2((m_i_prev_1[h_i] - m_i_1[h_i]) * sm_scale)

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s_1[h_i, bi_i] = T.exp2(acc_s_1[h_i, bi_i] * sm_scale - m_i_1[h_i] * sm_scale)

                    T.reduce_sum(acc_s_1, sumexp_i_1, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp_1[h_i] += sumexp_i_1[h_i]

                    # --- Step 3: O_R += P1 @ V1_R (Self-Attention) ---
                    T.copy(acc_s_1, S_shared_1)

                    T.barrier_arrive(bar_S_1_ready)

                    T.gemm(S_shared_1, KV_shared_1_r, acc_o_r, transpose_B=False, wg_wait=-1)

                    # --- Step 4: O_R += P0 @ V0_R (Cross-Attention) ---
                    T.barrier_wait(bar_S_0_ready, (i_i & 1))

                    T.gemm(S_shared_0, KV_shared_0_r, acc_o_r, transpose_B=False, wg_wait=-1)

                    T.wait_wgmma(1)
                    T.barrier_arrive(bar_k_1_free[0])
                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_k_0_free[0])

                T.copy(sumexp_1, row_sum_shared_1)
                T.barrier_arrive(bar_stats_1_ready)
                T.barrier_wait(bar_stats_0_ready, T.ceildiv(NI, 2) & 1)
                T.copy(row_sum_shared_0, sumexp_i_1)

                for h_i in T.Parallel(H_per_block):
                    sumexp_1[h_i] += sumexp_i_1[h_i]

                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_r[h_i, d_i] /= sumexp_1[h_i]

                T.copy(acc_o_r, O_shared_r)
                T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D // 2 : D])

    return main


def sparse_mla_fwd_interface(
    q, kv, indices, q_start_index_s, kv_stride, sm_scale=None, is_casual=True, return_kernel=False, print_kernel=False
):
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = 512

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    if q_start_index_s != 0:
        assert q_start_index_s > kv_stride, (
            "If it is because each cp has too short length, you should fix the logic involving CP0 (cp_rank == 0), to make sure q with pos < KV_Stride - 1 is masked (or you may just ignore how this is handled if nan in these q's Out would not effect others, which is reported to be likely to happen by wangding)"
        )
    CP0 = q_start_index_s == 0

    # Compile the kernel
    kernel = sparse_mla_fwd(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride, kv_group, sm_scale, is_casual, CP0)

    if print_kernel:
        print(kernel.get_kernel_source())

    if return_kernel:
        return kernel

    (
        out,
        lse,
    ) = kernel(q, kv, indices, torch.tensor([q_start_index_s], dtype=torch.int32, device="cuda"))
    if q_start_index_s == 0 and kv_stride > 1:
        # Set the output of the first (kv_stride - 1) positions to 0, since they cannot see any kv so no computation was performed
        out[:, : kv_stride - 1, :, :] = 0
    return out, lse


def ref_sparse_mla_fwd_interface(q, kv, indices, q_start_index_s, kv_stride=1, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape
    if q_start_index_s is None:
        q_start_index_s = sk * kv_stride - sq

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    num_kv_per_index = 1
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(q_start_index_s, sq + q_start_index_s, dtype=torch.int32, device="cuda").view(
        -1, 1
    ) >= torch.arange(kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32, device="cuda").view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def test_sparse_mla_fwd_pipelined(
    B=1,
    S=4096,
    SKV=8192,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    # Offset of query in global sequence position (or relative to kv)
    q_start_s_index=2048,
    check_correctness=True,
    profile=False,
):
    KV_stride = 1

    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True) / 10
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True) / 10
    q_start_s_index_t = torch.tensor([q_start_s_index], dtype=torch.int32, device="cuda")

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                # Add offset q_start_s_index to convert to global sequence position
                i_i = torch.randperm(min(max(1, ((t + q_start_s_index) // KV_stride)), SKV))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    print("index generation finished")

    kernel = sparse_mla_fwd_interface(q, kv, indices, q_start_s_index, KV_stride, return_kernel=True, print_kernel=True)

    def fn():
        return kernel(q, kv, indices, q_start_s_index_t)

    if check_correctness:
        tl_out, tl_lse = fn()
        assert KV_stride == 1, "KV_stride > 1 not supported"
        # if q_start_s_index == 0 and KV_stride > 1:
        #     tl_out[:, :KV_stride - 1, :, :] = 0
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, q_start_s_index, KV_stride)
        print(f"tl_out: {tl_out}")
        print(f"ref_out: {ref_out}")
        torch.testing.assert_close(tl_out, ref_out, rtol=1e-3, atol=1e-3)

    if profile:
        print("Profiling mode: running minimal iterations (1 warmup + 1 run)...")
        fn()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        return

    from tilelang.profiler import do_bench

    ms = do_bench(
        fn,
        rep=20,
        warmup=10,
    )
    print(f"Average time: {ms:.3f} ms")
    print(f"fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    tflops = (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12
    print(f"fwd tflops = {tflops:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.test_correctness:
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 1, 1024, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
        test_sparse_mla_fwd_pipelined(B, S, SKV, H, HKV, DQK, DV, topk, dtype, check_correctness=True, profile=args.profile)
    else:
        # Prefill Benchmark: long context
        print(" --- Prefill Benchmark --- ")
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 2, 4096, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
        test_sparse_mla_fwd_pipelined(
            B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start_s_index=4096, check_correctness=False, profile=args.profile
        )

        # Decode Benchmark: large batch size, high throughput generation
        print("\n --- Decode Benchmark --- ")
        # Increase batch size to saturate h100 for decode
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 128 * 16, 2, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
        test_sparse_mla_fwd_pipelined(
            B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start_s_index=2048 + 4096, check_correctness=False, profile=args.profile
        )
