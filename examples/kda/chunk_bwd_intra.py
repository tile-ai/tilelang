# Reference: FLA_KDA/fla_chunk_intra.py
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune

from FLA_KDA.fla_chunk_intra import chunk_kda_bwd_intra
from FLA_KDA.cumsum import chunk_local_cumsum
from test_utils_kda import compare_tensors, do_bench

import torch

torch.random.manual_seed(0)
torch.set_printoptions(profile="full")


def prepare_input(
    B,
    S,
    H,
    DK,
    chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
):
    BT = chunk_size
    q = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    k = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    g = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()
    beta = torch.randn(B, S, H, dtype=input_dtype).cuda()

    # dAqk and dAkk are gradients w.r.t. Aqk and Akk
    # Shape: (B, S, H, BT)
    dAqk = torch.randn(B, S, H, BT, dtype=input_dtype).cuda()
    dAkk = torch.randn(B, S, H, BT, dtype=input_dtype).cuda()

    # Initial gradients (will be updated by the kernel)
    dq = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    dk = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    db = torch.randn(B, S, H, dtype=input_dtype).cuda()
    dg = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()

    return q, k, g, beta, dAqk, dAkk, dq, dk, db, dg


def prepare_output(
    B,
    S,
    H,
    DK,
    chunk_size,
    NK,
    output_dtype,
    gate_dtype,
    state_dtype,
):
    dq = torch.empty(B, S, H, DK, dtype=output_dtype).cuda()
    dk = torch.empty(B, S, H, DK, dtype=output_dtype).cuda()
    db = torch.empty(NK, B, S, H, dtype=output_dtype).cuda()
    dg = torch.empty(B, S, H, DK, dtype=gate_dtype).cuda()
    return dq, dk, db, dg


def get_configs():
    import itertools

    threads = [32, 64, 128, 256]
    num_stages = [0, 1, 2, 3]
    _configs = list(itertools.product(threads, num_stages))

    configs = [{"threads": c[0], "num_stages": c[1]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=5, rep=5)
@tilelang.jit(
    out_idx=[-4, -3, -2, -1],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
)
def tilelang_chunk_bwd_intra(
    # task config
    B,
    S,
    H,
    DK,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    # kernel config
    block_DK,
    block_BC=16,
    threads=128,
    num_stages=0,
):
    BT = chunk_size
    BC = block_BC  # sub-chunk size, typically 16

    NC = BT // BC  # number of sub-chunks
    NT = T.ceildiv(S, BT)
    NK = T.ceildiv(DK, block_DK)  # number of K blocks

    K_shape = (B, S, H, DK)
    Beta_shape = (B, S, H)
    G_shape = (B, S, H, DK)
    BT_shape = (B, S, H, BT)  # for dAqk and dAkk

    dq_shape = (B, S, H, DK)
    dk_shape = (B, S, H, DK)
    db_shape = (B, S, H)
    db2_shape = (NK, B, S, H)
    dg_shape = (B, S, H, DK)

    @T.prim_func
    def kernel(
        # input
        q: T.Tensor(K_shape, dtype=input_dtype),
        k: T.Tensor(K_shape, dtype=input_dtype),
        g: T.Tensor(G_shape, dtype=gate_dtype),
        beta: T.Tensor(Beta_shape, dtype=input_dtype),
        dAqk: T.Tensor(BT_shape, dtype=input_dtype),
        dAkk: T.Tensor(BT_shape, dtype=input_dtype),
        dq: T.Tensor(dq_shape, dtype=input_dtype),
        dk: T.Tensor(dk_shape, dtype=input_dtype),
        db: T.Tensor(db_shape, dtype=input_dtype),
        dg: T.Tensor(dg_shape, dtype=gate_dtype),
        # output
        dq2: T.Tensor(dq_shape, dtype=output_dtype),
        dk2: T.Tensor(dk_shape, dtype=output_dtype),
        db2: T.Tensor(db2_shape, dtype=output_dtype),
        dg2: T.Tensor(dg_shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(DK, block_DK) * NC, NT, B * H, threads=threads) as (i_kc, i_t, i_bh):
            i_k, i_i = i_kc // NC, i_kc % NC
            bb, bh = i_bh // H, i_bh % H

            # actual sub-chunk index
            i_ti = i_t * BT + i_i * BC

            # current sub-chunk data
            q_shared = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            k_shared = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            beta_shared = T.alloc_shared((BC,), dtype=input_dtype)
            g_current_shared = T.alloc_shared((BC, block_DK), dtype=gate_dtype)
            gn_shared = T.alloc_shared((block_DK,), dtype=gate_dtype)  # last token's g in current sub-chunk

            dq_shared = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            dk_shared = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            dg_shared = T.alloc_shared((BC, block_DK), dtype=gate_dtype)

            # Allocate fragments
            dq2_fragment = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            dk2_fragment = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            dg2_fragment = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            db_fragment = T.alloc_fragment((BC,), dtype=accum_dtype)

            # Initialize fragments
            T.clear(dq2_fragment)
            T.clear(dk2_fragment)
            T.clear(dg2_fragment)
            T.clear(db_fragment)

            # Temporary shared memory for previous sub-chunks
            k_prev_shared = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            g_prev_shared = T.alloc_shared((BC, block_DK), dtype=gate_dtype)
            dAqk_prev_shared = T.alloc_shared((BC, BC), dtype=input_dtype)
            dAkk_prev_shared = T.alloc_shared((BC, BC), dtype=input_dtype)

            # Temporary fragment for b_kg computation
            kg_fragment = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)

            kj_shared = T.alloc_shared((block_DK,), dtype=T.float32)
            gkj_shared = T.alloc_shared((block_DK,), dtype=T.float32)
            kgj_fragment = T.alloc_fragment((BC, block_DK), dtype=T.float32)
            dAqk_col = T.alloc_shared((BC,), dtype=input_dtype)
            dAkk_col = T.alloc_shared((BC,), dtype=input_dtype)

            # Load g, q, k for current sub-chunk
            T.copy(q[bb, i_ti : i_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], q_shared)
            T.copy(k[bb, i_ti : i_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], k_shared)
            T.copy(g[bb, i_ti : i_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], g_current_shared)
            T.copy(beta[bb, i_ti : i_ti + BC, bh], beta_shared)

            if i_i > 0:
                chunk_first_idx = i_ti  # chunk first token idx

                T.copy(g[bb, chunk_first_idx, bh, i_k * block_DK : (i_k + 1) * block_DK], gn_shared)  # Get the first token's g value (b_gn)

                # Loop over previous sub-chunks (i_j from 0 to i_i-1)
                # Since i_i is computed from i_kc % NC and NC is small, we can use conditional blocks
                # Process each possible previous sub-chunk with conditional execution
                for i_j in T.Pipelined(i_i, num_stages=num_stages):  # i_j is index ofprevious sub_chunks
                    prev_ti = i_t * BT + i_j * BC
                    T.copy(k[bb, prev_ti : prev_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], k_prev_shared)
                    T.copy(g[bb, prev_ti : prev_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], g_prev_shared)

                    T.copy(dAqk[bb, i_ti : i_ti + BC, bh, i_j * BC : (i_j + 1) * BC], dAqk_prev_shared)
                    T.copy(dAkk[bb, i_ti : i_ti + BC, bh, i_j * BC : (i_j + 1) * BC], dAkk_prev_shared)

                    for i_bc, i_k2 in T.Parallel(BC, block_DK):
                        kg_fragment[i_bc, i_k2] = k_prev_shared[i_bc, i_k2] * T.exp2(gn_shared[i_k2] - g_prev_shared[i_bc, i_k2])

                    T.gemm(dAqk_prev_shared, kg_fragment, dq2_fragment, clear_accum=False)
                    T.gemm(dAkk_prev_shared, kg_fragment, dk2_fragment, clear_accum=False)

                for i_bc, i_k2 in T.Parallel(BC, block_DK):
                    gqn = T.exp2(g_current_shared[i_bc, i_k2] - gn_shared[i_k2])
                    dq2_fragment[i_bc, i_k2] = dq2_fragment[i_bc, i_k2] * gqn
                    dk2_fragment[i_bc, i_k2] = dk2_fragment[i_bc, i_k2] * gqn

            # Process current sub-chunk diagonal
            loop_length = T.min(BC, S - i_t * BT - i_i * BC)
            for j in T.Pipelined(loop_length, num_stages=num_stages):
                token_j_idx = i_ti + j

                T.copy(k[bb, token_j_idx, bh, i_k * block_DK : (i_k + 1) * block_DK], kj_shared)
                T.copy(g[bb, token_j_idx, bh, i_k * block_DK : (i_k + 1) * block_DK], gkj_shared)
                T.copy(dAqk[bb, i_ti : i_ti + BC, bh, i_i * BC + j], dAqk_col)
                T.copy(dAkk[bb, i_ti : i_ti + BC, bh, i_i * BC + j], dAkk_col)

                for i_bc, i_k2 in T.Parallel(BC, block_DK):
                    kgj_fragment[i_bc, i_k2] = kj_shared[i_k2] * T.exp2(g_current_shared[i_bc, i_k2] - gkj_shared[i_k2])
                    dq2_fragment[i_bc, i_k2] += T.if_then_else(i_bc >= j, dAqk_col[i_bc] * kgj_fragment[i_bc, i_k2], 0.0)
                    dk2_fragment[i_bc, i_k2] += T.if_then_else(i_bc >= j, dAkk_col[i_bc] * kgj_fragment[i_bc, i_k2], 0.0)

            # Compute b_db = sum(b_dk2 * b_k, dim=1)
            dk2_k_fragment = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            for i_bc, i_k2 in T.Parallel(BC, block_DK):
                dk2_k_fragment[i_bc, i_k2] = dk2_fragment[i_bc, i_k2] * k_shared[i_bc, i_k2]
            T.reduce_sum(dk2_k_fragment, db_fragment, dim=1, clear=True)

            # b_dk2 *= b_b[:, None]
            for i_bc, i_k2 in T.Parallel(BC, block_DK):
                dk2_fragment[i_bc, i_k2] = dk2_fragment[i_bc, i_k2] * beta_shared[i_bc]

            # Compute b_dg2 = b_q * b_dq2 (before adding dq to dq2)
            for i_bc, i_k2 in T.Parallel(BC, block_DK):
                dg2_fragment[i_bc, i_k2] = q_shared[i_bc, i_k2] * dq2_fragment[i_bc, i_k2]

            # Load dq and compute b_dq2 = b_dq2 + b_dq
            T.copy(dq[bb, i_ti : i_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], dq_shared)
            for i_bc, i_k2 in T.Parallel(BC, block_DK):
                dq2_fragment[i_bc, i_k2] = dq2_fragment[i_bc, i_k2] + dq_shared[i_bc, i_k2]

            # # Store results
            T.copy(dq2_fragment, dq2[bb, i_ti : i_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK])
            T.copy(db_fragment, db2[i_k, bb, i_ti : i_ti + BC, bh])

            # Initialize dkt_fragment for processing subsequent sub-chunks and lower triangular part
            dkt_fragment = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            T.clear(dkt_fragment)

            # Temporary shared memory for subsequent sub-chunks
            q_next_shared = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            k_next_shared = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            g_next_shared = T.alloc_shared((BC, block_DK), dtype=gate_dtype)
            beta_next_shared = T.alloc_shared((BC,), dtype=input_dtype)
            dAqk_next_shared = T.alloc_shared((BC, BC), dtype=input_dtype)
            dAkk_next_shared = T.alloc_shared((BC, BC), dtype=input_dtype)

            # Temporary fragments for computation
            gkn_shared = T.alloc_shared((BC, block_DK), dtype=accum_dtype)
            qg_shared = T.alloc_shared((BC, block_DK), dtype=accum_dtype)
            kbg_fragment = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            kbg_shared = T.alloc_shared((BC, block_DK), dtype=accum_dtype)
            dkt_temp_fragment = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            # T.use_swizzle(10)

            NC_actual = T.min(NC, T.ceildiv(S - i_t * BT, BC))  # Process subsequent sub-chunks (i_j from i_i+1 to NC-1)
            if i_i < NC_actual - 1:
                # Get the last token's g value in current sub-chunk
                chunk_last_idx = T.min(S, i_ti + BC) - 1
                gn_last_shared = T.alloc_shared((block_DK,), dtype=gate_dtype)
                T.copy(g[bb, chunk_last_idx, bh, i_k * block_DK : (i_k + 1) * block_DK], gn_last_shared)

                # Loop over subsequent sub-chunks
                for i_j in T.Pipelined(i_i + 1, NC_actual, num_stages=num_stages):
                    i_tj = i_t * BT + i_j * BC

                    T.copy(q[bb, i_tj : i_tj + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], q_next_shared)
                    T.copy(k[bb, i_tj : i_tj + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], k_next_shared)
                    T.copy(g[bb, i_tj : i_tj + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], g_next_shared)
                    T.copy(beta[bb, i_tj : i_tj + BC, bh], beta_next_shared)

                    T.copy(dAqk[bb, i_tj : i_tj + BC, bh, i_i * BC : (i_i + 1) * BC], dAqk_next_shared)  # [BC, BC] need transpose
                    T.copy(dAkk[bb, i_tj : i_tj + BC, bh, i_i * BC : (i_i + 1) * BC], dAkk_next_shared)  # [BC, BC] need transpose

                    for i_bc, i_k2 in T.Parallel(BC, block_DK):
                        # kbg = k * beta
                        kbg_fragment[i_bc, i_k2] = k_next_shared[i_bc, i_k2] * beta_next_shared[i_bc]
                        gkn_shared[i_bc, i_k2] = T.if_then_else(
                            i_tj + i_bc < S, T.exp2(g_next_shared[i_bc, i_k2] - gn_last_shared[i_k2]), 0.0
                        )

                    # Compute qg and kbg
                    for i_bc, i_k2 in T.Parallel(BC, block_DK):
                        qg_shared[i_bc, i_k2] = q_next_shared[i_bc, i_k2] * gkn_shared[i_bc, i_k2]
                        kbg_shared[i_bc, i_k2] = kbg_fragment[i_bc, i_k2] * gkn_shared[i_bc, i_k2]

                    # Accumulate: dkt += dAqk^T @ qg + dAkk^T @ kbg
                    # Use transpose_A=True because dAqk/dAkk are loaded in (T, BT) layout but we need (BT, T) for gemm
                    T.gemm(dAqk_next_shared, qg_shared, dkt_temp_fragment, transpose_A=True, clear_accum=True)
                    T.gemm(dAkk_next_shared, kbg_shared, dkt_temp_fragment, transpose_A=True, clear_accum=False)

                    for i_bc, i_k2 in T.Parallel(BC, block_DK):
                        dkt_fragment[i_bc, i_k2] = dkt_fragment[i_bc, i_k2] + dkt_temp_fragment[i_bc, i_k2]

                # Scale dkt by exp2(gn_last - g_current)
                for i_bc, i_k2 in T.Parallel(BC, block_DK):
                    g_scale = T.exp2(gn_last_shared[i_k2] - g_current_shared[i_bc, i_k2])
                    dkt_fragment[i_bc, i_k2] = dkt_fragment[i_bc, i_k2] * g_scale

            # Process lower triangular part of current sub-chunk diagonal
            # This corresponds to j <= i_bc in the diagonal block
            qj_shared = T.alloc_shared((block_DK,), dtype=T.float32)
            kj_shared_lower = T.alloc_shared((block_DK,), dtype=T.float32)
            gj_shared_lower = T.alloc_shared((block_DK,), dtype=T.float32)
            bj_local = T.alloc_local((1), dtype=input_dtype)
            dAqk_col_lower = T.alloc_shared((BC,), dtype=input_dtype)
            dAkk_col_lower = T.alloc_shared((BC,), dtype=input_dtype)

            gkq_fragment = T.alloc_fragment((BC, block_DK), dtype=T.float32)
            # dkt_lower_temp = T.alloc_fragment((BC, block_DK), dtype=T.float32)
            kbj_fragment = T.alloc_fragment((block_DK,), dtype=T.float32)

            max_token_j_idx = T.min(S, i_ti + BC)
            for j in T.Pipelined(BC, num_stages=num_stages):
                token_j_idx = i_ti + j

                if token_j_idx < max_token_j_idx:
                    T.copy(q[bb, token_j_idx, bh, i_k * block_DK : (i_k + 1) * block_DK], qj_shared)  # [BK]
                    T.copy(k[bb, token_j_idx, bh, i_k * block_DK : (i_k + 1) * block_DK], kj_shared_lower)
                    T.copy(g[bb, token_j_idx, bh, i_k * block_DK : (i_k + 1) * block_DK], gj_shared_lower)

                    bj_local[0] = beta[bb, token_j_idx, bh]
                    T.copy(dAqk[bb, token_j_idx, bh, i_i * BC : (i_i + 1) * BC], dAqk_col_lower)  # [BC]
                    T.copy(dAkk[bb, token_j_idx, bh, i_i * BC : (i_i + 1) * BC], dAkk_col_lower)

                    # Compute kbj = kj * bj
                    for i_k2 in T.Parallel(block_DK):
                        kbj_fragment[i_k2] = kj_shared_lower[i_k2] * bj_local[0]
                    # Compute gkq = exp2(gj - g_current)
                    for i_bc, i_k2 in T.Parallel(BC, block_DK):
                        gkq_fragment[i_bc, i_k2] = T.exp2(gj_shared_lower[i_k2] - g_current_shared[i_bc, i_k2])

                    # Accumulate: dkt += (dAkk * kbj + dAqk * qj) * gkq for i_bc <= j
                    for i_bc, i_k2 in T.Parallel(BC, block_DK):
                        dkt_fragment[i_bc, i_k2] += T.if_then_else(
                            i_bc <= j,
                            (dAkk_col_lower[i_bc] * kbj_fragment[i_k2] + dAqk_col_lower[i_bc] * qj_shared[i_k2]) * gkq_fragment[i_bc, i_k2],
                            0.0,
                        )

            # Load dk and dg
            T.copy(dk[bb, i_ti : i_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], dk_shared)
            T.copy(dg[bb, i_ti : i_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK], dg_shared)

            # Update dg2: dg2 += (dk2 - dkt) * k + dg
            for i_bc, i_k2 in T.Parallel(BC, block_DK):
                dg2_fragment[i_bc, i_k2] = (
                    dg2_fragment[i_bc, i_k2]
                    + (dk2_fragment[i_bc, i_k2] - dkt_fragment[i_bc, i_k2]) * k_shared[i_bc, i_k2]
                    + dg_shared[i_bc, i_k2]
                )

            # Update dk2: dk2 += dk + dkt
            for i_bc, i_k2 in T.Parallel(BC, block_DK):
                dk2_fragment[i_bc, i_k2] += dk_shared[i_bc, i_k2] + dkt_fragment[i_bc, i_k2]

            # Store dk2 and dg2
            T.copy(dk2_fragment, dk2[bb, i_ti : i_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK])
            T.copy(dg2_fragment, dg2[bb, i_ti : i_ti + BC, bh, i_k * block_DK : (i_k + 1) * block_DK])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    threads=128,
    num_stages=0,
    cu_seqlens=None,
    chunk_indices=None,
):
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg = prepare_input(
        B,
        S,
        H,
        DK,
        chunk_size,
        getattr(torch, input_dtype),
        getattr(torch, output_dtype),
        getattr(torch, accum_dtype),
        getattr(torch, gate_dtype),
        getattr(torch, state_dtype),
    )

    # Reference implementation
    dq_ref, dk_ref, db_ref, dg_ref = chunk_kda_bwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
    )
    block_DK = min(64, tilelang.math.next_power_of_2(DK))
    NK = (DK + block_DK - 1) // block_DK
    # TileLang implementation
    kernel = tilelang_chunk_bwd_intra(
        B=B,
        S=S,
        H=H,
        DK=DK,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        accum_dtype=accum_dtype,
        gate_dtype=gate_dtype,
        state_dtype=state_dtype,
        chunk_size=chunk_size,
        block_DK=block_DK,
    )

    dq_tilelang, dk_tilelang, db_tilelang, dg_tilelang = prepare_output(
        B, S, H, DK, chunk_size, NK, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )
    dq_tilelang, dk_tilelang, db_tilelang, dg_tilelang = kernel(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg)
    db_tilelang = db_tilelang.sum(0).add_(db)
    dg_tilelang = chunk_local_cumsum(
        dg_tilelang,
        chunk_size=chunk_size,
        reverse=True,
    )

    compare_tensors("dq", dq_tilelang, dq_ref)
    compare_tensors("dk", dk_tilelang, dk_ref)
    compare_tensors("db", db_tilelang, db_ref)
    compare_tensors("dg", dg_tilelang, dg_ref)

    fla_time = do_bench(
        chunk_kda_bwd_intra,
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
    )
    tilelang_time = do_bench(kernel, q, k, g, beta, dAqk, dAkk, dq, dk, db, dg)
    print(f"Fla time: {fla_time}")
    print(f"Tilelang time: {tilelang_time}")


def main():
    DK = 128
    run_test(
        B=1,
        S=8192,
        H=8,
        DK=DK,
        input_dtype=T.float32,
        output_dtype=T.float32,
        accum_dtype=T.float32,
        gate_dtype=T.float32,
        state_dtype=T.float32,
        chunk_size=64,
        threads=128,
        num_stages=0,
    )


if __name__ == "__main__":
    main()
