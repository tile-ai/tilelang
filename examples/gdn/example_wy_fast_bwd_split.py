# Reference: fla/ops/gated_delta_rule/wy_fast.py

import sys  # noqa: F401

import tilelang
import tilelang.language as T

# Add your fla repository path to sys.path
# Currently we use the fla repository from the flash-linear-attention project at commit id 00000000
# sys.path.insert(0, "/home/tzj/flash-linear-attention")
try:
    import fla

    print(fla.__file__)
    from fla.ops.gated_delta_rule.wy_fast import bwd_prepare_wy_repr
except ImportError:
    print("fla not found, using tilelang implementation")
    fla = None

import torch
import torch.nn.functional as F

torch.random.manual_seed(0)
torch.set_printoptions(profile="full")


def prepare_input_fake(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
):
    BS = chunk_size
    K = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    V = torch.ones(B, S, H, DV, dtype=input_dtype).cuda()
    Beta = torch.ones(B, S, H, dtype=input_dtype).cuda()
    G = torch.ones(B, S, H, dtype=gate_dtype).cuda()
    A = torch.ones(B, S, H, BS, dtype=input_dtype).cuda()
    dw = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    du = torch.ones(B, S, H, DV, dtype=input_dtype).cuda()
    return K, V, Beta, G, A, dw, du


def prepare_input(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
):
    BS = chunk_size
    K = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    K = F.normalize(K, dim=-1, p=2)
    V = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    V = F.normalize(V, dim=-1, p=2)
    Beta = torch.randn(B, S, H, dtype=input_dtype).cuda()
    G = torch.randn(B, S, H, dtype=gate_dtype).cuda()
    A = torch.randn(B, S, H, BS, dtype=input_dtype).cuda()
    dw = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    du = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    return K, V, Beta, G, A, dw, du


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    output_dtype,
    gate_dtype,
    state_dtype,
):
    dk = torch.empty(B, S, H, DK, dtype=output_dtype).cuda()
    dv = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    dbeta = torch.empty(B, S, H, dtype=output_dtype).cuda()
    dg = torch.empty(B, S, H, dtype=gate_dtype).cuda()
    return dk, dv, dbeta, dg


@tilelang.jit(
    out_idx=[-5, -4, -3, -2, -1],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def tilelang_wy_fast_bwd(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    # kernel config
    block_DK=64,
    block_DV=64,
    threads=128,
    num_stages=0,
):
    block_S = chunk_size
    BS = block_S

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    Beta_shape = (B, S, H)
    G_shape = (B, S, H)
    A_shape = (B, S, H, BS)
    dw_shape = (B, S, H, DK)
    du_shape = (B, S, H, DV)

    dk_shape = (B, S, H, DK)
    dv_shape = (B, S, H, DV)
    dbeta_shape = (B, S, H)
    dg_shape = (B, S, H)
    dA_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        # input
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        A: T.Tensor(A_shape, dtype=input_dtype),
        dw: T.Tensor(dw_shape, dtype=input_dtype),
        du: T.Tensor(du_shape, dtype=input_dtype),
        # output
        dA: T.Tensor(dA_shape, dtype=input_dtype),
        dk: T.Tensor(dk_shape, dtype=output_dtype),
        dv: T.Tensor(dv_shape, dtype=output_dtype),
        dbeta: T.Tensor(dbeta_shape, dtype=output_dtype),
        dg: T.Tensor(dg_shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H

            A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared_beta_g = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            V_shared_beta = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype)
            G_shared = T.alloc_shared((block_S,), dtype=gate_dtype)
            G_shared_exp = T.alloc_shared((block_S,), dtype=gate_dtype)
            dw_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            du_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)

            dA_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dk_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dk_fragment_beta_g = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dv_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dv_fragment_beta = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dbeta_fragment_k = T.alloc_fragment((block_S,), dtype=accum_dtype)
            dbeta_fragment_v = T.alloc_fragment((block_S,), dtype=accum_dtype)
            dbeta_fragment_reduce_tmpk = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dbeta_fragment_reduce_tmpv = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dg_fragment = T.alloc_fragment((block_S,), dtype=gate_dtype)
            dg_fragment_reduce_tmp = T.alloc_fragment((block_S, block_DK), dtype=gate_dtype)

            T.use_swizzle(10)

            T.clear(dA_fragment)
            T.clear(dk_fragment)
            T.clear(dk_fragment_beta_g)
            T.clear(dv_fragment)
            T.clear(dv_fragment_beta)
            T.clear(dbeta_fragment_k)
            T.clear(dbeta_fragment_v)
            T.clear(dg_fragment)

            T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, :], A_shared)
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]
                G_shared[i_s] = G[bb, bs * block_S + i_s, bh]
                G_shared_exp[i_s] = T.exp(G[bb, bs * block_S + i_s, bh])

            # Update dk
            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], K_shared)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    K_shared_beta_g[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s] * G_shared_exp[i_s]
                T.copy(dw[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dw_shared)
                T.gemm(dw_shared, K_shared_beta_g, dA_fragment, transpose_B=True)
                T.gemm(A_shared, dw_shared, dk_fragment_beta_g, clear_accum=True, transpose_A=True)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dk_fragment[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * Beta_shared[i_s] * G_shared_exp[i_s]
                # for i_s, i_k2 in T.Parallel(block_S, block_DK):
                #     dbeta_fragment[i_s] = dbeta_fragment[i_s] + dk_fragment_beta_g[i_s, i_k2] * K_shared[i_s, i_k2] * G_shared_exp[i_s]
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dbeta_fragment_reduce_tmpk[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * K_shared[i_s, i_k2] * G_shared_exp[i_s]
                T.reduce_sum(dbeta_fragment_reduce_tmpk, dbeta_fragment_k, dim=1, clear=False)

                # for i_s, i_k2 in T.Parallel(block_S, block_DK):
                #     dg_fragment[i_s] = dg_fragment[i_s] + dk_fragment_beta_g[i_s, i_k2] * K_shared[i_s, i_k2] * G_shared_exp[i_s] * Beta_shared[i_s]
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dg_fragment_reduce_tmp[i_s, i_k2] = (
                        dk_fragment_beta_g[i_s, i_k2] * K_shared[i_s, i_k2] * G_shared_exp[i_s] * Beta_shared[i_s]
                    )
                T.reduce_sum(dg_fragment_reduce_tmp, dg_fragment, dim=1, clear=False)

                # correct dk
                T.copy(dk_fragment, dk[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])

            # Update dv
            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(V[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], V_shared)
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    V_shared_beta[i_s, i_v2] = V_shared[i_s, i_v2] * Beta_shared[i_s]
                T.copy(du[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], du_shared)
                T.gemm(du_shared, V_shared_beta, dA_fragment, transpose_B=True)
                T.gemm(A_shared, du_shared, dv_fragment_beta, clear_accum=True, transpose_A=True)
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    dv_fragment[i_s, i_v2] = dv_fragment_beta[i_s, i_v2] * Beta_shared[i_s]
                # for i_s, i_v2 in T.Parallel(block_S, block_DV):
                #     dbeta_fragment[i_s] = dbeta_fragment[i_s] + dv_fragment_beta[i_s, i_v2] * V_shared[i_s, i_v2]
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    dbeta_fragment_reduce_tmpv[i_s, i_v2] = dv_fragment_beta[i_s, i_v2] * V_shared[i_s, i_v2]
                T.reduce_sum(dbeta_fragment_reduce_tmpv, dbeta_fragment_v, dim=1, clear=False)

                T.copy(dv_fragment, dv[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV])

            # Temporary store dbeta, dg and dA
            for i_s in T.Parallel(block_S):
                dbeta[bb, bs * block_S + i_s, bh] = dbeta_fragment_k[i_s] + dbeta_fragment_v[i_s]
                dg[bb, bs * block_S + i_s, bh] = dg_fragment[i_s]
            # correct dA
            T.copy(dA_fragment, dA[bb, bs * block_S : (bs + 1) * block_S, bh, :])

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def tilelang_wy_fast_bwd_split(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    # kernel config
    block_DK=64,
    block_DV=64,
    threads=128,
    num_stages=0,
):
    block_S = chunk_size
    BS = block_S

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    Beta_shape = (B, S, H)
    G_shape = (B, S, H)
    A_shape = (B, S, H, BS)
    dw_shape = (B, S, H, DK)
    du_shape = (B, S, H, DV)

    dk_shape = (B, S, H, DK)
    dv_shape = (B, S, H, DV)
    dbeta_shape = (B, S, H)
    dA_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        # input
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        A: T.Tensor(A_shape, dtype=input_dtype),
        dw: T.Tensor(dw_shape, dtype=input_dtype),
        du: T.Tensor(du_shape, dtype=input_dtype),
        dA: T.Tensor(dA_shape, dtype=input_dtype),
        dk: T.Tensor(dk_shape, dtype=output_dtype),
        dv: T.Tensor(dv_shape, dtype=output_dtype),
        dbeta_k: T.Tensor(dbeta_shape, dtype=output_dtype),
        dg_A_positive: T.Tensor(dA_shape, dtype=gate_dtype),
        dg_A_negative: T.Tensor(dA_shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H

            A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared_beta = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            dA_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            dA_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dA_A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dA_A_fragment_1 = T.alloc_fragment((block_S,), dtype=accum_dtype)
            dA_A_fragment_2 = T.alloc_fragment((block_S,), dtype=accum_dtype)
            dk_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            dk_shared_beta = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            dk_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dk_fragment_beta = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype)
            dbeta_fragment_reduce_tmpk = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dbeta_fragment_k = T.alloc_fragment((block_S,), dtype=accum_dtype)
            G_shared = T.alloc_shared((block_S,), dtype=gate_dtype)
            G_shared_exp = T.alloc_shared((block_S,), dtype=gate_dtype)

            T.clear(dbeta_fragment_reduce_tmpk)
            T.clear(dbeta_fragment_k)
            T.clear(dA_A_fragment_1)
            T.clear(dA_A_fragment_2)

            T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, :], A_shared)
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]
                G_shared[i_s] = G[bb, bs * block_S + i_s, bh]
            for i_s in T.Parallel(block_S):
                G_shared_exp[i_s] = T.exp(G_shared[i_s])

            # Load intermediate results
            # for i_s in T.Parallel(block_S):
            # dbeta_fragment[i_s] = dbeta[bb, bs * block_S + i_s, bh]
            # dg_fragment[i_s] = dg[bb, bs * block_S + i_s, bh]
            T.copy(dA[bb, bs * block_S : (bs + 1) * block_S, bh, :], dA_shared)
            # T.copy(dA_shared, dA[bb, bs * block_S:(bs + 1) * block_S, bh, :])

            # Update dA
            T.copy(dA_shared, dA_fragment)

            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                with T.If(i_s1 <= i_s2):  # noqa: SIM117
                    with T.Then():
                        dA_fragment[i_s1, i_s2] = 0
            T.copy(dA_fragment, dA_shared)
            T.gemm(dA_shared, A_shared, dA_fragment, clear_accum=True, transpose_B=True)
            T.copy(dA_fragment, dA_shared)
            T.gemm(A_shared, dA_shared, dA_fragment, clear_accum=True, transpose_A=True)
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                with T.If(i_s1 <= i_s2):
                    with T.Then():
                        dA_fragment[i_s1, i_s2] = 0
                    with T.Else():
                        dA_fragment[i_s1, i_s2] = -dA_fragment[i_s1, i_s2]

            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                with T.If(G[bb, bs * block_S + i_s1, bh] - G[bb, bs * block_S + i_s2, bh] <= 0):
                    with T.Then():
                        dA_fragment[i_s1, i_s2] *= T.exp(G[bb, bs * block_S + i_s1, bh] - G[bb, bs * block_S + i_s2, bh])
                    with T.Else():
                        dA_fragment[i_s1, i_s2] = 0
            T.copy(dA_fragment, dA_shared)

            # acceptable dA diff
            # T.copy(dA_fragment, dA[bb, bs * block_S:(bs + 1) * block_S, bh, :])

            # Update dk using previous dk
            T.clear(A_fragment)
            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], K_shared)
                T.copy(dk[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dk_shared)
                T.copy(dk_shared, dk_fragment)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    K_shared_beta[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s]
                T.gemm(K_shared_beta, K_shared, A_fragment, transpose_B=True)
                T.gemm(dA_shared, K_shared, dk_fragment_beta, clear_accum=True)
                # for i_s, i_k2 in T.Parallel(block_S, block_DK):
                #     dbeta_fragment[i_s] = dbeta_fragment[i_s] + dk_fragment_beta[i_s, i_k2] * K_shared[i_s, i_k2]
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dbeta_fragment_reduce_tmpk[i_s, i_k2] = dk_fragment_beta[i_s, i_k2] * K_shared[i_s, i_k2]
                T.reduce_sum(dbeta_fragment_reduce_tmpk, dbeta_fragment_k, dim=1, clear=False)
                T.gemm(dA_shared, K_shared_beta, dk_fragment, transpose_A=True)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dk_shared_beta[i_s, i_k2] = dk_fragment_beta[i_s, i_k2] * Beta_shared[i_s]
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dk_fragment[i_s, i_k2] = dk_fragment[i_s, i_k2] + dk_shared_beta[i_s, i_k2]
                T.copy(dk_fragment, dk[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])

            # Update dg and dbeta
            T.copy(A_fragment, A_shared)
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dA_A_fragment[i_s1, i_s2] = dA_fragment[i_s1, i_s2] * A_fragment[i_s1, i_s2]
            # Note: Reduce operation now not supported in shared memory
            # FIXME: reduce will cause incorrect result when dim != -1
            T.reduce_sum(dA_A_fragment, dA_A_fragment_1, dim=1)
            T.reduce_sum(dA_A_fragment, dA_A_fragment_2, dim=0)

            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dg_A_positive[bb, bs * block_S + i_s1, bh, i_s2] = dA_A_fragment[i_s1, i_s2]
                dg_A_negative[bb, bs * block_S + i_s2, bh, i_s1] = dA_A_fragment[i_s1, i_s2]

            for i_s in T.Parallel(block_S):
                dbeta_k[bb, bs * block_S + i_s, bh] = dbeta_fragment_k[i_s]

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    block_DK=64,
    block_DV=64,
    threads=128,
    num_stages=0,
):
    K, V, Beta, G, A, dw, du = prepare_input(
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        getattr(torch, input_dtype),
        getattr(torch, output_dtype),
        getattr(torch, accum_dtype),
        getattr(torch, gate_dtype),
        getattr(torch, state_dtype),
    )
    dk_ref, dv_ref, dbeta_ref, dg_ref = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )
    dk_tilelang, dv_tilelang, dbeta_tilelang, dg_tilelang = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )
    BS = chunk_size
    dA_tilelang = torch.empty(B, S, H, BS, dtype=getattr(torch, input_dtype)).cuda()
    dbeta_tilelang_k = torch.empty(B, S, H, dtype=getattr(torch, output_dtype)).cuda()
    dg_tilelang_A_positive = torch.empty(B, S, H, BS, dtype=getattr(torch, gate_dtype)).cuda()
    dg_tilelang_A_negative = torch.empty(B, S, H, BS, dtype=getattr(torch, gate_dtype)).cuda()

    # ref
    dk_ref, dv_ref, dbeta_ref, dg_ref = bwd_prepare_wy_repr(K, V, G, Beta, A, dw, du, cu_seqlens=None)

    # tilelang
    kernel = tilelang_wy_fast_bwd(
        B,
        S,
        H,
        DK,
        DV,
        input_dtype,
        output_dtype,
        accum_dtype,
        gate_dtype,
        state_dtype,
        chunk_size,
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    dA_tilelang, dk_tilelang, dv_tilelang, dbeta_tilelang, dg_tilelang = kernel(K, V, Beta, G, A, dw, du)
    torch.cuda.synchronize()
    kernel_split = tilelang_wy_fast_bwd_split(
        B,
        S,
        H,
        DK,
        DV,
        input_dtype,
        output_dtype,
        accum_dtype,
        gate_dtype,
        state_dtype,
        chunk_size,
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    kernel_split(
        K, V, Beta, G, A, dw, du, dA_tilelang, dk_tilelang, dv_tilelang, dbeta_tilelang_k, dg_tilelang_A_positive, dg_tilelang_A_negative
    )
    torch.cuda.synchronize()

    dbeta_tilelang = dbeta_tilelang_k + dbeta_tilelang
    dg_tilelang = dg_tilelang + dg_tilelang_A_positive.sum(dim=-1) - dg_tilelang_A_negative.sum(dim=-1)

    from test_utils import assert_similar

    assert_similar(dk_ref, dk_tilelang, eps=1e-5, name="dk", raise_assert=False)
    assert_similar(dv_ref, dv_tilelang, eps=1e-5, name="dv", raise_assert=False)
    assert_similar(dbeta_ref, dbeta_tilelang, eps=1e-5, name="dbeta", raise_assert=False)
    assert_similar(dg_ref, dg_tilelang, eps=1e-5, name="dg", raise_assert=False)


def main():
    DK = 128
    DV = 128
    run_test(
        B=1,
        S=32768,
        H=8,
        DK=DK,
        DV=DV,
        input_dtype=T.bfloat16,
        output_dtype=T.bfloat16,
        accum_dtype=T.float32,
        gate_dtype=T.float32,
        state_dtype=T.float32,
        chunk_size=64,
        block_DK=32,
        block_DV=32,
        threads=128,
        num_stages=0,
    )


if __name__ == "__main__":
    main()
