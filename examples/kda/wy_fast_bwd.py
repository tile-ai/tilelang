# Reference: fla/ops/gated_delta_rule/wy_fast.py

import sys  # noqa: F401

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from FLA_KDA.fla_wy_fast import prepare_wy_repr_bwd
from test_utils_kda import do_bench, compare_tensors

import torch

torch.random.manual_seed(0)
torch.set_printoptions(profile="full")


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
    V = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    Beta = torch.randn(B, S, H, dtype=input_dtype).cuda()
    GK = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()
    A = torch.randn(B, S, H, BS, dtype=input_dtype).cuda()
    dw = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    dv = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    dk = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    dg = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()

    return K, V, Beta, GK, A, dw, dv, dk, dg


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
    dg = torch.empty(B, S, H, DK, dtype=gate_dtype).cuda()
    dA = torch.empty(B, S, H, DK, dtype=output_dtype).cuda()
    return dk, dv, dbeta, dg, dA


def get_configs():
    import itertools

    block_DK = [32, 64, 128]
    block_DV = [32, 64, 128]
    threads = [32, 64, 128, 256]
    num_stages = [0, 1, 2, 3]
    _configs = list(itertools.product(block_DK, block_DV, threads, num_stages))

    configs = [{"block_DK": c[0], "block_DV": c[1], "threads": c[2], "num_stages": c[3]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=3, rep=5)
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
    G_shape = (B, S, H, DK)
    A_shape = (B, S, H, BS)
    dw_shape = (B, S, H, DK)
    du_shape = (B, S, H, DV)

    dk_shape = (B, S, H, DK)
    dv_shape = (B, S, H, DV)
    dbeta_shape = (B, S, H)
    dg_shape = (B, S, H, DK)
    dA_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        # input
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        GK: T.Tensor(G_shape, dtype=gate_dtype),
        A: T.Tensor(A_shape, dtype=input_dtype),
        dw: T.Tensor(dw_shape, dtype=input_dtype),
        du: T.Tensor(du_shape, dtype=input_dtype),
        dk: T.Tensor(dk_shape, dtype=input_dtype),
        dg: T.Tensor(dg_shape, dtype=gate_dtype),
        # output
        dA: T.Tensor(dA_shape, dtype=input_dtype),
        dk2: T.Tensor(dk_shape, dtype=output_dtype),
        dv: T.Tensor(dv_shape, dtype=output_dtype),
        dbeta: T.Tensor(dbeta_shape, dtype=output_dtype),
        dg2: T.Tensor(dg_shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H

            A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared_beta_g = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            V_shared_beta = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype)
            GK_shared = T.alloc_shared((block_S, block_DK), dtype=gate_dtype)
            GK_shared_exp = T.alloc_shared((block_S, block_DK), dtype=gate_dtype)
            dw_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            du_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)

            dk_old_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            dg_old_shared = T.alloc_shared((block_S, block_DK), dtype=gate_dtype)
            dA_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)

            dA_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dA_fragment_tmp1 = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dA_fragment_tmp2 = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            dk_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dk_fragment_beta_g = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dv_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dv_fragment_beta = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dbeta_fragment = T.alloc_fragment((block_S,), dtype=accum_dtype)
            dbeta_fragment_reduce_tmpk = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dbeta_fragment_reduce_tmpv = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dg_fragment = T.alloc_fragment((block_S, block_DK), dtype=gate_dtype)

            T.clear(dA_fragment)
            T.clear(dk_fragment)
            T.clear(dk_fragment_beta_g)
            T.clear(dv_fragment)
            T.clear(dv_fragment_beta)
            T.clear(dbeta_fragment)
            T.clear(dg_fragment)

            T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, :], A_shared)  # load A
            T.copy(Beta[bb, bs * block_S : (bs + 1) * block_S, bh], Beta_shared)

            # Update dk
            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], K_shared)
                T.copy(dk[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dk_old_shared)
                T.copy(dg[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dg_old_shared)
                T.copy(GK[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], GK_shared)

                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    GK_shared_exp[i_s, i_k2] = T.exp2(GK_shared[i_s, i_k2])
                    K_shared_beta_g[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s] * GK_shared_exp[i_s, i_k2]

                T.copy(dw[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dw_shared)
                T.gemm(dw_shared, K_shared_beta_g, dA_fragment, transpose_B=True, clear_accum=False)
                T.gemm(A_shared, dw_shared, dk_fragment_beta_g, transpose_A=True, clear_accum=True)

                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dk_fragment[i_s, i_k2] = (
                        dk_fragment_beta_g[i_s, i_k2] * GK_shared_exp[i_s, i_k2] * Beta_shared[i_s] + dk_old_shared[i_s, i_k2]
                    )

                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dbeta_fragment_reduce_tmpk[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * K_shared[i_s, i_k2] * GK_shared_exp[i_s, i_k2]
                T.reduce_sum(dbeta_fragment_reduce_tmpk, dbeta_fragment, dim=1, clear=False)

                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dg_fragment[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * K_shared_beta_g[i_s, i_k2] + dg_old_shared[i_s, i_k2]

                # correct dk, dg
                T.copy(dk_fragment, dk2[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])
                T.copy(dg_fragment, dg2[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])

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

                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    dbeta_fragment_reduce_tmpv[i_s, i_v2] = dv_fragment_beta[i_s, i_v2] * V_shared[i_s, i_v2]
                T.reduce_sum(dbeta_fragment_reduce_tmpv, dbeta_fragment, dim=1, clear=False)

                T.copy(dv_fragment, dv[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV])

            T.copy(dbeta_fragment, dbeta[bb, bs * block_S : (bs + 1) * block_S, bh])

            # correct dA
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dA_shared[i_s1, i_s2] = T.if_then_else(i_s1 > i_s2, dA_fragment[i_s1, i_s2], 0.0)
            T.gemm(dA_shared, A_shared, dA_fragment_tmp1, transpose_B=True, clear_accum=True)
            T.copy(dA_fragment_tmp1, dA_shared)
            T.gemm(A_shared, dA_shared, dA_fragment_tmp2, transpose_A=True, clear_accum=True)
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dA_fragment_tmp2[i_s1, i_s2] = T.if_then_else(i_s1 > i_s2, -dA_fragment_tmp2[i_s1, i_s2], 0.0)
            T.copy(dA_fragment_tmp2, dA[bb, bs * block_S : (bs + 1) * block_S, bh, :])

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
    K, V, Beta, GK, A, dw, dv, dk, dg = prepare_input(
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

    dk_tilelang, dv_tilelang, dbeta_tilelang, dg_tilelang, dA_tilelang = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )

    # ref
    dk_ref, dv_ref, dbeta_ref, dg_ref, dA_ref = prepare_wy_repr_bwd(
        k=K,
        v=V,
        gk=GK,
        beta=Beta,
        A=A,
        dw=dw,
        du=dv,
        dk=dk,
        dg=dg,
    )

    # # tilelang
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
    )
    dA_tilelang, dk_tilelang, dv_tilelang, dbeta_tilelang, dg_tilelang = kernel(K, V, Beta, GK, A, dw, dv, dk, dg)

    compare_tensors("dA", dA_tilelang, dA_ref)
    compare_tensors("dk", dk_tilelang, dk_ref)
    compare_tensors("dv", dv_tilelang, dv_ref)
    compare_tensors("dbeta", dbeta_tilelang, dbeta_ref)
    compare_tensors("dg", dg_tilelang, dg_ref)
    fla_time = do_bench(
        prepare_wy_repr_bwd,
        k=K,
        v=V,
        gk=GK,
        beta=Beta,
        A=A,
        dw=dw,
        du=dv,
        dk=dk,
        dg=dg,
    )
    tilelang_time = do_bench(kernel, K, V, Beta, GK, A, dw, dv, dk, dg)
    print(f"FLA_time: {fla_time}")
    print(f"TileLang_time: {tilelang_time}")


def main():
    DK = 128
    DV = 128
    run_test(
        B=1,
        S=32768,
        H=8,
        DK=DK,
        DV=DV,
        input_dtype=T.float32,
        output_dtype=T.float32,
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
