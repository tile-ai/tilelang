import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune

from FLA_KDA.fla_chunk_inter import chunk_kda_bwd_dqkwg
from test_utils_kda import do_bench, compare_tensors

import torch

torch.random.manual_seed(42)


def prepare_input(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    gate_dtype,
):
    BS = S // chunk_size
    q = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    k = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    v_new = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    w = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()
    g = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()
    h = torch.randn(B, BS, H, DK, DV, dtype=input_dtype).cuda()
    dv = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    do = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    dh = torch.randn(B, BS, H, DK, DV, dtype=input_dtype).cuda()

    return q, k, v_new, w, g, h, dv, do, dh


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    gate_dtype,
):
    dq = torch.randn(B, S, H, DK, dtype=torch.float32).cuda()
    dk = torch.randn(B, S, H, DK, dtype=torch.float32).cuda()
    dw = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()
    dg = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()
    return dq, dk, dw, dg


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
@tilelang.jit(out_idx=[-4, -3, -2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def chunk_bwd_dqkwg(
    B,
    S,
    H,
    DK,
    DV,
    scale,
    chunk_size,
    input_dtype,
    gate_dtype,
    block_DK=32,
    block_DV=32,
    threads=32,
    num_stages=0,
):
    block_S = chunk_size
    BS = S // block_S
    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    H_shape = (B, BS, H, DK, DV)

    @T.prim_func
    def kernel(
        Q: T.Tensor(K_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        G: T.Tensor(K_shape, dtype=gate_dtype),
        h: T.Tensor(H_shape, dtype=input_dtype),
        dv: T.Tensor(V_shape, dtype=input_dtype),
        DO: T.Tensor(V_shape, dtype=input_dtype),
        Dh: T.Tensor(H_shape, dtype=input_dtype),
        dq: T.Tensor(K_shape, dtype=T.float32),
        dk: T.Tensor(K_shape, dtype=T.float32),
        dw: T.Tensor(K_shape, dtype=gate_dtype),
        dg: T.Tensor(K_shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(DK, block_DK), T.ceildiv(S, block_S), B * H, threads=threads) as (bk, bs, bbh):
            bb, bh = bbh // H, bbh % H
            chunk_last_idx = T.min(S, (bs + 1) * block_S) - 1

            dgkn_fragment = T.alloc_fragment((block_DK), dtype=T.float32)
            dgkn_fragment_tmp = T.alloc_fragment((block_DK,), dtype=T.float32)
            dq_fragment = T.alloc_fragment((block_S, block_DK), dtype=T.float32)
            dk_fragment = T.alloc_fragment((block_S, block_DK), dtype=T.float32)
            dw_fragment = T.alloc_fragment((block_S, block_DK), dtype=T.float32)
            dgk_shared = T.alloc_shared((block_S, block_DK), dtype=T.float32)

            h_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            dh_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            dgkn_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)  # d of last token in a chunk
            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            DO_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            DV_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            G_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)  # chunk G
            Gn_shared = T.alloc_shared((block_DK), dtype=input_dtype)  # chunk last token G
            Q_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)

            dkkn_shared = T.alloc_shared((block_S, block_DK), dtype=T.float32)
            pp_shared = T.alloc_shared((block_DK), dtype=T.float32)

            T.clear(dgkn_fragment)
            T.clear(dq_fragment)
            T.clear(dk_fragment)
            T.clear(dw_fragment)

            T.copy(G[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK], G_shared)
            T.copy(G[bb, chunk_last_idx, bh, bk * block_DK : (bk + 1) * block_DK], Gn_shared)

            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(h[bb, bs, bh, bk * block_DK : (bk + 1) * block_DK, i_v * block_DV : (i_v + 1) * block_DV], h_shared)
                T.copy(Dh[bb, bs, bh, bk * block_DK : (bk + 1) * block_DK, i_v * block_DV : (i_v + 1) * block_DV], dh_shared)
                T.copy(V[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], V_shared)
                T.copy(DO[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], DO_shared)
                T.copy(dv[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], DV_shared)
                # += reduce_sum
                for i_k1, i_v1 in T.Parallel(block_DK, block_DV):
                    dgkn_shared[i_k1, i_v1] = h_shared[i_k1, i_v1] * dh_shared[i_k1, i_v1]
                T.reduce_sum(dgkn_shared, dgkn_fragment_tmp, dim=1, clear=True)  # [block_DK]
                for i_ks in T.Parallel(block_DK):
                    dgkn_fragment[i_ks] += dgkn_fragment_tmp[i_ks]
                T.gemm(DO_shared, h_shared, dq_fragment, transpose_B=True, clear_accum=False)  # [block_S, block_DK]
                T.gemm(V_shared, dh_shared, dk_fragment, transpose_B=True, clear_accum=False)  # [block_S, block_DK]
                T.gemm(DV_shared, h_shared, dw_fragment, transpose_B=True, clear_accum=False)  # [block_S, block_DK]
            # chunk last token
            for i_k0 in T.Parallel(block_DK):
                dgkn_fragment[i_k0] = dgkn_fragment[i_k0] * T.exp2(Gn_shared[i_k0])

            for i_s, i_k in T.Parallel(block_S, block_DK):
                dw_fragment[i_s, i_k] = -dw_fragment[i_s, i_k]
                dq_fragment[i_s, i_k] = dq_fragment[i_s, i_k] * scale * T.exp2(G_shared[i_s, i_k])
                dk_fragment[i_s, i_k] = dk_fragment[i_s, i_k] * T.exp2(Gn_shared[i_k] - G_shared[i_s, i_k])

            T.copy(dw_fragment, dw[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK])
            T.copy(dq_fragment, dq[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK])
            T.copy(dk_fragment, dk[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK])

            T.copy(Q[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK], Q_shared)
            T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK], K_shared)

            for i_s2, i_k2 in T.Parallel(block_S, block_DK):
                dkkn_shared[i_s2, i_k2] = dk_fragment[i_s2, i_k2] * K_shared[i_s2, i_k2]
            T.reduce_sum(dkkn_shared, pp_shared, dim=0, clear=True)
            for i_k3 in T.Parallel(block_DK):
                pp_shared[i_k3] += dgkn_fragment[i_k3]

            for i_s4, i_k4 in T.Parallel(block_S, block_DK):
                dgk_shared[i_s4, i_k4] = (
                    Q_shared[i_s4, i_k4] * dq_fragment[i_s4, i_k4]
                    - K_shared[i_s4, i_k4] * dk_fragment[i_s4, i_k4]
                    + T.if_then_else(chunk_last_idx == bs * block_S + i_s4, pp_shared[i_k4], 0.0)
                )

            T.copy(dgk_shared, dg[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    DV,
    scale,
    input_dtype,
    gate_dtype,
    qk_dtype,
    chunk_size,
    use_gk=True,
    use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
    block_DK=64,
    block_DV=32,
    threads=128,
    num_stages=0,
):
    q, k, v_new, w, g, h, dv, do, dh = prepare_input(B, S, H, DK, DV, chunk_size, getattr(torch, input_dtype), getattr(torch, gate_dtype))

    dq_ref, dk_ref, dw_ref, dg_ref = chunk_kda_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
    )

    dq, dk, dw, dg = prepare_output(B, S, H, DK, DV, chunk_size, getattr(torch, gate_dtype))
    kernel = chunk_bwd_dqkwg(
        B=B, S=S, H=H, DK=DK, DV=DV, scale=scale, chunk_size=chunk_size, input_dtype=input_dtype, gate_dtype=gate_dtype
    )
    dq, dk, dw, dg = kernel(q, k, v_new, g, h, dv, do, dh)

    compare_tensors("dq", dq_ref, dq)
    compare_tensors("dk", dk_ref, dk)
    compare_tensors("dw", dw_ref, dw)
    compare_tensors("dg", dg_ref, dg)

    fla_time = do_bench(
        chunk_kda_bwd_dqkwg,
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
    )
    tilelang_time = do_bench(kernel, q, k, v_new, g, h, dv, do, dh)
    print("fla_time:", fla_time)
    print("tilelang_time:", tilelang_time)


def main():
    run_test(
        B=1,
        S=8192,
        H=64,
        DK=128,
        DV=128,
        scale=1.0,
        input_dtype="float32",
        gate_dtype="float32",  # gate must be float32
        qk_dtype="float32",
        chunk_size=64,
        use_gk=True,
        use_initial_state=True,
        store_final_state=True,
        save_new_value=True,
        block_DK=32,
        block_DV=32,
        threads=128,
        num_stages=2,
    )


if __name__ == "__main__":
    main()
