# Reference: fla/ops/common/chunk_delta_h.py
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune

from FLA_KDA.fla_chunk_delta import chunk_gated_delta_rule_bwd_dhu
from FLA_KDA.cumsum import chunk_local_cumsum
from test_utils_kda import do_bench, compare_tensors

import torch
import torch.nn.functional as F

torch.random.manual_seed(42)


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
    Q = torch.randn(B, S, H, DK, dtype=input_dtype).cuda() * 0.01
    K = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    K = F.normalize(K, dim=-1, p=2)
    W = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    # Note: G should be in logspace and do chunkwise cumsum
    G = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()
    G = F.logsigmoid(G)
    G = chunk_local_cumsum(G, chunk_size)

    h0 = torch.randn(B, H, DK, DV, dtype=input_dtype).cuda()
    dht = torch.randn(B, H, DK, DV, dtype=input_dtype).cuda()
    dO = torch.randn(B, S, H, DV, dtype=input_dtype).cuda() * 0.01

    dv = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    return Q, K, W, G, h0, dht, dO, dv


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
    BS = S // chunk_size
    dh = torch.empty(B, BS, H, DK, DV, dtype=output_dtype).cuda()
    dh0 = torch.empty(B, H, DK, DV, dtype=state_dtype).cuda()
    dv2 = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return dh, dh0, dv2


def get_configs():
    import itertools

    block_DV = [32, 64, 128]
    threads = [32, 64, 128, 256]
    num_stages = [0, 1, 2, 3, 4]
    _configs = list(itertools.product(block_DV, threads, num_stages))

    configs = [{"block_DV": c[0], "threads": c[1], "num_stages": c[2]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit()
def tilelang_chunk_gated_delta_rule_bwd_dhu(
    Q,
    K,
    W,
    GK,
    h0,
    dht,
    dO,
    dv,
    # task config
    input_dtype: str = "bfloat16",
    output_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    gate_dtype: str = "float32",
    state_dtype: str = "float32",
    chunk_size: int = 64,
    scale: float = 1.0,
    use_gk: bool = True,
    use_initial_state: bool = True,
    use_final_state_gradient: bool = True,
    # kernel config
    block_DV: int = 64,
    threads: int = 256,
    num_stages: int = 0,
):
    block_S = chunk_size
    B, S, H, DK = T.const("B S H DK")
    DV = T.const("DV")
    # Should support cu_seqlen
    BS = S // block_S

    Q: T.Tensor[[B, S, H, DK], input_dtype]
    K: T.Tensor[[B, S, H, DK], input_dtype]
    W: T.Tensor[[B, S, H, DK], input_dtype]
    GK: T.Tensor[[B, S, H, DK], gate_dtype]
    h0: T.Tensor[[B, H, DK, DV], input_dtype]
    dht: T.Tensor[[B, H, DK, DV], input_dtype]
    dO: T.Tensor[[B, S, H, DV], input_dtype]
    dv: T.Tensor[[B, S, H, DV], input_dtype]
    dh = T.empty([B, BS, H, DK, DV], output_dtype)
    dh0 = T.empty([B, H, DK, DV], state_dtype)
    dv2 = T.empty([B, S, H, DV], output_dtype)

    with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
        bb, bh = bbh // H, bbh % H

        b_dh_shared = T.alloc_shared((DK, block_DV), dtype=output_dtype)
        b_dh_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
        b_dh_fragment_1 = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
        b_dh_fragment_2 = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
        dv_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
        dv_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
        dv_fragment_2 = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
        dO_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
        K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)

        Q_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
        W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)

        GK_last_shared = T.alloc_shared((DK,), dtype=gate_dtype)

        if use_final_state_gradient:
            T.copy(dht[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], b_dh_shared)
            T.copy(b_dh_shared, b_dh_fragment)
        else:
            T.clear(b_dh_fragment)

        for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
            # The gradient should be stored in the reverse order
            i_s_inv = T.ceildiv(S, block_S) - i_s - 1  # reverse indices
            # Store the updated dh
            T.copy(b_dh_fragment, b_dh_shared)
            T.copy(b_dh_shared, dh[bb, i_s_inv, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

            # Update dv
            T.copy(K[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, 0:DK], K_shared)
            T.gemm(K_shared, b_dh_shared, dv_fragment, clear_accum=True)
            T.copy(dv[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV], dv_shared)  # copy old dv
            T.copy(dv_shared, dv_fragment_2)
            for i_s2, i_v in T.Parallel(block_S, block_DV):
                dv_fragment[i_s2, i_v] = dv_fragment[i_s2, i_v] + dv_fragment_2[i_s2, i_v]
            # Store the updated dv
            T.copy(dv_fragment, dv_shared)
            T.copy(dv_shared, dv2[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV])

            # Update dh
            T.copy(Q[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, 0:DK], Q_shared)  # [block_S, DK]
            T.copy(W[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, 0:DK], W_shared)  # [block_S, DK]
            T.copy(
                dO[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV], dO_shared
            )  # [block_S, block_DV]

            if use_gk:
                last_idx = T.min((i_s_inv + 1) * block_S, S) - 1  # chunk last token gk
                T.copy(GK[bb, last_idx, bh, :], GK_last_shared)
                for i_k, i_v in T.Parallel(DK, block_DV):
                    b_dh_fragment[i_k, i_v] *= T.exp2(GK_last_shared[i_k])

            T.gemm(Q_shared, dO_shared, b_dh_fragment_1, transpose_A=True, clear_accum=True)  # [DK, block_DV]

            # dv_shared: [block_S, block_DV]
            T.gemm(W_shared, dv_shared, b_dh_fragment_2, transpose_A=True, clear_accum=True)  # [DK, block_DV]
            for i_k, i_v in T.Parallel(DK, block_DV):
                b_dh_fragment[i_k, i_v] += b_dh_fragment_1[i_k, i_v] * scale - b_dh_fragment_2[i_k, i_v]

        if use_initial_state:
            T.copy(b_dh_fragment, dh0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return dh, dh0, dv2


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
    scale,
    use_gk=True,
    use_initial_state=True,
    use_final_state_gradient=True,
    block_DV=64,
    threads=256,
    num_stages=0,
    use_torch=False,
):
    Q, K, W, G, h0, dht, dO, dv = prepare_input(
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

    dh_tilelang, dh0_tilelang, dv2_tilelang = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )

    # fla ref
    print("fla running...", flush=True)
    if use_gk:
        dh_ref, dh0_ref, dv2_ref = chunk_gated_delta_rule_bwd_dhu(
            q=Q, k=K, w=W, do=dO, dv=dv, gk=G, h0=h0, dht=dht, scale=scale, use_exp2=True
        )

    # tilelang
    print("tilelang running...", flush=True)
    dh_tilelang, dh0_tilelang, dv2_tilelang = tilelang_chunk_gated_delta_rule_bwd_dhu(
        Q,
        K,
        W,
        G,
        h0,
        dht,
        dO,
        dv,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        accum_dtype=accum_dtype,
        gate_dtype=gate_dtype,
        state_dtype=state_dtype,
        chunk_size=chunk_size,
        scale=scale,
        use_gk=use_gk,
        use_initial_state=use_initial_state,
        use_final_state_gradient=use_final_state_gradient,
    )

    fla_time = do_bench(
        chunk_gated_delta_rule_bwd_dhu, q=Q, k=K, w=W, do=dO, dv=dv, gk=G, h0=h0, dht=dht, scale=scale, chunk_size=chunk_size
    )
    tilelang_time = do_bench(
        tilelang_chunk_gated_delta_rule_bwd_dhu,
        Q,
        K,
        W,
        G,
        h0,
        dht,
        dO,
        dv,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        accum_dtype=accum_dtype,
        gate_dtype=gate_dtype,
        state_dtype=state_dtype,
        chunk_size=chunk_size,
        scale=scale,
        use_gk=use_gk,
        use_initial_state=use_initial_state,
        use_final_state_gradient=use_final_state_gradient,
    )

    print(f"fla time: {fla_time} ms")
    print(f"tilelang time: {tilelang_time} ms")

    compare_tensors("dh", dh_ref, dh_tilelang)
    compare_tensors("dh0", dh0_ref, dh0_tilelang)
    compare_tensors("dv2", dv2_ref, dv2_tilelang)


def main():
    DK = 128
    run_test(
        B=1,
        S=1024 * 8,
        H=64,
        DK=DK,
        DV=128,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        state_dtype="float32",
        chunk_size=64,
        scale=DK**-0.5,
        use_gk=True,
        use_initial_state=True,
        use_final_state_gradient=True,
        block_DV=32,
        threads=128,
        num_stages=1,
        use_torch=False,
    )


if __name__ == "__main__":
    main()
