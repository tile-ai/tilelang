import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
import torch
import torch.nn.functional as F
from FLA_KDA.fla_chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from FLA_KDA.cumsum import chunk_local_cumsum
from test_utils_kda import do_bench

torch.random.manual_seed(42)


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
):
    q = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    k = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    beta = torch.randn(B, S, H, dtype=input_dtype).cuda()
    gk = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()
    gk = F.logsigmoid(gk)
    gk = chunk_local_cumsum(gk, chunk_size)
    return q, k, gk, beta


def prepare_output(
    B,
    S,
    H,
    chunk_size,
    sub_chunk_size,
    output_dtype,
):
    Aqk = torch.empty(B, S, H, chunk_size, dtype=output_dtype).cuda()
    Akk = torch.empty(B, S, H, sub_chunk_size, dtype=output_dtype).cuda()
    return Aqk, Akk


def get_configs():
    import itertools

    block_H = [1, 2, 4, 8]
    threads = [128, 256]
    num_stages = [0, 1, 2, 3]
    _configs = list(itertools.product(block_H, threads, num_stages))

    configs = [{"block_H": c[0], "threads": c[1], "num_stages": c[2]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=3, rep=5)
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_chunk_kda_fwd_intra_token_parallel(
    B,
    S,
    H,
    DK,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    sub_chunk_size,
    block_H=1,
    threads=32,
    num_stages=1,
):
    CS = chunk_size
    SCS = sub_chunk_size
    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    GK_shape = (B, S, H, DK)
    Beta_shape = (B, S, H)
    Aqk_shape = (B, S, H, CS)
    Akk_shape = (B, S, H, SCS)

    @T.prim_func
    def kernel(
        Q: T.Tensor(Q_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        GK: T.Tensor(GK_shape, dtype=gate_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        Aqk: T.Tensor(Aqk_shape, dtype=output_dtype),
        Akk: T.Tensor(Akk_shape, dtype=output_dtype),
    ):
        with T.Kernel(B * S, T.ceildiv(H, block_H), threads=threads) as (bbs, bh):  # block_index_bs, block_index_dh
            bb, bs = bbs // S, bbs % S
            i_c = bs // CS  # indice chunk
            i_s = (bs % CS) // SCS  # indice subchunk
            i_tc = i_c * CS
            i_ts = i_tc + i_s * SCS
            loops = bs + 1 - i_ts

            Q_i_shared = T.alloc_shared((block_H, DK), dtype=input_dtype)
            K_i_shared = T.alloc_shared((block_H, DK), dtype=input_dtype)
            GK_i_shared = T.alloc_shared((block_H, DK), dtype=gate_dtype)
            Beta_shared = T.alloc_shared(
                (block_H,),
                dtype=input_dtype,
            )
            K_j_shared = T.alloc_shared((block_H, DK), dtype=input_dtype)
            GK_j_shared = T.alloc_shared((block_H, DK), dtype=gate_dtype)
            Aqk_shared = T.alloc_shared((block_H, DK), dtype=accum_dtype)
            Akk_shared = T.alloc_shared((block_H, DK), dtype=accum_dtype)
            Sum_Aqk_shared = T.alloc_shared((block_H, CS), dtype=output_dtype)
            Sum_Akk_shared = T.alloc_shared((block_H, SCS), dtype=output_dtype)

            Q_i_fragment = T.alloc_fragment(
                (block_H, DK),
                dtype=input_dtype,
            )
            K_i_fragment = T.alloc_fragment(
                (block_H, DK),
                dtype=input_dtype,
            )
            K_j_fragment = T.alloc_fragment(
                (block_H, DK),
                dtype=accum_dtype,
            )

            Sum_Aqk_fragment = T.alloc_fragment(
                (block_H,),
                dtype=accum_dtype,
            )
            Sum_Akk_fragment = T.alloc_fragment(
                (block_H,),
                dtype=accum_dtype,
            )

            T.copy(Q[bb, bs, bh * block_H : (bh + 1) * block_H, :], Q_i_shared)
            T.copy(K[bb, bs, bh * block_H : (bh + 1) * block_H, :], K_i_shared)
            T.copy(GK[bb, bs, bh * block_H : (bh + 1) * block_H, :], GK_i_shared)  # TMA

            T.disable_warp_group_reg_alloc()
            for i_h in T.Parallel(block_H):  # cannot use TMA
                Beta_shared[i_h] = Beta[bb, bs, bh * block_H + i_h]

            for i_h, i_k in T.Parallel(block_H, DK):
                K_i_fragment[i_h, i_k] = K_i_shared[i_h, i_k] * Beta_shared[i_h]
                Q_i_fragment[i_h, i_k] = Q_i_shared[i_h, i_k]

            T.clear(Sum_Akk_shared)
            T.clear(Sum_Aqk_shared)

            for d in T.Pipelined(loops, num_stages=num_stages):
                j = d + i_ts
                T.copy(K[bb, j, bh * block_H : (bh + 1) * block_H, :], K_j_shared)
                T.copy(GK[bb, j, bh * block_H : (bh + 1) * block_H, :], GK_j_shared)
                # T.copy(K_j_shared, K_j_fragment)
                for i_h, i_k in T.Parallel(block_H, DK):
                    K_j_fragment[i_h, i_k] = K_j_shared[i_h, i_k] * T.exp2(GK_i_shared[i_h, i_k] - GK_j_shared[i_h, i_k])
                    Aqk_shared[i_h, i_k] = Q_i_fragment[i_h, i_k] * K_j_fragment[i_h, i_k]
                    Akk_shared[i_h, i_k] = K_i_fragment[i_h, i_k] * K_j_fragment[i_h, i_k]

                T.reduce_sum(Aqk_shared, Sum_Aqk_fragment, dim=-1, clear=True)
                T.reduce_sum(Akk_shared, Sum_Akk_fragment, dim=-1, clear=True)

                T.copy(Sum_Aqk_fragment, Sum_Aqk_shared[:, j % CS])

                if j < bs:
                    T.copy(Sum_Akk_fragment, Sum_Akk_shared[:, d])

            T.copy(Sum_Aqk_shared, Aqk[bb, bs, bh * block_H : (bh + 1) * block_H, :])
            T.copy(Sum_Akk_shared, Akk[bb, bs, bh * block_H : (bh + 1) * block_H, :])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    scale,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    sub_chunk_size,
):
    q, k, gk, beta = prepare_input(
        B,
        S,
        H,
        DK,
        chunk_size,
        getattr(torch, input_dtype),
        getattr(torch, output_dtype),
        getattr(torch, accum_dtype),
        getattr(torch, gate_dtype),
    )
    Aqk_ref, Akk_ref = prepare_output(B, S, H, chunk_size, sub_chunk_size, getattr(torch, output_dtype))
    Aqk_tilelang, Akk_tilelang = prepare_output(B, S, H, chunk_size, sub_chunk_size, getattr(torch, output_dtype))

    Aqk_ref, Akk_ref = chunk_kda_fwd_intra_token_parallel(
        q=q, k=k, gk=gk, beta=beta, Aqk=Aqk_ref, Akk=Akk_ref, scale=scale, chunk_size=chunk_size, sub_chunk_size=sub_chunk_size
    )

    kernel = tilelang_chunk_kda_fwd_intra_token_parallel(
        B,
        S,
        H,
        DK,
        input_dtype,
        output_dtype,
        accum_dtype,
        gate_dtype,
        chunk_size,
        sub_chunk_size,
    )
    # kernel_source  = kernel.get_kernel_source()
    # print(kernel_source)
    # exit()
    # # scale 如何传值
    # r = torch.cuda.nvtx.range_start("TILELANG_KDA")
    Aqk_tilelang, Akk_tilelang = kernel(
        q,
        k,
        gk,
        beta,
    )
    # torch.cuda.nvtx.range_end(r)

    fla_time = do_bench(
        chunk_kda_fwd_intra_token_parallel,
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Aqk=Aqk_ref,
        Akk=Akk_ref,
        scale=scale,
        chunk_size=chunk_size,
        sub_chunk_size=sub_chunk_size,
    )
    tilelang_time = do_bench(
        kernel,
        q,
        k,
        gk,
        beta,
    )

    print(f"fla time: {fla_time} ms")
    print(f"tilelang time: {tilelang_time} ms")


def main():
    run_test(
        B=1,
        S=1024 * 8,  # 32768
        H=64,
        DK=128,
        scale=1.0,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        chunk_size=64,
        sub_chunk_size=16,
    )


if __name__ == "__main__":
    main()
