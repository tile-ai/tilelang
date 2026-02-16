import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune

from FLA_KDA.fla_chunk_o import chunk_gla_bwd_dA
from test_utils_kda import compare_tensors, do_bench

import torch

torch.random.manual_seed(1)


def prepare_input(
    B,
    S,
    H,
    DV,
    chunk_size,
    input_dtype,
    do_dtype,
):
    DO = torch.randn(B, S, H, DV, dtype=do_dtype).cuda()
    V_new = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    return DO, V_new


def prepare_output(
    B,
    S,
    H,
    DV,
    chunk_size,
    d_type,
):
    dA = torch.empty(B, S, H, chunk_size, dtype=d_type).cuda()
    return dA


def get_configs():
    import itertools

    block_DV = [32, 64, 128]
    threads = [32, 64, 128, 256]
    num_stages = [0, 1, 2, 3, 4]
    _configs = list(itertools.product(block_DV, threads, num_stages))
    configs = [{"block_DV": c[0], "threads": c[1], "num_stages": c[2]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=10, rep=5)
@tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_chunk_bwd_kernel_dv_local(
    B,
    S,
    H,
    DV,
    scale,
    input_dtype,
    da_dtype,
    do_dtype,
    chunk_size,
    block_DV=128,
    threads=128,
    num_stages=1,
):
    block_S = BS = chunk_size
    DO_shape = (B, S, H, DV)
    V_shape = (B, S, H, DV)
    dA_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        DO: T.Tensor(DO_shape, dtype=do_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        dA: T.Tensor(dA_shape, dtype=da_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            do_shared = T.alloc_shared((block_S, block_DV), dtype=do_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=do_dtype)
            dA_fragment = T.alloc_fragment((block_S, block_S), dtype=T.float32)

            T.clear(dA_fragment)
            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(DO[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], do_shared)
                T.copy(V[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], V_shared)
                T.gemm(do_shared, V_shared, dA_fragment, transpose_B=True)
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dA_fragment[i_s1, i_s2] = T.if_then_else(i_s1 >= i_s2, dA_fragment[i_s1, i_s2] * scale, 0.0)  # 下三角矩阵
            T.copy(dA_fragment, dA[bb, bs * block_S : (bs + 1) * block_S, bh, 0:block_S])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    DV,
    scale,
    input_dtype,
    do_dtype,
    da_dtype,
    chunk_size,
):
    DO, V_new = prepare_input(B, S, H, DV, chunk_size, getattr(torch, input_dtype), getattr(torch, do_dtype))
    print(DO.dtype, V_new.dtype)
    dA_ref = chunk_gla_bwd_dA(v=V_new, do=DO, scale=scale)

    dA_tilelang = prepare_output(B, S, H, DV, chunk_size, getattr(torch, da_dtype))
    kernel = tilelang_chunk_bwd_kernel_dv_local(
        B=B,
        S=S,
        H=H,
        DV=DV,
        scale=scale,
        input_dtype=input_dtype,
        da_dtype=da_dtype,
        do_dtype=do_dtype,
        chunk_size=chunk_size,
    )
    dA_tilelang = kernel(DO, V_new)
    compare_tensors("dA", dA_ref, dA_tilelang)
    fla_time = do_bench(chunk_gla_bwd_dA, v=V_new, do=DO, scale=scale)
    tilelang_time = do_bench(kernel, DO, V_new)
    print("fla_time:", fla_time)
    print("tilelang_time:", tilelang_time)


def main():
    run_test(
        B=1,
        S=1024 * 8,  # 32768
        H=64,
        DK=128,
        DV=128,
        scale=1.0,
        input_dtype="bfloat16",
        do_dtype="bfloat16",
        da_dtype="float32",
        chunk_size=64,
    )


if __name__ == "__main__":
    main()
