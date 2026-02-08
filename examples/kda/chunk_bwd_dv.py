import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
import sys  # noqa: F401

from FLA_KDA.fla_chunk_o import chunk_bwd_dv_local
from test_utils_kda import compare_tensors, do_bench

import torch

torch.random.manual_seed(1)


def prepare_input(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    do_dtype,
):
    q = torch.randn(B, S, H, DK, dtype=do_dtype).cuda()
    k = torch.randn(B, S, H, DK, dtype=do_dtype).cuda()
    DO = torch.randn(B, S, H, DV, dtype=do_dtype).cuda()
    A = torch.randn(B, S, H, chunk_size, dtype=input_dtype).cuda()
    return q, k, DO, A


def prepare_output(
    B,
    S,
    H,
    DV,
    chunk_size,
    output_dtype,
):
    dv = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return dv


def get_configs():
    import itertools

    block_DV = [32, 64, 128]
    threads = [32, 64, 128]
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
    input_dtype,
    output_dtype,
    do_dtype,
    chunk_size,
    block_DV=128,
    threads=128,
    num_stages=1,
):
    block_S = BS = chunk_size
    DO_shape = (B, S, H, DV)
    A_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        DO: T.Tensor(DO_shape, dtype=do_dtype),
        A: T.Tensor(A_shape, dtype=input_dtype),
        dv: T.Tensor(DO_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H

            A_shared = T.alloc_shared((BS, BS), dtype=do_dtype)
            DO_shared = T.alloc_shared((BS, block_DV), dtype=do_dtype)
            dv_fragment = T.alloc_fragment((BS, block_DV), dtype=T.float32)
            dv_shared = T.alloc_shared((BS, block_DV), dtype=output_dtype)

            T.copy(A[bb, bs * BS : (bs + 1) * BS, bh, :], A_shared)
            for i_s1, i_s2 in T.Parallel(BS, BS):
                A_shared[i_s1, i_s2] = T.if_then_else(i_s1 >= i_s2, A_shared[i_s1, i_s2], 0.0)
            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(DO[bb, bs * BS : (bs + 1) * BS, bh, i_v * block_DV : (i_v + 1) * block_DV], DO_shared)
                T.gemm(A_shared, DO_shared, dv_fragment, transpose_A=True, clear_accum=True)  # transpose_A: A^T
                T.copy(dv_fragment, dv_shared)
                T.copy(dv_shared, dv[bb, bs * BS : (bs + 1) * BS, bh, i_v * block_DV : (i_v + 1) * block_DV])

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
    output_dtype,
    chunk_size,
):
    q, k, DO, A = prepare_input(B, S, H, DK, DV, chunk_size, getattr(torch, input_dtype), getattr(torch, do_dtype))
    dv_ref = chunk_bwd_dv_local(q, k, do=DO, A=A)

    dv_tilelang = prepare_output(B, S, H, DV, chunk_size, getattr(torch, output_dtype))
    kernel = tilelang_chunk_bwd_kernel_dv_local(
        B=B,
        S=S,
        H=H,
        DV=DV,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        do_dtype=do_dtype,
        chunk_size=chunk_size,
    )
    dv_tilelang = kernel(DO, A)
    compare_tensors("dv", dv_ref, dv_tilelang)

    fla_time = do_bench(chunk_bwd_dv_local, q, k, do=DO, A=A)
    tilelang_time = do_bench(kernel, DO, A)
    print("fla_time: ", fla_time)
    print("tilelang_time: ", tilelang_time)


def main():
    run_test(
        B=1,
        S=1024 * 8,  # 32768
        H=64,
        DK=128,
        DV=128,
        scale=1.0,
        input_dtype="bfloat16",
        do_dtype="float32",
        output_dtype="bfloat16",
        chunk_size=64,
    )


if __name__ == "__main__":
    main()
