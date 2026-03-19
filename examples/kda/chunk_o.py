import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune

from FLA_KDA.fla_chunk_o import chunk_gla_fwd_o_gk
from test_utils_kda import compare_tensors

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
    output_dtype,
    accum_dtype,
    gate_dtype,
):
    BS = chunk_size
    Q = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    A = torch.randn(B, S, H, BS, dtype=input_dtype).cuda()
    V = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    HIDDEN = torch.randn(B, S // BS, H, DK, DV, dtype=input_dtype).cuda()
    G = torch.randn(B, S, H, DK, dtype=gate_dtype).cuda()
    return Q, V, G, A, HIDDEN


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    output_dtype,
):
    O = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return O


def get_configs():
    import itertools

    block_DK = [32, 64, 128]
    block_DV = [32, 64, 128]
    threads = [128, 256]
    num_stages = [0, 1, 2, 3, 4]
    _configs = list(itertools.product(block_DK, block_DV, threads, num_stages))

    configs = [{"block_DK": c[0], "block_DV": c[1], "threads": c[2], "num_stages": c[3]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=3, rep=5)
@tilelang.jit()
def tilelang_chunk_fwd_o(
    Q,
    V,
    GK,
    A,
    HIDDEN,
    # task config
    input_dtype: str = "float16",
    output_dtype: str = "float16",
    accum_dtype: str = "float32",
    gate_dtype: str = "float32",
    chunk_size: int = 64,
    scale: float = 1.0,
    # kernel config
    block_S: int = 64,
    block_DK: int = 64,
    block_DV: int = 64,
    threads: int = 256,
    num_stages: int = 0,
):
    assert chunk_size == block_S, "chunk_size must be equal to block_S"
    BS = chunk_size
    B, S, H, DK = T.const("B S H DK")
    DV = T.const("DV")
    Q: T.Tensor[[B, S, H, DK], input_dtype]
    V: T.Tensor[[B, S, H, DV], input_dtype]
    GK: T.Tensor[[B, S, H, DK], gate_dtype]
    A: T.Tensor[[B, S, H, BS], input_dtype]
    HIDDEN: T.Tensor[[B, S // BS, H, DK, DV], input_dtype]
    O = T.empty([B, S, H, DV], output_dtype)

    with T.Kernel(T.ceildiv(DV, block_DV), T.ceildiv(S, block_S), B * H, threads=threads) as (bv, bs, bbh):
        bb, bh = bbh // H, bbh % H
        Q_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
        V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
        HIDDEN_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
        A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
        O_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
        O_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
        GK_shared = T.alloc_shared((block_S, block_DK), dtype=gate_dtype)
        GQ_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)

        T.clear(O_fragment)

        for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
            T.copy(Q[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], Q_shared)  # [block_S, block_DK]
            T.copy(GK[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], GK_shared)  # [block_S, block_DK]
            T.copy(
                HIDDEN[bb, bs, bh, i_k * block_DK : (i_k + 1) * block_DK, bv * block_DV : (bv + 1) * block_DV], HIDDEN_shared
            )  # [block_DK, block_DV]
            for i_s, i_k2 in T.Parallel(block_S, block_DK):
                Q_shared[i_s, i_k2] = Q_shared[i_s, i_k2] * scale
                GQ_shared[i_s, i_k2] = Q_shared[i_s, i_k2] * T.exp2(GK_shared[i_s, i_k2])
            T.gemm(GQ_shared, HIDDEN_shared, O_fragment)  # O_fragment as accumulator
        T.copy(V[bb, bs * block_S : (bs + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV], V_shared)  # [block_S, block_DV]
        T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, 0:block_S], A_shared)  # [block_S, block_S]

        for i_s1, i_s2 in T.Parallel(block_S, block_S):
            A_shared[i_s1, i_s2] = T.if_then_else(i_s1 < i_s2, 0, A_shared[i_s1, i_s2])

        T.gemm(
            A_shared,
            V_shared,
            O_fragment,
        )

        T.copy(O_fragment, O_shared)

        T.copy(O_shared, O[bb, bs * block_S : (bs + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV])

    return O


def do_bench(fn, *args, warmup=10, rep=10, **kwargs):
    """
    Do benchmark for a function.
    """
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    for _ in range(warmup):
        fn(*args, **kwargs)

    torch.cuda.synchronize()
    for i in range(rep):
        start_event[i].record()
        fn(*args, **kwargs)
        end_event[i].record()
    torch.cuda.synchronize()

    # Record clocks
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )

    return times.mean().item()


def run_test(
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
    block_DK,
    block_DV,
    threads,
    num_stages,
):
    input_dtype_torch = getattr(torch, input_dtype)
    output_dtype_torch = getattr(torch, output_dtype)
    accum_dtype_torch = getattr(torch, accum_dtype)
    gate_dtype_torch = getattr(torch, gate_dtype)
    Q, V, G, A, HIDDEN = prepare_input(
        B, S, H, DK, DV, chunk_size, input_dtype_torch, output_dtype_torch, accum_dtype_torch, gate_dtype_torch
    )
    scale = 1.0 / DK**0.5
    # scale = 1.0

    O_ref = prepare_output(B, S, H, DK, DV, chunk_size, output_dtype_torch)
    O_ref = chunk_gla_fwd_o_gk(Q, V, G, A, HIDDEN, scale, chunk_size=chunk_size, use_exp2=True)

    block_S = chunk_size
    O_tilelang = tilelang_chunk_fwd_o(
        Q,
        V,
        G,
        A,
        HIDDEN,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        accum_dtype=accum_dtype,
        gate_dtype=gate_dtype,
        chunk_size=chunk_size,
        scale=scale,
        block_S=block_S,
    )
    compare_tensors("O", O_ref, O_tilelang)
    fla_time = do_bench(chunk_gla_fwd_o_gk, Q, V, G, A, HIDDEN, scale, chunk_size=chunk_size, use_exp2=True)
    tilelang_time = do_bench(
        tilelang_chunk_fwd_o,
        Q,
        V,
        G,
        A,
        HIDDEN,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        accum_dtype=accum_dtype,
        gate_dtype=gate_dtype,
        chunk_size=chunk_size,
        scale=scale,
        block_S=block_S,
    )
    print("fla_time:", fla_time)
    print("tilelang_time:", tilelang_time)


def main():
    run_test(
        B=1,
        S=8192,
        H=64,
        DK=128,
        DV=128,
        chunk_size=64,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        block_DK=32,
        block_DV=32,
        threads=128,
        num_stages=1,
    )


if __name__ == "__main__":
    main()
