import torch
import argparse
from tilelang.profiler import do_bench
from tilelang import language as T
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from example_gqa_sink_fwd_bhsd_wgmma_pipelined import flashattn, ref_program, gen_inputs
from typing import Optional


@triton.jit
def triton_kernel(
    Q,
    K,
    V,
    Sinks,
    sm_scale,
    Out,
    Z,
    H,
    N_Q_CTX,
    N_KV_CTX,
    HEAD_DIM: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BANDWIDTH: tl.constexpr,
    start_q: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # load attention sinks
    if Sinks is not None:  # noqa: SIM108
        sink = tl.load(Sinks + off_h).to(tl.float32)
    else:
        sink = 0

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])

    if BANDWIDTH:
        lo, hi = tl.maximum(0, start_q + start_m * BLOCK_M - BANDWIDTH), start_q + (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, start_q + (start_m + 1) * BLOCK_M

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]

        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        k = K.load([off_z, off_h // groups, start_n, 0]).reshape([BLOCK_N, HEAD_DIM]).T
        qk = tl.dot(q, k, allow_tf32=False)

        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp(qk)
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = V.load([off_z, off_h // groups, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        # v = v.to(tl.float32)
        p = p.to(v.dtype)  # We perform fp16 gemm to utilize tensor core
        acc = tl.dot(p, v, acc, allow_tf32=False)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    sink = tl.math.exp(sink - m_i)
    z = l_i + sink
    acc = acc / z[:, None]
    # m_i += tl.math.log(l_i)
    # m_ptrs = M + off_hz * N_Q_CTX + offs_m
    # tl.store(m_ptrs, m_i)
    acc = acc.to(Out.dtype)[None, None, :, :]
    Out.store([off_z, off_h, start_m * BLOCK_M, 0], acc)


def triton_program(Q, K, V, Sinks, window_size: Optional[int] = None) -> torch.Tensor:
    bs, n_heads, seq_q, head_dim = Q.shape
    _, n_heads_kv, seq_kv, _ = K.shape
    BLOCK_M = 64
    BLOCK_N = 64
    groups = n_heads // n_heads_kv

    o = torch.empty_like(Q)
    grid = (triton.cdiv(seq_q, BLOCK_M), bs * n_heads, 1)
    triton_kernel[grid](
        TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, head_dim]),
        TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, head_dim]),
        TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, head_dim]),
        Sinks,
        1.0 / head_dim**0.5,
        TensorDescriptor.from_tensor(o, [1, 1, BLOCK_M, head_dim]),
        bs,
        n_heads,
        N_Q_CTX=seq_q,
        N_KV_CTX=seq_kv,
        HEAD_DIM=head_dim,
        groups=groups,
        BANDWIDTH=window_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        start_q=seq_kv - seq_q,
    )
    return o


def main(
    batch: int = 1,
    heads: int = 32,
    seq_q: int = 256,
    seq_kv: int = 256,
    dim: int = 128,
    groups: int = 8,
    window_size: Optional[int] = None,
    dtype: str = "float16",
    tune: bool = False,
):
    dtype = T.dtype(dtype)
    torch_dtype = dtype.as_torch()
    if window_size is not None:
        print("Using sliding window attention.")
        assert window_size <= seq_q
        flops_per_matmul = 2.0 * batch * heads * min(window_size, seq_kv // 2) * seq_q * dim  # just a rough estimation
    else:
        print("Using full attention.")
        flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim * 0.5
    total_flops = 2 * flops_per_matmul

    if tune:
        kernel = flashattn(batch, heads, seq_q, seq_kv, dim, groups, window_size, dtype=dtype)
        print(f"Best latency: {kernel.latency}")
        print(f"Best TFlops: {total_flops / kernel.latency * 1e-9}")
        print(f"Best config: {kernel.config}")
    else:
        block_M = 128
        block_N = 128
        num_stages = 2
        threads = 256
        print(f"{block_M=}, {block_N=}, {num_stages=}, {threads=}")

        kernel = flashattn(
            batch,
            heads,
            seq_q,
            seq_kv,
            dim,
            groups,
            window_size,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
            dtype=dtype,
        )

        Q, K, V, sinks = gen_inputs(batch, heads, seq_q, seq_kv, dim, groups, dtype=torch_dtype)

        if torch.allclose(
            triton_program(Q, K, V, sinks, window_size), ref_program(Q, K, V, sinks, window_size, dtype=torch_dtype), rtol=1e-2, atol=1e-2
        ):
            print("Checks for triton passed.✅")
        else:
            print("Checks for triton failed.❌")

        # Benchmark triton
        latency_triton = do_bench(lambda: triton_program(Q, K, V, sinks, window_size), warmup=500)
        print("Triton: {:.2f} ms".format(latency_triton))
        print("Triton: {:.2f} TFlops".format(total_flops / latency_triton * 1e-9))

        # Benchmark tilelang
        latency_tilelang = do_bench(lambda: kernel(Q, K, V, sinks), warmup=500)
        print("Tilelang: {:.2f} ms".format(latency_tilelang))
        print("Tilelang: {:.2f} TFlops".format(total_flops / latency_tilelang * 1e-9))

        print("Speedup: {:.2f}x".format(latency_triton / latency_tilelang))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="heads")
    parser.add_argument("--seq_q", type=int, default=2048, help="sequence length of query")
    parser.add_argument("--seq_kv", type=int, default=2048, help="sequence length of key/value")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--groups", type=int, default=8, help="groups")
    parser.add_argument("--window_size", type=int, default=None, help="window size (default: None, which means full attention)")
    parser.add_argument("--dtype", type=str, default="float16", help="dtype, can be float16 or bfloat16")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_q, args.seq_kv, args.dim, args.groups, args.window_size, args.dtype, args.tune)
