import argparse
import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, repeat
import itertools
import math
from tilelang.profiler import do_bench

try:
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
except ImportError as err:
    raise ImportError("Please install mamba-ssm to use the triton chunk scan operator.") from err

try:
    import helion
    from helion._testing import run_example
    import helion.language as hl
except ImportError as err:
    raise ImportError("Please install helion to use the helion chunk scan operator.") from err


def ref_program(cb, x, dt, dA_cumsum, C, prev_states, D):
    """
    Argument:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    _, _, ngroups, _, _ = cb.shape
    batch, seqlen, nheads, headdim = x.shape
    # _, _, ngroups, dstate = B.shape
    # assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    # assert C.shape == B.shape
    # B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    cb = repeat(cb, "b c g l s -> b c (g h) l s", h=nheads // ngroups)
    # CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
    #                   rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum(
        "bchls,bhcs,bcshp->bclhp", scores_decay.to(x.dtype), dt.to(x.dtype), rearrange(x, "b (c s) h p -> b c s h p", c=nchunks)
    )
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = (
        torch.einsum("bclhn,bchpn->bclhp", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks), prev_states.to(C.dtype)) * state_decay_out
    )
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out


def chunk_scan_triton(cb, x, dt, dA_cumsum, C, states, D):
    out, _ = _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D)
    return out


def chunk_scan_helion(cb, x, dt, dA_cumsum, C, states, D):
    @helion.kernel()
    def helion_mamba2_chunk_scan_kernel(
        cb: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        C: torch.Tensor,
        prev_states: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        Argument:
            cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
            x: (batch, seqlen, nheads, headdim)
            dt: (batch, nheads, nchunks, chunk_size)
            dA_cumsum: (batch, nheads, nchunks, chunk_size)
            C: (batch, seqlen, ngroups, dstate)
            prev_states: (batch, nchunks, nheads, headdim, dstate)
            D: (nheads,)
        Return:
            out: (batch, seqlen, nheads, headdim)
        """

        batch, nchunks, ngroups, chunk_size, _ = cb.shape
        _, seqlen, nheads, headdim = x.shape
        _, _, _, dstate = C.shape
        assert nchunks == (seqlen + chunk_size - 1) // chunk_size

        block_m = hl.register_block_size(chunk_size)
        block_n = hl.register_block_size(headdim)
        block_k = hl.register_block_size(64, 64)
        dstate = hl.specialize(dstate)

        assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
        assert x.shape == (batch, seqlen, nheads, headdim)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        assert C.shape == (batch, seqlen, ngroups, dstate)
        assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
        assert D.shape == (nheads,)

        dtype = cb.dtype
        accum_dtype = torch.float32
        assert x.dtype == dt.dtype == dA_cumsum.dtype == C.dtype == prev_states.dtype == D.dtype == dtype

        out = torch.empty_like(x)

        p = 1.44269504

        for tile_h, tile_m, tile_n, tile_b, tile_c in hl.tile(
            [nheads, chunk_size, headdim, batch, nchunks],
            block_size=[1, block_m, block_n, 1, 1],
        ):
            acc_o = hl.zeros([tile_m, tile_n], dtype=accum_dtype)
            dA_cumsum_local_m = dA_cumsum[tile_b.begin, tile_h.begin, tile_c.begin, tile_m].to(torch.float32)
            scale_m_local = torch.exp2(dA_cumsum_local_m * p)

            C_local = C[
                tile_b.begin,
                tile_m.index + tile_c.begin * chunk_size,
                tile_h.begin // (nheads // ngroups),
                :,
            ]
            prev_states_local = prev_states[tile_b.begin, tile_c.begin, tile_h.begin, tile_n, :]
            acc_o = hl.dot(C_local, prev_states_local.T, acc=acc_o)
            acc_o *= scale_m_local[:, None]

            for tile_k in hl.tile((tile_m.id + 1) * block_m, block_size=block_k):
                cb_local = cb[
                    tile_b.begin,
                    tile_c.begin,
                    tile_h.begin // (nheads // ngroups),
                    tile_m,
                    tile_k,
                ]
                dA_cumsum_local_k = dA_cumsum[tile_b.begin, tile_h.begin, tile_c.begin, tile_k].to(torch.float32)
                cb_local *= torch.exp2(dA_cumsum_local_m[:, None] * p - dA_cumsum_local_k[None, :] * p)
                dt_local = dt[tile_b.begin, tile_h.begin, tile_c.begin, tile_k].to(torch.float32)
                cb_local = (cb_local * dt_local[None, :]).to(dtype)
                pred = (tile_m.index + 0)[:, None] >= (tile_k.index + 0)[None, :]
                cb_local = torch.where(pred, cb_local, torch.zeros_like(cb_local))
                x_local = x[
                    tile_b.begin,
                    tile_c.begin * chunk_size + tile_k.index,
                    tile_h.begin,
                    tile_n,
                ]
                acc_o = hl.dot(cb_local, x_local, acc=acc_o)

            D_local = D[tile_h.begin].to(torch.float32)
            x_residual = x[tile_b.begin, tile_c.begin * chunk_size + tile_m.index, tile_h.begin, tile_n].to(torch.float32)
            acc_o += x_residual * D_local
            out[tile_b.begin, tile_c.begin * chunk_size + tile_m.index, tile_h.begin, tile_n] = acc_o.to(dtype=dtype)

        return out

    args = (cb, x, dt, dA_cumsum, C, states, D)
    run_example(helion_mamba2_chunk_scan_kernel, ref_program, args)


def get_configs():
    iter_params = dict(block_M=[64, 128, 256], block_N=[32, 64], block_K=[64, 128, 256], block_Dstate=[128], num_stages=[1, 2, 3, 4, 5])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[7],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def chunk_scan_fwd(
    batch,
    seqlen,
    chunk_size,
    ngroups,
    nheads,
    headdim,
    dstate,
    block_M=64,
    block_N=64,
    block_K=64,
    block_Dstate=128,
    num_stages=2,
    threads=128,
):
    dtype = T.float16
    accum_dtype = T.float32
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504

    @T.prim_func
    def main(
        cb: T.Tensor((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),  # type: ignore
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
        dt: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),  # type: ignore
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),  # type: ignore
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),  # type: ignore
        prev_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), dtype),  # type: ignore
        D: T.Tensor((nheads), dtype),  # type: ignore
        Output: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
    ):
        with T.Kernel(nheads, T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N), batch * nchunks, threads=threads) as (
            bz,
            bx,
            by,
        ):
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            cb_shared = T.alloc_shared((block_M, block_K), dtype)
            cb_local = T.alloc_fragment((block_M, block_K), dtype)
            dA_cs_k_shared = T.alloc_shared((block_K), dtype)
            dA_cs_k_local = T.alloc_fragment((block_K), accum_dtype)
            dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_K), dtype)
            dt_local = T.alloc_fragment((block_K), accum_dtype)
            x_shared = T.alloc_shared((block_K, block_N), dtype)
            dA_cs_m_shared = T.alloc_shared((block_M), dtype)
            scale_m_local = T.alloc_fragment((block_M), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)
            D_local = T.alloc_fragment((1), accum_dtype)
            x_residual_shared = T.alloc_shared((block_M, block_N), dtype)
            x_residual_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            batch_idx = by % batch
            chunk_idx = by // batch
            # m: chunk_size
            # n : headdim
            m_idx = bx // T.ceildiv(headdim, block_N)
            n_idx = bx % T.ceildiv(headdim, block_N)

            T.annotate_layout(
                {
                    cb_shared: tilelang.layout.make_swizzled_layout(cb_shared),
                    x_residual_shared: tilelang.layout.make_swizzled_layout(x_residual_shared),
                }
            )

            T.no_set_max_nreg()

            T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M], dA_cs_m_shared)
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)

            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[
                    batch_idx,
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz // (nheads // ngroups),
                    0:block_Dstate,
                ],
                C_shared,
            )
            T.copy(prev_states[batch_idx, chunk_idx, bz, n_idx * block_N : (n_idx + 1) * block_N, 0:block_Dstate], prev_state_shared)
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    cb[
                        batch_idx,
                        chunk_idx,
                        bz // (nheads // ngroups),
                        m_idx * block_M : (m_idx + 1) * block_M,
                        k * block_K : (k + 1) * block_K,
                    ],
                    cb_shared,
                )
                T.copy(cb_shared, cb_local)
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cs_k_shared)
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0)
                T.copy(
                    x[
                        batch_idx,
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                        bz,
                        n_idx * block_N : (n_idx + 1) * block_N,
                    ],
                    x_shared,
                )
                T.gemm(cb_local, x_shared, acc_o)

            D_local[0] = D[bz]
            T.copy(
                x[
                    batch_idx,
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                ],
                x_residual_shared,
            )
            T.copy(x_residual_shared, x_residual_local)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] += x_residual_local[i, j] * D_local[0]

            T.copy(acc_o, acc_o_shared)
            T.copy(
                acc_o_shared,
                Output[
                    batch_idx,
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                ],
            )

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=80, help="heads")
    parser.add_argument("--groups", type=int, default=1, help="groups")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--chunk_size", type=int, default=256, help="chunk size")
    parser.add_argument("--dim", type=int, default=64, help="dim")
    parser.add_argument("--dstate", type=int, default=128, help="dstate")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    args = parser.parse_args()
    batch, heads, groups, seq_len, chunk_size, dim, dstate = (
        args.batch,
        args.heads,
        args.groups,
        args.seq_len,
        args.chunk_size,
        args.dim,
        args.dstate,
    )
    nchunks = math.ceil(seq_len / chunk_size)
    total_flops = 2 * batch * seq_len * chunk_size * heads * dim * 0.5 + 2 * batch * seq_len * heads * dim * dstate

    print("Benchmarking TileLang...")
    kernel = chunk_scan_fwd(batch, seq_len, chunk_size, groups, heads, dim, dstate)
    best_latency = kernel.latency
    best_config = kernel.config
    ref_latency = kernel.ref_latency
    print(f"Best latency: {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    print(f"Best config: {best_config}")

    cb = torch.randn(batch, nchunks, groups, chunk_size, chunk_size).half().cuda()
    x = torch.randn(batch, seq_len, heads, dim).half().cuda()
    dt = torch.randn(batch, heads, nchunks, chunk_size).half().cuda()
    dA_cumsum = torch.randn(batch, heads, nchunks, chunk_size).half().cuda()
    C = torch.randn(batch, seq_len, groups, dstate).half().cuda()
    states = torch.randn(batch, nchunks, heads, dim, dstate).half().cuda()
    D = torch.randn(heads).half().cuda()

    print("Benchmarking Triton...")
    triton_latency = do_bench(lambda: chunk_scan_triton(cb, x, dt, dA_cumsum, C, states, D), _n_warmup=10, _n_repeat=10)
    print(f"Triton TFlops: {total_flops / triton_latency * 1e-9}")

    print("Benchmarking Helion...")
    chunk_scan_helion(cb, x, dt, dA_cumsum, C, states, D)
