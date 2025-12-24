import argparse
import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, repeat
import itertools


def chunk_state_triton(B, x, dt, dA_cumsum):
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd

    return _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=False)


def ref_program(B, x, dt, dA_cumsum):
    """
    Argument:
        B: (batch, seqlen, ngroups, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)


def get_configs():
    iter_params = dict(block_M=[64, 128], block_N=[32, 64, 128], block_K=[32, 64], num_stages=[1, 2, 3, 4, 5])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[4])
def chunk_state_fwd(
    batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, block_M=64, block_N=64, block_K=64, num_stages=2, threads=128
):
    dtype = T.float16
    accum_dtype = T.float32
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504

    @T.prim_func
    def main(
        B: T.Tensor((batch, seqlen, ngroups, dstate), dtype),
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        dt: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),
        Output: T.Tensor((batch, nchunks, nheads, headdim, dstate), dtype),
    ):
        with T.Kernel(nheads, T.ceildiv(headdim, block_M) * T.ceildiv(dstate, block_N), batch * nchunks, threads=threads) as (bz, bx, by):
            x_shared = T.alloc_shared((block_K, block_M), dtype)
            x_local = T.alloc_fragment((block_K, block_M), dtype)
            xt_local = T.alloc_fragment((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            dt_shared = T.alloc_shared((block_K), dtype)
            dA_cumsum_shared = T.alloc_shared((block_K), dtype)
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            scale = T.alloc_fragment((block_K), accum_dtype)
            dA_cs_last = T.alloc_fragment((1), accum_dtype)
            dA_cumsum_local = T.alloc_fragment((block_K), accum_dtype)
            dt_local = T.alloc_fragment((block_K), accum_dtype)

            loop_range = T.ceildiv(chunk_size, block_K)

            batch_idx = by % batch
            chunk_idx = by // batch
            m_idx = bx // T.ceildiv(dstate, block_N)
            n_idx = bx % T.ceildiv(dstate, block_N)

            T.annotate_layout({x_shared: tilelang.layout.make_swizzled_layout(x_shared)})

            dA_cs_last[0] = dA_cumsum[batch_idx, bz, chunk_idx, chunk_size - 1]
            T.clear(acc_o)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    x[
                        batch_idx,
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                        bz,
                        m_idx * block_M : (m_idx + 1) * block_M,
                    ],
                    x_shared,
                )
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cumsum_shared)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dA_cumsum_shared, dA_cumsum_local)
                T.copy(dt_shared, dt_local)
                for i in T.Parallel(block_K):
                    scale[i] = T.exp2(dA_cs_last[0] * p - dA_cumsum_local[i] * p) * dt_local[i]
                T.copy(x_shared, x_local)
                for i, j in T.Parallel(block_M, block_K):
                    xt_local[i, j] = x_local[j, i] * scale[j]
                T.copy(
                    B[
                        batch_idx,
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                        bz // (nheads // ngroups),
                        n_idx * block_N : (n_idx + 1) * block_N,
                    ],
                    B_shared,
                )
                T.gemm(xt_local, B_shared, acc_o)
            T.copy(acc_o, acc_o_shared)
            T.copy(
                acc_o_shared,
                Output[batch_idx, chunk_idx, bz, m_idx * block_M : (m_idx + 1) * block_M, n_idx * block_N : (n_idx + 1) * block_N],
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
    total_flops = 2 * batch * seq_len * heads * dim * dstate

    if not args.tune:
        kernel = chunk_state_fwd(
            batch, seq_len, chunk_size, groups, heads, dim, dstate, block_M=64, block_N=128, block_K=64, num_stages=4, threads=128
        )
        profiler = kernel.get_profiler(tilelang.TensorSupplyType.Normal)
        profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = profiler.do_bench(ref_program, warmup=500)
        print("Ref: {:.2f} ms".format(latency))
        print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(warmup=500)
        print("Tile-lang: {:.2f} ms".format(latency))
        print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    else:
        best_result = chunk_state_fwd(batch, seq_len, chunk_size, groups, heads, dim, dstate)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
