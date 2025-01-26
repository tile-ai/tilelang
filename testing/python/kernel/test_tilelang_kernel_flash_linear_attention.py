# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import tilelang.language as T


def chunk_scan_fwd(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, block_M, block_N,
                   block_K, block_Dstate, num_stages, threads):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504

    @T.prim_func
    def main(cb: T.Buffer((batch, nchunks, ngroups, chunk_size, chunk_size), dtype), x: T.Buffer(
        (batch, seqlen, nheads, headdim), dtype), dt: T.Buffer(
            (batch, nheads, nchunks, chunk_size), dtype), dA_cumsum: T.Buffer(
                (batch, nheads, nchunks, chunk_size), dtype),
             C: T.Buffer((batch, seqlen, ngroups, dstate), dtype), prev_states: T.Buffer(
                 (batch, nchunks, nheads, headdim, dstate), dtype), D: T.Buffer(
                     (nheads), dtype), Output: T.Buffer((batch, seqlen, nheads, headdim), dtype)):
        with T.Kernel(
                nheads,
                T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N),
                batch * nchunks,
                threads=threads) as (bz, bx, by):
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            cb_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.dyn")
            cb_local = T.alloc_fragment((block_M, block_K), dtype)
            dA_cs_k_shared = T.alloc_shared((block_K), dtype, scope="shared")
            dA_cs_k_local = T.alloc_fragment((block_K), accum_dtype)
            dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_K), dtype, scope="shared")
            dt_local = T.alloc_fragment((block_K), accum_dtype)
            x_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.dyn")
            dA_cs_m_shared = T.alloc_shared((block_M), dtype, scope="shared")
            scale_m_local = T.alloc_fragment((block_M), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)
            D_local = T.alloc_fragment((1), accum_dtype)
            x_residual_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.dyn")
            x_residual_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            batch_idx = by % batch
            chunk_idx = by // batch
            # m: chunk_size
            # n : headdim
            m_idx = bx // T.ceildiv(headdim, block_N)
            n_idx = bx % T.ceildiv(headdim, block_N)

            T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M:(m_idx + 1) * block_M],
                   dA_cs_m_shared)
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)

            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[batch_idx, chunk_idx * chunk_size + m_idx * block_M:chunk_idx * chunk_size +
                  (m_idx + 1) * block_M, bz // (nheads // ngroups), 0:block_Dstate], C_shared)
            T.copy(
                prev_states[batch_idx, chunk_idx, bz, n_idx * block_N:(n_idx + 1) * block_N,
                            0:block_Dstate], prev_state_shared)
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    cb[batch_idx, chunk_idx, bz // (nheads // ngroups),
                       m_idx * block_M:(m_idx + 1) * block_M, k * block_K:(k + 1) * block_K],
                    cb_shared)
                T.copy(cb_shared, cb_local)
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K:(k + 1) * block_K],
                       dA_cs_k_shared)
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i,
                             j] = cb_local[i,
                                           j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K:(k + 1) * block_K], dt_shared)
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(m_idx * block_M + i >= k * block_K + j,
                                                    cb_local[i, j], 0)
                T.copy(
                    x[batch_idx, chunk_idx * chunk_size + k * block_K:chunk_idx * chunk_size +
                      (k + 1) * block_K, bz, n_idx * block_N:(n_idx + 1) * block_N], x_shared)
                T.gemm(cb_local, x_shared, acc_o)

            D_local[0] = D[bz]
            T.copy(
                x[batch_idx, chunk_idx * chunk_size + m_idx * block_M:chunk_idx * chunk_size +
                  (m_idx + 1) * block_M, bz, n_idx * block_N:(n_idx + 1) * block_N],
                x_residual_shared)
            T.copy(x_residual_shared, x_residual_local)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] += x_residual_local[i, j] * D_local[0]

            T.copy(
                acc_o,
                Output[batch_idx, chunk_idx * chunk_size + m_idx * block_M:chunk_idx * chunk_size +
                       (m_idx + 1) * block_M, bz, n_idx * block_N:(n_idx + 1) * block_N])

    return main


def run_chunk_scan(batch,
                   seqlen,
                   chunk_size,
                   ngroups,
                   nheads,
                   headdim,
                   dstate,
                   block_M,
                   block_N,
                   block_K,
                   block_Dstate,
                   num_stages=2,
                   threads=128):
    program = chunk_scan_fwd(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, block_M,
                             block_N, block_K, block_Dstate, num_stages, threads)

    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [7], tl.TensorSupplyType.Integer)

    def ref_program(cb, x, dt, dA_cumsum, C, prev_states, D):
        import torch
        from einops import rearrange, repeat
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
        causal_mask = torch.tril(
            torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
        scores_decay = scores_decay.masked_fill(~causal_mask, 0)
        out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                           rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
        state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
        out_prev = torch.einsum('bclhn,bchpn->bclhp',
                                rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                                prev_states.to(C.dtype)) * state_decay_out
        out = out + out_prev
        out = rearrange(out, "b c l h p -> b (c l) h p")
        if D is not None:
            if D.dim() == 1:
                D = rearrange(D, "h -> h 1")
            out = out + x * D
        return out

    mod.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def chunk_state_fwd(batch,
                    seqlen,
                    chunk_size,
                    ngroups,
                    nheads,
                    headdim,
                    dstate,
                    block_M,
                    block_N,
                    block_K,
                    num_stages=2,
                    threads=128):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504

    @T.prim_func
    def main(B: T.Buffer((batch, seqlen, ngroups, dstate), dtype), x: T.Buffer(
        (batch, seqlen, nheads, headdim), dtype), dt: T.Buffer(
            (batch, nheads, nchunks, chunk_size), dtype), dA_cumsum: T.Buffer(
                (batch, nheads, nchunks, chunk_size), dtype), Output: T.Buffer(
                    (batch, nchunks, nheads, headdim, dstate), dtype)):
        with T.Kernel(
                nheads,
                T.ceildiv(headdim, block_M) * T.ceildiv(dstate, block_N),
                batch * nchunks,
                threads=threads) as (bz, bx, by):
            x_shared = T.alloc_shared((block_K, block_M), dtype)
            x_local = T.alloc_fragment((block_K, block_M), dtype)
            xt_local = T.alloc_fragment((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            dt_shared = T.alloc_shared((block_K), dtype)
            dA_cumsum_shared = T.alloc_shared((block_K), dtype)
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            scale = T.alloc_fragment((block_K), accum_dtype)
            dA_cs_last = T.alloc_fragment((1), accum_dtype)
            dA_cumsum_local = T.alloc_fragment((block_K), accum_dtype)
            dt_local = T.alloc_fragment((block_K), accum_dtype)

            loop_range = T.ceildiv(chunk_size, block_K)

            batch_idx = by % batch
            chunk_idx = by // batch
            m_idx = bx // T.ceildiv(dstate, block_N)
            n_idx = bx % T.ceildiv(dstate, block_N)

            dA_cs_last[0] = dA_cumsum[batch_idx, bz, chunk_idx, chunk_size - 1]
            T.clear(acc_o)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    x[batch_idx, chunk_idx * chunk_size + k * block_K:chunk_idx * chunk_size +
                      (k + 1) * block_K, bz, m_idx * block_M:(m_idx + 1) * block_M], x_shared)
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K:(k + 1) * block_K],
                       dA_cumsum_shared)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K:(k + 1) * block_K], dt_shared)
                T.copy(dA_cumsum_shared, dA_cumsum_local)
                T.copy(dt_shared, dt_local)
                for i in T.Parallel(block_K):
                    scale[i] = T.exp2(dA_cs_last[0] * p - dA_cumsum_local[i] * p) * dt_local[i]
                T.copy(x_shared, x_local)
                for i, j in T.Parallel(block_M, block_K):
                    xt_local[i, j] = x_local[j, i] * scale[j]
                T.copy(
                    B[batch_idx, chunk_idx * chunk_size + k * block_K:chunk_idx * chunk_size +
                      (k + 1) * block_K, bz // (nheads // ngroups),
                      n_idx * block_N:(n_idx + 1) * block_N], B_shared)
                T.gemm(xt_local, B_shared, acc_o)
            T.copy(
                acc_o, Output[batch_idx, chunk_idx, bz, m_idx * block_M:(m_idx + 1) * block_M,
                              n_idx * block_N:(n_idx + 1) * block_N])

    return main


def run_chunk_state(batch,
                    seqlen,
                    chunk_size,
                    ngroups,
                    nheads,
                    headdim,
                    dstate,
                    block_M,
                    block_N,
                    block_K,
                    num_stages=2,
                    threads=128):
    program = chunk_state_fwd(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, block_M,
                              block_N, block_K, num_stages, threads)

    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [4], tl.TensorSupplyType.Integer)

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
        import torch
        import torch.nn.functional as F
        from einops import rearrange, repeat

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
        return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype),
                            dt.to(x.dtype), x)

    mod.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_chunk_scan():
    run_chunk_scan(
        batch=8,
        seqlen=2048,
        chunk_size=256,
        ngroups=1,
        nheads=8,
        headdim=64,
        dstate=128,
        block_M=64,
        block_N=64,
        block_K=64,
        block_Dstate=128,
        num_stages=2,
        threads=128)


def test_chunk_state():
    run_chunk_state(
        batch=8,
        seqlen=2048,
        chunk_size=256,
        ngroups=1,
        nheads=8,
        headdim=64,
        dstate=128,
        block_M=64,
        block_N=64,
        block_K=64,
        num_stages=2,
        threads=128)


if __name__ == "__main__":
    tilelang.testing.main()
