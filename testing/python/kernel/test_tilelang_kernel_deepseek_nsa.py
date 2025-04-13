# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
# ruff: noqa
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import torch
from typing import Optional, Union
from einops import rearrange, repeat

tilelang.testing.set_random_seed(42)


def naive_nsa_ref(q: torch.Tensor,
                  k: torch.Tensor,
                  v: torch.Tensor,
                  g_slc: torch.Tensor,
                  g_swa: torch.Tensor,
                  block_indices: torch.LongTensor,
                  block_counts: Optional[Union[torch.LongTensor, int]] = None,
                  block_size: int = 64,
                  window_size: int = 0,
                  scale: Optional[float] = None,
                  cu_seqlens: Optional[torch.LongTensor] = None,
                  head_first: bool = False) -> torch.Tensor:

    if scale is None:
        scale = k.shape[-1]**-0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
        if head_first:
            raise RuntimeError(
                "Sequences with variable lengths are not supported for head-first mode")
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'),
                                     (q, k, v, block_indices))
        g_slc, g_swa = map(lambda x: rearrange(x, 'b h t -> b t h'), (g_slc, g_swa))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, 'b h t -> b t h')

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    S = block_indices.shape[-1]
    k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
    if isinstance(block_counts, torch.Tensor):
        block_counts = repeat(block_counts, 'b t h -> b t (h g)', g=G)
    c = torch.arange(S).repeat_interleave(BS).unsqueeze(1).expand(-1, q.shape[2]).to(q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))

    o_slc = torch.zeros_like(v)
    o_swa = torch.zeros_like(v) if window_size > 0 else None
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat(
            [block_indices.new_tensor(range(0, B * T, T)),
             block_indices.new_tensor([B * T])])

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = q[i], k[i], v[i], g_slc[i], g_swa[
                i], block_indices[i]
            if isinstance(block_counts, torch.Tensor):
                s_b = block_counts[i]
            else:
                s_b = block_counts
        else:
            T = cu_seqlens[i + 1] - cu_seqlens[i]
            q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = map(
                lambda x: x[0][cu_seqlens[i]:cu_seqlens[i + 1]],
                (q, k, v, g_slc, g_swa, block_indices))
            if isinstance(block_counts, torch.Tensor):
                s_b = block_counts[0][cu_seqlens[i]:cu_seqlens[i + 1]]
            else:
                s_b = block_counts

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, S*BS, HQ]
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [HQ]
            g_slc_i = g_slc_b[i_q]
            # [HQ]
            g_swa_i = g_swa_b[i_q]
            # [S*BS, HQ]
            i_i = i_b[i_q]
            # [HQ]
            if isinstance(block_counts, torch.Tensor):
                s_i = s_b[i_q]
            else:
                s_i = s_b
            # [S*BS, HQ, -1]
            k_i_slc, v_i_slc = map(
                lambda x: x.gather(
                    0,
                    i_i.clamp(0, T - 1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])), (k_b, v_b))
            # [S*BS, HQ]
            attn_slc = torch.einsum('h d, n h d -> n h', q_i, k_i_slc).masked_fill(
                torch.logical_or(i_i < 0, i_i > i_q) |
                (c >= s_i if block_counts is not None else False), float('-inf')).softmax(0)
            if not varlen:
                o_slc[i, i_q] = torch.einsum('n h, n h v -> h v', attn_slc,
                                             v_i_slc) * g_slc_i.unsqueeze(-1)
            else:
                o_slc[0][cu_seqlens[i] + i_q] = torch.einsum('n h, n h v -> h v', attn_slc,
                                                             v_i_slc) * g_slc_i.unsqueeze(-1)
            if window_size > 0:
                k_i_swa, v_i_swa = map(lambda x: x[max(0, i_q - window_size + 1):i_q + 1],
                                       (k_b, v_b))
                attn_swa = torch.einsum('h d, n h d -> n h', q_i, k_i_swa).softmax(0)
                if not varlen:
                    o_swa[i, i_q] = torch.einsum('n h, n h v -> h v', attn_swa,
                                                 v_i_swa) * g_swa_i.unsqueeze(-1)
                else:
                    o_swa[0][cu_seqlens[i] + i_q] = torch.einsum('n h, n h v -> h v', attn_swa,
                                                                 v_i_swa) * g_swa_i.unsqueeze(-1)

    if head_first:
        o_slc = rearrange(o_slc, 'b t h d -> b h t d')
        o_swa = rearrange(o_swa, 'b t h d -> b h t d')

    return o_slc.to(dtype) + o_swa.to(dtype) if o_swa is not None else o_slc.to(dtype)


def native_sparse_attention(batch,
                            heads,
                            seq_len,
                            dim,
                            is_causal,
                            scale=None,
                            block_size=64,
                            groups=16,
                            selected_blocks=16,
                            num_stages=0,
                            threads=32):
    if scale is None:
        scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    dtype = "float16"
    accum_dtype = "float"
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(dim))

    NK = tilelang.cdiv(dim, block_T)
    NV = tilelang.cdiv(dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T

    @T.prim_func
    def native_sparse_attention(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([G, BV], dtype)

            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_max_prev = T.alloc_fragment([G], accum_dtype)
            scores_scale = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            logsum = T.alloc_fragment([G], accum_dtype)

            i_t, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            T.copy(Q[i_b, i_t, i_h * G:(i_h + 1) * G, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i in T.Pipelined(NS, num_stages=num_stages):
                i_s = BlockIndices[i_b, i_t, i_h, i] * BS
                if i_s <= i_t and i_s >= 0:
                    # [BS, BK]
                    T.copy(K[i_b, i_s:i_s + BS, i_h, :], K_shared)

                    if is_causal:
                        for i, j in T.Parallel(G, BS):
                            acc_s[i, j] = T.if_then_else(i_t >= (i_s + j), 0,
                                                         -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)

                    # Softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                    for i in T.Parallel(G):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(G, BS):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(G):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    # Rescale
                    for i, j in T.Parallel(G, BV):
                        acc_o[i, j] *= scores_scale[i]

                    # V * softmax(Q * K)
                    T.copy(V[i_b, i_s:i_s + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(G, BV):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, i_t, i_h * G:(i_h + 1) * G, i_v * BV:(i_v + 1) * BV])

    return native_sparse_attention


def run_native_sparse_attention(batch,
                                heads,
                                seq_len,
                                dim,
                                is_causal,
                                scale=None,
                                block_size=64,
                                groups=16,
                                selected_blocks=16,
                                num_stages=0,
                                threads=32):
    dtype = torch.float16
    head_kv = heads // groups
    program = native_sparse_attention(batch, heads, seq_len, dim, is_causal, scale, block_size,
                                      groups, selected_blocks, num_stages, threads)
    kernel = tilelang.compile(program, out_idx=-1)
    Q = torch.randn((batch, seq_len, heads, dim), dtype=dtype).cuda()
    K = torch.randn((batch, seq_len, head_kv, dim), dtype=dtype).cuda()
    V = torch.randn((batch, seq_len, head_kv, dim), dtype=dtype).cuda()
    g_slc = torch.ones((batch, seq_len, heads), dtype=dtype).cuda()
    g_swa = torch.ones((batch, seq_len, heads), dtype=dtype).cuda()

    block_indices = torch.full((batch, seq_len, head_kv, selected_blocks),
                               seq_len,
                               dtype=torch.long,
                               device='cuda')
    for b in range(batch):
        for t in range(seq_len):
            for h in range(head_kv):
                i_i = torch.randperm(max(1, (t // block_size)))[:selected_blocks]
                block_indices[b, t, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    block_counts = torch.randint(1, selected_blocks + 1, (batch, seq_len, head_kv), device='cuda')

    out = kernel(Q, K, V, block_indices.to(torch.int32))

    ref = naive_nsa_ref(
        q=Q,
        k=K,
        v=V,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale,
    )
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)


def test_tilelang_kernel_deepseek_nsa():
    # disable pipeline
    run_native_sparse_attention(
        batch=2,
        heads=64,
        seq_len=1,
        dim=16,
        is_causal=True,
        scale=None,
        block_size=32,
        groups=16,
        selected_blocks=16,
        num_stages=0,
        threads=32)
    # enable pipeline
    run_native_sparse_attention(
        batch=2,
        heads=64,
        seq_len=1,
        dim=16,
        is_causal=True,
        scale=None,
        block_size=32,
        groups=16,
        selected_blocks=16,
        num_stages=2,
        threads=32)


if __name__ == "__main__":
    tilelang.testing.main()
