# ruff: noqa
import torch
from typing import Optional, Union
from packaging.version import parse

import tilelang
from tilelang import language as T
import tilelang.testing

import fla

if parse(fla.__version__) < parse("0.2.1"):
    from fla.ops.common.utils import prepare_token_indices
else:
    from fla.ops.utils import prepare_token_indices
from reference import naive_nsa
from einops import rearrange


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def native_sparse_attention_varlen(batch, heads, c_seq_len, dim, is_causal, scale=None, block_size=64, groups=1, selected_blocks=16):
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [c_seq_len, heads, dim]
    kv_shape = [c_seq_len, head_kv, dim]
    o_slc_shape = [c_seq_len, heads, dim]
    o_swa_shape = [c_seq_len, heads, dim]
    lse_slc_shape = [c_seq_len, heads]
    lse_swa_shape = [c_seq_len, heads]
    block_indices_shape = [c_seq_len, head_kv, selected_blocks]
    block_counts_shape = [c_seq_len, head_kv]
    offsets_shape = [batch + 1]
    token_indices_shape = [c_seq_len, 2]
    block_indices_dtype = T.int32
    block_counts_dtype = T.int32
    offsets_dtype = T.int32
    token_indices_dtype = T.int32
    dtype = T.float16
    accum_dtype = T.float32
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(dim))

    NK = tilelang.cdiv(dim, block_T)
    NV = tilelang.cdiv(dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T
    num_stages = 0
    threads = 32

    @T.prim_func
    def native_sparse_attention_varlen(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        O_slc: T.Tensor(o_slc_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
        BlockCounts: T.Tensor(block_counts_shape, block_counts_dtype),
        Offsets: T.Tensor(offsets_shape, offsets_dtype),
        TokenIndices: T.Tensor(token_indices_shape, token_indices_dtype),
    ):
        with T.Kernel(c_seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
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

            i_c, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            i_n, i_t = TokenIndices[i_c, 0], TokenIndices[i_c, 1]

            bos = Offsets[i_n]
            eos = Offsets[i_n + 1]
            current_seq_len = eos - bos

            NS = BlockCounts[i_t, i_h]
            T.copy(Q[bos + i_t, i_h * G : (i_h + 1) * G, :BK], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i in T.Pipelined(NS, num_stages=num_stages):
                i_s = BlockIndices[bos + i_t, i_h, i] * BS
                if i_s <= i_t and i_s >= 0:
                    # [BS, BK]
                    # Lei: may have some padding issues
                    # we should learn from mha varlen templates to handle this
                    T.copy(K[bos + i_s : bos + i_s + BS, i_h, :BK], K_shared)

                    if is_causal:
                        for i, j in T.Parallel(G, BS):
                            acc_s[i, j] = T.if_then_else(i_t >= (i_s + j), 0, -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

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
                    T.copy(V[bos + i_s : bos + i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(G, BV):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, O_slc[bos + i_t, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV])

    return native_sparse_attention_varlen


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    window_size: int,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, C_SEQ_LEN, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]

    batch = len(offsets) - 1
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    WS = window_size

    kernel = native_sparse_attention_varlen(
        batch=batch,
        heads=HQ,
        c_seq_len=C_SEQ_LEN,
        dim=K,
        is_causal=True,
        block_size=block_size,
        groups=G,
        selected_blocks=S,
    )

    o_slc = torch.empty(B, C_SEQ_LEN, HQ, V, dtype=v.dtype, device=q.device)
    kernel(
        q.view(C_SEQ_LEN, HQ, D),
        k.view(C_SEQ_LEN, H, D),
        v.view(C_SEQ_LEN, H, D),
        o_slc.view(C_SEQ_LEN, HQ, V),
        block_indices.to(torch.int32).view(C_SEQ_LEN, H, S),
        block_counts.to(torch.int32).view(C_SEQ_LEN, H),
        offsets.to(torch.int32),
        token_indices.to(torch.int32),
    )
    return o_slc


@torch.compile
class ParallelNSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_indices, block_counts, block_size, window_size, scale, offsets):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the offsets of tokens in each sequence
        # for example, if the passed `offsets` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(offsets) if offsets is not None else None

        o_slc = parallel_nsa_fwd(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            window_size=window_size,
            scale=scale,
            offsets=offsets,
            token_indices=token_indices,
        )
        return o_slc.to(q.dtype)


def parallel_nsa(
    q: torch.Tensor,
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
    head_first: bool = False,
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
        block_counts (Union[torch.LongTensor, int]):
            Number of selected blocks for each token.
            If a tensor is provided, with shape `[B, T, H]` if `head_first=True` else `[B, T, H]`,
            each token can select the same number of blocks.
            If not provided, it will default to `S`, Default: `None`
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, "b h t d -> b t h d"), (q, k, v, block_indices))
        g_slc, g_swa = map(lambda x: rearrange(x, "b h t -> b t h"), (g_slc, g_swa))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, "b h t -> b t h")
    assert q.shape[2] % (k.shape[2] * 16) == 0, "Group size must be a multiple of 16 in NSA"

    if isinstance(block_counts, int):
        block_indices = block_indices[:, :, :, :block_counts]
        block_counts = None

    o_slc = ParallelNSAFunction.apply(q, k, v, block_indices, block_counts, block_size, window_size, scale, cu_seqlens)
    if window_size > 0:
        assert False, "Window size is not supported yet"
    else:
        o = o_slc * g_slc.unsqueeze(-1)
    if head_first:
        o = rearrange(o, "b t h d -> b h t d")
    return o


if __name__ == "__main__":
    N, C_SEQ_LEN, H, HQ, D, S, block_size, dtype = 2, 64, 1, 16, 64, 1, 32, torch.float16
    torch.manual_seed(42)
    # randomly split the sequence into N segments
    offsets = (
        torch.cat(
            [
                torch.tensor([0], dtype=torch.long),
                torch.arange(16, C_SEQ_LEN)[torch.randperm(C_SEQ_LEN - 1)[: N - 1]],
                torch.tensor([C_SEQ_LEN], dtype=torch.long),
            ],
            0,
        )
        .cuda()
        .sort()[0]
    )

    # seq-first required for inputs with variable lengths
    perm_q = torch.randperm(C_SEQ_LEN, device="cuda")
    perm_k = torch.randperm(C_SEQ_LEN, device="cuda")
    perm_v = torch.randperm(C_SEQ_LEN, device="cuda")
    q = (
        torch.linspace(0, 1, steps=C_SEQ_LEN, dtype=dtype, device="cuda")[perm_q]
        .view(1, C_SEQ_LEN, 1, 1)
        .expand(1, C_SEQ_LEN, HQ, D)
        .clone()
        .requires_grad_(True)
    )
    k = (
        torch.linspace(0, 1, steps=C_SEQ_LEN, dtype=dtype, device="cuda")[perm_k]
        .view(1, C_SEQ_LEN, 1, 1)
        .expand(1, C_SEQ_LEN, H, D)
        .clone()
        .requires_grad_(True)
    )
    v = (
        torch.linspace(0, 1, steps=C_SEQ_LEN, dtype=dtype, device="cuda")[perm_v]
        .view(1, C_SEQ_LEN, 1, 1)
        .expand(1, C_SEQ_LEN, H, D)
        .clone()
        .requires_grad_(True)
    )
    g_slc = torch.rand((1, C_SEQ_LEN, HQ), dtype=dtype, device="cuda").requires_grad_(True)
    g_swa = torch.rand((1, C_SEQ_LEN, HQ), dtype=dtype, device="cuda").requires_grad_(True)
    do = torch.randn((1, C_SEQ_LEN, HQ, D), dtype=dtype, device="cuda")

    token_indices = prepare_token_indices(offsets).tolist()
    block_indices = torch.full((1, C_SEQ_LEN, H, S), C_SEQ_LEN, dtype=torch.long, device="cuda")
    for i in range(C_SEQ_LEN):
        _, t = token_indices[i]
        for h in range(H):
            i_i = torch.randperm(max(1, tilelang.cdiv(t, block_size)))[:S]
            block_indices[0, i, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    block_counts = torch.randint(1, S + 1, (1, C_SEQ_LEN, H), device="cuda")

    ref = naive_nsa(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        cu_seqlens=offsets,
    )

    tri = parallel_nsa(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        cu_seqlens=offsets,
    )

    print("tri", tri)
    print("ref", ref)

    torch.testing.assert_close(ref, tri, atol=1e-2, rtol=1e-2)
