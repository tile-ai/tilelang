# ruff: noqa

import torch
import time
import argparse
import tilelang
from tilelang import language as T
import tilelang.testing
from typing import Optional, Union
from einops import rearrange, repeat
import triton
import triton.language as tl
from fla.ops.utils import prepare_token_indices
from fla.utils import autocast_custom_fwd, contiguous


@triton.heuristics(
    {
        "USE_OFFSETS": lambda args: args["offsets"] is not None,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1]],
    key=["BS", "BK", "BV"],
)
@triton.jit
def parallel_nsa_fwd_kernel(
    q,
    k,
    v,
    o_slc,
    o_swa,
    lse_slc,
    lse_swa,
    scale,
    block_indices,
    block_counts,
    offsets,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    WS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    NS = S

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o_slc = tl.make_block_ptr(o_slc + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse_slc = lse_slc + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    # [G, BV]
    b_o_slc = tl.zeros([G, BV], dtype=tl.float32)

    b_m_slc = tl.full([G], float("-inf"), dtype=tl.float32)
    b_acc_slc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k_slc = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v_slc = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [BK, BS]
            b_k_slc = tl.load(p_k_slc, boundary_check=(0, 1))
            # [BS, BV]
            b_v_slc = tl.load(p_v_slc, boundary_check=(0, 1))
            # [G, BS]
            b_s_slc = tl.dot(b_q, b_k_slc)
            b_s_slc = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s_slc, float("-inf"))

            # [G]
            b_m_slc, b_mp_slc = tl.maximum(b_m_slc, tl.max(b_s_slc, 1)), b_m_slc
            b_r_slc = tl.exp(b_mp_slc - b_m_slc)
            # [G, BS]
            b_p_slc = tl.exp(b_s_slc - b_m_slc[:, None])
            # [G]
            b_acc_slc = b_acc_slc * b_r_slc + tl.sum(b_p_slc, 1)
            # [G, BV]
            b_o_slc = b_o_slc * b_r_slc[:, None] + tl.dot(b_p_slc.to(b_q.dtype), b_v_slc)

            b_mp_slc = b_m_slc
    b_o_slc = b_o_slc / b_acc_slc[:, None]
    b_m_slc += tl.log(b_acc_slc)

    tl.store(p_o_slc, b_o_slc.to(p_o_slc.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse_slc, b_m_slc.to(p_lse_slc.dtype.element_ty))


class ParallelNSAFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_size, scale, offsets):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the offsets of tokens in each sequence
        # for example, if the passed `offsets` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(offsets) if offsets is not None else None

        o, lse = parallel_nsa_fwd(q=q, k=k, v=v, block_indices=block_indices, block_size=block_size, scale=scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_indices = block_indices
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype)


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o_slc: torch.Tensor,
    o_swa: Optional[torch.Tensor],
    lse_slc: torch.Tensor,
    lse_swa: Optional[torch.Tensor],
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    window_size: int,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    WS = window_size
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    grid = (T, NV, B * H)

    parallel_nsa_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o_slc=o_slc,
        o_swa=o_swa,
        lse_slc=lse_slc,
        lse_swa=lse_swa,
        scale=scale,
        block_indices=block_indices,
        block_counts=block_counts,
        offsets=offsets,
        token_indices=token_indices,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        WS=WS,
        BK=BK,
        BV=BV,
    )
    return o_slc, lse_slc, o_swa, lse_swa


@torch.compile
class ParallelNSAFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_counts, block_size, window_size, scale, offsets):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the offsets of tokens in each sequence
        # for example, if the passed `offsets` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(offsets) if offsets is not None else None

        o_slc, lse_slc, o_swa, lse_swa = parallel_nsa_fwd(
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
        ctx.save_for_backward(q, k, v, o_slc, lse_slc, o_swa, lse_swa)
        ctx.block_indices = block_indices
        ctx.block_counts = block_counts
        ctx.offsets = offsets
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.window_size = window_size
        ctx.scale = scale
        return o_slc.to(q.dtype), o_swa.to(q.dtype) if o_swa is not None else o_swa


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

    o_slc, o_swa = ParallelNSAFunction.apply(q, k, v, block_indices, block_counts, block_size, window_size, scale, cu_seqlens)
    if window_size > 0:
        o = torch.addcmul(o_slc * g_slc.unsqueeze(-1), o_swa, g_swa.unsqueeze(-1))
    else:
        o = o_slc * g_slc.unsqueeze(-1)
    if head_first:
        o = rearrange(o, "b t h d -> b h t d")
    return o


def naive_nsa(
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
            Queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the maximum number of selected blocks for each query token, which is set to 16 in the paper.
        block_counts (Union[torch.LongTensor, int]):
            Number of selected blocks for each token.
            If a tensor is provided, with shape `[B, T, H]` if `head_first=True` else `[B, T, H]`,
            each token can select the same number of blocks.
            If not provided, it will default to `S`, Default: `None`.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, "b h t d -> b t h d"), (q, k, v, block_indices))
        g_slc, g_swa = map(lambda x: rearrange(x, "b h t -> b t h"), (g_slc, g_swa))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, "b h t -> b t h")

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    S = block_indices.shape[-1]
    k, v, block_indices = (repeat(x, "b t h d -> b t (h g) d", g=G) for x in (k, v, block_indices))
    if isinstance(block_counts, torch.Tensor):
        block_counts = repeat(block_counts, "b t h -> b t (h g)", g=G)
    c = torch.arange(S).repeat_interleave(BS).unsqueeze(1).expand(-1, q.shape[2]).to(q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))

    o_slc = torch.zeros_like(v)
    o_swa = torch.zeros_like(v) if window_size > 0 else None
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat([block_indices.new_tensor(range(0, B * T, T)), block_indices.new_tensor([B * T])])

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = q[i], k[i], v[i], g_slc[i], g_swa[i], block_indices[i]
            if isinstance(block_counts, torch.Tensor):
                s_b = block_counts[i]
            else:
                s_b = block_counts
        else:
            T = cu_seqlens[i + 1] - cu_seqlens[i]
            q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = map(
                lambda x: x[0][cu_seqlens[i] : cu_seqlens[i + 1]], (q, k, v, g_slc, g_swa, block_indices)
            )
            if isinstance(block_counts, torch.Tensor):
                s_b = block_counts[0][cu_seqlens[i] : cu_seqlens[i + 1]]
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
            k_i_slc, v_i_slc = map(lambda x: x.gather(0, i_i.clamp(0, T - 1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])), (k_b, v_b))
            # [S*BS, HQ]
            attn_slc = (
                torch.einsum("h d, n h d -> n h", q_i, k_i_slc)
                .masked_fill(torch.logical_or(i_i < 0, i_i > i_q) | (c >= s_i if block_counts is not None else False), float("-inf"))
                .softmax(0)
            )
            if not varlen:
                o_slc[i, i_q] = torch.einsum("n h, n h v -> h v", attn_slc, v_i_slc) * g_slc_i.unsqueeze(-1)
            else:
                o_slc[0][cu_seqlens[i] + i_q] = torch.einsum("n h, n h v -> h v", attn_slc, v_i_slc) * g_slc_i.unsqueeze(-1)
            if window_size > 0:
                k_i_swa, v_i_swa = map(lambda x: x[max(0, i_q - window_size + 1) : i_q + 1], (k_b, v_b))
                attn_swa = torch.einsum("h d, n h d -> n h", q_i, k_i_swa).softmax(0)
                if not varlen:
                    o_swa[i, i_q] = torch.einsum("n h, n h v -> h v", attn_swa, v_i_swa) * g_swa_i.unsqueeze(-1)
                else:
                    o_swa[0][cu_seqlens[i] + i_q] = torch.einsum("n h, n h v -> h v", attn_swa, v_i_swa) * g_swa_i.unsqueeze(-1)

    if head_first:
        o_slc = rearrange(o_slc, "b t h d -> b h t d")
        o_swa = rearrange(o_swa, "b t h d -> b h t d")

    return o_slc.to(dtype) + o_swa.to(dtype) if o_swa is not None else o_slc.to(dtype)


def get_configs():
    import itertools

    iter_params = dict(
        block_T=[128, 256, 512],
        num_stages=[0, 1, 2, 4, 5],
        threads=[32, 64, 128, 256, 512],
    )
    return [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]


@tilelang.autotune(
    configs=get_configs(),
)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def tilelang_sparse_attention(
    batch, heads, seq_len, dim, is_causal, scale=None, block_size=64, groups=1, selected_blocks=16, block_T=128, num_stages=2, threads=32
):
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
    block_indices_dtype = T.int32
    dtype = T.float16
    accum_dtype = T.float32
    block_S = block_size
    block_T = min(block_T, tilelang.math.next_power_of_2(dim))

    NK = tilelang.cdiv(dim, block_T)
    NV = tilelang.cdiv(dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T

    @T.prim_func
    def tilelang_sparse_attention(
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
            acc_s_cast = T.alloc_shared([G, BS], dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_max_prev = T.alloc_fragment([G], accum_dtype)
            scores_scale = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            logsum = T.alloc_fragment([G], accum_dtype)

            i_t, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i in T.Pipelined(NS, num_stages=num_stages):
                i_s = BlockIndices[i_b, i_t, i_h, i] * BS
                if i_s <= i_t and i_s >= 0:
                    # [BS, BK]
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

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
                    T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(G, BV):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, i_t, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV])

    return tilelang_sparse_attention


def generate_block_indices(batch, seq_len, heads, selected_blocks, block_size):
    """Generate random block indices for the benchmark."""
    block_indices = torch.full((batch, seq_len, heads, selected_blocks), seq_len, dtype=torch.long, device="cuda")

    for b in range(batch):
        for t in range(seq_len):
            for h in range(heads):
                i_i = torch.randperm(max(1, (t // block_size)))[:selected_blocks]
                block_indices[b, t, h, : len(i_i)] = i_i

    return block_indices.sort(-1)[0]


def benchmark_nsa(
    batch_size, seq_len, heads, head_query, dim, selected_blocks, block_size, dtype, scale, warmup=10, iterations=100, validate=False
):
    """Benchmark the TileLang Sparse Attention implementation."""

    # Set random seed for reproducibility
    tilelang.testing.set_random_seed(0)
    torch.random.manual_seed(0)

    # Compile the NSA kernel
    kernel = tilelang_sparse_attention(
        batch=batch_size,
        heads=head_query,
        seq_len=seq_len,
        dim=dim,
        is_causal=True,
        block_size=block_size,
        groups=head_query // heads,
        selected_blocks=selected_blocks,
        scale=scale,
    )

    profiler = kernel.get_profiler()

    profiler_latency = profiler.do_bench()
    print(f"Profiler latency: {profiler_latency} ms")

    # Create input tensors
    Q = torch.randn((batch_size, seq_len, head_query, dim), dtype=dtype, device="cuda")
    K = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype, device="cuda")
    V = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype, device="cuda")
    out = torch.empty((batch_size, seq_len, head_query, dim), dtype=dtype, device="cuda")

    # Generate block indices
    block_indices = generate_block_indices(batch_size, seq_len, heads, selected_blocks, block_size).to(torch.int32)

    # Warmup
    for _ in range(warmup):
        kernel(Q, K, V, block_indices, out)

    # Synchronize before timing
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        kernel(Q, K, V, block_indices, out)
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate metrics
    elapsed_time = end_time - start_time
    avg_time = elapsed_time / iterations * 1000  # ms

    # Calculate FLOPs (approximate for NSA)
    # Each token attends to selected_blocks * block_size tokens
    # Each attention calculation involves 2*dim FLOPs for QK
    # And another 2*dim FLOPs for attention * V
    flops_per_token = 4 * dim * selected_blocks * block_size
    total_flops = batch_size * seq_len * head_query * flops_per_token
    flops_per_sec = total_flops / (elapsed_time / iterations)
    tflops = flops_per_sec / 1e12

    # Validate result against reference if requested
    if validate:
        g_slc = torch.ones((batch_size, seq_len, head_query), dtype=dtype, device="cuda")
        g_swa = torch.ones((batch_size, seq_len, head_query), dtype=dtype, device="cuda")
        block_counts = torch.randint(1, selected_blocks + 1, (batch_size, seq_len, heads), device="cuda")

        ref = naive_nsa(
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

        is_valid = torch.allclose(ref, out, atol=1e-2, rtol=1e-2)
        if is_valid:
            print("Validation: PASSED")
        else:
            print("Validation: FAILED")
            print(f"Max difference: {(ref - out).abs().max().item()}")

    # Return benchmark results
    return {
        "avg_time_ms": avg_time,
        "tflops": tflops,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "heads": heads,
        "head_query": head_query,
        "dim": dim,
        "selected_blocks": selected_blocks,
        "block_size": block_size,
    }


def benchmark_triton_nsa(
    batch_size, seq_len, heads, head_query, dim, selected_blocks, block_size, dtype, scale, warmup=10, iterations=100, validate=False
):
    """Benchmark the Triton-based TileLang Sparse Attention implementation."""

    # Set random seed for reproducibility
    tilelang.testing.set_random_seed(0)
    torch.random.manual_seed(0)

    # Create input tensors
    Q = torch.randn((batch_size, seq_len, head_query, dim), dtype=dtype, device="cuda")
    K = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype, device="cuda")
    V = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype, device="cuda")
    g_slc = torch.ones((batch_size, seq_len, head_query), dtype=dtype, device="cuda")
    g_swa = torch.ones((batch_size, seq_len, head_query), dtype=dtype, device="cuda")

    # Generate block indices
    block_indices = generate_block_indices(batch_size, seq_len, heads, selected_blocks, block_size)
    block_counts = torch.randint(1, selected_blocks + 1, (batch_size, seq_len, heads), device="cuda")
    o_slc = torch.empty((batch_size, seq_len, head_query, dim), dtype=dtype, device="cuda")
    lse_slc = torch.empty((batch_size, seq_len, head_query), dtype=torch.float, device="cuda")

    # Warmup
    for _ in range(warmup):
        out = parallel_nsa_fwd(
            q=Q,
            k=K,
            v=V,
            o_slc=o_slc,
            o_swa=None,
            lse_slc=lse_slc,
            lse_swa=None,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            window_size=0,
            scale=scale,
        )

    # Synchronize before timing
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        out = parallel_nsa_fwd(
            q=Q,
            k=K,
            v=V,
            o_slc=o_slc,
            o_swa=None,
            lse_slc=lse_slc,
            lse_swa=None,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            window_size=0,
            scale=scale,
        )
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate metrics
    elapsed_time = end_time - start_time
    avg_time = elapsed_time / iterations * 1000  # ms

    # Calculate FLOPs (approximate for NSA)
    flops_per_token = 4 * dim * selected_blocks * block_size
    total_flops = batch_size * seq_len * head_query * flops_per_token
    flops_per_sec = total_flops / (elapsed_time / iterations)
    tflops = flops_per_sec / 1e12

    # Validate result against reference if requested
    if validate:
        ref = naive_nsa(
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

        is_valid = torch.allclose(ref, out, atol=1e-2, rtol=1e-2)
        if is_valid:
            print("Validation: PASSED")
        else:
            print("Validation: FAILED")
            print(f"Max difference: {(ref - out).abs().max().item()}")

    # Return benchmark results
    return {
        "avg_time_ms": avg_time,
        "tflops": tflops,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "heads": heads,
        "head_query": head_query,
        "dim": dim,
        "selected_blocks": selected_blocks,
        "block_size": block_size,
    }


def run_benchmark_suite(impl="all"):
    """Run a suite of benchmarks with different configurations."""

    # Define configurations to benchmark
    configs = [
        # Small model config - Note: head_query must be a multiple of heads*16 for Triton
        {"batch_size": 2, "seq_len": 1024, "heads": 8, "head_query": 8 * 16, "dim": 64, "selected_blocks": 8, "block_size": 32},
        # Medium model config
        {"batch_size": 2, "seq_len": 2048, "heads": 16, "head_query": 16 * 16, "dim": 64, "selected_blocks": 16, "block_size": 64},
        # Large model config
        {"batch_size": 1, "seq_len": 4096, "heads": 32, "head_query": 32 * 16, "dim": 128, "selected_blocks": 32, "block_size": 128},
    ]

    results = []
    for config in configs:
        print(f"Running benchmark with config: {config}")

        if impl in ["all", "tilelang"]:
            print("Benchmarking TileLang implementation:")
            result = benchmark_nsa(
                batch_size=config["batch_size"],
                seq_len=config["seq_len"],
                heads=config["heads"],
                head_query=config["head_query"],
                dim=config["dim"],
                selected_blocks=config["selected_blocks"],
                block_size=config["block_size"],
                dtype=torch.float16,
                scale=0.1,
                validate=False,
            )
            results.append({"impl": "tilelang", **result})
            print(f"Average time: {result['avg_time_ms']:.2f} ms")
            print(f"Performance: {result['tflops']:.2f} TFLOPs")

        if impl in ["all", "triton"]:
            print("Benchmarking Triton implementation:")
            result = benchmark_triton_nsa(
                batch_size=config["batch_size"],
                seq_len=config["seq_len"],
                heads=config["heads"],
                head_query=config["head_query"],
                dim=config["dim"],
                selected_blocks=config["selected_blocks"],
                block_size=config["block_size"],
                dtype=torch.float16,
                scale=0.1,
                validate=False,
            )
            results.append({"impl": "triton", **result})
            print(f"Average time: {result['avg_time_ms']:.2f} ms")
            print(f"Performance: {result['tflops']:.2f} TFLOPs")

        if impl in ["all"]:
            # Print comparison if both implementations were run
            tilelang_result = next(
                r
                for r in results
                if r["impl"] == "tilelang" and r["batch_size"] == config["batch_size"] and r["seq_len"] == config["seq_len"]
            )
            triton_result = next(
                r
                for r in results
                if r["impl"] == "triton" and r["batch_size"] == config["batch_size"] and r["seq_len"] == config["seq_len"]
            )
            speedup = tilelang_result["avg_time_ms"] / triton_result["avg_time_ms"]
            print(f"Speedup (Triton vs TileLang): {speedup:.2f}x")

        print("-" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TileLang Sparse Attention")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--head_query", type=int, default=16, help="Number of query heads")
    parser.add_argument("--dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--selected_blocks", type=int, default=16, help="Number of selected blocks")
    parser.add_argument("--block_size", type=int, default=32, help="Block size")
    parser.add_argument("--dtype", type=str, default=T.float16, help="Data type (float16 or float32)")
    parser.add_argument("--scale", type=float, default=0.1, help="Attention scale factor")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--validate", action="store_true", help="Validate against reference")
    parser.add_argument("--suite", action="store_true", help="Run benchmark suite")
    parser.add_argument(
        "--impl",
        type=str,
        default="all",
        choices=["tilelang", "triton", "all"],
        help="Implementation to benchmark (tilelang, triton, or all)",
    )

    args = parser.parse_args()

    # For Triton impl, ensure head_query is a multiple of heads*16
    if args.impl in ["triton", "all"] and args.head_query % (args.heads * 16) != 0:
        # Adjust head_query to nearest valid value
        args.head_query = ((args.head_query // (args.heads * 16)) + 1) * (args.heads * 16)
        print(f"Adjusted head_query to {args.head_query} to be compatible with Triton implementation")

    if args.suite:
        run_benchmark_suite(impl=args.impl)
    else:
        dtype = torch.float16 if args.dtype == T.float16 else torch.float32

        if args.impl in ["tilelang", "all"]:
            print("Benchmarking TileLang implementation:")
            result = benchmark_nsa(
                batch_size=args.batch,
                seq_len=args.seq_len,
                heads=args.heads,
                head_query=args.head_query,
                dim=args.dim,
                selected_blocks=args.selected_blocks,
                block_size=args.block_size,
                dtype=dtype,
                scale=args.scale,
                warmup=args.warmup,
                iterations=args.iterations,
                validate=args.validate,
            )
            print("\nBenchmark Results (TileLang):")
            print(
                f"Configuration: batch={args.batch}, seq_len={args.seq_len}, heads={args.heads}, "
                + f"head_query={args.head_query}, dim={args.dim}, blocks={args.selected_blocks}, "
                + f"block_size={args.block_size}"
            )
            print(f"Average time: {result['avg_time_ms']:.2f} ms")
            print(f"Performance: {result['tflops']:.2f} TFLOPs")

        if args.impl in ["triton", "all"]:
            print("Benchmarking Triton implementation:")
            result = benchmark_triton_nsa(
                batch_size=args.batch,
                seq_len=args.seq_len,
                heads=args.heads,
                head_query=args.head_query,
                dim=args.dim,
                selected_blocks=args.selected_blocks,
                block_size=args.block_size,
                dtype=dtype,
                scale=args.scale,
                warmup=args.warmup,
                iterations=args.iterations,
                validate=args.validate,
            )
            print("\nBenchmark Results (Triton):")
            print(
                f"Configuration: batch={args.batch}, seq_len={args.seq_len}, heads={args.heads}, "
                + f"head_query={args.head_query}, dim={args.dim}, blocks={args.selected_blocks}, "
                + f"block_size={args.block_size}"
            )
            print(f"Average time: {result['avg_time_ms']:.2f} ms")
            print(f"Performance: {result['tflops']:.2f} TFLOPs")
