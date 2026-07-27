# ruff: noqa
from typing import Optional

import torch
from typing import Union
from einops import rearrange, repeat


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


def naive_nsa_simple(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor,
    block_size: int = 64,
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
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the maximum number of selected blocks for each query token, which is set to 16 in the paper.
        block_counts (torch.LongTensor):
            Block counts of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        block_size (int):
            Selected block size. Default: 64.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    scale = k.shape[-1] ** -0.5

    dtype = q.dtype
    HQ = q.shape[2]
    H = k.shape[2]
    D = k.shape[-1]
    G = HQ // H
    BS = block_size
    S = block_indices.shape[-1]
    SELECTED_BLOCKS_SIZE = S * BS
    k, v, block_indices = (repeat(x, "b t h d -> b t (h g) d", g=G) for x in (k, v, block_indices))
    block_counts = repeat(block_counts, "b t h -> b t (h g)", g=G)
    c = torch.arange(S).repeat_interleave(BS).unsqueeze(1).expand(-1, q.shape[2]).to(q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))
    o = torch.zeros_like(v)
    B, T = q.shape[:2]

    for i in range(B):
        q_b, k_b, v_b, i_b, s_b = q[i], k[i], v[i], block_indices[i], block_counts[i]
        # [T, HQ, S, BS] -> [T, HQ, S*BS]
        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, HQ, S*BS] -> [T, S*BS, HQ]
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [S*BS, HQ] -> represents selected blocks for each query token
            i_i = i_b[i_q]
            # [HQ] -> represents the number of selected blocks for each query token
            s_i = s_b[i_q]

            k_i = torch.zeros((S * BS, HQ, D), device=k_b.device, dtype=k_b.dtype)
            v_i = torch.zeros((S * BS, HQ, D), device=v_b.device, dtype=v_b.dtype)

            for h in range(HQ):
                for t in range(SELECTED_BLOCKS_SIZE):
                    selected_block_index = i_i[t, h]
                    k_i[t, h] = k_b[selected_block_index, h, :]
                    v_i[t, h] = v_b[selected_block_index, h, :]

            # [S*BS, HQ]
            attn = torch.einsum("h d, n h d -> n h", q_i, k_i)
            attn = attn.masked_fill((i_i > i_q) | (c >= s_i), float("-inf"))
            attn = torch.softmax(attn, dim=0)
            o[i, i_q] = torch.einsum("n h, n h v -> h v", attn, v_i)

    return o.to(dtype)


def naive_nsa_simple_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor,
    block_size: int = 64,
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, 1, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, 1, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the maximum number of selected blocks for each query token, which is set to 16 in the paper.
        block_counts (torch.LongTensor):
            Block counts of shape `[B, 1, H]` if `head_first=False` else `[B, H, T]`.
        block_size (int):
            Selected block size. Default: 64.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, 1, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    scale = k.shape[-1] ** -0.5

    dtype = q.dtype
    HQ = q.shape[2]
    H = k.shape[2]
    D = k.shape[-1]
    G = HQ // H
    BS = block_size
    S = block_indices.shape[-1]
    SELECTED_BLOCKS_SIZE = S * BS
    k, v, block_indices = (repeat(x, "b t h d -> b t (h g) d", g=G) for x in (k, v, block_indices))
    block_counts = repeat(block_counts, "b t h -> b t (h g)", g=G)
    c = torch.arange(S).repeat_interleave(BS).unsqueeze(1).expand(-1, q.shape[2]).to(q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))
    o = torch.zeros_like(q)
    B, T = q.shape[:2]

    for i in range(B):
        q_b, k_b, v_b, i_b, s_b = q[i], k[i], v[i], block_indices[i], block_counts[i]
        # [T, HQ, S, BS] -> [T, HQ, S*BS]
        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, HQ, S*BS] -> [T, S*BS, HQ]
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)

        # [HQ, D]
        q_i = q_b[0] * scale
        # [S*BS, HQ] -> represents selected blocks for each query token
        i_i = i_b[0]
        # [HQ] -> represents the number of selected blocks for each query token
        s_i = s_b[0]

        k_i = torch.zeros((S * BS, HQ, D), device=k_b.device, dtype=k_b.dtype)
        v_i = torch.zeros((S * BS, HQ, D), device=v_b.device, dtype=v_b.dtype)

        for h in range(HQ):
            for t in range(SELECTED_BLOCKS_SIZE):
                selected_block_index = i_i[t, h]
                k_i[t, h] = k_b[selected_block_index, h, :]
                v_i[t, h] = v_b[selected_block_index, h, :]

        # [S*BS, HQ]
        attn = torch.einsum("h d, n h d -> n h", q_i, k_i)
        attn = attn.masked_fill((c >= s_i), float("-inf"))
        attn = torch.softmax(attn, dim=0)
        o[i, 0] = torch.einsum("n h, n h v -> h v", attn, v_i)

    return o.to(dtype)
