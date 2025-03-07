# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ruff: noqa
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def naive_nsa(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              block_indices: torch.LongTensor,
              block_counts: torch.LongTensor,
              block_size: int = 64,
              scale: Optional[float] = None,
              head_first: bool = False,
              cu_seqlens: Optional[torch.LongTensor] = None) -> torch.Tensor:
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
        scale = k.shape[-1]**-0.5
    if cu_seqlens is not None:
        if head_first:
            raise RuntimeError(
                "Sequences with variable lengths are not supported for head-first mode")
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'),
                                     (q, k, v, block_indices))
        block_counts = rearrange(block_counts, 'b h t -> b t h')

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    S = block_indices.shape[-1]
    k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
    block_counts = repeat(block_counts, 'b t h -> b t (h g)', g=G)
    c = torch.arange(S).repeat_interleave(BS).unsqueeze(1).expand(-1, q.shape[2]).to(q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))

    o = torch.zeros_like(v)
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat(
            [block_indices.new_tensor(range(0, B * T, T)),
             block_indices.new_tensor([B * T])])

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, i_b, s_b = q[i], k[i], v[i], block_indices[i], block_counts[i]
        else:
            T = cu_seqlens[i + 1] - cu_seqlens[i]
            q_b, k_b, v_b, i_b, s_b = map(lambda x: x[0][cu_seqlens[i]:cu_seqlens[i + 1]],
                                          (q, k, v, block_indices, block_counts))

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, S*BS, HQ]
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [S*BS, HQ]
            i_i = i_b[i_q]
            # [1, HQ]
            s_i = s_b[i_q]
            # [S*BS, HQ, -1]
            k_i, v_i = map(
                lambda x: x.gather(
                    0,
                    i_i.clamp(0, T - 1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])), (k_b, v_b))
            # [S*BS, HQ]
            attn = torch.einsum('h d, n h d -> n h', q_i, k_i).masked_fill((i_i > i_q) | (c >= s_i),
                                                                           float('-inf')).softmax(0)
            if not varlen:
                o[i, i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)
            else:
                o[0][cu_seqlens[i] + i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)

    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')
    return o.to(dtype)


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
    scale = k.shape[-1]**-0.5

    dtype = q.dtype
    HQ = q.shape[2]
    H = k.shape[2]
    D = k.shape[-1]
    G = HQ // H
    BS = block_size
    S = block_indices.shape[-1]
    SELECTED_BLOCKS_SIZE = S * BS
    k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
    block_counts = repeat(block_counts, 'b t h -> b t (h g)', g=G)
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
            attn = torch.einsum('h d, n h d -> n h', q_i, k_i)
            attn = attn.masked_fill((i_i > i_q) | (c >= s_i), float('-inf'))
            attn = torch.softmax(attn, dim=0)
            o[i, i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)

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
    scale = k.shape[-1]**-0.5

    dtype = q.dtype
    HQ = q.shape[2]
    H = k.shape[2]
    D = k.shape[-1]
    G = HQ // H
    BS = block_size
    S = block_indices.shape[-1]
    SELECTED_BLOCKS_SIZE = S * BS
    k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
    block_counts = repeat(block_counts, 'b t h -> b t (h g)', g=G)
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
        attn = torch.einsum('h d, n h d -> n h', q_i, k_i)
        attn = attn.masked_fill((c >= s_i), float('-inf'))
        attn = torch.softmax(attn, dim=0)
        o[i, 0] = torch.einsum('n h, n h v -> h v', attn, v_i)

    return o.to(dtype)
