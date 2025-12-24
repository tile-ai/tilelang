# ruff: noqa
import torch
from typing import Optional, Union
from packaging.version import parse

import torch
import triton
import triton.language as tl

import fla

if parse(fla.__version__) < parse("0.2.1"):
    from fla.ops.common.utils import prepare_token_indices
else:
    from fla.ops.utils import prepare_token_indices
from fla.utils import autocast_custom_fwd, contiguous
from reference import naive_nsa
from einops import rearrange


@triton.heuristics(
    {
        "USE_OFFSETS": lambda args: args["offsets"] is not None,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
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

    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
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

    if WS > 0:
        p_o_swa = tl.make_block_ptr(o_swa + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
        p_lse_swa = lse_swa + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
        # [G, BV]
        b_o_swa = tl.zeros([G, BV], dtype=tl.float32)

        b_m_swa = tl.full([G], float("-inf"), dtype=tl.float32)
        b_acc_swa = tl.zeros([G], dtype=tl.float32)
        for i_s in range(max(0, i_t - WS + 1), i_t + 1, BS):
            p_k_swa = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v_swa = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [BK, BS]
            b_k_swa = tl.load(p_k_swa, boundary_check=(0, 1))
            # [BS, BV]
            b_v_swa = tl.load(p_v_swa, boundary_check=(0, 1))
            # [G, BS]
            b_s_swa = tl.dot(b_q, b_k_swa)
            b_s_swa = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s_swa, float("-inf"))

            # [G]
            b_m_swa, b_mp_swa = tl.maximum(b_m_swa, tl.max(b_s_swa, 1)), b_m_swa
            b_r_swa = tl.exp(b_mp_swa - b_m_swa)
            # [G, BS]
            b_p_swa = tl.exp(b_s_swa - b_m_swa[:, None])
            # [G]
            b_acc_swa = b_acc_swa * b_r_swa + tl.sum(b_p_swa, 1)
            # [G, BV]
            b_o_swa = b_o_swa * b_r_swa[:, None] + tl.dot(b_p_swa.to(b_q.dtype), b_v_swa)

            b_mp_swa = b_m_swa
        b_o_swa = b_o_swa / b_acc_swa[:, None]
        b_m_swa += tl.log(b_acc_swa)
        tl.store(p_o_swa, b_o_swa.to(p_o_swa.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_lse_swa, b_m_swa.to(p_lse_swa.dtype.element_ty))


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
    o_slc = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    o_swa = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device) if window_size > 0 else None
    lse_slc = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    lse_swa = torch.empty(B, T, HQ, dtype=torch.float, device=q.device) if window_size > 0 else None

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


if __name__ == "__main__":
    N, T, H, HQ, D, S, block_size, dtype = 2, 64, 1, 16, 64, 1, 32, torch.float16
    torch.manual_seed(42)
    # randomly split the sequence into N segments
    offsets = (
        torch.cat(
            [torch.tensor([0], dtype=torch.long), torch.arange(16, T)[torch.randperm(T - 1)[: N - 1]], torch.tensor([T], dtype=torch.long)],
            0,
        )
        .cuda()
        .sort()[0]
    )
    # offsets.shape is [N+1]
    # seq-first required for inputs with variable lengths
    perm_q = torch.randperm(T, device="cuda")
    perm_k = torch.randperm(T, device="cuda")
    perm_v = torch.randperm(T, device="cuda")
    q = torch.linspace(0, 1, steps=T, dtype=dtype, device="cuda")[perm_q].view(1, T, 1, 1).expand(1, T, HQ, D).clone().requires_grad_(True)
    k = torch.linspace(0, 1, steps=T, dtype=dtype, device="cuda")[perm_k].view(1, T, 1, 1).expand(1, T, H, D).clone().requires_grad_(True)
    v = torch.linspace(0, 1, steps=T, dtype=dtype, device="cuda")[perm_v].view(1, T, 1, 1).expand(1, T, H, D).clone().requires_grad_(True)
    g_slc = torch.rand((1, T, HQ), dtype=dtype, device="cuda").requires_grad_(True)
    g_swa = torch.rand((1, T, HQ), dtype=dtype, device="cuda").requires_grad_(True)
    do = torch.randn((1, T, HQ, D), dtype=dtype, device="cuda")

    token_indices = prepare_token_indices(offsets).tolist()
    block_indices = torch.full((1, T, H, S), T, dtype=torch.long, device="cuda")
    for i in range(T):
        _, t = token_indices[i]
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            block_indices[0, i, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    block_counts = torch.randint(1, S + 1, (1, T, H), device="cuda")

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
