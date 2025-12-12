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
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from reference import naive_nsa
from einops import rearrange


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

    # if USE_BLOCK_COUNTS:
    #     NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    # else:
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

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do_slc, do_swa):
        q, k, v, o_slc, lse_slc, o_swa, lse_swa = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o_slc=o_slc,
            o_swa=o_swa,
            lse_slc=lse_slc,
            lse_swa=lse_swa,
            do_slc=do_slc,
            do_swa=do_swa,
            block_indices=ctx.block_indices,
            block_counts=ctx.block_counts,
            block_size=ctx.block_size,
            window_size=ctx.window_size,
            scale=ctx.scale,
            offsets=ctx.offsets,
            token_indices=ctx.token_indices,
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None, None, None, None, None


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


@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["BS", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_bwd_kernel_dkv(
    q,
    k,
    v,
    lse_slc,
    lse_swa,
    delta_slc,
    delta_swa,
    do_slc,
    do_swa,
    dk,
    dv,
    block_mask,
    offsets,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    WS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * B * T * H + bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))

    # [BS, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    # [BS, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_m_slc = tl.load(block_mask + (bos + i) * H * M + i_h * M + i_s)
        if b_m_slc:
            p_q = tl.make_block_ptr(q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
            # [G, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)

            p_do_slc = tl.make_block_ptr(do_slc + (bos + i) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
            p_lse_slc = lse_slc + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta_slc = delta_slc + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            # [G, BV]
            b_do_slc = tl.load(p_do_slc, boundary_check=(0, 1))
            # [G]
            b_lse_slc = tl.load(p_lse_slc)
            b_delta_slc = tl.load(p_delta_slc)
            # [BS, G]
            b_s_slc = tl.dot(b_k, tl.trans(b_q))
            b_p_slc = tl.exp(b_s_slc - b_lse_slc[None, :])
            b_p_slc = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p_slc, 0)
            # [BS, G] @ [G, BV] -> [BS, BV]
            b_dv += tl.dot(b_p_slc.to(b_do_slc.dtype), b_do_slc)
            # [BS, BV] @ [BV, G] -> [BS, G]
            b_dp_slc = tl.dot(b_v, tl.trans(b_do_slc))
            # [BS, G]
            b_ds_slc = b_p_slc * (b_dp_slc - b_delta_slc[None, :])
            # [BS, G] @ [G, BK] -> [BS, BK]
            b_dk += tl.dot(b_ds_slc.to(b_q.dtype), b_q)

        if WS > 0:
            o_s = i_s * BS + tl.arange(0, BS)
            if max(i_s * BS, i - WS + 1) < min((i_s + 1) * BS, i + 1):
                p_q = tl.make_block_ptr(q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
                # [G, BK]
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_q = (b_q * scale).to(b_q.dtype)

                p_do_swa = tl.make_block_ptr(do_swa + (bos + i) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
                p_lse_swa = lse_swa + (bos + i) * HQ + i_h * G + tl.arange(0, G)
                p_delta_swa = delta_swa + (bos + i) * HQ + i_h * G + tl.arange(0, G)
                # [G, BV]
                b_do_swa = tl.load(p_do_swa, boundary_check=(0, 1))
                # [G]
                b_lse_swa = tl.load(p_lse_swa)
                b_delta_swa = tl.load(p_delta_swa)
                # [BS, G]
                b_s_swa = tl.dot(b_k, tl.trans(b_q))
                b_p_swa = tl.exp(b_s_swa - b_lse_swa[None, :])
                b_p_swa = tl.where((i >= o_s and (i - WS) < o_s)[:, None], b_p_swa, 0)
                # [BS, G] @ [G, BV] -> [BS, BV]
                b_dv += tl.dot(b_p_swa.to(b_do_swa.dtype), b_do_swa)
                # [BS, BV] @ [BV, G] -> [BS, G]
                b_dp_swa = tl.dot(b_v, tl.trans(b_do_swa))
                # [BS, G]
                b_ds_swa = b_p_swa * (b_dp_swa - b_delta_swa[None, :])
                # [BS, G] @ [G, BK] -> [BS, BK]
                b_dk += tl.dot(b_ds_swa.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor)})
@triton.jit
def parallel_nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(block_counts + i_b * T * H + i_t * H + i_h)
    else:
        b_m = b_i * BS <= i_t

    if b_i < NS and b_i >= 0:
        tl.store(block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i, b_m.to(block_mask.dtype.element_ty))


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
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_bwd_kernel_dq(
    q,
    k,
    v,
    lse_slc,
    delta_slc,
    do_slc,
    lse_swa,
    delta_swa,
    do_swa,
    dq,
    scale,
    block_indices,
    block_counts,
    offsets,
    token_indices,
    T,
    B: tl.constexpr,
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

    q += (bos + i_t) * HQ * K
    do_slc += (bos + i_t) * HQ * V
    lse_slc += (bos + i_t) * HQ
    delta_slc += (bos + i_t) * HQ
    if WS > 0:
        do_swa += (bos + i_t) * HQ * V
        lse_swa += (bos + i_t) * HQ
        delta_swa += (bos + i_t) * HQ
    dq += (i_v * B * T + bos + i_t) * HQ * K
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do_slc = tl.make_block_ptr(do_slc, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse_slc = lse_slc + i_h * G + tl.arange(0, G)
    p_delta_slc = delta_slc + i_h * G + tl.arange(0, G)

    # [G, BV]
    b_do_slc = tl.load(p_do_slc, boundary_check=(0, 1))
    # [G]
    b_lse_slc = tl.load(p_lse_slc)
    b_delta_slc = tl.load(p_delta_slc)

    # [G, BK]
    b_dq_slc = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k_slc = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v_slc = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
            # [BK, BS]
            b_k_slc = tl.load(p_k_slc, boundary_check=(0, 1))
            # [BV, BS]
            b_v_slc = tl.load(p_v_slc, boundary_check=(0, 1))

            # [G, BS]
            b_s_slc = tl.dot(b_q, b_k_slc)
            b_p_slc = tl.exp(b_s_slc - b_lse_slc[:, None])
            b_p_slc = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p_slc, 0)

            # [G, BV] @ [BV, BS] -> [G, BS]
            b_dp_slc = tl.dot(b_do_slc, b_v_slc)
            b_ds_slc = b_p_slc * (b_dp_slc.to(tl.float32) - b_delta_slc[:, None])
            # [G, BS] @ [BS, BK] -> [G, BK]
            b_dq_slc += tl.dot(b_ds_slc.to(b_k_slc.dtype), tl.trans(b_k_slc))
    b_dq_slc *= scale

    if WS > 0:
        p_do_swa = tl.make_block_ptr(do_swa, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
        p_lse_swa = lse_swa + i_h * G + tl.arange(0, G)
        p_delta_swa = delta_swa + i_h * G + tl.arange(0, G)

        # [G, BV]
        b_do_swa = tl.load(p_do_swa, boundary_check=(0, 1))
        # [G]
        b_lse_swa = tl.load(p_lse_swa)
        b_delta_swa = tl.load(p_delta_swa)

        # [G, BK]
        b_dq_swa = tl.zeros([G, BK], dtype=tl.float32)
        for i_s in range(max(0, i_t - WS + 1), i_t + 1, BS):
            p_k_swa = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v_swa = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
            # [BK, BS]
            b_k_swa = tl.load(p_k_swa, boundary_check=(0, 1))
            # [BV, BS]
            b_v_swa = tl.load(p_v_swa, boundary_check=(0, 1))

            # [G, BS]
            b_s_swa = tl.dot(b_q, b_k_swa)
            b_p_swa = tl.exp(b_s_swa - b_lse_swa[:, None])
            b_p_swa = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p_swa, 0)

            # [G, BV] @ [BV, BS] -> [G, BS]
            b_dp_swa = tl.dot(b_do_swa, b_v_swa)
            b_ds_swa = b_p_swa * (b_dp_swa.to(tl.float32) - b_delta_swa[:, None])
            # [G, BS] @ [BS, BK] -> [G, BK]
            b_dq_swa += tl.dot(b_ds_swa.to(b_k_swa.dtype), tl.trans(b_k_swa))
        b_dq_swa *= scale

    if WS == 0:
        tl.store(p_dq, b_dq_slc.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    else:
        tl.store(p_dq, (b_dq_slc + b_dq_swa).to(p_dq.dtype.element_ty), boundary_check=(0, 1))


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


@triton.jit
def parallel_nsa_bwd_kernel_preprocess(o, do, delta, B: tl.constexpr, V: tl.constexpr):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


def parallel_nsa_block_mask(
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    offsets: torch.LongTensor,
    block_size: int,
):
    B, T, H, S = block_indices.shape
    BS = block_size
    if offsets is not None:
        NS = triton.cdiv(prepare_lens(offsets).max().item(), BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    parallel_nsa_kernel_mask[(T, B, H * S)](
        block_indices=block_indices, block_counts=block_counts, block_mask=block_mask, T=T, H=H, S=S, BS=BS, NS=NS
    )
    return block_mask


def parallel_nsa_bwd_preprocess(o: torch.Tensor, do: torch.Tensor):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float32)
    parallel_nsa_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def parallel_nsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o_slc: torch.Tensor,
    lse_slc: torch.Tensor,
    do_slc: torch.Tensor,
    o_swa: torch.Tensor,
    lse_swa: torch.Tensor,
    do_swa: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    window_size: int = 0,
    scale: float = None,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    WS = window_size
    BK = triton.next_power_of_2(K)
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NV = triton.cdiv(V, BV)

    delta_slc = parallel_nsa_bwd_preprocess(o_slc, do_slc)
    delta_swa = parallel_nsa_bwd_preprocess(o_swa, do_swa) if window_size > 0 else None

    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    grid = (T, NV, B * H)
    parallel_nsa_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse_slc=lse_slc,
        delta_slc=delta_slc,
        do_slc=do_slc,
        lse_swa=lse_swa,
        delta_swa=delta_swa,
        do_swa=do_swa,
        dq=dq,
        block_indices=block_indices,
        block_counts=block_counts,
        offsets=offsets,
        token_indices=token_indices,
        scale=scale,
        T=T,
        B=B,
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
    dq = dq.sum(0)

    if offsets is not None:
        chunk_indices = prepare_chunk_indices(offsets, BS)
        NS = len(chunk_indices)
    else:
        chunk_indices = None
        NS = triton.cdiv(T, BS)

    # [B, T, H, M]
    block_mask = parallel_nsa_block_mask(block_indices, block_counts, offsets, block_size)
    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NS, B * H)
    parallel_nsa_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse_slc=lse_slc,
        lse_swa=lse_swa,
        delta_slc=delta_slc,
        delta_swa=delta_swa,
        do_slc=do_slc,
        do_swa=do_swa,
        dk=dk,
        dv=dv,
        block_mask=block_mask,
        offsets=offsets,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        M=block_mask.shape[-1],
        BS=BS,
        WS=WS,
        BK=BK,
        BV=BV,
    )
    dk = dk.sum(0)
    return dq, dk, dv


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

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do_slc, do_swa):
        q, k, v, o_slc, lse_slc, o_swa, lse_swa = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o_slc=o_slc,
            o_swa=o_swa,
            lse_slc=lse_slc,
            lse_swa=lse_swa,
            do_slc=do_slc,
            do_swa=do_swa,
            block_indices=ctx.block_indices,
            block_counts=ctx.block_counts,
            block_size=ctx.block_size,
            window_size=ctx.window_size,
            scale=ctx.scale,
            offsets=ctx.offsets,
            token_indices=ctx.token_indices,
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None, None, None, None, None


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
    B, T, H, HQ, D, S, block_size, dtype = 2, 64, 1, 16, 32, 1, 32, torch.float16
    torch.random.manual_seed(0)
    q = torch.randn((B, T, HQ, D), dtype=dtype, device="cuda").requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device="cuda").requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device="cuda").requires_grad_(True)
    g_slc = torch.ones((B, T, HQ), dtype=dtype, device="cuda").requires_grad_(True)
    g_swa = torch.ones((B, T, HQ), dtype=dtype, device="cuda").requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device="cuda")

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device="cuda")
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, (t // block_size)))[:S]
                block_indices[b, t, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    block_counts = torch.randint(1, S + 1, (B, T, H), device="cuda")

    ref = naive_nsa(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg_slc, g_slc.grad = g_slc.grad.clone(), None

    tri = parallel_nsa(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_size=block_size,
        block_counts=block_counts,
    )
    print("tri", tri)
    print("ref", ref)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg_slc, g_slc.grad = g_slc.grad.clone(), None

    # assert_close(" o", ref, tri, 0.004)
    torch.testing.assert_close(ref, tri, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dq, tri_dq, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dk, tri_dk, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dg_slc, tri_dg_slc, atol=1e-2, rtol=1e-2)
