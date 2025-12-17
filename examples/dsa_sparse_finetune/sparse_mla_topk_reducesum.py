# ruff: noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import tilelang
from tilelang import language as T
from einops import repeat, rearrange, einsum
from index import prepare_token_indices
from utils import get_abs_err, get_err_ratio

BF16 = T.bfloat16
FP32 = T.float32
INT32 = T.int32

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=pass_configs)
def tl_sparse_mla_topk_reducesum_impl(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_I=32,
    num_stages=2,
    threads=128,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

    batch_plus_one = T.symbolic("batch_plus_one")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = heads // kv_group
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    q_shape = [seq_len, heads, dim + tail_dim]
    kv_shape = [seq_len_kv, kv_group, dim + tail_dim]
    indices_shape = [seq_len, kv_group, topk]
    lse_shape = [seq_len, heads]
    reducesum_shape = [seq_len, kv_group, REPLICATE_H, topk]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]

    @T.prim_func
    def tl_sparse_mla_topk_reducesum_kernel(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Offsets: T.Tensor(offsets_shape, indices_dtype),  # type: ignore
        TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # type: ignore
        ReduceSum: T.Tensor(reducesum_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, kv_group, threads=threads) as (
            bx,
            by,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            reducesum = T.alloc_fragment([BI], accum_dtype)
            lse = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(lse, 0)

            b_s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
            bos, eos = Offsets[b_i], Offsets[b_i + 1]
            r_i = bx % REPLICATE_H
            g_i = by
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[bos + s_i, H0:H1, :D], Q_shared)
            T.copy(Q[bos + s_i, H0:H1, D:], Q_tail_shared)
            T.copy(Lse[bos + s_i, H0:H1], lse)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = (Indices[bos + s_i, g_i, i_i * BI + bi_i] <= max_kv_i) & (Indices[bos + s_i, g_i, i_i * BI + bi_i] != -1)

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp(acc_s[h_i, bi_i] * sm_scale - lse[h_i])
                T.reduce_sum(acc_s, reducesum, dim=0)
                T.copy(reducesum, ReduceSum[bos + s_i, g_i, r_i, i_i * BI : i_i * BI + BI])

    return tl_sparse_mla_topk_reducesum_kernel


def sparse_mla_topk_reducesum_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    lse: torch.Tensor,
    offsets: torch.Tensor,
    dim_v: int,
):
    assert kv.shape[-2] == 1
    seq_len, heads, dim_plus_tail_dim, topk = *q.shape, topk_indices.shape[-1]
    REPLICATE_H = max(heads // 64, 1)
    tail_dim = dim_plus_tail_dim - dim_v
    token_indices = prepare_token_indices(offsets)

    reducesum = torch.zeros([seq_len, 1, REPLICATE_H, topk], dtype=torch.float32, device=q.device)
    kernel = tl_sparse_mla_topk_reducesum_impl(heads=heads, dim=dim_v, tail_dim=tail_dim, topk=topk)
    kernel(q, kv, topk_indices, lse, offsets, token_indices, reducesum)
    reducesum = reducesum.sum(dim=-2)  # [batch, seq_len, 1, RH, topk] -> [batch, seq_len, 1, topk]
    attn_score = reducesum / reducesum.sum(dim=-1, keepdim=True)

    return attn_score


def ref_mla_topk_softmax(Q: torch.Tensor, K: torch.Tensor, TopkIndices: torch.Tensor, offsets: torch.Tensor):
    # q: [batch, seq_len, heads, dim]
    # k: [batch, seq_len, dim]
    sm_scale = Q.shape[-1] ** -0.5
    all_lse = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        q = Q[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        topk_indices = TopkIndices[offsets[i] : offsets[i + 1]]
        seq_len = q.shape[0]
        mask = (torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]).unsqueeze(-2).cuda()
        logits = einsum(q, k, "s1 h d, s2 d -> s1 h s2") * sm_scale
        logits = torch.where(mask, logits, float("-inf"))
        score = F.softmax(logits, dim=-1, dtype=torch.float32)
        score_sum = score.sum(dim=-2)
        topk_score = torch.gather(score_sum, dim=-1, index=topk_indices.to(torch.int64))
        topk_score = topk_score / topk_score.sum(dim=-1, keepdim=True)
        max_logits = logits.amax(dim=-1).to(torch.float32)
        lse = torch.log((logits - max_logits.unsqueeze(-1).to(torch.float32)).exp().sum(dim=-1)) + max_logits
        all_lse.append(lse)
        all_topk_score.append(topk_score)
    lse = torch.cat(all_lse, dim=0)
    topk_score = torch.cat(all_topk_score, dim=0)
    return lse, topk_score


def test_kernel(
    B=1,
    S=2048,
    H=16,
    D=512,
    tail_D=64,
    topk=128,
):
    torch.manual_seed(42)

    q = torch.randn((S, H, D + tail_D)).cuda().bfloat16()
    kv = torch.randn((S, D + tail_D)).cuda().bfloat16()
    offsets = torch.tensor([0, 1023, S], dtype=torch.int32).cuda()

    topk_indices = repeat(torch.arange(topk, dtype=torch.int32).cuda(), "k -> s k", s=S).contiguous()

    lse, ref_attn_score = ref_mla_topk_softmax(q, kv, topk_indices, offsets)

    kv = kv.unsqueeze(-2)
    topk_indices = topk_indices.unsqueeze(-2)

    attn_score = sparse_mla_topk_reducesum_interface(q, kv, topk_indices, lse, offsets, dim_v=D).squeeze(-2)
    print(f"attn_score err: {get_abs_err(attn_score, ref_attn_score):.6f} ratio: {get_err_ratio(attn_score, ref_attn_score):.6f}")


if __name__ == "__main__":
    test_kernel()
