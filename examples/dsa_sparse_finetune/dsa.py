from typing import Optional
import torch
import torch.nn.functional as F
from indexer_topk_reducesum import indexer_topk_reducesum_interface
from indexer_bwd import indexer_bwd_interface
from sparse_mla_fwd import sparse_mla_fwd_interface
from sparse_mla_bwd import sparse_mla_bwd
from sparse_mla_topk_reducesum import sparse_mla_topk_reducesum_interface
from einops import einsum, repeat
from utils import get_abs_err, get_err_ratio


class RegsiterLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        ctx.save_for_backward(loss)
        return x

    @staticmethod
    def backward(ctx, grad):
        loss = ctx.saved_tensors
        return grad, torch.ones(1, dtype=loss[0].dtype, device=loss[0].device)


register_loss = RegsiterLossFunction.apply


def ref_deepseek_sparse_attention_innner(
    q: torch.Tensor,
    kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    dim_v: int,
    sm_scale: Optional[float] = None,
    index_sm_scale: Optional[float] = None,
):
    dtype = q.dtype
    q, kv, index_q, index_k, weights = map(lambda x: x.to(torch.float32), (q, kv, index_q, index_k, weights))

    index_sm_scale = index_q.shape[-1] ** -0.5
    b, s = index_q.shape[:2]

    # tl_topk_indices = tl_topk_indices.to(torch.int64)
    # tl_topk_indices[tl_topk_indices == -1] = s

    casual_mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
    index_logits = einsum(index_q, index_k, "b s1 h k, b s2 k -> b s1 h s2")
    index_logits = F.relu(index_logits)
    index_logits = (index_logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * index_sm_scale
    index_logits = torch.where(casual_mask, index_logits, float("-inf"))
    topk_indices = torch.topk(index_logits, k=topk, dim=-1).indices
    topk_logits = torch.gather(F.pad(index_logits, (0, 1), value=float("-inf")), dim=-1, index=topk_indices)
    topk_score = F.log_softmax(topk_logits, dim=-1, dtype=torch.float32)
    index_topk_score = topk_score

    if sm_scale is None:
        sm_scale = kv.shape[-1] ** -0.5

    h = q.shape[-2]
    index_mask = torch.zeros((b, s, s + 1), dtype=torch.bool, device="cuda").scatter_(
        dim=-1, index=topk_indices, src=torch.ones_like(topk_indices, dtype=torch.bool)
    )[:, :, :-1]
    mask = repeat(casual_mask & index_mask, "b s1 s2 -> b s1 h s2", h=h)
    k, v = kv, kv[..., :dim_v]
    logits = einsum(q, k, "b s1 h d, b s2 d -> b s1 h s2") * sm_scale
    logits = torch.where(mask, logits, float("-inf"))
    attn_score = F.softmax(logits, dim=-1, dtype=torch.float32)
    o = einsum(attn_score, v, "b s1 h s2, b s2 d -> b s1 h d")

    attn_score = attn_score.sum(dim=-2)  # [b, s1, s2]
    attn_topk_score = torch.gather(F.pad(attn_score, (0, 1)), dim=-1, index=topk_indices)
    attn_topk_score = attn_topk_score / attn_topk_score.sum(dim=-1, keepdim=True)

    loss = F.kl_div(index_topk_score.clip(-100, 0), attn_topk_score.detach().log().clip(-100, 0), log_target=True, reduction="sum")
    o = register_loss(o, loss)

    return o.to(dtype), topk_indices


def ref_deepseek_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    dim_v: int,
    sm_scale: Optional[float] = None,
    index_sm_scale: Optional[float] = None,
):
    all_o, all_topk_indices = [], []
    for i in range(offsets.shape[0] - 1):
        o, topk_indices = ref_deepseek_sparse_attention_innner(
            q[None, offsets[i] : offsets[i + 1]],
            kv[None, offsets[i] : offsets[i + 1]],
            index_q[None, offsets[i] : offsets[i + 1]],
            index_k[None, offsets[i] : offsets[i + 1]],
            weights[None, offsets[i] : offsets[i + 1]],
            topk,
            dim_v,
            sm_scale,
            index_sm_scale,
        )
        all_o.append(o.squeeze(0))
        all_topk_indices.append(topk_indices.squeeze(0))
    o = torch.cat(all_o, dim=0)
    topk_indices = torch.cat(all_topk_indices, dim=0)
    return o, topk_indices


class DSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
        dim_v: int,
        sm_scale: Optional[float] = None,
    ):
        # topk_indices, index_score = ref_index_score(index_q, weights, index_k, topk)
        topk_indices, index_score = indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)
        o, lse = sparse_mla_fwd_interface(q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, sm_scale=sm_scale, d_v=dim_v)
        ctx.save_for_backward(q, kv, index_q, index_k, weights, topk_indices, index_score, o, lse, offsets)
        ctx.topk = topk
        ctx.dim_v = dim_v
        ctx.sm_scale = sm_scale
        return o, topk_indices

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor,
        _1: torch.Tensor,
    ):
        q, kv, index_q, index_k, weights, topk_indices, index_score, o, lse, offsets = ctx.saved_tensors
        attn_score = sparse_mla_topk_reducesum_interface(
            q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), lse, offsets, dim_v=ctx.dim_v
        ).squeeze(-2)
        dq, dkv = sparse_mla_bwd(q, kv.unsqueeze(-2), o, do, topk_indices.unsqueeze(-2), lse, offsets, sm_scale=ctx.sm_scale)
        dindex_q, dweights, dindex_k = indexer_bwd_interface(index_q, weights, index_k, attn_score, index_score, topk_indices, offsets)
        return dq, dkv.squeeze(-2), dindex_q, dindex_k, dweights, None, None, None, None


def deepseek_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    dim_v: int,
    sm_scale: Optional[float] = None,
):
    return DSAFunction.apply(q, kv, index_q, index_k, weights, offsets, topk, dim_v, sm_scale)


def test_kernel(
    B=1,
    S=2048,
    H=16,
    D=512,
    tail_D=64,
    index_D=128,
    topk=64,
):
    torch.manual_seed(42)
    q = torch.randn((S, H, D + tail_D)).cuda().bfloat16().requires_grad_()
    kv = torch.randn((S, D + tail_D)).cuda().bfloat16().requires_grad_()
    index_q = torch.randn((S, H, index_D)).cuda().bfloat16().requires_grad_()
    weights = torch.randn((S, H)).cuda().bfloat16().requires_grad_()
    index_k = torch.randn((S, index_D)).cuda().bfloat16().requires_grad_()
    do = torch.randn((S, H, D)).cuda().bfloat16().requires_grad_()
    offsets = torch.tensor([0, S // 2, S], dtype=torch.int32).cuda()

    o, topk_indices = deepseek_sparse_attention(q, kv, index_q, index_k, weights, offsets, topk, D)
    o.backward(do)
    q_grad, q.grad = q.grad, None
    kv_grad, kv.grad = kv.grad, None
    index_q_grad, index_q.grad = index_q.grad, None
    index_k_grad, index_k.grad = index_k.grad, None
    weights_grad, weights.grad = weights.grad, None

    ref_o, ref_topk_indices = ref_deepseek_sparse_attention(q, kv, index_q, index_k, weights, offsets, topk, D)
    ref_o.backward(do)
    ref_q_grad, q.grad = q.grad, None
    ref_kv_grad, kv.grad = kv.grad, None
    ref_index_q_grad, index_q.grad = index_q.grad, None
    ref_index_k_grad, index_k.grad = index_k.grad, None
    ref_weights_grad, weights.grad = weights.grad, None

    print(f"o err: {get_abs_err(o, ref_o):.6f} ratio: {get_err_ratio(o, ref_o):.6f}")
    print(f"q.grad err: {get_abs_err(q_grad, ref_q_grad):.6f} ratio: {get_err_ratio(q_grad, ref_q_grad):.6f}")
    print(f"kv.grad err: {get_abs_err(kv_grad, ref_kv_grad):.6f} ratio: {get_err_ratio(kv_grad, ref_kv_grad):.6f}")
    print(
        f"index_q.grad err: {get_abs_err(index_q_grad[:, :64, :], ref_index_q_grad[:, :64, :]):.6f} ratio: {get_err_ratio(index_q_grad[:, :64, :], ref_index_q_grad[:, :64, :]):.6f}"
    )
    print(f"index_k.grad err: {get_abs_err(index_k_grad, ref_index_k_grad):.6f} ratio: {get_err_ratio(index_k_grad, ref_index_k_grad):.6f}")
    print(f"weights.grad err: {get_abs_err(weights_grad, ref_weights_grad):.6f} ratio: {get_err_ratio(weights_grad, ref_weights_grad):.6f}")

    intersections = []
    for j in range(S):
        ref_np = ref_topk_indices[j].cpu().to(torch.int32).numpy()
        trt_np = topk_indices[j].cpu().to(torch.int32).numpy()

        mask = trt_np != -1

        set_ref = set(ref_np[mask])
        set_trt = set(trt_np[mask])
        intersection = set_ref & set_trt
        intersections.append(len(intersection) / len(set_ref))
    print("average intersections: {:.4f}".format(sum(intersections) / len(intersections)))


test_kernel()
