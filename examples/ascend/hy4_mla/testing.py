"""Reproducible inputs, the FP32 reference, and numerical checks for the BF16 core."""

import math

import torch


def make_inputs(nq, nk, topk=2048, *, seed=42, component="both", variable_lengths=True):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn((nq, 64, 576), generator=generator) * 0.3).to(torch.bfloat16)
    kv = (torch.randn((nk, 576), generator=generator) * 0.3).to(torch.bfloat16)
    if component == "rope":
        q[..., :512] = 0
    elif component == "latent":
        q[..., 512:] = 0
    elif component != "both":
        raise ValueError("component must be both, rope or latent")
    indices = torch.randint(nk, (nq, topk), generator=generator, dtype=torch.int32)
    # Mixed invalid pairs exercise zero-burst DMA and nonadjacent gathers.
    indices[:, ::17] = -1
    indices[:, 5::29] = nk
    indices[:, 7::31] = -9
    indices[:, 11::37] = nk + 17
    attn_sink = torch.linspace(-3, 3, 64, dtype=torch.float32)
    boundaries = torch.tensor([topk, 0, 1, 63, 64, 65, 639, 640, 641, topk - 1], dtype=torch.int32).clamp(max=topk)
    lengths = boundaries.repeat((nq + len(boundaries) - 1) // len(boundaries))[:nq].contiguous()
    if not variable_lengths:
        lengths.fill_(topk)
    return q, kv, indices, attn_sink, lengths


def sparse_mla_reference(q, kv, indices, attn_sink, lengths=None, *, scale, sink=True):
    """Independent FP32 gather/softmax reference; no TileLang or NPU imports.

    Max and LSE exclude the sink. Empty rows use FlashMLA's compatibility
    sentinels (zero output, -inf maximum, +inf LSE).
    """
    if q.ndim != 3 or q.shape[1:] != (64, 576):
        raise ValueError("Q must have shape [nq, 64, 576]")
    if kv.ndim != 2 or kv.shape[1] != 576 or kv.shape[0] == 0:
        raise ValueError("KV must have nonempty shape [nk, 576]")
    if indices.ndim != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError("Indices must have shape [nq, K]")
    if attn_sink.shape != (64,):
        raise ValueError("Sink must have shape [64]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    nq, _, _ = q.shape
    topk = indices.shape[1]
    if lengths is not None and (lengths.shape != (nq,) or bool(((lengths < 0) | (lengths > topk)).any())):
        raise ValueError("Lengths must have shape [nq] and values in [0, K]")
    output = torch.zeros((nq, 64, 512), dtype=q.dtype, device=q.device)
    maximum = torch.full((nq, 64), -torch.inf, dtype=torch.float32, device=q.device)
    lse = torch.full_like(maximum, torch.inf)
    # Keep reference memory O(K*576), rather than materializing [nq,K,576].
    for t in range(nq):
        limit = int(lengths[t]) if lengths is not None else topk
        selected = indices[t, :limit].long()
        valid = (selected >= 0) & (selected < kv.shape[0])
        selected = selected[valid]
        if selected.numel() == 0:
            continue
        focused = kv[selected].float()
        scores = (q[t].float() @ focused.T) * scale
        maximum[t] = scores.max(dim=-1).values
        lse[t] = torch.logsumexp(scores, dim=-1)
        # A zero-value sink column gives a stable denominator even if its
        # logit is much larger than all data logits. Duplicates retain weight.
        # FlashMLA defines a +inf sink as zero output, rather than softmax's
        # undefined inf-inf case. -inf (including padded heads) is a no-op.
        positive_inf_sink = torch.isposinf(attn_sink) if sink else None
        safe_sink = attn_sink.float().masked_fill(positive_inf_sink, 0) if sink else None
        logits = torch.cat((scores, safe_sink[:, None]), dim=-1) if sink else scores
        probabilities = logits.softmax(dim=-1)[:, : selected.numel()]
        if sink:
            probabilities = probabilities.masked_fill(positive_inf_sink[:, None], 0)
        output[t] = (probabilities @ focused[:, :512]).to(q.dtype)
    return output, maximum, lse


def check_outputs(actual, expected):
    out, maximum, lse = (t.detach().cpu() for t in actual)
    ref_out, ref_max, ref_lse = (t.detach().cpu() for t in expected)
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(maximum, ref_max, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-4, atol=2e-4)
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"
    delta = out.float() - ref_out.float()
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / ref_out.float().norm().clamp_min(1e-12)).item(),
    }
