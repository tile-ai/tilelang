# ruff: noqa
import functools

import torch
from einops import rearrange, repeat
from bert_padding import pad_input, unpad_input


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    return padding_mask


def generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(output_unpad, indices_q, batch_size, seqlen_q)
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device)
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(output_unpad, "(b s) h d -> b s h d", b=batch_size)

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device)
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


@functools.lru_cache(maxsize=None)
def flash_attn_varlen_supported(device_index: int = 0) -> bool:
    """Whether the installed flash-attn has kernels for this device.

    flash-attn refuses architectures it was not built for at the first call
    (`RuntimeError: FlashAttention only supports Ampere GPUs or newer.`), so
    probe once with a minimal varlen call instead of guessing from versions.
    """
    import flash_attn

    device = torch.device("cuda", device_index)
    q = torch.zeros(8, 1, 16, device=device, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    try:
        flash_attn.flash_attn_varlen_func(q, q, q, cu_seqlens, cu_seqlens, 4, 4)
    except Exception:
        return False
    return True


def varlen_attention_ref(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, causal=False, window_size=None):
    """Plain-PyTorch varlen attention for devices flash-attn cannot run on.

    Takes the flash-attn varlen layout, `(total_tokens, heads, dim)` with
    `cu_seqlens_*` holding each sequence's offsets, and returns the same layout.
    """
    import torch.nn.functional as F

    heads = q_unpad.size(1)
    groups = heads // k_unpad.size(1)
    outs = []
    for b in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        k_start, k_end = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
        if q_end == q_start:
            continue
        qb = q_unpad[q_start:q_end].transpose(0, 1).unsqueeze(0).float()
        kb = k_unpad[k_start:k_end].transpose(0, 1).unsqueeze(0).float()
        vb = v_unpad[k_start:k_end].transpose(0, 1).unsqueeze(0).float()
        if groups > 1:
            kb = kb.repeat_interleave(groups, dim=1)
            vb = vb.repeat_interleave(groups, dim=1)

        mask = None
        if causal or window_size is not None:
            rows = torch.arange(q_end - q_start, device=q_unpad.device).unsqueeze(1)
            cols = torch.arange(k_end - k_start, device=q_unpad.device).unsqueeze(0)
            # flash-attn aligns a causal mask to the bottom-right corner.
            diagonal = cols - rows - (k_end - k_start - (q_end - q_start))
            allowed = diagonal <= 0 if causal else torch.ones_like(diagonal, dtype=torch.bool)
            if window_size is not None:
                left, right = window_size
                if left is not None and left >= 0:
                    allowed = allowed & (diagonal >= -left)
                if right is not None and right >= 0:
                    allowed = allowed & (diagonal <= right)
            mask = torch.zeros_like(diagonal, dtype=qb.dtype).masked_fill(~allowed, float("-inf"))

        out = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask)
        outs.append(out.squeeze(0).transpose(0, 1).to(q_unpad.dtype))
    return torch.cat(outs, dim=0)
