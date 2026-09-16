"""SageAttention3 end to end in TileLang on SM120: bf16 ``q, k, v`` in, bf16 attention out.

``sageattn3_tl(q, k, v)`` computes what ``sageattn3_blackwell(q, k, v)`` computes (non-causal,
``head_dim`` 128, ``seq_len % 128 == 0``, same Q/K sequence length) with four device launches:

1. ``sa3_means`` -- K sequence mean and Q 128-row block means,
2. ``sa3_quant`` -- Q, K and V^T quantized into the attention kernel's layouts (and, from 2K
   tokens up, the smoothed K; below that ``k - km`` in torch is cheaper),
3. ``torch.matmul`` -- delta_s = QM @ K_smoothed^T in bf16,
4. the persistent warp-specialized attention kernel, taking delta_s in bf16.

delta_s stays bf16 because that is all the information the original carries: it multiplies in
bf16 and widens afterwards. The attention kernel widens it in its producer, which saves the
original's full-size fp32 copy (4.5 ms of 86 at 32K) and gives output bit-identical to feeding
the same values in fp32. The bf16-delta_s attention kernel is compiled at ptxas register-usage
level 5, the fastest of the four code generations measured for it.

Unlike the original, the inputs are not modified (the original subtracts the K mean in place).
"""

import torch

import tilelang
import tilelang.language as T

from examples.attention_sm120 import sageattn3_prep as prep
from examples.attention_sm120 import sm120_sageattn3_fwd_ws as ws

# below this many tokens the smoothed K is cheaper as a torch subtraction than as a kernel output
KSM_FROM_KERNEL_MIN_SEQ = 2048
_KERNELS: dict = {}


def _kernels(batch: int, heads: int, seq_len: int, dim: int):
    key = (batch, heads, seq_len, dim)
    if key not in _KERNELS:
        emit_ksm = seq_len >= KSM_FROM_KERNEL_MIN_SEQ
        attn = ws.sm120_sageattn3_fwd(
            batch,
            heads,
            seq_len,
            dim,
            ds_dtype=T.bfloat16,
            pass_configs={tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 5},
        )
        _KERNELS[key] = (
            prep.sa3_means(batch, heads, seq_len, dim),
            prep.sa3_quant(batch, heads, seq_len, dim, emit_ksm=emit_ksm),
            attn,
            emit_ksm,
        )
    return _KERNELS[key]


def sageattn3_tl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """bf16 [B, H, N, 128] q, k, v -> bf16 [B, H, N, 128] attention output."""
    for name, x in (("q", q), ("k", k), ("v", v)):
        if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 4:
            raise ValueError(f"{name} must be a 4-D CUDA bfloat16 tensor, got {tuple(x.shape)} {x.dtype} on {x.device}")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            f"q, k, v must have the same shape (non-causal self-attention), got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    b, h, n, d = q.shape
    if d != 128:
        raise ValueError(f"head_dim must be 128, got {d}")
    if n % 128 != 0:
        raise ValueError(f"seq_len must be a multiple of 128, got {n} (padding is not supported yet)")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    means, quant, attn, emit_ksm = _kernels(b, h, n, d)
    qm, km = means(q, k)
    qp, kp, vtp, sfq, sfk, sfv, ksm = quant(q, k, v, qm, km)
    if not emit_ksm:
        ksm = k - km
    ds = torch.matmul(qm, ksm.transpose(-2, -1))
    # Release the smoothed K before the attention call: the output has the same size, so the
    # allocator hands the attention kernel that just-written (cache-hot) block. Holding it through
    # the call measures 0.2265 ms instead of 0.2070 at 1K (B=4, H=32).
    del ksm
    return attn(
        qp.view(torch.int8),
        kp.view(torch.int8),
        vtp.view(torch.int8),
        sfq.view(torch.uint32),
        sfk.view(torch.uint32),
        sfv.view(torch.uint32),
        ds,
    )
