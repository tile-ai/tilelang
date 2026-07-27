# ruff: noqa: E712
import math
import torch

import triton
import triton.language as tl
import torch.nn.functional as F


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_sparse_attn_mask_from_topk(x, topk, use_dense_for_last_block=False):
    bsz, num_head, downsample_len, _ = x.shape
    # N_CTX = downsample_len * BLOCK
    sparse_index = torch.topk(x, topk, dim=-1).indices
    dense_mask = torch.full([bsz, num_head, downsample_len, downsample_len], False, dtype=torch.bool, device=x.device)
    dense_mask.scatter_(-1, sparse_index, True)
    if use_dense_for_last_block:
        dense_mask[:, :, -2:, :] = True
    dense_mask.tril_()
    return dense_mask


def get_sparse_attn_mask_from_threshold(x, threshold, use_dense_for_last_block=False):
    dense_mask = x > threshold
    if use_dense_for_last_block:
        dense_mask[:, :, -2:, :] = True
    dense_mask.tril_()
    return dense_mask


@triton.jit
def _fwd_kernel_inner(
    acc,
    l_i,
    m_i,
    q,
    k_block_col_idx,
    block_mask_ptr,
    k_ptrs,
    v_ptrs,
    offs_m,
    offs_n,
    stride_kt,
    stride_vt,
    stride_bmask_n,
    sm_scale,
    past_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    mask_val = tl.load(block_mask_ptr + k_block_col_idx * stride_bmask_n)

    if mask_val == True:
        start_n = k_block_col_idx * BLOCK_N
        # -- compute qk ----

        k = tl.load(k_ptrs + start_n * stride_kt)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        qk *= sm_scale

        # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
        qk += tl.where(offs_m[:, None] + past_len >= (start_n + offs_n[None, :]), 0, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(v_ptrs + start_n * stride_vt)

        p = p.to(v.type.element_ty)

        acc += tl.dot(p, v)
        # update m_i and l_i
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    block_mask_ptr,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_bmz,
    stride_bmh,
    stride_bmm,
    stride_bmn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    H,
    N_CTX,
    PAST_LEN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    Q_LEN = N_CTX - PAST_LEN
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    # off_k = offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    mask_ptrs = block_mask_ptr + start_m * stride_bmm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)

    k_block_start = 0
    k_block_end = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)

    # loop over k, v and update accumulator
    for col_idx in range(k_block_start, k_block_end):
        acc, l_i, m_i = _fwd_kernel_inner(
            acc,
            l_i,
            m_i,
            q,
            col_idx,
            mask_ptrs,
            k_ptrs,
            v_ptrs,
            offs_m,
            offs_n,
            stride_kn,
            stride_vn,
            stride_bmn,
            sm_scale,
            PAST_LEN,
            BLOCK_M,
            BLOCK_N,
        )

    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(Out.dtype.element_ty)

    off_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def _forward(ctx, q, k, v, block_sparse_mask, sm_scale, BLOCK_M=64, BLOCK_N=64, num_warps=None, num_stages=1, out=None):
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2]
    o = out if out is not None else torch.empty_like(q).contiguous()
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])

    assert q.shape[-1] in [64, 128]
    BLOCK_DMODEL = q.shape[-1]

    if is_hip():
        num_warps, num_stages = 8, 1
    else:
        num_warps, num_stages = 4, 2

    N_CTX = k.shape[2]
    PAST_LEN = N_CTX - q.shape[2]
    print("PAST_LEN", PAST_LEN)
    H = q.shape[1]

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        block_sparse_mask,
        o,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *block_sparse_mask.stride(),
        *o.stride(),
        H,
        N_CTX,
        PAST_LEN,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o


class _sparse_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_sparse_dense, sm_scale):
        # shape constraints
        return _forward(ctx, q, k, v, block_sparse_dense, sm_scale)

    @staticmethod
    def backward(ctx, do):
        # No gradient propagation.
        raise NotImplementedError("It does not support gradient propagation yet")
        return None, None, None, None, None


block_sparse_triton_fn = _sparse_attention.apply


def test_topk_sparse_attention():
    # Config
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 1, 1, 256, 64
    TOPK = 2  # Keep top 8 elements per row
    BLOCK = 64
    torch.manual_seed(0)

    # Create inputs
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.bfloat16)
    sm_scale = 1.0 / (D_HEAD**0.5)

    # Create sparse mask (downsampled to block level)
    downsample_factor = BLOCK
    downsample_len = math.ceil(SEQ_LEN / downsample_factor)
    print("downsample_len", downsample_len)

    x_ds = torch.randn([BATCH, N_HEADS, downsample_len, downsample_len], device="cuda", dtype=torch.bfloat16)
    x_ds[:, :, :, 0] = 100
    print("x_ds.shape", x_ds.shape)
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)
    # print("block_mask", block_mask)
    print("block_mask.shape", block_mask.shape)

    # Run Triton kernel
    triton_output = block_sparse_triton_fn(q, k, v, block_mask, sm_scale)

    # Compute reference
    # Expand block mask to full attention matrix
    full_mask = torch.kron(block_mask.float(), torch.ones(BLOCK, BLOCK, device="cuda"))
    full_mask = full_mask[..., :SEQ_LEN, :SEQ_LEN].bool()
    full_mask = full_mask & torch.tril(torch.ones_like(full_mask))  # Apply causal

    # PyTorch reference implementation
    attn = torch.einsum("bhsd,bhtd->bhst", q, k) * sm_scale
    attn = attn.masked_fill(~full_mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    ref_output = torch.einsum("bhst,bhtd->bhsd", attn, v)

    # print("ref_output", ref_output)
    # print("triton_output", triton_output)

    # Verify accuracy
    assert torch.allclose(triton_output, ref_output, atol=1e-2, rtol=1e-2), "Triton output doesn't match reference"
    print("Pass topk sparse attention test with qlen == klen")


def test_topk_sparse_attention_qlt_kl():
    BATCH, N_HEADS = 1, 1
    Q_LEN, K_LEN, D_HEAD = 64, 256, 64  # qlen < klen; here, past_len = 256 - 128 = 128.
    TOPK = 1
    BLOCK = 64  # block size used in downsampling
    torch.manual_seed(0)

    # Create inputs.
    q = torch.randn(BATCH, N_HEADS, Q_LEN, D_HEAD, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(BATCH, N_HEADS, K_LEN, D_HEAD, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(BATCH, N_HEADS, K_LEN, D_HEAD, device="cuda", dtype=torch.bfloat16)
    # softmax scale
    sm_scale = 1.0 / (D_HEAD**0.5)

    downsample_factor = BLOCK
    downsample_len = math.ceil(K_LEN / downsample_factor)  # number of blocks along one dimension
    x_ds = torch.randn(BATCH, N_HEADS, downsample_len, downsample_len, device="cuda", dtype=torch.bfloat16)
    # Force the first column to be high so that the first block is always selected.
    x_ds[:, :, :, 0] = 100
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)
    # Run Triton kernel.
    triton_output = block_sparse_triton_fn(q, k, v, block_mask, sm_scale)

    past_len = K_LEN - Q_LEN

    attn = torch.einsum("bhsd,bhtd->bhst", q, k) * sm_scale

    full_mask_full = torch.kron(block_mask.float(), torch.ones(BLOCK, BLOCK, device="cuda")).bool()
    full_mask_full = full_mask_full[..., :K_LEN, :K_LEN]

    effective_mask = full_mask_full[..., past_len:K_LEN, :]  # shape: (B, H, Q_LEN, K_LEN)

    i_global = torch.arange(past_len, K_LEN, device=k.device).unsqueeze(1)  # shape: (Q_LEN, 1)
    j_global = torch.arange(K_LEN, device=k.device).unsqueeze(0)  # shape: (1, K_LEN)
    causal_mask = j_global <= i_global  # shape: (Q_LEN, K_LEN)

    final_mask = effective_mask & causal_mask  # shape: (B, H, Q_LEN, K_LEN)

    attn = attn.masked_fill(~final_mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    ref_output = torch.einsum("bhst,bhtd->bhsd", attn, v)

    # Verify accuracy.
    assert torch.allclose(triton_output, ref_output, atol=1e-2, rtol=1e-2), "Triton output doesn't match reference when qlen < klen"

    print("Pass topk sparse attention test with qlen < klen")


if __name__ == "__main__":
    test_topk_sparse_attention()
    test_topk_sparse_attention_qlt_kl()
