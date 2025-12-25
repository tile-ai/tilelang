import torch
import triton
import triton.language as tl
import math
import argparse
import tilelang
import tilelang.language as T

torch.manual_seed(0)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@triton.jit
def _fwd_inner(
    q,
    k_ptrs,
    v_ptrs,
    s_ptrs,
    m_i,
    l_i,
    acc,
    offs_h,
    mask_h,
    offs_n,
    seqlen,
    softmax_scale,
    lo,
    hi,
    stride_kt,
    stride_vt,
    stride_sh,
    stride_sn,
    BLOCK_N: tl.constexpr,
):
    """Inner loop computation for attention"""

    for blk_idx in tl.range(lo, hi):
        start_n = blk_idx * BLOCK_N
        k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen)
        v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen)

        qk = tl.dot(q, k)
        qk *= softmax_scale
        qk += tl.where(offs_n[None, :] + start_n < seqlen, 0, -1.0e9)

        row_max = tl.max(qk, 1)
        tl.store(s_ptrs + offs_h * stride_sh + blk_idx * stride_sn, row_max, mask=mask_h)

        m_ij = tl.maximum(m_i, row_max)
        qk -= m_ij[:, None]
        p = tl.math.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc *= alpha[:, None]
        p = p.to(v.type.element_ty)
        acc += tl.dot(p, v)

    return m_i, l_i, acc


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [4, 8] for num_stages in [2, 4]],
    key=["gqa_group_size", "BLOCK_N", "BLOCK_D", "BLOCK_H"],
)
@triton.jit
def _fwd_kernel_varlen(
    Q,  # [token_q = b, h_q, dim]
    K,  # [token_k, h_kv, dim]
    V,
    O,
    S,
    s_aux,
    softmax_scale,
    cu_seqlens_k,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_sb,
    stride_sh,
    stride_sn,  # bmask shape [b, q_h, seq/BLOCK_N]
    gqa_group_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_h_for_kv = tl.program_id(1)
    off_h_q = off_h_for_kv * gqa_group_size

    cu_k_start = tl.load(cu_seqlens_k + off_z)
    cu_k_end = tl.load(cu_seqlens_k + off_z + 1)

    seqlen_k = cu_k_end - cu_k_start

    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    Q_ptrs = Q + off_z * stride_qt + off_h_q * stride_qh
    K_ptrs = K + (cu_k_start) * stride_kt + off_h_for_kv * stride_kh
    V_ptrs = V + (cu_k_start) * stride_vt + off_h_for_kv * stride_vh
    O_ptrs = O + off_z * stride_ot + off_h_q * stride_oh
    S_ptrs = S + off_z * stride_sb + off_h_q * stride_sh

    mask_h = offs_h < gqa_group_size
    q = tl.load(Q_ptrs + offs_d[None, :] * stride_qd + offs_h[:, None] * stride_qh, mask=mask_h[:, None])

    if s_aux is not None:
        sink = tl.load(s_aux + off_h_q + offs_h, mask=mask_h).to(tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        m_i = tl.zeros([BLOCK_H], dtype=tl.float32) + sink
    else:
        l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
        m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)

    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    k_ptrs = K_ptrs + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V_ptrs + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

    lo, hi = 0, tl.cdiv(seqlen_k, BLOCK_N)
    m_i, l_i, acc = _fwd_inner(
        q,
        k_ptrs,
        v_ptrs,
        S_ptrs,
        m_i,
        l_i,
        acc,
        offs_h,
        mask_h,
        offs_n,
        seqlen_k,
        softmax_scale,
        lo,
        hi,
        stride_kt,
        stride_vt,
        stride_sh,
        stride_sn,
        BLOCK_N,
    )

    if s_aux is not None:
        sink = tl.math.exp(sink - m_i)
        l_i = l_i + sink
        acc = acc / l_i[:, None]

    else:
        l_recip = 1 / l_i[:, None]
        acc = acc * l_recip

    for blk_idx in tl.range(lo, hi):
        s = tl.load(S_ptrs + offs_h * stride_sh + blk_idx * stride_sn, mask=mask_h)
        s = tl.exp(s - m_i) / l_i
        tl.store(S_ptrs + offs_h * stride_sh + blk_idx * stride_sn, s, mask=mask_h)

    acc = acc.to(O.dtype.element_ty)

    tl.store(O_ptrs + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od, acc, mask=mask_h[:, None])


def get_configs():
    import itertools

    block_N = [64, 128]
    block_H = [64]
    num_split = [1]
    num_stages = [1, 2, 3]
    threads = [128]
    _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

    configs = [{"block_N": c[0], "block_H": c[1], "num_split": c[2], "num_stages": c[3], "threads": c[4]} for c in _configs]
    return configs


@tilelang.jit(out_idx=[-2, -1])
def flashattn(
    batch, heads, k_heads, max_seqlen_kv, total_seqlen_k, dim, has_sink, block_N=128, block_H=64, num_split=1, num_stages=1, threads=128
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape_q = [batch, heads, dim]
    shape_k = [total_seqlen_k, k_heads, dim]
    shape_v = [total_seqlen_k, k_heads, dim]
    shape_o = [batch, heads, dim]
    shape_s = [batch, heads, math.ceil(max_seqlen_kv / block_N)]
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // k_heads

    valid_block_H = min(block_H, kv_group_num)
    # TODO: check if max_seqlen_kv is correct for varlen case

    @T.prim_func
    def flashattn_gqa_decode_no_split(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        cu_seqlens_k: T.Tensor([batch + 1], T.int32),
        s_aux: T.Tensor([heads], T.float32),
        Output: T.Tensor(shape_o, dtype),
        S: T.Tensor(shape_s, dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            S_shared = T.alloc_shared([block_H, math.ceil(max_seqlen_kv / block_N)], dtype)
            # S_fragment = T.alloc_fragment([block_H, math.ceil(max_seqlen_kv / block_N)], accum_dtype)
            s_aux_shared = T.alloc_shared([block_H], T.float32)

            bid = bx
            hid = by
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            cur_start_k = cu_seqlens_k[bid]
            cur_end_k = cu_seqlens_k[bid + 1]
            cur_seqlen_k = cur_end_k - cur_start_k

            T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
            loop_range = T.ceildiv((cur_seqlen_k // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[cur_start_k + k * block_N : cur_start_k + (k + 1) * block_N, cur_kv_head, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    # acc_s[i, j] = T.if_then_else(mask_local[j] != 0 and k * block_N + j < cur_seqlen_k, acc_s[i, j],
                    #                              -T.infinity(accum_dtype))
                    acc_s[i, j] = T.if_then_else(k * block_N + j < cur_seqlen_k, acc_s[i, j], -T.infinity(accum_dtype))
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                # scores_max_prev is m_i
                # scores_max is row_max->m_ij in triton
                T.copy(scores_max, S_shared[:, k])
                # scores_scale is alpha in triton
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                # scores_sum is l_ij in triton
                # logsum is l_i in triton
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(V[cur_start_k + k * block_N : cur_start_k + (k + 1) * block_N, cur_kv_head, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            if has_sink:
                T.copy(s_aux[hid * valid_block_H : hid * valid_block_H + block_H], s_aux_shared)
                for i in T.Parallel(block_H):
                    logsum[i] += s_aux_shared[i]
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            for h, k in T.Parallel(block_H, math.ceil(max_seqlen_kv / block_N)):
                S_shared[h, k] = T.exp2((S_shared[h, k] - scores_max[h]) * scale) / logsum[h]
            # T.copy(S_shared, S_fragment)
            # for h, k in T.Parallel(block_H, math.ceil(max_seqlen_kv / block_N)):
            #     S_fragment[h, k] = T.exp2((S_fragment[h, k] - scores_max[h]) * scale) / logsum[h]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])
            # T.copy(S_fragment, S_shared)
            T.copy(S_shared[:valid_block_H, :], S[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

    # TODO: split version
    return flashattn_gqa_decode_no_split


def flash_attn_with_attn_pool_decode_tilelang(
    Q: torch.Tensor,  ## [tq = b, q_h, q_dim]
    K: torch.Tensor,  ## [tk, k_h, k_dim]
    V: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    real_max_k_seqlen: int,
    num_split: int,
    softmax_scale: float,
    s_aux: torch.Tensor = None,
    block_size: int = 64,
    use_per_kv_head_sparse_index: bool = False,
    tl_kernel=None,
):
    num_tokens, q_h, head_size = Q.shape
    batch = cu_seqlens_k.size(0) - 1
    k_h = K.size(1)

    assert Q.dim() == K.dim() == 3
    assert Q.size(2) == K.size(2)
    assert cu_seqlens_k.dim() == 1
    assert head_size in {64, 128, 256}
    assert Q.is_contiguous()
    # assert K.is_contiguous()
    # assert V.is_contiguous()

    gqa_group_size = q_h // k_h

    O_tl = torch.zeros_like(Q)
    S_tl = torch.zeros((batch, q_h, math.ceil(real_max_k_seqlen / block_size)), dtype=Q.dtype, device=Q.device)
    O_tl, S_tl = tl_kernel(Q, K, V, cu_seqlens_k, s_aux)

    if use_per_kv_head_sparse_index:
        S_tl = torch.max_pool2d(S_tl, kernel_size=(gqa_group_size, 1), stride=(gqa_group_size, 1))
    else:
        S_tl = torch.max_pool2d(S_tl, kernel_size=(q_h, 1), stride=(q_h, 1))

    return O_tl, S_tl


def flash_attn_with_attn_pool_decode(
    Q: torch.Tensor,  ## [tq = b, q_h, q_dim]
    K: torch.Tensor,  ## [tk, k_h, k_dim]
    V: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    real_max_k_seqlen: int,
    num_split: int,
    softmax_scale: float,
    s_aux: torch.Tensor = None,
    block_size: int = 64,
    use_per_kv_head_sparse_index: bool = False,
):
    num_tokens, q_h, head_size = Q.shape
    batch = cu_seqlens_k.size(0) - 1
    k_h = K.size(1)

    assert Q.dim() == K.dim() == 3
    assert Q.size(2) == K.size(2)
    assert cu_seqlens_k.dim() == 1
    assert head_size in {64, 128, 256}
    assert Q.is_contiguous()
    # assert K.is_contiguous()
    # assert V.is_contiguous()

    gqa_group_size = q_h // k_h

    BLOCK_D = head_size
    BLOCK_N = block_size
    BLOCK_H = 64

    O = torch.zeros_like(Q)
    S = torch.zeros((batch, q_h, math.ceil(max_seqlen_k / block_size)), dtype=Q.dtype, device=Q.device)

    def grid(META):
        return (batch, k_h)

    with torch.cuda.device(Q.device.index):
        _fwd_kernel_varlen[grid](
            Q,
            K,
            V,
            O,
            S,
            s_aux,
            softmax_scale,
            cu_seqlens_k,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *S.stride(),
            gqa_group_size,
            BLOCK_H=BLOCK_H,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

    if use_per_kv_head_sparse_index:
        S = torch.max_pool2d(S, kernel_size=(gqa_group_size, 1), stride=(gqa_group_size, 1))
    else:
        S = torch.max_pool2d(S, kernel_size=(q_h, 1), stride=(q_h, 1))

    return O, S


def test_varlen_decode_main(args):
    """Test decode kernel with variable sequence lengths"""
    batch_size = args.batch_size
    q_heads = args.q_heads
    kv_heads = args.kv_heads
    max_k_seqlen = args.k_seqlen  # Use as max sequence length
    real_max_k_seqlen = args.k_seqlen
    head_size = args.head_size
    block_size = args.block_size
    dtype = torch.bfloat16 if args.dtype == T.bfloat16 else torch.float16

    print(f"Testing decode kernel with variable sequence lengths (max_k_seqlen={max_k_seqlen})")

    # Generate sink values if needed
    sink = None
    if args.test_sink:
        sink = torch.randn(q_heads, device="cuda", dtype=torch.float32) * 0.1  # Small sink values
        print(f"Using sink attention with sink values: {sink}")

    # Generate variable length k sequences
    k_seqlens = torch.randint(max_k_seqlen // 4, max_k_seqlen + 1, size=(batch_size,))
    print(f"k_seqlens: {k_seqlens}")

    # Generate cumulative sequence lengths for k
    cu_seqlens_k = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
    total_k_tokens = 0
    for i in range(batch_size):
        cu_seqlens_k[i] = total_k_tokens
        total_k_tokens += k_seqlens[i]
    cu_seqlens_k[batch_size] = total_k_tokens

    print(f"cu_seqlens_k: {cu_seqlens_k}")

    # Generate tensors - Q is [batch_size, q_heads, head_size] for decode
    q_decode = torch.randn(batch_size, q_heads, head_size, device="cuda", dtype=dtype)
    k_varlen = torch.randn(total_k_tokens, kv_heads, head_size, device="cuda", dtype=dtype)
    v_varlen = torch.randn(total_k_tokens, kv_heads, head_size, device="cuda", dtype=dtype)

    softmax_scale = 1.0 / math.sqrt(head_size)
    max_seqlen_k = int(k_seqlens.max())

    print(f"Actual max_seqlen_k: {max_seqlen_k}")
    print(f"q_decode shape: {q_decode.shape}")
    print(f"k_varlen shape: {k_varlen.shape}")
    print(f"v_varlen shape: {v_varlen.shape}")

    num_tokens, q_h, head_size = q_decode.shape
    batch = cu_seqlens_k.size(0) - 1
    k_h = k_varlen.size(1)
    tl_kernel = flashattn(batch, q_h, k_h, args.k_seqlen, cu_seqlens_k[-1].item(), head_size, args.test_sink)

    # Test our decode kernel
    O_triton, S_triton = flash_attn_with_attn_pool_decode(
        q_decode,
        k_varlen,
        v_varlen,
        cu_seqlens_k,
        max_seqlen_k,
        real_max_k_seqlen,
        args.num_split,
        softmax_scale,
        s_aux=sink,
        block_size=block_size,
    )
    O_tilelang, S_tilelang = flash_attn_with_attn_pool_decode_tilelang(
        q_decode,
        k_varlen,
        v_varlen,
        cu_seqlens_k,
        max_seqlen_k,
        real_max_k_seqlen,
        args.num_split,
        softmax_scale,
        s_aux=sink,
        block_size=block_size,
        tl_kernel=tl_kernel,
    )
    for i in range(batch_size):
        S_tilelang[i, :, math.ceil((cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item()) / block_size) :] = 0

    # Create torch reference - pad tensors for comparison
    k_padded_list = []
    v_padded_list = []

    for i in range(batch_size):
        actual_k_len = k_seqlens[i]

        # Extract and pad k, v for this batch
        k_start = cu_seqlens_k[i]
        k_end = cu_seqlens_k[i + 1]

        # Pad to max_seqlen_k
        k_padded = torch.zeros(max_seqlen_k, kv_heads, head_size, device="cuda", dtype=dtype)
        v_padded = torch.zeros(max_seqlen_k, kv_heads, head_size, device="cuda", dtype=dtype)

        k_padded[:actual_k_len] = k_varlen[k_start:k_end]
        v_padded[:actual_k_len] = v_varlen[k_start:k_end]

        k_padded_list.append(k_padded)
        v_padded_list.append(v_padded)

    # Stack to create batched tensors [b, max_seqlen, kv_heads, head_size]
    k_padded_batched = torch.stack(k_padded_list, dim=0).transpose(1, 2)  # [b, kv_heads, max_seqlen, head_size]
    v_padded_batched = torch.stack(v_padded_list, dim=0).transpose(1, 2)  # [b, kv_heads, max_seqlen, head_size]

    # Expand q to match kv heads: [b, q_heads, 1, head_size]
    q_expanded = q_decode.unsqueeze(2)  # [b, q_heads, 1, head_size]

    print(f"q_expanded shape: {q_expanded.shape}")
    print(f"k_padded_batched shape: {k_padded_batched.shape}")
    print(f"v_padded_batched shape: {v_padded_batched.shape}")

    # Compute torch reference
    k_repeat = repeat_kv(k_padded_batched, q_heads // kv_heads)  # [b, q_heads, max_seqlen, head_size]
    v_repeat = repeat_kv(v_padded_batched, q_heads // kv_heads)  # [b, q_heads, max_seqlen, head_size]

    if sink is None:
        # Standard attention computation: [b, q_heads, 1, head_size] @ [b, q_heads, head_size, max_seqlen]
        attn_score = torch.matmul(q_expanded, k_repeat.transpose(-2, -1)) * softmax_scale  # [b, q_heads, 1, max_seqlen]

        # Apply sequence length masking
        for i in range(batch_size):
            actual_k_len = k_seqlens[i]
            attn_score[i, :, :, actual_k_len:] = float("-inf")

        attn_weights = attn_score.softmax(dim=-1)  # [b, q_heads, 1, max_seqlen]

        # Mask out invalid positions
        for i in range(batch_size):
            actual_k_len = k_seqlens[i]
            attn_weights[i, :, :, actual_k_len:] = 0.0

        # Compute output: [b, q_heads, 1, max_seqlen] @ [b, q_heads, max_seqlen, head_size]
        O_torch = torch.matmul(attn_weights, v_repeat)  # [b, q_heads, 1, head_size]
    else:
        # s_aux attention
        logits = torch.matmul(q_expanded, k_repeat.transpose(-2, -1)) * softmax_scale  # [b, q_heads, 1, max_seqlen]

        # Apply sequence length masking
        for i in range(batch_size):
            actual_k_len = k_seqlens[i]
            logits[i, :, :, actual_k_len:] = float("-inf")

        sink_expanded = sink.view(1, q_heads, 1, 1)  # [1, q_heads, 1, 1]
        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        logits_or_sinks_max = torch.maximum(logits_max, sink_expanded)
        sinks = torch.exp(sink_expanded - logits_or_sinks_max)
        unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
        attn_weights = unnormalized_scores / normalizer

        # Mask out invalid positions
        for i in range(batch_size):
            actual_k_len = k_seqlens[i]
            attn_weights[i, :, :, actual_k_len:] = 0.0

        # Compute output: [b, q_heads, 1, max_seqlen] @ [b, q_heads, max_seqlen, head_size]
        O_torch = torch.matmul(attn_weights.to(v_repeat.dtype), v_repeat)  # [b, q_heads, 1, head_size]

    O_torch = O_torch.squeeze(2)  # [b, q_heads, head_size]

    # Compute attention score pooling for S
    attn_score_pooled = torch.max_pool2d(
        attn_weights.squeeze(2),  # [b, q_heads, max_seqlen]
        kernel_size=(q_heads, block_size),
        stride=(q_heads, block_size),
        ceil_mode=True,
    ).to(dtype=torch.float16)  # [b, 1, ceil(max_seqlen/block_size)]

    print(f"O_triton shape: {O_triton.shape}")
    print(f"O_tilelang shape: {O_tilelang.shape}")
    print(f"O_torch shape: {O_torch.shape}")
    print(f"S_triton shape: {S_triton.shape}")
    print(f"S_tilelang shape: {S_tilelang.shape}")
    print(f"attn_score_pooled shape: {attn_score_pooled.shape}")

    # Compare results
    max_diff_o = torch.max(torch.abs(O_triton - O_torch))
    max_diff_o_tl = torch.max(torch.abs(O_tilelang - O_torch))
    print(f"Max difference in O: {max_diff_o.item()}")
    print(f"Max difference in O_tilelang: {max_diff_o_tl.item()}")

    max_diff_s = torch.max(torch.abs(S_triton - attn_score_pooled))
    max_diff_s_tl = torch.max(
        torch.abs(
            S_tilelang[:, :, : math.ceil(max_seqlen_k / block_size)] - attn_score_pooled[:, :, : math.ceil(max_seqlen_k / block_size)]
        )
    )
    print(f"Max difference in S: {max_diff_s.item()}")
    print(f"Max difference in S_tilelang: {max_diff_s_tl.item()}")

    assert torch.allclose(O_triton, O_torch, atol=1e-2, rtol=1e-2), f"Output mismatch: {max_diff_o.item()}"
    assert torch.allclose(S_triton, attn_score_pooled, atol=1e-2, rtol=1e-2), f"Score mismatch: {max_diff_s.item()}"
    assert torch.allclose(O_tilelang, O_torch, atol=1e-2, rtol=1e-2), f"Output mismatch: {max_diff_o_tl.item()}"
    assert torch.allclose(
        S_tilelang[:, :, : math.ceil(max_seqlen_k / block_size)],
        attn_score_pooled[:, :, : math.ceil(max_seqlen_k / block_size)],
        atol=1e-2,
        rtol=1e-2,
    ), f"Score mismatch: {max_diff_s_tl.item()}"

    print("✅ All tests passed!")


def do_bench(fn, *args, warmup=10, rep=10, **kwargs):
    """
    Do benchmark for a function.
    """
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    for _ in range(warmup):
        fn(*args, **kwargs)

    torch.cuda.synchronize()
    for i in range(rep):
        start_event[i].record()
        fn(*args, **kwargs)
        end_event[i].record()
    torch.cuda.synchronize()

    # Record clocks
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )

    return times.mean().item()


def speed_benchmark_decode_comparison(args):
    """Speed benchmark for decode kernel"""
    batch_size = args.batch_size
    q_heads = args.q_heads
    kv_heads = args.kv_heads
    max_k_seqlen = args.k_seqlen
    head_size = args.head_size
    block_size = args.block_size
    dtype = torch.bfloat16 if args.dtype == T.bfloat16 else torch.float16

    print("\n=== Decode Speed Benchmark Comparison ===")
    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Q heads: {q_heads}, KV heads: {kv_heads}")
    print(f"  Max K sequence length: {max_k_seqlen}")
    print(f"  Head size: {head_size}")
    print(f"  Block size: {block_size}")
    print(f"  Data type: {dtype}")
    print(f"  Variable lengths: {args.test_varlen}")
    print(f"  s_aux attention: {args.test_sink}")
    print()

    # Generate input data
    if args.test_varlen:
        k_seqlens = torch.randint(max_k_seqlen // 4, max_k_seqlen + 1, size=(batch_size,))
    else:
        k_seqlens = torch.full((batch_size,), max_k_seqlen, dtype=int)

    # Generate cumulative sequence lengths for k
    cu_seqlens_k = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
    total_k_tokens = 0
    for i in range(batch_size):
        cu_seqlens_k[i] = total_k_tokens
        total_k_tokens += k_seqlens[i]
    cu_seqlens_k[batch_size] = total_k_tokens

    # Generate tensors
    q_decode = torch.randn(batch_size, q_heads, head_size, device="cuda", dtype=dtype)
    k_varlen = torch.randn(total_k_tokens, kv_heads, head_size, device="cuda", dtype=dtype)
    v_varlen = torch.randn(total_k_tokens, kv_heads, head_size, device="cuda", dtype=dtype)

    softmax_scale = 1.0 / math.sqrt(head_size)
    max_seqlen_k = int(k_seqlens.max())

    # Generate sink values if needed
    sink = None
    if args.test_sink:
        sink = torch.randn(q_heads, device="cuda", dtype=torch.float32) * 0.1  # Small sink values
        print("  Using sink attention with sink values")

    print("Setup complete:")
    print(f"  Total K tokens: {total_k_tokens}")
    print(f"  Actual max K seq len: {max_seqlen_k}")
    if args.test_varlen:
        print(f"  K sequence lengths: {k_seqlens.tolist()}")

    # Warmup
    num_tokens, q_h, head_size = q_decode.shape
    batch = cu_seqlens_k.size(0) - 1
    k_h = k_varlen.size(1)
    tl_kernel = flashattn(batch, q_h, k_h, args.k_seqlen, cu_seqlens_k[-1].item(), head_size, args.test_sink)

    # Benchmark
    print("⚡ Benchmarking Tilelang kernel (100 iterations)...")
    tilelang_time = do_bench(
        flash_attn_with_attn_pool_decode_tilelang,
        q_decode,
        k_varlen,
        v_varlen,
        cu_seqlens_k,
        max_seqlen_k,
        args.k_seqlen,
        1,
        softmax_scale,
        sink,
        block_size,
        False,
        tl_kernel,
    )
    print(f"Average decode kernel time Tilelang: {tilelang_time:.3f} ms")

    # Benchmark
    print("⚡ Benchmarking Triton kernel (100 iterations)...")
    triton_time = do_bench(
        flash_attn_with_attn_pool_decode,
        q_decode,
        k_varlen,
        v_varlen,
        cu_seqlens_k,
        max_seqlen_k,
        args.k_seqlen,
        1,
        softmax_scale,
        sink,
        block_size,
    )
    print(f"Average decode kernel time Triton: {triton_time:.3f} ms")

    print(f"Speedup: {(triton_time / tilelang_time):.3f}")


def main():
    args = argparse.Namespace(
        batch_size=1,
        q_heads=32,
        kv_heads=8,
        k_seqlen=8192,
        head_size=128,
        block_size=128,
        dtype=T.float16,
    )
    args.test_sink = True
    args.test_varlen = True
    args.dtype = T.float16
    args.num_split = 1
    test_varlen_decode_main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention Decode with Attention Pooling")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--q_heads", type=int, default=32, help="Number of query heads")
    parser.add_argument("--kv_heads", type=int, default=8, help="Number of key-value heads")
    parser.add_argument("--k_seqlen", type=int, default=8192, help="Key sequence length")
    parser.add_argument("--head_size", type=int, default=128, choices=[64, 128, 256], help="Head dimension")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for computation")
    parser.add_argument("--dtype", type=str, default=T.bfloat16, choices=[T.float16, T.bfloat16], help="Data type")
    parser.add_argument("--test_varlen", action="store_true", help="Test with truly variable sequence lengths")
    parser.add_argument("--test_sink", action="store_true", help="Test with sink attention mechanism")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    parser.add_argument("--num_split", type=int, default=1, choices=[1, 16], help="Number of splits")
    args = parser.parse_args()
    args.test_sink = True
    args.test_varlen = True
    args.dtype = T.float16
    args.num_split = 1

    if args.benchmark:
        speed_benchmark_decode_comparison(args)
    else:
        test_varlen_decode_main(args)
