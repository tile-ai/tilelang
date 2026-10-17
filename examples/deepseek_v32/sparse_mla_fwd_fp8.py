# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from utils import assert_tensors_similar


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def sparse_mla_fwd_fp8(
    Q,
    KV,
    Indices,
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    """FP8 (e4m3) sparse MLA forward for DeepSeek V3.2, with a bf16 output.

    On SM90 the FP8 form of `wgmma.mma_async` has no `imm-trans-a`/`imm-trans-b`
    operands, unlike the bf16/fp16 forms, so both GEMM operands must be K-major.
    The QK product satisfies this (both Q and KV are contiguous along `dim`),
    but the PV product does not: its contraction axis is `block_I`, while KV is
    laid out `[block_I, dim]`. Without a K-major V, `T.gemm` falls back to
    `mma.sync` and the kernel runs ~1.8x slower.

    Storing V column-major in global memory (as FlashAttention-3 does for dense
    FP8 attention) is not an option here, because the top-k gather reads one
    contiguous row per selected token. This kernel keeps the coalesced gather
    and transposes V inside shared memory instead. The transpose sits before the
    QK GEMM so that its CUDA-core work overlaps with the asynchronous wgmma.

    Args:
        Q: Query tensor, `[batch, seq_len, heads, dim + tail_dim]`, fp8 e4m3.
        KV: Key/value cache, `[batch, seq_len_kv, kv_group, dim + tail_dim]`, fp8 e4m3.
        Indices: Selected KV positions, `[batch, seq_len, kv_group, topk]`, int32.
        heads: Number of query heads.
        dim: Latent (value) head dim, 512 for V3.2.
        tail_dim: RoPE head dim, 64 for V3.2.
        topk: Number of selected KV tokens per query.
        kv_group: Number of KV groups.
        sm_scale: Softmax scale; defaults to `1 / sqrt(dim + tail_dim)`.
        is_causal: Must be True.
        CP0: Kept for signature parity with the bf16 kernel.
        block_I: KV tile size along topk.
        num_stages: Pipeline depth of the topk loop.
        threads: Threads per block.

    Returns:
        Tuple of the bf16 output `[batch, seq_len, heads, dim]` and the fp32
        log-sum-exp `[batch, seq_len, heads]`.
    """
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.float8_e4m3
    out_dtype = T.bfloat16
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

    # A shared-memory access moves at most 128 bits, so an fp8 vector holds 16
    # elements. TM is the number of rows each thread transposes at once; 16 was
    # the best of {4, 8, 16, 32} on H800 and keeps the staging buffer at 256 B.
    TN = 16
    TM = 16
    assert BI % TM == 0 and D % TN == 0, f"block_I={BI} must be a multiple of {TM}, dim={D} of {TN}"

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    Q: T.Tensor(q_shape, dtype)  # type: ignore
    KV: T.Tensor(kv_shape, dtype)  # type: ignore
    Indices: T.Tensor(indices_shape, indices_dtype)  # type: ignore
    Output = T.empty(o_shape, out_dtype)
    Lse = T.empty(lse_shape, accum_dtype)

    with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
        bx,
        by,
        bz,
    ):
        # Q is split in two buffers instead of one [H, D + D_tail] as in the bf16
        # kernel: an fp8 swizzled layout needs the contiguous extent to be a
        # multiple of 128 elements, and 576 is not.
        Q_shared = T.alloc_shared([H_per_block, D], dtype)
        Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
        KV_shared = T.alloc_shared([BI, D], dtype)
        # K-major copy of V, so that the PV GEMM can use wgmma.
        KV_T_shared = T.alloc_shared([D, BI], dtype)
        K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
        mask = T.alloc_fragment([BI], "bool")

        acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
        acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
        S_shared = T.alloc_shared([H_per_block, BI], dtype)
        kv_t_frag = T.alloc_local([TM, TN], dtype)
        sumexp = T.alloc_fragment([H_per_block], accum_dtype)
        sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
        alpha = T.alloc_fragment([H_per_block], accum_dtype)
        m_i = T.alloc_fragment([H_per_block], accum_dtype)
        m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

        T.fill(acc_o, 0)
        T.fill(sumexp, 0)
        T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

        b_i, g_i = by, bz
        s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
        q_i = s_i
        max_kv_i = q_i

        H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
        H1 = H0 + H_per_block

        # TODO: merge the statements when the compiler has better support for non-power-of-2 extents.
        T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
        T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

        for i_i in T.Pipelined(NI, num_stages=num_stages):
            for bi_i in T.Parallel(BI):
                mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

            for bi_i, d_i in T.Parallel(BI, D):
                KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
            for bi_i, d_i in T.Parallel(BI, D_tail):
                K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

            # Transpose V for the PV GEMM. Reads are 128-bit vectors along dim,
            # the staging buffer does the swap, writes go out row by row.
            # This only depends on KV_shared, so placing it ahead of the QK GEMM
            # lets it overlap with the asynchronous wgmma that follows.
            for bi_o, d_o in T.Parallel(BI // TM, D // TN):
                for i in T.serial(TM):
                    for j in T.vectorized(TN):
                        kv_t_frag[i, j] = KV_shared[bi_o * TM + i, d_o * TN + j]
                for j in T.serial(TN):
                    for i in T.serial(TM):
                        KV_T_shared[d_o * TN + j, bi_o * TM + i] = kv_t_frag[i, j]

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
            T.copy(m_i, m_i_prev)
            T.reduce_max(acc_s, m_i, dim=1, clear=False)
            for h_i in T.Parallel(H_per_block):
                m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
            for h_i in T.Parallel(H_per_block):
                alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
            for h_i, bi_i in T.Parallel(H_per_block, BI):
                acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
            T.reduce_sum(acc_s, sumexp_i, dim=1)
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

            T.copy(acc_s, S_shared)
            T.gemm(S_shared, KV_T_shared, acc_o, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        # Rescale
        for h_i, d_i in T.Parallel(H_per_block, D):
            acc_o[h_i, d_i] /= sumexp[h_i]
        for h_i in T.Parallel(H_per_block):
            sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

        T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
        T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return Output, Lse


def sparse_mla_fwd_fp8_interface(q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512, block_I=64, num_stages=2, threads=256):
    """Run the FP8 kernel on fp8 e4m3 tensors, returning a bf16 output and fp32 lse."""
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert q.dtype == torch.float8_e4m3fn and kv.dtype == torch.float8_e4m3fn, "expected fp8 e4m3 inputs"
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    out, lse = sparse_mla_fwd_fp8(
        q, kv, indices, heads, dim, tail_dim, topk, kv_group, sm_scale, is_casual, block_I=block_I, num_stages=num_stages, threads=threads
    )
    return out, lse


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    """Float32 reference. Takes dequantized (bf16) tensors so that only the kernel is under test."""
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
        1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda"
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : 1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def _make_inputs(B, S, SKV, H, HKV, DQK, topk):
    """Build fp8 inputs plus their dequantized bf16 twins, and a causal top-k index set."""
    torch.random.manual_seed(0)
    q_bf16 = torch.randn((B, S, H, DQK), dtype=torch.bfloat16, device="cuda")
    kv_bf16 = torch.randn((B, SKV, HKV, DQK), dtype=torch.bfloat16, device="cuda")
    q = q_bf16.to(torch.float8_e4m3fn)
    kv = kv_bf16.to(torch.float8_e4m3fn)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i
    return q, kv, indices


def test_sparse_mla_fwd_fp8(
    B=1,
    S=4096,
    SKV=8192,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    check_correctness=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    """Check the FP8 kernel against the float32 reference and report throughput."""
    q, kv, indices = _make_inputs(B, S, SKV, H, HKV, DQK, topk)

    tl_out, tl_lse = sparse_mla_fwd_fp8_interface(q, kv, indices, block_I=block_I, num_stages=num_stages, threads=threads)

    if check_correctness:
        # Compare against the dequantized inputs, so the error reflects the
        # kernel rather than the fp8 rounding of its inputs.
        ref_out = ref_sparse_mla_fwd_interface(q.to(torch.bfloat16), kv.to(torch.bfloat16), indices)
        assert_tensors_similar(tl_out, ref_out, eps=5e-2, name="out")
        print("assert_tensors_similar passed")

    def fn():
        return sparse_mla_fwd_fp8_interface(q, kv, indices, block_I=block_I, num_stages=num_stages, threads=threads)

    from tilelang.profiler import do_bench

    ms = do_bench(fn, warmup=100, rep=250)
    print(f"Average time: {ms:.3f} ms")
    print("fwd io bandwidth = ", (B * S * DQK * topk) / (ms * 1e-3) / 1e12)
    print("fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)


def run_regression_perf(B=1, S=4096, SKV=8192, H=128, HKV=1, DQK=576, DV=512, topk=2048, block_I=64, num_stages=2, threads=256):
    """Time the kernel only, for the performance regression suite."""
    q, kv, indices = _make_inputs(B, S, SKV, H, HKV, DQK, topk)
    dim = DV
    tail_dim = DQK - dim
    _, _, heads, _ = q.shape
    _, _, kv_group, _ = kv.shape

    def run_kernel_only():
        sparse_mla_fwd_fp8(
            q, kv, indices, heads, dim, tail_dim, topk, kv_group, None, True, block_I=block_I, num_stages=num_stages, threads=threads
        )

    from tilelang.profiler import do_bench

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    test_sparse_mla_fwd_fp8()
