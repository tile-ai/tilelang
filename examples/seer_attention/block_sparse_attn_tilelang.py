import math
import torch

import tilelang
import tilelang.language as T
import torch.nn.functional as F


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


@tilelang.jit(
    out_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def blocksparse_flashattn(batch, heads, seq_q, seq_kv, dim, downsample_len, is_causal):
    block_M = 64
    block_N = 64
    num_stages = 0
    threads = 128
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    block_mask_shape = [batch, heads, downsample_len, downsample_len]

    dtype = T.float16
    accum_dtype = T.float32
    block_mask_dtype = T.int8

    def kernel_func(block_M, block_N, num_stages, threads):
        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            BlockSparseMask: T.Tensor(block_mask_shape, block_mask_dtype),
            Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                block_mask = T.alloc_fragment([downsample_len], block_mask_dtype)

                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                T.copy(BlockSparseMask[bz, by, bx, :], block_mask)

                loop_range = T.ceildiv(seq_kv, block_N)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    if block_mask[k] != 0:
                        T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                        if is_causal:
                            past_len = seq_kv - seq_q
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.if_then_else(bx * block_M + i + past_len >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                        else:
                            T.clear(acc_s)
                        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
                        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
                        # in the first ceil_div(kBlockM, kBlockN) steps.
                        # for i in T.Parallel(block_M):
                        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
                        for i in T.Parallel(block_M):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_M, block_N):
                            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                            # max * log_2(e)) This allows the compiler to use the ffma
                            # instruction instead of fadd and fmul separately.
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)

                        for i, j in T.Parallel(block_M, dim):
                            acc_o[i, j] *= scores_scale[i]

                        T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]

                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

        return main

    return kernel_func(block_M, block_N, num_stages, threads)


def test_topk_sparse_attention():
    # Config
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 4, 2, 256, 64
    TOPK = 2  # Keep top 8 elements per row
    BLOCK = 64
    torch.manual_seed(0)

    # Create inputs
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)

    sm_scale = 1.0 / (D_HEAD**0.5)

    # Create sparse mask (downsampled to block level)
    downsample_factor = BLOCK
    downsample_len = math.ceil(SEQ_LEN / downsample_factor)
    x_ds = torch.randn([BATCH, N_HEADS, downsample_len, downsample_len], device="cuda", dtype=torch.float16)
    x_ds[:, :, :, 0] = 100
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

    # Run tilelang kernel
    kernel = blocksparse_flashattn(BATCH, N_HEADS, SEQ_LEN, SEQ_LEN, D_HEAD, downsample_len, is_causal=True)
    tilelang_output = kernel(q, k, v, block_mask.to(torch.int8))

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

    print("ref_output", ref_output)
    print("tilelang_output", tilelang_output)

    # Verify accuracy
    assert torch.allclose(tilelang_output, ref_output, atol=1e-2, rtol=1e-2), "TileLang output doesn't match reference"
    print("Pass topk sparse attention test with qlen == klen")


def test_topk_sparse_attention_qlen_lt_klen():
    # Config
    BATCH, N_HEADS = 1, 1
    Q_LEN, K_LEN, D_HEAD = 128, 256, 64  # qlen < klen; here, past_len = 256 - 128 = 128.
    TOPK = 1
    BLOCK = 64  # block size used in downsampling
    torch.manual_seed(0)

    # Create inputs.
    q = torch.randn(BATCH, N_HEADS, Q_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, K_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, K_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    sm_scale = 1.0 / (D_HEAD**0.5)

    downsample_factor = BLOCK
    downsample_len = math.ceil(K_LEN / downsample_factor)  # number of blocks along one dimension
    x_ds = torch.randn(BATCH, N_HEADS, downsample_len, downsample_len, device="cuda", dtype=torch.float16)
    # Force the first column to be high so that the first block is always selected.
    x_ds[:, :, :, 0] = 100
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

    kernel = blocksparse_flashattn(BATCH, N_HEADS, Q_LEN, K_LEN, D_HEAD, downsample_len, is_causal=True)
    print(kernel.get_kernel_source())
    tilelang_output = kernel(q, k, v, block_mask.to(torch.int8))

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

    print("ref_output", ref_output)
    print("tilelang_output", tilelang_output)

    # Verify accuracy.
    torch.testing.assert_close(tilelang_output, ref_output, atol=1e-2, rtol=1e-2)

    print("Pass topk sparse attention test with qlen < klen")


def main():
    test_topk_sparse_attention()
    test_topk_sparse_attention_qlen_lt_klen()


def run_regression_perf():
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 4, 2, 256, 64
    TOPK = 2
    BLOCK = 64
    torch.manual_seed(0)

    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)

    downsample_factor = BLOCK
    downsample_len = math.ceil(SEQ_LEN / downsample_factor)
    x_ds = torch.randn([BATCH, N_HEADS, downsample_len, downsample_len], device="cuda", dtype=torch.float16)
    x_ds[:, :, :, 0] = 100
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

    kernel = blocksparse_flashattn(BATCH, N_HEADS, SEQ_LEN, SEQ_LEN, D_HEAD, downsample_len, is_causal=True)
    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(q, k, v, block_mask.to(torch.int8))

    latency_1 = do_bench(run_kernel_only, backend="cupti")

    BATCH, N_HEADS = 1, 1
    Q_LEN, K_LEN, D_HEAD = 128, 256, 64
    TOPK = 1
    BLOCK = 64
    torch.manual_seed(0)

    q = torch.randn(BATCH, N_HEADS, Q_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, K_LEN, D_HEAD, device="cuda", dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, K_LEN, D_HEAD, device="cuda", dtype=torch.float16)

    downsample_factor = BLOCK
    downsample_len = math.ceil(K_LEN / downsample_factor)
    x_ds = torch.randn(BATCH, N_HEADS, downsample_len, downsample_len, device="cuda", dtype=torch.float16)
    x_ds[:, :, :, 0] = 100
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

    kernel = blocksparse_flashattn(BATCH, N_HEADS, Q_LEN, K_LEN, D_HEAD, downsample_len, is_causal=True)
    print(kernel.get_kernel_source())

    def run_kernel_only2():
        kernel(q, k, v, block_mask.to(torch.int8))

    latency_2 = do_bench(run_kernel_only2, backend="cupti")

    return (latency_1 + latency_2) / 2


if __name__ == "__main__":
    main()
