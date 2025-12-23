# ruff: noqa
import torch
from reference import naive_nsa_simple_inference
import tilelang
from tilelang import language as T
import tilelang.testing

tilelang.testing.set_random_seed(42)


# TODO(lei): workaround, as threads is not divisible by warp group size,
# auto warp specialization may have some bugs.
@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def native_sparse_attention(
    batch,
    heads,
    seq_len,  # Length of K/V sequences (context window size)
    dim,  # Embedding dimension per head
    scale=None,
    block_size=64,  # Tile size for attention computation
    groups=1,  # Grouped query attention (GQA) groups
    selected_blocks=16,  # Number of blocks to select per attention head
):
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    # Modified shapes for inference (q has seq_len=1)a
    q_shape = [batch, 1, heads, dim]  # Changed seq_len to 1
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, 1, head_kv, selected_blocks]  # Changed seq_len to 1
    block_indices_dtype = T.int32
    dtype = T.float16
    accum_dtype = T.float32
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(dim))

    NK = tilelang.cdiv(dim, block_T)
    NV = tilelang.cdiv(dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T
    num_stages = 0
    threads = 32

    @T.prim_func
    def native_sparse_attention(
        Q: T.Tensor(q_shape, dtype),  # [batch, 1, heads, dim]
        K: T.Tensor(kv_shape, dtype),  # [batch, seq_len, head_kv, dim]
        V: T.Tensor(kv_shape, dtype),  # Same shape as K
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),  # Selected block indices
        Output: T.Tensor(q_shape, dtype),  # Output attention tensor
    ):
        with T.Kernel(1, NV, batch * head_kv, threads=threads) as (bx, by, bz):
            # Shared memory allocations for tile storage
            Q_shared = T.alloc_shared([G, BK], dtype)  # Current query block
            K_shared = T.alloc_shared([BS, BK], dtype)  # Current key block
            V_shared = T.alloc_shared([BS, BV], dtype)  # Current value block
            O_shared = T.alloc_shared([G, BV], dtype)  # Output accumulator

            # Attention computation buffers
            acc_s = T.alloc_fragment([G, BS], accum_dtype)  # QK^T scores
            acc_s_cast = T.alloc_fragment([G, BS], dtype)  # Casted scores for softmax
            acc_o = T.alloc_fragment([G, BV], accum_dtype)  # Output accumulator
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_max_prev = T.alloc_fragment([G], accum_dtype)
            scores_scale = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            logsum = T.alloc_fragment([G], accum_dtype)

            i_v, i_bh = by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            # Copy Q for the single position
            T.copy(Q[i_b, 0, i_h * G : (i_h + 1) * G, :], Q_shared)  # Changed i_t to 0

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # Main attention computation loop over selected blocks
            for i in T.Pipelined(NS, num_stages=num_stages):
                i_s = BlockIndices[i_b, 0, i_h, i] * BS  # Get block offset
                if i_s >= 0:  # Skip invalid/padding blocks
                    # Load current key block to shared memory
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    # Compute QK^T attention scores
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # Online softmax with numerical stability
                    # 1. Compute max for scaling
                    # 2. Compute exponentials and sum
                    # 3. Maintain running logsum for normalization
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for i in T.Parallel(G):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(G, BS):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(G):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    # Accumulate attention-weighted values
                    T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Final normalization and output
            for i, j in T.Parallel(G, BV):
                acc_o[i, j] /= logsum[i]  # Normalize by logsum
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, 0, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV])  # Changed i_t to 0

    return native_sparse_attention


def main():
    B, SEQ_LEN, H, HQ, D, S, block_size, dtype = 2, 64, 1, 16, 16, 1, 32, torch.float16
    groups = HQ // H
    SEQ_LEN_Q = 1
    kernel = native_sparse_attention(
        batch=B,
        heads=HQ,
        seq_len=SEQ_LEN,
        dim=D,
        block_size=block_size,
        groups=HQ // H,
        selected_blocks=S,
    )

    Q = torch.randn((B, SEQ_LEN_Q, HQ, D), dtype=dtype, device="cuda").requires_grad_(True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda").requires_grad_(True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda").requires_grad_(True)

    mask = torch.randint(0, 2, (B, SEQ_LEN, groups), device="cuda")
    DO = torch.randn((B, SEQ_LEN_Q, HQ, D), dtype=dtype, device="cuda")

    block_indices = torch.full((B, SEQ_LEN_Q, H, S), SEQ_LEN, dtype=torch.long, device="cuda")
    for b in range(B):
        for t in range(SEQ_LEN_Q):
            for h in range(H):
                i_i = torch.randperm(max(1, (t // block_size)))[:S]
                block_indices[b, t, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    block_counts = torch.randint(1, S + 1, (B, SEQ_LEN_Q, H), device="cuda")

    out = kernel(Q, K, V, block_indices.to(torch.int32))

    ref = naive_nsa_simple_inference(
        q=Q,
        k=K,
        v=V,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
    )
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)


def run_regression_perf():
    B, SEQ_LEN, H, HQ, D, S, block_size, dtype = 2, 64, 1, 16, 16, 1, 32, torch.float16
    groups = HQ // H
    SEQ_LEN_Q = 1
    kernel = native_sparse_attention(
        batch=B,
        heads=HQ,
        seq_len=SEQ_LEN,
        dim=D,
        block_size=block_size,
        groups=HQ // H,
        selected_blocks=S,
    )

    Q = torch.randn((B, SEQ_LEN_Q, HQ, D), dtype=dtype, device="cuda").requires_grad_(True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda").requires_grad_(True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda").requires_grad_(True)
    block_indices = torch.full((B, SEQ_LEN_Q, H, S), SEQ_LEN, dtype=torch.long, device="cuda")
    for b in range(B):
        for t in range(SEQ_LEN_Q):
            for h in range(H):
                i_i = torch.randperm(max(1, (t // block_size)))[:S]
                block_indices[b, t, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(Q, K, V, block_indices.to(torch.int32))

    return do_bench(run_kernel_only, warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    main()
