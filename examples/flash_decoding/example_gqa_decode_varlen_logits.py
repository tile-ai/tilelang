import torch
import math
import argparse
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

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
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bid, hid, bz):
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
            S_shared = T.alloc_shared([block_H, math.ceil(max_seqlen_kv / block_N)], accum_dtype)
            S_shared_cast = T.alloc_shared([block_H, math.ceil(max_seqlen_kv / block_N)], dtype)
            s_aux_shared = T.alloc_shared([block_H], T.float32)

            cur_kv_head = hid // (kv_group_num // valid_block_H)

            cur_start_k = cu_seqlens_k[bid]
            cur_end_k = cu_seqlens_k[bid + 1]
            cur_seqlen_k = cur_end_k - cur_start_k

            T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((cur_seqlen_k // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[cur_start_k + k * block_N : cur_start_k + (k + 1) * block_N, cur_kv_head, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(k * block_N + j < cur_seqlen_k, acc_s[i, j], -T.infinity(accum_dtype))
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                T.copy(scores_max, S_shared[:, k])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
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
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(O_shared, Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])
            T.copy(S_shared, S_shared_cast)
            T.copy(S_shared_cast[:valid_block_H, :], S[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :])

    return flashattn_gqa_decode_no_split


def ref_attention(q, k, v, k_seqlens, q_heads, sink=None):
    """
    Compute reference attention output and weights.
    Args:
        q: [b, q_heads, head_size]
        k, v: [b, kv_heads, max_seqlen, head_size]
        k_seqlens: [b] actual sequence lengths
        sink: [q_heads] optional sink values
    Returns: output [b, q_heads, head_size], attn_weights [b, q_heads, max_seqlen]
    """
    batch_size, kv_heads, max_seqlen, head_size = k.shape
    softmax_scale = 1.0 / math.sqrt(head_size)

    # Expand KV heads and compute attention scores
    k = repeat_kv(k, q_heads // kv_heads)
    v = repeat_kv(v, q_heads // kv_heads)
    logits = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) * softmax_scale  # [b, q_heads, 1, max_seqlen]

    # Mask invalid positions
    mask = torch.arange(max_seqlen, device=q.device).expand(batch_size, -1) >= k_seqlens.unsqueeze(1)
    logits.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    if sink is None:
        attn_weights = logits.softmax(dim=-1)
    else:
        # Sink attention: softmax with additional sink term
        sink_expanded = sink.view(1, q_heads, 1, 1)
        logits_max = torch.maximum(logits.max(dim=-1, keepdim=True).values, sink_expanded)
        exp_logits = torch.exp(logits - logits_max)
        attn_weights = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + torch.exp(sink_expanded - logits_max))

    attn_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(2), 0.0)
    output = torch.matmul(attn_weights.to(v.dtype), v).squeeze(2)
    return output, attn_weights.squeeze(2)


def test_varlen_decode_main(args):
    """Test decode kernel with variable sequence lengths."""
    batch_size, q_heads, kv_heads = args.batch_size, args.q_heads, args.kv_heads
    max_k_seqlen, head_size, block_size = args.k_seqlen, args.head_size, args.block_size
    dtype = torch.bfloat16 if args.dtype == T.bfloat16 else torch.float16

    # Generate variable length sequences and cumulative lengths
    k_seqlens = torch.randint(max_k_seqlen // 4, max_k_seqlen + 1, size=(batch_size,))
    cu_seqlens_k = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
    cu_seqlens_k[1:] = torch.cumsum(k_seqlens, dim=0).to(torch.int32).cuda()
    total_k_tokens = cu_seqlens_k[-1].item()

    # Generate input tensors
    q = torch.randn(batch_size, q_heads, head_size, device="cuda", dtype=dtype)
    k_varlen = torch.randn(total_k_tokens, kv_heads, head_size, device="cuda", dtype=dtype)
    v_varlen = torch.randn(total_k_tokens, kv_heads, head_size, device="cuda", dtype=dtype)
    sink = torch.randn(q_heads, device="cuda", dtype=torch.float32) * 0.1 if args.test_sink else None

    # Run tilelang kernel
    tilelang.disable_cache()
    tl_kernel = flashattn(batch_size, q_heads, kv_heads, max_k_seqlen, total_k_tokens, head_size, args.test_sink)
    O_tl, S_tl = tl_kernel(q, k_varlen, v_varlen, cu_seqlens_k, sink)
    S_tl = torch.max_pool2d(S_tl, kernel_size=(q_heads, 1), stride=(q_heads, 1))

    # Mask out invalid S positions
    for i in range(batch_size):
        valid_blocks = math.ceil(k_seqlens[i].item() / block_size)
        S_tl[i, :, valid_blocks:] = 0

    # Prepare padded tensors for reference
    actual_max = int(k_seqlens.max())
    k_padded = torch.zeros(batch_size, kv_heads, actual_max, head_size, device="cuda", dtype=dtype)
    v_padded = torch.zeros(batch_size, kv_heads, actual_max, head_size, device="cuda", dtype=dtype)
    for i in range(batch_size):
        seq_len = k_seqlens[i].item()
        k_padded[i, :, :seq_len] = k_varlen[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1)
        v_padded[i, :, :seq_len] = v_varlen[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1)

    # Compute reference
    O_ref, attn_weights = ref_attention(q, k_padded, v_padded, k_seqlens.cuda(), q_heads, sink)
    S_ref = torch.max_pool2d(attn_weights, kernel_size=(q_heads, block_size), stride=(q_heads, block_size), ceil_mode=True).to(dtype)

    # Compare results
    num_blocks = math.ceil(actual_max / block_size)
    assert torch.allclose(O_tl, O_ref, atol=1e-2, rtol=1e-2), f"Output mismatch: {(O_tl - O_ref).abs().max()}"
    assert torch.allclose(S_tl[:, :, :num_blocks], S_ref[:, :, :num_blocks], atol=1e-2, rtol=1e-2), "Score mismatch"
    print("✅ All tests passed!")


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
    sink = torch.randn(q_heads, device="cuda", dtype=torch.float32) * 0.1 if args.test_sink else None
    if args.test_varlen:
        print(f"  K sequence lengths: {k_seqlens.tolist()}")

    _, q_h, head_size = q_decode.shape
    batch = cu_seqlens_k.size(0) - 1
    k_h = k_varlen.size(1)
    tl_kernel = flashattn(batch, q_h, k_h, args.k_seqlen, cu_seqlens_k[-1].item(), head_size, args.test_sink)

    def run_once():
        tl_kernel(q_decode, k_varlen, v_varlen, cu_seqlens_k, sink)

    # Benchmark
    print("⚡ Benchmarking Tilelang kernel (100 iterations)...")
    tilelang_time = do_bench(
        run_once,
    )
    print(f"Average decode kernel time Tilelang: {tilelang_time:.3f} ms")


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

    # if args.benchmark:
    #     speed_benchmark_decode_comparison(args)
    # else:
    #     test_varlen_decode_main(args)

    speed_benchmark_decode_comparison(args)
