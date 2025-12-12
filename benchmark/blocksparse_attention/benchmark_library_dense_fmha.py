# ruff: noqa
import torch
from tilelang.profiler import do_bench


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


def benchmark_topk_sparse_attention():
    from benchmark_configs import configs

    torch.manual_seed(0)

    # Config
    for BATCH, N_HEADS, SEQ_LEN, D_HEAD, TOPK, BLOCK in configs:
        # Create inputs
        q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
        k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)
        v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device="cuda", dtype=torch.float16)

        import flash_attn

        def benchmark_fn():
            flash_attn.flash_attn_func(q, k, v, causal=True)

        ref_latency = do_bench(
            benchmark_fn,
            warmup=10,
            rep=100,
        )
        print(
            f"BATCH: {BATCH}, N_HEADS: {N_HEADS}, SEQ_LEN: {SEQ_LEN}, D_HEAD: {D_HEAD}, TOPK: {TOPK}, BLOCK: {BLOCK}, ref_latency: {ref_latency}"
        )


if __name__ == "__main__":
    benchmark_topk_sparse_attention()
