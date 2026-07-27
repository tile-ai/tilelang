# ruff: noqa
import math
import torch

import torch.nn.functional as F
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

        sm_scale = 1.0 / (D_HEAD**0.5)

        # Create sparse mask (downsampled to block level)
        downsample_factor = BLOCK
        downsample_len = math.ceil(SEQ_LEN / downsample_factor)
        x_ds = torch.randn([BATCH, N_HEADS, downsample_len, downsample_len], device="cuda", dtype=torch.bfloat16)
        x_ds[:, :, :, 0] = 100
        block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

        def benchmark_fn():
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
            return ref_output

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
