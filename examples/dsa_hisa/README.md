# tilelang_kernels — hisa prefill pipeline

Tilelang prefill implementation of **hisa** (HIerarchical Sparse Attention).
Paper: <https://arxiv.org/pdf/2603.28458>.

## What is HISA?

HISA optimizes DeepSeek sparse attention by a plug-and-play replacement
for the indexer that rewrites the search path from a flat token scan into
a two-stage hierarchical procedure.

**Stage 1 — coarse block-level selection.** Group K tokens into pool blocks
of `k_block_size` tokens, mean-pool each block, then score each query
against all pool blocks and pick the top `block_topk` blocks per query.

**Stage 2 — fine-grained token-level scoring.** For each query, run a
full-resolution MQA over the raw tokens inside its selected blocks, then
pick the top `topk_tokens` tokens per query.

## Files

| file | step | role |
|---|---|---|
| `fp8_block_mean_pooling.py` | 1.1 | Mean-pool raw K into pool blocks (fp8 + per-block f32 scale) |
| `pool_mqa_fp8.py`           | 1.2 | fp8×fp8 score `Q · pooled_K` → one logit per (query, pool block) |
| `clean_and_maintain_logits.py` | 1.3 | In-place mask on stage-1 logits: -inf outside per-query range, +inf at first/last valid block |
| `block_sparse_mqa_fp8.py`   | 2.1 | fp8×fp8 fine-grained score over the raw tokens of the `block_topk` selected blocks |
| `hisa.py`                   | —   | End-to-end orchestration: all four kernels + the two `torch.topk` steps + the index-translation post-processing |

Each per-kernel file has one `test_*` entry that (a) runs the kernel +
torch ref, (b) asserts via `torch.testing.assert_close`, (c) prints the
latency of the kernel. `hisa.py` has `test_hisa` that runs the full
pipeline, checks the output-index mask invariant, and prints end-to-end
latency.

## Per-kernel reference

### 1.1 `fp8_block_mean_pooling.py`

**Function**: `fp8_native_block_mean_pooling`

**Meaning**: flat per-block mean of the chunk's K tokens, re-quantized to
fp8 with a per-block f32 scale. Groups `N` K tokens into
`ceildiv(N, k_block_size)` pool blocks.

**Interface**:
```python
blocked_k, blocked_k_scale = fp8_native_block_mean_pooling_interface(
    k,           # [N, D] fp8
    k_scale,     # [N] f32  — per-token scale from indexer_k_quant_and_cache
    k_block_size,
)
# blocked_k:       [num_blocks, D] fp8
# blocked_k_scale: [num_blocks]    f32
```

**What it does**: per pool block `b` of size `kb = k_block_size`,
1. dequantize each of the `kb` tokens: `k_f[i] = k_fp8[i] * k_scale[i]`
2. average across the block in f32: `mean = sum_i k_f[i] / kb` (or the
   actual valid count for the ragged tail block)
3. re-quantize the f32 mean to fp8 with a per-block scale
   `block_scale = max(max_abs(mean) / 448, 1e-10)`, writing
   `blocked_k[b] = fp8(mean / block_scale)` and `blocked_k_scale[b] = block_scale`.

### 1.2 `pool_mqa_fp8.py`

**Function**: `pool_mqa_attn_return_logits_fp8`

**Meaning**: coarse-grained fp8 multi-query attention over the **pooled** K
(one vector per pool block). Produces one logit per (query, pool-block).

**Interface**:
```python
block_k_score = pool_mqa_attn_return_logits_fp8_interface(
    q_fp8,                    # [M, H, D] fp8
    blocked_kv_fp8,           # [Nb, D]   fp8     (from step 1.1)
    blocked_kv_scale,         # [Nb]      f32     (from step 1.1)
    weights_f32,              # [M, H]    f32
    cu_seqlen_blocked_ks,     # [M] int32 — per-query start in pool-block coords
    cu_seqlen_blocked_ke,     # [M] int32 — per-query end   in pool-block coords
)
# block_k_score: [M, Nb] f32
```

**What it does**: for each query `m` and each pool block `n` in
`[cu_seqlen_blocked_ks[m], cu_seqlen_blocked_ke[m])`,
```
block_k_score[m, n] = sum_h ReLU(q[m, h] · blocked_k[n]) * blocked_k_scale[n] * weights[m, h]
```
Uses tile-level fp8×fp8→f32 Tensor Core GEMM; the per-block scale is
applied post-GEMM. The kernel processes queries in tiles of size
`block_Q × block_N` and **writes the union of the tile's queries' visible
K ranges** — entries outside an individual query's range inside that
union still carry raw dot-product values (they will be masked by
step 1.3 next). Entries outside the tile union are left at their
zero-init value.

### 1.3 `clean_and_maintain_logits.py`

**Function**: `clean_and_maintain_logits_`

**Meaning**: in-place post-kernel mask on the stage-1 logits.

**Interface**:
```python
clean_and_maintain_logits_interface(
    logits,        # [M, Nb] f32 — stage-1 output; modified in place
    cu_seqlen_ks,  # [M] int32 — per-row start (inclusive)
    cu_seqlen_ke,  # [M] int32 — per-row end   (exclusive)
)
```

**What it does**: for each row `m`,
- positions outside `[cu_seqlen_ks[m], cu_seqlen_ke[m])` → set to `-inf`
  (so `torch.topk` ignores them),
- positions `cu_seqlen_ks[m]` and `cu_seqlen_ke[m] - 1`      → set to `+inf`
  (force-maintain the boundary blocks: they are always picked by the
  subsequent top-block selection — a standard hisa trick to preserve
  sink and local blocks).

### 2.1 `block_sparse_mqa_fp8.py`

**Function**: `fp8_native_block_sparse_mqa_attn_return_logits`

**Meaning**: fine-grained fp8 MQA over only the **raw K tokens** inside the
top-`block_topk` pool blocks selected per query. Two kernel variants are
auto-dispatched by the factory:
- general (`kv_block_size > block_N`): pipelined sub-block inner loop
- small-pooling-size (`kv_block_size == block_N`): single pass, no pipeline

**Interface**:
```python
block_sparse_logits = fp8_native_block_sparse_mqa_attn_return_logits_interface(
    q,                  # [M, H, D] fp8
    k,                  # [N, D]    fp8
    k_scale,            # [N]       f32
    topk_block_index,   # [M, block_topk] int64 — from torch.topk over stage-1 scores
    kv_block_size,      # == k_block_size
    weights,            # [M, H] f32
    cu_seqlen_ks,       # [M] int32 — per-query K start (absolute, in raw tokens)
    cu_seqlen_ke,       # [M] int32 — per-query K end
)
# block_sparse_logits: [M, block_topk * kv_block_size] f32
```

**What it does**: for each query `m`, for each selected block
`t ∈ [0, block_topk)` with `blk = topk_block_index[m, t]`, for each
in-block offset `i ∈ [0, kv_block_size)`,
```
k_abs = blk * kv_block_size + i
if k_abs ∉ [cu_seqlen_ks[m], cu_seqlen_ke[m]) or k_abs >= N:
    block_sparse_logits[m, t * kv_block_size + i] = -inf
else:
    block_sparse_logits[m, t * kv_block_size + i] =
        sum_h ReLU(q[m, h] · k[k_abs]) * k_scale[k_abs] * weights[m, h]
```
The out-of-range mask is written directly by this kernel — no separate
mask pass is needed here (unlike stage 1).

### End-to-end `hisa.py`

**Function**: `hisa_indexer`

**Meaning**: single entry point that runs the full pipeline below.

**Interface**:
```python
topk_indices = hisa_indexer(
    q,                # [M, H, D] fp8
    k,                # [N, D]    fp8
    k_scale,          # [N]       f32
    weights,          # [M, H]    f32
    cu_seqlen_ks,     # [M]       int32 — per-query K start
    cu_seqlen_ke,     # [M]       int32 — per-query K end
    *,
    k_block_size,     # pool block size (=128 in DeepSeek-V3.2)
    block_topk,       # number of top pool blocks kept per query
    topk_tokens,      # final top-k size handed to the sparse attention
)
# topk_indices: [M, topk_tokens] int32 — each row is the query's top-k K
# positions expressed as offsets within its own [cu_ks, cu_ke) window.
# Out-of-range slots are -1.
```

**Pipeline**:

```
(1.1) fp8_native_block_mean_pooling            K, k_scale → blocked_k, blocked_k_scale
(1.2) pool_mqa_attn_return_logits_fp8          Q × blocked_k → block_k_score[M, Nb]
(1.3) clean_and_maintain_logits                in-place mask (-inf/+inf) on block_k_score
(1.4) torch.topk(block_k_score.bfloat16(),     → topk_block_indices[M, block_topk] int64
                 k=block_topk, sorted=False)
(2.1) fp8_native_block_sparse_mqa_…            Q × K[selected] → block_sparse_logits
                                                  [M, block_topk * k_block_size]
(2.2) torch.topk(block_sparse_logits,          → relevant_topk_indices[M, topk_tokens] int64
                 k=topk_tokens)
(2.3) (Python) gather topk_block_indices +     → absolute K positions, then subtract
      arith + subtract cu_seqlen_ks + mask        cu_seqlen_ks for per-query-relative offsets
                                                  → topk_indices[M, topk_tokens] int32
```
