# GLM-5.3 k-pool compression

These examples contain the standalone fused compressor, paged-cache writer,
and rolling decode-tail maintenance used by the GLM-5.3-Flash sparse-attention
indexer:

1. Apply a per-dimension softmax over `slot_score + ape` across each pool.
2. Pool BF16 K vectors with those probabilities.
3. Round the pooled vector through BF16 and apply normalized Hadamard-128.
4. Round through BF16 again and quantize with one FP32 absmax scale per vector.
5. Write the FP8 vector and scale to a caller-owned physical cache location.

During prefill, `glm53_kpool_seed_tail_cache` uses cumulative request boundaries
to copy each request's final four-token rolling window into a caller-owned BF16
tail cache. Each request mapping starts at phase zero and advances by one modulo
the pool size. During plain or speculative decode,
`glm53_kpool_decode_update_and_maybe_write_cache` processes each request's
tokens in position order, updates the tail, and invokes the same compression
math whenever a token closes a pool.

The published GLM-5.3-Flash specialization is `pool_size=4`, `head_dim=128`.
The wrapper also accepts other positive pool sizes for focused testing, but
deliberately rejects other head dimensions.

## Cache contract

The production vLLM kernel packs FP8 values and FP32 scales into one
interleaved `uint8` allocation. TileLang tensors have one element type, so this
example uses two caller-owned tensors instead:

- `k_cache`: `[num_blocks, page_size, 128]`, using the platform-selected E4M3
  dtype (`e4m3fnuz` on pre-gfx950 ROCm and `e4m3fn` on gfx950/CUDA).
- `scale_cache`: `[num_blocks, page_size]`, FP32.
- `tail_cache`: `[num_tail_blocks, 2, 4, 128]`, BF16, with K in lane 0
  and gate scores in lane 1.

`loc` contains flat page-major physical slots. An optional Boolean
`write_mask` disables rows without relying on an invalid address. Before the
launch, the host wrapper rejects incompatible tensors, out-of-range active
locations, and duplicate active locations. Masked locations are not read by
the device kernel and may contain a sentinel such as `-1`.

The decode wrapper requires tokens for each request to be contiguous and
position-ordered. Active requests must own distinct tail blocks. A nonnegative
compressed-cache location is accepted exactly on pool-closing tokens; all
padding and non-closing tokens use negative sentinels. These checks prevent
cross-request tail races and duplicate compressed-cache writes.

This increment intentionally excludes pool Top-K, pool-to-token expansion,
preshuffled framework cache layouts, and vLLM or SGLang integration.

Run the ROCm correctness tests with:

```bash
pytest -q testing/python/amd/test_tilelang_hip_glm53_kpool_compress.py
pytest -q testing/python/amd/test_tilelang_hip_glm53_kpool_decode_tail.py
```
