# GLM-5.3 k-pool compression

This example contains the standalone fused compressor and paged-cache writer
used by the GLM-5.3-Flash sparse-attention indexer:

1. Apply a per-dimension softmax over `slot_score + ape` across each pool.
2. Pool BF16 K vectors with those probabilities.
3. Round the pooled vector through BF16 and apply normalized Hadamard-128.
4. Round through BF16 again and quantize with one FP32 absmax scale per vector.
5. Write the FP8 vector and scale to a caller-owned physical cache location.

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

`loc` contains flat page-major physical slots. An optional Boolean
`write_mask` disables rows without relying on an invalid address. Before the
launch, the host wrapper rejects incompatible tensors, out-of-range active
locations, and duplicate active locations. Masked locations are not read by
the device kernel and may contain a sentinel such as `-1`.

This first increment intentionally excludes decode-tail maintenance, pool
Top-K, pool-to-token expansion, preshuffled framework cache layouts, and vLLM
or SGLang integration.

Run the ROCm correctness tests with:

```bash
pytest -q testing/python/amd/test_tilelang_hip_glm53_kpool_compress.py
```
