# FlashAttention Compile-Time Precomputation

## Optimization Goal

Scale, block count, offsets, and mask boundaries are repeatedly computed in the hot KV loop.

## PTO Implementation

Use a Python factory to compute BR, BC, NUM_KV_BLOCKS, scale, and static strides; retain only task-dependent offsets inside the loop. Generate separate causal and full kernels to eliminate runtime mode branches from the hot path.

For FlashAttention, refer to `examples/ascend/flash_attention/example_mha.py`, `examples/ascend/flash_attention/core.py`, and the accompanying `test_mha.py`. Verify QK/PV, online softmax, masking, buffer layouts, and the pipeline against the current source; the presence of an example does not mean it has been validated on the current backend version. After changing the tile, stage, mask, or forwarding, run the corresponding accuracy cases. The mask must be applied before the row maximum is computed.

## Accuracy Gates

Use sufficient precision for precomputed values, especially the scale and log2/exp conversion constants.

Compare against the torch scaled_dot_product_attention reference, covering fully masked rows, the causal diagonal, sequence/block boundaries, extremely large positive and negative logits, repeated maxima, the NaN/Inf contract, and supported dtypes. Keep online state and output accumulation in fp32.

## Performance Gates

Inspect scalar instructions, register usage, and code size in the generated code; account for excessive specialization in the compilation cache.

Record full-kernel latency, TFLOPS, Q/K/V/O GM bytes, Cube/Vector/MTE time, L1/L0/UB usage, and stage. Change only one optimization variable at a time, and compare the relevant full test suite only after all targeted accuracy tests pass.
