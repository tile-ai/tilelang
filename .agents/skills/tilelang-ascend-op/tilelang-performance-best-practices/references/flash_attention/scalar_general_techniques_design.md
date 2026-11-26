# Streamlining Scalar Hot Loops in FlashAttention

## Optimization Goal

Address, loop, and state management in the KV loop consumes significant scalar cycles.

## PTO Implementation

Shorten local-variable lifetimes, reuse static offsets, and avoid constructing complex views inside loops. Express fixed row/vector loops with range/T.Unroll, and confine dynamic sequence handling to the block count and tail block.

For FlashAttention, refer to `examples/ascend/flash_attention/example_mha.py`, `examples/ascend/flash_attention/core.py`, and the accompanying `test_mha.py`. Verify QK/PV, online softmax, masking, buffer layouts, and the pipeline against the current source; the presence of an example does not mean it has been validated on the current backend version. After changing the tile, stage, mask, or forwarding, run the corresponding accuracy cases. The mask must be applied before the row maximum is computed.

## Accuracy Gates

Streamlining must not remove genuine mask, tail-block, or state-update dependencies.

Compare against the torch scaled_dot_product_attention reference, covering fully masked rows, the causal diagonal, sequence/block boundaries, extremely large positive and negative logits, repeated maxima, the NaN/Inf contract, and supported dtypes. Keep online state and output accumulation in fp32.

## Performance Gates

Use generated source and timeline data to measure scalar gaps, and retain a change only when end-to-end latency improves.

Record full-kernel latency, TFLOPS, Q/K/V/O GM bytes, Cube/Vector/MTE time, L1/L0/UB usage, and stage. Change only one optimization variable at a time, and compare the relevant full test suite only after all targeted accuracy tests pass.
