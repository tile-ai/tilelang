# Reducing FlashAttention Scalar-Memory Pressure

## Optimization Objective

UB loads/stores of scalar `m`/`l`/`alpha` state create gaps in Vector execution.

## PTO Implementation

Organize contiguous `m`/`l`/`alpha` arrays in a SIMD-broadcastable form, process rows in pairs or groups, and reuse register values within the same update stage. Resident UB state must be written back across KV blocks; do not assume that registers remain live across pipeline iterations.

For FlashAttention, refer to `examples/ascend/flash_attention/example_mha.py`, `examples/ascend/flash_attention/core.py`, and the accompanying `test_mha.py`. Verify QK/PV, online softmax, masks, buffer layouts, and pipelining against the current source; the existence of an example does not mean that the current round's backend has been validated. After changing tile, stage, mask, or forwarding, run the corresponding correctness cases. Apply the mask before row max.

## Correctness Gates

Strictly isolate state for every row, and validate odd row-block counts and the final half-block.

Compare against the torch `scaled_dot_product_attention` reference. Cover fully masked rows, the causal diagonal, sequence/block boundaries, extremely positive and negative logits, duplicate maxima, the NaN/Inf contract, and supported dtypes. Keep online state and output accumulation in fp32.

## Performance Gates

Compare state load/store count, Vector busy time, and register pressure; avoid spills caused by large unroll factors.

Record complete kernel latency, TFLOPS, Q/K/V/O GM bytes, Cube/Vector/MTE time, L1/L0/UB occupancy, and stage count. Change only one optimization variable at a time, and compare the relevant complete test suite only after every targeted correctness test passes.
