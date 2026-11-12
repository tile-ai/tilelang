# PTO FlashAttention Optimization Index

## Optimization Objective

Implement `softmax(QK^T·scale)·V` efficiently without explicitly writing the complete score/probability matrix.

## PTO Implementation

Assign kernel tasks by query block. Keep Q resident and stream KV in blocks. For every KV block, update fp32 `m`, `l`, and the O accumulator, then write back `O/l` at the end. The factory selects BR/BC and stage according to D, sequence, causal mode, and dtype.

For FlashAttention, refer to `examples/ascend/flash_attention/example_mha.py`, `examples/ascend/flash_attention/core.py`, and the accompanying `test_mha.py`. Verify QK/PV, online softmax, masks, buffer layouts, and pipelining against the current source; the existence of an example does not mean that the current round's backend has been validated. After changing tile, stage, mask, or forwarding, run the corresponding correctness cases. Apply the mask before row max.

## Correctness Gates

Validate both the mathematical output and per-row softmax normalization. Do not conceal errors with a lower-precision reference.

Compare against the torch `scaled_dot_product_attention` reference. Cover fully masked rows, the causal diagonal, sequence/block boundaries, extremely positive and negative logits, duplicate maxima, the NaN/Inf contract, and supported dtypes. Keep online state and output accumulation in fp32.

## Performance Gates

A/B test the baseline, tile, stage, mask specialization, and Q residency separately.

Record complete kernel latency, TFLOPS, Q/K/V/O GM bytes, Cube/Vector/MTE time, L1/L0/UB occupancy, and stage count. Change only one optimization variable at a time, and compare the relevant complete test suite only after every targeted correctness test passes.
