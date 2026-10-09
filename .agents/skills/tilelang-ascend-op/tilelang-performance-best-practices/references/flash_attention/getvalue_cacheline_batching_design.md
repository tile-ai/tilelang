# Contiguous Batched KV Transfers

## Optimization Objective

Small-fragment or noncontiguous access to K/V blocks causes too many copy transactions.

## PTO Implementation

Choose BC so that K[BC,D] and transposed V[D,BC] form contiguous rectangular `T.copy` operations. Use the validated `transpose=True` path for V. Multiversion adjacent KV blocks by `Pipelined` stage; do not write scalar per-element loads.

For FlashAttention, refer to `examples/ascend/flash_attention/example_mha.py`, `examples/ascend/flash_attention/core.py`, and the accompanying `test_mha.py`. Verify QK/PV, online softmax, masks, buffer layouts, and pipelining against the current source; the existence of an example does not mean that the current round's backend has been validated. After changing tile, stage, mask, or forwarding, run the corresponding correctness cases. Apply the mask before row max.

## Correctness Gates

Validate K/V layouts, head stride, GQA/MQA sharing rules, and KV tail blocks.

Compare against the torch `scaled_dot_product_attention` reference. Cover fully masked rows, the causal diagonal, sequence/block boundaries, extremely positive and negative logits, duplicate maxima, the NaN/Inf contract, and supported dtypes. Keep online state and output accumulation in fp32.

## Performance Gates

Compare copy count, burst behavior, MTE2 time, and L1/L0 pressure after increasing BC.

Record complete kernel latency, TFLOPS, Q/K/V/O GM bytes, Cube/Vector/MTE time, L1/L0/UB occupancy, and stage count. Change only one optimization variable at a time, and compare the relevant complete test suite only after every targeted correctness test passes.
