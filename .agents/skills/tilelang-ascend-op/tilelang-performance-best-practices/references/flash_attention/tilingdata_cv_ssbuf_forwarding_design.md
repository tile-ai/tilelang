# Cube-Vector Buffer Forwarding

## Optimization Goal

Pass the fp32 QK scores to Vector softmax and then return probabilities to Cube PV, minimizing unnecessary GM traffic and intermediate copies.

## PTO Implementation

Use T.dual_copy from L0C→UB, perform SimdVF softmax in UB, use T.dual_copy from UB→L1, and then execute T.gemm(P,V). Specialize BR/BC and buffer sizes in the Python factory.

For FlashAttention, refer to `examples/ascend/flash_attention/example_mha.py`, `examples/ascend/flash_attention/core.py`, and the accompanying `test_mha.py`. Verify QK/PV, online softmax, masks, buffer layouts, and pipelining against the current source. The existence of an example does not mean it has been validated for the current backend revision. After changing tiles, stages, masks, or forwarding, run the corresponding correctness cases. Apply the mask before the row maximum.

## Correctness Gate

Complete max, exp, and sum in fp32 before converting probabilities to bf16. Evaluate the effect of conversion error on the final O.

Compare against the torch scaled_dot_product_attention reference. Cover fully masked rows, the causal diagonal, sequence/block boundaries, extremely positive and negative logits, repeated maxima, the NaN/Inf contract, and supported dtypes. Keep online state and output accumulation in fp32.

## Performance Gate

Measure L0C/UB/L1 transfers and synchronization gaps. Increase buffer sizes only when the overlap benefit exceeds the capacity cost.

Record complete kernel latency, TFLOPS, Q/K/V/O GM bytes, Cube/Vector/MTE time, L1/L0/UB usage, and stage count. Change only one optimization variable at a time, and compare the complete relevant test suite only after every targeted correctness test passes.
