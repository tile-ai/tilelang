# Coalesced Block-Scale Loading

## Applicability

Applies when scale/bias/LUT data is contiguous along K, each individual segment is too small, and fixed copy overhead is significant.

## TileLang/PTO Implementation Flow

Enlarge the L1 tile for scales and transfer the data for multiple K iterations at once; select the current scale subsegment in the inner loop using a static offset. Refer directly to xsf_l1/wsf_l1 and SF_LOAD_CHUNK_SIZE in `examples/ascend/example_blockscaled_gemm*.py`.

Reuse `examples/ascend/example_gemm.py` for the baseline implementation: B has the physical layout [N,K], call T.gemm(..., transpose_B=True), use fp32 for L0C, and set clear_accum=True only for the first K tile. The core count for output tiles must not exceed the number of independent tasks; M/N/K tail blocks must use a validated padded-copy path or a dedicated fallback.

## Accuracy Gates

The scale for every K block must align exactly with its operand block. Cover K tails and different scale dtypes.

Test every input dtype and output dtype separately. Cover tile±1 for M/N/K, long-K cancellation error, positive-negative cancellation, mixtures of large and small magnitudes, zero, and the NaN/Inf contract. Keep both partial results from every K partition and the final reduction in fp32; do not relax tolerances to conceal accumulation-order or output-conversion errors.

## Performance Gates

Sweep the coalescing factor and record the MTE2 instruction count, effective bandwidth, and L1 usage; do not substitute a fixed 20KB threshold for measurements.

For each candidate, run compilation, targeted PTO accuracy tests, and a standardized benchmark. Report latency, TFLOPS, Cube/MTE2 time, GM bytes, L1/L0/UB usage, core utilization, and generated-code size. Add a candidate to dispatch only when the complete operator is faster end to end with no accuracy regression.
