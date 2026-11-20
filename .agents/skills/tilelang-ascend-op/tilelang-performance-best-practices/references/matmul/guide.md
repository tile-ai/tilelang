# PTO GEMM Optimization Index

## Applicability

Applies to bf16/fp16/fp32 and block-scaled GEMM. First establish a correct Cube baseline, then select tiling, pipelining, residency, or shape-specific scheduling according to the bottleneck.

## TileLang/PTO Implementation Flow

Use a Python factory to enumerate BM/BN/BK, MAD tiles, core counts, and stages while constraining L0A/L0B/L0C/L1 capacity and shape divisibility. Use T.AscendTileScheduler for ordinary output tiles; route dynamic cases or tail blocks to a correct fallback.

Reuse `examples/ascend/example_gemm.py` for the baseline implementation: B has the physical layout [N,K], call T.gemm(..., transpose_B=True), use fp32 for L0C, and set clear_accum=True only for the first K tile. The core count for output tiles must not exceed the number of independent tasks; M/N/K tail blocks must use a validated padded-copy path or a dedicated fallback.

## Accuracy Gates

A GEMM path may serve as an optimization baseline only after lowering, finite-value checks, and target-tolerance validation succeed with the current TileLang and repository versions. Do not include historical observations in conclusions without a commit, commands, and raw logs.

Test every input dtype and output dtype separately. Cover tile±1 for M/N/K, long-K cancellation error, positive-negative cancellation, mixtures of large and small magnitudes, zero, and the NaN/Inf contract. Keep both partial results from every K partition and the final reduction in fp32; do not relax tolerances to conceal accumulation-order or output-conversion errors.

## Performance Gates

Compare the baseline, 2/3-stage, full-load, L2-control, and shape-specialized variants in order; change only one variable at a time.

For each candidate, run compilation, targeted PTO accuracy tests, and a standardized benchmark. Report latency, TFLOPS, Cube/MTE2 time, GM bytes, L1/L0/UB usage, core utilization, and generated-code size. Add a candidate to dispatch only when the complete operator is faster end to end with no accuracy regression.
