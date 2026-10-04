# GEMM Early Copy Issuance and Deep Pipelining

## Applicability

Use this approach when the basic 2-stage pipeline is correct, but the generated code starts copying the next K tile later than the earliest valid opportunity.

## TileLang/PTO Implementation Process

Express prefetching by first increasing T.Pipelined stages, adjusting buffer versions, and restructuring loop nesting. Retain a configuration only when generated source and the timeline prove that the copy moved earlier. Do not simulate low-level events in the DSL.

Reuse the baseline implementation in examples/ascend/example_gemm.py: B uses physical layout [N,K], the implementation calls T.gemm(..., transpose_B=True), L0C uses fp32, and clear_accum=True is set only for the first K tile. The core count for output tiles must not exceed the number of independent tasks. M/N/K tail tiles must use a validated padded-copy path or a dedicated fallback.

## Correctness Gate

A prefetched version must not overwrite an L1 tile that T.gemm is still consuming.

Test every input dtype and output dtype separately. Cover tile±1 on M/N/K, long-K accumulation error, positive/negative cancellation, mixtures of large and small magnitudes, zero, and the NaN/Inf contract. Keep partial results from every K partition and the final reduction in fp32. Do not hide accumulation-order or output-conversion errors by widening tolerances.

## Performance Gate

Verify actual earlier issuance in generated code and the timeline. Roll back if memory usage increases without shortening the gap.

For every candidate, run compilation, targeted PTO correctness tests, and the standardized benchmark. Report latency, TFLOPS, Cube/MTE2 time, GM bytes, L1/L0/UB usage, core utilization, and generated-code size. Add a candidate to dispatch only when the complete operator is faster end to end without a correctness regression.
