# GEMM Multistage Pipeline

## Applicability

Use this pattern when there are at least two K-tile iterations and overlap opportunities exist among MTE2, MTE1, and Cube.

## TileLang/PTO Implementation Flow

Set two stages for the A/B L1 buffers with `T.annotate_buffer_versions`, and use `T.Pipelined` for the K loop. If L0A/L0B sub-tiles are explicit, use capacity-constrained `T.Pipelined` in the inner loop as well. Let the PTO scheduler manage dependencies; do not write manual events.

Reuse `examples/ascend/example_gemm.py` as the base implementation: B has physical layout `[N,K]`, calls `T.gemm(..., transpose_B=True)`, uses fp32 L0C, and sets `clear_accum=True` only for the first K tile. The number of cores for output tiles must not exceed the independent-task count. M/N/K tails must use a validated padded copy or dedicated fallback.

## Correctness Gates

The `clear_accum` condition must account for the first iteration of both the outer `kt` and inner `sk` loops.

Test every input dtype and output dtype separately. Cover tile±1 for M/N/K, long-K cancellation error, positive/negative cancellation, mixed large and small values, zero, and the NaN/Inf contract. Keep every K-shard partial and the final reduction in fp32. Do not relax tolerances to conceal accumulation-order or output-conversion errors.

## Performance Gates

Measure two stages first, then three stages. If adding a stage increases L1 pressure and reduces occupancy, retain the shallower pipeline.

For every candidate, run compilation, targeted PTO correctness tests, and the unified benchmark. Report latency, TFLOPS, Cube/MTE2 time, GM bytes, L1/L0/UB occupancy, core utilization, and generated-code size. Admit a candidate into dispatch only when the complete operator is faster end to end without a correctness regression.
