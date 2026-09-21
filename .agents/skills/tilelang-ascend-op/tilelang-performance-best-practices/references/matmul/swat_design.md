# Shape-Aware Output-Tile Scheduling

## Applicability

Use this pattern when there are enough M/N output tiles but tail work is imbalanced, or when traversal order affects A/B reuse in L2.

## TileLang/PTO Implementation Flow

Use `T.AscendTileScheduler` as the base. When serpentine traversal or tail-block specialization is required, generate static scheduler variants in the Python factory. Every task must still own exactly one output tile and must not depend on cross-core synchronization.

Reuse `examples/ascend/example_gemm.py` as the base implementation: B has physical layout `[N,K]`, calls `T.gemm(..., transpose_B=True)`, uses fp32 L0C, and sets `clear_accum=True` only for the first K tile. The number of cores for output tiles must not exceed the independent-task count. M/N/K tails must use a validated padded copy or dedicated fallback.

## Correctness Gates

Any reordering must cover every logical tile exactly once. Use a tile-ID bitmap to test for duplicates and omissions.

Test every input dtype and output dtype separately. Cover tile±1 for M/N/K, long-K cancellation error, positive/negative cancellation, mixed large and small values, zero, and the NaN/Inf contract. Keep every K-shard partial and the final reduction in fp32. Do not relax tolerances to conceal accumulation-order or output-conversion errors.

## Performance Gates

Measure the tail wave, overall latency, L2 hit behavior, and inter-core load separately. Do not retain complex scheduling without profiling evidence.

For every candidate, run compilation, targeted PTO correctness tests, and the unified benchmark. Report latency, TFLOPS, Cube/MTE2 time, GM bytes, L1/L0/UB occupancy, core utilization, and generated-code size. Admit a candidate into dispatch only when the complete operator is faster end to end without a correctness regression.
