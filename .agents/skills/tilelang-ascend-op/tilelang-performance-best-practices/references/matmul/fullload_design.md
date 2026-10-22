# GEMM Single-Side Residency

## Applicability

Use this pattern when a complete reusable panel of A or B, together with its scale, fits in L1 and can be reused by multiple output tiles on the opposite side.

## TileLang/PTO Implementation Flow

Move the resident panel's `T.copy` outside output-tile traversal; continue transferring the streaming side by K tile. The scheduler's traversal order should keep the reused side unchanged across consecutive tasks. The L1 budget includes the resident panel, scale, streaming buffer versions, and a safety margin.

Reuse `examples/ascend/example_gemm.py` as the base implementation: B has physical layout `[N,K]`, calls `T.gemm(..., transpose_B=True)`, uses fp32 L0C, and sets `clear_accum=True` only for the first K tile. The number of cores for output tiles must not exceed the independent-task count. M/N/K tails must use a validated padded copy or dedicated fallback.

## Correctness Gates

The indexing, block size, and application order of a quantization scale must match the block-scaled reference.

Test every input dtype and output dtype separately. Cover tile±1 for M/N/K, long-K cancellation error, positive/negative cancellation, mixed large and small values, zero, and the NaN/Inf contract. Keep every K-shard partial and the final reduction in fp32. Do not relax tolerances to conceal accumulation-order or output-conversion errors.

## Performance Gates

Compare saved GM bytes against resident-initialization cost; do not enable residency when the reuse count is less than two.

For every candidate, run compilation, targeted PTO correctness tests, and the unified benchmark. Report latency, TFLOPS, Cube/MTE2 time, GM bytes, L1/L0/UB occupancy, core utilization, and generated-code size. Admit a candidate into dispatch only when the complete operator is faster end to end without a correctness regression.
