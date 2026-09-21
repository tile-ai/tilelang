# PTO Stream-K Two-Stage Implementation

## Applicability

Use Stream-K when the number of regular M/N tiles is significantly smaller than the number of Cube cores, K is very large, and conventional tile adjustments still cannot provide sufficient parallelism.

## TileLang/PTO Implementation Flow

Kernel A generates fp32 partial results in the workspace for each output-tile×K-partition pair. Kernel B assigns each output tile to a unique owner, which reduces the partial results in a deterministic order and converts the output. A Python host wrapper invokes the two kernels sequentially, without relying on an unverified grid barrier or atomic accumulation.

The baseline implementation reuses `examples/ascend/example_gemm.py`: B has the physical layout `[N, K]`, the implementation calls `T.gemm(..., transpose_B=True)`, L0C uses fp32, and `clear_accum=True` is set only for the first K tile. The number of cores assigned to output tiles must not exceed the number of independent tasks. M/N/K tail tiles must use a validated padded-copy path or a dedicated fallback.

## Correctness Gate

Use fp32 for both the workspace and final accumulation. Evaluate addition-order error separately for each partition count.

Test every input dtype and output dtype separately. Cover tile±1 for M/N/K, cancellation error with large K, positive-negative cancellation, mixed magnitudes, zero, and the NaN/Inf contract. Keep every K-partition partial result and the final reduction in fp32. Do not relax tolerances to conceal accumulation-order or output-conversion errors.

## Performance Gate

Measure both launches and workspace GM traffic end to end. Dispatch this path only when the complete path outperforms regular GEMM; comparing Kernel A alone is insufficient.

For each candidate, run compilation, targeted PTO correctness tests, and the standardized benchmark. Report latency, TFLOPS, Cube/MTE2 time, GM bytes, L1/L0/UB usage, core utilization, and generated-code size. Add a candidate to dispatch only when the complete operator is faster end to end and has no correctness regression.
