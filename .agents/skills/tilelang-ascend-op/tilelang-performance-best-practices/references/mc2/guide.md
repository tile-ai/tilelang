# Communication-Compute Combined Optimization in the Current Repository

## Target Implementation

The current PTO DSL does not provide a reusable collective-communication kernel API in this repository. Therefore, use `examples/ascend/example_gemm.py` for computation and the current runtime's official distributed API for communication. A Python host wrapper orchestrates local GEMM, asynchronous collectives, remote GEMM, and streams/events; do not emulate communication or cross-core synchronization in T.prim_func.

## Workflow

1. Validate local GEMM accuracy and latency independently.
2. Validate the collective's tensor layout, rank offset, stream, and completion semantics independently.
3. Establish a non-fused communication→computation or computation→communication reference.
4. Launch asynchronous communication by chunk and execute the corresponding GEMM on an independent stream; use only public runtime APIs.
5. After synchronized multi-rank validation, search the chunk size and local-first arrangement.

Accuracy testing must cover every rank, partial chunks, different world sizes, and an fp32 GEMM accumulator. Performance reporting must include communication latency, computation latency, end-to-end latency, overlap ratio, and inflation. Explicitly reject this optimization when the required runtime API is unavailable.
