# TileLang/PTO Scalar Scheduling and Indexing

Use a Python kernel factory and static DSL structure to reduce runtime scalar work: select shape/dtype/mode/tile/stage before compilation, and keep only actual input dimensions and the task ID dynamic. Use `T.Persistent`, `T.Pipelined`, `T.Parallel`, and `T.Unroll` in hot loops; store common address subexpressions in local variables.

Use generated source code and the timeline as evidence for every optimization. Check for dynamic branches, repeated division and modulo operations, 64-bit addressing, register spills, and code-size growth from unrolling. Do not remove genuine tail handling, masks, or dynamic-shape contracts.

Run targeted correctness tests first. For performance, conduct single-variable A/B tests with identical kernel semantics.
