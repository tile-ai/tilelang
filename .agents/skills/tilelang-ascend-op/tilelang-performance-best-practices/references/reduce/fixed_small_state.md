# Iterative Normalization with Fixed Small State

Applicable to multi-round grouped normalization and its backward propagation over a fixed small matrix or tensor.

**Evidence level: implemented and measured.** The structures below have been implemented in real Ascend kernels and have evidence from compilation/execution, correctness, and same-method performance comparisons. This level indicates candidate credibility only; it does not replace ranking by the current bottleneck or revalidation on target cases.

## Optimization Structures

- Keep the entire small state resident in SIMD local memory across multiple computation rounds. Prefer preserving the logical access axes in local dimensions; compare lowering for one-dimensional flattening against multidimensional layouts.
- Compute each normalization group's reciprocal only once and reuse multiplication. Forward does not materialize unused intermediate state; backward saves only snapshots required by the backward pass.
- Place a barrier only at genuine cross-stage read/write dependencies; do not mechanically inherit or remove barriers from an old implementation.
- The register-residency duration, local layout, snapshots, synchronization points, and unrolled code size jointly form the feature fingerprint. Reproducing only some of them does not justify inheriting the complete performance conclusion.

## Validated Structural Combinations

| Path | Implemented Physical Structure |
|---|---|
| Iterative normalization | Keep two-dimensional local state resident throughout all rounds of grouped normalization and compute one reciprocal per group; do not add synchronization without a data dependency before the final writeout |
| Backward recomputation and gradient propagation | Keep forward-recomputed state and backward gradients resident separately, save only snapshots genuinely required by backward, and synchronize only between snapshot production and consumption |

This combination applies only when the complete state can remain resident on chip. For other scales, reevaluate layout, code size, and measured performance.
