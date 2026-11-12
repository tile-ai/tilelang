# Indexed Short-Record Transformation

Applicable to gather/scatter, affine combinations, and parameter-gradient reduction over many independent short records.

**Evidence level: implemented and measured.** The structures below have been implemented in real Ascend kernels and have evidence from compilation/execution, correctness, and same-method performance comparisons. This level indicates candidate credibility only; it does not replace ranking by the current bottleneck or revalidation on target cases.

## Optimization Structures

- Pack multiple records along the outer dimension and use rectangular DMA plus SIMD gather/scatter to avoid small per-record accesses.
- When adjacent fields share the same short record, valid region, and index family, process them together in one SIMD traversal whenever possible. If they are split into multiple hot-loop passes, account for repeated indexing, loads/stores, and loop overhead.
- Hoist indices, constants, and loop invariants out of the Persistent hot loop, and confirm from generated code that they are not recomputed per tile.
- When a previous stage has materialized a value from which a derivative or intermediate expression can be recovered, compare the added read against the eliminated recomputation. Complete lowering and measurement in the context of current record transfers, task mapping, and lifetimes; do not decide from input bytes or operation count alone.
- For cross-tile state such as parameter gradients, separately evaluate register permutation/group reduction and UB strided gather + cross-lane reduction. Select based on the accumulator's physical layout, post-loop instructions, correctness, and measurements.
- When partials have compatible consumers, reduction axes, and lifetimes, perform a unified lane reduction after Persistent and pack them into one contiguous partial record. Verify that this genuinely reduces the design to one GM buffer, one copy, and one host reduction over the corresponding dimension.
- Treat index hoisting, intermediate-value reuse, accumulator layout, reduction lowering, partial layout/copy/host-reduction count, and launcher JIT configuration as separate candidates to verify. Implementing one item must not mark the others complete or ineffective in bulk.
- Configure multiversioning only for streaming buffers that genuinely need cross-iteration overlap. Keep accumulation state and single-lifetime data single-version, and A/B test each version combination.
- A static full-tile candidate must jointly verify compile-time extent, elimination of tail masks/branches, index generation, tile size, and core count. Validate with controlled A/B tests or a small combination search. A negative result rejects only the measured combination; retain a general tail path for other shapes.
- Pipeline, fast-math, and large-tile instructions must each pass lowering, correctness, and same-method performance validation independently.

## Applicability Assessment

| Optimization Candidate | Applicable Conditions | Key Checks |
|---|---|---|
| Multi-record coalesced transfer and SIMD | Individual records are short and access is regular; per-record scheduling or small DMA accounts for a significant share | UB capacity, valid tail region, index mapping, and actual DMA count |
| Shared-field hot-loop fusion | Multiple fields share a record, valid region, or index family | Traversal count, repeated indexing and loads/stores, live state, and lowering |
| Loop-invariant hoisting | Indices or constants are invariant in the hot loop | Generation location, hot-loop instructions, and per-tile recomputation count |
| Intermediate-value reuse | An adjacent stage has materialized a value from which the current expression can be recovered | Interface and lifetime, added reads, eliminated recomputation, and combined lowering |
| Accumulator and reduction lowering | Cross-tile accumulation state is short and field grouping is fixed | Lane layout, hot-loop reordering, post-loop instructions, correctness, and measurements |
| Merge partials | Multiple partials have compatible consumers, reduction axes, and lifetimes | Contiguous partial layout and the counts of GM buffers/copies and host reductions |
| Static full tiles and JIT tiling | Some inputs can be proven tail-free, and different scales require different tile/core parameters | Extent, tail handling, index lowering, tile/core combinations, and the general tail path |
| Multistage pipeline | The hot loop contains overlap-capable transfers and computation, with enough iterations to amortize pipeline overhead | Buffer versions, UB occupancy, lowering, and same-method A/B performance |

Rank candidates independently against the current bottleneck. Retain an optimization only after its physical layout, lowering, correctness, and same-method performance are all validated as effective.
