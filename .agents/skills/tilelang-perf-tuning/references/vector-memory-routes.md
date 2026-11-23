# Vector and Memory Routes

Applies to elementwise, gather/scatter, transpose/layout, quant, short-record packing, and other vector or irregular-memory-access kernels.

## Work Tree and Tile Boundaries

First express the computation as:

```text
outer item (row/token/segment/patch)
└── inner independent chunk (channel/vector tile/pixel round)
```

If the outer items are fewer than the available cores and the chunks have no reduction, ordering dependency, or overlapping writes, flattening `(item, chunk)` must be a candidate; compare the fixed overhead that overly small tasks would introduce. `T.serial` strided loops and `T.Persistent` are both just candidates; never judge speed by name.

Tile/group candidates come from five classes of real boundaries: the largest fully contiguous semantic unit; SIMD lanes and the effective payload; DMA burst/whole 2D rows/alignment/stride; capacity after accounting for residency, padding, and versions; and the ratio of one-wave, two-wave, steady multi-wave, and tail tiles. Deduplicate into a small set. The failure of one small group only vetoes that value; it cannot close the full-contiguous-unit or capacity-bound candidates.

## Four Physical Routes

| Route | Physical structure | Dominant costs |
|---|---|---|
| `SCALAR_OR_SIMT` | Per-thread independent indexing, branches, and reads/writes | Thread context, scalar addressing, in-flight loads, active threads |
| `MEMORY_INDEXED_VECTOR` | Vector lanes gather/scatter across UB or supported levels | Discrete accesses, index registers, in-flight widening, lane utilization |
| `CONTIGUOUS_LOAD_REGISTER_REORDER` | Contiguous loads followed by register select/shuffle/interleave/pack | Contiguous reads/writes, reorder instructions, cross-register and live state |
| `MATERIALIZED_TRANSFORM` | Materialize planar/transpose/scratch in UB/L1 first, then compute | Extra full reads/writes, capacity, synchronization, materialization passes |

The final arithmetic living in vector registers does not make the whole route register-resident; as soon as the full payload is ever written to a temporary layout, it is a materialized transform.

## Indexed vs. Register Reorder

For every repeatable hot-loop window, record `source_span_bytes`, `consumed_unique_bytes`, density, `window_vregs`, the indexing pattern, and the reuse count.

- The window fits a reasonable register footprint with high density or reuse: prioritize contiguous load + register reorder.
- The span is large, only a few elements are consumed, or a full window would over-fetch or overflow badly: prioritize indexed vector.
- Segmented-contiguous: compare segmented contiguous load + in-segment reorder against gather.
- When native MTE can express the contiguous segments, compare full GM/UB routes against per-lane access.

Verify each intrinsic's input/output dtypes, lanes, parts, widening, predicates, base/index units, and lowering. An implementation that contains redundant conversions, lane fixups, or intermediate materialization only eliminates that full chain; the route must be reopened whenever a lower-cost representative still exists.

When values span multiple registers, remap indices into a local window first, then merge or write out; a global lane index must not be fed directly into a single-register select. A dynamic lane offset and a dynamic intrinsic base are different capabilities: the failure of one dynamic stage base cannot be extrapolated to "all dynamic indexing is unsupported".

## Orthogonal Composition

A candidate identity includes at least `(tile/group, task mapping, physical dataflow, precision, pipeline, tail)`. One axis can change another axis's contiguous payload, fixed cost, or dominant bottleneck, so "baseline tile + new route", "new tile + baseline route", and "new tile + new route" are three separate adjudications. Two single-axis failures cannot replace the combination.

When comparing across routes, fix the other axes where possible; when they cannot be fixed, list the differences explicitly and let the result adjudicate only the full implementation chain.

## Residency, Transfer, and L2

- Reduce GM round trips and repeated reads first; pursue overlap afterwards.
- Weights, scales, small tables, indices, and states that are invariant across iterations should stay resident in a single version.
- Prefer wide stores for contiguous outputs; handle random writes, repeated targets, and atomic semantics separately.
- A/B an L2 policy only when the current API/lowering supports it; inputs, weights, and outputs may need different policies. Revert when the hit rate rises but total time does not fall.
- A large tile simultaneously affects bursts, task count, tails, version footprint, and register pressure; never chase the capacity ceiling alone.

## Pipeline Eligibility and Realization

Create a pipeline candidate only when all of the following hold: profiling shows meaningful time on both MTE and Compute (the overlap itself need not exist yet — introducing it is the candidate's purpose, and the realization proof below verifies it); each core has enough independent full iterations; UB/L1 can hold the versions of every buffer that stays live across iterations; and the dependencies allow the next round's copy in to overlap the previous round's Compute/copy out.

First draw each buffer's `copy in → compute → copy out` lifetime. Inputs, outputs, indices, and temporaries must all be versioned as long as they stay live across iterations; read-only resident data keeps a single version. Full waves use fixed extents; tails are handled separately.

Automatic versioning requires a single logical tile, unique producers/consumers, fixed rank/extent, no hidden aliasing, and every active buffer being recognized. `num_stages > 1`, two buffers, a successful compilation, or a correct run never proves the pipeline took effect.

A failed automatic composition eliminates only that combination. While capacity and the API allow, keep adjudicating explicit stage storage, manual annotations, static stage bodies, and `T.Pipelined` full-wave. Once the automatic path takes full effect, no equivalent manual implementation is needed.

A pipeline's realization must be proven by a stage-1/2 pairing with the same tiles/arithmetic/task mapping: a latency improvement beyond noise, an improvement in effective bandwidth or compute, the matching increase in overlap, and, when necessary, correct version rotation and synchronization in the generated IR. Otherwise record "compiled but not realized".

## Tails and Ordering Stability

- GM accesses valid elements only; UB is allocated by the full instruction footprint.
- Full/tail splitting must not change public semantics or omit the allowed-remainder class.
- Candidates with multiple versions, synchronization, `Persistent`, or compiler warnings cover all cases in one process, and are re-measured with reversed or interleaved case orders.
- Order-dependent drift, sporadic errors, deadlocks, or memory corruption eliminate the exact combination and preserve the evidence.
