# Operation, Pipeline, and Layout Semantic Rules

## Contents

1. [TileOp contracts](#tileop-contracts)
2. [Software-pipeline contracts](#software-pipeline-contracts)
3. [Reducer contracts](#reducer-contracts)
4. [Layout and distribution contracts](#layout-and-distribution-contracts)
5. [Kernel and target contracts](#kernel-and-target-contracts)
6. [Type and value contracts](#type-and-value-contracts)
7. [Current coverage and priorities](#current-coverage-and-priorities)
8. [Boundary tests](#boundary-tests)

## TileOp contracts

Give every TileOp an explicit semantic contract covering:

- accepted operand kinds (`Buffer`, `BufferLoad`, `BufferRegion`, pointer);
- logical rank, shape, and region extents;
- operand, accumulator, and result dtypes;
- required storage scopes and address spaces;
- read, write, and opaque effect regions;
- legal source/destination aliasing and overlap;
- initialization/accumulation behavior;
- participant/uniformity requirements;
- target-independent semantics and target-specific capabilities.

Apply representative rules:

- `T.copy`: require compatible regions unless the op explicitly defines
  broadcast, padding, gather/scatter, or overlap behavior.
- `T.gemm` and block-scaled variants: validate M/N/K, transpose interpretation,
  batch dimensions, accumulator shape/dtype, operand scopes, scale layout, and
  clear/accumulate semantics.
- `T.clear`/`T.fill`: require a writable destination region and a convertible
  value dtype.
- reductions/scans: validate axis/rank, source/destination shape, identity,
  dtype, and in-place support.
- atomics: use the memory/concurrency contract for dtype, alignment, scope,
  ordering, and return-value semantics.

Do not use Python `assert` for new user-facing checks. Raise `TypeError` for an
invalid operand kind and `ValueError` for an invalid value/shape combination;
include operand names and expected versus actual values. Convert existing
assertions opportunistically when touching an API.

Run frontend checks when Python has exact shapes/types. Repeat critical
invariants in native operator parsing when imported or generated TIR can bypass
the frontend.

## Software-pipeline contracts

For each requested pipeline, require:

- `order` and `stage` metadata to be present together and aligned with the
  schedulable statement list;
- stage/order/group arrays to use valid values and consistent lengths;
- replayable scalar binds to be pure and independent of buffers written by the
  pipeline;
- scalar and buffer dependencies to respect the proposed schedule;
- no conflicting overlapping writes to one buffer version;
- multi-versioned buffer allocation and stage indexing to agree;
- explicit barriers and async groups to use the pipeline's depth and phase;
- a conditional producer/consumer schedule to preserve definition and wait
  semantics.

Classify failures carefully:

- malformed metadata or an impossible dependency schedule is invalid;
- nested pipeline-requested loops are legal at the language level; accept or
  reject them according to the selected backend's hierarchical-pipeline
  capability;
- a requested pipeline with no profitable overlap is valid and may fall back
  to serial execution with a warning;
- a backend without validated async pipelining should produce a target-specific
  fallback/unsupported diagnostic, not a universal semantic error.

Move checks earlier only when source statement counting is stable. Replayable
bind normalization changes the schedulable list, so preserve compatibility with
both accepted annotation forms where the injector already does so.

## Reducer contracts

Preserve the existing first-class reducer lifecycle:

1. allocate one reducer state;
2. initialize exactly one static epoch site;
3. perform zero or more updates between init and finalize;
4. update only through `reducer_update` in `T.Parallel`;
5. finalize exactly once in a compatible control-flow scope;
6. use an ordinary destination buffer with a compatible dtype;
7. forbid ordinary loads, stores, clear/fill/copy, aliasing, and pointer escape
   of reducer state;
8. reject reducer constructs that survive materialization.

Keep init/finalize collective execution uniform. Distinguish a skipped finalize
that leaves an unread partial from a path that reads an unfinalized or stale
value.

Treat `VerifyReducerEpoch` and `VerifyReducerConsumed` as the model for strong
lifecycle diagnostics: validate early against user-written structure, then
assert that no semantic marker survives its consuming lowering pass.

## Layout and distribution contracts

Validate layout structure without imposing false equality requirements:

- require a parallel-loop layout annotation on the outermost consecutive
  parallel loop after inference;
- forbid independent layout annotations on inner loops of the same parallel
  nest;
- require layout input dimension to match the parallel-nest depth;
- require buffer rank and layout logical rank to be compatible;
- require Fragment thread extent/range to fit the kernel or
  warp-specialization participant range selected for that use;
- validate logical index domains against layout input domains and guarded tails;
- require stores through replicated fragments to select one owner or use the
  canonical replica-zero guard;
- require injectivity only when the consumer contract needs unique ownership;
  replication and reductions may intentionally be non-injective;
- preserve address-space and swizzle semantics through layout conversion;
- treat failed TileLang-to-CuTe conversion as unsupported input unless the
  semantic contract itself is violated.

Do not require loop extent to equal launched thread count or explicit layout
shape. Loop partitioning, replication, thread ranges, and tail guards make
those differences legal. Use `$tilelang-layout` and the layout verification
harness before changing these rules.

## Kernel and target contracts

Separate target-independent kernel structure from target capability:

- require integer, positive grid/block extents where statically known;
- enforce dimensionality limits and prohibit malformed/nested launch regions;
- require unique and compatible thread bindings per launch dimension;
- validate cluster dimensions, masks, and cluster-shared operations together;
- require cooperative launch support for grid-wide synchronization;
- validate target limits such as threads per block, shared-memory capacity,
  alignment, and instruction availability as target-specific support errors;
- require warp/warp-group collectives to use compatible participant geometry;
- preserve pointer address spaces through host/device split and codegen.

Do not label a missing TMA, WGMMA, MFMA, cluster, or vector instruction as a
language-semantic error when a legal fallback exists. Select the fallback or
emit a precise target-support diagnostic.

## Type and value contracts

Require:

- integer, nonnegative shape/region extents when statically known;
- nonzero loop steps and legal annotation values;
- BufferStore values convertible under the language's explicit cast rules;
- vector lane counts and lane extraction indices to be valid;
- reinterpret/view operations to preserve total storage size and alignment;
- atomics, reducers, scans, and MMA operands to use supported dtypes;
- stochastic rounding parameters only with the corresponding rounding mode;
- memory-order and scope strings/enums to remain valid through lowering.

Distinguish implicit numeric conversion permitted by TileLang from reinterpret
casts that change storage interpretation. Reject silent address-space or packed
storage changes.

## Current coverage and priorities

| Area | Current state | Next step |
|---|---|---|
| TileOp operands/shapes | Many frontend checks, often `assert`; native builders also check | Normalize errors and define per-op contracts |
| Pipeline dependencies | Strong late planning/injection checks | Add source-friendly metadata and backend capability diagnostics |
| Reducer lifecycle | Strong early and post-consumption verification | Preserve and reuse this pattern |
| Parallel layout annotations | Structural post-inference validation | Add thread-range and ownership compatibility checks |
| Kernel launch arguments | Basic frontend validation | Add target capability diagnostics |
| Dtype/value checks | Distributed across APIs/codegen | Centralize reusable predicates and diagnostics |

Prioritize consistent TileOp contracts and user diagnostics after the
memory/concurrency safety work. Keep target support and optimization fallback
separate from universal legality.

## Boundary tests

For TileOps:

- wrong rank/shape/dtype/scope versus the nearest supported combination;
- legal aliasing versus a proven forbidden overlap;
- clear-accumulate and initialized-accumulator variants;
- frontend construction and imported/generated TIR that bypasses it.

For pipelines:

- malformed stage/order metadata and valid replayable-bind compatibility;
- cyclic/conflicting dependencies versus a serial fallback;
- nested pipeline requests on supporting and unsupported backends, plus bare
  nested `T.Pipelined` and siblings;
- conditional async producer/consumer paths.

For layouts and kernels:

- annotation rank mismatch versus legal guarded-tail shape differences;
- replicated store with and without owner selection;
- compatible versus out-of-range thread ranges under warp specialization;
- target-independent valid kernel on supported and unsupported targets.

For types:

- legal numeric cast versus invalid reinterpret/packed-storage conversion;
- supported atomic/reducer dtype versus unsupported dtype;
- valid vector lane extraction versus compile-time out-of-range lane.
