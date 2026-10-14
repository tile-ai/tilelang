# Memory and Concurrency Semantic Rules

## Contents

1. [Severity model](#severity-model)
2. [Buffer rank and bounds](#buffer-rank-and-bounds)
3. [Initialization](#initialization)
4. [Storage ownership and aliasing](#storage-ownership-and-aliasing)
5. [Parallel races and atomics](#parallel-races-and-atomics)
6. [Collectives and barriers](#collectives-and-barriers)
7. [Asynchronous operations](#asynchronous-operations)
8. [Current coverage and priorities](#current-coverage-and-priorities)
9. [Boundary tests](#boundary-tests)

## Severity model

Use a hard error only for a proven violation. Use a guard, warning, or opt-in
analysis when validity cannot be established.

| Result | Treatment |
|---|---|
| Proven rank mismatch, OOB, race, lifecycle violation, or invalid scope | Error |
| Access may be unsafe but proof is inconclusive | Guard or warning |
| Valid operation unsupported by one target | Target-specific unsupported diagnostic |
| Safe operation that prevents vectorization or another optimization | Optimization warning at most |

## Buffer rank and bounds

Enforce these contracts for `BufferLoad`, `BufferStore`, buffer regions, and
pointer-style accesses:

- Require the number of indices to match the logical buffer rank unless the
  operation explicitly defines flattened or region semantics.
- Reject an access when analysis proves any scalar or vector lane is below
  zero or at/above its dimension extent.
- Preserve Python-style negative-index legalization when the language API
  explicitly supports it; validate the normalized index instead.
- Check every lane of `Ramp`, `Broadcast`, and `Shuffle` indices. Predicate
  partially valid vector accesses instead of accepting one in-range lane as
  proof for the whole vector.
- Validate the base, extent, and read/write mask of `T.access_ptr` together;
  checking only the base element is insufficient.
- Require a buffer region to fit the source and destination shapes of the op
  that consumes it.

Current caveat: `SafeMemChecker` skips constant indices and treats local/shared
OOB primarily as optional warnings. Do not rely on that behavior as the
language contract. Add a default hard error for statically proven OOB while
retaining guards/warnings for uncertain accesses.

## Initialization

Apply initialization semantics according to ownership:

- Treat global buffers as caller-owned unless the API declares an output or
  initialization obligation.
- Require local and fragment values to be defined before use on every path that
  reaches the read when that failure is provable.
- Model read-modify-write as a read followed by a write; `x += y` does not
  initialize `x`.
- Account for tile operators and writable pointer escapes as potential writes,
  but do not assume an opaque read-only call initialized a buffer.
- Treat shared memory as cross-thread state. Source order alone does not prove
  initialization under warp specialization; require a producer plus the
  synchronization contract that makes it visible.
- Distinguish “nothing ever writes this buffer” from path-sensitive partial
  initialization. The former can be a low-noise default diagnostic; the latter
  needs stronger control-flow and synchronization analysis.

`VerifyBufferInit` currently implements the low-noise first tier. Extend it
incrementally rather than claiming it proves definite assignment.

## Storage ownership and aliasing

Use storage scope as a semantic ownership contract:

- `local`: private to one physical thread; do not infer cross-thread ownership
  from a logical parallel index.
- `local.fragment`: distributed or replicated according to a Fragment layout;
  validate accesses against that ownership map.
- `shared`: visible to participating threads in one CTA only after the required
  synchronization.
- cluster-shared scopes: visible only within the declared cluster and through
  supported cluster operations.
- `global`: visible across CTAs; cross-CTA coordination requires atomics or a
  supported grid/cluster synchronization mechanism.

For aliases and pointer escapes:

- Preserve address space and storage scope through `address_of`, views, and
  access-pointer lowering.
- Require aliased regions to have compatible dtype, alignment, extent, and
  access permissions.
- Reject overlapping source/destination regions only when the operation does
  not define overlap semantics.
- Keep a buffer alive until every synchronous or asynchronous user has
  completed.
- Include writes through `access_ptr`, external calls, and TileOps in effect
  analysis when their contracts permit mutation.

## Parallel races and atomics

For a non-atomic shared/global store inside logical parallel execution:

- report a hard error when two distinct iterations are proven to write the
  same location with conflicting values;
- accept a proven injective address mapping;
- follow the project's explicit same-value-write contract rather than silently
  changing it;
- exempt first-class reducers and supported atomics from ordinary-store race
  rules, then validate their own contracts;
- warn or require opt-in analysis when address disjointness is merely
  unprovable.

For atomic operations, validate:

- supported operand and result dtype;
- natural alignment and vector-lane contiguity;
- memory scope and backend support;
- legal memory-order values and preservation through lowering;
- whether return-previous semantics are available for the selected operation;
- address/region shape for vector atomics.

Split the existing race verifier into a low-false-positive proven-race tier and
an optional potential-race tier instead of enabling every solver failure as an
error.

## Collectives and barriers

Treat each collective operation as declaring a participant set. Require every
participant to execute compatible operations in a compatible order and count.

- Validate `sync_threads` against CTA participants and `sync_grid` against a
  cooperative grid launch.
- Reject a collective when a condition is proven non-uniform across its
  participants.
- Reject unequal proven execution counts, including loop trip-count and early
  exit differences.
- Allow a barrier inside `T.Parallel` when loop partitioning/replication makes
  the participant count and execution count uniform; placement alone is not a
  violation.
- Treat warp-group and cluster collectives according to their narrower
  participant sets rather than requiring whole-CTA uniformity.

For mbarriers, validate the lifecycle:

1. initialize before arrive, expect-tx, or wait;
2. keep the barrier in a supported shared scope;
3. use a compatible expected-arrival/transaction count;
4. pair wait phase/parity with the corresponding producer epoch;
5. do not reuse or destroy an epoch before all required waits complete;
6. keep producer and consumer references within the same ownership domain.

Generated pipeline barriers may satisfy these rules by construction. Preserve
annotations that distinguish generated ownership from user-managed barriers.

## Asynchronous operations

For `T.async_copy`, cp.async, TMA, and other async producers:

- require a completion mechanism before the first consuming read;
- prevent destination overwrite or buffer-version reuse before completion;
- prevent source/destination lifetime end before completion;
- pair commit/wait groups and enforce backend limits on outstanding groups;
- validate barrier ownership, phase, transaction count, and pipeline version;
- validate source/destination scopes, direction, alignment, and region shape;
- account for conditional producers so a consumer never waits for work that
  was not issued or consumes data whose producer was skipped.

Prefer a unified async-dependency verifier over isolated instruction-specific
checks. Run it while source TileOps and explicit waits are still recognizable,
or preserve effect annotations through lowering.

## Current coverage and priorities

| Area | Current state | Next step |
|---|---|---|
| Buffer initialization | Default low-noise warning; not path-sensitive | Add proven local/fragment definite-assignment errors |
| Global bounds | Runtime guards for uncertain accesses | Add proven constant/static OOB errors |
| Local/shared bounds | Optional warning | Split proven violation from inconclusive warning |
| Parallel race | Optional potential-race warning | Enable a proven-race tier by default |
| Atomic contracts | Distributed across APIs/codegen | Centralize dtype/alignment/order checks |
| Collective uniformity | Fragmented | Add participant/path/count analysis |
| Barrier lifecycle | Mostly lowering-local | Add user-visible lifecycle verification |
| Async dependencies | Pipeline/instruction-specific | Add wait/use/reuse verifier |

Prioritize proven OOB/rank mismatch, collective/barrier correctness, async
wait/use/reuse, and proven data races. These failures can produce silent memory
corruption or deadlock.

## Boundary tests

For each rule, include both the violation and its nearest valid neighbor:

- constant OOB versus a symbolic access protected by a valid predicate;
- rank mismatch versus an explicitly flattened access API;
- local read-before-write versus initialization on every branch;
- shared producer/consumer with and without the required synchronization;
- proven colliding store versus injective store and supported atomic update;
- divergent collective versus a uniform collective inside `T.Parallel`;
- wait-before-use versus async use-before-wait and overwrite-before-wait;
- barrier phase match versus stale parity and mismatched ownership.

Assert source spans and actionable diagnostics whenever the source IR carries
them.
