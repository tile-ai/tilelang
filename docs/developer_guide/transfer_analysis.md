# Transfer semantics and producer scheduling

`src/transform/common/transfer_analysis.{h,cc}` separates three questions that
used to share the `copy_stage` classification:

1. What value and memory effects does the statement produce?
2. Where should the scheduler place it?
3. Which physical transfer instruction can implement it?

## Shared semantic analysis

`AnalyzeTransferValue` describes a source `BufferLoad`, optional zero-fill data
predicate, and whether the value undergoes conversion. Pipeline planning and
the CUDA/ROCm async injectors use the same matcher. Arithmetic, nonzero padding,
negative floating-point zero and masked loads with unspecified fill semantics
are not byte-preserving zero-fill transfers.

`AnalyzeTransfers` summarizes a whole scheduling unit, including TileOps and
ordinary loops. It traverses address and predicate dependencies as well as the
stored value. Side effects, intra-producer shared/local reads and non-shared
writes prevent treating the entire unit as an implicit async region. Mixed
copy/compute regions remain synchronous rather than skipping their internal
dependencies.

An expression `if_then_else(p, load, 0)` writes zero when `p` is false. An
`IfThenElse` statement may skip a write entirely. The latter is an execution
guard and is not automatically assigned an async completion protocol.

## Independent policy and capability

`PipelineStageInfo::prefetch_stage` is an early-placement policy. Direct,
converting and zero-fill transfers are preferred for prefetch. This policy is
deliberately independent from `async_candidate`: conversion may be worth
prefetching but cannot be implemented by byte-preserving async copy. Explicit
manual stage assignments are retained even when a transfer is async-capable.

Placement also considers consumers. A synchronous producer used only by direct
shared-to-fragment copies stays with those consumers: advancing the entire
load/convert/store relay can increase shared-buffer versioning and register
lifetimes without useful overlap. Async-capable transfers, TMA, shared operands
read by computation, and explicit schedules retain their existing policy.
Dependency propagation follows this placement decision, so unrelated async
transfers in the same loop remain pipelined. This is a cost policy, not an
instruction-legality rule or a guarantee of improvement in every reuse context.

An initial policy kept guarded transfers late to preserve the previous schedule.
Paired measurements found a regression in automatic guarded GEMM. An independent
early-placement ablation improved all four affected automatic configurations,
including three convolutions, so the final policy removes that blanket exclusion.

Manual and automatic pipelines consume the same capability facts. CUDA warp
specialization uses the broader `IsSimtProducer` effect query for **role**
placement: producer warps may execute synchronous arithmetic or conversions.
That role is not a promise to emit `cp.async`.

TMA im2col remains an opaque backend-owned transfer. SIMT im2col becomes ordinary
guarded accesses through logical lowering, and receives no operator-name
exception in producer analysis.

## Completion dependencies and physical realization

The existing per-statement producer/group annotations encode candidate
completion regions. `InjectSoftwarePipeline` manages buffer versions and
wait-before-use dependencies. A consumer can depend on producers in **multiple
stages**; wait analysis visits every relevant producer stage instead of assuming
a single owner. Per-stage commit counts are conservative on a shared hardware
queue: commits from other stages cannot make these waits too weak, although
they can cause more waiting than a globally optimized issue timeline would.

Layout inference and physical loop lowering still precede backend instruction
selection. CUDA/ROCm injectors check the transfer's physical width, indexing and
target constraints. A candidate with illegal physical geometry remains ordinary
loads/stores, with normal shared-memory synchronization. Empty commit groups
retain their numbering; removing them locally would invalidate downstream
inflight counts.

This design reuses the existing completion representation and synchronization
passes. It does not add a public transfer dialect, a second pipeline IR, or
claim to implement global issue-timeline optimization.

## Physical effects through vectorization and synchronization

Vectorization widens declared access-pointer regions together with cp.async
instruction counts. Later passes must see the whole transfer, not the scalar
extent of its first element.

ThreadSync compares pointer-involving accesses in physical bytes, including
buffer strides, element offsets, vector lanes, packed subbyte types and shared
alias offsets. Independent thread bindings and mutable-index snapshots remain
independent; loop-carried comparisons advance both the next address and its
constraints. Existing indexed-access ownership analysis remains in place.

Known instructions own their actual access extent and direction: cp.async
counts, vector atomic widths and matrix shared-row widths override stale
pointer metadata. An unbounded `address_of` does not imply a one-element access.
Unknown footprints cannot establish disjointness. These proofs remove barriers
between disjoint regions, not completion waits or explicit synchronization.

## Validation boundaries

Changes to instruction eligibility must be measured on convolution as well as
producer microbenchmarks; numerical
correctness alone is not evidence that a new schedule is profitable.
