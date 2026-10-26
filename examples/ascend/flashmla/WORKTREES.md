# FlashMLA compiler prerequisites

This example contains the manual and persistent staged kernels, runners,
historical measurement records, and support for native NZ copies, AIV stream
selection, and manual pipe ordering.
Apply the remaining compiler prerequisites before building the staged kernel.
The source does not use `T.IssuePhase`.

Included compiler support:

- Native NZ output for L0C-to-UB copies.
- AIV stream selection outside scheduled pipelines.
- Manual source order across physical phase ties on each shared core and pipe.

Remaining compiler prerequisites:

- [#3623](https://github.com/tile-ai/tilelang/pull/3623) preserves physical
  allocation/view remapping, pointer element units, and independent Cube view
  layouts on shared storage rings.
- [#3624](https://github.com/tile-ai/tilelang/pull/3624) preserves Cube
  configuration-register dependencies across manual stages.

The staged kernel uses `T.annotate_buffer_addresses` for exact L1 byte addresses
and supplies an explicit fixed `T.annotate_buffer_versions` count for every
placed buffer. `q_l1` has one physical allocation on both query paths: the
prefetched-query path already includes its two slots in the buffer shape.
Its `T.annotate_manual_multi_buffer` annotation tracks the full query-slot
period without adding another physical version dimension.

The staged kernel requires Cube-view layout support across aliases. QK allocates
one four-version L0B buffer; PV uses a same-shape `T.view` with a different layout.
The shared `.data` gives both views one version clock and a physical 64 KiB
ring without an `assume_conflict` hint. Both views use BF16 and the same padded
physical size. Copy and GEMM lowering preserve the selected view's layout.
The raw FP8 and FP4 buffers retain independent storage and three-slot histories.

## Historical integrated measurements

The historical measurements below used the integrated kernel at
`00e7b9bb64c892c5323f42bede764e8940ceb4cd`, before the L0B view change.
That compiler was based on upstream main `876a59e`; its documentation was
committed as `6bdfe6e`. The historical integration still contained an unused
IssuePhase API. These identities describe the recorded experiments, not the
compiler available on an unmodified checkout of this example.

- [RESULTS.md](RESULTS.md) contains the historical measurement tables and limitations.
- [Rebase record](results/staged-main-rebase-2026-10-20.json) records the native build, host checks, and lowering identities.
- [Prefill samples](results/staged-main-rebase-prefill-bench-2026-10-20.json) and [decode samples](results/staged-main-rebase-decode-bench-2026-10-20.json) retain the complete measured launches.
- The reference is FlashMLA commit `a123d0b0191e0da7aa1e044f8644e0989cc63220`.

The benchmark records are not performance measurements of the independently
split compiler branches. Re-run the documented commands on a chosen integrated
compiler revision before attributing those timings to a new build.

## Dependencies already in main

Complete periodic buffer lifetimes (#3606), exact on-chip buffer placement
(#3607), and partial UB-to-L1 NZ copies with correct pitches, compact allocations,
and access footprints (#3530) are already upstream.

SS buffer versioning and fixed-version UB reuse (#3514), independent AIV
execution and cross-core synchronization (#3485), partial L0 copies and layout
lookup (#3511), and the SIMD/PTO code-generation foundations are already upstream.
No additional TVM or CUTLASS modifications are required.

The optional zero-burst API and pipeline-peeling experiments are not used.
The kernel retains its documented zero-burst code-generation workaround.
PV unit-flag selection remains in kernel code rather than requiring a separate
dynamic unit-flag compiler patch.
