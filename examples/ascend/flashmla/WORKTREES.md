# FlashMLA compiler prerequisites

The kernel branch contains the manual and persistent staged kernels, runners,
historical measurement records, and support for native NZ copies, AIV stream
selection, and manual pipe ordering. The compiler snapshot is based on upstream
`d85a70a97218fec25856f8b5c969f167ad96dc1e`.
Apply the remaining compiler prerequisites before building the staged kernel.
The source does not use `T.IssuePhase`.

Included compiler support:

- Native NZ output for L0C-to-UB copies.
- AIV stream selection outside scheduled pipelines.
- Manual source order across physical phase ties on each shared core and pipe.

Remaining compiler support:

- Preserve Cube configuration-register dependencies across manual stages.
- Preserve complete periodic buffer lifetimes, including externally preloaded
  values and sibling accesses: [#3606](https://github.com/tile-ai/tilelang/pull/3606).
- Place versioned on-chip buffers at exact byte offsets:
  [#3607](https://github.com/tile-ai/tilelang/pull/3607).
- Preserve physical allocation/view remapping and pointer element units during
  layout lowering.
- Preserve independent Cube view layouts on shared storage rings.
- Support partial UB-to-L1 NZ copies with correct pitches, compact allocations,
  and access footprints: [#3530](https://github.com/tile-ai/tilelang/pull/3530).

The buffer-placement prerequisite now provides `T.annotate_buffer_offsets` for
UB, L1, L0A/L0B/L0C, and SS buffers, with an explicit fixed
`T.annotate_buffer_versions` count required for every placed buffer. The staged
kernel still calls the earlier `T.annotate_l1_offsets` API. Integrating the new
placement prerequisite also requires migrating that call and adding the explicit
fixed version count for `q_l1` on the prefetched-query path.

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
compiler available on an unmodified checkout of the new example branch.

- [RESULTS.md](RESULTS.md) contains the historical measurement tables and limitations.
- [Rebase record](results/staged-main-rebase-2026-10-20.json) records the native build, host checks, and lowering identities.
- [Prefill samples](results/staged-main-rebase-prefill-bench-2026-10-20.json) and [decode samples](results/staged-main-rebase-decode-bench-2026-10-20.json) retain the complete measured launches.
- The reference is FlashMLA commit `a123d0b0191e0da7aa1e044f8644e0989cc63220`.

The benchmark records are not performance measurements of the independently
split compiler branches. Re-run the documented commands on a chosen integrated
compiler revision before attributing those timings to a new build.

## Dependencies already in main

SS buffer versioning and fixed-version UB reuse (#3514), independent AIV
execution and cross-core synchronization (#3485), partial L0 copies and layout
lookup (#3511), and the SIMD/PTO code-generation foundations are already upstream.
No additional TVM or CUTLASS modifications are required.

The optional zero-burst API and pipeline-peeling experiments are not used.
The kernel retains its documented zero-burst code-generation workaround.
PV unit-flag selection remains in kernel code rather than requiring a separate
dynamic unit-flag compiler patch.
