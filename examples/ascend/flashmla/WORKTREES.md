# FlashMLA compiler prerequisites

The kernel branch contains the kernels, runners, historical measurement records,
and the compiler changes from fork PRs #5, #7, and #12. It starts from
`d85a70a97218fec25856f8b5c969f167ad96dc1e`, synchronized with upstream on 2026-10-21.
Apply the remaining compiler prerequisites before building the staged kernel.
The source does not use `T.IssuePhase`, and its API is excluded from these proposals.

| Included compiler change | Source pull request |
| --- | --- |
| Support native NZ output for L0C-to-UB copies | [Denverjin/tilelang#5](https://github.com/Denverjin/tilelang/pull/5) |
| Select AIV streams outside scheduled pipelines | [Denverjin/tilelang#7](https://github.com/Denverjin/tilelang/pull/7) |
| Preserve manual source order across physical phase ties | [Denverjin/tilelang#12](https://github.com/Denverjin/tilelang/pull/12) |

| Remaining compiler change | Pull request | Fork branch |
| --- | --- | --- |
| Preserve Cube configuration phases across manual stages | [#6](https://github.com/Denverjin/tilelang/pull/6) | [codex/ascend-cube-register-phases](https://github.com/Denverjin/tilelang/tree/codex/ascend-cube-register-phases) |
| Preserve complete periodic buffer lifetimes | [tile-ai/tilelang#3606](https://github.com/tile-ai/tilelang/pull/3606), from [fork #8](https://github.com/Denverjin/tilelang/pull/8) | [ascend-periodic-buffer-lifetimes](https://github.com/Denverjin/tilelang/tree/ascend-periodic-buffer-lifetimes) |
| Support exact placement of versioned on-chip buffers | [tile-ai/tilelang#3607](https://github.com/tile-ai/tilelang/pull/3607), from [fork #9](https://github.com/Denverjin/tilelang/pull/9) | [ascend-buffer-placement-hints](https://github.com/Denverjin/tilelang/tree/ascend-buffer-placement-hints) |
| Preserve physical views and pointer units during layout lowering | [#10](https://github.com/Denverjin/tilelang/pull/10) | [codex/ascend-layout-pointer-units](https://github.com/Denverjin/tilelang/tree/codex/ascend-layout-pointer-units) |
| Preserve independent Cube view layouts on shared storage rings | [#11](https://github.com/Denverjin/tilelang/pull/11) | [codex/ascend-conflict-storage-rings](https://github.com/Denverjin/tilelang/tree/codex/ascend-conflict-storage-rings) |

The additional NZ pitch, compact-allocation, and access-footprint prerequisite
is already proposed upstream as [tile-ai/tilelang#3530](https://github.com/tile-ai/tilelang/pull/3530).
It is not duplicated as a new fork PR. The pointer-unit branch also carries
physical allocation/view remapping fixes that are outside that upstream PR.

The buffer-placement prerequisite now provides `T.annotate_buffer_offsets` for
UB, L1, L0A/L0B/L0C, and SS buffers, with an explicit fixed
`T.annotate_buffer_versions` count required for every placed buffer. The staged
kernel still calls the earlier `T.annotate_l1_offsets` API. Integrating the new
placement prerequisite also requires migrating that call and adding the explicit
fixed version count for `q_l1` on the prefetched-query path.

The staged kernel requires PR #11's Cube-view layout support from commit
`393a0bde46d60a3b7cfaabeeb65ebc67a7a14ec7` or later. QK allocates one
four-version L0B buffer; PV uses a same-shape `T.view` with a different layout.
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
