# FlashMLA compiler prerequisites

The example branch contains the kernels, runners, and historical measurement
records. Compiler changes are proposed separately against `Denverjin/tilelang:main`,
synchronized to upstream `d85a70a97218fec25856f8b5c969f167ad96dc1e` on 2026-10-21.
Apply the compiler prerequisites before building the staged kernel.
The source does not use `T.IssuePhase`, and its API is excluded from these proposals.

| Compiler change | Pull request | Fork branch |
| --- | --- | --- |
| Support native NZ output for L0C-to-UB copies | [#5](https://github.com/Denverjin/tilelang/pull/5) | [codex/ascend-l0c-ub-nz](https://github.com/Denverjin/tilelang/tree/codex/ascend-l0c-ub-nz) |
| Preserve Cube configuration phases across manual stages | [#6](https://github.com/Denverjin/tilelang/pull/6) | [codex/ascend-cube-register-phases](https://github.com/Denverjin/tilelang/tree/codex/ascend-cube-register-phases) |
| Select AIV streams outside scheduled pipelines | [#7](https://github.com/Denverjin/tilelang/pull/7) | [codex/ascend-aiv-stream-selection](https://github.com/Denverjin/tilelang/tree/codex/ascend-aiv-stream-selection) |
| Preserve complete periodic buffer lifetimes | [#8](https://github.com/Denverjin/tilelang/pull/8) | [codex/ascend-periodic-buffer-lifetimes](https://github.com/Denverjin/tilelang/tree/codex/ascend-periodic-buffer-lifetimes) |
| Establish buffer placement hints from exact L1 offsets | [#9](https://github.com/Denverjin/tilelang/pull/9) | [codex/ascend-buffer-placement-hints](https://github.com/Denverjin/tilelang/tree/codex/ascend-buffer-placement-hints) |
| Preserve physical views and pointer units during layout lowering | [#10](https://github.com/Denverjin/tilelang/pull/10) | [codex/ascend-layout-pointer-units](https://github.com/Denverjin/tilelang/tree/codex/ascend-layout-pointer-units) |
| Preserve independent Cube view layouts on shared storage rings | [#11](https://github.com/Denverjin/tilelang/pull/11) | [codex/ascend-conflict-storage-rings](https://github.com/Denverjin/tilelang/tree/codex/ascend-conflict-storage-rings) |
| Preserve manual source order across physical phase ties | [#12](https://github.com/Denverjin/tilelang/pull/12) | [codex/ascend-manual-pipe-order](https://github.com/Denverjin/tilelang/tree/codex/ascend-manual-pipe-order) |

The additional NZ pitch, compact-allocation, and access-footprint prerequisite
is already proposed upstream as [tile-ai/tilelang#3530](https://github.com/tile-ai/tilelang/pull/3530).
It is not duplicated as a new fork PR. The pointer-unit branch also carries
physical allocation/view remapping fixes that are outside that upstream PR.

The buffer-placement branch currently implements exact L1 byte offsets.
Its PR describes the intended extension to manual placement hints for every
buffer kind. That extension is not implemented yet; the kernel still calls
`T.annotate_l1_offsets` and uses the existing L1 placement contract.

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
