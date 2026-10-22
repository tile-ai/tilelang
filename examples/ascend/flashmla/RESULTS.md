# Measurement report

Measured on 2026-10-09 with one Ascend950DT_9581 (32 AIC + 64 AIV), physical
device 5, CANN 9.2.0, Bisheng clang 15.0.5 (2026-08-27), PyTorch 2.13.0+cpu,
and torch_npu 2.13.0rc1. TileLang base: `a35f8ddf45eba16c21211ec8822d56ce5363036f`.
FlashMLA reference: `a123d0b0191e0da7aa1e044f8644e0989cc63220`.
The unchanged reference extension and manual kernel ran on the same NPU.

## Performance

Each entry is the median of three round means, ten launches per round, using
the PR's msprof harness and 8 GB L2 flush. A/B order alternates by round.
Speedup is reference latency divided by manual latency. Small differences
near 1.00x should be interpreted together with the recorded round variation.
These are attention-kernel times; input reshapes, allocations and other kernels
are excluded. Decode applies the same per-call flattening as the PR wrapper,
so copies of strided inputs affect the initial cache state in both variants.

All prefill cases use nq=4096, H=64, D=512, topk=640 and attention sinks.

| KV length | Reference μs | Manual μs | Reference TFLOPS | Manual TFLOPS | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | 858.04 | 848.67 | 400.44 | 404.86 | 1.011x |
| 8192 | 853.78 | 851.15 | 402.44 | 403.68 | 1.003x |
| 32768 | 862.15 | 862.05 | 398.54 | 398.58 | 1.000x |

Decode uses FP8 main KV, topk=128, extra topk=512, H=64, D=512 and attention
sinks. The first eight cases have sq=4, variable cache sequence lengths,
main sequence length 256 and extra sequence length 2048. The last two are the
PR's shared-cache prefill-as-decode cases, with b=1, sq=4096 and both sequence
lengths 4096. They have no topk-length tensors.

| Extra cache | Batch | Queries/batch | Reference μs | Manual μs | Reference TFLOPS | Manual TFLOPS | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 | 64 | 4 | 78.44 | 76.70 | 273.78 | 279.99 | 1.023x |
| FP8 | 128 | 4 | 144.46 | 142.34 | 297.32 | 301.75 | 1.015x |
| FP8 | 256 | 4 | 274.83 | 271.30 | 312.56 | 316.63 | 1.013x |
| FP8 | 512 | 4 | 539.45 | 533.56 | 318.47 | 321.99 | 1.011x |
| FP4 | 64 | 4 | 70.60 | 71.38 | 304.19 | 300.84 | 0.989x |
| FP4 | 128 | 4 | 131.34 | 131.99 | 327.00 | 325.40 | 0.995x |
| FP4 | 256 | 4 | 247.43 | 248.10 | 347.17 | 346.23 | 0.997x |
| FP4 | 512 | 4 | 489.66 | 497.20 | 350.85 | 345.53 | 0.985x |
| FP8 | 1 | 4096 | 1045.89 | 1032.69 | 328.52 | 332.72 | 1.013x |
| FP4 | 1 | 4096 | 936.84 | 953.41 | 366.76 | 360.39 | 0.983x |

The FP4 shared-cache case had one anomalous round in the full sweep: its
manual mean was 6749.84 μs, with eight launches between 6.25 and 10.57 ms.
The other two round means were 953.41 and 953.17 μs. All samples remain in the
JSON and the table follows the same median-of-round-means rule as every case.
A separate five-round recheck gave manual round means of
952.22, 953.05, 952.43, 953.13, 952.65 μs, with median
952.65 μs, versus 939.97 μs
for the reference. The cause of the anomalous full-sweep capture is unresolved;
see [the full recheck](results/decode-shared-fp4-recheck.json).

## Correctness and scope

The correctness runner checks both variants against the PR's PyTorch
reference, using its output cosine/error tolerances and logits/LSE checks.
Prefill includes 14 cases: sinks, strided inputs, per-query lengths including
zero, invalid indices, partial index groups, and amplified KV values that
trigger rescaling. Decode includes 9 cases: FP8-only, FP8+FP8, FP8+FP4,
zero/all-invalid requests, per-query main/extra lengths, and odd page sizes.
Performance cases additionally compare the two implementations' outputs.

The supported example scope is H=64, D=512 and compact KV pages. Each quantized
cache must fit in 32-bit byte offsets. Runtime topk lengths are implemented as
masks over the static topk loop, while the reference shortens the loop. Thus
variable-topk performance is not claimed by these fixed-topk measurements.

## Controlled changes

The diagnostic case uses nq=4096, nk=8192, topk=640. Intermediate measurements
below use one ten-launch round, except the initial measurement (three rounds).
They guide tuning; the final comparison above uses three rounds for every case.

| Manual implementation | Latency μs |
| --- | ---: |
| Initial explicit pipeline | 1625.56 |
| Match gather / skip-scale notification ordering | 1251.82 |
| Match softmax row grouping and packed conversion | 1242.92 |
| Rely on explicit masks and zero-burst DMA | 1206.52 |
| Hoist scalar indices out of VF | 1206.47 |
| Specialize first-block / accumulation VFs and process two output rows together | 850.94 |

The last experiment combines VF specialization with the reference's paired-row
output epilogue; it does not isolate the contribution of either change alone.
Dynamic-UB alignment had no material effect in the measured case.

## Unroll and code size

The source explicitly keeps persistent and SIMD row-group loops rolled, while
unrolling the same small QK/PV, gather and ND-to-NZ groups as the reference.
The native flags include `-Ofast -mllvm -enable-hiipu-vf-loop-unroll`.
The ND-to-NZ VF is 536 bytes in both binaries. For the representative prefill
specialization, total device `.text` is 14,440 bytes for the reference and
15,560 bytes for the manual version. All manual SIMD VFs report zero stack
usage; the largest reported vector-register count is 12. This inspection did
not collect hardware ICache-miss counters.

## Evidence and commands

[README.md](README.md) contains the exact reproduction commands. Raw launch
samples and round means are saved in [results/prefill-bench-final.json](results/prefill-bench-final.json)
and [results/decode-bench-final.json](results/decode-bench-final.json). The
correctness JSON files, source hashes, compiler version and function sizes
are in [results](results).

These hardware results were collected on the pinned TileLang revision above.
The publication branch targets a newer `main`. The current pair gathers use
native `T.copy` with the temporary `T.assume` workaround documented in `README.md`;
the performance numbers above predate this change. The 2026-10-18 section below
reports the newer native-copy and staged decode measurements separately.
`results/environment.json` retains the original measurement and publication
source hashes as historical evidence. The current attention core and intrinsic
header differ from those snapshots.

On 2026-10-14, the native pair-copy version passed all 14 prefill and 9 decode
correctness cases through the TileLang runtime, including zero/all-invalid
pairs. This recheck used the unchanged compiler from `b683a26`, CANN 9.2.0,
torch_npu 2.13.0rc1, and Ascend950DT_9581 physical devices 7 (prefill) and 6
(decode). Generated code retained runtime 0/2 DMA counts and decode address
clamping. For representative BF16 prefill and FP8+FP4 decode specializations,
Bisheng assembly retained the same DMA instruction counts without increasing
conditional-jump counts. This was a correctness/code-generation recheck,
not a performance measurement.

The full local debug evidence is under `.artifacts/flashmla-pr229/` in the
worktree: `final-suite.log`, generated `.asc` files, intermediate experiments,
`prefill-specialized-resources.log`, `prefill-specialized-symbols.txt`,
`reference-prefill-symbols.txt`, and the pinned reference checkout. These large
scratch artifacts are ignored by Git. Transient device initialization
failures were retried before the final suite; failed starts provide no samples.

## Compiler-managed decode, 2026-10-18

The native-copy manual kernel and compiler-managed `staged.py` were measured
against the same pinned FlashMLA reference on the same Ascend950DT_9581,
physical device 5. Each value is the median of three round means with ten
launches per round and an 8 GB L2 flush before every launch. All selected
implementations pass the numerical checks before timing. The shape definitions
and fixed-topk scope above also apply here.

| Extra cache | Batch | Queries/batch | Reference μs | Manual μs | Staged μs | Manual / staged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 | 64 | 4 | 78.49 | 76.56 | 71.98 | 1.064x |
| FP8 | 128 | 4 | 144.43 | 143.03 | 136.07 | 1.051x |
| FP8 | 256 | 4 | 276.97 | 273.28 | 260.16 | 1.050x |
| FP8 | 512 | 4 | 546.29 | 540.71 | 511.31 | 1.057x |
| FP4 | 64 | 4 | 70.86 | 70.14 | 68.95 | 1.017x |
| FP4 | 128 | 4 | 130.75 | 131.15 | 129.01 | 1.017x |
| FP4 | 256 | 4 | 251.05 | 252.51 | 246.31 | 1.025x |
| FP4 | 512 | 4 | 497.03 | 500.63 | 479.67 | 1.044x |
| FP8 | 1 | 4096 | 1045.70 | 1034.47 | 974.18 | 1.062x |
| FP4 | 1 | 4096 | 937.63 | 954.60 | 935.75 | 1.020x |

All ten staged decode cases are faster than the manual kernel in this sweep.
The staged implementation also passed the nine decode correctness cases and
four independent main/extra-length boundary cases. Raw samples are in
[staged-decode-bench-2026-10-18.json](results/staged-decode-bench-2026-10-18.json).

These device measurements used compiler `0d86dd0b886624bb2e35bbfdf83e9e30cc358c98`.
After the pointer-layout fix in `34894d2ee0bf1c311e3587dd530ddb989ae8fa70`, all
23 unique decode instances were compiled again and loaded in fresh processes.
Their complete device executables, host sources, device sources, and parameter
files are byte-identical to the measured and correctness-tested versions;
[the compiler equivalence record](results/staged-decode-compiler-equivalence-2026-10-18.json)
contains the per-case hashes. This is a compilation equivalence check, not a
second device timing run. Kernel source and native-library hashes remain in
the benchmark provenance.

## 2026-10-20: high-level QK/PV with a shared L0B ring

Commit `e924481` changed `staged.py` to use independent MajorK and MajorMN B4 buffers, ordinary
`T.copy` / `T.gemm`, and default `T.assume_conflict(bq, bv)`. Compiler commit
`c536475` gives the compatible pair one ring clock while preserving its layouts.
The allocation remains 64 KiB. The original A4 and single S-to-A copy are retained.
PV uses two GEMM branches in one `T.Task`; the two-tile path holds the unit flag
in a scalar variable so intermediate rescale drains keep their M/FIX synchronization.

The preceding best kernel is preserved in local commit `9e691b0` and in the
frozen source identified by SHA256 `1cd8a2282ab2d960623fb85c4777c0b4d1679d158b9b6c87170e8fbafde55697`.
Both versions below were built with the same compiler and compared on identical
inputs, alternating order over three rounds of ten launches on physical NPU 5.
Times are the median of the three round means; a positive change means slower.

| BF16 prefill, nk | Previous best, μs | High-level B4, μs | Change |
| ---: | ---: | ---: | ---: |
| 4096 | 865.48 | 863.65 | -0.21% |
| 8192 | 864.46 | 866.13 | +0.19% |
| 32768 | 873.02 | 873.40 | +0.04% |

Validation passed all 14 BF16 correctness cases, all nine decode correctness
cases, six permanent BF16 regressions, and four independent decode-length
boundary regressions. The compiler change passed 741 host tests and its device
regression with unequal owner lengths and repeated launches. This change's
performance sweep covers BF16; the earlier ten-case decode timing table above
belongs to its separately identified baseline.

Source/native identities, correctness counts, and all timing samples are in
[staged-b4-layouts-2026-10-20.json](results/staged-b4-layouts-2026-10-20.json).
[WORKTREES.md](WORKTREES.md) records the separately proposed compiler
prerequisites and the identities of the historical integrated builds.

## 2026-10-20: manual issue order without IssuePhase, before main rebase

Commit `9c46d1c` removes all `T.IssuePhase` calls from FlashMLA. The source
orders operations by their intended issue order while `T.Stage` identifies
their logical KV tiles. The compiler now separates physical issue phases
when the larger-stage-first reconstruction rule would reverse a manual
source-order edge on a shared core/pipe. Generated AIV code issues gather
before waiting for softmax's delta and publishing SS. That historical compiler
still contained the unused IssuePhase API; the split compiler proposals omit it.

Existing validation records report 68 focused tests, 776 AutoSchedule tests,
eight schedule comparisons, and 11 FlashMLA device regressions passing.
The device suite covers pipeline and flat execution, cross-query rescaling,
alternating rescale/accumulate paths, index-group tails, Q-prefetch boundaries
at 5/15/16 tiles, and four independent main/extra decode-length cases.
The source and native library checked before rebasing match these recorded
SHA256 values. The kernel source is unchanged by the rebase; the rebuilt
compiler is a separate validation target.

```text
staged.py:      986817d6f361116825ea0cb9a51db576e7a3fc9730c02936c1a0826fc4d826a2
libtilelang.so: f63a9bebad45fe496bb7e89c621d29efaf819fca41a61e612648175c65885713
```

The performance comparison used Ascend950DT_9581 physical NPU 6, identical
inputs, kernelkit/msprof with L2 flushing, and three alternating rounds of
ten launches. Values below are medians of round means. Timing started after
the device regression suite completed.

| BF16 prefill (nq, nk, topk) | Intermediate baseline, μs | Before rebase, μs | Change |
| --- | ---: | ---: | ---: |
| (4096, 4096, 640) | 866.10 | 867.44 | +0.154% |

This baseline is the intermediate source with one remaining IssuePhase,
SHA256 `b837367b6c3a0e9d8d9faabaac3946726361012ba6bf1b87050163a5b827425e`.
Its frozen ASC from the previous compiler was recompiled after checking the
kernel ABI and returned source identity. Outputs match that baseline bitwise.
This is a separate comparison from the three-case B4 table above; it does
not directly compare against the original `e924481` or `9e691b0` source.

[The manual-order record](results/staged-manual-order-2026-10-20.json) preserves
source/library identities, schedule comparisons, and the three round means
for each variant. Per-launch samples were not saved in that record.
The original local summary was written before committing, with `e924481` as
its parent HEAD; the validated changes were subsequently committed as `9c46d1c`.
This pre-rebase record does not include a rerun of the full 14-case BF16 or nine-case
decode correctness runners, nor the full three-case BF16 and ten-case decode
timing matrices. Earlier measurements retain their original revision scope.

## 2026-10-20: main rebase, build and host validation

The integrated kernel branch at this measurement was `00e7b9b`, based on upstream main `876a59e`.
The rebase removes upstreamed SS, independent-AIV, partial-copy, and SIMD/PTO
changes from the local delta. The composite prerequisite commit `35a66f4`
becomes `d3042c7`, reduced from 77 files to 22 while retaining the remaining
NZ pitch, compact-allocation, and access-footprint fixes. Main's newer codegen
base is preserved. All committed FlashMLA files are unchanged from `9c46d1c`.

The incremental native build passed, followed by 987 host tests covering
AutoSchedule and the relevant copy, ND2NZ, L1-offset, layout, and PTO paths.
Full TileLang lowering and ASC source generation passed for BF16
`(nq=65, nk=2048, topk=640)` and FP8 main plus FP4 extra
`(nq=65, nk=2048, topk=128, extra_topk=512)`, with variable lengths enabled.
The initial host selection included unmarked device tests, which failed at
NPU initialization with device visibility disabled. The final run explicitly
selects host tests from those layout files; both logs are retained.

The rebuilt `libtilelang.so` SHA256 is
`489c5ace819791fec4a2058c93c0ee9a50fddde505cad2a172f0929619ddda3b`.
The [rebase validation record](results/staged-main-rebase-2026-10-20.json)
contains the commit mapping, exact test command, dependency identities,
library hashes, generated-source hashes, and lowering warnings.
That initial check did not include Bisheng compilation of these generated
kernels, NPU correctness runs, or performance measurements. The subsequent
device performance sweep on the same compiler is recorded below.

## 2026-10-20: complete performance sweep after main rebase

The integrated `00e7b9b` branch, based on main `876a59e`, was measured on
Ascend950DT_9581 physical NPU 6 with CANN 9.2.0, Bisheng 15.0.5,
torch 2.13.0+cpu, and torch_npu 2.13.0rc1. The native library SHA256 remains
`489c5ace819791fec4a2058c93c0ee9a50fddde505cad2a172f0929619ddda3b`.
All 26 manual/staged kernels were freshly compiled and passed allocation
checks before device execution. Runtime cache hits were checked against their
source and binary identities, with new compilation forbidden during the sweep.
The pinned FlashMLA reference remains `a123d0b0191e0da7aa1e044f8644e0989cc63220`.

All three implementations used the same inputs at each of the three BF16
prefill and ten decode shapes. Manual and staged outputs passed the existing
runners' comparisons against FlashMLA before timing. Each measurement used
kernelkit/msprof, an 8 GB buffer to flush L2, ten warmup launches, and ten measured
launches per round. Implementation order reversed on alternate rounds.
Tables report the median of three round means in microseconds; a positive
change means staged is slower than manual. All 1,170 measured attention-kernel
durations are retained. Compilation, allocations, and wrapper kernels are
excluded from the reported attention-kernel time.

BF16 prefill uses `nq=4096, topk=640, H=64, Dqk=Dv=512`:

| nk | Reference, μs | Manual, μs | Staged, μs | Staged vs manual |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 853.32 | 848.60 | 864.24 | +1.84% |
| 8192 | 854.87 | 850.44 | 866.68 | +1.91% |
| 32768 | 861.95 | 861.72 | 873.97 | +1.42% |

Decode uses FP8 main KV, `topk=128`, and `extra_topk=512`. The extra-cache
format and batch/query dimensions vary:

| Extra format | b | sq | Reference, μs | Manual, μs | Staged, μs | Staged vs manual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 | 64 | 4 | 78.77 | 77.13 | 72.84 | -5.56% |
| FP8 | 128 | 4 | 145.32 | 143.10 | 137.16 | -4.15% |
| FP8 | 256 | 4 | 275.85 | 271.63 | 264.53 | -2.62% |
| FP8 | 512 | 4 | 541.40 | 535.04 | 519.54 | -2.90% |
| FP4 | 64 | 4 | 70.62 | 70.26 | 67.93 | -3.33% |
| FP4 | 128 | 4 | 129.65 | 130.13 | 127.20 | -2.25% |
| FP4 | 256 | 4 | 247.19 | 249.05 | 245.40 | -1.46% |
| FP4 | 512 | 4 | 490.12 | 495.86 | 479.41 | -3.32% |
| FP8 | 1 | 4096 | 1046.16 | 1034.05 | 998.09 | -3.48% |
| FP4 | 1 | 4096 | 936.10 | 953.46 | 957.61 | +0.44% |

Staged remains 1.42–1.91% slower than manual on BF16 prefill. It is
1.46–5.56% faster on nine decode cases; FP8+FP4 with `b=1, sq=4096` is
0.44% slower than manual and 2.30% slower than the reference.
The [prefill record](results/staged-main-rebase-prefill-bench-2026-10-20.json)
and [decode record](results/staged-main-rebase-decode-bench-2026-10-20.json)
contain every launch duration, exact compile arguments, resource usage,
and source/native/generated-binary hashes.

This sweep validates outputs at the benchmark shapes; the separate 14-case
BF16 and nine-case decode correctness runners were not rerun. Earlier tables
retain their original source/compiler/device scope. The simultaneous
reference/manual/staged measurements above do not by themselves isolate the
performance effect of rebasing. Profiler logs reported unsupported
`AICORE_SHAPE` metadata, but every capture contained the expected ten
attention-kernel records.
