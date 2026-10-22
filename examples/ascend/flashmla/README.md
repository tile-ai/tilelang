# FlashMLA PR #229: manual and compiler-managed Ascend scheduling

This example reproduces the Ascend 950 sparse attention kernel in
[FlashMLA PR #229](https://github.com/deepseek-ai/FlashMLA/pull/229), pinned to
`a123d0b0191e0da7aa1e044f8644e0989cc63220`, with explicit TileLang scheduling.
The code and scheduling algorithms adapted from FlashMLA retain its MIT notice
in `LICENSE.flashmla`.

`core.py` owns the complete AIC/AIV pipeline, online softmax, gather2 index
processing, skip-scale accumulation, and synchronization. `quantization.py`
implements FP8 / FP4 dequantization in TileLang SIMD. `manual_intrinsics.h`
only wraps individual CANN transfer / MMA instructions and SS-buffer access;
it contains no attention loop and does not call the reference kernel.
`TL_ENABLE_AUTO_SCHEDULE=False` and explicit `T.Cube()` / `T.Vector()` are used.

`staged.py` expresses the same attention algorithm as logical tiles with
`T.Stage` and compiler-managed versions and synchronization. A persistent KV
pipeline spans query jobs. Source order and manual pipe-order constraints
preserve issue order without `T.IssuePhase` annotations. Q uses two
explicit query slots when the periodic lifetime fits the event budget
(`4 <= nb <= 15`); other shapes retain the single-query-slot path. The two
32-row KV halves have explicit L1 offsets in separate bank groups.

QK and PV use two views of one four-version L0B allocation. QK's `bq` has a
MajorK layout, while `bv = T.view(bq, shape=(64, 128))` has a MajorMN layout.
The views share `.data`, one version clock, and a 64 KiB ring, so no
`T.assume_conflict` hint is needed. Each producer writes the layout consumed
through its view; creating the view does not convert existing data. Their copies
and matrix products use `T.copy` and `T.gemm`. PV dispatches its nonzero and zero unit flags
through two GEMM branches in one `T.Task`; the two-KV-tile path keeps the flag in
a scalar variable so rescale synchronization survives constant folding.

For decode with FP8 main KV and FP4 extra KV, each conditional pair-copy is one
`T.Task`, and the conditional dequantization is another. This gives the two
format alternatives a common producer/consumer ownership boundary. `raw8` and
`raw4` retain separate storage, layouts, and three-slot histories: a zero-burst
copy must preserve the previous contents of the selected format's own slot.
They are not interchangeable views of one raw buffer. `dequantize(..., slot=None)`
serves these logical raw tiles; the manual kernel's explicit-slot path is unchanged.

The manual pair gathers use two-dimensional `T.make_tensor_from_addr` views;
the staged kernel uses `T.StridedTensor`. Both use `T.copy`,
with runtime source strides and a runtime row count of zero or two. A zero
count preserves the UB contents; it does not clear them. The local
`T.assume(n_burst > 0)` is a temporary code-generation workaround that removes
the copy's positive-count guard in the current compiler while retaining the
runtime zero-or-two DMA argument. It is **not a valid input invariant**: fully
invalid pairs still have zero bursts. Recheck the generated code and invalid-pair
cases when upgrading the compiler; this needs an explicit zero-burst copy
contract to become a supported use of `T.copy`.

The manual softmax handoff uses the existing `tl.ascend_copy_ubuf_to_cbuf` intrinsic through
`T.call_intrin`: one DMA with four 1 KiB bursts and 1 KiB source/destination gaps.
This preserves the in-place BF16 packing, shared L1 tile layout, instruction
granularity, and manual synchronization. Ordinary `T.copy` still requires
contiguous regions on this raw UB-to-L1 path. The staged kernel instead uses a
compact 32-row S tile with an annotated layout and ordinary `T.copy`.

Supported shapes are H=64, Dqk=Dv=512, BF16 Q and output, and topk multiples of
64. Prefill takes contiguous BF16 KV and allows strided Q / indices. Decode
accepts the PR's compact FP8 main cache and optional FP8 or FP4 extra cache.
Both support attention sinks, invalid indices, and per-query topk lengths.
Packed decode pages must be contiguous, and each cache must be smaller than
4 GiB because gather uses 32-bit byte offsets. Runtime topk lengths mask the
fixed loop extent; they do not shorten this example's pipeline loop.

## Reproduce

The kernel branch includes the native NZ-copy, AIV-stream-selection, and manual
pipe-order compiler changes. It starts at main `d85a70a`; apply the remaining
compiler prerequisites and placement-API migration listed in
[WORKTREES.md](WORKTREES.md) before compiling the staged kernel.
The historical integrated compiler was based on main `876a59e`; its
[rebase validation record](results/staged-main-rebase-2026-10-20.json) covers
the initial native build, host tests, and kernel source generation. The
[subsequent full performance sweep](RESULTS.md#2026-10-20-complete-performance-sweep-after-main-rebase)
records three BF16 prefill and ten decode cases on that integrated compiler,
including output checks and same-input reference/manual/staged comparisons.

Build TileLang with the prerequisite changes and Ascend enabled. The original performance
measurements in `RESULTS.md` used `a35f8ddf45eba16c21211ec8822d56ce5363036f`
and the earlier C-adapter pair gathers. The 2026-10-18 section reports the
native-copy manual and compiler-managed decode comparison with its exact
compiler and source identities. Build the pinned FlashMLA reference separately:

```bash
cd /path/to/FlashMLA
MAX_JOBS=16 FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND python setup.py build_ext --inplace
```

From the TileLang worktree, select an idle physical NPU and run:

```bash
export ASCEND_RT_VISIBLE_DEVICES=5  # select a free device on your machine
export PYTHONPATH=.
python examples/ascend/flashmla/run.py --reference /path/to/FlashMLA \
  --mode correctness --output results/prefill-correctness.json
python examples/ascend/flashmla/run.py --reference /path/to/FlashMLA \
  --mode bench --rounds 3 --repeat 10 --output results/prefill-bench.json
python examples/ascend/flashmla/run_decode.py --reference /path/to/FlashMLA \
  --mode correctness --output results/decode-correctness.json
python examples/ascend/flashmla/run_decode.py --reference /path/to/FlashMLA \
  --mode bench --rounds 3 --repeat 10 --output results/decode-bench.json
```

Both runners accept `--implementation reference`, `manual`, or `staged` to
select one implementation. The default `both` preserves the reference/manual
comparison; `all` includes the staged kernel. For example, add
`--implementation all` to the correctness or benchmark commands above to compare
all three on the same inputs. A selected implementation run alone still receives
the mathematical-reference correctness check before timing.

The scripts use the pinned PR's input generator, numerical reference,
quantization, FLOP accounting, and `kernelkit.bench` profiler. All selected variants
receive the same tensors, including the reference's non-contiguous strides.
Each timed launch follows an 8 GB L2 flush; JIT compilation and warm-up are
excluded. They reverse implementation order on alternate rounds, require exactly 10
attention-kernel samples per round, and report the median of three round means.
JSON files retain every launch duration. Timing covers the attention kernel;
host allocations, input reshapes, and ancillary device kernels are excluded,
as in the PR's attention-kernel performance metric.

`--nk 8192 --detail` selects one prefill performance case and additionally
collects AIC/AIV pipe statistics. `--case N` selects a zero-based case in either runner.
The full performance sets contain three prefill cases and ten decode cases,
including the two prefill-as-decode cases.

## Scheduling

The manual kernel preserves these reference choices: four QK/PV tiles and sixteen gather2 DMA
instructions are unrolled; the BF16 ND-to-NZ row micro-loop is unrolled 32
ways with post-increment pointers; softmax row groups and the persistent
pipeline stay rolled. Softmax exp/sum processes eight rows per rolled
iteration, and the output epilogue processes two rows per rolled iteration.
First-block softmax and output-accumulation conditions select specialized VFs
outside SIMD row loops, matching the reference's C++ template specialization.
The vendor flags include `-Ofast -mllvm -enable-hiipu-vf-loop-unroll`.

The manual kernel's synchronization uses explicit buffer locks, cross-core flags, unit flags,
and the shared scalar buffer. The skip-scale notification is issued before
gather only at an index-group boundary, and after gather otherwise. Changing
this ordering measurably reduced pipeline overlap in the initial version.
For complete 8/10-block prefill index groups, two live index versions suffice;
shorter groups keep four. This keeps dynamic UB within CANN's 248 KiB budget.
Bounds legalization is disabled because index preprocessing and zero-burst
DMA implement the masks explicitly; invalid pairs must not issue a GM read.

The staged kernel keeps QK and PV as nested scheduled loops. In the common
10-tile query, metadata leads pair gather by one iteration, and gather leads
ND-to-NZ or dequantization by two. The three KV/raw slots cover gather through
transpose/dequantization. For BF16 queries with 5–10 tiles, gather and softmax
read two query slots of index masks directly. The final softmax read precedes
reuse of its query slot; the compiler tracks the full `2 * nb` period. Other
shapes snapshot tile masks into six versions when metadata leads gather, or
five versions otherwise. The compiler emits synchronization and buffer-version
counters for these ownership intervals. Queries outside the Q-slot event
budget use the single-query-slot schedule.

The pipelined source orders conversion, QK, softmax/statistics, metadata,
PV/output, gather, and SS publication by their intended issue order; `T.Stage`
still identifies their logical KV tile. Manual pipe-order constraints keep
gather before the scalar wait for softmax's `delta`, including when the
higher-stage SS store would otherwise tie with gather. No `T.IssuePhase`
annotations are needed. With `pipeline_stages=False`, metadata and gather
remain before their consumers, and SS is published immediately after softmax.

`test_staged.py` covers cross-query rescaling, accumulation, partial index groups,
the flat execution path, and Q-prefetch boundaries at 5, 15, and 16 tiles.
Its independent main/extra decode length tests additionally require
`FLASHMLA_REFERENCE=/path/to/FlashMLA`.

See `RESULTS.md` for the measured results, environment, and limitations.
The historical manual-kernel measurements and newer staged measurements are
reported separately, including the pre-rebase `9c46d1c` manual-order validation in
[staged-manual-order-2026-10-20.json](results/staged-manual-order-2026-10-20.json).
The current compiler's complete performance results are in
[prefill](results/staged-main-rebase-prefill-bench-2026-10-20.json) and
[decode](results/staged-main-rebase-decode-bench-2026-10-20.json), with every
measured launch retained.
Use the per-case runner JSON to compare correctness and
latency for the exact implementation, shape, and compiler build.
