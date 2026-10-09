# FlashMLA PR #229: manual Ascend scheduling

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

Supported shapes are H=64, Dqk=Dv=512, BF16 Q and output, and topk multiples of
64. Prefill takes contiguous BF16 KV and allows strided Q / indices. Decode
accepts the PR's compact FP8 main cache and optional FP8 or FP4 extra cache.
Both support attention sinks, invalid indices, and per-query topk lengths.
Packed decode pages must be contiguous, and each cache must be smaller than
4 GiB because gather uses 32-bit byte offsets. Runtime topk lengths mask the
fixed loop extent; they do not shorten this example's pipeline loop.

## Reproduce

Build TileLang at `a35f8ddf45eba16c21211ec8822d56ce5363036f` with Ascend enabled.
Build the pinned FlashMLA reference separately:

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

The prefill runner also accepts `--implementation reference` or
`--implementation manual` to run either variant alone; its default is `both`.
The decode runner always compares the original PR and manual TileLang.

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

## Unrolling and scheduling

Preserved reference choices: four QK/PV tiles and sixteen gather2 DMA
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

See `RESULTS.md` for the measured results, environment, and limitations.
