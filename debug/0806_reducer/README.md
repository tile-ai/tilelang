# Reducer v2 performance benchmarks

This directory compares the legacy reducer in `tilelang_ref` with the reducer-v2
implementation in the current checkout. Every kernel is checked against a
PyTorch reference before its latency is accepted. If either implementation is
incorrect, the report does not calculate a speedup for that case.

The two checkouts are loaded in separate Python processes. This is required
because TileLang loads native libraries and registers TVM global functions at
import time; importing both versions in one process can silently mix compiler
components.

## Benchmark families

| Family | What it measures | Expected use in the analysis |
|---|---|---|
| `unique_owner` | Independent output elements, with no cross-thread reduction | Measures the v2 LocalComplete plan against legacy `replication="none"`; generated v2 code should have compact partial storage and no collective |
| `row_reduce` | One or more output rows reduced across a K dimension | Compares the normal legacy fully replicated reducer with the v2 canonical fallback |
| `streaming_gemv` | A reducer epoch kept alive across several K tiles | Measures realistic deferred accumulation, including initialization, updates, register pressure, and finalization |

Two correctness diagnostics are included:

- `replica_stress_m1_k8_b512_t128` is the original `BD=8`, 128-thread
  contribution-multiplicity case. The legacy result is expected to be 16 times
  too large, while v2 must return the exact sum.
- `legacy_batch4_m128_k64_b64_t256` exercises the old `run_batch` path, whose
  correctness is not assumed. The first v2 implementation treats `batch` only
  as a hint and uses its scalar fallback.

## Prerequisites

Both source checkouts must already be built in development mode:

```text
<current checkout>/build/lib/libtilelang.so
<legacy checkout>/build/lib/libtilelang.so
```

By default, the driver compares:

```text
v2:     the current TileLang checkout
legacy: the sibling ../tilelang_ref checkout
```

Use `--v2-repo` and `--legacy-repo` to override them.

## Running

First list the complete matrix:

```bash
python debug/0806_reducer/run_benchmarks.py --list
```

Run the quick performance suite:

```bash
python debug/0806_reducer/run_benchmarks.py --suite quick --device 0
```

Run correctness and code generation only, without timing:

```bash
python debug/0806_reducer/run_benchmarks.py --suite quick --no-benchmark
```

Run the full matrix with a longer timing window:

```bash
python debug/0806_reducer/run_benchmarks.py \
  --suite full \
  --warmup-ms 50 \
  --rep-ms 300 \
  --device 0
```

Run the known-problem diagnostics:

```bash
python debug/0806_reducer/run_benchmarks.py --suite diagnostic --no-benchmark
```

Select exact cases or patterns:

```bash
python debug/0806_reducer/run_benchmarks.py \
  --case 'unique_owner_*' \
  --case row_reduce_m1_k128_b512_t128
```

Incorrect kernels are deliberately not timed. For low-level diagnosis only,
`--benchmark-incorrect` opts into collecting their latency; the comparison
still suppresses their speedup.

## Outputs

Each run creates a timestamped directory under `results/` unless
`--output-dir` is supplied:

```text
results/<timestamp>/
├── _cache/{legacy,v2}/
├── comparison.csv
├── legacy.json
├── summary.md
├── v2.json
└── sources/
    ├── legacy/*.cu
    └── v2/*.cu
```

The headline ratio is:

```text
legacy latency / v2 latency
```

A value greater than 1 means v2 is faster. Raw JSON also records compilation
time, error samples, tolerances, source size, and textual counts of generated
`AllReduce`, `run_batch`, named-barrier, and thread-synchronization sites.

## Methodology and caveats

- Both variants use float32 inputs, the same deterministic random seed, the
  same launch geometry, and the same PyTorch reference.
- Warp specialization and TMA lowering are disabled for both versions so the
  reducer comparison is not obscured by unrelated scheduling differences.
- Output buffers are allocated once and passed explicitly, so reported latency
  does not include per-call output allocation.
- Timing uses the median CUDA-event result from TileLang's profiler, including
  its L2-cache flush behavior.
- The comparison spans two repository revisions, so a latency difference can
  include changes outside the reducer. Generated CUDA is retained for checking
  such differences.
- For serious measurements, run both `--order legacy-first` and
  `--order v2-first` into different output directories, keep the GPU idle, and
  control clocks/power state if the machine permits it.
- This matrix evaluates the v2 one-allocation/one-epoch implementation. It
  includes the conservative direct-ownership LocalComplete specialization but
  does not assume a subgroup fast path or a general affine/Fragment ownership
  planner.

Add or tune cases in `benchmark_cases.py`; the worker and comparison report use
that single shared definition.
