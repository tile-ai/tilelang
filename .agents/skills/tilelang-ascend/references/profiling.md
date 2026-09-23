# Ascend profiling and comparison recipes

Contents: [components](#measurement-and-tuning-components),
[benchmark API](#do_bench-units-and-return-types),
[regression drivers](#existing-regression-entry-points),
[source experiments](#generated-source-experiments),
[VF simulation](#vf-simulation-and-latency-estimates).

## Measurement and tuning components

| Component | What it owns |
|---|---|
| `tilelang/profiler/` | Prepared kernel benchmarking, input/reference checks, timing backend dispatch, msprof capture and interpretation |
| `tilelang/autotuner/` | Candidate elaboration/compilation, correctness and benchmark work, selection and persisted tuning results |
| `tilelang/carver/` | Template/architecture-based configuration hints; inspect supported architecture policies before assuming Ascend applicability |
| `tilelang/ascend/transform/z3_scheduler.py` | In-kernel task scheduling model; solved II is not measured device latency |
| `tilelang/ascend/language/tile_schedule.py` | Persistent core-to-tensor-tile mapping; affects work distribution and locality |
| `maint/scripts/ascend_perf_regression.py` | Ascend benchmark case inventory and execution |
| `tilelang/instrumentation/`, `tilelang/tools/pass_timing.py` | Compiler-phase diagnostics; compile time is separate from steady-state execution |

For an autotuning comparison, record the candidate set, compiled configuration,
input/reference supplier, output contract, measurement boundary, and cached
result identity. Check which tuner features support the selected target.

## `do_bench` units and return types

Read `tilelang/profiler/bench.py` for the selected revision. The current API uses
`warmup` and `rep` in milliseconds; `_n_warmup` and `_n_repeat` are iteration
overrides. The current API accepts an explicit NPU device object or string;
integer device indices select CUDA/HIP. Select the NPU before preparing tensors
and keep it consistent with the benchmark's device context.

```python
import statistics
from tilelang.profiler import do_bench

# run() launches the prepared workload on the selected current device.
samples_ms = [do_bench(run, backend="msprof", warmup=25, rep=100, early_stop_baseline=None) for _ in range(5)]
latency_ms = statistics.median(samples_ms)
profile = do_bench(run, backend="msprof_detail", early_stop_baseline=None)
detail_latency_ms = profile.dur_ns / 1_000_000
```

Use the original harness's parameters when comparing existing results; these
values illustrate the API. The median above aggregates complete benchmark
samples. The msprof helper does not use `return_mode` or `quantiles` as a
per-kernel median selector.

| Backend | Current result |
|---|---|
| `msprof` | Numeric latency in milliseconds |
| `msprof_detail` | `KernelProfile`, including `dur_ns`, AIC/AIV cycle counts, and pipe ratios |

`early_stop_baseline` can return an event-based estimate before msprof runs.
Disable it for controlled profiler comparisons. The helper warms its cache
flush kernel before capture, filters that kernel out of parsed profiles, and
sums remaining kernel durations. If the callable launches multiple kernels,
interpret the aggregate accordingly and inspect individual records as needed.

Check profiler startup and parsing logs. Empty records, startup failures, or
missing target kernels invalidate a profiler claim even if a surrounding test
passes. Detailed ratios are relative to the corresponding AIC or AIV cycles;
do not add them as mutually exclusive fractions of one common time interval.

## Existing regression entry points

The Ascend benchmark suite is listed in `_ENTRIES` in
`maint/scripts/ascend_perf_regression.py`; each entry calls an example's
`run_regression_perf` function. Select an entry with `TL_PERF_REGRESSION_ENTRY`,
or run the full suite with the following command. Set the per-entry timeout in
seconds with `TL_PERF_REGRESSION_TIMEOUT`.

```bash
python maint/scripts/ascend_perf_regression.py
```

Compare emitted names and result counts with the expected entries; caught
failures can leave empty results without a failing exit code. Retain the logs
and `__TILELANG_PERF_RESULTS_JSON__=` records with the measurement results.
Avoid parallel benchmark workers contending for the same device.

For another project's harness, inspect its current CLI, case generation, and
baseline format. Preserve its measured callable, cache-flush policy, and
required initialization when extracting a smaller benchmark.

## Generated-source experiments

`tilelang.ascend.callback.register_ascend_postproc_callback` accepts a function
`(code, target) -> code`. See
`examples/ascend/example_ascend_postproc_callback.py`. It is a global callback,
so scope experiments to an isolated process and filter by target and the exact
source marker. Verify a replacement hits exactly the intended operation and
save the original and modified source.

A postprocessing change can isolate an instruction, cache hint, or ordering
hypothesis without first rewriting a pass. It does not establish that the
frontend or dependency model already supports that change. Confirm a fresh
compile actually applies the callback; cached artifacts can bypass the
experiment. Validate results before interpreting faster timing.

## VF simulation and latency estimates

### Isolate and compile the VF

Save the kernel's generated `.asc` with its intended pass settings. Copy the
selected VF definition, includes, and helper dependencies into a small `.asc`
file, with a `__global__ __vector__` entry that invokes it once. Preserve the
original SIMD call or SIMT `asc_vf_call<vf>(cce::dim3(x, y, z), ...)`, including
all thread dimensions and captured arguments.

Provide aligned GM/UB storage covering every accessed offset and preserving
alias relationships. Initialize representative inputs, masks, and scalar
parameters; finish input preparation and synchronization before the VF runs.
Keep measurement programs and traces in a temporary directory for the task.

Reuse the kernel's CANN installation, target, optimization flags, TileLang
template include path, and launch ABI. Compile the isolated program with
Bisheng and provide a host launcher that selects a device, allocates buffers,
launches the entry, and synchronizes its stream. Use matching Bisheng, runtime,
and simulator versions; load that installation's `set_env.sh` if needed.

### Record execution

Use `npusim` and inspect `npusim record --help`.
Select the model matching the compiler target and a fresh output directory.
For example, an Ascend950 run uses:

```bash
npusim record -s Ascend950 -o <output-dir> <executable-launcher>
```

Keep the selected CANN runtime and simulator libraries in the launch
environment. Wait for kernel completion and recording flush, then check the
log. Missing or truncated vector instructions invalidate the cycle estimate.

### Read the cycle interval

Locate `instr.bin` or `chip*_instr.bin`, possibly under an `npusim_*/record`
directory. Decode it with the same CANN installation's
`cannsim.core.public.instr_decoder`, or use its instruction timeline report.
Check timestamp units and issue/completion semantics before calculating cycles.

For the selected VF on one chip/core/subcore, measure from the first vector
instruction's start to the last one's completion. Select its RVEC/VECTOR events,
excluding buffer preparation and scalar launch work. For duration events,
the interval is `max(start + duration) - min(start)`; express it in cycles
using the report's units. Keep the trace showing those boundaries. A dispatch
interval or host wall time measures a different quantity.
