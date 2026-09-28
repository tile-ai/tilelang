# Ascend profiling and comparison recipes

Contents: [components](#measurement-and-tuning-components),
[benchmark API](#do_bench-units-and-return-types),
[regression drivers](#existing-regression-entry-points),
[source experiments](#generated-source-experiments),
[cycle measurements](#operation-cycle-measurements).

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

## Operation cycle measurements

### Choose the measured region

Start from the operation whose cost or overlap you want to understand in your
kernel. State whether the result covers one operation, a dependent chain,
independent repeated work, or the complete kernel. These boundaries answer
different questions.

| Region | Keep in the experiment |
|---|---|
| SIMD/SIMT VF | Actual function and call, thread dimensions, masks, loop counts, loads/stores, and synchronization |
| Data transfer | Source/destination memory spaces, physical bytes, layout conversion, strides, alignment, and buffer reuse |
| Cube operation | Operand dtype and tile geometry, initialized inputs, accumulator clear/accumulate mode, and dependencies |
| Synchronization handoff | Producer, signal/wait, and consumer; an idle signal alone does not measure readiness under load |
| Combined pipeline | Operation order, buffer versions, overlap, first-iteration fill, and final drain |

### Isolate and compile the operation

Save the kernel's generated `.asc` with its intended pass settings. Copy the
selected operation, includes, and helper dependencies into a small `.asc`
file. Match its execution model with a Cube, Vector, or mixed entry. For a VF,
use a `__global__ __vector__` entry and preserve the original SIMD call or SIMT
`asc_vf_call<vf>(cce::dim3(x, y, z), ...)`, including captured arguments.

Provide aligned storage in the original memory spaces, covering every accessed
offset and preserving alias relationships. Initialize representative inputs,
masks, and scalar parameters; finish preparation and synchronization before
the measured region. For a copy-only or compute-only experiment, make its
inputs ready beforehand. Add the producer back when measuring the dependency
chain or overlap, and report that as a separate boundary.
Keep measurement programs and traces in a temporary directory for the task.

Check the output against a reference before interpreting timing. Use inputs
that distinguish the relevant paths and buffer versions. If intermediate state
cannot be read back, consume it in a checked result after the measured interval.
Inspect the generated code and captured instruction count to confirm the
intended work remains after optimization.

Reuse the kernel's CANN installation, target, optimization flags, TileLang
template include path, and launch ABI. Compile the isolated program with
Bisheng and provide a host launcher that selects a device, allocates buffers,
launches the entry, and synchronizes its stream. Use matching Bisheng, runtime,
and simulator versions; load that installation's `set_env.sh` if needed.

### Record execution

Use `npusim` and inspect `npusim record --help`.
Select the model matching the compiler target and a fresh output directory.
For example, an Ascend950 run with a Python launcher uses:

```bash
npusim record -s Ascend950 -o ./cycle-profile -g -n 0 \
  'python /absolute/path/to/kernel_launcher.py'
```

In the current CLI, `-n 0` enables core 0 logs; it does not select a kernel or
change the launch grid. Select the kernel's core count in the launcher. Use
`-n all` or the required core range when the measurement spans several cores;
one core's completion does not establish whole-kernel completion.

For a Python launcher, prepare inputs and reference results on the CPU, then
transfer the inputs to the device. Avoid NPU random generation, quantization,
or warmup kernels before the intended capture. Confirm the captured kernel
identity and launch count instead of assuming the recorder selected the intended
wrapper. Use separate processes for independent captures; several cases may
share one wrapper if each has a distinct trace interval and completed setup.

Keep the selected CANN runtime and simulator libraries in the launch
environment, including the child process started by the recorder. If imports
or device initialization fail, inspect that child's interpreter, library paths,
and simulator-visible device IDs. Preserve simulator library precedence when
adding a missing CANN runtime path; host device visibility may not match the
simulator's devices. Wait for kernel completion and recording flush, then
check the log. Missing or truncated target instructions invalidate the cycle
estimate even if the launcher exits successfully.

### Read the cycle interval

Locate `instr.bin` or `chip*_instr.bin`, possibly under an `npusim_*/record`
directory. Decode it with the same CANN installation's
`cannsim.core.public.instr_decoder`, or use its instruction timeline report.
Check timestamp units and issue/completion semantics before calculating cycles.

Pair events using the installed reader's semantics and the launch, chip, core,
subcore, instruction ID, and PC. A PC alone is insufficient for repeated loop
iterations. Do not infer timing semantics from the name `is_popped`: in the
verified CANN 9.2 reader, `1` marks dispatch and `0` marks completion. Recheck
this mapping for another tool version, and compare extracted intervals with
the generated timeline and a known producer/consumer dependency.

For the selected region, measure from its first instruction's start to its
last instruction's completion. A VF uses its RVEC/VECTOR events; a copy, Cube,
or combined region uses the corresponding pipe events and required handoffs.
Exclude preparation outside the chosen boundary. For duration events,
the interval is `max(start + duration) - min(start)`; express it in cycles
using the report's units. State whether the start event is dispatch or actual
execution: dispatch-to-completion includes queue and dependency waits, so it
does not by itself isolate instruction service time. Include loads, stores,
and synchronization inside the original region when they are part of its cost.
Keep the trace showing the selected boundaries; summing overlapping instruction
durations does not give elapsed time. For a complete kernel, include the launch
and all participating cores through final completion, rather than reporting
only the busiest operation's interval.

Keep raw cycles alongside any time conversion. Verify the report's clock
domain and frequency before converting to nanoseconds; simulator host wall
time is not device latency. Record whether input data is already resident and
whether cache state is controlled. A fresh process alone does not prove a cold
cache in the model.

### Repeated operations and throughput

When measuring repeated work, distinguish an isolated operation's latency
from the initiation interval (II) of a stream of operations. A dependent chain
measures a different limit from independent operations. For the latter, keep
physical buffers or registers distinct and live until their results are
consumed; different source-level names need not imply independent storage.
Reusing a destination can introduce write-after-write or consumer dependencies.

Sweep a few repetition counts and relevant work sizes, masks, strides, and
thread dimensions. Keep preparation outside the interval and place required
drain synchronization after the stream; a barrier after every operation
serializes the experiment. Compare steady issue and completion spacings with
the burst-duration slope. A relation such as
`T(P) = L + (P - 1) * II` is useful only when the trace shows a stable pipeline;
startup, queue limits, scalar issue cost, and buffer reuse may change the slope.
An empty wrapper can expose fixed overhead, but subtracting overlapping
pipeline intervals mechanically can produce a misleading estimate.

Check a conclusion on a different work size or repetition count before using
it to change the kernel. Preserve the source, binary, build flags, target,
tool versions, input conditions, correctness result, and raw trace with each
measurement. Report simulation coverage separately from real-device testing,
and validate the actual kernel after applying an annotation or rescheduling.

### Override automatic task estimates

Use `T.Task(latency=..., ii=...)` to supply measured cycle costs to AutoSchedule
for a specific operation or group of operations in your kernel. Both arguments
are optional compile-time Python integers. Each supplied value replaces that
cost for the **complete task body**; an omitted value is still estimated.
The final pair must satisfy `latency >= ii > 0`, including when one value comes
from automatic estimation.

```python
# measured_latency and measured_ii are Python integers from an experiment
# matching this copy's shape, dtype, layout, and memory spaces.
with T.Task(latency=measured_latency, ii=measured_ii):
    T.copy(src, dst)
```

- `latency` is the number of cycles from task issue until its outputs are ready.
  Use a measurement with inputs ready and unrelated queues drained; a copy
  waiting behind other work does not establish its intrinsic task cost.
- `ii` is the minimum interval before the same hardware pipe can issue another
  instance. Estimate it from a sustained stream of independent work, checking
  issue/completion spacings and the burst-duration slope. It is not the solved
  II of the enclosing loop.
- With several operations inside one `T.Task`, measure and annotate the whole
  group. Do not reuse a single instruction's latency or II as the group's cost,
  or divide total burst cycles by the count and use that as both values.

All statements in one `T.Task` must use the same hardware pipe and AIC/AIV core
affinity. They issue in source order without internal synchronization. Split
cross-pipe sequences, such as a data load followed by Cube computation, into
separate tasks; see [task boundaries](../SKILL.md#24-task-boundaries).

The overrides change the scheduler's cost assumptions. They do not insert
waits inside the task or make an otherwise invalid dependency sequence legal.
Keep the matching measurement conditions with the values, recompile to apply
them, then check correctness and benchmark the complete kernel. Remeasure when
the task body, shape, layout, target, or toolchain changes.
