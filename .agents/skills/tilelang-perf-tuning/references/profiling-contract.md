# Profiling Contract

This file defines the unified device-side collection methodology for performance tuning. When a CSV field is uncertain, read the actual header; do not maintain an easily stale field catalog.

## Entry Gate

Every B0, B*, and Ci regenerates a standalone profiling entry from its latest source. The entry:

- Embeds all `CASES` in the locked order;
- Accepts only `--device` and the optional `--warm-up`, `--launch-count`;
- Selects the physical device before creating NPU tensors;
- Calls the local target kernel inside the entry, or a production wrapper that has been explicitly verified;
- Never uses host timing as kernel performance;
- Records the operator source SHA, case SHA, import path, actual arguments, kernel names, and launch mapping.

First invoke the same entry directly and confirm compilation, execution, and the full case order:

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<target> \
  python <entry.py> --device <physical_id>
```

An old entry must never measure new source. Collecting a baseline indirectly through a third-party benchmarking wrapper and mixing its host overhead into the kernel methodology is forbidden.

## `msprof op`

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<target> \
  msprof op --warm-up=10 --launch-count=<total_target_launches> \
  --output=<new_dir> --kernel-name=<resolved_kernel> \
  python <entry.py> --device <physical_id>
```

`--launch-count` caps the number of collected launches, not the number of cases. Set it to the total number of target-kernel launches across all cases: sum the per-case launch counts from the launch mapping, counting repeated launches of the target kernel within one case. If the installed profiler counts warm-up launches toward this limit, include them; confirm the counting behavior before fixing the value. A value derived from the case count alone can stop collection before the complete chain is captured.

Require the complete kernel directory count, launch index, case order, and the eight regular CSV categories. Each retry uses a new output directory; never overwrite or mix old data.

Run the device health check only once and select the physical device by status and occupancy. Do not wait globally just because an msprof process exists on another device. When device ID mapping looks wrong, first remove the inherited `ASCEND_RT_VISIBLE_DEVICES` and pass the physical ID explicitly; do not attribute it to a kernel performance failure.

## Regular `msprof` Fallback

Use it only when all of the following hold: the same latest entry runs correctly when invoked directly; the failure is clearly in DBI/instrumentation/export rather than compilation, execution, precision, kernel names, or case mapping; and a brand-new output directory is used.

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<target> \
  msprof --output=<new_dir> \
  --application="python <entry.py> --device <physical_id> --warm-up 10 --launch-count 2" \
  --task-time=on --aic-mode=task-based --aic-metrics=PipeUtilization
```

The entry performs warmup followed by formal launches for each case. After export, filter by the complete target kernel name; the matched row count must equal `Σ per-case target-kernel launches × (warmup + formal)`. Derive the expected count and the warmup groups from each case's launch mapping, including repeated target-kernel launches. Group rows by case order and drop each group's warmup. When kernel names, row counts, order, or SHAs are inconsistent, stop — never guess the mapping.

The fallback summary states the collection method and the missing fields. Final comparisons prefer a single collection method; when one side falls back, the other side is re-collected with regular msprof as well.

## Archiving and Summary

Use the following for `msprof op` results:

```bash
python3 <skill_dir>/scripts/perf_summary.py \
  <launch_dir> <variant_case_dir> \
  --kernel-name <resolved_kernel> --round-name <round>
```

The script archives the raw CSVs, normalizes the file names, verifies the target kernel is unique, and generates `summary.txt`. Read the summary first; open the raw CSVs/timeline only for anomalies, unclear directions, near-noise results, or precise attribution.

Regular msprof cannot use the script that requires the eight CSVs; archive the task/op summaries, filtered results, case mapping, warmup/formal counts, and SHAs yourself, and produce a per-case summary noting the missing metrics.

## Time Methodology

- AIV-only: use the device-side time of the target AIV kernel.
- AIC-only: use the device-side time of the target AIC kernel.
- Mixed AIC/AIV: define the critical path from the actual cooperation and overlap; never simply add all pipe ratios.
- Multiple kernels: record each case's branches, kernel names, and invocation counts; whole-chain acceptance covers every launch, workspace, event, and synchronization the public call needs. Per-core collection is for attribution only.
- Repeated launches of the same version follow the pre-declared mean/median/stable-value rule; an anomalous round can be dropped only with device frequency, concurrency, or tooling evidence.

When a missed kernel, missed launch, or wrong aggregation is discovered, discard the bad baseline and re-measure; never modify the shared timer to manufacture an improvement.

## Attribution, Comparison, and Noise

The summary reports at least the kernel time, inter-core min/avg/max, available pipes, frequency, and anomalies. Combine it with the source to compute logical GM bytes, Vector/Cube work, and lower bounds:

```text
effective_bandwidth = logical_bytes / kernel_time
effective_compute = logical_ops / kernel_time
```

Effective bandwidth is not physical HBM utilization. A single pipe ratio cannot be mapped directly onto a source-level operation; verify with generated code, traffic, timelines, or isolated candidates. Pipeline overlap cannot be proven by summed ratios, `num_stages`, or buffer counts alone.

Keep the same inputs, dtypes, output semantics, device, frequency, warmup, formal counts, concurrency, and kernel aggregation methodology between baseline and candidates. Every candidate collects all target cases. When differences approach noise, alternate re-measurements as `B* → Ci → B* → Ci`. pytest benchmarks or host timers cannot support final promotion.

A profiler instrumentation/export failure is recorded as an environment/tooling block; it never eliminates a candidate or declares convergence.
