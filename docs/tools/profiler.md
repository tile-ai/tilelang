# Profiling and Benchmarking

TileLang keeps one `Profiler` and one `do_bench` entry point across runtime
devices. The `backend` argument selects a **timing method**, not a compiler
target or an execution backend such as `tvm_ffi`, `nvrtc`, or `cython`.

## Timing Methods

| Runtime device | Method | Measurement |
| --- | --- | --- |
| CUDA / ROCm | `event` (default) | Device-event intervals, with cache flushing before each sample. |
| CUDA / ROCm | `cupti` | Torch profiler kernel time, excluding the annotated cache-flush work. |
| CUDA / ROCm | `cudagraph` | Graph replay time divided by captured call count; cache flushing precedes each replay. |
| Metal / CPU | `wall` | Synchronized batch wall time divided by call count, including host overhead and without cache flushing. |

CUDA and ROCm reuse the existing Torch GPU timing implementations. The historical
`cupti` name remains an API compatibility spelling for the Torch profiler path.
Availability of tracing and graph capture still depends on the installed runtime
and on whether the callable supports the requested operation.

The default remains `event`. Select `wall` explicitly for Metal or CPU. Unsupported
device/method combinations raise before executing the callable; no method is
silently substituted. Other runtime device types are not currently supported.

All methods return **milliseconds**, but they measure different quantities.
Do not compare an event interval, summed kernel time, and a wall-clock batch as
if they were interchangeable measurements.

## Device Selection

For a kernel profiler, an explicit `device` takes precedence over input-device
inference. Without either, the already-resolved kernel target determines the
device family, and the current device determines the GPU ordinal. Input tensors
must share the selected device and be compatible with the kernel target.

Input generation, reference checks, kernel checks, and timing use the same
device. A CPU-targeted kernel does not allocate CUDA inputs merely because the
host also has a GPU. Explicit device scopes are restored after profiling.

Standalone `do_bench` does not inspect a callable's closure. Pass `device` when
benchmarking work outside the default runtime device. Integer devices retain
their meaning as CUDA/ROCm device ordinals.

```python
from tilelang.profiler import do_bench

latency_ms = do_bench(lambda: operation(inputs), device=inputs.device)

cpu_latency_ms = do_bench(lambda: cpu_operation(cpu_inputs), device="cpu", backend="wall")

metal_latency_ms = metal_kernel.get_profiler().do_bench(backend="wall")
```

Autotuning accepts the same method through `set_profile_args(backend="wall")`.
This does not extend the existing multi-GPU worker scheduling to other device
families.

## Iteration Counts and Results

`warmup` and `rep` are time budgets in milliseconds. Positive `_n_warmup` and
`_n_repeat` override iteration counts in `do_bench`; the equivalent kernel
profiler arguments are `n_warmup` and `n_repeat`.

The wall-clock method measures ten batches. A manual repeat count applies to
each batch; automatic repeat selection divides the measurement budget among
them. Its statistics describe per-call batch averages, not individual call
latencies. Synchronization is outside the timed interval before the first batch
and included at the end of every batch.

`return_mode` supports `min`, `max`, `mean`, and `median`. A single requested
quantile returns a scalar; multiple quantiles return a list. The historical
`cupti` method returns mean kernel time only. An early-stop estimate preserves
the existing behavior of returning one estimate per requested quantile,
including a one-element list for a single quantile.

`fast_flush` and `cache_size` apply only to GPU methods. Wall-clock timing does
not allocate a cache-flush buffer. Keep the timing method and cache policy fixed
when comparing autotuning candidates.
