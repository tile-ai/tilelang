# Autotuning

TileLang includes a built‑in autotuner that searches configuration spaces
for the best performing kernel, compiles candidates in parallel, validates
correctness, benchmarks them, and caches the best result for reuse.

This guide covers two workflows:
- Decorator‑based: `@tilelang.autotune(configs=...)` stacked on `@tilelang.jit`
- Programmatic: `AutoTuner.from_kernel(...).set_*().run()`

It also explains input tensor supply, validation, caching, and environment
variables that affect parallelism and cache behavior.

## 1) Decorator‑based Autotune

Use `@tilelang.autotune` above `@tilelang.jit` and expose tunable parameters as
function arguments with defaults. The autotuner overrides these parameters with
values from your config space.

```python
import tilelang
import tilelang.language as T

def matmul_configs(M, N, K):
    # Example space — tailor to your target
    tiles = [64, 128]
    stages = [2, 3]
    threads = [128, 256]
    return [
        dict(block_M=BM, block_N=BN, block_K=BK, num_stages=S, threads=TH)
        for BM in tiles
        for BN in tiles
        for BK in [32, 64]
        for S in stages
        for TH in threads
    ]

@tilelang.autotune(configs=matmul_configs, warmup=25, rep=100, timeout=60)
@tilelang.jit(out_idx=[-1])
def matmul(M: int, N: int, K: int,
           block_M: int = 128, block_N: int = 128, block_K: int = 32,
           threads: int = 128, num_stages: int = 3,
           dtype: str = 'float16', accum_dtype: str = 'float32'):

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype),
               B: T.Tensor((K, N), dtype),
               C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_f = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_f)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_f)

            T.copy(C_f, C[by * block_M, bx * block_N])

    return kernel

# Usage
# Provide inputs via context (recommended for reproducibility across configs)
import torch
M = N = K = 1024
A = torch.randn(M, K, device='cuda', dtype=torch.float16)
B = torch.randn(K, N, device='cuda', dtype=torch.float16)
C = torch.empty(M, N, device='cuda', dtype=torch.float16)

from tilelang.autotuner import set_autotune_inputs
with set_autotune_inputs(A, B, C):
    tuned_kernel = matmul(M, N, K)   # compiles, tunes, returns best kernel
    tuned_kernel(A, B, C)            # run best kernel
```

Notes
- `configs` can be a list of dicts or a callable `(args...) -> list[dict]`. Each
  dict’s keys must match the tunable function arguments (e.g., `block_M`).
- The decorator returns a callable that runs autotune once per argument tuple
  and caches the resulting best kernel in‑process.
- For explicit input control during tuning, wrap the call with
  `set_autotune_inputs(...)`. Otherwise, `supply_type` (below) is used.

## 2) Programmatic Autotune

Use the `AutoTuner` class to manage configs and arguments more explicitly.

```python
from tilelang.autotuner import AutoTuner

kernel_factory = matmul  # the function above (already @tilelang.jit)
tuner = AutoTuner.from_kernel(kernel_factory(M, N, K), configs=matmul_configs(M, N, K))

tuner.set_profile_args(
    warmup=25, rep=100, timeout=60,
    supply_type=tilelang.TensorSupplyType.Auto,  # or provide supply_prog/ref_prog
    ref_prog=lambda A, B, C: torch.allclose(C, (A @ B).to(C.dtype), rtol=1e-2, atol=1e-2),
)

tuner.set_compile_args(
    target='auto',                  # or 'cuda'/'hip'/'metal'
    execution_backend='auto',       # resolves per-target
    out_idx=[-1],                   # which outputs to return if multiple
    pass_configs={                  # optional TVM passes/flags
        # tilelang.PassConfigKey.EXAMPLE_KEY: value,
    },
)

artifact = tuner.run()             # compiles + runs + validates all configs
best_kernel = artifact.kernel      # JITKernel
best_latency = artifact.latency
best_config = artifact.config

# Reuse best kernel
best_kernel(A, B, C)
```

### Example Gallery (in repo)
- examples/gdn/example_chunk_delta_h.py:101 — uses `@autotune` to sweep configs
- examples/deepseek_nsa/benchmark/benchmark_nsa_fwd.py:451 — uses `@tilelang.autotune`
- examples/quickstart.py:84 — profiles a tuned kernel with `get_profiler`
- examples/hadamard_transform/example_hadamard.py:152 — profiler with custom warmup
- examples/dynamic_shape/example_dynamic.py:94 — profiler for dynamic shapes
- examples/gemm/example_gemm_persistent.py:135 — compare persistent vs non‑persistent

Click any path to open the code and compare patterns.

## Input Tensor Supply

The tuner needs inputs to compile and benchmark kernels. Provide them in one of
three ways (priority order):

1) Context manager (fixed inputs across configs)
```python
with set_autotune_inputs(A, B, C):
    tuned = matmul(M, N, K)
```

2) Custom supplier program
```python
def supply_prog(signature):
    # signature holds KernelParam objects describing shapes/dtypes
    # Return a list of torch tensors matching the kernel’s arguments
    return [A, B, C]

tuner.set_profile_args(supply_prog=supply_prog)
```

3) Built‑in generators via `supply_type`
- `TensorSupplyType.Auto` (default): heuristic per dtype (uniform ints / fp ranges)
- `Integer`, `Uniform`, `Normal`, `Randn`, `Zero`, `One`

Important
- Built‑in generators require static shapes; if your PrimFunc uses symbolic
  dimensions (T.dyn), supply concrete inputs via (1) or (2).
- Float8 dtypes require PyTorch 2.1+ for `torch.float8_*` support.

## Correctness Checking and Tolerances

Use one of the following validation methods:
- `ref_prog`: Provide a reference program that receives the same inputs and
  checks results. You can return a boolean or raise on mismatch.
- `manual_check_prog`: A callable that inspects outputs and raises on mismatch.
- `skip_check=True`: Skip correctness checks (faster, use with caution).

Control numeric drift via:
- `rtol` and `atol` (defaults 1e‑2)
- `max_mismatched_ratio` (default 1%)

## Configuration Spaces and Best Practices

What to tune
- Tile sizes: `block_M`, `block_N`, `block_K`
- Software pipelining: `num_stages`
- Threads per block: `threads` (or (x, y) tuple)
- Optional: dtype variants, epilogues, small scheduling knobs

Tips
- Start from a working baseline. Tune a small, meaningful space first.
- Respect hardware limits (shared memory bytes, registers per thread/block,
  max threads per block). Eliminate impossible configs up‑front.
- Keep block sizes multiples of vector widths and warp sizes when relevant.
- Use `set_autotune_inputs` to ensure each config is measured on identical data.
- Record your best configs and bake them as defaults when stable.

## Parallel Compilation/Benchmarking and Timeouts

The tuner compiles configurations in parallel using a thread pool and benchmarks
them with a per‑config timeout. On CUDA, each worker sets the current device to
avoid context issues.

Notes
- `timeout` uses POSIX signals; on non‑Unix systems, it may not take effect.
- Logs are written to `autotuner.log` in the working directory.

## Caching

The autotuner caches best artifacts both in‑memory (per process) and on disk under
`$TILELANG_CACHE_DIR/autotuner`. The cache key includes:
- TileLang version, function source, closure free‑vars
- Config list, compile args, profile args

Disk cache contents (per key)
- Best config and latency: `best_config.json`, `latency.json`
- Kernel sources and library: `device_kernel.cu`, `host_kernel.cu`, `kernel_lib.so` (or `kernel.cubin`/`executable.so` depending on backend)
- Function and params: `function.pkl`, `params.pkl`

Control via env vars (tilelang.env)
- `TILELANG_CACHE_DIR` (default `~/.tilelang/cache`)
- `TILELANG_TMP_DIR` (default `$TILELANG_CACHE_DIR/tmp`)
- Disable all kernel caches: `TILELANG_DISABLE_CACHE=1`
- Disable autotune disk cache only: `TILELANG_AUTO_TUNING_DISABLE_CACHE=1`

CPU worker control
- `TILELANG_AUTO_TUNING_CPU_UTILITIES` (fraction, default 0.9)
- `TILELANG_AUTO_TUNING_CPU_COUNTS` (int, `-1` auto)
- `TILELANG_AUTO_TUNING_MAX_CPU_COUNT` (int, `-1` unlimited)

Backend notes
- NVRTC backend persists `.cubin` and a Python launcher.
- Torch/DLPack backend may not save artifacts to disk; in this case, only
  in‑memory caching applies and a warning is logged.

## Alternative: Manual Sweeps with par_compile

If you prefer manual control, use `JITImpl.par_compile` to compile a batch of
configs and drive your own benchmarking:

```python
@tilelang.jit
def factory(M, N, K, block_M=128, block_N=128, block_K=32):
    @T.prim_func
    def k(A: T.Tensor((M, K), 'float16'),
           B: T.Tensor((K, N), 'float16'),
           C: T.Tensor((M, N), 'float16')):
        ...
    return k

impl = factory  # JITImpl
cfgs = [
    dict(block_M=64, block_N=128, block_K=32),
    dict(block_M=128, block_N=128, block_K=64),
]
kernels = impl.par_compile(cfgs, num_workers=4)
# Now benchmark kernels[i](A, B, C) yourself
```

## Recording and Reusing Best Configs

The programmatic path returns an `AutotuneResult` that can be saved and later
reloaded. This is useful for CI, multi‑host workflows, or shipping tuned configs.

```python
artifact = tuner.run()  # AutotuneResult

# Save to disk
from pathlib import Path
save_dir = Path('out/best/matmul_1024')
artifact.save_to_disk(save_dir, verbose=True)

# Reload later
from tilelang.autotuner.param import AutotuneResult, CompileArgs
restored = AutotuneResult.load_from_disk(save_dir, CompileArgs())
best = restored.kernel
best(A, B, C)
```

Notes
- DLPack/Torch execution backend may not persist compiled binaries; in that
  case, re‑compilation is needed on load or use a different backend.
- The directory contains human‑readable JSONs (best config/latency) and sources.

## Advanced: Config Space Callables

Derive config spaces from problem sizes to keep searches targeted and legal:

```python
def matmul_configs(M, N, K):
    large = min(M, N, K) >= 1024
    tiles = [128] if large else [64, 128]
    for BM in tiles:
        for BN in tiles:
            for BK in [32, 64]:
                for S in [2, 3]:
                    for TH in [128, 256]:
                        yield dict(block_M=BM, block_N=BN, block_K=BK,
                                    num_stages=S, threads=TH)
```

## Device and Backend Selection

Tune compile‑time options explicitly:
- `target='auto'|'cuda'|'hip'|'metal'` (normalized to a TVM Target)
- `execution_backend='auto'|'tvm_ffi'|'cython'|'nvrtc'|'torch'`
- `pass_configs={...}` to toggle TileLang/TVM passes for experiments

On CUDA with multiple GPUs, the tuner sets the current device per worker thread
to avoid context mixups.

## Troubleshooting
- “No configurations to tune”: Ensure `configs` is a non‑empty list or callable.
- Timeouts: Increase `timeout`; ensure inputs fit device memory; verify that
  your reference check isn’t the bottleneck.
- Dynamic shapes: Provide concrete inputs via `set_autotune_inputs` or a custom
  `supply_prog`.
- Disk cache disabled: Check `TILELANG_AUTO_TUNING_DISABLE_CACHE` and backend.
