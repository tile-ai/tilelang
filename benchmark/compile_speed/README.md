# Cold Parallel-Compilation Benchmark

Compiles a diverse zoo of realistic transformer inference kernels from a *cold*
cache and reports the wall-clock before/after this PR. It exercises the parallel
AOT path (`par_compile` → `KernelCache.cached` → nvcc) in the *many light
kernels* regime (parallel AOT / autotune sweeps), where launch and disk-save
serialization dominate.

## The zoo

`kernel_zoo.py` builds one translation unit per `(family, shape, tile)` over
real model dims (Qwen2.5 / Llama-3). Families:

| family | kernel |
|--------|--------|
| `gemm` | dense q/k/v/o/gate/up/down projections |
| `gqa`  | GQA flash-attention forward (online softmax) |
| `rmsnorm` | RMSNorm over the hidden dim |
| `silu` | fused SwiGLU activation |
| `softmax` | row-wise softmax (logits / router) |

Default `--scale 3` → **126 distinct kernels**; each is a genuine cold cache miss
(throwaway `TILELANG_CACHE_DIR`, disjoint shapes) that actually runs nvcc.

## Environment

- GPU: `NVIDIA H20`, driver `535.161.08`, 180 CPU cores
- nvcc: CUDA `12.9`

## How to Reproduce

```bash
cd benchmark/compile_speed
python benchmark_compile_speed.py            # ~126 kernels
python benchmark_compile_speed.py --scale 6  # larger zoo for many-core boxes
```

`baseline` reconstructs the pre-PR behavior (disk-save under the global lock,
capped at 32 workers); `current` is the shipped default (lock-free save, worker
count scaled to cores).

## Results

126 cold kernels on an H20, `baseline` = global lock + `min(32, cores)` workers,
`current` = lock-free save + `cores` workers. Speedup vs. core count (best of 2):

| cores | baseline (s) | current (s) | speedup |
|-------|-------------|-------------|---------|
| 8     | 50.7        | 39.2        | 1.29x   |
| 16    | 26.2        | 22.4        | 1.17x   |
| 32    | 19.9        | 17.8        | 1.12x   |
| 64    | 18.3        | 15.3        | 1.19x   |
| 180   | 18.4        | 14.7        | 1.25x   |

Two effects compose. The **lock removal** helps at every core count (it is the
whole win at `cores <= 32`, where both configs use the same worker count). The
**worker scaling** only adds on top past 32 cores, where `current` outgrows the
old `min(32, cores)` cap.

The largest speedups appear on a *fully cold* cache (first run of a fresh
checkout / CI), where per-process PCH build is not yet amortized: 50.3s -> 16.2s
= **3.1x** on the 180-core box. The table above is warm-toolchain (best of 2),
so it isolates the compile-scheduling win from one-time startup cost.

Absolute numbers depend on core count, nvcc version, and disk speed; the
reproducible result is the shape — lock removal is universal, worker scaling
grows with cores and kernel count.
