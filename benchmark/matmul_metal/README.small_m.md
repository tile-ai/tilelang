# Small-M Metal GEMM

`benchmark_small_m.py` demonstrates a tunable SIMD-group GEMM for small row
counts, such as decode workloads. It compares a conventional 32x32x32 tile
with narrower M tiles. Tile sizes are kernel parameters, not a new compiler
dispatch policy. A narrower tile is a candidate to measure, not a universal
default.

The kernel uses FP16 inputs, FP32 accumulation and output, and 128 threads.
It stages the accumulator through shared memory before copying to the output,
including partial tiles. It requires Apple Silicon with a Metal-enabled
TileLang build and PyTorch MPS; it does not require Metal 4 or MLX.

From the repository root:

```sh
python benchmark/matmul_metal/benchmark_small_m.py --rows 1 2 4 8 16 32 --output sweep.json
python benchmark/matmul_metal/benchmark_small_m.py --rows 1 8 --n 4096 --k 1024 --focused --output wide.json
python benchmark/matmul_metal/benchmark_small_m.py --rows 1 8 --n 1024 --k 4096 --transpose-b --focused --output transposed.json
python benchmark/matmul_metal/benchmark_small_m.py --rows 7 --n 1025 --k 513 --transpose-b --focused --output tails.json
```

Normally B is contiguous `(K, N)`. With `--transpose-b`, B is contiguous
`(N, K)` and the operation is `A @ B.T`, a common weight layout. `--nk`
provides a fallback for either dimension when `--n` or `--k` is omitted.
`--focused` compares only 32x32x32 and 8x32x32; the default also tries
8x64x32, 8x64x64, and 16x64x64.

Each candidate is checked against CPU FP32 matmul of the same FP16 values
before and after timing (`rtol=atol=2e-3`). A sentinel after the output checks
trailing overwrites; it is not a memory sanitizer. A failed check stops the
run rather than silently excluding the candidate.

The default measurement uses seven shuffled rounds of 1,000 calls after
warmup, synchronizing around each batch. JSON contains all samples, medians,
shapes, weight layout, and the imported TileLang checkout revision (or `null`
when Git metadata is unavailable, such as a wheel installation). Latency
includes Python/FFI dispatch and amortized synchronization, not just GPU
execution. Inputs and outputs are reused, so results represent warm-data
throughput per invocation, not isolated request latency. Compilation is
outside the measurement; `compile_ms` times the `tilelang.compile` call
only, not the entire first invocation. Use `TILELANG_DISABLE_CACHE=1` when
checking fresh compilation.

`--torch-reference` optionally includes `torch.mm` with FP32 inputs converted
from the same FP16 values before timing, and a preallocated FP32 output.
It checks the same numerical result, but doubles input storage and excludes
conversion costs. It is a reference path, not an equivalent FP16 performance
baseline. No performance advantage over PyTorch or MLX is implied.

Measure on your target device and dimensions before selecting a tile. These
experiments do not establish a global heuristic, performance on other Apple
GPU generations, or end-to-end serving speedups.
