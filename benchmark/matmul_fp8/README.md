# FP8 Matmul Benchmark (8192×8192)

This document records the throughput achieved by `benchmark_matmul.py` when multiplying FP8 matrices sized `M = N = 8192` across different `K` dimensions. Each measurement relies on the default autotuning search space bundled with the benchmark.

## Environment

- Repository commit: `6b1faf71faf18c564f5f77e0f5c1671cd91dfbc3`
- GPUs: `NVIDIA H800 SXM` on driver `560.35.05`

## How to Reproduce

```bash
cd benchmark/matmul_fp8
python - <<'PY'
from benchmark_matmul import matmul

M = 8192
N = 8192
for K in [256, 512, 1024, 2048, 4096, 8192, 16384]:
    res = matmul(M, N, K, False)
    tflops = 2 * M * N * K / res.latency * 1e-12
    print(f"K={K:5d}  latency={res.latency:.6f}s  TFlops={tflops:.3f}")
PY
```

## Results

| K     | Latency (s) | Throughput (TFLOPs) |
|-------|-------------|---------------------|
|   256 | 0.091488    | 376                 |
|   512 | 0.110496    | 622                 |
|  1024 | 0.148256    | 927                 |
|  2048 | 0.234080    | 1174                |
|  4096 | 0.398944    | 1378                |
|  8192 | 0.752416    | 1461                |
| 16384 | 1.443808    | 1523                |
