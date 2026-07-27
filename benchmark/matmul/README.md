# FP16 Matmul Benchmark (8192×8192)

This document records the throughput achieved by `benchmark_matmul.py` when multiplying FP16 matrices sized `M = N = 8192` across different `K` dimensions using the default autotuning search space.

## Environment

- Repository commit: `17bd0a6c651f599bec1397e0b91830c3ddc93076`
- GPUs: `NVIDIA H800 SXM` on driver `560.35.05`

## How to Reproduce

```bash
cd benchmark/matmul
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
|   256 | 0.089056    | 386                 |
|   512 | 0.132064    | 520                 |
|  1024 | 0.218816    | 628                 |
|  2048 | 0.390112    | 705                 |
|  4096 | 0.746752    | 736                 |
|  8192 | 1.449888    | 758                 |
| 16384 | 2.871168    | 766                 |
