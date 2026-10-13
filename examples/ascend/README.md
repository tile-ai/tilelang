# Ascend examples

These examples demonstrate complete kernels on Ascend 950. Use the explicit
`tilelang.ascend.language` dialect and an Ascend-enabled build; see the
[backend guide](../../tilelang/ascend/README.md) for setup.
Inputs, reference computations and correctness checks run on the NPU.

| Example | Purpose |
| --- | --- |
| [GEMM](example_gemm.py) | Persistent matrix multiplication with swizzling, precision modes and a mixed-core epilogue |
| [Split-K GEMM](example_gemm_splitk.py) | Atomic or deterministic reduction across K partitions |
| [Vector addition](example_vecadd.py) | SIMT and explicit SIMD with pipelined data movement |
| [RMSNorm](example_rmsnorm.py) | SIMT fragments, reduction and pipelined row processing |
| [FP8 quantization](example_per_token_cast_to_fp8.py) | The same group-wise quantization in SIMT and explicit SIMD |

For advanced GEMM implementations and tuning, see [deepgemm](deepgemm).
Attention kernels are in [flash_attention](flash_attention/README.md).

Run an example directly to check correctness and report device performance.
GEMM and attention report latency and TFLOPS; vector addition, RMSNorm and FP8
quantization report latency and effective bandwidth in GB/s.
Bandwidth counts each logical input and output tensor's bytes once, including
auxiliary outputs such as FP8 scales and RMSNorm RSTD.
GEMM throughput uses `2 * M * N * K` operations, including for Split-K.
Use `--no-bench` on a top-level example to run only correctness checks.

GEMM runs FP8, BF16, FP32, HF32, BF16 output and both accumulation cases.
It uses `M = K = N = 8192`.
Vector addition and FP8 quantization run both SIMT and SIMD by default;
use `--mode simt` or `--mode simd` to select one. Each selected case is timed by default.

```bash
python examples/ascend/example_gemm.py
python examples/ascend/example_gemm_splitk.py --deterministic
python examples/ascend/example_vecadd.py
python examples/ascend/example_per_token_cast_to_fp8.py
python examples/ascend/example_per_token_cast_to_fp8.py --mode simd
python examples/ascend/example_rmsnorm.py --no-bench
python -m pytest examples/ascend/test_rmsnorm.py -q
```

Each top-level `example_<name>.py` has one `test_<name>.py`. Correctness tests use
smaller inputs than performance runs. Feature-level and compiler regressions
live in [testing/ascend](../../testing/ascend/README.md), independently of examples.
For copy padding, buffer annotations and scheduling APIs, consult the
[Ascend programming skill](../../.agents/skills/tilelang-ascend/SKILL.md).

The [performance regression driver](../../maint/scripts/ascend_perf_regression.py)
benchmarks these applications, including both vector programming modes, DeepGEMM and attention.
Language-feature tests are not performance entries.
