Auto-Tuning Techniques for Performance Optimization
===================================================
<div style="text-align: left;">
<em>Author:</em> <a href="https://github.com/yyttt6">yyttt6</a>
</div>

## Overview

Auto-tuning a Tile Language program involves three main steps:

1. Implement the target program using Tile Language with reserved optimization parameters
2. Provide explicit candidate configurations for the native TileLang kernel.
3. Parallel compile and benchmark candidate configurations to identify the best performance

## Matrix Multiplication Example

The following example demonstrates auto-tuning matrix multiplication. Code has been simplified for readability - see `examples/gemm/example_gemm.py` for complete implementation.

### Step 1: Implement with Reserved Parameters
Users can implement matrix multiplication in Tile Language while reserving parameters for optimization:
```python
# Reserved parameters for optimization
def kernel(
    block_M=None,
    block_N=None,
    block_K=None,
    num_stages=None,
    thread_num=None,
    enable_rasteration=None,
):
    dtype = "float16"
    accum_dtype = "float"

    # Matrix multiplication implementation
    @T.prim_func
    def main(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((M, N), dtype),
    ):
        # ...existing code...

    return main
```
### Step 2: Generate Candidate Configurations
Manually define configurations or use combinatorial generation:
```python
configs = [
    {"block_M": 128, "block_N": 128, "block_K": 128, "num_stages": 3, "thread_num": 128, "enable_rasteration": True},
    {"block_M": 32, "block_N": 32, "block_K": 32, "num_stages": 0, "thread_num": 32, "enable_rasteration": False},
    # ...additional configurations...
]
```
It can also be given by combinatorial traversal of different parameters
```python
import itertools

block_M = [64, 128, 256]
block_N = [64, 128, 256]
block_K = [32, 64]
num_stages = [0, 1, 2, 3]
thread_num = [128, 256]
enable_rasterization = [True, False]
_configs = list(
    itertools.product(
        block_M,
        block_N,
        block_K,
        num_stages,
        thread_num,
        enable_rasterization,
    )
)

configs = [
    {"block_M": c[0], "block_N": c[1], "block_K": c[2], "num_stages": c[3], "thread_num": c[4], "enable_rasteration": c[5]}
    for c in _configs
]
```
### Step 3: Compile and Benchmark
Configure JIT compilation and benchmarking settings:
```python
autotuner = AutoTuner.from_kernel(kernel=kernel, configs=get_configs(M, N, K)).set_compile_args(
    out_idx=[-1],
    supply_type=tl.TensorSupplyType.Integer,
    ref_prog=ref_program,
    skip_check=False,
    target="auto",
)
result = autotuner.run(warmup=3, rep=20)
out_c = result.kernel(a, b)
```
The result object contains optimized kernel implementation which can be used by users directly

## Standalone TIRx migration

Autotuning compiles and benchmarks native TileLang kernels using `tvm-tirx`.
Supply candidate dictionaries directly, as in the examples above; no TVM
Schedule or TE representation is required.

The legacy Carver template/policy generator is retired. Hardware queries under
`tilelang.carver.arch` remain available to filter candidates by shared-memory,
thread-count, and architecture limits. The former `with_roller=True` example
option raises a migration error instead of loading another compiler.
