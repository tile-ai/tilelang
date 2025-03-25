Auto-Tuning Techniques for Performance Optimization
===================================================
<div style="text-align: left;">
<em>Author:</em> <a href="https://github.com/yyttt6">yyttt6</a>
</div>

## Overview

Auto-tuning a Tile Language program involves three main steps:

1. Implement the target program using Tile Language with reserved optimization parameters
2. Generate top-k candidate configurations using carver
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
### Step 2: Generate Candidate Configurations with Carver
Use carver to generate optimized configurations based on templates:
```python
# Configure Matmul template
arch = CUDA("cuda")
carve_template = MatmulTemplate(
    M=M,
    N=N,
    K=K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float",
).with_arch(arch)

# Generate top-k optimization hints (topk=10 recommended)
roller_hints = carve_template.recommend_hints(topk=10)

# Configure candidate parameters
for hint in roller_hints:

    # ...existing code...

    config["block_M"] = block_m
    config["block_N"] = block_n
    config["block_K"] = hint.rstep[0]
    config["num_stages"] = hint.pipeline_stage
    config["thread_num"] = block_rows * block_cols * 32
    config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization

```
### Step 3: Compile and Benchmark
Configure JIT compilation and benchmarking settings:
```python
autotuner = AutoTuner.from_kernel(
    kernel=kernel, configs=get_configs(M, N, K, with_roller)).set_compile_args(
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
