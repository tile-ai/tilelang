# TileLang/PTO Implementation Guide

## Contents

- [Sources of Truth](#sources-of-truth)
- [Execution Domains](#execution-domains)
- [Memory and Pipelining](#memory-and-pipelining)
- [Correctness Rules](#correctness-rules)
- [Performance Rules](#performance-rules)
- [Operator Patterns](#operator-patterns)
- [Validation Matrix](#validation-matrix)

## Sources of Truth

Confirm APIs in this priority order:

1. The current target operator and `examples/ascend/**/*.py`: implementation structure and API references; determine performance status from each specific example and current validation results.
2. Matching `references/*/code/*_asc.py` files in this Skill: complete Ascend host dispatch and kernel structures for generation and optimization ideas; no template maturity is assigned to this source-reading corpus.
3. Locate the actually imported source with `python -c 'import tilelang; print(tilelang.__file__)'`; do not infer the version from a sibling directory name.
4. `examples/ascend/` and `testing/ascend/` in the actual TileLang source: runnable examples, API boundaries, and lowering regressions.
5. `tilelang/language/` and PTO lowering in the actual TileLang source: API definitions and backend constraints.
6. PTO codegen/lowering: confirm that support genuinely exists under `TILELANG_DEFAULT_TARGET=pto`.

## Execution Domains

```python
import tilelang
from tilelang import language as T
from tilelang.language import simd as S


@tilelang.jit(out_idx=-1)
def make_kernel(n: int, num_cores: int):
    # num_cores is supplied from confirmed target hardware resources.
    @T.prim_func
    def kernel(x: T.Tensor[(n,), T.float32], y: T.Tensor[(n,), T.float32]):
        with T.Kernel(num_cores) as core_id:
            ...

    return kernel
```

- `T.SimdVF()`: A 2048-bit vector domain with 64 lanes for fp32 and 128 lanes for fp16/bf16. Commonly used for regular contiguous elementwise computation.
- `T.SimtVF(threads=N)`: A thread domain. In addition to scattered indexing, complex branches, and atomic operations, it can host repository-validated `T.reduce_*` Norm/Reduction paths.
- Prefer similar repository implementations over abstract preferences. For example, first reuse the SimtVF reduction structure in `examples/ascend/example_rmsnorm.py` for RMSNorm and validate current cases; then treat SimdVF as an optimization candidate requiring independent validation.
- L1/L0A/L0B/L0C and `T.gemm`: Cube matrix computation.
- `T.MixedKernel`: Use only when `sid` must manually partition two AIV subcores.

Do not port the GPU form `T.Kernel(..., threads=N)` to Ascend.

## Memory and Pipelining

- Use `T.alloc_shared` for UB. For GM/UB/L1/L0 transfers, use the `T.copy` form validated in the corresponding repository path.
- Transfer only the `valid` range for a dynamic tail block, but allocate the UB row width for the complete aligned footprint that DMA/SIMD may access.
- Move reused data such as weights, scales, and small tables into UB outside the Persistent loop whenever possible, using a single version.
- Multiversion tile inputs and outputs by stage:

```python
x_ub = T.alloc_shared((block,), T.float32)
y_ub = T.alloc_shared((block,), T.float32)
T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})

for task in T.Persistent(
    [T.ceildiv(n, block)],
    num_cores,
    core_id,
    group_size=1,
    num_stages=num_stages,
):
    offset = task * block
    valid = T.min(block, n - offset)
    T.copy(x[offset : offset + valid], x_ub[:valid])  # Schematic: verify this dynamic slice on PTO.
    ...
    T.copy(y_ub[:valid], y[offset : offset + valid])
```

UB budget formula: `Σ(buffer_elems × dtype_bytes × versions) + padding + resident + safety_margin`.

## Correctness Rules

- Convert fp16/bf16 inputs to fp32 before reduction, variance, rsqrt, softmax max/sum, and long multiply-add chains.
- GEMM uses an fp32 L0C accumulator by default and converts to the output dtype only after completion.
- Use the stable softmax form `exp(x - max(x)) / sum(exp(x - max(x)))`.
- Add `eps` to the fp32 statistics for RMSNorm/LayerNorm.
- A tail block must not read uninitialized UB lanes. Full-register access requires zeroing, filling, or a correct mask.
- Use tolerances from similar repository tests together with the numerical scale. Do not expand tolerances to conceal dtype, tail-block, or accumulation errors.

## Performance Rules

1. Reduce GM traffic first: keep reused data resident, fuse intermediate results, and avoid repeated writeback.
2. Then establish MTE/Vector/Cube overlap: test Persistent with 2/3 stages and select using profiling.
3. Prefer a static vector count in SIMD inner loops; avoid unnecessary dynamic loops, scalar branches, and excessive unrolling.
4. Core count must not exceed `min(hardware core count, independent task count)`. When each core performs fixed work such as constant transfers or index construction, also A/B test a lower core count with multiple Persistent tasks per core. Select based on duplicated fixed-work cost, waves, load balance, and measurements.
5. UB, registers, and instruction volume jointly constrain tile size. When performance declines, inspect spills, code bloat, and occupancy.
6. Keep input distribution, shape, dtype, output semantics, warmup/repeat, and concurrency settings identical in performance comparisons.

## Operator Patterns

### Elementwise

GM→UB, compute with `T.SimdVF`, then UB→GM. A scalar weight may use `S.vld(addr, dist='BRC_B32')`. For bf16/fp16 transcendental functions, unpack/convert to fp32 before computation, then pack afterward.

### Reduction / Norm

- Whenever possible, assign each output to one core to avoid atomic accumulation.
- Use single-version resident UB for cross-tile state and an fp32 vector accumulator per tile.
- For a large reduction, evaluate split-K partials + final reduction, including the additional GM traffic.
- If the public interface includes backward, validate its fp32 accumulation and tails separately. For a forward-only primitive, explicitly record that backward is out of scope.

### Transpose

- Use SIMD gather/scatter when supported by the shape/dtype. Pad both UB dimensions to the vector-access footprint.
- For dtypes that cannot be expressed by an index vector, use a repository-validated SIMT path.
- `T.assume` may express only constraints genuinely guaranteed by the caller. Tests must cover every permitted remainder class.

### GEMM

```python
with T.Kernel(num_cube_cores) as block_id:
    a_l1 = T.alloc_l1((tile_m, tile_k), dtype)
    b_l1 = T.alloc_l1((tile_n, tile_k), dtype)  # Canonical [N, K] in current Ascend examples.
    c_l0 = T.alloc_l0c((tile_m, tile_n), T.float32)
    for kt in T.Pipelined(T.ceildiv(k, tile_k), num_stages=num_stages):
        T.copy(a_gm_slice, a_l1)
        T.copy(b_gm_slice, b_l1)
        T.gemm(a_l1, b_l1, c_l0, transpose_B=True, clear_accum=(kt == 0))
    T.copy(c_l0, c_gm_slice)
```

This is a structural illustration from current Ascend examples, not the only legal `T.gemm` layout for every target. Adapt slicing, padding, and output transfers from similar examples that have run successfully. With nested K subloops, clear only on the first valid sub-tile, for example, `kt == 0 and sk == 0`. Do not assume arbitrary M/N/K tails are handled correctly automatically.

## Validation Matrix

| Dimension | Required Checks |
|---|---|
| Shape | Minimum, common, maximum, remainder classes allowed by the public interface, and dynamic dimensions; add tile±1 only when the interface promises general tail-block support |
| Dtype | Every supported input/output dtype; compare low-precision inputs with an fp32 reference |
| Numerical Values | Zero, positive and negative values, extremes, small values, and duplicates; cover NaN/Inf when required |
| Paths | Interface-required forward/backward, tail blocks, resident/fallback, and single-core/multicore paths |
| Performance | After targeted correctness tests pass, compare latency on representative shapes and record the regression threshold |

```bash
TILELANG_DEFAULT_TARGET=pto pytest <test_file> -x
TILELANG_DEFAULT_TARGET=pto pytest <test_file>
```

Add `-n` only after confirming worker device binding and isolation, with concurrency no greater than the available-device count; otherwise, run serially. On OOM, reduce concurrency without dropping cases. Benchmarks require exclusive device access.

For failures, separately record collection count, pass count, first failing case, exception type, and root cause. `NotImplementedError` means the test reached an unimplemented repository path, not a tolerance issue. After implementing that path, restart validation from the targeted correctness test.
