# TileLang Bottleneck Optimization Quick Reference

After using the msprof CSV files to identify the bottleneck of the target TileLang kernel, use this reference to map hardware symptoms to verifiable TileLang changes. Change only one primary variable at a time, then collect a new profile using the same `cases.csv`, kernel-filtering rules, and timing methodology.

## Contents

1. [VEC Bound](#1-vec-bound-vector-compute-bottleneck)
2. [MTE2/MTE3 Bound](#2-mte2mte3-bound-data-transfer-bottleneck)
3. [CUBE Bound](#3-cube-bound-matrix-compute-bottleneck)
4. [SCALAR Bound and Launch Overhead](#4-scalar-bound-and-launch-overhead)
5. [Inter-Core Load Imbalance](#5-inter-core-load-imbalance)
6. [Bank Conflict](#6-bank-conflict)
7. [Insufficient Pipeline Overlap](#7-insufficient-pipeline-overlap)
8. [Low L2 Cache Hit Rate](#8-low-l2-cache-hit-rate)
9. [Cross-Metric Diagnostics](#9-cross-metric-diagnostics)
10. [Repository Implementation References](#10-repository-implementation-references)

## Usage Gates

- Analyze only the runtime kernel uniquely corresponding to the current `operator_file`; exclude input-construction and other auxiliary kernels.
- First determine whether the kernel is AIV-only, AIC-only, or mixed AIC/AIV, then select the corresponding fields.
- After changing the `T.Kernel` block count, tile shape, `num_stages`, buffer versions, or data layout, first run the complete accuracy regression suite relevant to the original pytest, then recollect all performance cases.
- Tile sizes, buffer counts, and pipeline stages must fit the target device's UB/L1/L0 capacity. Revert the parameter if compilation fails or resource limits are exceeded.

---

## 1. VEC Bound (Vector Compute Bottleneck)

**Criterion**: The target kernel has the highest `aiv_vec_ratio`, and the actual Vector duration is significantly higher than that of the other pipelines.

### TileLang Optimization Order

| # | Method | TileLang implementation | Applicable signal |
|---|---|---|---|
| 1 | Fuse intermediate results | Use `T.alloc_shared` to retain UB intermediate values, avoiding a write back to GM followed by another `T.copy` load | GM round trips occur between computation steps |
| 2 | Register reuse | Use `T.alloc_fragment` to hold reused values and complete multiple computation steps within the same `T.SimtVF`/`T.SimdVF` region | The same input is read from UB multiple times |
| 3 | Reduce casts | Combine dtype conversions, converting values in batches in a `T.Parallel` loop and reusing the results | The fp32/fp16/bf16 instruction ratio differs from expectations |
| 4 | Combine element-wise expressions | Complete adjacent element-wise computations in the same `T.Parallel` loop | The full tile is traversed multiple times |
| 5 | Optimize reductions | Use `T.alloc_reducer` and `T.finalize_reducer` instead of serial scalar accumulation | High scalar ratio in a reduction |
| 6 | Adjust parallel granularity | Adjust `T.SimtVF(threads=...)`, `T.SimdVF()`, and the workload assigned to `T.Parallel` | Insufficient vector width or thread utilization |

### Basic Structure

```python
in_ub = T.alloc_shared((tile_elems,), dtype)
out_ub = T.alloc_shared((tile_elems,), dtype)

for tile in T.Pipelined(num_tiles, num_stages=2):
    T.copy(x[tile * tile_elems], in_ub)
    with T.SimtVF(threads=threads):
        values = T.alloc_fragment((tile_elems,), compute_dtype)
        for i in T.Parallel(tile_elems):
            values[i] = T.cast(in_ub[i], compute_dtype)
            out_ub[i] = fused_compute(values[i])
    T.copy(out_ub, y[tile * tile_elems])
```

Do not allocate a fragment merely to follow the template. Use one only when it actually eliminates repeated UB reads or redundant computation.

---

## 2. MTE2/MTE3 Bound (Data Transfer Bottleneck)

**Criterion**: `ai*_mte2_ratio` or `ai*_mte3_ratio` is the highest ratio, and the GM transfer volume, instruction count, and bandwidth utilization in `Memory.csv` confirm the bottleneck.

### First Determine Whether Performance Is Near the Bandwidth Limit

```text
Theoretical transfer time (us) = bytes transferred / currently available device bandwidth (Byte/s) × 1e6
```

- Actual time close to theoretical time: First hide transfers with pipelining or reduce the total transfer volume.
- Actual time significantly above theoretical time: Check contiguity, alignment, per-`T.copy` granularity, tail tiles, and L2 behavior.

### TileLang Optimization Order

| # | Method | TileLang implementation |
|---|---|---|
| 1 | Reduce GM round trips | Place fusible computations in the same kernel and the same UB tile |
| 2 | Increase contiguous transfer granularity | Increase the tile size and use contiguous slices for `T.copy` |
| 3 | Handle unaligned tail tiles | Reserve aligned space for the target buffer and use `T.copy(..., pad_value=...)` or an explicit valid range |
| 4 | Reuse constants or weights | Retain data reused across iterations in `T.alloc_shared` or `T.alloc_l1` |
| 5 | Overlap transfer and computation | Use `T.Pipelined(..., num_stages=N)` and multi-version buffers |
| 6 | Adjust the L2 policy | After confirming the access and reuse pattern, set and compare `l2_cache_ctrl` for `T.copy` |

Do not pursue larger tiles in isolation. Always check UB/L1 usage, wasted work on tail tiles, and the number of tiles available for parallel execution.

---

## 3. CUBE Bound (Matrix Compute Bottleneck)

**Criterion**: `aic_cube_ratio` is the highest ratio. Use `aic_cube_fops`, L0/L1 bandwidth, and actual FLOPS to determine whether performance is near the compute limit.

### TileLang Optimization Order

| # | Method | TileLang implementation |
|---|---|---|
| 1 | Adjust the matrix tile | Jointly tune `TILE_M/TILE_N/TILE_K` to balance compute efficiency, L1/L0 capacity, and the number of parallel tiles |
| 2 | Reuse data in L1 | Use `T.alloc_l1` to retain K tiles or data reusable across output tiles |
| 3 | Accumulate in L0 | Use `T.alloc_l0c` and `T.gemm(..., clear_accum=(k == 0))` to accumulate across K tiles |
| 4 | Pipeline K | Use `T.Pipelined(K_TILES, num_stages=N)` to overlap GM→L1, L1→L0, and computation |
| 5 | Persistent scheduling | Use `T.Persistent` to traverse output tiles, reducing launch overhead and tail-tile imbalance |
| 6 | Output path | Select `T.copy` or `T.dual_copy` based on the output dtype and mixed-kernel structure to avoid extra intermediate transfers |

### Basic Structure

```python
a_l1 = T.alloc_l1((block_m, block_k), dtype)
b_l1 = T.alloc_l1((block_n, block_k), dtype)
acc_l0 = T.alloc_l0c((block_m, block_n), accum_dtype)

for k in T.Pipelined(num_k_tiles, num_stages=num_stages):
    T.copy(a_gm[..., k * block_k], a_l1)
    T.copy(b_gm[..., k * block_k], b_l1)
    T.gemm(a_l1, b_l1, acc_l0, transpose_B=True, clear_accum=(k == 0))
```

---

## 4. SCALAR Bound and Launch Overhead

**Criterion**: `ai*_scalar_ratio` is high, or the `Task Duration` for a small shape is dominated by launch, dynamic branches, and scalar loops.

### TileLang Optimization Order

| # | Method | TileLang implementation |
|---|---|---|
| 1 | Compile-time specialization | Move dtype, fixed shapes, and mode switches into JIT parameters or Python branches to reduce dynamic device-side decisions |
| 2 | Hoist loop-invariant expressions | Move indices, scales, or constant computations that do not vary by tile out of the loop |
| 3 | Reduce `T.serial` | Replace parallelizable element processing with `T.Parallel` and use `T.alloc_reducer` for reductions |
| 4 | Reduce dynamic scalar accesses | Avoid repeatedly reading dynamic Tensor elements or constructing `T.alloc_var` in inner loops |
| 5 | Adjust the block count | For small workloads, reduce the block count in `T.Kernel(num_blocks)` to avoid assigning too little work to each core |
| 6 | Combine small tiles | Increase the workload per block to reduce the proportion of loop-control and launch overhead |

High launch overhead for small shapes does not necessarily indicate an implementation error. Report both the absolute duration and the theoretical optimization headroom.

---

## 5. Inter-Core Load Imbalance

**Criterion**: The target kernel's per-core `ai*_time(us)` values in `PipeUtilization.csv` differ by more than 10%.

```python
times = [row["aiv_time(us)"] for row in target_kernel_rows]
imbalance = (max(times) - min(times)) / max(times) * 100
```

### TileLang Optimization Order

1. Check whether `T.Kernel(num_blocks)` is much larger than the number of valid tiles or mismatched with the data partitioning.
2. Use `T.ceildiv` to calculate the tile count and let each block process a similar number of complete tiles.
3. Distribute tail tiles across multiple blocks instead of assigning one large tail exclusively to the final block.
4. Use `T.Persistent` for irregular workloads so blocks continuously claim output tiles.
5. If individual tiles differ greatly in workload, redesign the tile dimensions or specialize the partitioning strategy by case.

---

## 6. Bank Conflict

**Criterion**: `aiv_vec_total_cflt_ratio`, `aiv_vec_bankgroup_cflt_ratio`, or `aiv_vec_bank_cflt_ratio` in `ResourceConflictRatio.csv` exceeds its threshold.

### TileLang Optimization Order

| Conflict type | TileLang change |
|---|---|
| High bankgroup | Adjust the two-dimensional shape and row stride of `T.alloc_shared`, along with the `T.Parallel` index mapping, to prevent parallel accesses from concentrating on the same bank group |
| High bank | Add padding to UB rows or adjacent operands and change starting offsets to prevent multiple operands from hitting the same bank in the same cycle |
| High resource conflict | Split overly long fused expressions and adjust Vector/Cube computation order and pipeline stages |
| High MTE conflict | Adjust `T.Pipelined` stages, buffer versions, and `T.copy` placement to stagger transfers and Vector accesses to the same UB buffer |

Adjust only one padding, stride, or index mapping at a time, then check the measured conflict ratio again. Do not infer the optimal value solely from the physical UB structure.

---

## 7. Insufficient Pipeline Overlap

**Criterion**: The pipeline view shows that MTE2, VEC, CUBE, or MTE3 execute mostly in series. Pay particular attention when the sum of the `vec + scalar + mte2 + mte3` ratios approaches 100%.

### Automatic Multi-Buffering

```python
buf = T.alloc_shared((tile_elems,), dtype)
T.annotate_buffer_versions({buf: num_stages})

for tile in T.Pipelined(
    num_tiles,
    num_stages=num_stages,
    annotations={"multi_buffer_eligible": [buf]},
):
    T.copy(x[tile * tile_elems], buf)
    compute(buf)
```

### Checklist

1. Is `num_stages` at least 2, and does it match the number of buffer versions?
2. Do different iterations access independent data? Real RAW/WAR dependencies prevent overlap.
3. Are `T.copy` and the computation inside the same pipeline-compatible loop?
4. Are there unnecessary synchronizations, serial inner loops, or cross-iteration write-after-read dependencies?
5. If automatic multi-buffering cannot express a manual ring buffer, use an explicit version dimension and `T.annotate_manual_multi_buffer`.
6. After adding stages, does increased UB/L1 usage cause compilation failure, reduce the block count, or regress performance?

---

## 8. Low L2 Cache Hit Rate

**Criterion**: The target kernel's `ai*_total_hit_rate(%)` in `L2Cache.csv` is low, and the Memory/Pipe data shows that GM access is the primary bottleneck.

### TileLang Optimization Order

1. First increase data locality through tiling and persistent scheduling, preventing different blocks from redundantly reading the same data in an unordered pattern.
2. For data with clear reuse, use a retention policy such as `T.copy(..., l2_cache_ctrl="NORMAL_FV")`.
3. For data streamed only once that would pollute the cache, use a `NOTALLOC_*` policy verified against the current TileLang version.
4. Profile the input, weights, and output separately with A/B comparisons; do not apply one L2 policy to all directions.
5. Check the hit rate together with total GM bytes. Revert the change if the hit rate rises without reducing total duration.

The repository's `example_gemm_bypass_l2.py` demonstrates how to use `T.copy(..., l2_cache_ctrl=...)` with different policies for input weights and output.

---

## 9. Cross-Metric Diagnostics

| Symptom combination | Root-cause hypothesis | TileLang items to inspect first |
|---|---|---|
| High vec_ratio + high bank conflict | The UB layout amplifies Vector duration | `T.alloc_shared` shape/padding and parallel index mapping |
| High mte2_time + low L2 hit rate | The data-reuse pattern or L2 policy is inappropriate | Tile scheduling, `T.Persistent`, and `T.copy(l2_cache_ctrl=...)` |
| High fixpipe_ratio | The output path or address alignment is inefficient | Output tile, valid range, and `T.copy`/`T.dual_copy` path |
| High mte2 + high mte3 | Bidirectional GM transfers are saturated | Fuse intermediate results, increase tile size, and reduce GM round trips |
| Low Block Dim + high Duration | Too few parallel tiles or an insufficient block count | `T.Kernel` block count, tile shape, and `T.Persistent` |
| Large per-core duration variance | Uneven partitioning of tiles or tail tiles | `T.ceildiv`, tail-tile distribution, and persistent scheduling |
| High scalar + small shape | High proportion of dynamic control and launch overhead | Compile-time specialization, fewer `T.serial` loops, and a lower block count |
| Serialized MTE/VEC | Multi-buffering did not take effect | `T.Pipelined`, buffer versions, and dependencies |

Every root-cause hypothesis must be validated through source inspection and profiling after the change.

---

## 10. Repository Implementation References

Prefer extracting patterns from files whose structure resembles the target operator:

| Optimization pattern | Reference file |
|---|---|
| Vector tiling, UB transfers, and pipelining | `examples/ascend/example_rmsnorm.py` |
| L1/L0C/UB data flow and pipelining for GEMM | `examples/ascend/example_gemm_mixedkernel.py` |
| Buffer versioning | `examples/ascend/example_buffer_version_annotation.py` |
| `T.Pipelined` + `T.annotate_buffer_versions` | `examples/ascend/example_simdvf_vecadd.py` |
| Fragment reuse and reduction | `examples/ascend/example_rmsnorm.py` |
| `T.alloc_l1`/`T.alloc_l0c`/`T.gemm` | `examples/ascend/example_gemm.py` |
| `T.copy(..., l2_cache_ctrl=...)` | `examples/ascend/example_gemm_bypass_l2.py` |
| Automatic and manual multi-buffering | `examples/ascend/example_manual_multibuffer.py` |

When extracting an optimization pattern, preserve the target operator's interface, data layout, boundary handling, and accuracy semantics. Never replace the target kernel wholesale.
