# Complete Reference for TileLang msprof Op CSV Fields

Data source: MindStudio 8.3.0 CSV field definitions. This reference is used to analyze TileLang kernels running on Atlas A2/A3.

Before analysis, always filter the current `profiling_file` by `expected_kernel` to select the target kernel and exclude input construction, data conversion, and other auxiliary kernels. If a field is absent, empty, or `N/A`, skip it; never fill it using a value from another kernel.

> The `ai*` prefix in field names represents `aic` (Cube Core) or `aiv` (Vector Core). First use the target kernel's actual execution records to determine whether it is AIV-only, AIC-only, or a mixed AIC/AIV kernel.

## Contents

1. [OpBasicInfo.csv](#1-opbasicinfocsv)
2. [PipeUtilization.csv](#2-pipeutilizationcsv)
3. [ArithmeticUtilization.csv](#3-arithmeticutilizationcsv)
4. [Memory.csv](#4-memorycsv)
5. [MemoryL0.csv](#5-memoryl0csv)
6. [MemoryUB.csv](#6-memoryubcsv)
7. [L2Cache.csv](#7-l2cachecsv)
8. [ResourceConflictRatio.csv](#8-resourceconflictratiocsv)
9. [TileLang Cross-File Diagnostic Sequence](#9-tilelang-cross-file-diagnostic-sequence)

---

## 1. OpBasicInfo.csv

Start with this file to confirm the target kernel, overall duration, block count, and operating frequency.

| Field | Meaning | TileLang focus |
|---|---|---|
| Op Name | Runtime kernel name | Must uniquely correspond to the target kernel in the current TileLang source code |
| Op Type | Kernel type | Used to distinguish AIV, AIC, and mixed kernels |
| Task Duration(us) | Total task duration, including scheduling, execution, and response | Used to assess launch/response overhead; for each variant, consistently calculate `kernel_time` from the target aiv/aic critical path |
| Block Dim | Number of logical blocks in the task | Compare with `T.Kernel(num_blocks)`, the number of valid tiles, and the actual device core count |
| Mix Block Dim | Secondary-core blockDim for a mixed kernel; N/A for a non-mixed kernel | Check the core allocation ratio of mixed AIC/AIV kernels |
| Device ID | NPU device ID | Ensure that the baseline and all variants are collected on the same device |
| PID | Process ID | Verify that each process runs only one case |
| Current Freq | Current operating frequency | Compare with Rated Freq to identify dynamic downclocking |
| Rated Freq | Rated frequency | Results from different runs may not be comparable when Current < Rated |

### Corresponding TileLang Constructs

- `Block Dim`: Check whether the block count in `with T.Kernel(num_blocks)` matches the number of output tiles.
- Mixed kernel: Preserve both the raw aic and aiv times, and use the critical path rather than their simple sum as `kernel_time`.
- Small shape: Compare `Task Duration` with the on-core `aiv_time/aic_time`; when the difference is large, investigate launch and control overhead first.

---

## 2. PipeUtilization.csv

The most important file for bottleneck localization. Each row corresponds to a core or sub-core, identified by `block_id` and `sub_block_id`.

### Common Fields

| Field | Meaning |
|---|---|
| block_id | Logical block identifier of the target task |
| sub_block_id | Name and index of the Vector/Cube sub-core within the block |
| aic_time(us) | Cube Core execution time |
| aic_total_cycles | Total Cube Core cycles |
| aiv_time(us) | Vector Core execution time |
| aiv_total_cycles | Total Vector Core cycles |

### Pipeline Unit Duration and Ratio

| Field | Meaning | Bottleneck threshold |
|---|---|---|
| aiv_vec_time(us) | Vector instruction duration | — |
| aiv_vec_ratio | Percentage of cycles spent on Vector instructions | When >50%, investigate VEC Bound first |
| aic_cube_time(us) | Cube instruction duration | — |
| aic_cube_ratio | Percentage of cycles spent on Cube instructions | Usually dominant in MatMul; should be close to 0 for a pure Vector kernel |
| ai*_scalar_time(us) | Scalar instruction duration | — |
| ai*_scalar_ratio | Percentage of cycles spent on Scalar instructions | When >30%, investigate SCALAR Bound first |
| aic_fixpipe_time(us) | FixPipe (L0C→GM/L1) duration | — |
| aic_fixpipe_ratio | Percentage of cycles spent on FixPipe | When >15%, check the output tile, path, and address alignment |
| aic_mte1_time(us) | MTE1 (L1→L0A/L0B) duration, excluding wait time | — |
| aic_mte1_ratio | Percentage of cycles spent on MTE1 | — |
| ai*_mte2_time(us) | MTE2 load duration | — |
| ai*_mte2_ratio | Percentage of cycles spent on MTE2 | When >50%, investigate the load bottleneck first |
| ai*_mte3_time(us) | MTE3 store duration | — |
| ai*_mte3_ratio | Percentage of cycles spent on MTE3 | Evaluate together with MTE2 to assess bidirectional GM transfer pressure |
| ai*_icache_miss_rate | ICache miss rate | When >15%, check branches and generated code size |

### Active Bandwidth (A2/A3 Only)

| Field | Meaning |
|---|---|
| aiv_mte2_active_bw(GB/s) | Vector-core MTE2 active bandwidth |
| aiv_mte3_active_bw(GB/s) | Vector-core MTE3 active bandwidth |
| aic_mte1_active_bw(GB/s) | Cube-core MTE1 active bandwidth; requires MemoryDetail |
| aic_mte2_active_bw(GB/s) | Cube-core MTE2 active bandwidth; requires MemoryDetail |
| aic_mte3_active_bw(GB/s) | Cube-core MTE3 active bandwidth |
| aic_fixpipe_active_bw(GB/s) | Cube-core FixPipe active bandwidth |

### Corresponding TileLang Constructs

| Profiling symptom | TileLang constructs to inspect first |
|---|---|
| High VEC | `T.SimtVF`/`T.SimdVF`, `T.Parallel`, `T.alloc_fragment`, casts, and element-wise fusion |
| High CUBE | `T.gemm`, `T.alloc_l1`, `T.alloc_l0c`, and matrix tile shape |
| High SCALAR | Device-side dynamic branches, `T.serial`, inner-loop index calculations, and dynamic Tensor scalar reads |
| High MTE2/MTE3 | Number of `T.copy` operations, contiguous slices, tile size, and GM round trips |
| Serialized pipeline | `T.Pipelined(num_stages=...)`, `T.annotate_buffer_versions`, and cross-iteration dependencies |
| Large per-core time variance | `T.Kernel` block count, `T.ceildiv` tail-tile partitioning, and `T.Persistent` scheduling |

---

## 3. ArithmeticUtilization.csv

Inspect Cube and Vector instruction types, instruction counts, and computation volume.

### Cube Instruction Fields

| Field | Meaning | TileLang relationship |
|---|---|---|
| aic_cube_ratio | Percentage of cycles spent on Cube instructions | Whether `T.gemm` is on the main path |
| aic_cube_fp16_ratio | Percentage of Cube fp16 instructions | Input dtype and the `T.gemm` computation path |
| aic_cube_int8_ratio | Percentage of Cube int8 instructions | Quantized matrix computation path |
| aic_cube_fops | Total number of Cube floating-point operations | Compare with theoretical FLOPS to calculate compute utilization |
| aic_cube_total_instr_number | Total number of Cube instructions | The instruction count may be excessive when tiles are too small |
| aic_cube_fp_instr_number | Number of Cube floating-point instructions | — |
| aic_cube_int_instr_number | Number of Cube integer instructions | — |

### Vector Instruction Fields

| Field | Meaning | TileLang relationship |
|---|---|---|
| aiv_vec_ratio | Percentage of cycles spent on Vector instructions | Whether the main computation in `T.Parallel` becomes a bottleneck |
| aiv_vec_fp32_ratio | Percentage of Vector fp32 instructions | Check for unnecessary fp32 casts/computation |
| aiv_vec_fp16_ratio | Percentage of Vector fp16 instructions | Compare with the expected computation dtype |
| aiv_vec_int32_ratio | Percentage of Vector int32 instructions | Check the proportion of indexing and integer operations |
| aiv_vec_int16_ratio | Percentage of Vector int16 instructions | — |
| aiv_vec_misc_ratio | Percentage of miscellaneous Vector instructions | When high, check special functions, type conversions, and complex control flow |
| aiv_vec_fops | Total number of Vector floating-point operations | Compare with the algorithm's expected FLOPS to identify redundant computation |

### TileLang Diagnostics

- fp32 ratio far above expectations: Locate `T.cast`, the fragment dtype, and the reduction accumulation dtype.
- High instruction count but low effective FLOPS: Combine multiple `T.Parallel` traversals and reuse intermediate values in `T.alloc_fragment`.
- High Scalar usage in reductions: Use `T.alloc_reducer` and `T.finalize_reducer` instead of accumulation with `T.serial`.
- Excessive CUBE instruction count: Increase the K tile or output tile, while also checking L1/L0 capacity and parallelism.

---

## 4. Memory.csv

Inspect memory bandwidth, transfer instruction counts, and data volume.

### Bandwidth Rates

| Field | Meaning |
|---|---|
| aiv_gm_to_ub_bw(GB/s) | GM→UB bandwidth |
| aiv_ub_to_gm_bw(GB/s) | UB→GM bandwidth |
| aic_l1_read_bw(GB/s) | L1 read bandwidth |
| aic_l1_write_bw(GB/s) | L1 write bandwidth |
| ai*_main_mem_read_bw(GB/s) | Main-memory read bandwidth |
| ai*_main_mem_write_bw(GB/s) | Main-memory write bandwidth |

### Instruction Statistics

| Field | Meaning |
|---|---|
| aic_mte1_instructions | Number of MTE1 instructions |
| aic_mte1_ratio | Percentage of cycles spent on MTE1 |
| ai*_mte2_instructions | Number of MTE2 instructions |
| ai*_mte2_ratio | Percentage of cycles spent on MTE2 |
| ai*_mte3_instructions | Number of MTE3 instructions |
| ai*_mte3_ratio | Percentage of cycles spent on MTE3 |

### Data Transfer Volume

| Field | Meaning |
|---|---|
| read_main_memory_datas(KB) | Total volume read from main memory |
| write_main_memory_datas(KB) | Total volume written to main memory |
| GM_to_L1_datas(KB) | GM→L1 transfer volume |
| L1_to_GM_datas(KB)(estimate) | L1→GM transfer volume (estimated) |
| L0C_to_L1_datas(KB) | L0C→L1 transfer volume |
| L0C_to_GM_datas(KB) | L0C→GM transfer volume |
| GM_to_UB_datas(KB) | GM→UB transfer volume |
| UB_to_GM_datas(KB) | UB→GM transfer volume |

### Bandwidth Utilization

| Field | Meaning | Reference criterion |
|---|---|---|
| GM_to_L1_bw_usage_rate(%) | GM→L1 bandwidth utilization | >60% is usually good |
| L1_to_GM_bw_usage_rate(%)(estimate) | L1→GM bandwidth utilization | >60% is usually good |
| L0C_to_L1_bw_usage_rate(%) | L0C→L1 bandwidth utilization | Analyze together with the FixPipe path |
| L0C_to_GM_bw_usage_rate(%) | L0C→GM bandwidth utilization | Analyze together with the output path |
| GM_to_UB_bw_usage_rate(%) | GM→UB bandwidth utilization | >60% is usually good |
| UB_to_GM_bw_usage_rate(%) | UB→GM bandwidth utilization | >60% is usually good |

### Corresponding TileLang Constructs

- GM↔UB: `T.copy` and `T.alloc_shared`.
- GM↔L1: `T.copy` and `T.alloc_l1`.
- L1/L0: `T.alloc_l1`, `T.alloc_l0a/l0b/l0c`, and `T.gemm`.
- GM byte count above the algorithmic minimum: Check whether intermediate results can remain in shared/fragment storage and whether computation can be fused.
- Many transfer instructions but low total data volume: Increase the contiguous tile size and reduce fragmented `T.copy` operations.

Use the following formula consistently for theoretical transfer time:

```text
Theoretical time (us) = bytes transferred / currently available device bandwidth (Byte/s) × 1e6
```

---

## 5. MemoryL0.csv

L0A/L0B/L0C bandwidth, primarily for analyzing TileLang kernels that use `T.gemm`.

| Field | Meaning | TileLang focus |
|---|---|---|
| aic_l0a_read_bw(GB/s) | L0A read bandwidth | L0 access for the A tile |
| aic_l0a_write_bw(GB/s) | L0A write bandwidth | `T.alloc_l0a`/automatic L0A path |
| aic_l0b_read_bw(GB/s) | L0B read bandwidth | L0 access for the B tile |
| aic_l0b_write_bw(GB/s) | L0B write bandwidth | `T.alloc_l0b`/automatic L0B path |
| aic_l0c_read_bw_cube(GB/s) | Cube bandwidth for reading from L0C | Accumulation across K tiles and the output path |
| aic_l0c_write_bw_cube(GB/s) | Cube bandwidth for writing to L0C | `T.alloc_l0c` and `T.gemm` accumulation |

When L0 bandwidth is abnormal, jointly inspect `TILE_M/N/K`, `clear_accum`, `T.Pipelined` on the K loop, and the number of L1→L0 copies.

---

## 6. MemoryUB.csv

Vector and Scalar read/write bandwidth to UB.

| Field | Meaning | TileLang focus |
|---|---|---|
| aiv_ub_read_bw_vector(GB/s) | Vector bandwidth for reading from UB | Whether `T.alloc_shared` is repeatedly read in full |
| aiv_ub_write_bw_vector(GB/s) | Vector bandwidth for writing to UB | Whether intermediate results cause multiple UB writes |
| aiv_ub_read_bw_scalar(GB/s) | Scalar bandwidth for reading from UB | Whether there are too many dynamic scalar accesses in inner loops |
| aiv_ub_write_bw_scalar(GB/s) | Scalar bandwidth for writing to UB | Whether there are too many scalar loops and fine-grained writes |

When Vector UB bandwidth and VEC time are both high, try reuse with `T.alloc_fragment` first. When Scalar UB bandwidth is high, reduce inner-loop Tensor scalar reads and `T.serial` loops.

---

## 7. L2Cache.csv

Inspect the L2 hit count and hit rate of the target kernel.

| Field | Meaning | Reference criterion |
|---|---|---|
| ai*_write_cache_hit | Number of write-cache hits | — |
| ai*_write_cache_miss_allocate | Number of allocations after write-cache misses | — |
| ai*_r*_read_cache_hit | Number of cache hits on each read channel | — |
| ai*_r*_read_cache_miss_allocate | Number of allocations after misses on each read channel | — |
| ai*_write_hit_rate(%) | Write-cache hit rate | >80% is usually good |
| ai*_read_hit_rate(%) | Read-cache hit rate | >80% is usually good |
| ai*_total_hit_rate(%) | Overall hit rate | >80% is usually good; investigate carefully when <50% |

### Corresponding TileLang Constructs

1. First check the output-tile order and data locality of `T.Persistent`.
2. Then, based on the data reuse pattern, run per-input A/B tests for `T.copy(..., l2_cache_ctrl=...)`.
3. Reused data such as weights and streaming input/output may require different strategies.
4. Always evaluate the hit rate together with total GM bytes and `kernel_time`; never use it as an isolated optimization objective.

---

## 8. ResourceConflictRatio.csv

Inspect UB bank-group conflicts, bank conflicts, resource conflicts, and wait ratios. This file is produced by on-device collection and may be absent from simulation results.

### Core Conflict Metrics

| Field | Meaning | Reference criterion |
|---|---|---|
| aiv_vec_total_cflt_ratio | Percentage of total Vector instruction stalls | <5% is usually good; >15% is severe |
| aiv_vec_bankgroup_cflt_ratio | Percentage of stalls caused by bank-group conflicts | <3% |
| aiv_vec_bank_cflt_ratio | Percentage of stalls caused by bank conflicts | <3% |
| aiv_vec_resc_cflt_ratio | Percentage of compute-unit resource conflicts | <5% |
| aiv_vec_mte_cflt_ratio | Percentage of Vector/MTE conflicts | <3% |

### Wait Metrics

| Field | Meaning |
|---|---|
| aic_cube_wait_ratio | Cube-unit wait ratio |
| aiv_vec_wait_ratio | Vector-unit wait ratio |
| ai*_mte1_wait_ratio | MTE1 wait ratio |
| ai*_mte2_wait_ratio | MTE2 wait ratio |
| ai*_mte3_wait_ratio | MTE3 wait ratio |

### TileLang Optimization Mapping

| Conflict type | Preferred change |
|---|---|
| High bankgroup_cflt | Adjust the `T.alloc_shared` shape, row stride, padding, and `T.Parallel` index mapping |
| High bank_cflt | Change the starting offsets or row widths of different UB operands to avoid concentrated access to the same bank in the same cycle |
| High resc_cflt | Adjust fused expressions, Vector/Cube ordering, and pipeline stages to reduce contention for the same execution resources |
| High mte_cflt | Adjust `T.copy` placement, `T.Pipelined` stages, and buffer versions to prevent transfers and computation from accessing the same version |
| High wait ratio | Check real data dependencies, tile granularity, multi-buffer versions, and unnecessary serial loops |

After changing the layout, rerun accuracy tests to ensure that padding, slices, or boundary indices have not changed the semantics.

---

## 9. TileLang Cross-File Diagnostic Sequence

1. Use `OpBasicInfo.csv` to verify the kernel name, frequency, Task Duration, and Block Dim.
2. Use `PipeUtilization.csv` to identify the dominant pipeline, establish a consistent `kernel_time`, and assess per-core variance.
3. Use `ArithmeticUtilization.csv` to evaluate effective computation, casts, and the Vector/Cube instruction mix.
4. Use `Memory.csv` to calculate total transfer volume, bandwidth utilization, and theoretical transfer time.
5. Read `MemoryL0.csv` for CUBE kernels and `MemoryUB.csv` for Vector kernels.
6. Read `L2Cache.csv` and `ResourceConflictRatio.csv` to validate cache and conflict hypotheses.
7. Map the conclusions to concrete changes in `T.Kernel`, tile shape, `T.copy`, buffer scope, `T.Pipelined`, `T.Persistent`, `T.Parallel`, or `T.gemm`.
8. After completing the full pytest-related accuracy regression suite, collect a new profile using the same case and timing methodology.
