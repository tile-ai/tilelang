---
name: tilelang-performance-best-practices
description: Reference for implementing and optimizing TileLang Ascend/PTO operators. Use when adding, porting, reviewing, or optimizing an Ascend operator in the current repository. Verify Kernel, Persistent, SimdVF/SimtVF, UB/L1/L0, GEMM, reduction, transpose, tail-block, and pipeline APIs against the current repository's framework source and actually installed version, and decide whether to reuse templates based on maturity, PTO lowering, device correctness, and same-method performance data.
---

# TileLang Ascend/PTO Performance Best Practices

Use sources of truth in this priority order:

1. The target operator, `examples/ascend/`, `testing/ascend/`, and public-interface constraints in the current repository.
2. TileLang source located by `python -c 'import tilelang; print(tilelang.__file__)'`, along with its `examples/ascend/` and `testing/ascend/`.
3. Maturity-classified Python baselines, strategy metadata, and design documents under `references/`.

All source paths in this document are relative to the current repository root. Locate implementations through `src/ascend/`, `src/backend/`, `tilelang/`, and `examples/ascend/`, and verify that the actually imported version matches the source.

## Workflow

1. Search `examples/ascend/**/*.py` for similar operators, prioritizing schedules and APIs already validated on the current branch.
2. Locate the actually imported TileLang. Verify uncertain interfaces against its `examples/ascend/`, `testing/ascend/`, API definitions, and PTO lowering. Do not infer the installed version from a directory name.
3. When capacity, core count, architecture, or peak performance affects the implementation, read [Hardware Resource and Effective Compiler Capacity Discovery](references/common/hardware_resource_discovery.md). Obtain or reuse complete hardware evidence, record confirmed physical resources and the source of SKU specifications, and verify execution-domain reservations against actual TileLang lowering.
4. Read the [Optimization Index](references/index.md), [Template Maturity](references/template_status.md), and the corresponding operator-family documentation. Only `VERIFIED` or `PRODUCTION_REFERENCE` may be considered direct-reuse candidates.
5. First implement the smallest correctness-preserving path, then optimize from benchmark data. Do not obtain a "pass" by relaxing tolerances, reducing reference precision, or skipping shapes.
6. Run the targeted PTO correctness test first:

   ```bash
   TILELANG_DEFAULT_TARGET=pto pytest <test_file> -x
   ```

7. After all targeted correctness tests pass, run the relevant complete test suite; run expanded tests according to project requirements before integration. Performance comparisons must keep inputs, dtype, warmup, repeat, device, and concurrency identical.
8. Report tolerances, pass count, first failure, covered shapes, kernel latency, and baseline. Without measured data, call the result only a "design recommendation".
9. Distinguish numerical failures, compilation failures, and repository-raised `NotImplementedError`. Record unimplemented paths accurately and complete their implementation. Do not relax tolerances, skip tests, or describe a previously passing subset as a complete pass.

## Script Tools

- `scripts/validate_templates.py`: By default, validates structure, syntax, stale paths, and key tiling regressions. When `TILELANG_DEFAULT_TARGET=pto` is set and `--npu --num-cores <queried_AIV_core_count>` is passed, it compiles and runs representative Broadcast, EuclideanNorm, Softmax, Scan, and RoPE cases.

## Code Admission Rules

- Kernel implementations use only the Python TileLang DSL; tiling, dispatch, and mathematical references use ordinary Python. Derive from a similar implementation in the current repository first, then select a bundled reference according to [Template Maturity](references/template_status.md).
- Do not hard-code `target="ascend"` in `@tilelang.jit`. Let `TILELANG_DEFAULT_TARGET=pto` select PTO unless a test explicitly compares targets.
- Ascend `T.Kernel` uses only a one-dimensional block grid and does not receive `threads=`; express the thread domain in `T.SimtVF(threads=N)`.
- Prefer execution domains used by similar validated implementations: regular elementwise work normally uses `T.SimdVF()`, while Norm/Reduction may use the repository-validated `T.SimtVF + T.reduce_*`; use `T.SimtVF()` for scattered indexing, complex branching, or atomic operations. Any execution-domain change requires measured correctness and performance validation.
- For current PTO Ascend GEMM, prefer the validated L1 B layout `[N, K]` with `transpose_B=True`. Other layouts are not prohibited by the API, but require validation against target-version lowering and the device. Zero fp32 L0C for the first valid K subtile and accumulate subsequent subtiles.
- Use fp32 accumulation as the starting point for fp16/bf16 reductions, normalization, softmax statistics, and matrix multiplication. Internal mixed precision, HF32, and approximate instructions may be experimental, but must preserve original tests, tolerances, and public contracts; keep them only when correctness and same-method performance justify doing so.
- For Vector, Cube, and mixed kernels, respectively use hardware-query-confirmed available AIV, AIC, and paired resources. Pass the core count explicitly when constructing a template; for small workloads, do not use more cores than independent tasks.
- Elementwise, gather/scatter, or layout-conversion optimization must build a task tree and sourced tile boundaries according to [Tiling, Task, and Vector Dataflow Search](references/elementwise/tiling_task_vector_search.md). Classify physical routes by the complete data path: an intermediate planar/scratch payload is a materialized transform and cannot substitute for direct contiguous loads followed by register reordering. Treat tile/task, dataflow, precision, and pipeline as orthogonal axes, and adjudicate interacting combinations through the same actually activated candidate.
- A UB budget must distinguish confirmed physical capacity, execution-domain reservations in the final kernel body made by current lowering, and the explicit buffer footprint; recompute it especially after adding or removing `T.SimtVF`. The budget also includes buffer versions, alignment padding, resident data, and a justified safety margin. Copy only `valid` elements for a GM tail block, while still allocating the UB for a complete SIMD/DMA footprint.
- Pipeline work must establish cross-iteration dependencies according to [Multiversion Pipeline Design](references/elementwise/double_buffer_design.md). Version every still-live buffer; use explicit stage storage when automatic analysis does not apply. Determine activation from generated code and same-method latency/overlap.
- Verify multi-axis tile domains according to [Persistent Multi-Axis Task Mapping](references/common/persistent_task_mapping.md) to avoid unnecessary flattened-index decoding in hot loops.
- When reduction compute is light and the contiguous non-reduction axis is wide, use [Wide-Output-Axis Tiling for Reduction](references/reduce/wide_output_tiling.md) to evaluate merging contiguous output tiles.
- Determine the shape matrix from the public-interface contract. If the interface does not support arbitrary tail blocks, cover only permitted remainder classes rather than treating a new fallback as an existing requirement. Also check forward/backward, dynamic shapes, very small/large shapes, NaN/Inf semantics, and empty-task boundaries.

## Operator-Family Routing

| Operator Family | Preferred Pattern | Source Reference (Validate Before Reuse) |
|---|---|---|
| Elementwise / Quant | Persistent + UB staging + SimdVF; for short-record index transformations, read the [structural reference](references/elementwise/indexed_short_record.md) | `examples/ascend/example_simdvf_per_token_cast_to_fp8.py` |
| Rowwise Reduction + Elementwise Epilogue | Exact reduction-axis UB + resident broadcast parameters + SIMD fp32 reduction; evaluate both single-row pipelining and cross-row batching, and read [Rowwise Reduction with Elementwise Writeback](references/reduce/rowwise_reduce_epilogue.md) and [Batched Short Reduction](references/reduce/batched_short_reduction.md) | Validate against the target operator |
| Reduction / Norm / Softmax | fp32 accumulation and hierarchical reduction; select [Batched Short Reduction](references/reduce/batched_short_reduction.md) or [Fixed Small State](references/reduce/fixed_small_state.md) according to structure | TileLang `examples/ascend/example_rmsnorm.py` |
| Transpose / Gather | Choose contiguous load + register reordering, SIMD gather, or temporary materialization based on source-window span and density; fall back to SIMT only for unsupported paths | `testing/ascend/layout/test_ascend_l0_transpose.py` (L0 layout semantics), `tilelang/ascend/language/copy_op.py` (transfer API); validate general transpose implementations against the target operator |
| MatMul | L1/L0C, validated layouts, K-dimension pipelining; Stream-K/full-load/SWAT are candidates | TileLang `examples/ascend/example_gemm*.py` |
| FlashAttention | Cube/Vector dataflow, online softmax, fp32 state | TileLang `examples/ascend/flash_attention/example_mha.py` |
| TopK / irregular | SimdVF instructions or SimtVF | `examples/ascend/example_simdvf_topk_gate.py`, `examples/ascend/example_simdvf_*topk*.py` |

## Reference Routing

Select the operator family from [references/index.md](references/index.md) and determine availability using [template_status.md](references/template_status.md). `EXECUTABLE_BASELINE` is only a correctness starting point. Do not describe `PARTIAL` or `DESIGN_ONLY` as directly copyable optimized implementations. A status upgrade must include lowering, correctness, and performance evidence.
