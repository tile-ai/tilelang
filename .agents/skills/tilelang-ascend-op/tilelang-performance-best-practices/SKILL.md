---
name: tilelang-performance-best-practices
description: Reference for implementing and optimizing TileLang Ascend/PTO operators. Use when adding, porting, reviewing, or optimizing an Ascend operator in the current repository. Route to bundled Ascend host-and-kernel implementation references for generation and optimization ideas, verify APIs against the current framework source, and keep reference code distinct from maturity-rated executable templates.
---

# TileLang Ascend/PTO Performance Best Practices

Use sources of truth in this priority order:

1. The target operator, `examples/ascend/`, `testing/ascend/`, and public-interface constraints in the current repository.
2. TileLang source located by `python -c 'import tilelang; print(tilelang.__file__)'`, along with its `examples/ascend/` and `testing/ascend/`.
3. Ascend host-and-kernel implementation references under `references/*/code/*_asc.py`.
4. Maturity-classified executable templates, strategy metadata, and design documents under `references/`.

All source paths in this document are relative to the current repository root. Locate implementations through `src/ascend/`, `src/backend/`, `tilelang/`, and `examples/ascend/`, and verify that the actually imported version matches the source.

## Workflow

1. Classify the target by operator family, then search `examples/ascend/**/*.py` and the matching entries in the [Optimization Index](references/index.md).
2. Read only the relevant `references/*/code/*_asc.py` files for complete host dispatch, tiling, buffer, execution-domain, pipeline, and kernel structures. Use them both when generating an implementation and when forming performance hypotheses.
3. Locate the actually imported TileLang. Verify uncertain interfaces against its `examples/ascend/`, `testing/ascend/`, API definitions, and PTO lowering. Do not infer the installed version from a directory name.
4. When capacity, core count, architecture, or peak performance affects the implementation, read [Hardware Resource and Effective Compiler Capacity Discovery](references/common/hardware_resource_discovery.md). Obtain or reuse complete hardware evidence, record confirmed physical resources and the source of SKU specifications, and verify execution-domain reservations against actual TileLang lowering.
5. Read [Template Maturity](references/template_status.md) only when selecting an executable template. Only `VERIFIED` or `PRODUCTION_REFERENCE` templates may be considered direct-reuse candidates.
6. First implement the smallest correctness-preserving path, then optimize from benchmark data. Do not obtain a "pass" by relaxing tolerances, reducing reference precision, or skipping shapes.
7. Run the targeted PTO correctness test first:

   ```bash
   TILELANG_DEFAULT_TARGET=pto pytest <test_file> -x
   ```

8. After all targeted correctness tests pass, run the relevant complete test suite; run expanded tests according to project requirements before integration. Performance comparisons must keep inputs, dtype, warmup, repeat, device, and concurrency identical.
9. Report tolerances, pass count, first failure, covered shapes, kernel latency, and baseline. Without measured data, call the result only a "design recommendation".
10. Distinguish numerical failures, compilation failures, and repository-raised `NotImplementedError`. Record unimplemented paths accurately and complete their implementation. Do not relax tolerances, skip tests, or describe a previously passing subset as a complete pass.

## Script Tools

- `scripts/validate_templates.py`: By default, validates structure, syntax, stale paths, and key tiling regressions. When `TILELANG_DEFAULT_TARGET=pto` is set and `--npu --num-cores <queried_AIV_core_count>` is passed, it compiles and runs representative Broadcast, EuclideanNorm, Softmax, Scan, and RoPE cases.

## Bundled Ascend Implementation References

- Files under `references/*/code/*_asc.py` contain Ascend host and kernel implementations for source reading. They are not tests, launch examples, or maturity-rated templates.
- An `_asc.py` reference may retain `target="pto"`; PTO is the corresponding Ascend backend path and does not make the implementation CUDA code.
- Use only `_asc.py` implementations from this corpus. Do not route to or copy CUDA implementations.
- These references do not require template-status registration or separate lowering, device-correctness, or performance validation merely to remain in the Skill. When their ideas are applied to a target operator, validate the resulting target implementation under that operator's normal generation or tuning workflow.

## Code Admission Rules

- Kernel implementations use only the Python TileLang DSL; tiling, dispatch, and mathematical references use ordinary Python. Derive from a similar implementation in the current repository or a matching bundled `_asc.py` implementation reference. Apply [Template Maturity](references/template_status.md) only to executable templates, not to the source-reading corpus.
- Do not hard-code `target="ascend"` in `@tilelang.jit`. Let `TILELANG_DEFAULT_TARGET=pto` select PTO unless a test explicitly compares targets.
- Ascend `T.Kernel` uses only a one-dimensional block grid and does not receive `threads=`; express the thread domain in `T.SimtVF(threads=N)`.
- Prefer execution domains used by similar validated implementations: regular elementwise work normally uses `T.SimdVF()`, while Norm/Reduction may use the repository-validated `T.SimtVF + T.reduce_*`; use `T.SimtVF()` for scattered indexing, complex branching, or atomic operations. Any execution-domain change requires measured correctness and performance validation.
- For current PTO Ascend GEMM, prefer the validated L1 B layout `[N, K]` with `transpose_B=True`. Other layouts are not prohibited by the API, but require validation against target-version lowering and the device. Zero fp32 L0C for the first valid K sub-tile and accumulate subsequent sub-tiles.
- Use fp32 accumulation as the starting point for fp16/bf16 reductions, normalization, softmax statistics, and matrix multiplication. Internal mixed precision, HF32, and approximate instructions may be experimental, but must preserve original tests, tolerances, and public contracts; keep them only when correctness and same-method performance justify doing so.
- For Vector, Cube, and mixed kernels, respectively use hardware-query-confirmed available AIV, AIC, and paired resources. Pass the core count explicitly when constructing a template; for small workloads, do not use more cores than independent tasks.
- Elementwise, gather/scatter, or layout-conversion optimization must build a task tree and sourced tile boundaries according to [Tiling, Task, and Vector Dataflow Search](references/elementwise/tiling_task_vector_search.md). Classify physical routes by the complete data path: an intermediate planar/scratch payload is a materialized transform and cannot substitute for direct contiguous loads followed by register reordering. Treat tile/task, dataflow, precision, and pipeline as orthogonal axes, and adjudicate interacting combinations through the same actually activated candidate.
- A UB budget must distinguish confirmed physical capacity, execution-domain reservations in the final kernel body made by current lowering, and the explicit buffer footprint; recompute it especially after adding or removing `T.SimtVF`. The budget also includes buffer versions, alignment padding, resident data, and a justified safety margin. Copy only `valid` elements for a GM tail block, while still allocating the UB for a complete SIMD/DMA footprint.
- Pipeline work must establish cross-iteration dependencies according to [Multiversion Pipeline Design](references/elementwise/double_buffer_design.md). Version every still-live buffer; use explicit stage storage when automatic analysis does not apply. Determine activation from generated code and same-method latency/overlap.
- Verify multi-axis tile domains according to [Persistent Multi-Axis Task Mapping](references/common/persistent_task_mapping.md) to avoid unnecessary flattened-index decoding in hot loops.
- When reduction compute is light and the contiguous non-reduction axis is wide, use [Wide-Output-Axis Tiling for Reduction](references/reduce/wide_output_tiling.md) to evaluate merging contiguous output tiles.
- Determine the shape matrix from the public-interface contract. If the interface does not support arbitrary tail blocks, cover only permitted remainder classes rather than treating a new fallback as an existing requirement. Also check forward/backward, dynamic shapes, very small/large shapes, NaN/Inf semantics, and empty-task boundaries.

## Operator-Family Routing

| Operator Family | Preferred Pattern | Implementation References |
|---|---|---|
| Elementwise / Fusion / Irregular | Persistent + UB staging + SimdVF; use SimtVF for irregular indices and split forward/backward kernels when useful | `references/elementwise/code/swiglu_*_asc.py`, `engram_gate_asc.py`, `engram_hash_asc.py` |
| Quant | Per-token/per-channel/per-block scaling, packed scale layout, stochastic rounding, segmented dispatch | [Quant guide](references/quant/guide.md) and `references/quant/code/` |
| Rowwise Reduction + Elementwise Epilogue | Exact reduction-axis UB + resident broadcast parameters + SIMD fp32 reduction; evaluate both single-row pipelining and cross-row batching | `references/reduce/code/normalize_weight_asc.py`, `engram_grad_w_reduce_asc.py`, plus [rowwise](references/reduce/rowwise_reduce_epilogue.md) and [batched short reduction](references/reduce/batched_short_reduction.md) |
| Reduction / Norm / Iterative State | fp32 accumulation, hierarchical reduction, fixed state, forward/backward, and fused GEMM epilogues | `references/reduce/code/sinkhorn_asc.py`, `norm_fn_asc.py`, and TileLang `examples/ascend/example_rmsnorm.py` |
| Transpose / Gather | Choose contiguous load + register reordering, SIMD gather, or temporary materialization based on source-window span and density; fall back to SIMT only for unsupported paths | `references/conversion/code/batched_transpose_asc.py`, `testing/ascend/layout/test_ascend_l0_transpose.py` |
| MatMul | L1/L0C, validated layouts, K-dimension pipelining; Stream-K/full-load/SWAT are candidates | TileLang `examples/ascend/example_gemm*.py` |
| FlashAttention | Cube/Vector dataflow, online softmax, fp32 state | TileLang `examples/ascend/flash_attention/example_mha.py` |
| TopK / Routing | SimdVF selection/reduction, grouped experts, padding, physical routing | `references/sort/code/moe_topk_gate_asc.py`, `examples/ascend/example_simdvf_topk_gate.py` |

## Reference Routing

Select the operator family from [references/index.md](references/index.md). Read matching `_asc.py` files as implementation references without assigning them template maturity. When selecting an executable template instead, determine availability using [template_status.md](references/template_status.md): `EXECUTABLE_BASELINE` is only a correctness starting point, and `PARTIAL`/`DESIGN_ONLY` are not directly copyable optimized templates.
