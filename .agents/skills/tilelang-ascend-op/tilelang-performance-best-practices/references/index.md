# TileLang/PTO Reference Index

First read the [General Implementation Guide](tilelang_pto_guide.md), [Template Maturity](template_status.md), and [Validation Gates](validation.md). Reference existing implementations directly from the source of truth in the target repository. Locate TileLang APIs through the actual import path rather than copying a second version into the Skill.

Structural documents provide only distilled optimization decisions. They do not index or require reading corresponding historical optimization implementations; implement and validate independently from the current target baseline.

| Operator Family | Documentation | Code References | Key Optimizations |
|---|---|---|---|
| Common | Skill named `npu-arch` · [copy](common/datacopy_optimization_design.md) · [tail](common/tail_block_design.md) · [resident](common/ub_resident_design.md) · [task map](common/persistent_task_mapping.md) | TileLang `examples/ascend/` | Hardware facts, merged movement, padding, residency, multiversioning, multi-axis task mapping |
| Broadcast | [guide](broadcast/broadcast_design.md) | [broadcast code](broadcast/code/) | Single-/multi-axis, resident input, shape specialization |
| Conversion | [guide](conversion/guide.md) | `testing/ascend/layout/test_ascend_l0_transpose.py` | L0 layout testing and movement-API reference; generic transpose requires a validated factory |
| Elementwise / Gather | [vector](elementwise/vector_efficiency_design.md) · [tiling/task](elementwise/tiling_task_vector_search.md) · [pipeline](elementwise/double_buffer_design.md) · [indexed short record](elementwise/indexed_short_record.md) | `examples/ascend/example_simdvf_per_token_cast_to_fp8.py`, `testing/ascend/layout/` | SimdVF, short-record packing, shared-field fusion, partial packing, static full-tile/JIT, contiguous movement, task flattening, instruction dataflow, 2/3 stages |
| MatMul | [guide](matmul/guide.md) | TileLang `examples/ascend/example_gemm.py` | Tile search, L1/L0, pipelining, residency, Stream-K |
| Reduction | [guide](reduce/guide.md) · [rowwise reduce epilogue](reduce/rowwise_reduce_epilogue.md) · [wide output](reduce/wide_output_tiling.md) · [batched short reduction](reduce/batched_short_reduction.md) · [fixed small state](reduce/fixed_small_state.md) | [reduction strategies](reduce/templates/dav310/) | fp32 state, full-load, fused row reduction and elementwise writeback, wide-output tiles, recompute, online, SIMD short reduction, resident small state, split-axis |
| FlashAttention | [guide](flash_attention/guide.md) | TileLang `examples/ascend/flash_attention/example_mha.py` | Resident Q, online softmax, Cube/Vector forwarding |
| Scan | [guide](scan/guide.md) | [scan strategies](scan/templates/dav310/) | Row owner, resident carry, lane scan, three-stage split |
| RoPE | [guide](rope/guide.md) | [rope_vf_common.py](rope/code/rope_vf_common.py) | fp32 pair rotation, layout specialization |
| Scalar | [guide](scalar/guide.md) | Per-operator Python factories | Constant folding, address reuse, lifetime, code size |
| SIMT | [guide](simt/optimization-guide.md) | Reduction templates and SIMT examples in the current repository | Threads, fragments, branch specialization |
| Sort/TopK | [guide](sort/radix_sort.md) | `examples/ascend/example_simdvf_topk_gate.py` | Padded loads, local selection, hierarchical merge |
| Conv | [guide](conv/guide.md) | PTO GEMM primitive | Tiled im2col/grouped GEMM |
| Compute/Communication Fusion | [guide](mc2/guide.md) | Local GEMM + production distributed runtime | Asynchronous collectives, chunk pipeline |

## Validation Requirements

- Templates copied from the target repository must still rerun their corresponding tests on the current branch.
- For Scan, RoPE, and converted legacy templates, perform PTO lowering before device-accuracy and end-to-end performance testing.
- First classify maturity according to [template_status.md](template_status.md); `PARTIAL`/`DESIGN_ONLY` must not be listed as directly copyable templates.
- fp16/bf16 reduction, norm, softmax, and GEMM use fp32 statistics or accumulation state by default.
- The test matrix must include tails, dynamic shapes, NaN/Inf, very small tasks, and multistage workspaces.
