# TileLang/PTO Reference Index

First read the [General Implementation Guide](tilelang_pto_guide.md) and select the relevant family below. Full Ascend host-and-kernel implementations live in family `code/` directories and may be read directly for generation and optimization ideas. They are source-reading references, not maturity-rated templates. Use [Template Maturity](template_status.md) and [Validation Gates](validation.md) only when selecting or applying executable templates.

Structural documents provide distilled optimization decisions, while family `code/` directories preserve complete Ascend host-and-kernel structures. Read only the matching family and treat the code as implementation vocabulary rather than inherited correctness or performance evidence for the current target.

| Operator Family | Documentation | Code References | Key Optimizations |
|---|---|---|---|
| Common | Skill named `npu-arch` · [copy](common/datacopy_optimization_design.md) · [tail](common/tail_block_design.md) · [resident](common/ub_resident_design.md) · [task map](common/persistent_task_mapping.md) | TileLang `examples/ascend/` | Hardware facts, merged movement, padding, residency, multiversioning, multi-axis task mapping |
| Broadcast | [guide](broadcast/broadcast_design.md) | [broadcast code](broadcast/code/) | Single-/multi-axis, resident input, shape specialization |
| Conversion | [guide](conversion/guide.md) | [batched transpose](conversion/code/batched_transpose_asc.py) · `testing/ascend/layout/test_ascend_l0_transpose.py` | Multi-axis Persistent mapping, contiguous copy, UB padding, SIMD/SIMT transpose |
| Elementwise / Fusion / Irregular | [vector](elementwise/vector_efficiency_design.md) · [tiling/task](elementwise/tiling_task_vector_search.md) · [pipeline](elementwise/double_buffer_design.md) · [indexed short record](elementwise/indexed_short_record.md) | [SwiGLU forward](elementwise/code/swiglu_forward_asc.py) · [SwiGLU backward](elementwise/code/swiglu_backward_asc.py) · [Engram gate](elementwise/code/engram_gate_asc.py) · [Engram hash](elementwise/code/engram_hash_asc.py) | SimdVF, fused forward/backward, short-record indexing, resident metadata, dynamic tails, 2/3-stage pipelines |
| Quant | [guide](quant/guide.md) | [per-token cast](quant/code/per_token_cast_asc.py) · [segmented per-channel cast](quant/code/per_channel_cast_with_psum_asc.py) · [lossless per-block cast](quant/code/per_block_cast_lossless_asc.py) | Multi-path dispatch, scale layout, stochastic rounding, expert segmentation, packed low-precision conversion |
| MatMul | [guide](matmul/guide.md) | TileLang `examples/ascend/example_gemm.py` | Tile search, L1/L0, pipelining, residency, Stream-K |
| Reduction | [guide](reduce/guide.md) · [rowwise reduce epilogue](reduce/rowwise_reduce_epilogue.md) · [wide output](reduce/wide_output_tiling.md) · [batched short reduction](reduce/batched_short_reduction.md) · [fixed small state](reduce/fixed_small_state.md) | [normalize weight](reduce/code/normalize_weight_asc.py) · [partial gradient reduction](reduce/code/engram_grad_w_reduce_asc.py) · [Sinkhorn](reduce/code/sinkhorn_asc.py) · [fused norm](reduce/code/norm_fn_asc.py) · [templates](reduce/templates/dav310/) | fp32 state, rowwise and cross-row reduction, forward/backward, fixed resident state, fused GEMM epilogues, split-axis |
| FlashAttention | [guide](flash_attention/guide.md) | TileLang `examples/ascend/flash_attention/example_mha.py` | Resident Q, online softmax, Cube/Vector forwarding |
| Scan | [guide](scan/guide.md) | [scan strategies](scan/templates/dav310/) | Row owner, resident carry, lane scan, three-stage split |
| RoPE | [guide](rope/guide.md) | [rope_vf_common.py](rope/code/rope_vf_common.py) | fp32 pair rotation, layout specialization |
| Scalar | [guide](scalar/guide.md) | Per-operator Python factories | Constant folding, address reuse, lifetime, code size |
| SIMT | [guide](simt/optimization-guide.md) | Reduction templates and SIMT examples in the current repository | Threads, fragments, branch specialization |
| Sort/TopK | [guide](sort/radix_sort.md) | [production MoE TopK](sort/code/moe_topk_gate_asc.py) · `examples/ascend/example_simdvf_topk_gate.py` | Padded loads, local selection, grouped experts, hierarchical merge, physical routing |
| Conv | [guide](conv/guide.md) | PTO GEMM primitive | Tiled im2col/grouped GEMM |
| Compute/Communication Fusion | [guide](mc2/guide.md) | Local GEMM + production distributed runtime | Asynchronous collectives, chunk pipeline |

## Validation Requirements

- `references/*/code/*_asc.py` is an implementation-reading corpus. Keeping these files in the Skill does not require template-maturity registration, lowering, device correctness, or performance validation.
- Templates copied from the target repository must still rerun their corresponding tests on the current branch.
- For Scan, RoPE, and converted legacy templates, perform PTO lowering before device-accuracy and end-to-end performance testing.
- First classify maturity according to [template_status.md](template_status.md); `PARTIAL`/`DESIGN_ONLY` must not be listed as directly copyable templates.
- fp16/bf16 reduction, norm, softmax, and GEMM use fp32 statistics or accumulation state by default.
- The test matrix must include tails, dynamic shapes, NaN/Inf, very small tasks, and multistage workspaces.
