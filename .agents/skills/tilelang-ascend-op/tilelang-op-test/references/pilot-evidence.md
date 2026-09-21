# Initial Pilot Evidence

The following historical pilot records predate the Skill migration. They illustrate methods for reviewing contracts and evidence; they are not an index of files in the current repository or proof that current code passes. Only historical filenames and symbols are retained here. After migration, locate the source again, establish tests, and execute validation for the current operator.

## Rerun After Skill Changes

On 2026-09-09, after selecting compatible headers for Bisheng and GCC 11, three targeted pilot paths were rerun serially through `pytest_st_evidence.py`:

| Pilot | Exact Validation Point | Result |
|---|---|---|
| `engram_hash` | Exact comparison for an Ascend token tail block | 1 collected, 1 passed |
| SwiGLU | Invalid-input rejection contract for two optional parameters | 2 collected, 2 passed |
| `bmp_to_patches` | Source-embedded Ascend boundary case with patch size 17 | 1 collected, 1 passed |

Every execution-evidence record reports `EXECUTED_PASS_REQUIRES_CREDIBILITY_REVIEW`: execution succeeded, but a delivery `PASS` still requires independently satisfying the contract, oracle, assertion, and coverage gates.

## `engram_hash`: Exact Integer Comparison and Tail-Block Coverage

Evidence:

- Public signature/docstring: `engram_hash_kernel.py::engram_hash`;
- Independent PyTorch oracle: `engram.py::engram_hash_ref`;
- CUDA and Ascend kernels: `engram_hash_cuda.py`, `engram_hash_asc.py`;
- Historical pytest: `test_engram_hash.py`.

The outputs are integer indices and therefore require exact comparison. Existing correctness parameters vary `num_tokens` by test level but fix `max_ngram_size=3`, `num_ngram_layers=2`, and `num_embed_table_per_ngram=8`. The Ascend implementation derives token tiling from those structural parameters. Before adding a case, review whether valid structural combinations or token/core tails are missing. Unless an invalid-input rejection contract is added, the vocabulary size must remain positive so that modulo semantics are valid.

## SwiGLU: Parameter Dependencies and Auxiliary State

Evidence:

- Public dispatch/parameter validation: `swiglu_forward_kernel.py`;
- Independent formula and optional-parameter behavior: `swiglu.py::swiglu_forward`;
- CUDA/Ascend implementations: `swiglu_forward_cuda.py`, `swiglu_forward_asc.py`;
- Historical pytest: `test_swiglu_forward.py`;
- Historical description of the Ascend support scope: `ascend_migration_effort_analysis.md`.

Relevant dependencies include `routed_scaling_factor → topk_weights` and `clamped_count → clamp_value`; the count buffer contains four elements. The Ascend path currently accepts unscaled BF16/FP32 output rather than CUDA's FP8 variant. Test numerical output and the clipping counter separately. Use the PyTorch reference as the primary oracle and CUDA only as supporting evidence when both backends support the relevant feature.

The PyTorch docstring currently calls the prefix and input a mask in one parameter description. Use the function signature and `get_mapping_from_psum` behavior as authoritative, report the documentation inconsistency, and do not copy it directly into the contract.

## `bmp_to_patches`: Source-Embedded Ascend Boundary Tests

Evidence is concentrated in `image_embed_ops.py`:

- `bmp_to_patches` documents parameters, transformations, output shape, and dtype;
- `bmp_to_patches_ref` uses CPU/PyTorch to remove BMP row padding, optionally flip vertically, convert BGR to RGB, normalize, convert to BF16, and reshape into patches;
- `test_bmp_to_patches_ascend` uses `rtol=0, atol=0.01` to cover multiple patch sizes, cropping layouts, and flip combinations;
- CUDA and Ascend kernels provide differential and path evidence.

The test is embedded in a file that does not follow the `test_*.py` naming convention, so it must be selected explicitly. Existing cases already include an irregular patch size; do not add another prime number merely to claim boundary coverage. Review genuine grouped-row/fallback paths, persistent-task tails, row padding, flipping, and the `fp16_norm_safe` normalization branch. Unless a more reliable source exists, `std <= 0` and other undocumented scalar ranges remain contract gaps.

## Generalization Checks

After the three pilots, `state_cache.py` was used to validate stateful and in-place mutation rules. It contains explicit parameter validation, a CPU reference, cyclic indexing, dynamic strides, and protected-storage checks. This case validated a stateful-review rule: comparing only the logical view cannot prove that adjacent storage remains intact.

On 2026-09-09, after the recorded `CPLUS_INCLUDE_PATH` aligned Bisheng with GCC 11 headers, all three parameterized cases of `test_load_state_to_cache_ascend_strided` were collected and passed on Ascend. The same target had previously failed to compile with GCC 13 headers. Classify the earlier run as blocked by the compilation environment and the later run as executed and passed; however, a pass conclusion must still satisfy the contract, oracle, assertion, and coverage reviews required by this Skill.

`mhc_post` was used to validate multiple outputs and gradients. Its existing comprehensive test compares the forward result and four gradients against `mhc_post_ref`. Because the public wrapper has almost no written documentation, new boundary conclusions still require more reliable contract evidence.
