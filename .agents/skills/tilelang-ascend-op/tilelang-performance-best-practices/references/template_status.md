# Template Maturity and Compatibility

Use this table to assess code availability before reading operator-family documentation. `PRODUCTION_REFERENCE` is a feature-gated status: the current kernel and launcher must contain the core structures summarized in this table and its linked documents. If those structures are missing, downgrade the current path to `EXECUTABLE_BASELINE`; historical optimization knowledge becomes a candidate only, and its performance conclusions do not carry over. The existence of a file does not mean that its optimization strategy has been implemented.

This maturity table applies to executable templates and strategy implementations. Full Ascend host-and-kernel files under `references/*/code/*_asc.py` are a source-reading corpus for generation and optimization work; they are intentionally not registered or classified here.

"Implemented and measured" in a structural document is a knowledge-evidence level, not a code-template maturity status, and does not determine candidate implementation priority. Applying it to the current implementation still requires checking applicability and revalidation.

| Status | Usage Rule |
|---|---|
| `PRODUCTION_REFERENCE` | The source of truth is in the current repository; reference the production file directly instead of copying its kernel into the Skill |
| `VERIFIED` | PTO lowering, device execution, and correctness were validated on the versions listed below; revalidate after porting to another version |
| `EXECUTABLE_BASELINE` | Executable correctness starting point with no performance conclusion; must not enter production dispatch directly |
| `PARTIAL` | Contains only part of the strategy structure or has regressed to a baseline; complete and validate it |
| `DESIGN_ONLY` | Contains only a design, tiling metadata, or Python helper; it is not a TileLang kernel |

## Agent Presentation Categories

Maturity describes implementation completeness; by itself, it is not a performance ranking. The agent must inspect both the "Validated Scope or Gap" column in this table and the corresponding operator-family documentation before using these presentation categories:

- ✅ **Optimization pattern can be copied directly**: A `PRODUCTION_REFERENCE` or current-version `VERIFIED` TileLang `.py` whose optimized structure is genuinely implemented. Its implementation skeleton may be reused. Describe it as a high-performance template or expected to be faster only when performance evidence applies to the current comparison methodology.
- ⚠️ **Executable baseline or partial implementation**: `EXECUTABLE_BASELINE`, `PARTIAL`, or a runnable implementation without the corresponding optimized structure/performance evidence. Use it only as a correctness starting point, implementation reference, or candidate requiring completion.
- ❌ **Design reference only**: `DESIGN_ONLY`, or material without an executable TileLang kernel for the current version. It must not enter a direct implementation plan.

Do not classify a file as "Optimization pattern can be copied directly" merely because a complete `.py` file exists. Do not claim that code outperforms the target repository's current implementation merely because it resides in this Skill.

## Historical Pre-Migration Validation Baseline

| Item | Validated Version |
|---|---|
| Historical operator-implementation snapshot | commit `5802406` |
| Historical framework snapshot | commit `84855805` |
| TileLang Python package | `0.1.12+cuda.git84855805` |
| PTOAS | `0.58`, commit `f8174330` |
| Backend | `TILELANG_DEFAULT_TARGET=pto`, Ascend NPU |
| Validation date | 2026-08-11 |

Templates whose build interfaces have changed in the table below must be revalidated in the current repository; historical records are not proof of a pass after migration. Before execution, run `python -c 'import tilelang; print(tilelang.__file__, tilelang.__version__)'`. A different version or source path is an unvalidated combination.

## Bundled Code Status

| Path/Strategy | Status | Validated Scope or Gap |
|---|---|---|
| `broadcast/code/broadcast_add_kernel.py` | `EXECUTABLE_BASELINE` | Requires revalidation after explicitly passing the queried core count; historically covered fp16 `3x129` and `2x257`; UB/fragment padded to 256, while GM accesses only valid elements |
| `broadcast/code/onedim_add_kernel.py` | `EXECUTABLE_BASELINE` | Shares the same implementation and tiler with broadcast add; it is not a distinct optimization strategy |
| `reduce/templates/dav310/kernel_utils.py:euclidean_norm` | `EXECUTABLE_BASELINE` | Requires revalidation after explicitly passing the queried core count; historically covered fp16 `65x65`, `2x129`, and `2x257`; padded with `next_power_of_2(R)`; supports only padded R `<=4096` |
| `reduce/.../euclidean_norm_*tail*.py` | `EXECUTABLE_BASELINE` | Calls the same padded full-load baseline; does not represent a dedicated tail optimization |
| `reduce/.../euclidean_norm_group_block_split_r.py` | `DESIGN_ONLY` | No verified split-R kernel; `build()` actively rejects use |
| `reduce/.../softmax_v2_base.py`, `ar_full_load.py` | `EXECUTABLE_BASELINE` | Stable serial fp32 full-load correctness path; no production performance conclusion |
| `reduce/.../softmax_v2_ar_small_r.py` | `PARTIAL` | Still reuses serial full-load; multi-row batching is not implemented |
| Other `softmax_v2_*` strategy files | `DESIGN_ONLY` | Contain only strategy metadata or an online-merge helper; no schedulable kernel |
| `scan/templates/dav310/scan_base.py` | `EXECUTABLE_BASELINE` | Requires revalidation after explicitly passing the queried core count; historically covered bf16→fp32 `2x257`; serial inner scan for correctness only |
| `cum_streaming_scan.py`, `cum_tile_resident_scan.py` | `EXECUTABLE_BASELINE` | Share the scan baseline; do not represent lane-parallel optimization |
| Other `cum_*` Python strategies | `DESIGN_ONLY` | Contain only strategy metadata, workspace logic, or dependency-pair helpers |
| `rope/code/rope_vf_common.py` | `EXECUTABLE_BASELINE` | Requires revalidation after explicitly passing the queried core count; historically covered bf16 half-split `2x2x128`; contiguous baseline with compile-time position offset |
| `conversion/templates/dav3510/transpose_base.py` | `PARTIAL` | Provides only interface adaptation and the existing 64-alignment constraint check; requires a kernel factory validated in the current repository and does not contain a general transpose kernel |
| Other conversion Python strategies | `PARTIAL` or `DESIGN_ONLY` | Most only perform selection/constraint logic and do not provide an independent PTO kernel; inspect each file before use |

## Status Upgrade Gate

When upgrading `PARTIAL` or `DESIGN_ONLY` to `VERIFIED`, record all of the following: target hardware; CANN, TileLang, and current repository versions; complete command; shape/dtype; tolerance; lowering status; first failure; performance methodology; and raw result files. Without same-condition measured data, call it only an "implementation candidate" or "correctness baseline".
