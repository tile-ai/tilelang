# TileLang GPU Schedule Templates

This document summarizes the GPU schedule rules in `tilelang/schedule/gpu/`, their workflows, and the changes made during development.

---

## Overview

TileLang's schedule rules automatically transform loop-level `te.compute` expressions into optimized tile-level GPU kernels. Rules are tried in priority order — the first rule that returns a non-`None` schedule wins.

**Default rule order** (`tilelang/schedule/gpu/__init__.py`):
1. **Matmul** — Matrix multiplication with tensor cores
2. **GEMV** — Matrix-vector products (split-K fast path)
3. **GeneralReduction** — All reduction patterns (simple, multi-step, multi-source)
4. **Transpose** — Tensor transpose via shared memory
5. **ElementWise** — Injective element-wise operations
6. **Fallback** — Generic catch-all

All rules extend `GPUScheduleRule` from `tvm.dlight.ScheduleRule`.

---

## 1. Matmul (`matmul.py`)

**Pattern**: Matrix multiplication `C[i,j] = sum_k(A[i,k] * B[k,j])`

**Workflow**:
1. Normalize and inline trivial blocks
2. Detect GEMM pattern via `is_gemv` (rejects) + block structure analysis
3. Select architecture-specific tile config:
   - Hopper (sm>=90): 128×256×64, 4 pipeline stages
   - Ampere (sm>=80): 128×128×32, 3 stages
   - Generic: 64×64×32, 2 stages
4. Schedule pattern:
   ```
   fuse batch+spatial → split [blockIdx.x, blockIdx.y]
   split K → [outer_k, inner_k]
   fill_at(blockIdx, ..., 0)                    # zero-init accumulator
   cache_read_at(outer_k, A, "shared.dyn")      # stage A tile
   cache_read_at(outer_k, B, "shared.dyn")      # stage B tile
   gemm_at(outer_k, ...)                        # tile-level T.gemm
   pipeline(outer_k, num_stages)                # software pipelining
   annotate_layout(A_shared, swizzle_layout)    # bank-conflict-free
   annotate_layout(B_shared, swizzle_layout)
   cache_write_at(blockIdx, C, "local.fragment") # write-back
   launch_thread(root, threads)
   ```

**Key feature**: Uses `gemm_at` which lowers to tensor core instructions (WMMA/MMA). Fragment layouts are inferred by `LayoutInference` to match hardware MMA shapes.

---

## 2. GEMV (`gemv.py`)

**Pattern**: Matrix-vector product `y[i] = sum_k(A[i,k] * x[k])` (inner reduction)

**Workflow**:
1. Normalize, inline, detect GEMV via `is_gemv()` analysis
2. Verify inner-reduction orientation via `normalize()`
3. Identify matrix, vector, and output buffers
4. Delegate to `_apply_splitk_fast_path`

**Split-K fast path** (CUDA only, uses `tir.Schedule` not `TileSchedule`):
1. Check eligibility: float dtypes, 16/32-bit, no accumulator promotion needed
2. Choose parameters via `_choose_splitk_schedule_params`:
   | Architecture | output_tile | reduce_threads | vec |
   |-------------|-------------|----------------|-----|
   | sm>=90, mixed (fp16→fp32) | 1 | 256 | 2 |
   | sm>=90, fp32 | 1 | 256 | 1 |
   | Generic, fp16/bf16 | 2 | 128 | 2 |
   | Generic, fp32 | 1 | 256 | 1 |
3. Schedule:
   ```
   split S → [bo, bi(=output_tile)]
   split R → [ro, tx(=reduce_threads), vec_loop] (if vec>1)
   reorder(batch, bo, bi, ro, tx, vec_loop, c)
   fuse(vec_loop, c)  or  fuse(tx, c) when vec=1
   fuse(batch, bo) → bind blockIdx.x
   bind bi → threadIdx.y (if output_tile>1)
   bind tx → threadIdx.x
   ```
4. Epilogue handling: If epilogue exists, set GEMV output scope to `shared.dyn`, then `reverse_compute_at`
5. Lowering: `LowerCrossThreadReduction → LowerInitBlock → ConvertBlocksToOpaque → CanonicalizeScheduledTIR`

**Performance**: ~82-83% of torch.compile (cuBLAS) on H100. The gap is due to scalar loads — `LowerCrossThreadReduction` generates serial accumulation loops that prevent vectorized memory access. This appears to be a fundamental limit of the split-K + cross-thread-reduction approach.

**What was explored**:
- Vectorized loads via `sch.vectorize()`: TVM's VectorizeLoop cannot vectorize reduction accumulation patterns
- Loop unroll annotations: Fails because `LowerCrossThreadReduction` generates variables referencing the loop
- rfactor-based approach: Enables compute vectorization but cache_read generates inefficient memory patterns
- TileSchedule-based approach: Fails with undefined variables in `MakePackedAPI`
- `__launch_bounds__` tuning: No performance impact
- nvcc flags (`--use_fast_math`, `-O3`): No improvement

---

## 3. GeneralReduction (`general_reduction.py`)

**Pattern**: Arbitrary reductions — simple (sum/max/min), multi-step (softmax, layernorm), multi-source

**History**: Created by merging the former standalone `Reduction` rule into a unified handler. `reduction.py` is now a backward-compatible shim that re-exports `GeneralReduction as Reduction`.

**Workflow**:
1. Normalize, inline contiguous spatial blocks
2. Classify: count reduction blocks, identify epilogues
3. Route to appropriate handler:

### Single-source reduction (most common)
```
fuse spatial → split [blockIdx, inner_s(=1)]
split reduction → [outer_chunk, inner_chunk]
cache_read_at(outer_chunk, input, "local.fragment")
cache_reduce_at(blockIdx, output, "local.fragment", init_value)
reduce_at(outer_chunk, ...)
bind + launch_thread
```

### Two-reduction chain (softmax pattern: max → subtract+exp → sum → divide)
```
Same as single-source but applied twice:
  1st reduction: cache_read + cache_reduce + reduce_at
  bridge blocks: cache_read + cache_write + parallel
  2nd reduction: cache_read + cache_reduce + reduce_at
  trailing output: cache_read + parallel
All under same blockIdx binding, shared launch_thread
```

### Multi-source reduction
When `_collect_input_buffers` finds multiple distinct inputs:
```
decompose_reduction (separate init from update)
fill_at(blockIdx, ..., init_value)
cache_read_at for each input buffer
(no reduce_at — explicit update loop retained, num_threads=1)
```

**Key helpers** (in `reduction_utils.py`):
- `_analyze_reduction_update`: Detects sum/max/min/and/or from the update expression
- `_infer_init_value`: Returns identity element (0 for sum, -inf for max, etc.)
- `_choose_num_threads`: Selects thread count based on reduction extent
- `_choose_reduction_step`: Determines reduction tile size for cache staging

---

## 4. LayerNormLike (`layernorm_like.py`)

**Pattern**: Two-reduction with center bridge — specifically `mean → center(x-mean) → variance(center²) → normalize`

**Workflow**:
1. Normalize, find exactly 2 reduction blocks
2. Inline all injective blocks before first reduction
3. Verify pattern: both reductions are "sum", second reduction's input is the square of the center bridge output
4. Identify center bridge block (writes to the buffer that the second reduction reads)
5. Inline non-center bridge blocks and trailing non-output blocks
6. Schedule:
   ```
   output block: fuse spatial → split [blockIdx, inner]
   compute_at all blocks under blockIdx (reverse order: 2nd reduction, center, 1st reduction)
   _schedule_single_source_reduction for 1st reduction (reduce_type="sum")
   _schedule_center_bridge: cache_read global inputs, cache_write, parallel
   _schedule_single_source_reduction for 2nd reduction (force_reduce_type="sumsq")
   output block: cache_read remaining global inputs, parallel
   launch_thread(root, max(thread extents))
   ```

**Key detail**: The second reduction uses `force_reduce_type="sumsq"` even though the IR shows a plain sum — because the input is `(x - mean)²` and the `reduce_at` primitive has a specialized `sumsq` lowering path.

---

## 5. Transpose (`transpose.py`)

**Pattern**: Tensor permutation requiring non-trivial data movement

**Workflow**:
1. Normalize, inline all blocks except the last (output)
2. Verify single block with injective iteration
3. Schedule:
   ```
   fuse all spatial loops → split [blockIdx, tile_outer, tile_inner]
   tile sizes: 128 (outer) × 128 (inner) — tuned for shared memory banks
   cache_read_at(tile_outer, input, "shared.dyn", disable_tma=True)
   annotate_layout(shared_buf, swizzle_layout)  # bank-conflict-free
   cache_write_at(tile_outer, output, "local.fragment")
   parallel(tile_inner)
   bind(blockIdx, "blockIdx.x")
   launch_thread(root, num_threads)
   ```

**Key feature**: Uses shared memory with swizzled layout as an intermediate staging area. The swizzle eliminates bank conflicts during the transposed read pattern. TMA is disabled because the transpose pattern doesn't benefit from it.

---

## 6. ElementWise (`element_wise.py`)

**Pattern**: Injective operations (no reduction) — activations, broadcasts, type casts

**Workflow**:
1. Normalize, inline all blocks except the last
2. Verify purely injective (all spatial loops)
3. Schedule:
   ```
   fuse all loops → split [blockIdx, inner_tile]
   # For each read buffer (if fragment-cacheable):
   cache_read_at(blockIdx, block, buf_idx, "local.fragment")
   cache_write_at(blockIdx, block, 0, "local.fragment")
   parallel(inner_tile)
   bind(blockIdx, "blockIdx.x")
   launch_thread(root, num_threads)
   ```

**Fragment caching check** (`_tile_aligns_with_suffix`): A read buffer is only cached into `local.fragment` if the tile's suffix dimensions align with the buffer's shape — specifically, the suffix product of the buffer shape must divide the tile size. This prevents misaligned fragment access patterns.

**What was fixed** (prior conversation): The suffix-product divisibility check was added to prevent crashes when buffer shapes don't evenly tile into the fragment dimensions.

---

## 7. Fallback (`fallback.py`)

**Pattern**: Anything not matched by other rules

**Workflow**:
1. Normalize, inline everything possible
2. Simple mapping:
   ```
   fuse all loops → split [blockIdx, inner]
   bind(blockIdx, "blockIdx.x")
   launch_thread(root, num_threads)
   ```

No caching, no fragment staging. Pure loop-to-thread mapping.

---

## Changes Made Across Development

### Reduction → GeneralReduction Merge
- **`general_reduction.py`**: New unified file combining simple reduction, two-reduction chains (softmax), and multi-source reductions
- **`reduction.py`**: Rewritten as backward-compatible shim (`GeneralReduction as Reduction`)
- **`reduction_utils.py`**: New file with 17 shared helper functions extracted from the original reduction code
- **`layernorm_like.py`**: Updated imports from `.reduction` to `.reduction_utils`
- **`__init__.py`**: Replaced `Reduction()` with `GeneralReduction()` in default rules

### ElementWise Fix
- Added `_tile_aligns_with_suffix` guard for fragment caching eligibility

### GEMV Investigation (no changes landed)
- Explored vectorization, rfactor, TileSchedule approaches — all hit fundamental limitations
- Performance ceiling identified at ~83% of cuBLAS for the split-K cross-thread-reduction approach

---

## Lowering Pipelines

### TileSchedule-based rules (Matmul, GeneralReduction, LayerNormLike, Transpose, ElementWise, Fallback)
```
Simplify → LowerCrossThreadReduction → LowerInitBlock →
PlanAndUpdateBufferAllocationLocation → ConvertBlocksToOpaque →
UnifyThreadBinding → CompactBufferAllocation → Simplify → CanonicalizeScheduledTIR
```
Then compiled via `tilelang.compile(mod["main"])`.

### GEMV (tir.Schedule-based)
```
LowerCrossThreadReduction → LowerInitBlock →
ConvertBlocksToOpaque → CanonicalizeScheduledTIR
```
Wrapped in `TileSchedule(mod)` for compatibility with the compilation pipeline.

**Critical ordering constraints**:
- `LowerCrossThreadReduction` before `LowerInitBlock` (needs `T.init()` block structure)
- `UnifyThreadBinding` before `CompactBufferAllocation` (GPU buffer compaction)
