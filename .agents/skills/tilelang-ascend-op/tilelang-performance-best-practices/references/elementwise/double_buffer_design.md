# Elementwise Multiversion Pipelines

## Goals and Applicability

A multiversion pipeline allows the transfer-in for tile `i+1`, computation for tile `i`, and transfer-out for tile `i-1` to overlap when the hardware permits. It applies to operators where both GM transfers and Vector/Cube computation take measurable time, each core has enough independent tiles, and UB capacity can accommodate all versions. For small tasks, a single iteration, or a pipeline stage already near its theoretical lower bound, multiversioning may only add UB usage, synchronization, and prologue/epilogue overhead.

Before implementation, map the dependencies for each logical iteration:

```text
GM --CopyIn--> input/local buffer --Compute--> output/local buffer --CopyOut--> GM
```

Version every UB/L1 buffer that will be written in the next iteration while it may still be read by Compute or CopyOut from the preceding iteration. Determine input, output, index, and temporary-intermediate versions from their actual lifetimes; do not assume that "double-buffering only the input" is sufficient. Read-only resident data that remains unchanged across iterations usually stays single-versioned.

Also account for the total footprint of all versions, padding, resident data, and the number of effective iterations per core. Fall back to a single stage when the number of effective iterations is less than the stage count. Consider three stages only when a pipeline gap remains after using two stages and capacity permits it.

## Automatic and Manual Multiversioning

When access relations are simple and affine, and the compiler can uniquely identify producers and consumers, try automatic versioning first. Prefer peeling unconditional, fixed-extent full iterations away from tail handling, and use `T.Pipelined` to expose a canonical CopyIn -> Compute -> CopyOut body:

```python
input_ub = T.alloc_shared((tile_elems,), dtype)
output_ub = T.alloc_shared((tile_elems,), dtype)
T.annotate_buffer_versions({input_ub: 2, output_ub: 2})

for tile in T.Pipelined(full_tile_count, num_stages=2):
    T.copy(src[tile], input_ub)
    compute(input_ub, output_ub)
    T.copy(output_ub, dst[tile])

# Process the remainder tile through a separate single-stage path, outside the steady-state body above.
if has_tail:
    process_tail(...)
```

This automatic template has four structural conditions that must all be satisfied:

1. Allocate `input_ub/output_ub` for **one logical tile**. Do not first add an explicit `[2, ...]` stage dimension and then call `annotate_buffer_versions(...: 2)`. That requests both manual and automatic version expansion and may cause incorrect extent rewriting.
2. Version only mutable buffers that genuinely remain live across iterations in the steady-state loop. Keep read-only LUT/index buffers, tail buffers outside the pipeline, and fallback buffers not involved in this body single-versioned.
3. Each execution of the `T.Pipelined` body must use the same rank, shape, and copy extent for CopyIn/Compute/CopyOut. Place full/tail branches that alter the accessed range, zero-work branches, and dynamic remainder handling outside the body.
4. The first automatic representative should preferably access the original logical shape of the versioned buffer directly. Do not access it through a `view` with a hidden extent, a flattened alias, or an explicit stage subscript. If an intrinsic needs a base pointer, it must still be traceable to the same logical buffer and a fixed range.

`T.Persistent(..., num_stages=2)` is not prohibited, but the presence of `num_stages` alone does not make it the standard form of an automatic pipeline. Treat it as an automatic candidate only when it likewise provides a steady-state task body with an unconditional fixed extent and unique producers/consumers. Otherwise, use the `T.Pipelined(full_tile_count, ...)` representative above to evaluate automatic versioning first. CANN/compiler upgrades do not replace these structural conditions.

`num_stages` expresses scheduling intent only; it cannot replace buffer versions or dependency verification. Views/aliases, flattened indices, or gather/scatter may cause automatic analysis to claim the wrong storage, expand an extent, or lose a version relation. Minimize the case and identify the failure point first, then switch to explicit stage storage and a manual annotation that actually exists in the currently installed version:

```python
input_ub = T.alloc_shared((2, tile_elems), dtype)
output_ub = T.alloc_shared((2, tile_elems), dtype)
T.annotate_manual_multi_buffer(input_ub, output_ub)

for wave in T.Pipelined(full_waves, num_stages=2):
    stage = wave % 2
    copy_in(wave, stage)
    compute(stage)
    copy_out(wave, stage)
```

Use the same stage for every buffer that remains live across iterations. Distinguish ordinary dynamic UB indexing from a dynamic base pointer passed to an explicit SIMD intrinsic: scalar/SIMT `ub[index]`, dynamic slots used by `T.copy`, and per-lane offsets used by `vgather2/vscatter` must not be generalized as "dynamic addresses are unsupported" because one case failed. The risky combination is using `ub[stage, ...]` as the base pointer of an intrinsic such as `vld/vsts/vgather2/vscatter`; the current backend may fail during version analysis, instruction selection, or address legalization.

If the failure evidence points to such a dynamic base, eliminate only the "dynamic stage addressing" combination and invoke a generic body with a compile-time constant stage from an outer stage branch:

```python
stage = wave % 2
copy_in(wave, stage)  # Whether T.copy can use a dynamic slot must still be verified against the current lowering.
if stage == 0:
    compute(input_ub[0], output_ub[0])
    copy_out(output_ub[0], wave)
else:
    compute(input_ub[1], output_ub[1])
    copy_out(output_ub[1], wave)
```

This gives the SIMD body a compile-time constant base while keeping the algorithm, tile, and runtime kernel generic; do not specialize by test shape. A static stage must not be promoted based only on successful compilation. It must still be paired with a stage-1 version that uses the same arithmetic and task mapping to validate latency and overlap. Verify APIs and lowering against the actually imported TileLang/PTO implementation or an implementation validated in the repository.

Record pipeline candidates across four dimensions, and let a failure reject only the exact combination:

1. **Storage**: automatic versions, or explicit `[stage, ...]` storage with a manual annotation;
2. **Addressing**: a dynamic stage, or static stage 0/1 bodies formed by an outer stage branch/macro;
3. **Control flow**: a conditional loop, or an unconditional complete full-wave after peeling the tail;
4. **Scheduler**: `Persistent`/`Pipelined`, the stage count, and the number of steady-state iterations per core.

A pipeline candidate is admissible when each core has enough independent iterations and profiling shows that both MTE and Compute have time that can overlap. After admission, try the automatic combination first. Consider the automatic combination complete only if every buffer live across iterations is correctly versioned, the generated code contains the expected switching/synchronization, and the latency/overlap improvement in paired stage-1/stage-2 measurements is reproducible.

Automatic versioning that is `not eligible`, extent/alias errors, lowering failures, partial versioning of live buffers, or no measured effect eliminates only the automatic combination. Regardless of whether the user provides a numerical target, before declaring that an admitted pipeline direction has failed or converged, complete the following manual representative or provide conclusive evidence that it cannot be implemented because of the current lowering, capacity, dependencies, or APIs:

- Make the input, output, and every temporary buffer live across iterations explicitly multiversioned;
- Use a manual annotation, with static stage bodies when necessary;
- Use an unconditional full-wave with `T.Pipelined(..., num_stages=2)`, handling the tail separately;
- Benchmark it immediately alongside a stage-1 control with the same tile, arithmetic, and task mapping.

When automatic versioning is already fully effective, the equivalent manual implementation need not be tested, but record `NOT_NEEDED_AUTO_COMPLETE` explicitly. Automatic versioning failing for one work granularity or data flow is not evidence that manual pipelining is infeasible for a different work granularity or data flow.

Handwritten `set_flag`/`wait_flag` or event ordering is not the default alternative. Consider it only when the current repository or actually installed source contains a verified example with the same access pattern. A deadlock eliminates only the corresponding event scheme.

The scheduler must be able to see the steady-state pipeline body; do not wrap the entire CopyIn/Compute/CopyOut sequence in a runtime condition. Peel the tail with generic bounds:

```text
full_waves = work_items // items_per_wave
pipeline(full_waves):
    unconditionally execute a complete CopyIn -> Compute -> CopyOut
if work_items % items_per_wave != 0:
    process the final tail wave separately
```

Dynamic shapes, out-of-bounds protection, and tail-block semantics must still be preserved. Choose `Pipelined` or `Persistent` according to the iteration mapping of the current API, not merely by its name.

## Criteria for Effectiveness

Successful compilation, the presence of two buffers, `num_stages > 1`, correct execution, or a change in a single pipe ratio cannot independently prove that the pipeline is effective. Compare stages 1 and 2 as a pair using the same measurement methodology:

- Kernel latency improves beyond measurement noise, reproducibly;
- Effective bandwidth calculated from actual GM bytes in the source, or actual compute throughput, improves;
- MTE/Vector/Cube overlap improves as hypothesized, and total execution time decreases accordingly;
- When necessary, inspect version switching, synchronization, and full-wave scheduling structure in the generated IR/code.

If latency/overlap does not improve beyond noise, record "compiled successfully but ineffective." Record the complete combination identifier for a stage-2 failure; do not extrapolate it to other tiles, Vector data flows, or control flows.

The pipeline must not change computation, casts, or output semantics. Initialize every version of a tail tile independently and transfer only its valid range. Cover stage/tile boundaries, unaligned tail blocks, and extrema required by the interface. A candidate with multiversion, synchronization, or scheduling warnings must run all target cases in the same process and be retested with a different case order. Eliminate it if order-dependent drift, intermittent errors, or memory corruption occurs. Do not relax accuracy or skip cases.

## Executable Code and Evidence

First search the current repository for real uses of `annotate_buffer_versions`, `annotate_manual_multi_buffer`, `num_stages`, `T.Pipelined`, and `T.Persistent`; then inspect the actually imported TileLang source and PTO lowering. Begin with `examples/ascend/example_simdvf_vecadd.py`, `examples/ascend/example_manual_multibuffer.py`, `examples/ascend/example_crosslevel_multibuffer.py`, `examples/ascend/example_rmsnorm.py`, and other matching Ascend implementations in the repository. For every reuse, revalidate the applicable access pattern, version, and device evidence.
