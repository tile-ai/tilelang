# Searching Tiling, Task Mapping, and Vector Data Flows for Element-Wise / Gather Kernels

This guide applies to Vector kernels for output-layout transformations, gather/scatter, element-wise conversion, and image/sequence packing. Its goal is to enumerate high-value structures systematically, avoid fixing tiling too early with conservative constants that have not been compared, and avoid optimizing only outer tasks while overlooking parallel work inside them.

## 1. Draw the Work-Decomposition Tree First

Express the computation with at least two levels:

```text
outer logical item (row / patch / token / segment)
└── inner independent chunk (pixel round / channel chunk / vector tile)
```

Record the number of outer items, inner chunks, available Vector Cores, and iterations per core separately. If there are fewer outer items than cores but each item contains multiple mutually independent chunks, flatten `(item, chunk)` into a candidate task space and compare it with having one core process the entire item serially. Outputs may still be written back as disjoint slices. If chunks have reductions, ordering dependencies, or overlapping writes, document explicitly why they cannot be flattened.

Flattening proves only that the task count increases; it does not prove that the scheduling form is optimal. `T.serial(core_id, total_tasks, num_cores)` gives each core an ordinary strided serial loop and is suitable for short per-core iterations. `T.Persistent([total_tasks], num_cores, core_id, ...)` explicitly exposes a persistent cross-core task space to the scheduler and is better suited to outer loops that need unified task mapping or later pipelining, but it may also add fixed overhead. If a slow case requires inner flattening to occupy all cores, compare these two outer mappings with identical tiles, arithmetic, memory access, and tail handling, or prove their equivalence from the current lowering/generated code. Never generalize one operator's measurements into “Persistent is always faster than serial.” The actual gate is that a task-scheduling candidate cannot be closed after testing only one cross-core mapping.

When a large item is substantially slower than other cases, do not skip this analysis merely because it uses the fallback. If flattening makes each task too small, also compare a granularity where one task combines several adjacent chunks.

Before the first modification, record each case's work tree, outer/flattened waves, payload, measured time, lower bound, and dominant pipe. Structurally triggered directions requiring validation are independent of the current candidate pool; changes in candidate priority must not make them disappear. The optimization workflow invoking this guide defines the concrete persistence format and statuses.

## 2. Enumerate Sourced Tile Boundaries

Do not derive candidate tiles/groups from one empirical constant. At minimum, list the values from the following boundaries that apply to the current layout:

1. **Complete contiguous semantic unit**: A full row, one contiguous segment, one channel plane, or the largest contiguous range that does not cross metadata/object boundaries.
2. **SIMD boundary**: Lane count, effective payload of one load/select/gather/scatter instruction, and integral numbers of vector chunks.
3. **DMA boundary**: Contiguous bursts, full row width of a two-dimensional copy, alignment, and stride restrictions.
4. **Capacity boundary**: UB/L1 maxima after including all resident buffers, indices, padding, safety margin, and 1/2/3 versions.
5. **Parallelism boundary**: Task counts required for at least one wave, two waves, and multiple steady-state waves; also record the tail-tile ratio.

Deduplicate these boundaries to form a small candidate set. First eliminate invalid values with resource formulas, then measure values that imply different data flows. A safety margin is allowed, but state the specific lowering behavior or temporary storage it protects. Do not replace a complete contiguous unit or capacity-limit candidate with an arbitrary cap.

Record these five boundary classes and their source values for every case. A measured candidate can decide only a boundary matching its actual granularity; a small group, empirical cap, or particular parallel granularity cannot substitute for a complete contiguous unit. Before stopping, provide measurements for every boundary class or explicit capacity/lowering evidence that it is infeasible. Do not silently discard one.

## 3. Build Vector Data-Flow Families and an Instruction-Cost Table

Classify candidates first by the storage levels and access methods through which data actually passes, rather than presupposing an answer from an operator's API spelling. Common, non-exhaustive data-flow families include:

| Data-flow family | Representative physical path | Primary costs |
|---|---|---|
| Scalar/SIMT | Per-thread indexing, branches, arithmetic, reads, and writes | Thread context, scalar indexing, branches, and active-thread count |
| memory-indexed Vector | Vector lanes gather/scatter UB/GM or read a table/LUT | Discrete accesses, index registers, lane utilization, and table initialization |
| register-resident Vector | Contiguous Vector load followed by in-register select/shuffle/pack/convert/arithmetic and final write | Contiguous loads/stores, register rearrangement, conversions, and live-register pressure |
| materialized transform | Materialize an intermediate layout in UB/L1, then transpose/permute/DMA or write contiguously | Additional full reads/writes, temporary capacity, synchronization, and alignment |

These names describe physical data routes; they do not prescribe fixed intrinsics. One implementation may combine multiple families. A family may be closed with per-case evidence only when current interface semantics, actual TileLang APIs/lowering, or data dependencies prove that it does not apply.

Classify a physical route by every storage level traversed from source to result, not merely by the final arithmetic instruction. Writing interleaved/discrete data into a complete planar, transposed, or scratch layout before loading it contiguously into registers is `MATERIALIZED_TRANSFORM`. Only loading contiguously from the original or segmented source window and completing select/shuffle/pack in registers without writing back a complete intermediate payload is `CONTIGUOUS_LOAD_REGISTER_REORDER`.

### 3.1 Two Common Physical Routes for the Same Index Rearrangement

For `y[j] = f(x[index[j]])`, per-lane gather and “contiguous load + register select/shuffle” solve the same problem at different indexing levels:

```text
memory-indexed:
    generate/load index → each lane accesses UB/GM by index → compute

register-resident:
    contiguous load of source window → lane index within window → register select/shuffle → compute
```

Before creating the first candidate pool, record these source-window characteristics for every case:

```text
source_span_bytes = minimum contiguous window in bytes covering all required source elements
source_density = distinct source bytes actually consumed / source_span_bytes
window_vregs = number of Vector registers required to hold the source window
index_regularity = fixed/periodic/affine or runtime-dynamic/data-dependent (affects index generation but does not decide the route alone)
source_reuse = number of valid outputs or reuse instances produced from the same window
```

| Source layout and access characteristics | Preferred candidate | Rationale |
|---|---|---|
| Candidate source range fits a reasonable register footprint; density is high or the source can be reused across lanes/iterations | Contiguous load + register select/shuffle/pack | Replaces per-lane memory access with contiguous access; indices may be regular or data-dependent |
| Only a few elements are selected from a wide span; the candidate range does not fit selectable registers; a full window would overfetch or overflow | gather / memory-indexed Vector | Reads only required elements and avoids complex cross-register rearrangement |
| Segments are internally contiguous but discrete from one another | Segmented contiguous loads + in-segment register rearrangement, compared pairwise with gather | Neither local contiguity nor inter-segment discreteness alone determines the global result |

When indices cross row, segment, or object boundaries, calculate a valid window for each region; do not fabricate contiguity with one large bounding interval. Obtain register payload, dtype, cross-register selection, and lowering restrictions from the current TileLang/PTO version and target architecture rather than hard-coding platform widths. When converting an interleaved record to a field-separated layout such as `[a0,b0,c0,a1,b1,c1,...]`, consuming most fields in the final window makes it a high-density window and warrants contiguous-load deinterleaving. Selecting only a few fields across a wide span is usually better suited to gather.

To determine whether register select/shuffle covers the target mapping, verify the dtype, single-/multi-register selection range, and PTO lowering. Confirm gather index units, source dtype, widening performed along the gather path, and mask semantics from actual source. Repository examples prove only that a spelling exists, not that it wins for the current data distribution.

For the same effective output volume, count contiguous and indexed GM/UB accesses, select/shuffle/pack operations, table/LUT initialization, indices, casts/arithmetic, predicates, temporary materialization, lane utilization, and live-register pressure. Candidates with different source spelling but identical generated instructions and memory accesses may be merged. Candidates with different physical routes must not be merged merely because both use `T.SimdVF`.

Keep orthogonal axes fixed as much as possible during route comparison: use the same tile/group, task mapping, numerical precision, output-storage method, and stage count, and replace only the input/rearrangement route being tested. If a candidate simultaneously changes precision, the cast chain, output scatter/store, or pipelining, its result decides only that **complete instruction chain**. It cannot directly establish that “`vld+vselr` is slower than gather” or the reverse. When axes cannot be held completely fixed, list the differences and create a new candidate for the lowest-cost representative that may still have high value.

Initial candidate records must include the window metrics above, estimated costs for both routes, and status. When both routes are implementable and may affect the dominant bottleneck, test one lowest-cost representative of each. Keeping only one route requires capacity, semantic, lowering, or strict cost-dominance evidence. If a bounded window fits in registers, closing contiguous load + register reorder also requires the actual searched paths, symbols, and lowering conclusions. After the kernel becomes Vector-bound, if a remaining route can reduce indexed accesses or full materialization, or increase effective lanes, add or reopen it; do not converge merely because semantic GM bytes approach their lower bound. A claim of a “minimal sequence” must also state the minimum semantic operations, generated instruction/access counts, API search scope, and decisions for alternative routes.

### 3.2 TileLang `vld + vselr` Implementation Template

Keep these three concepts distinct:

- **Contiguous window**: `S.vld` first loads source data contiguously from UB into a Vector register.
- **Register-resident**: Subsequent rearrangement touches only that register and never writes a complete intermediate layout back to UB/L1.
- **Lane index**: In `S.vselr(src, index)`, `index[lane]` is a lane number inside the `src` register, not a UB address. It need not have a fixed stride, but every index must remain within the register window selectable by that operation.

The neutral example below extracts 64 `a` values from a 256B interleaved `uint16` window `[a0,b0,a1,b1,...]`. It demonstrates only the physical route; a real kernel must adapt it for dtype, valid predicates, tails, and cross-register ranges:

```python
from tilelang.language import simd as S

with T.SimdVF():
    # A uint16 register has 128 lanes. Clamp indices in the high 64 lanes to
    # 0..127 as well, so vselr never receives an out-of-bounds index even when
    # the final store is masked.
    lanes = T.reinterpret(S.vci(0, T.int16), "uint16x128")
    local_lanes = S.vand(lanes, S.vdup(T.uint16(63), T.uint16))
    even_index = S.vmuls(local_lanes, T.uint16(2))

    interleaved = S.vld(src_ub[src_base])
    field_a = S.vselr(interleaved, even_index)
    valid64 = S.pset(16, "PAT_VL64")
    S.vsts(dst_ub[dst_base], field_a, valid64, dist="NORM_B16")
```

If the required window spans multiple registers, never pass global indices directly to one `vselr`. Partition the source range into provable local windows, remap indices for each segment, then merge/write the results. If cross-register selection and packing cost more than per-lane UB access, compare against `vgather2` with measurements. The index for `vgather2(base_ub, index)` is a per-lane element offset relative to the UB base. For a `uint8/int8` source, current TileLang widens gather results to `uint16/int16`, so include conversion, lane count, and subsequent arithmetic in the cost.

### 3.2.1 Lowest-Cost Type and Lane-Chain Gate

The first representative of the contiguous-load route must be an evidence-backed low-cost chain under the current API/lowering. Do not treat the cost of repairing lane layout after selection as intrinsic to the route. Before implementation, record each intrinsic's input/output dtype, valid lanes, `part` semantics, and whether it creates empty slots, then check:

1. Whether gather already widens along its path while the contiguous-load route requires explicit integer widening;
2. Whether direct conversion from a narrow integer to floating point produces a lane layout through `part=even/odd` that requires multiple `vintlv` repairs;
3. Whether a shorter chain first performs one integer widening and then converts to the target floating-point type;
4. When target precision permits, whether affine computation can be fused on a narrower floating-point type with operations such as `vmadd/vaxpy`, avoiding unnecessary FP32 widening, interleaving, and final `vpack`;
5. Whether the current mapping matches `vld2`/load distribution, store distribution, or another natively lowered layout operation, avoiding a manual duplicate of that operation.

For example, the following is only a low-cost shape to verify for a narrow integer field selected in registers; it is not a fixed answer for every dtype:

```python
raw = S.vld(src_ub[src_base])
field_u8 = S.vselr(raw, lane_index)
field_u16 = S.vcvt(field_u8, T.uint16, part=0)
field_f16 = S.vcvt(T.reinterpret(field_u16, "int16x128"), T.float16)
S.vmadd(field_f16, scale_f16, bias_f16, valid)
```

If a direct `u8 -> f16 -> f32` representative requires multiple split conversions, several `vintlv` operations, and `vpack`, while the integer-widening path above or another current lowering route remains undecided, mark the failed candidate as nonrepresentative. The workflow may eliminate that candidate according to its status, but it must not close the direct register-rearrangement route, and it must create a new lowest-cost representative. Conversely, widening performed along the gather path must count as an advantage; do not compare only the single instruction names `vld` and `vgather2`.

Do not rely on memory when locating evidence. From the actually imported TileLang root, inspect `vld`, `vgather2`, and `vselr` in `tilelang/ascend/language/simd.py`, and their lowering in `src/ascend/codegen/codegen_pto.cc`. Refer to `examples/ascend/example_simdvf_per_token_cast_to_fp8.py` and `examples/ascend/example_buffer_version_annotation.py` for SIMD and multi-version usage, respectively. Each concrete gather/select combination must still be verified against the current API, lowering, and tests; examples do not replace correctness and profiling for the target shapes.

Also distinguish an intrinsic's base from its index operand. Ordinary dynamic `ub[index]` is not universally prohibited, and the lane index of `vgather2/vscatter` is inherently a dynamic vector. However, when `ub[stage, ...]` becomes the base pointer for `vld/vsts/vgather2/vscatter`, it may trigger addressing or instruction-selection restrictions in the current backend. One failure with a dynamic base rejects only that addressing combination. Multi-version pipelining must still test a static base body formed by an outer stage branch. Base conclusions on current-version lowering, generated code, and a minimal reproduction; do not generalize them into permanent hardware restrictions.

Distinguish “change only task/group while inheriting the parent data flow” from “introduce or replace a physical route such as gather, contiguous load + register reorder, or full materialization.” If a register window is confirmed implementable and neither the baseline nor definitive evidence has closed direct register rearrangement, the first candidate whose hypothesis concerns the physical route must validate it. Familiarity with a gather example is not evidence for reordering candidates. The optimization workflow invoking this guide defines candidate fields and statuses.

### 3.3 Comparison Route: `vgather2 + vsts` in TileLang

When required inputs span multiple contiguous windows but UB indices can be generated in final-output order, also consider per-lane collection with `vgather2` followed by contiguous `vsts` output:

```text
output-order index → vgather2(raw UB) → Vector compute → vsts(contiguous output UB)
```

This route places layout-conversion cost on the input side. The index lanes for `vgather2` follow contiguous output-lane order, allowing `vsts` to store directly to output UB after computation. Compare it with “contiguous `vld` + register `vselr`/shuffle + `vscatter` or segmented store” using the same effective output payload. Count indexed UB reads, index generation/loading, widening along the path, cross-register rearrangement, and output stores separately. Failure or poor performance of one input route does not automatically reject another output-storage combination, and vice versa. Verify the actual implementation's dtype, mask, and `vsts` distribution against the current API/lowering.

## 4. Orthogonal-Axis Combinations and Reopening Gates

Do not record “one candidate” as only a name. For layout-conversion and gather/scatter operators, record it as a combination containing at least these orthogonal axes:

```text
(tile/group boundary, task mapping, Vector data flow, computation precision, buffer versions/pipeline, tail strategy)
```

A change on one axis may alter the assumptions of another. A larger group increases contiguous payload and amortizes LUT/metadata cost, but may also shift the bottleneck from DMA/scheduling to Vector rearrangement. A SIMD data flow may not pay off for a small tile but become critical for a longer contiguous tile. Therefore, failure of `group + original data flow` and failure of `original tile + SIMD` do not imply failure of `group + SIMD`.

The six-axis tuple is the candidate's identity, not descriptive text. Create a new candidate and retain the original result whenever tile/group, task mapping, physical data flow, precision, pipeline, or tail strategy changes materially. Never reuse an old ID for another combination. The current candidate pool is only an execution queue; the optimization workflow invoking this guide maintains the complete set of directions awaiting decisions.

If a tiling/task change improves contiguity or reduces fixed cost per task, and a Vector change can eliminate the dominant compute/rearrangement cost of that new tile, they form an interacting high-value combination. A candidate must actually cover the corresponding granularity and physical route before it can decide the combination. Before stopping, complete at least the following decision matrix, or provide current lowering, capacity, data-dependency, or instruction-cost evidence for every unexecuted quadrant:

| | Baseline Vector data flow | Candidate Vector data flow |
|---|---|---|
| Baseline tile/task | Baseline | Decide the Vector change independently |
| Candidate tile/task | Decide the tiling change independently | **Decide the combination** |

For an operator whose input order differs from its output layout, rewrite the layout mapping after grouping. If a native permute/transpose operation is absent or its lowering is unverified, do not fall back directly to element-wise SIMT and declare structural convergence. Compare memory-indexed Vector, register-resident Vector, temporary layout materialization, and other implementable routes against the actual APIs. When the producer can write the final layout directly, prefer fusing conversion into computation and include both eliminated UB reads/writes and added index instructions in the cost table.

If a complete contiguous unit can be defined, `FULL_CONTIGUOUS_UNIT × CONTIGUOUS_LOAD_REGISTER_REORDER` is a general interaction that must be decided independently. After testing a small grouping segment, do not close it by claiming that “a larger group is the same route.” A complete unit changes DMA length, task count, index amortization, and the available register-rearrangement window simultaneously.

After every measurement, determine whether the bottleneck shifted. If another axis in the candidate pool directly addresses the new bottleneck, immediately add or reopen the corresponding combination as a high-value candidate. Do not silently skip it merely because the initial list omitted that combination.

## 5. Limit Candidate Failures to Their Exact Scope

A candidate record must include at least tile/group, task mapping, execution domain, critical instruction sequence, buffer versions, pipeline control flow, and tail strategy. One failure rejects only that combination:

- A correctness failure in one gather index format does not reject other SIMD select/shuffle operations or index formats.
- Failure of planar/materialized Vector does not reject contiguous load + register reorder without a complete intermediate payload.
- Failure or insufficient benefit of a small group does not reject a complete contiguous semantic unit derived from another source value.
- A slower stage 2 for one tile does not reject stage 2 for another tile/control flow.
- Ineligibility for automatic buffer versioning does not reject manual multi-versioning with explicit storage.
- Failure to lower a dynamic stage address does not reject a static stage body.

Merge failure conclusions only when generated code proves that candidates are materially equivalent. If two changes lie on different orthogonal axes, the candidate record must state explicitly whether their combination was measured, has no interaction, or is infeasible based on definitive evidence.

To close an unimplemented route as “strictly dominated,” compare memory-access levels, loads/stores/gathers/scatters, conversions, temporary materialization, synchronization, and active lanes under the same effective payload, and provide current API/lowering feasibility evidence. Generic claims such as “more instructions” or “more complex code” are not sufficient closure evidence.

## 6. Structural Coverage Audit Before Stopping

If the performance target remains unmet or any case is still clearly slow, complete the following table before stopping:

| Structure family | Highest-value candidate tested | Result/evidence | Reason not implemented |
|---|---|---|---|
| Contiguous transfers and tile/group boundaries |  |  |  |
| Outer/inner task mapping |  |  |  |
| Indexed source-window diagnosis (span/density/register footprint/regularity) |  |  |  |
| memory-indexed Vector route |  |  |  |
| register-resident Vector route |  |  |  |
| Temporary materialization or another implementable Vector route |  |  |  |
| Interacting tile/task × Vector combination |  |  |  |
| Single-stage and complete multi-version pipelines |  |  |  |
| Low-overhead tail and small-task paths |  |  |  |

Every initially identified or subsequently reopened high-value candidate must have measured results or definitive inapplicability/elimination evidence. If an undecided item remains, the evidence-convergence stop condition is not satisfied. The optimization workflow invoking this guide defines the specific status names.
