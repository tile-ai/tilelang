# Row-Wise Reduction with Elementwise Writeback

Applies to operators that first compute a reduction scalar for each row and then use that scalar and optional broadcast parameters to write back the entire row.

## Implementation Essentials

- When the reduction axis already satisfies vector alignment, allocate UB according to the actual axis length; avoid letting `next_power_of_2` inflate multiversion buffers.
- Keep broadcast parameters that are invariant across rows resident as a single version outside the row loop; multiversion streaming inputs and outputs by stage. Assign contiguous rows to cores and use pipelining to overlap MTE/Vector. Determine the stage count through measurements based on UB and row width.
- Use fp32 vector accumulation and horizontal reduction in `T.SimdVF`. In a second pass, reread the original row from UB to generate the output, trading a small amount of recomputation for less register and intermediate-buffer pressure.
- Combine multiple scalar factors that are invariant within a row before entering the elementwise writeback.

## Outer-Record Batching Gates

- From the baseline, calculate the contiguous GM transfer bytes per record, the number of Persistent tasks, the number of waves per core, and UB usage including padding, resident values, and buffer versions. Use profiling to determine whether small DMA operations and fixed per-task overhead limit throughput.
- When the reduction axis is short, per-record DMA granularity is small, and enough outer records are available, the candidate pool must include at least one contiguous multi-record design with `rows_per_tile > 1`. Also read [Batched Short Reductions](batched_short_reduction.md). Meeting an external performance target does not replace A/B validation of this candidate.
- Search `rows_per_tile` geometrically, constrained jointly by contiguous layout, UB capacity, DMA granularity, task waves, and tail blocks; do not hard-code operator- or shape-specific heuristics. For multi-record inputs and outputs, prefer rectangular UB shaped `(rows_per_tile, reduce_width)` with contiguous DMA. GM tail blocks must access only valid records.
- A/B-test `rows_per_tile` and `num_stages` separately: first compare single-record and multi-record data flows, then search the pipeline stage count for each feasible data flow. If batching causes insufficient task parallelism, exceeds UB capacity, increases tail-block cost, or degrades measured latency, retain the single-record fallback.

## Selection and Validation

Evaluate this structure when the reduction axis can reside in UB, but do not infer "only one row can be processed at a time" from "one row can reside in UB." A/B-test exact axis length versus padding, single-record versus multi-record tiles, stage counts, and a two-pass SIMD recomputation versus a single-pass fragment implementation. Also check for accuracy or coverage changes caused by reduction order, tail records, and shape dispatch.
