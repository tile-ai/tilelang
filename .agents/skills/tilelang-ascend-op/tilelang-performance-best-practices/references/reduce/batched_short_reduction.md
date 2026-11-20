# Batched Short Reductions with Per-Record Writeback

Applies when each reduction is short, the outer dimension contains many records, and per-record normalization, scaling, or another elementwise epilogue follows the reduction. When each row first produces a reduction scalar and then writes back the entire row, also read [Row-Wise Reduction with Elementwise Epilogue](rowwise_reduce_epilogue.md).

**Evidence level: implemented and measured.** The structures below have been implemented in real Ascend kernels, with evidence from compilation and execution, accuracy tests, and like-for-like performance comparisons. This level indicates candidate credibility only; it does not replace current bottleneck prioritization or revalidation on target cases.

## Optimization Structure

- Process records in blocks, but DMA must follow the actual contiguous axis: inputs/outputs contiguous by record can be transferred as a whole block, while split-major inputs generally transfer one contiguous `[split, feature]` rectangle per record.
- Pad the UB feature axis for `SimdVF` to the vector lane count. Transfer only valid records/features in GM, use a valid-lane mask for short-axis reduction and writeback, and write scalar results through a single lane.
- Accumulate vector partials along the short feature axis, and combine scalar partials with broadcast loads. A dot product may use masked `vcadd`/`vdupv`; verify the reduction order.
- For UB with an explicit stage dimension, select the slot using each core's wave and apply `annotate_manual_multi_buffer`. For pipelined buffers without an explicit stage dimension, use `annotate_buffer_versions`. Do not mix the two addressing schemes on the same buffer.
- Keep `Persistent(..., num_stages=N)`, buffer versions, and any required `enable_offset` consistent, and verify actual overlap from generated addresses and events. Keep resident values single-versioned.
- When batching, SIMD footprint, DMA form, and buffer lifetimes share the same layout, first implement them as one data-flow hypothesis, then A/B-test the parts that can be separated independently. Tail blocks must access only valid GM; computing the full UB footprint is allowed when it has no side effects.
- Treat `rows_per_tile` and `num_stages` as independent search axes. First select geometric candidates for batching granularity according to contiguous layout, UB budget, DMA granularity, and task waves; then measure the pipeline stage count for feasible tiles. A stage search on a single record cannot replace multi-record candidates.
- Combining partials within the producer and the contiguous `n_splits==1` fast path are independent candidates; evaluate them only when data dependencies and target cases match.

Before reuse, verify batching granularity, reduction order, partial count, transfer form, buffer versions, and launcher parameters. A negative result from a different context eliminates only the current combination.

## Validated Structural Combinations

| Path | Implemented Physical Structure |
|---|---|
| Normalization after reduction | Multi-record batching and lane-padded UB; transfer one contiguous rectangle per record for split-major input, pair explicitly staged input with automatically versioned output, and perform batched writeback after accumulating broadcast scalars and vector partials |
| Elementwise epilogue after reduction | Use rectangular UB and batched DMA for contiguous records; independently generate an fp32 reduction scalar for each record, then write back per record within the tile, with the tail tile accessing only valid GM |
| Reduction with parameter gradients | Batched DMA for input and output, with multiversioned streaming buffers; use masked lane reduction/broadcast for the dot product and fill-back, and write scalar results only through valid lanes |
