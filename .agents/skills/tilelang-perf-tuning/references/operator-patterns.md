# Operator Patterns

Read only the sections that match the target operator. Patterns generate candidates; they are not copy-paste templates.

## Reduction / Norm / Softmax

- Low-precision inputs start from fp32 statistics/accumulation state; stable softmax uses `max` and `sum(exp(x-max))`.
- When the reduction axis fits on-chip, compare the exact axis length against power-of-two padding; padding amplifies DMA, UB, and multi-version footprints.
- When each row's statistics are written back elementwise, keep broadcast parameters resident in a single version and reuse statistics in registers/UB where possible; compare re-reading the row against saving intermediates.
- With short reductions and many outer records, testing contiguous multi-record DMA with `rows_per_tile > 1` is mandatory; A/B `rows_per_tile` and stages independently.
- When the non-reduction contiguous axis is wide, increase the output tile to amortize task/DMA/addressing setup, and recompute the task waves and core count.
- Row-wise and column-wise reductions have different data contiguity; model and route them independently instead of hiding both paths behind one generic tile.
- With a fixed small state iterated over many rounds, compare keeping the state resident across rounds, saving only the snapshots the backward pass needs, and reducing barriers beyond the true dependencies.

## GEMM / Quant / LSTM

- Adapt GEMMs from the Cube paths already running in this repository; confirm A/B layouts, `transpose_B`, L1/L0 allocation, and the output path.
- L0C defaults to fp32: zero it at the first valid K tile, then accumulate; nested K sub-tiles may only be zeroed at the global first tile.
- Jointly search M/N/K tiles, L1/L0 footprint, the number of independent output tiles, and K pipelining; never pursue larger tiles in isolation.
- When weights are reused across time steps, batches, or output tiles, prefer comparing converted-weight L1/UB residency against precomputed input contributions.
- Quantization jointly models packed loads, unpack/dequantization, scale broadcast, ND→NZ/layout conversion, and MMAD; fusing dequantization inside the producer can avoid a full weight materialization.
- Native low-precision input/residual paths and fast math must pass the original tolerances; recurrent states and long-chain accumulation are usually more sensitive than independent input contributions.
- For small M/B, check padded MMAD waste and batch parallelism; grouping improves local efficiency but may reduce the task count.

## Scan / Recurrent / Stateful

- First mark the state axis that must stay serial and the batch/channel/feature axes that can be parallel; never attempt to flatten true dependencies.
- Keep states, carries, loop invariants, and broadcast parameters resident across steps where possible; choose registers or UB based on state size, live registers, and code volume.
- Compare how loop orders affect state round trips, contiguous access, and task count; after changing the loop order, re-model tails and synchronization.
- A scalar broadcast can cost more than the arithmetic itself; verify the real instruction costs of broadcast loads, vdup/vmuls/vaxpy, and the current lowering.
- Splitting the state axis across cores becomes a candidate only when cross-core reduction/boundary synchronization cost is controllable; first quantify the per-step synchronization count and worst-case latency.
- For long sequences, state the per-state minimal operation count and the dependency-chain lower bound, separating the algorithmic limit from the code-generation throughput gap.

## Attention / Sparse Attention

- Model the physical dataflow of Q/K/V together with the online softmax: Q residency, K/V tiling, max/sum states, output accumulation, and gather paths.
- Softmax max/sum and GEMM accumulators default to fp32; check causal/all-masked rows, tails, and output dtypes.
- GQA/MLA, Dk/Dv, topK, and S1 change Cube padding, random gather traffic, and task count; group them by the actual path.
- Divisions/modulos in the hot loop, sub-block routing, and index-table generation should be hoisted or replaced by affine task mapping first.
- When random indexing has no batched DMA, adjudicate scalar/SIMT, indexed vector, bounded contiguous windows, and materialized gather separately; missing one gather API never proves convergence.
- Split-K, multi-kernel partial/merge, and Cube/Vector forwarding must account for all workspaces, launches, synchronization, and whole-chain time.
- For small G/M, Cube padding can be far larger than the useful FLOPs; report both the useful FLOPs and the actual padded-compute lower bound.

## Elementwise / Broadcast / RoPE

- Contiguous elementwise work prefers GM→UB→Vector→UB→GM; move constants and reuse tables out of the Persistent hot loop.
- For broadcast, compare resident inputs, cross-row/cross-token reuse, and shape-generic tails; avoid reloading small tables per task.
- RoPE half-split and interleaved variants use different physical reorderings; check whether sin/cos cover the half dimension or the full dimension, and avoid duplicate loads.
- Token/row-aligned blocking can turn hot-loop addresses affine and eliminate divisions/modulos; first quantify the cost of those scalar operations in the generated code.
- For interleaved/short records, prefer comparing contiguous load + register select/interleave against indexed gather before writing any full planar temporary layout.
- For fp16/bf16 transcendental functions and rotation statistics, choose fp32 or a verified fast path per the original precision requirements, converting the output last.

## Gather / Scatter / Layout / Slice

- Separate sequential index/output streams from random source/destination streams; use wide DMA transactions for the sequential stream, then choose SIMT, indexed vector, or windowed reorder for the random stream.
- For regular mappings such as stride-2/4, interleave/deinterleave, and transpose, verify contiguous load + register reorder first; check for overlapping loads, in-flight widening, and redundant relays.
- Scatter must distinguish unique targets, repeated targets, and atomic semantics; a race-free test never proves plain stores correct.
- Model slice/transpose by the longest contiguous suffix and stride across 2D copies, whole-row copies, register reorder, and gather fallbacks.
- When the output exceeds L2 or physical traffic already sits near the HBM lower bound, do not drive evidence-free micro-tuning toward impossible targets; still adjudicate the routes that can reduce physical traffic.
- A dtype change alters lane counts, load alignment, and register packing; int8/int64 usually need an independent cost chain and cannot directly inherit the fp16 route.

## General Acceptance

Every pattern must return to the current source and cases: confirm the API/lowering, capacity, precision, the branches actually activated, and same-methodology performance. A pattern hit does not imply a gain, let alone an inherited historical speedup.
