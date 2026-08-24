---
name: tilelang-layout
description: Use when working with TileLang layouts — the Layout/Fragment classes, layout inference and its cost model, the CuTe layout algebra and TileLang-to-CuTe conversion, swizzles, or when debugging layout-related passes and their annotations.
---

# TileLang Layout System

## Core types (`src/layout/layout.h`, Python `tilelang/layout/`)

**`tl::Layout`** maps logical coordinates to physical coordinates:
`input_size_` (logical shape) + `forward_index_` (one PrimExpr per output
axis). Expressions are written in process-wide placeholder Vars
`InputPlaceholder(i)` (printed `_i`, `_j`, ...). Key APIs:
`InputShape/OutputShape/GetForwardIndex/Forward(vars)/Inverse/
InverseWithLevel/Reshape/DetectInjective/DebugOutput`. `OutputShape()` is
DERIVED from the expressions' ranges by the Analyzer, not stored.

**`tl::Fragment : Layout`** adds a thread dimension for register buffers:
`forward_thread_` (PrimExpr, may reference `ReplicationPlaceholder()`,
printed `_rep`) and `replicate_size_`. `ThreadExtent()` =
`max(forward_thread)+1` under bound domains — also derived, not stored.
`thread_range_` (via `BindThreadRange`) offsets the thread space for warp
specialization; the forward map itself stays in normalized `[0, T)`.
Replication semantics: each logical point is held by `replicate_size`
threads; `Replicate/DeReplicate/CondenseReplicateVar` manipulate the rep
axis.

**Gotchas (verified, easy to trip on):**

- `FragmentNode::GetForwardVars()` PREPENDS `ReplicationPlaceholder()`
  when `replicate_size > 1` (`src/layout/layout.cc:1103`). The input
  placeholders are always the TRAILING `InputDim()` entries. Python code
  that zips `get_forward_vars()` against the shape positionally is wrong
  for replicated fragments.
- The canonical (thread, slot, rep) packing lives in
  `FragmentNode::InverseWithLevel` (`src/layout/layout.cc:1133`): rep
  becomes a trailing ORDINARY input dim of extent `ReplicateExtent()`
  (substituting the placeholder), thread a trailing output. The inverse
  then maps `(slot..., thread) -> (coords..., rep)` with rep last;
  `loop_partition.cc` consumes exactly this ordering and guards
  replica-zero for stores. Mirror this packing when you need a fragment
  as a plain multi-output layout.
- Layouts print via `DebugOutput()` / Python `repr`:
  `Fragment((2,) -> (2,), replicate: 256, thread: _rep, index: (_i,),
  thread_range: I.Range(0, 256))`. Structured fields are directly
  accessible from Python: `replicate_size`, `get_thread_size()`,
  `forward_thread`, `forward_index`, `thread_range`, `get_input_shape()`.

**Swizzles** (`tilelang/layout/swizzle.py`, `src/layout/swizzle_mode.*`)
are XOR-based shared-memory layouts. They are NOT expressible as strided
layouts; the CuTe side models them as a separate `Swizzle` functor
(`ComposedLayout`), never as (shape, stride) modes.

## CuTe layout algebra (`src/layout/cute_layout.{h,cc}`, `tilelang/layout/cute.py`)

A full CuTe implementation maintained for the MMA/TMA backends, exposed to
C++ (`namespace tvm::tl::cute`) and Python (`tilelang.layout.cute`):

- Layouts are `(shape, stride)` IntTuple trees, **COLUMN-major (first mode
  fastest)** — opposite of TileLang's row-major intuition. Multi-output
  codomains use `ScaledBasis` strides (`v@axis`, `E<i>`).
- Algebra: `Coalesce`, `RightInverse`, `LeftInverse`, `Composition`,
  `Complement`, `Cosize`, `LogicalDivide`, `Filter`, `Restrict`,
  `Layout::WithShape`, `Parse`/`Print` (exact CuTe spelling).
- Converters: `LayoutFromTileLang` (single flat output; multi-output
  tl::Layouts are serialized ROW-major, so outputs `[thread, slot]` yield
  `thread * slots + slot`), `LayoutFromTileLangHierarchical` (per-axis
  recovery onto basis axes), `ComposedLayoutFromTileLang` (swizzle
  recovery). All use probe-then-prove: numerically probe strides at
  one-hot points, then symbolically PROVE equivalence — a wrong recovery
  cannot slip through; failure returns `None`/nullopt.

**Failure conventions (must respect):** only the three `*FromTileLang`
converters return Optional. Every other algebra op ICHECK-crashes on its
preconditions (composition divisibility, non-constant extents in the
probe, complement non-injectivity, restrict rank...). Wrap calls when the
input is untrusted. `RightInverse` is PARTIAL: it silently drops stride-0
and non-const-stride modes and inverts only the maximal contiguous chain —
check bijectivity via `size(right_inverse(F)) == size(F)`.

**Conversion caveats:** the converters read only `GetForwardIndex()`; a
Fragment's `forward_thread` and replication are silently ignored unless
you pack them into a plain multi-output Layout first (see the canonical
packing above; substitute `ReplicationPlaceholder` with a trailing input
var — if it leaks into the probe, conversion gracefully returns None).

## Layout inference (`src/transform/layout_inference/`)

`tl.LayoutInference` assigns a layout to every fragment buffer and
parallel loop nest. Three levels: **strict** (annotations, MMA-imposed) →
**common** (BFS propagation through shared buffers) → **free** (per
connected component, try every member as inference root, keep the
cheapest attempt). Results are stored as IR annotations: `layout_map`
(Buffer -> Layout, on the SBlock) and `parallel_loop_layout` (Fragment,
on the outermost parallel For); `ParallelLoopLayoutValidator` enforces the
annotation contract.

"Cheapest" is a pluggable policy (`layout_cost_model.{h,cc}`):

- `tl.layout_cost_model="io-aware"` (opt-in): every
  fragment<->global copy and global-touching parallel loop is charged
  `max(bandwidth bytes, issue bytes)` under the attempt's layouts, scored
  symbolically on the CuTe algebra (pack -> `LayoutFromTileLang` ->
  `RightInverse` -> `Composition`; vector width read off coalesced modes;
  segments counted at warp/step granularity). Registers are the tiebreak.
  Statements outside the model are charged a conservative worst case — an
  attempt must never profit from opacity.
- `tl.layout_cost_model="register-count"` (default): register-slots-only
  ordering.
- The scoring formulas are guarded by the Python parity check
  `maint/layout_inference/run.py --cute` (symbolic scorer vs an
  independent NumPy enumeration oracle); keep `cute_model.py` in lockstep
  when changing `layout_cost_model.cc`.

Hardware geometry is parameterized: lane width via `MaxVectorLoadBits`
(`src/transform/loop_vectorize.h`, SHARED with the vectorizer so the
model's width beliefs match codegen), warp size from the target's
`thread_warp_size`, segment granularity 128B (see `BindMemoryGeometry`).

## Downstream consumers

- `loop_partition.cc`: partitions parallel loops per-thread via the
  fragment inverse; emits replica-zero guards for replicated stores.
- `loop_vectorize.cc`: plans vector widths (`GetVectorizeSize`,
  `IndicesCanVectorize`); the cost model mirrors its judgment.
- TMA/MMA lowering: `src/cuda/op/tma_layout.cc` and
  `producer_consumer_ws.cc` recover shared-buffer swizzles via
  `ComposedLayoutFromTileLang`; tcgen05/wgmma macro generators
  (`tilelang/cuda/intrinsics/macro/`) round-trip TMEM layouts through
  `to_tilelang`/`from_tilelang_hierarchical`.

## Tools

- **`maint/layout_inference/`** — the layout verification harness (see its
  README): golden layout snapshots per constructed case (`run.py`),
  lowered vector-width anchors (`--anchor`), and symbolic-vs-oracle parity
  (`--cute`, backed by `cute_model.py` + the numpy `oracle.py`). Extend it
  with a new case whenever you change inference, the cost model, or the
  converters; goldens are recorded then human-reviewed (`--record`).
- **Diagnostics**: the inference and cost-model passes log decisions via
  DLOG (debug builds); `tl.layout_visualization_enable` renders layouts.
- **Python inspection**: run `tl.transform.LayoutInference()` on a module
  and read the annotations (see `maint/layout_inference/common.py` for
  the extraction idiom), or `cute.Layout.from_tilelang(...)` to see a
  layout's (shape, stride) normal form.

## File map

| area | files |
|---|---|
| core types | `src/layout/layout.{h,cc}`, `tilelang/layout/{layout,fragment}.py` |
| swizzle | `src/layout/swizzle_mode.*`, `tilelang/layout/swizzle*.py` |
| CuTe algebra | `src/layout/cute_layout.{h,cc}`, `tilelang/layout/cute.py` |
| inference | `src/transform/layout_inference/layout_inference.cc` |
| cost model | `src/transform/layout_inference/layout_cost_model.{h,cc}` |
| MMA layouts | `src/layout/gemm_layouts.cc`, `src/layout/tcgen05_layout.*` |
| verification | `maint/layout_inference/` |
