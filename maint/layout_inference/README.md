# Layout inference verification harness

Constructed IR cases with reviewed expected layouts, for validating the
free-mode layout search — in particular the selection policy behind
`tl.layout_cost_model` ("register-count" = the default ordering,
"io-aware" = the opt-in global-memory model in
`src/transform/layout_inference/layout_cost_model.cc`).

Why this exists:

- The cost model's score decides which layout attempt wins; any change to
  the model (or a future fast-path/slow-path split inside it) can silently
  flip the winner. Golden layouts pin the current answers so a flip shows
  up as a reviewable diff, not a runtime perf mystery.
- Cases where the two policies **disagree** are the calibration corpus:
  each disagreement is a concrete claim ("the io-aware pick is faster on
  hardware") that can be benchmarked.

## Usage

The default `register-count` model also tries one scalar plan at each
unannotated reducer-update root. Both native and scalar plans use the unchanged
spill/register-slot score; native plans win same-root ties. This can remove
replicated column accumulators without forcing scalar layouts for full
reductions. Explicit widths and layouts remain authoritative. The opt-in
`io-aware` model retains its existing candidate search and scoring.

```bash
python run.py                # verify all cases against the pinned goldens
python run.py --case NAME    # substring filter
python run.py --show         # also print the inferred layouts
python run.py --record       # rewrite goldens from current behavior
python run.py --target NAME  # pin another target suite (see below)
python run.py --list-targets # list the pinned suites and their configs
python run.py --anchor       # lower fully; check per-buffer vector widths
                             # in device TIR against VECTOR_ANCHOR
python run.py --cute         # compare symbolic scores with the exact oracle
```

## Targets are pinned, and goldens are per target

Layout inference is **target-dependent**: the cost model's vector-width
arithmetic reads `thread_warp_size`, the reduction paths read the target's
warp/thread geometry, and those differ across `sm_90` / `sm_100` / `Metal`.
Historically this driver ran `determine_target("auto")`, which resolves to the
**host GPU's** compute capability — so the same source produced a different
answer per machine, and a run on an `sm_100`/`sm_103` box drifted across most
of the suite at once. That is a property of the question, not of the layouts.

So the driver pins a target and stores goldens per suite:

```
expected/<suite>/<case>.json    # {variant: {model: {"buffers": ..., "loops": ...}}}
expected/<suite>/target.json    # the config the suite was recorded under
expected/<suite>/excluded.json  # {case: reason} cases this suite does not cover
```

`--target` accepts a pinned suite name, `auto` (host-detected, for ad-hoc
investigation), or an inline JSON target config. Running against a suite whose
recorded `target.json` does not match the run's target is an error, not a
drift report, and a suite with no goldens says exactly how to record it. A case
listed in `excluded.json` is skipped by name with its reason printed, for
build configurations that cannot produce its answers at all.

| suite | target | notes |
|---|---|---|
| `cuda-sm90` (default) | `{"kind": "cuda", "arch": "sm_90"}` | where the original goldens were recorded |
| `cuda-sm100`, `cuda-sm103` | same, newer arch | layout selection genuinely differs from `sm_90` |
| `metal` | `{"kind": "metal"}` | host suite for Apple silicon; excludes the reducer-v2 cases |

A non-CUDA build can still run the `cuda-*` suites: layout inference only
needs the backend's `tl.copy` implementation to be *registered*, not a GPU or
a toolkit. `src/cuda/CMakeLists.txt` keeps `op/copy.cc` and `op/tma_layout.cc`
in the always-compiled source list for that reason (≈6 s per file on a
non-CUDA build, ≈0.6 MB in `libtilelang`). Where a build genuinely cannot
infer a case, the driver reports it; `--allow-unsupported` downgrades those to
a counted skip instead of a failure. `--anchor` needs the full lowering
pipeline and therefore supports fewer builds than the golden check — it
honours the same flag.

## Build configuration matters too

Pinning the target makes a run reproducible across *machines*, but not across
*build configurations*: layout inference calls into backend code that a build
may or may not compile. `-DUSE_CUDA=OFF` leaves `src/cuda/op/copy.cc` out, so an
`sm_90` target resolves a different `tl.copy` implementation and
`reducer_scalar_candidates` comes out fully replicated — including the `width=4`
variant of #3171's unit test, which then reports `combine_size == 128` instead
of the `4` it asserts. The same assertions pass on a CUDA-enabled build, which
is where the `cuda-sm90` goldens are meaningful.

So: run the `cuda-*` suites against a CUDA-enabled build (the CI gate does, via
`if: contains(matrix.runner.toolkit, 'CUDA')`). Drift reported by a non-CUDA
build on these cases is a build-capability mismatch, not a layout regression.

`--anchor` closes the loop between the model and the real vectorizer: the
cost model scores a layout assuming a vector width, and the anchor reads
back what the vectorizer actually emitted for the winning layout under an
explicit `io-aware` config. Each case declares `VECTOR_ANCHOR = {variant: {buffer:
lanes}}`; variants without one print observed widths for review. A
mismatch means the model's width belief and codegen diverged — exactly the
drift the shared MaxVectorLoadBits policy is supposed to prevent.

## CuTe parity checking (`--cute`)

`python run.py --cute` validates the symbolic formulation used by the
io-aware scorer on the in-tree CuTe layout algebra
(`tilelang/layout/cute.py`, `src/layout/cute_layout.cc`):

- `cute_model.py` packs each fragment as
  `(coords..., rep) -> [thread, slot]` (the canonical packing of
  `FragmentNode::InverseWithLevel`), converts it with
  `cute.Layout.from_tilelang` into ONE plain strided layout computing the
  enumerator's cell index, then derives the byte-address layout with
  `right_inverse` + `composition` and reads the vector width off the
  coalesced slot modes. Segment counts evaluate the derived layout at
  warp/step granularity, once per issued vector lane rather than once per
  logical point and replica.
- `oracle.py` is the independent arbiter: a numpy implementation of the
  retired exact-enumeration formulas, with whole-grid evaluation of the
  fragment's own forward expressions.
- Every fragment golden layout is scored by both paths, as a load and as a
  store (replication gating included), and (V, issue, bw, segments) must match
  exactly. `CUTE_STATEMENTS` in a case supplies real enclosing-buffer
  shapes where they differ from the fragment shape (offset_region_copy).

Current status: 112/112 statements match with a 100% conversion hit rate.
The production scorer in `layout_cost_model.cc` uses this formulation; its
mode arithmetic was additionally audited once, in-tree, against a full
exact-enumeration oracle across the layout-relevant test corpus (~264
tests + this harness, zero disagreements, zero over-cap skips) before that
oracle was removed. This `--cute` check is now the standing guard: any
change to the scoring formulas must keep it green (update `cute_model.py`
in lockstep with `layout_cost_model.cc`).

Recording is not approval: after `--record`, read the diff under
`expected/<suite>/` and convince yourself every changed layout is intended
before committing. Structural invariants (each case's optional `check`) are
enforced even in record mode, so a recording can never bless a layout that
violates a case's documented contract.

## Layout of a case

Each `cases/*.py` defines:

- `VARIANTS: dict[name -> callable]` — each callable returns a fresh
  `PrimFunc` (lazy so construction happens under the driver's target).
- optional `check(variant, model, result)` — assertions that must hold
  regardless of the golden, e.g. "this fragment must be fully replicated
  under the io-aware model".

`result` is `{"buffers": {name: layout}, "loops": {nest_key: layout}}`
where each layout is a STRUCTURED dict (`common.layout_to_dict`):

```json
{"kind": "Fragment", "input_shape": [2], "output_shape": [2],
 "forward_index": ["_i"], "replicate": 256, "threads": 256,
 "forward_thread": "_rep", "thread_range": [0, 256]}
```

Golden diffs therefore point at the exact field that moved
(`replicate: expected 1, got 128`), and checks assert on fields
(`frag["replicate"] == 1`) instead of substring-matching a print format.
The driver snapshots the `layout_map` block annotation plus each parallel
nest's `parallel_loop_layout` annotation (see `common.py`).

## Current cases

| case | what it pins |
|---|---|
| `elementwise_copy` | Baseline: both models must agree on the coalesced, vectorized roundtrip layout. Primary equal-score anchor. |
| `fp8_copy` | 1-byte dtype: the 16-element vector width at the wide end of the shared width policy. |
| `broadcast_read` | Issue #1729. The models disagree **by design**: register-count keeps the thread-collapsed legacy pathology (golden documents it); io-aware must pick full replication + a non-replicated coalesced loop (enforced by `check`). |
| `transposed_store` | Load and store pull the layout in opposite directions; goldens record each model's trade-off. fp32 variant: the models pick different layouts — benchmark-worthy. |
| `mixed_dtype_chain` | fp16/fp32 fragment pair in one component; vectorization is sized by conflicting dtypes. The models pick different vector splits — benchmark-worthy. |
| `reduce_broadcast` | Softmax-shaped row-reduce + broadcast consume: the most common real-kernel component. Both models agree, including the reduced fragment's canonical partial replication. |
| `offset_region_copy` | Multi-block tiled copies whose region mins carry block indices (the model's "foreign vars"): offset regions must rank exactly like zero-offset ones. |
| `shared_staging` | global→shared→fragment→global chain: the shared-side copy is outside the io model, so the fragment is decided by the copy-out alone; goldens would surface any change to that boundary. |
| `reducer_scalar_candidates` | Register-count can choose scalar column ownership while preserving native packed layouts for a full reduction; io-aware keeps its existing search. |
