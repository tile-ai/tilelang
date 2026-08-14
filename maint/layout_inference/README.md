# Layout inference verification harness

Constructed IR cases with reviewed expected layouts, for validating the
free-mode layout search — in particular the selection policy behind
`tl.layout_cost_model` (off = legacy register count, on = the io-aware
model from `src/transform/layout_inference/layout_cost_model.cc`).

Why this exists:

- The cost model's score decides which layout attempt wins; any change to
  the model (or a future fast-path/slow-path split inside it) can silently
  flip the winner. Golden layouts pin the current answers so a flip shows
  up as a reviewable diff, not a runtime perf mystery.
- Cases where the two policies **disagree** are the calibration corpus:
  each disagreement is a concrete claim ("the io-aware pick is faster on
  hardware") that can be benchmarked.

## Usage

```bash
python run.py                # verify all cases against expected/
python run.py --case NAME    # substring filter
python run.py --show         # also print the inferred layouts
python run.py --record       # rewrite goldens from current behavior
```

Recording is not approval: after `--record`, read the diff under
`expected/` and convince yourself every changed layout is intended before
committing. Structural invariants (each case's optional `check`) are
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
