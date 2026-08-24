# TileLang Loop Semantic Rules

## Contents

1. [Scope and terminology](#scope-and-terminology)
2. [Source representation](#source-representation)
3. [Established rules](#established-rules)
4. [Nested pipelines are backend capabilities](#nested-pipelines-are-backend-capabilities)
5. [Vectorized loops are not required to be leaves](#vectorized-loops-are-not-required-to-be-leaves)
6. [Non-rules](#non-rules)
7. [Rule-design checklist](#rule-design-checklist)

## Scope and terminology

Apply a nesting rule to one lexical path: an ancestor chain from a function
body to one leaf statement. Two sequential sibling loops are on different
paths.

Distinguish:

- **semantic rule:** required for correctness on every supported lowering;
- **implementation contract:** required by current passes but potentially
  relaxable in the future;
- **optimization condition:** determines whether an optimization applies and
  must not be reported as semantic invalidity.

Call a `T.Pipelined` loop **pipeline-requested** when source TIR carries
`num_stages` or manual `tl_pipeline_order`/`tl_pipeline_stage` metadata. After
planning, canonical markers also include `tl_pipelined_num_stages` and
software-pipeline stage/order attrs. Auxiliary `group` or `sync` metadata does
not request lowering by itself. A bare `T.Pipelined(n)` with none of these
markers behaves structurally like a serial loop and is not
pipeline-requested.

## Source representation

| Source construct | Source-TIR identity | Important consequence |
|---|---|---|
| `T.serial`, `T.grid` | serial `For` nest | Treat as ordinary sequential iteration |
| `T.unroll` | `ForKind::kUnrolled` | Compile-time expansion intent; may contain lower-level loops |
| `T.vectorized` | `ForKind::kVectorized` | SIMD intent; current planner selects only innermost loops |
| `T.Parallel` | one or more `ForKind::kParallel` loops | Consecutive loops form one multidimensional parallel region |
| `T.Pipelined` | serial `For` plus annotations | Detect through annotations, not `ForKind` |
| `T.Persistent` | binds, guards, `loop_break`, and serial `For` | Preserve a marker before enforcing persistent-specific structure |
| `while` | `While` | Often acts as a dynamic/persistent scheduler in real kernels |

Always inspect the representation at the intended checker stage. A rule that
looks obvious in Python may no longer be recognizable after frontend expansion.

## Established rules

These contracts are already enforced or covered by repository tests:

### Consecutive parallel dimensions are one region

Allow a strict chain:

```python
for i in T.Parallel(M):
    for j in T.Parallel(N):
        B[i, j] = A[i, j]
```

Reject re-entering `T.Parallel` after executable statements or another loop
has broken the chain:

```python
for i in T.Parallel(M):
    B[i, 0] = 0
    for j in T.Parallel(N):
        B[i, j] = A[i, j]
```

The current implementation is `tilelang/analysis/nested_loop_checker.py`.

### Pipeline may contain parallel; parallel may not contain pipeline

Allow the canonical tile loop:

```python
for k in T.Pipelined(K, num_stages=3):
    for i in T.Parallel(M):
        ...
```

Reject a pipeline-requested loop under a parallel region. Parallel lowering
distributes elementwise work; software-pipeline planning owns a sequential
producer/consumer timeline and cannot be introduced inside that region.

### Tile operators do not belong inside a parallel region

Reject `T.copy`, `T.gemm`, and other calls with `TLOpBuilder` inside
`T.Parallel`. Allow per-element intrinsics such as supported atomics and
reducer updates; do not classify every effectful call as a tile operator.

### Parallel indexing follows storage ownership

- Reject direct use of an enclosing parallel variable as an index into a
  thread-private local buffer.
- Allow parallel-independent local accesses such as replicated scalar reads.
- Use fragment storage when the indexed logical dimension is distributed
  across threads.
- Retain the existing symbolic-range restriction for fragment indexing.

These checks live in `parallel_local_index_checker.py` and
`fragment_loop_checker.py`.

## Nested pipelines are backend capabilities

Do not impose a language-wide limit on the number of pipeline-requested loops
along one lexical path. Hierarchical software pipelines are semantically
meaningful, and known Ascend kernels intentionally pipeline an outer tile loop
and inner GEMM reduction loops at the same time.

Comments in one backend's pipeline planning or multi-versioning passes establish
only that backend's current implementation limit. They do not justify a
backend-independent `PreLowerSemanticCheck`, which runs before the target
pipeline has selected its lowering strategy.

Use `pipeline-requested` classification for inventory and backend dispatch, not
for global rejection. When reviewing nested pipelines:

1. Resolve the selected target and backend pipeline.
2. Determine whether that backend supports hierarchical pipeline planning,
   buffer versioning, barriers, and warp/core specialization.
3. Allow nesting when the backend contract supports it.
4. If unsupported, diagnose it inside that backend after target resolution:
   `Backend <name> does not support nested software pipelines`.
5. Never silently discard one requested schedule if doing so can change async
   or multi-buffer semantics.

Keep the source scanner informational. It should report shapes such as:

```python
for ko in T.Pipelined(K, num_stages=3):
    for ki in T.Pipelined(BK, num_stages=2):
        ...
```

but exit successfully without deciding whether the target can lower them.

Cover this boundary matrix:

- a supporting backend accepts nested `num_stages` and manual stage/order
  schedules;
- an unsupported backend rejects the same source with a target-specific
  diagnostic;
- target-agnostic pre-lower validation accepts both cases;
- bare and pipeline-requested nested loops remain distinguishable for
  analysis;
- sequential sibling pipelines remain independent.

## Vectorized loops are not required to be leaves

Do not introduce a blanket rule that `T.vectorized` may not contain loops.
This is semantically meaningful when the inner loop has lane-invariant bounds:

```python
for i in T.vectorized(M):
    for k in T.serial(K):
        C[i] += A[i, k] * B[k]
```

The serial loop executes for each SIMD lane. `T.unroll` is similarly possible.
If the inner bound depends on `i`, lanes may have different trip counts and the
construct needs predication or scalarization; treat that as a separate rule.

Current implementation caveat: `VectorizePlanner` plans only innermost loops.
An outer `T.vectorized` containing `T.serial` may therefore be lowered as
serial with a warning. That is an optimization limitation, not semantic
invalidity. Test both semantic acceptance and whether vectorization actually
occurs before changing this behavior.

## Non-rules

Do not adopt these statements as general semantic rules:

- “Every nested `T.Pipelined` is invalid” or “one lexical path may contain at
  most one pipeline-requested loop.” Nested pipeline support is a backend
  capability; bare outer pipeline syntax is also serial-like.
- “`T.vectorized` must be an AST leaf.” Sequential inner loops can be valid.
- “`T.Parallel` extent must equal launched thread count.” Loop partitioning
  and replication intentionally support different extents.
- “Explicit loop-layout input shape must equal loop extent.” Guarded tails and
  non-bijective layouts can be intentional.
- “Every local access inside `T.Parallel` is invalid.” Only ownership-breaking
  dependencies are forbidden; replicated or per-iteration scratch accesses can
  be valid.
- “No examples use this form, therefore it is invalid.” Confirm a downstream
  correctness requirement.

## Rule-design checklist

Before approving a new loop rule, answer:

1. Is this semantic invalidity, a current implementation contract, or an
   optimization condition?
2. What exact TIR shape identifies the construct at the checker stage?
3. What is the smallest invalid example?
4. What is the closest valid example?
5. Do siblings differ from nested ancestors?
6. Does a neutral wrapper (`serial`, `unroll`, `if`, `SeqStmt`) change the
   answer?
7. Does generated IR use the same shape and require an exemption?
8. Can the checker prove the violation, or only fail to prove safety?
9. Which pass first relies on the invariant?
10. Does the diagnostic suggest the intended legal rewrite?
