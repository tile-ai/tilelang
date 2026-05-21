# Software Pipeline Annotations

TileLang can infer common producer/consumer pipelines from
`T.Pipelined(..., num_stages=...)`. For regular GEMM-like kernels, this is the
preferred entry point:

```python
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[by * BM, ko * BK], A_shared)
    T.copy(B[ko * BK, bx * BN], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

For kernels whose loop body has unusual ordering, extra post-processing, or
manual async-copy grouping, you can provide explicit pipeline annotations. This
guide explains how to write those annotations and why scalar `Bind` statements
are not part of the user-visible schedule.

## The User Model

Pipeline annotations describe executable pipeline statements, not every IR node
inside the loop body.

Annotate the statements that do work:

- copies, including global-to-shared and TMA copies
- fills
- GEMM or other tile operations
- reductions
- stores and atomics
- explicit waits, commits, or synchronization statements when they are part of
  the manual pipeline

Do not annotate scalar aliases:

```python
base: T.int32 = ko * BK
offset: T.int32 = base + tx
```

These scalar `Bind` statements are pure value definitions. The compiler places
them automatically at each use.

## Stage and Order

Each scheduled statement has two numbers:

- `stage`: the logical pipeline stage.
- `order`: the order used when emitting scheduled statements.

Lower stages run earlier in the pipeline. A typical copy/compute pipeline has
copy statements in stage 0 and compute statements in stage 1:

```python
for ko in T.Pipelined(
    T.ceildiv(K, BK),
    num_stages=3,
    stage=[0, 0, 1],
    order=[0, 1, 2],
):
    T.copy(A[ko * BK], A_shared)
    T.copy(B[ko * BK], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

The two arrays are aligned to the scheduled statements in source order. In the
example above:

```text
statement 0: copy A  -> stage 0, order 0
statement 1: copy B  -> stage 0, order 1
statement 2: gemm    -> stage 1, order 2
```

The compiler checks buffer dependencies after applying the annotations. If one
scheduled statement produces data for another, the producer must not be placed
after the consumer. In practice:

- If producer and consumer are in the same stage, producer order must be smaller
  than consumer order.
- If they are in different stages, producer stage must be less than or equal to
  consumer stage.

`num_stages` controls the pipeline depth, namely the number of in-flight buffer
versions retained by the rewritten loop. It is separate from the largest
`stage` value, although common kernels use small stage numbers such as 0 and 1.

## Annotating a Reordered Pipeline

The `order` array is useful when the pipeline should issue a future-iteration
producer before a current-iteration consumer.

```python
for ko in T.Pipelined(
    num_tiles,
    num_stages=2,
    stage=[0, 1],
    order=[1, 0],
):
    T.copy(A[ko * BK], A_shared)
    T.gemm(A_shared, B_shared, C_local)
```

This says there are two scheduled statements:

```text
copy -> stage 0, order 1
gemm -> stage 1, order 0
```

During pipeline rewriting, the compiler emits prologue, steady-state body, and
epilogue pieces with the proper logical iteration for each stage. The source
order still determines which annotation entry belongs to which scheduled
statement; `order` controls the emission order after the statements have been
classified.

## Scalar Bind Statements

A scalar `Bind` may appear as a statement in the loop body:

```python
for ko in T.Pipelined(
    num_tiles,
    num_stages=2,
    stage=[0, 1],
    order=[1, 0],
):
    base: T.int32 = ko * BK
    T.copy(A[base + tx], A_shared[tx])
    T.gemm(A_shared, B_shared, C_local)
```

The `stage` and `order` arrays contain two entries, not three. They annotate the
copy and the GEMM. The scalar `base` definition is intentionally omitted.

This rule matters because a scalar `Bind` does not have a unique pipeline
position. It is closer to an SSA value or a local alias than to an executable
operation. If the same scalar is used by statements in different stages, each
consumer may need the value under a different logical pipeline iteration.

For example:

```python
for ko in T.Pipelined(
    num_tiles,
    num_stages=2,
    stage=[0, 1],
    order=[1, 0],
):
    base: T.int32 = ko * BK
    T.copy(A[base + tx], A_shared[tx])
    B[base + tx] = A_shared[tx]
```

After pipeline rewriting, the copy may need `base` for a future producer
iteration while the store needs `base` for the current consumer iteration. A
single user-selected stage/order for `base` would either be out of scope or
would describe the wrong logical iteration for one of the uses.

TileLang therefore treats scalar `Bind` as non-schedulable:

```text
stage/order annotate only scheduled, effectful statements
scalar Bind definitions are replayed at each consumer
```

The replay is use-driven. If a consumer statement references a scalar bound in
the pipeline body, the compiler recreates the needed `Bind` immediately before
that consumer and substitutes the consumer's logical access index.

## Bind Dependencies

Scalar binds can depend on earlier scalar binds:

```python
base: T.int32 = ko * BK
offset: T.int32 = base + tx
T.copy(A[offset], A_shared[tx])
```

The compiler replays dependencies before their users:

```text
base
offset
consumer statement
```

This keeps manual annotations focused on actual pipeline operations while still
preserving the lexical scalar dependencies from the original loop body.

## Legacy Annotation Form

Older code may include scalar `Bind` statements in the annotation arrays:

```python
for ko in T.Pipelined(
    num_tiles,
    num_stages=2,
    stage=[3, 0, 1],
    order=[1, 2, 0],
):
    base: T.int32 = ko * BK
    T.copy(A[base + tx], A_shared[tx])
    B[base + tx] = A_shared[tx]
```

This form is accepted for compatibility. The scalar `Bind` entry is ignored,
and the remaining entries are applied to the scheduled statements:

```text
base Bind -> ignored
copy      -> stage 0, order 2
store     -> stage 1, order 0
```

When a legacy scalar `Bind` annotation is used by multiple scheduled statements,
TileLang may warn that the scalar annotation is ignored and the bind is replayed
at each use. New code should prefer the shorter annotation arrays that omit
scalar binds entirely.

## Manual `T.serial` Annotations

Most user code should use `T.Pipelined`. Lower-level tests or transform-level
code may use the raw loop annotations directly:

```python
for ko in T.serial(
    0,
    num_tiles,
    annotations={
        "software_pipeline_stage": [0, 1],
        "software_pipeline_order": [1, 0],
        "software_pipeline_async_stages": [0],
    },
):
    base: T.int32 = ko * BK
    T.copy(A[base + tx], A_shared[tx])
    B[base + tx] = A_shared[tx]
```

The same rule applies: `software_pipeline_stage` and
`software_pipeline_order` describe only scheduled statements. Scalar `Bind`
statements do not need entries.

## Design Rationale

The pipeline pass separates the loop body into two concepts:

```text
scheduled statements: executable operations controlled by stage/order
scalar bindings: pure local definitions placed by use-def analysis
```

This split keeps the API stable for users and avoids ambiguous schedules. A
real pipeline operation has a clear execution point, can read or write buffers,
and may require async-copy bookkeeping or synchronization. It is appropriate for
the user to assign a stage and order to that operation.

A scalar `Bind` has none of those properties:

- It has no side effect.
- It owns no buffer storage.
- It can be used by multiple scheduled statements.
- Its correct value may depend on the consumer's logical pipeline iteration.

Forcing users to annotate such a statement would require them to choose one
stage/order even when there is no single correct answer. Replaying scalar binds
at each consumer makes the intended semantics explicit in the generated IR and
keeps the user-facing schedule focused on real pipeline work.

## Checklist

When writing manual pipeline annotations:

- Count only scheduled statements when building `stage` and `order`.
- Omit scalar aliases such as `base = ko * BK`.
- Keep each scheduled statement's `order` unique.
- Keep producers before consumers according to stage/order dependency rules.
- Use `software_pipeline_async_producers` and
  `software_pipeline_async_producer_groups` with the same length as the
  scheduled statement list.
- Prefer the bind-free form for new code; rely on legacy bind slots only when
  maintaining old kernels.
