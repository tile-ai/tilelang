# Reducers

TileLang's first-class reducer represents a deferred reduction epoch. Logical
iterations contribute values; the compiler chooses the physical partial layout
and performs the required cross-thread collective exactly once at finalize.
This keeps the mathematical reduction independent of loop-layout replication.

## Which Reduction API to Use

| Need | API |
| --- | --- |
| Reduce one register value across a warp | `T.warp_reduce_sum/max/min/...` |
| Reduce one fragment axis immediately | `T.reduce_sum/max/min/...` |
| Accumulate across one or more parallel loops or serial tiles | First-class reducer epoch |

The epoch API is:

```python
acc = T.alloc_reducer(shape, dtype, op="sum")
T.reducer_init(acc, init=None)
T.reducer_update(acc[indices], value)       # inside T.Parallel
T.finalize_reducer(acc, dst, batch=1, annotations=None)
```

`dst` must be a separate ordinary fragment with the same logical shape and
dtype as `acc`.

## Basic Example

```python
@T.prim_func
def row_sum(
    A: T.Tensor((M, K), T.float32),
    B: T.Tensor((M,), T.float32),
):
    with T.Kernel(1, threads=128):
        values = T.alloc_fragment((M, K), T.float32)
        result = T.alloc_fragment((M,), T.float32)
        acc = T.alloc_reducer((M,), T.float32, op="sum")

        T.copy(A, values)
        T.reducer_init(acc)
        for i, k in T.Parallel(M, K):
            T.reducer_update(acc[i], values[i, k])
        T.finalize_reducer(acc, result)
        T.copy(result, B)
```

The logical result is `B[i] = sum(A[i, :])`. If layout inference replicates a
logical `(i, k)` iteration over several physical threads, that replication does
not multiply its contribution.

## Epoch Contract

A reducer allocation has one static lifecycle:

```text
alloc -> init -> zero or more updates -> finalize -> read dst
```

The compiler enforces these rules:

- `T.reducer_init` appears exactly once and precedes every update.
- `T.reducer_update` appears inside `T.Parallel`; its first argument is written
  directly as `acc[indices]`.
- `T.finalize_reducer` appears once in a control-flow scope compatible with the
  init and writes an independent destination.
- The reducer handle cannot be read, assigned, copied, cleared, filled,
  aliased, or passed through a pointer. It is not an ordinary buffer.
- The handle is dead after finalize. Read the destination fragment instead.

A complete epoch may be inside a thread-uniform serial loop or conditional. In
that case the same static epoch reopens on each dynamic execution. Do not put a
collective epoch behind a thread-divergent condition, and do not put finalize
in an extra loop that does not also contain init.

Multiple update sites are allowed between init and finalize. This is useful for
accumulating across serial tiles while paying for one final collective.

## Operations and Identities

`op` accepts the following values:

| Operation | Identity used when `init` is omitted | Dtype |
| --- | --- | --- |
| `"sum"` | zero | numeric |
| `"max"` | `-inf` for floats; minimum for signed ints; zero for unsigned ints | numeric |
| `"min"` | `+inf` for floats; maximum integer value | numeric |
| `"bitand"` | all bits set | integer or bool |
| `"bitor"` | zero | integer or bool |
| `"bitxor"` | zero | integer or bool |

Python numeric seeds are converted to the reducer dtype. A `PrimExpr` seed must
already have the matching dtype.

## Seeds Are Logical Values

`T.reducer_init(acc, seed)` captures `seed` at the init site and combines it
exactly once into every logical output:

```python
T.reducer_init(acc, running_sum[0])
```

It is not equivalent to filling every physical partial with `seed`; doing that
would multiply the seed by the participant or replication count. Later changes
to a buffer referenced by the seed expression do not change the captured
starting value.

## Layout and Replication

The compiler infers a `PartialFragment` for reducer storage. Its replica axis
distinguishes:

- **combine lanes**, which hold different partials and are combined; and
- **copy groups**, which compute equal results and are not combined together.

This is intentionally different from a normal `Fragment`, whose replicas are
all equal copies. See [Fragment Layouts and Replication](fragment_layout.md).

Most kernels should let the compiler choose the plan. Experts can pin a
`T.PartialFragment(..., replicate=R, combine=C)` with
`T.annotate_layout({acc: layout})`; the annotation must be compatible with
every update site's parallel mapping. `combine` must evenly divide
`replicate`.

`finalize_reducer(..., batch=N)` batches `N` adjacent collective values through
one lowering group. It is a performance knob and does not change logical
results; leave it at `1` unless generated code or measurement shows a benefit.

## Pipelines and Warp Specialization

An epoch may span a `T.Pipelined` loop, or a complete epoch may execute inside
each pipeline iteration. Pipeline scheduling still has to preserve the
init/update/finalize dependency order.

A reducer epoch must not be split across independent warp-specialized roles.
Keep all three epoch operations in one participant domain. If automatic warp
specialization would split such a loop, disable it for that kernel and inspect
the lowered program before attempting a manual schedule.

## Deprecated Reducer Form

`T.alloc_reducer(..., replication="all" | "none")` selects the legacy v1
fragment reducer. Its in-place `T.finalize_reducer(acc)` form is deprecated.
New code should omit `replication`, use `T.reducer_update`, and finalize into a
separate destination.

## Common Mistakes

```python
# Wrong: ordinary read-modify-write bypasses reducer semantics.
acc[0] += value

# Wrong: the handle does not contain a finalized value.
out[0] = acc[0]

# Wrong: update multiplicity outside T.Parallel is ambiguous.
T.reducer_update(acc[0], value)

# Right:
for i in T.Parallel(N):
    T.reducer_update(acc[0], values[i])
T.finalize_reducer(acc, result)
out[0] = result[0]
```
