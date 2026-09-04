# Fragment Layouts and Replication

A TileLang layout separates the logical shape written in the program from its
physical representation. In normal kernels, layout inference chooses this
mapping. Explicit layouts are for cases where an instruction contract, a known
vectorization pattern, or a carefully tuned data distribution must be pinned.

## API Summary

```python
T.Layout(shape, forward_fn)

T.Fragment(
    shape,
    forward_fn=None,
    forward_thread_fn=None,
    replicate=1,
    forward_index_fn=None,
)

T.PartialFragment(
    shape,
    forward_fn=None,
    forward_thread_fn=None,
    replicate=1,
    forward_index_fn=None,
    combine=None,
)

T.annotate_layout({buffer: layout, ...})
T.Parallel(*extents, loop_layout=fragment)
T.copy(src, dst, loop_layout=fragment)
```

`T.Layout` maps logical coordinates to physical memory coordinates. A
`T.Fragment` additionally maps each logical coordinate to a physical thread and
one or more per-thread local indices:

```text
logical indices (+ optional replica) -> (thread, local index)
```

`T.PartialFragment` uses the same mapping algebra, but is reserved for the
uncombined physical state of a first-class reducer.

## `Layout`

Construct a layout with a logical shape and a forward mapping:

```python
layout = T.Layout((M, N), lambda i, j: (j, i))
```

The callable receives one index per logical dimension and returns one physical
index or a sequence of physical indices. Important methods are:

| API | Meaning |
| --- | --- |
| `layout(*indices)` / `map_forward_index(indices)` | Apply the mapping |
| `get_input_shape()` | Logical domain shape |
| `get_output_shape()` | Physical index-space shape |
| `get_linearized_forward_index()` | Row-major physical offset expression |
| `repeat(dim, factor)` | Tile one logical dimension |
| `expand(leading_shape)` | Prepend pass-through dimensions |
| `inverse()` | Construct the inverse mapping when one exists |
| `reshape(shape, rescale_num=1, rescale_den=1)` | Reshape/recast the logical domain |

The output shape is derived from the ranges of the forward expressions; it is
not a second shape supplied by the caller. For a `Fragment`,
`get_thread_size()` is derived in the same way from the thread expression.

Shared-memory swizzles are ordinary `Layout` objects. Prefer the supplied
constructors such as `tilelang.layout.make_swizzled_layout(buffer)` over a
hand-written swizzle, because TMA and matrix instructions accept only specific
encodings.

## `Fragment`

The most direct constructor supplies one function returning `(thread, local)`:

```python
def mapping(i, j):
    linear = i * 32 + j
    thread = (linear // 4) % 128
    local = (linear // (128 * 4)) * 4 + linear % 4
    return thread, local

loop_layout = T.Fragment((128, 32), forward_fn=mapping)

for i, j in T.Parallel(128, 32, loop_layout=loop_layout):
    B[i, j] = A[i, j]
```

The example assigns four consecutive elements at a time to a thread. The
physical local index is an index into that thread's fragment/register storage;
it is not a shared-memory address.

The split form is equivalent when thread and local mappings are easier to
write separately:

```python
fragment = T.Fragment(
    (M, N),
    forward_thread_fn=lambda i, j: thread_expr,
    forward_index_fn=lambda i, j: local_expr,
)
```

Use a fragment in either of these places:

- `T.annotate_layout({frag_buffer: fragment})` pins the physical distribution
  of a `local.fragment` buffer.
- `T.Parallel(..., loop_layout=fragment)` pins how a logical parallel loop is
  partitioned over threads.
- `T.copy(..., loop_layout=fragment)` pins the loop layout of a normal SIMT
  copy. It is incompatible with TMA, LDSM/STSM, and TMEM copy paths.

For a nested `T.Parallel(i_extent, j_extent, ...)`, the fragment input rank must
equal the number of loop axes. The annotation belongs to the outermost loop;
`T.Parallel` handles that placement. The fragment's thread range must also fit
the kernel or warp-specialized role that executes it.

## Replication Means Equal Copies

`replicate=N` adds a physical replica coordinate without changing the logical
shape. When `N > 1`, `forward_fn` (or `forward_thread_fn`) receives an extra
`rep` argument:

```python
def replicated_mapping(i, j, rep):
    linear = i * 32 + j
    thread = (linear // 4) % 64 + rep * 64
    local = (linear // (64 * 4)) * 4 + linear % 4
    return thread, local

layout = T.Fragment((128, 32), forward_fn=replicated_mapping, replicate=2)
```

The constructor callback receives `rep` after the logical indices. One
low-level inspection API has a different canonical order:
`get_forward_vars()` returns `(rep, *logical_indices)` for a replicated
fragment. Account for that order when manually substituting these variables;
normal mapping callbacks do not need to.

For fragment storage, both replicas represent the **same logical value**.
Replication can make a small value available to more threads or avoid
thread-dependent register indexing.

When the same fragment is used as a `T.Parallel` loop layout, however, every
replica executes the loop body. Do not assume that an arbitrary shared/global
store, atomic, or opaque call is automatically reduced to one owner. Keep such
effects replica-invariant and safe to repeat, select a canonical owner with an
explicit thread predicate, or use an unreplicated loop layout. Reducer updates
are a special case: their lowering carries an execution-multiplicity marker so
one logical contribution is counted once even when the loop is replicated.

Fragment-to-fragment operations must agree with the destination fragment's
ownership and replication. An incompatible explicit loop/buffer pair is
rejected as a layout inference conflict.

Do not use ordinary `Fragment` replication to represent partial sums. Reducer
partials are deliberately not equal and use `PartialFragment` semantics.

For a value that every thread should hold, use the provided helper:

```python
fragment = T.alloc_fragment((N,), T.float32)
layout = tilelang.layout.make_fully_replicated_layout_fragment(fragment, threads)
T.annotate_layout({fragment: layout})
```

All physical replicas hold the same logical values when their producers obey
the replication contract. When publishing to shared or global memory, still
audit the selected copy/loop mapping: an explicitly replicated loop may issue
redundant same-value stores unless the operation defines unique-owner
semantics.

## `PartialFragment` and Reducers

For a `PartialFragment`, the replication coordinate has two parts:

```text
rep % combine  -> distinct addend lanes, combined at finalize
rep // combine -> equal-value copy groups, never combined with each other
```

`combine` defaults to `replicate` and must divide it evenly. For example,
`replicate=8, combine=4` describes two equal copy groups, each containing four
partials that must be reduced.

Layout inference creates these layouts for `local.reducer` buffers. Pinning one
with `T.annotate_layout` is an expert tuning mechanism: it constrains the
collective plan and compilation fails if an update site's `T.Parallel` mapping
cannot realize it. See [Reducers](reducer.md).

## Transforming Fragments

`fragment.repeat(repeats, repeat_on_thread=False, lower_dim_first=True)` tiles
the logical fragment. `repeat_on_thread` controls whether repeats expand the
thread mapping instead of the per-thread local mapping.

`fragment.replicate(n)` keeps the logical domain and adds equal physical
copies. `fragment.condense_rep_var()` folds the explicit replica coordinate
into the ordinary mapping when a downstream operation no longer needs it as a
separate axis.

These transformations preserve mapping algebra, but they do not make an
otherwise invalid instruction layout legal. The consuming copy/MMA/TMA
operation still checks its own shape, ownership, and encoding requirements.

## Recommendations

- Start with inferred layouts. Pin only the smallest set of buffers or loops
  needed to express the intended mapping.
- Keep the logical shape equal to the program's loop/buffer shape. Padding and
  tails belong in the physical mapping and guards, not in fictitious logical
  elements.
- Make vector lanes contiguous in the local index when coalescing is the goal.
- Treat `replicate` as an equality promise. If replicas can differ, use a
  reducer or explicit communication.
- Keep non-idempotent side effects, especially atomics, out of replicated loop
  bodies unless you explicitly elect one owner.
- Prefer TileLang's swizzle helpers for shared memory, especially when TMA is
  possible.
- Use the [layout visualization tool](../tools/layout_visualization.md) and
  inspect lowered code when tuning an explicit layout.
