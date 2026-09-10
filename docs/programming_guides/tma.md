# Tensor Memory Accelerator (TMA)

NVIDIA TMA moves multidimensional tiles between global and shared memory with a
small number of issuing threads. TileLang exposes both a high-level synchronous
copy contract and low-level split-phase operations.

## Choose the Copy Contract First

| API | Source-level completion | Fallback |
| --- | --- | --- |
| `T.copy(src, dst)` | Synchronous; compiler owns waits and synchronization | May select another legal copy path |
| `T.copy(src, dst, prefer_instruction="tma")` | Synchronous | No: compilation fails if TMA cannot be selected |
| `T.tma_copy(src, dst, ...)` | Split-phase; caller owns completion | No: explicit TMA never silently becomes a normal copy |

Use `T.copy` unless overlap itself is part of the algorithm. An implementation
may use TMA internally without changing `T.copy`'s source-level contract.
Use `disable_tma=True` to prevent automatic TMA selection for one `T.copy`.
The global pass config `tl.disable_tma_lower=True` provides deprecated
compatibility behavior; prefer the per-copy option in new code. Combining
either form with `prefer_instruction="tma"` is an error.

## Explicit API

```python
T.tma_copy(
    src,
    dst,
    *,
    barrier=None,
    cluster_mask=None,
    leader_scope_threads=None,
    eviction_policy=None,
    annotations=None,
)
```

The direction determines the protocol.

### Global to Shared Load

For a load, `barrier` is required. `T.tma_copy` elects one leader, adds the
transfer byte count to that mbarrier, and issues the TMA load. It deliberately
does not contribute the ordinary thread arrivals and does not wait.

A one-shot block-wide protocol is:

```python
shared = T.alloc_shared((BM, BK), T.float16)
ready = T.alloc_barrier(threads)

# Reached by every participating thread. One elected leader issues TMA and
# expect_tx; every thread contributes one ordinary arrival.
T.tma_copy(A[block_m * BM, 0], shared, barrier=ready)
T.mbarrier_arrive(ready)
T.mbarrier_wait_parity(ready, 0)

# shared is ready for this thread after its wait succeeds.
consume(shared)
```

The barrier's arrival count is the number of threads that call
`T.mbarrier_arrive`, not the number of bytes. Transaction bytes are accounted
for separately by `T.tma_copy`. Several loads may share one epoch/barrier; issue
them all before the common arrival and wait.

`leader_scope_threads` changes the election domain and must be a positive
multiple of 32. The default is the current thread extent. For example, a value
of `32` elects one issuer per warp; use a distinct destination and barrier for
each such scope unless duplicate loads are intentional.

### Shared to Global Store

For a store, no mbarrier is used. `T.tma_copy` issues the store and commits its
bulk async group (`T.tma_store_arrive`) but does not wait:

```python
T.tma_copy(result_shared, C[block_m * BM, block_n * BN])

# Independent work can overlap here.

T.tma_store_wait(0)               # source shared-memory reads are complete
# or
T.tma_store_wait(0, read=False)   # full store completion is required
```

`count` is the maximum number of committed groups allowed to remain
outstanding. The default `read=True` emits `cp.async.bulk.wait_group.read` and
is sufficient before reusing or overwriting the shared-memory source. Use
`read=False` when subsequent work requires destination writes themselves to be
complete and visible.

Do not add `T.tma_store_arrive()` after `T.tma_copy`: the explicit copy already
emits it. The standalone arrive API is for lower-level raw TMA operations such
as `T.tma_scatter4`.

## Barrier Rings and Parity

For `depth` reusable slots, allocate one mbarrier per slot:

```python
ready = T.alloc_barrier([threads] * depth)

for k in T.serial(num_tiles):
    slot = k % depth
    phase = (k // depth) & 1
    T.tma_copy(A[k * TILE], shared[slot], barrier=ready[slot])
    T.mbarrier_arrive(ready[slot])
    T.mbarrier_wait_parity(ready[slot], phase)
    consume(shared[slot])
```

The stage selects a physical barrier; parity counts how often that same barrier
has completed. `k % 2` is correct only when one barrier is completed every
iteration. With a depth-`N` ring, use `(k // N) & 1`.

The sketch shows the ready side only. A true producer/consumer pipeline also
needs back-pressure so a producer cannot overwrite a stage still used by a
consumer. Prefer `T.Pipelined` or a `WSSchedule` with `WSPipeline`, which
construct the full/empty protocol, instead of hand-writing it.

## Multicast and Clusters

For a load, `cluster_mask` is a positive compile-time bitmask of destination CTA
ranks. The lowest-ranked CTA in the mask issues one multicast transaction;
other in-mask CTAs receive it, while CTAs outside the mask issue their own
unicast load. Each CTA uses and waits on its local barrier instance.

The kernel must use `T.ClusterKernel` with compatible cluster dimensions. See
[Cluster TMA](cluster_tma.md) for multicast and SM-to-SM copy examples.

## Gather4 and Scatter4

Blackwell (`sm_100a`) exposes four-row gather/scatter operations:

```python
T.tma_gather4(src, dst, col, rows, *, barrier, eviction_policy=None)
T.tma_gather4_bytes(K_box, dtype)
T.tma_scatter4(src, dst, col, rows, *, eviction_policy=None)
```

Both global tensors and shared tiles are rank 2, dtypes must match, the shared
tile shape is `(4, K_box)`, `rows` contains exactly four row indices, and the
global innermost stride is one.

These operations are fully fire-and-forget. The caller supplies leader
election and all synchronization:

```python
mbar = T.alloc_barrier(threads)
if T.shuffle_elect(threads):
    T.mbarrier_expect_tx(mbar, T.tma_gather4_bytes(K_box, dtype))
    T.tma_gather4(A, tile, col, rows, barrier=mbar)
T.mbarrier_arrive(mbar)
T.mbarrier_wait_parity(mbar, 0)

if T.shuffle_elect(threads):
    T.tma_scatter4(tile, B, col, rows)
    T.tma_store_arrive()
T.tma_store_wait(0, read=False)
```

Set a shared layout with `T.annotate_layout`; the `swizzle=` argument on these
APIs is deprecated.

## Shape, Layout, and Alignment Requirements

Descriptor-based TMA is available on CUDA Hopper and newer and supports tensor
ranks from 1 through 5. A legal copy needs:

- a global/shared or shared/global direction;
- compatible source and destination region sizes and dtypes;
- an encodable, bijective shared-memory layout;
- a contiguous innermost global stride;
- an innermost transfer width that is a whole 16-byte multiple; and
- outer global byte strides that are 16-byte aligned and below the hardware
  limit.

The compiler can split a wide legal tile into several TMA boxes. It also
propagates the alignment required by a 32/64/128-B shared-memory swizzle to the
merged shared allocation. Prefer TileLang's layout helpers so this relationship
remains visible to the compiler.

`T.copy` may fall back when these conditions are not met. `T.tma_copy`,
`prefer_instruction="tma"`, multicast, gather4, and scatter4 are explicit
requests and fail with the unsupported reason instead.

## Recommendations

- Start with `T.copy`; force TMA only after checking generated code and
  measuring the kernel.
- Prefer compiler-managed `T.Pipelined` or a declarative `WSSchedule` with
  `WSPipeline` for a multi-stage load/compute protocol.
- Keep issue, arrive, wait, and stage reuse visibly paired in manual code.
- Do not treat an mbarrier wait as a general CTA rendezvous or a proxy fence.
- Use `T.tma_store_wait(read=False)` before a same-kernel consumer that needs
  the global destination to be complete.
- Keep explicit TMA target-specific and provide a separate kernel path when
  supporting pre-Hopper GPUs or non-CUDA backends.
