# Synchronization and Memory Ordering

Synchronization APIs are not interchangeable. Before inserting one, identify
the guarantee the consumer actually needs.

| Need | Typical API | What it does not provide by itself |
| --- | --- | --- |
| Warp rendezvous | `T.sync_warp(mask=None)` | CTA-wide rendezvous |
| CTA rendezvous and shared-memory visibility | `T.sync_threads()` | Completion of an async engine |
| Named subset-of-CTA rendezvous | `T.sync_threads(id, count)` | One-sided signaling |
| Named one-sided arrival | `T.named_barrier_arrive(id, count)` | Waiting |
| Async TMA-load completion | `T.mbarrier_wait_parity(...)` | Participation by threads that never wait |
| `cp.async` group completion | `T.ptx_wait_group(n)` | Cross-thread rendezvous |
| TMA-store completion | `T.tma_store_wait(...)` | CTA rendezvous |
| WGMMA completion | `T.warpgroup_wait(...)` / `T.wait_wgmma(...)` | Shared-memory proxy ordering |
| Cluster rendezvous | `T.cluster_sync()` | Grid-wide rendezvous |
| Cooperative-grid rendezvous | `T.sync_grid()` | Ordinary-launch support |
| Generic-to-async shared ordering | `T.fence_proxy_async()` | Rendezvous or async completion |

Compiler-managed `T.copy`, `T.Pipelined`, automatic warp specialization, and
`WSSchedule` pipelines insert their required synchronization. Add manual
operations only when using an explicit async API or implementing a protocol the
compiler cannot infer.

## CTA and Warp Barriers

```python
T.sync_threads(barrier_id=None, arrive_count=None)
T.sync_warp(mask=None)
T.syncthreads_count(predicate)
T.syncthreads_and(predicate)
T.syncthreads_or(predicate)
```

`T.sync_threads()` is a full CTA barrier and makes prior shared-memory writes
visible to threads that leave the barrier. Every non-exited CTA thread must
reach compatible barriers in the same dynamic order. Placing it under a
thread-dependent branch can deadlock.

On CUDA, passing `barrier_id` selects a named barrier; passing `arrive_count`
also sets the number of participating threads. Named barrier IDs are a scarce
hardware resource (normally 0 through 15), and every participant must agree on
the ID, count, and sequence.

`T.sync_warp()` synchronizes the lanes named by its CUDA mask, whose default is
the full 32-lane warp. All named, active lanes must execute it consistently. On
HIP, the mask is ignored and the operation acts as a wavefront compiler
barrier; do not use CUDA sub-warp mask assumptions in portable kernels.

The `syncthreads_*` forms perform the same CTA rendezvous and also return a
count, all, or any reduction of the predicate.

## Split Named Barriers

```python
T.named_barrier_arrive(barrier_id, thread_count)
T.sync_threads(barrier_id, thread_count)
```

On CUDA, `named_barrier_arrive` contributes a one-sided `bar.arrive` and returns
immediately. A corresponding `sync_threads(id, count)` performs `bar.sync` and
waits for the declared participant count. This is useful for a hand-written
producer/consumer protocol, but unlike `WSPipeline` it does not version buffers
or validate lifetimes.

## Mbarriers

TileLang exposes mbarriers on CUDA Hopper-class targets (`sm_90` or newer).

```python
T.alloc_barrier(arrive_count_or_list)
T.alloc_cluster_barrier(arrive_count_or_list)  # CUDA cluster scope

T.mbarrier_expect_tx(barrier, bytes)
T.mbarrier_arrive_expect_tx(barrier, bytes)
T.mbarrier_arrive(barrier, cta_id=None)
T.mbarrier_wait_parity(barrier, parity)

# Exact aliases for the common local forms:
T.barrier_arrive(barrier)
T.barrier_wait(barrier, parity)
```

An allocation accepts one positive arrival count or a list containing one
positive count per barrier. Initialization is emitted by the compiler.

An mbarrier epoch completes only after both obligations reach zero:

```text
expected ordinary arrivals + expected async transaction bytes
                              |
                              v
                       wait for parity
```

`mbarrier_expect_tx` adds an expected byte count without arriving.
`mbarrier_arrive_expect_tx` combines one arrival with the byte expectation.
`mbarrier_arrive` contributes one ordinary arrival. `T.tma_copy` loads issue
their own `expect_tx`, so the surrounding threads normally add only their
ordinary arrivals.

The parity passed to a wait identifies the epoch's starting phase (the phase
being waited through), not the phase value after completion. A freshly
initialized barrier starts with parity `0`; toggle the parity each time the
same barrier slot is reused. For a ring of `depth` barriers:

```python
slot = iteration % depth
parity = (iteration // depth) & 1
T.mbarrier_wait_parity(barriers[slot], parity)
```

Waiting is local to the calling thread; every consumer that reads the produced
data must execute the appropriate wait or be covered by a subsequent valid
rendezvous. Never reuse a barrier slot or its protected buffer while the prior
epoch can still be live.

`T.alloc_cluster_barrier` places the object in cluster-visible shared memory.
`T.mbarrier_arrive(barrier, cta_id=rank)` performs a remote arrival and is only
valid for such a barrier.

## Async Completion

### `cp.async`

`T.async_copy` emits a `cp.async` sequence and commits it, but does not wait.
Call `T.ptx_wait_group(n)` before consuming the destination. If other threads
consume data written by the issuing threads, a warp or CTA barrier is also
required. A wait completes the async copies; it does not by itself rendezvous
those consumers. If that shared destination subsequently feeds an async-proxy
operation such as WGMMA, proxy ordering is a third, separate requirement; the
default `InjectFenceProxy` pass inserts the fence when both operations remain
visible to it.

### TMA

A `T.tma_copy` load completes through its mbarrier. A store is committed by the
operation and completes through `T.tma_store_wait(count, read)`; see
[Tensor Memory Accelerator](tma.md) for the exact load and store protocols.

### Warpgroup MMA

`T.warpgroup_arrive()`, `T.warpgroup_commit_batch()`, and
`T.warpgroup_wait(num_mma)` control WGMMA groups. `T.wait_wgmma(id)` is the
corresponding lower-level wait helper used by TileLang's generated paths.
Prefer `T.gemm` and compiler-managed scheduling unless the complete WGMMA
lifetime is explicit in the kernel.

`T.warpgroup_fence_operand(...)` is a register/compiler fence for accumulator
operands; it prevents the compiler from moving uses across the related WGMMA.
It is not a shared-memory fence or a thread barrier.

## Cluster and Grid Synchronization

```python
T.cluster_arrive_relaxed()
T.cluster_arrive()
T.cluster_wait()
T.cluster_sync()       # arrive + wait
T.sync_grid()
```

Cluster operations are CUDA Hopper+ APIs and require `T.ClusterKernel` with
compatible cluster dimensions. Use `cluster_arrive` plus `cluster_wait` to
split a cluster barrier; use the relaxed arrival only when the weaker memory
ordering is intentional. See [Cluster TMA](cluster_tma.md).

`T.sync_grid()` lowers through CUDA cooperative groups and requires a
cooperative launch. Every block in the cooperative grid must reach it in a
compatible order. It is much more expensive and restrictive than CTA or
cluster synchronization.

`T.sync_global()` is a legacy exported name whose CUDA lowering is no longer
supported. New code must use `T.sync_grid()` when a cooperative grid barrier is
actually required.

Programmatic dependent launch APIs `T.pdl_trigger()` and `T.pdl_sync()` order
dependent kernel launches on supported CUDA targets. They do not synchronize
threads inside the current CTA and are not substitutes for memory barriers.

## Proxy Fences

```python
T.fence_proxy_async()
```

On Hopper and newer NVIDIA GPUs, generic shared-memory accesses and operations
such as TMA/WGMMA run through distinct memory proxies. A
`fence.proxy.async.shared::cta` orders prior generic-proxy shared-memory writes
before subsequent async-proxy operations can observe that memory.

For example, low-level code that constructs data or a descriptor through
ordinary shared-memory stores and then launches WGMMA needs a proxy fence on
that transition:

```text
generic shared writes -> CTA/role synchronization as needed
                      -> fence.proxy.async -> WGMMA/TMA async-proxy use
```

The fence has three important non-properties:

- It does not wait for TMA, `cp.async`, or WGMMA completion.
- It does not make threads rendezvous.
- It does not replace an mbarrier, wait-group operation, or `T.sync_threads`.

The default CUDA lowering pipeline runs `InjectFenceProxy`, which tracks
shared-memory stores and recognized async-proxy operations through sequences,
branches, and loops. It inserts and hoists fences conservatively. Existing
explicit `T.fence_proxy_async()` calls reset the tracked proxy state, so the
pass does not add an immediately redundant fence.

Manual fences are mainly for custom or opaque low-level operations whose proxy
effects are not visible to the pass. An opaque call carrying a writable shared
pointer through `T.access_ptr(..., "w" | "rw")` is recognized conservatively;
this is how ordinary `cp.async` and STMatrix writes are tracked. LDSM is only a
shared-memory read and does not create this write dependency. An unknown call
without visible shared-memory effects is not tracked. When adding a new async
intrinsic, its compiler lowering must teach `InjectFenceProxy` about the proxy
transition.

For pass details, see [InjectFenceProxy Pass](../compiler_internals/inject_fence_proxy.md).

## Safety Checklist

- Define the exact participants before choosing a barrier.
- Keep collective barriers out of thread-divergent control flow.
- Match arrival counts, async byte counts, stage slots, and parities.
- Wait for async completion before the first read or before reusing its source
  or destination storage.
- Add a rendezvous as well when completion by one thread must be observed by
  other threads.
- Treat proxy ordering as a separate requirement from both completion and
  rendezvous.
