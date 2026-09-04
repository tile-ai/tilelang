# Warp Specialization

Warp specialization assigns different warps in one CTA to different roles,
such as loading, matrix multiplication, and epilogue work. The roles make
progress independently and exchange ownership of versioned buffers through
barriers.

Use the following levels in order:

1. Write an eligible `T.Pipelined` loop and let CUDA lowering specialize it.
2. Use `WSSchedule` when role placement or a multi-pipeline dataflow must be
   explicit.
3. Use `T.ws(...)` only for fully manual kernels whose barriers and lifetimes
   are managed by hand.

## Automatic Producer/Consumer Specialization

On a CUDA target with TMA support, an eligible loop such as this can be split
automatically:

```python
for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[block_m * BM, k * BK], A_shared)
    T.copy(B[k * BK, block_n * BN], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

The pass recognizes TMA-capable global-to-shared producers, creates buffer
versions, reserves a producer warp group, remaps the original threads into the
consumer partition, and inserts full/back-pressure mbarriers. It also preserves
the source-level synchronous contract of `T.copy`.

Automatic specialization currently requires:

- a CUDA target that supports TMA (Hopper or newer);
- a `T.Pipelined` loop with `num_stages >= 1`;
- at least one global-to-shared copy that can use TMA; and
- staging-buffer layouts that TMA can encode (linear or supported 32/64/128-B
  swizzles).

Keep the producer/consumer loop structurally simple and inspect the lowered IR
when eligibility matters. The transform is deliberately conservative and may
leave an unsupported pattern unspecialized.

Disable automatic specialization for a kernel with:

```python
@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def kernel(...):
    ...
```

This is useful for debugging, for a manually synchronized kernel, or when a
collective such as a reducer epoch must remain in one participant domain.

## Declarative `WSSchedule`

A schedule describes roles, protected pipelines, and the operation order in
each loop scope:

```python
T.WSRole(name, *, warps_lo, warps_hi, max_nreg=0)
T.WSPipeline(name, buffers, depth)
T.WSSync.producer_acquire(pipeline, stage=0)
T.WSSync.producer_commit(pipeline, stage=0)
T.WSSync.consumer_wait(pipeline, stage=0)
T.WSSync.consumer_release(pipeline, stage=0)
T.WSScope(id, bodies)
T.WSSchedule(num_warps, roles, pipelines, scopes)
T.annotate_ws_schedule(schedule)
```

Operations and scope loops need stable IDs. Most tile operations accept an
`annotations` dictionary; `T.ws_op(id)` wraps statements that do not:

```python
A_shared = T.alloc_shared((BM, BK), T.float16)
A_fragment = T.alloc_fragment((BM, BK), T.float16)

T.annotate_ws_schedule(
    T.WSSchedule(
        num_warps=8,
        roles=[
            T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
            T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
        ],
        pipelines=[
            T.WSPipeline("input", [A_shared], depth=2),
        ],
        scopes=[
            T.WSScope(
                "k_loop",
                {
                    "Producer": [
                        T.WSSync.producer_acquire("input", stage=0),
                        "load",
                        T.WSSync.producer_commit("input", stage=0),
                    ],
                    "Consumer": [
                        T.WSSync.consumer_wait("input", stage=1),
                        "consume",
                        T.WSSync.consumer_release("input", stage=1),
                    ],
                },
            ),
            T.WSScope(
                T.WSScope.ROOT,
                {"Producer": ["k_loop"], "Consumer": ["k_loop"]},
            ),
        ],
    )
)

for k in T.Pipelined(tiles, num_stages=2, annotations={T.WSID: "k_loop"}):
    T.copy(A[k * BM, 0], A_shared, annotations={T.WSID: "load"})
    T.copy(A_shared, A_fragment, annotations={T.WSID: "consume"})
```

Plain strings in a scope body are shorthand for `T.WSOpRef(id)`.

### Roles

`WSRole` owns the half-open warp range `[warps_lo, warps_hi)`. Role names must
be unique and ranges must not overlap. `WSSchedule.num_warps` overrides the
kernel's thread extent and must be a positive multiple of four, because
`setmaxnreg` operates on complete warpgroups.

`max_nreg=0` leaves register allocation unchanged. A nonzero budget is applied
per warpgroup; roles that share a warpgroup must request a compatible budget.
Leave gaps in the role ranges only intentionally.

### Pipelines

`WSPipeline(name, buffers, depth)` creates a **full** and an **empty** mbarrier
for each of `depth` versions and multi-versions every listed buffer:

```text
producer acquire(empty) -> write version -> commit(full)
consumer wait(full)      -> read version  -> release(empty)
```

Several buffers may share a pipeline when they have the same lifetime. A
buffer may belong to only one pipeline. The stage on a sync entry is a logical
iteration offset; a common depth-`N` pipeline uses producer stage `0` and
consumer stage `N - 1`.

An operation that accesses a protected buffer must be inside the corresponding
open acquire/commit or wait/release span. One role cannot be both producer and
consumer of the same pipeline.

### Scopes and Operation IDs

`WSScope` schedules one serial loop, a `T.ws_op`-wrapped `while` loop, or the
implicit root. Every schedulable statement inside a declared scope must have an
ID and be placed by the schedule; allocations and the schedule annotation
itself are not scheduled operations. Each child scope is entered once per
participating role, and an operation is placed at most once per role.

An operation may be repeated in several roles only if it touches no
pipeline-protected buffer. Scope guards and `while` conditions must be uniform
across roles and must not read protected pipeline buffers.

Use `annotations={T.WSID: "name"}` on tile ops and supported loops. Use
`with T.ws_op("name"):` for scalar bindings, scheduler calls, or a group of
statements that should be one opaque scheduled operation.

The complete runnable reference is
[`examples/aws/gemm.py`](https://github.com/tile-ai/tilelang/blob/main/examples/aws/gemm.py).

## Manual `T.ws`

```python
with T.Kernel(..., threads=256):
    with T.ws(1):
        # warpgroup 1: threads 128..255
        produce()
    with T.ws(0):
        # warpgroup 0: threads 0..127
        consume()
```

`T.ws(*warp_group_idx)` selects one or more fixed 128-thread CUDA warpgroups.
It only creates role branches. It does **not** infer dependencies, version
buffers, choose barrier counts, wait for asynchronous work, or protect shared
memory. The programmer owns every one of those obligations.

Prefer `WSSchedule` for new explicit schedules because it validates role,
scope, buffer, and full/empty-barrier relationships. Reserve `T.ws` for
low-level experiments and existing hand-written protocols.

## Correctness Checklist

- Every role agrees on which logical buffer version an operation accesses.
- A producer cannot overwrite a version until all consumers release it.
- A consumer cannot read a version until its producer commits it and any async
  transaction has completed.
- Barrier arrivals and waits use the same participant set and phase sequence.
- Branches around synchronization are role-uniform.
- A reducer epoch and other collective cannot accidentally straddle roles.
- Register budgets and the larger specialized block size still satisfy target
  occupancy and launch limits.

See [Software Pipelines](software_pipeline.md), [TMA](tma.md), and
[Synchronization and Memory Ordering](synchronization.md) for the underlying
contracts.
