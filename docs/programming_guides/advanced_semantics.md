# Advanced Semantics

TileLang's high-level operators hide most thread mapping and synchronization,
but performance-sensitive kernels sometimes need to control those decisions.
This section documents the contracts behind the advanced APIs. These contracts
are part of program correctness, not merely optimization hints.

## Choose the Highest Useful Level

| Goal | Preferred API | Drop lower only when |
| --- | --- | --- |
| Move a tile | `T.copy` | You need an explicitly asynchronous lifetime |
| Pipeline a regular loop | `T.Pipelined(..., num_stages=N)` | Inference cannot express the required ordering |
| Distribute elementwise work | `T.Parallel(...)` with inferred layout | A fixed thread/local mapping is required |
| Reduce logical contributions | `T.alloc_reducer` and its epoch operations | A fixed warp-local reduction is sufficient |
| Specialize producer and consumer warps | Automatic warp specialization | You need an explicit role schedule |
| Synchronize shared-memory users | Let the compiler insert synchronization | You are writing a manual async or role pipeline |
| Use TMA | `T.copy`, optionally with `prefer_instruction="tma"` | You need to overlap issue and completion yourself |

The lower-level APIs expose more overlap, but they also transfer ownership of
barrier counts, phases, completion, and buffer reuse to the program.

## A Useful Mental Model

Keep these four concepts separate:

1. **Logical work** says which logical element or reduction contribution is
   evaluated. `T.Parallel` and reducer updates live at this level.
2. **Physical placement** says which thread and local slot hold that work.
   `Layout`, `Fragment`, and replication describe this level.
3. **Completion and rendezvous** say when participants or asynchronous engines
   have finished. Thread barriers, mbarriers, copy-group waits, and WGMMA waits
   have different participant sets.
4. **Memory ordering** says when one access domain may observe another.
   In particular, an NVIDIA proxy fence is not a thread barrier or an async
   completion wait.

Confusing two of these levels is a common source of duplicated reductions,
stale shared-memory reads, and deadlocks.

## Guides

- [Fragment Layouts and Replication](fragment_layout.md) explains logical to
  physical mappings, explicit layout annotations, ownership, and the distinct
  meaning of reducer partial layouts.
- [Reducers](reducer.md) documents the deferred reduction epoch API and its
  lifecycle rules.
- [Software Pipelines](software_pipeline.md) covers inferred and manually
  scheduled pipelines.
- [Warp Specialization](warp_specialization.md) covers automatic,
  schedule-driven, and fully manual warp specialization.
- [Tensor Memory Accelerator](tma.md) covers synchronous selection and explicit
  split-phase TMA operations.
- [Synchronization and Memory Ordering](synchronization.md) covers CTA, warp,
  cluster, grid, mbarrier, async completion, and proxy-fence semantics.
- [Cluster TMA](cluster_tma.md) covers multicast and distributed shared-memory
  copies across a thread-block cluster.

## Target Scope

`Layout`, `Fragment`, reducers, and the logical `T.Pipelined` contract are
target-independent concepts, although available lowering strategies differ by
backend. TMA, warp specialization, mbarriers, and proxy fences in these guides
refer to NVIDIA CUDA. TileLang's TMA and mbarrier APIs, along with proxy fences,
require Hopper-class targets (`sm_90` or newer); gather/scatter TMA and TMEM
features described here require Blackwell-class targets where noted.

When an API names a hardware mechanism explicitly, unsupported hardware is an
error. High-level APIs such as `T.copy` may instead choose a legal fallback.
