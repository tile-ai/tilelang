# AllReduce Barrier Lowering

TileLang's CUDA AllReduce implementation uses a barrier between cross-warp
butterfly steps. The selected barrier policy depends on the target:

- **SM80 and later:** `tl::NamedBarrier<count>`, lowered to PTX
  `bar.sync phase, count`.
- **Earlier CUDA architectures:** `tl::SyncThreadsBarrier`, lowered to
  `__syncthreads()`.
- **Single-warp reductions:** shuffle instructions only; no CTA barrier is
  emitted.

`bar.sync` named barriers are distinct from Hopper's shared-memory
`mbarrier` instructions. The AllReduce path only requires the former, so it is
enabled starting with Ampere rather than Hopper.

The pre-SM80 fallback is a whole-CTA synchronization path. Partial-CTA scalar
AllReduce therefore requires SM80 or later.

## Partial-CTA Participation

A scalar reduction may be executed by only part of a CTA. Emitting a whole-CTA
barrier inside that guarded region would deadlock, so lowering derives the
exact participating thread range from the reduce fragment layout.

The forward thread expression is converted to an absolute CTA thread ID by
adding the fragment's `ThreadRange.min`. The compiler then:

1. binds every logical layout dimension and the replication dimension;
2. computes the minimum and maximum thread IDs with the arithmetic analyzer;
3. uses Z3 model enumeration to count distinct thread IDs in the image; and
4. requires the count to equal `max - min + 1`.

The final equality proves that the image is one contiguous range instead of a
sparse set that merely has the same bounds. The range must also lie inside the
CTA and be warp aligned. Its minimum becomes the AllReduce workspace offset,
and its exact size becomes the named-barrier arrival count.

This analysis enumerates at most the CTA thread count, not the Cartesian
product of layout coordinates. Large logical dimensions that do not affect the
thread expression therefore do not increase the search space.
