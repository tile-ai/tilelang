# Multimem Allreduce Notes

This example implements intranode NVSwitch allreduce with TileLang multimem
primitives. The main target is a Hopper-class 8 GPU node with multicast VA
support.

## Strategies

- `one_shot`: every rank computes the full output with
  `T.multimem_ld_reduce`. This is simple, but every rank performs all reduce
  loads.
- `two_shot`: each rank computes only its output shard with
  `T.multimem_ld_reduce`, then broadcasts that shard with `T.multimem_st`.
  This matches the high-throughput ThunderKittens pattern on Hopper.
- `two_shot_tma_copy`: experimental path that stores the shard through ordinary
  Hopper TMA store to the multicast VA. It is correct on H100, but currently
  slightly slower than packed `multimem.st.bf16x2`.
## Development Lessons

Start by matching the reference shape and data movement, not just the high
level algorithm. The useful reference for this operator is ThunderKittens:
first reduce a shard with `multimem.ld_reduce`, then broadcast it. Once the
same amount of data moves through the fabric, benchmark numbers become
comparable.

Always inspect generated CUDA with `kernel.get_kernel_source()`. For this
operator the source made several issues obvious: whether bf16 used packed
`bf16x2` multimem instructions, whether stores were ordinary `multimem.st` or
bulk async multimem TMA, and whether TMA waits used the read or non-read form.

Multimem layout inference matters. Fragment-only flows such as
`ld_reduce -> multimem_st` need the multimem op to contribute the same SIMT
loop layout as a normal copy. Otherwise lowering can silently fall back to
incorrect scalar loads or miss the intended vectorization.

The rewriter must match the real lowered expression shape. Predicated
`ld_reduce` can hide the multicast `BufferLoad` inside nested expressions, so
the lowering pass should search recursively rather than assuming a flat load.
Plain `multimem_st` also has no reduce op, so it must not parse a reduce tag.

For bf16/fp16 on Hopper, packed `x2` instructions are the stable baseline.
The current tuned path uses coalesced width 2 for 16-bit types and 256 threads.
Larger vector forms are not automatically faster; the generated instruction mix
and memory coalescing must be checked.

TMA store to multicast VA is usable but not clearly better here. The
`two_shot_tma_copy` experiment generated ordinary `tl::tma_store` to the mcast
address and passed correctness. On the 32 MiB/rank bf16 shape it measured about
`144.77 us`, compared with about `141.99 us` for `two_shot` with
`multimem.st.bf16x2`.

Use `T.tma_copy` for this Hopper experiment. It lowers to ordinary TMA store to
the multicast VA, which is the path tested by this example.

Benchmark comparisons need consistent timing semantics. The TileLang example
uses `do_bench(..., group=group)`, which reports the cross-rank synchronized
time. Some external examples print per-rank CUDA event times, so compare the
same statistic before drawing performance conclusions.

Correctness should test each strategy separately. `one_shot` validates the full
output buffer. Two-shot variants only overwrite this rank's output shard through
the multicast VA, so tests check the local shard and reset the physical local
buffer between benchmark iterations.
