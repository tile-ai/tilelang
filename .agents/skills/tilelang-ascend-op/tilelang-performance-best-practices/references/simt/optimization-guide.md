# TileLang/PTO SIMT Optimization

## Usage

Within a one-dimensional `T.Kernel`, enter `with T.SimtVF(threads=N)`, use `T.alloc_fragment` for local data, and use `T.Parallel` for parallel dimensions. Prefer `T.SimdVF` for regular contiguous elementwise operations. For reductions, scattered indexing, complex control flow, and byte transposes, use a SIMT path validated in the target repository.

Have a Python factory generate a separate `T.prim_func` for each mode instead of branching by mode. Benchmark `threads` among validated candidates such as `64/128/256`. Handle the element tail with `if i<valid` or complete padding; do not assume that the thread count automatically masks out-of-bounds accesses.

## Gates

Cover every mode, thread-count candidate, lane/warp boundary, and tail. Inspect the generated source for branch divergence, vectorization, fragment size, and spills. Replace the SIMD/fallback path only when correctness passes and latency improves. Executable examples are available in the actual TileLang sources at `examples/ascend/example_rmsnorm.py` and `testing/ascend/language/test_tilelang_ascend_reduce.py`; for transpose-layout semantics, also consult `testing/ascend/layout/test_ascend_l0_transpose.py`.
