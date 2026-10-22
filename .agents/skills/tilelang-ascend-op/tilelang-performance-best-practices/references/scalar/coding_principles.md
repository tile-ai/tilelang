# TileLang/PTO Hot-Path Coding Principles

1. Use a Python factory to generate mode-specific `T.prim_func` functions, eliminating runtime mode branches.
2. Use local scalars to cache repeated task-offset, tile-ID, and stride computations.
3. Separate the main loop from tail blocks so that the main loop retains static full tiles.
4. Use `T.Unroll` only for fixed small loops; use `T.serial` or `T.Pipelined` for the rest.
5. Shorten the live ranges of fragments, local vectors, and scalars to reduce spills.
6. Place regular contiguous computation in `T.SimdVF`; use `T.SimtVF` implementations validated in the target repository for scattered indexing, complex branches, and reduction.
7. Keep resident buffers single-version and explicitly apply `annotate_buffer_versions` to pipelined tiles.
8. Do not truncate a 64-bit total offset to save instructions; reduce its width only after proving the range.
9. After every structural change, inspect generated source, correctness, and end-to-end latency.
