# Persistent Multidimensional Task Mapping

## Goals and Applicability

Use this approach for task domains that naturally decompose into multiple tile axes and where division, modulo, or address preparation for a flattened task ID becomes measurable overhead. Apply it only when profiling or generated code supports this conclusion.

## TileLang/PTO Implementation

- When supported by the current version, use `T.Persistent([tiles_0, tiles_1, ...], ...)` to obtain each axis's tile ID directly, avoiding flattening followed by coordinate reconstruction with `//` and `%` in the hot loop.
- Keep `T.Kernel` as a one-dimensional block grid; the multidimensional list describes only the task domain. Select axis order, `group_size`, and `num_stages` according to contiguous memory access, load balance, and pipeline requirements.
- Retain flattened mapping when multidimensional lowering is unavailable, cannot express the required traversal order, or does not improve the generated code.

## Validation Gate

Verify that the Cartesian product of tasks has no omissions or duplicates. Cover 1, tile±1, tail tiles, dynamic ranges, and empty tasks on every axis. Compare integer division/modulo operations and address instructions in the generated code, per-core task tail imbalance, and latency measured with the same methodology.
