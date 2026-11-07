# Row Kogge-Stone Scan

## Applicability

Use this approach when the scan axis is static and a tile can remain resident in UB, low logical depth is a priority, and the additional combine operations and register traffic at each level are acceptable.

## Algorithm

At level `k`, the distance is `d=2^k`; all elements with `i>=d` execute the following operation in parallel:

```text
next[i] = combine(prev[i - d], prev[i])
```

Each level must read from the previous level's snapshot; newly computed values at the current level must not update in place and contaminate subsequent lanes. Execute `ceil(log2(R))` levels in total. For a partial tail, update only valid elements. `dav310/cum_row_kogge_stone.py::level_pairs` provides the dependency edges for each level.

Handle carry across tiles in the same way as Sklansky, with a unique row owner maintaining fp32 state. If PTO SIMD cannot safely express snapshots between levels or causes register spills, fall back to the streaming correctness baseline.

## Correctness and Performance Gate

Cover `R=1`, every power-of-two boundary, tile±1, carry across multiple tiles, cancellation, mixed magnitudes, and the dtype overflow contract. Compare latency, SIMD instruction count, temporary UB/register usage, and code size against Sklansky.
