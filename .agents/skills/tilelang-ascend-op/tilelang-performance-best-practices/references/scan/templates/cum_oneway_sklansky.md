# Sklansky Prefix Network

## Applicability

Use this template when the scan axis is static, a complete tile can remain resident in UB, and PTO SIMD lane shift/gather/broadcast operations can efficiently express the tree combine. Do not directly use this template for scans that propagate argmin/argmax indices.

## Algorithm

Expand a resident tile of length `R` across `ceil(log2(R))` levels. At level `k`, the half-group width is `2^k` and the group width is `2^(k+1)`. Each group uses the last element of its lower half as the anchor and combines it with every target in the upper half:

```text
half  = 1 << k
group = half << 1
anchor = group_start + half - 1
target = [group_start + half, min(group_start + group, R))
```

The combine operation is addition for Cumsum and multiplication for Cumprod; for Cummin/Cummax without indices, it is min/max, respectively. A tail group updates only real targets, and padding does not participate in the result. `dav310/cum_oneway_sklansky.py::level_pairs` defines the exact anchor/target relation at each level.

Across tiles, use the final prefix of the current tile as an fp32 carry, chained into the next tile by the same row owner. Until PTO SIMD microtests pass, use the serial implementation in `scan_base.py` as the accuracy oracle; do not call it a Sklansky kernel.

## Accuracy and Performance Gates

Cover every `2^k-1/2^k/2^k+1` boundary, partial tails, multi-tile carry, positive-negative cancellation, and the overflow and NaN/Inf semantics specified by the interface. For performance, compare fanout, instruction count, register pressure, and generated-code length against Kogge-Stone.
