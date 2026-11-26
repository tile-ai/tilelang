# PTO TopK and Sorting

## Implementation

Read [moe_topk_gate_asc.py](code/moe_topk_gate_asc.py) for complete host dispatch, grouped-expert reduction, padding, and physical routing, and refer to `examples/ascend/example_simdvf_topk_gate.py` plus its test for the compact runnable example. Define the fill value for out-of-bounds lanes and the tie-breaking rule for duplicate values according to the target interface; when tail-block transfers are involved, verify the copy pad value. Do not generalize an example's fixed shape or behavior into a guarantee for all inputs.

Use a hierarchical structure for large sorts: Kernel A produces sorted candidates for each tile, and Kernel B merges candidates by row. When a global radix/histogram is required, use a separate workspace and sequential kernel launches; do not depend on an unvalidated grid barrier. The sortable key transform must be consistent with the dtype, sign bit, and NaN ordering contract.

## Accuracy and Performance

Cover K=1, boundary K, full-length K, tie-breaking for duplicate values, ±0, NaN, ±Inf, index dtype, tail blocks, and stability. Compare complete end-to-end latency, workspace, GM bytes, and Vector/MTE time against the baseline; multi-kernel designs must account for every launch.
