# Multi-Kernel Chain Timing

Use this procedure for operators whose single public invocation launches multiple kernels.

1. Record the actual branch, kernel names, invocation counts, and source version for each case.
2. Verify how the timer aggregates identically named kernels and AIC/AIV execution to avoid omissions or double counting.
3. For per-kernel collection, `launch-count` is the invocation count of that kernel; for full-chain collection, it is the total invocation count of all matching kernels.
4. Full-chain acceptance must invoke the real public entry point and retain all required conversion, zero-padding, and temporary-workspace overhead.
5. After modifying the source, reconfirm the entry point, import path, and kernel names; do not reuse an old embedded implementation.
6. Use per-kernel collection only for attribution; do not mix it with end-to-end results from the public entry point under the same measurement methodology.
7. The final report must cover all cases, warm-up and repetition counts, cache policy, anomalous runs, and independent retest results.

If timing omissions are found, discard the invalid baseline and rerun the measurements. Do not manufacture an improvement by modifying the shared timer.
