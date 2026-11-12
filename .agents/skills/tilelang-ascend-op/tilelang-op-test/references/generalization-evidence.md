# Generalization Evidence

The following entries are historical test records from before this skill was migrated. They illustrate validation methods and common defects only; they do not imply that the current repository contains the same operators or plugins, or that it has passed these tests. Evidence must be rerun and recorded for the current delivery.


This record applies the skill to operators beyond the three initial pilots to demonstrate that additional workflow paths have been exercised. It is not a delivery-acceptance conclusion for the entire repository.

## Round 1: 2026-09-10

### `normalize_weight`: Deterministic Row Tail and Non-Power-of-Two Reduction

The public API and PyTorch reference define an FP32 sum and normalization for each row. The Ascend kernel processes 128 rows per tile, but the existing random generator could not explicitly guarantee a tail containing exactly one row. The new deterministic case uses 129 rows and top-k 7.

This case compares both complete outputs against an independent reference while also checking shape, dtype, device, input immutability, and the sum-to-one invariant, and it passed on Ascend. The existing targeted accuracy cases for top-k 6 and 8 subsequently passed as well.

### `mhc_post`: Gradient Reduction Across a Hidden-Dimension Tail

The existing comprehensive tests compare the forward output and four gradients, but all their hidden sizes are divisible by the current Ascend partitioning scheme. The new case uses hidden 4160: the host selects chunks of 2112 elements, leaving 2048 valid elements and 64 physical padding channels in the final tile.

The first run failed without changing the existing tolerances. The forward output and gradients for `x` and `residual` matched, but the reduction gradients for `post_layer_mix` and `comb_res_mix` contained NaN/Inf or extreme values. Analysis showed that padding in the final UB participated in the reduction. The defect was fixed by initializing the input UB padding to zero only for incomplete tiles. The same nodeid then passed, as did the existing comprehensive regressions for hidden 2560 and 5120.

The repaired run produced a TileLang automatic-scheduling warning: the maximum available `S_MTE2` flag ID was 8, but 9 were allocated. Execution completed successfully, but this warning still requires follow-up in the compiler or scheduler layer and must not be obscured by the correctness pass.

After the fix, all 12 existing forward/backward benchmark cases were run. Initially, 9 remained within the A5 performance gate, while 3 measurements were flagged. Two backward results were respectively 2.72x and 6.23x slower, although another case in the same batch had measured 1.00x only minutes earlier. When those 3 cases were rerun individually, their ratios to baseline were 1.00x, 1.00x, and 1.01x, with no regression, so the earlier slowdown could not be reproduced. All hidden sizes in the existing benchmark (4096, 5120, and 7168) form complete tiles, so the new tail-only zero-fill logic does not execute. Non-divisible production shapes perform extra zero filling in the final tile; its cost has not yet been benchmarked with a production workload.

### `transpose`: Replacing an Empty Case with Real Assertions

The project-required extended test already generated a zero-row input, but `test_transpose` returned immediately after invoking the operator, so the case could pass regardless of the empty output. After removing the early return, the repository's exact comparator checks output shape, dtype, device, and byte-for-byte contents against `x.T.contiguous()`.

The empty-input nodeids for BF16 and E4M3 both passed, as did the corresponding nonempty strided regressions. The empty-input case validates the public-interface boundary without launching a kernel, while the nonempty cases validate the actual Ascend kernel path.

## Round 2: 2026-09-14

### `mhc_pre_big_fuse`: Static Core-Distribution Tail and Benchmark-Only Branch

The Ascend static path used integer division to distribute 512, 1024, and 2048 tokens evenly across the device's 72 vector cores, but did not assign the remainder, leaving the final 8, 16, or 32 output rows unwritten. An existing exact numerical pytest detected the 512 case; the 1024 static case reproduced the same all-zero tail, while the 509 dynamic case passed.

The static schedule now distributes remainder tokens among the cores. The special mapping for 8192 tokens explicitly launches 64 cores because the algorithm is defined as two waves of 4096 tokens, with groups of 64 tokens. Static cases 512, 1024, 2048, and 8192, along with the 509 dynamic control case, all passed correctness testing.

The wave16 branch for 8192×7168 had previously appeared only in benchmarks, which proved runnability and latency but not correctness. A new dedicated exact-reference pytest covers the branch and passed. The benchmark after the fix measured 8.7 microseconds for 512×4096 against an 11.1-microsecond baseline, and 172.2 microseconds for 8192×7168 against a 173.0-microsecond baseline, with no regression in either case.

### `mhc_pre_apply_mix`: Execution-Order-Sensitive Hidden-Tail Reduction

All existing hidden sizes are divisible by the selected Ascend backward chunk size. The hidden 4160 case selects chunks of 1088 elements, leaving 896 valid values and 192 physical padding channels in the final tile. The complete forward result and `x` gradient comparison passed, but mix reduction gradients depended on padding contents.

The new nodeid passed once when run alone before the fix, then failed under a fixed mixed-test order, with 9 of 12 mix-gradient values exceeding tolerance. This is useful and noncontradictory evidence: uninitialized UB may happen to contain harmless values after some allocation histories. Clearing the `o_grad` and `x` UB buffers before copying the final incomplete tile removed this dependency. The nodeid then passed both alone and under the same mixed order, as did existing complete-tile regressions for hidden 5120 and hidden 7168. The two backward benchmarks measured 1.01x and 0.99x relative to baseline, with no regression.

### Backend-Specific Test Collection

A saved representative-case list contained 4 parameterized nodeids generated under CUDA that did not exist under Ascend. Serial collection for the target backend exposed the mismatch: 14 valid Quant nodeids passed, followed by 4 regenerated Ascend nodeids. The workflow now requires concrete parameterized nodeids to come from collection in the same backend/device environment and to be validated without xdist before batch execution.

### Isolation of Source-Embedded Tests and Evidence for Optional Oracles

An old source-embedded cache test changed PyTorch's process-wide default device without restoring it. In a multi-node run, 3 subsequent tests consequently created reference Tensors intended for the CPU on the NPU and failed before producing a valid comparison. The conftest for source-embedded tests now saves and restores the default device before and after every test; the same set of 6 nodes subsequently passed.

The core `test_logsoftmax` first completed comparisons against an independent online algorithm, CPU double precision, and public variants, then called `pytest.skip()` when the optional hai-llm module was missing. This caused pytest to record the entire core test as skipped. The hai-llm comparison now has an independent nodeid: the core test passes, while the optional cross-check remains explicitly recorded as skipped. This preserves both facts without overstating the acceptance conclusion.

## Round 3: Repository-Wide Representative Scan

Static discovery found 134 current correctness test functions and 81 benchmark functions. After selecting one target-backend node for each correctness function and completing the fixes above, all 56 standalone test functions obtained Ascend pass evidence, while 29 of 78 source-embedded test functions obtained pass evidence. Four additional MHC gaps were then given executable tests: the incomplete hidden tile in `pre_apply_mix`, and token-tile tails in `expand`, `head_compute_mix`, and `pre_split_mixes`. The first case exposed the execution-order-sensitive defect described above. After the fix, complete forward/gradient reference comparisons passed for all 4 cases. The repository-wide conclusion remained `NOT_VERIFIED`: 43 source-embedded functions could not be collected because project-specific vLLM APIs were missing, another 2 failed during vLLM-dependent test-data preparation before kernel execution, 3 tests hard-coded CUDA, and 1 optional hai-llm oracle was skipped separately.

Attempting installation with the then-available upstream vLLM 0.23.0 did not resolve the block. Collection still failed for 15 files because that version lacked `vllm.testing` or repository-required `vllm.platform` APIs, and a transitive `regex` dependency error also occurred. The workflow therefore recorded the exact dependency mismatch instead of adding local stubs that might alter the runtime contract.

The execution-evidence classifier was validated with simulated pytest failures, missing dependencies, and Ascend device-startup output. An end-to-end smoke test collected and passed one negative-contract node, producing the conclusion `EXECUTED_PASS_REQUIRES_CREDIBILITY_REVIEW`. This demonstrates that the execution recorder distinguishes execution facts from ST acceptance: successful command execution is not automatically promoted to a contract or coverage conclusion.

## Capabilities Validated by These Applications

- Static discovery can find standalone tests and their reference calls.
- Contract and implementation-path analysis can distinguish genuine tails from arbitrarily chosen unusual numbers.
- Complete assertions can expose defects hidden by otherwise comprehensive tests.
- Failure at a valid boundary remains a failure and does not justify relaxing tolerances.
- Execution evidence preserves failure, repair, and regression stages separately.
- The coverage inventory records remaining gaps explicitly and does not overstate targeted-case passes as certification of an entire operator.

During these applications, the mappings among source, requirements, cases, coverage, and nodeids were reviewed. This file summarizes the results.
