# Plans and Validation

## Plan Admission

Every plan must meet these requirements before implementation:

1. At least one bottleneck finding from the current run supports its hypothesis.
2. The relevant TileLang APIs and invocation structure can be located in the current repository or the actually imported TileLang source.
3. When reusing a bundled reference from the Skill named `tilelang-performance-best-practices`, record the Skill name, internal Python file's relative path, and maturity. When a plan relies only on the current repository or actual TileLang source, record the corresponding source paths and direct validation evidence; prior registration in the template status table is not required.
4. Clearly identify covered cases, mutually exclusive branches, accuracy risks, and rollback conditions.
5. Provide reproducible lowering, accuracy, and performance validation methods.

Mark a plan `DESIGN_ONLY` when it does not meet item 2; it cannot be passed to direct implementation. When reusing a bundled reference without satisfying item 3, do not treat that reference as a directly reusable template. However, if the current repository or actual TileLang source separately provides complete evidence for the API, lowering, accuracy, and applicability, the plan may be admitted independently on that direct evidence.

## Report Template

### 1. Environment and Baseline

- Current repository path/commit, actually imported TileLang path/version, PTOAS version, and device.
- Compilation, test, and profiling commands and key environment variables.
- Dispatch, accuracy status, latency, and bottleneck evidence for every case.

### 2. Admitted-Plan Overview

| Priority | Plan | Core Hypothesis | Status | Covered Cases | Expected Observation |
|---:|---|---|---|---|---|

Admit at most three plans, using only these statuses:

- `IMPLEMENTABLE`: complete evidence exists for APIs, reference implementation, and lowering.
- `EXPERIMENT`: executable, but benefit or applicability still requires an isolated experiment.

`DESIGN_ONLY` is not an admitted plan. List it separately under "Non-Admitted Candidates"; it does not count toward the three-plan limit and is not passed to implementation.

### 3. Each Plan

- Bottleneck evidence and a falsifiable hypothesis.
- TileLang files, functions, dispatch branches, and parameters to modify.
- Current value -> candidate value, with Tiling/buffer calculations.
- Reference Python path, template maturity, and key TileLang structures.
- Per-case coverage table.
- Compilation, accuracy, and performance acceptance criteria, plus rollback conditions.

Output at most three admitted plans. If several optimizations must occur together to form a valid dataflow, they may be combined into one plan with their dependencies explained. Otherwise, prefer validating them separately for clear attribution.

### 4. Non-Admitted Candidates (Optional)

| Candidate | Status | Reason for Non-Admission | Missing Evidence | Follow-Up Validation |
|---|---|---|---|---|

List only `DESIGN_ONLY` here. This section preserves potentially valuable directions that cannot currently be implemented directly in TileLang/PTO; they do not participate in plan ranking, implementation, or performance comparison.

### 5. Complete Case Coverage

| Case | Baseline Group | Dispatch | Plan | Expected Change | Validation Status |
|---|---|---|---|---|---|

Every case must appear. Mark a case with no applicable plan as "No changes for now" and explain why.

## Validation Order

Follow the target repository's testing standards:

1. PTO lowering/compilation: run the smallest relevant case first and retain complete error information.
2. Accuracy: use the repository's existing reference and thresholds, covering every branch, boundary, and tail in the plan.
3. Performance: keep device, input, warmup, repeat, concurrency, and timing layer identical for the baseline and candidate; record kernel latency per case.
4. Expanded regression: after the minimal set passes, run the targeted accuracy tests, complete relevant test suite, and pre-merge extended tests required by the current repository.

Common PTO test entry:

```bash
TILELANG_DEFAULT_TARGET=pto pytest <test_file> -x
```

If the target workflow specifies another command, follow the target repository's current standard.

## Performance Conclusions

Per-case speedup is `baseline_latency / candidate_latency`. A summary value must state its algorithm, such as geometric mean, and separately list the worst regressing case. Call a result a performance optimization only after every accuracy gate passes and same-methodology measurements are complete; otherwise, report only experimental results or design recommendations.
