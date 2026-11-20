# Code Review Methodology

This document is for per-rule review subagents. Side classification, rule routing, grouped dispatch, and report generation are handled by `steps/` and `workflows/`; this document defines only the evidence-analysis and decision methodology for an individual rule.

The formal review scope, file types, and code-side classifications are determined by the upstream code summary and review plan. Related files may be used only to verify call chains, data flow, or defenses; this document does not automatically add them to the formal review scope.

## Fundamental Semantic Distinctions

When reviewing Python and the TileLang DSL, first distinguish among:

- Ordinary Python values;
- Compile-time values known during kernel construction;
- Runtime dynamic values such as `T.dynamic`;
- Tensors and buffers, including their indices, offsets, and memory scopes;
- Host-side input validation and defenses inside the kernel.

Do not infer the code side from the filename alone, and do not treat a value as validated merely because it “comes from a Python caller” or “looks constant.”

For Markdown, the formal review targets are the original document text, structure, commands, paths, links, and factual references. Apply the standard per-rule decision process in this document.

## Hypothesis-Testing Process

Perform the following steps in order for each rule.

### Step 1: Identify Candidate Locations

Using the current rule's applicability scope, identify the relevant functions, statement blocks, data flows, or documentation sections across all formal files specified by the plan. If no corresponding pattern exists, assign `PASS`; do not extend the rule to unrelated issues.

### Step 2: Establish the Hypotheses

- Null hypothesis H0: The candidate location complies with the current rule and has no reportable issue;
- Alternative hypothesis H1: The candidate location presents the risk described by the current rule;
- Confidence starts at `0%`.

### Step 3: Collect Positive Evidence

Positive evidence supports H1. Record only evidence that has actually been verified and is reproducible by another reviewer.

| Evidence type | Score | Condition for use |
|---|---:|---|
| Direct specification violation | +40% | The code or documentation directly contradicts a mandatory condition of the current rule |
| Missing contextual defense | +20% | The current scope and necessary call sites have been checked, but the defense required by the rule was not found |
| Reachable data flow | +20% | Inputs, assignments, or the call chain prove that the risky value can reach the problematic location |
| Reproducible validation | +30% | A unit test, compiler diagnostic, minimal reproduction, command execution, or performance data collected with the same methodology confirms the issue |
| Authoritative target constraint | +20% | The currently installed TileLang/PTO source, lowering, target constraint, or active repository contract explicitly supports the conclusion |

Do not split the same fact into differently named items and score it more than once. For example, a single compilation failure cannot be counted both as “reproducible validation” and as another synonymous piece of evidence.

### Step 4: Collect Negative Evidence

Negative evidence uses the AscendC scores and represents existing defenses or limitations of the current review scope.

| Evidence type | Score | Condition for use |
|---|---:|---|
| Defense present | -20% | The current scope contains an explicit check or protective override matching the risky variable |
| Upstream validation | -15% | A call site performs effective validation, and the assignment chain proves that it protects the same variable and path |
| Out of scope | -50% | PR review only: The candidate issue is outside the scope of the current diff |

Negative scores are deliberately smaller to prevent a subagent from stopping its analysis after finding one ambiguous check. Every item of negative evidence must include the file path, exact line number, and corresponding source text; without a reproducible citation, it scores `0%`.

When verifying a defense, confirm all of the following:

- The checked object is the same as the risky variable, or there is a provably equivalent assignment chain;
- The check occurs before the risky operation and covers the actual execution path;
- The check condition covers the entire risky value domain relevant to the current rule;
- Compile-time constants, hardware constraints, and caller guarantees are supported by actual code or an authoritative implementation.

If only some values or branches have been validated, record the corresponding negative evidence, but do not conclude that the risk has been fully eliminated.

### Step 5: Eliminate Invalid Candidate Issues

If any of the following conditions holds, the candidate issue is invalid and must immediately receive `PASS`, without continuing to accumulate scores:

- Complete defenses cover every risky value and execution path addressed by the rule;
- Context proves that the risky path is unreachable;
- The actual API, data type, or execution semantics prove that the current rule does not apply;
- The candidate pattern does not correspond to the current rule's issue description, decision method, or exclusion rules.

When negative evidence totals `-50%`, the result must not be `FAIL`. If concrete, unexcluded risk evidence remains, use the final score to determine whether the result is `SUSPICIOUS`.

### Step 6: Calculate Confidence and Assign a Decision

```text
confidence = clamp(sum of positive evidence scores + sum of negative evidence scores, 0, 100)
```

| Final condition | status | confidence | Handling |
|---|---|---|---|
| 80%–100% | `FAIL` | `HIGH` | Clear, reproducible evidence of an issue exists |
| 70%–79% | `SUSPICIOUS` | `MED` | Strong indications exist, but manual confirmation is still required |
| Below 70%, but concrete and unexcluded risk evidence exists | `SUSPICIOUS` | `LOW` | Evidence is insufficient to establish a definite violation |
| No concrete risk evidence exists, or the candidate issue has been eliminated | `PASS` | Omit | Do not report issue evidence or confidence |

A `PASS` result must not include `confidence` or `evidence`. A `FAIL/SUSPICIOUS` result must include all positive and negative evidence and the final `confidence_value`.

Severity and evidence confidence are different concepts. A red-line rule or high severity affects review priority and issue impact, but cannot replace factual evidence or automatically increase confidence.

## Evidence-Analysis Requirements

- Use the code summary to locate entry points, definitions, callers, and potential defenses, then verify the actual content by searching the source. Do not treat conclusions from the summary itself as final evidence.
- When encountering a function call, member variable, or cross-file transfer, continue along the shortest call chain necessary for the current risk.
- Related files may establish upstream validation or calling constraints, but an issue result's `code_snippet.file_path` must point to the actual issue location in a formal review file.
- When claiming that a value is validated on the host side, locate the specific condition, assertion, or parameter constraint and prove that it covers the same value used on the kernel side.
- When claiming that a value is a compile-time constant, locate the constant definition, build parameter, or code that proves it is known at compile time.
- Never fabricate missing defenses, constraints, or API evidence, and do not score assumptions such as “this is usually the case.”

## Verifying TileLang APIs and Target Constraints

Only when a decision under the current rule depends on specific API semantics should you consult the currently installed TileLang/PTO source, lowering implementation, or valid repository examples as needed. Relevant topics include:

- Supported memory scopes, alignment, tail-tile handling, and synchronization semantics for data transfers;
- `T.Pipelined`, buffer versions, and dependencies;
- Dynamic shapes, indices, offsets, and type conversions;
- Whether the current backend supports a specific parameter combination or execution structure.

Limit verification to the specific API involved in the suspicious code. Do not scan every API, produce an API research report, or infer version-dependent behavior solely from model memory.

Compilation legality, transfer constraints, on-chip capacity, and hardware behavior may serve as decisive evidence for `FAIL/HIGH` only when the installed source, lowering, target-environment compiler, or device provides reproducible support. If the necessary environment is unavailable, accurately record the evidence limitation; do not classify the absence of a toolchain, permission, or device as a code issue.

## Evidence Threshold for Performance Issues

A performance issue may be assigned `FAIL/HIGH` only when at least one of the following conditions holds:

- A benchmark or profile collected on the same target device with the same inputs, precision, warm-up, and statistical methodology proves a regression;
- Source code and execution semantics statically prove the addition of unnecessary GM accesses, synchronization, loops, transfers, or deterministic serial dependencies.

When the only evidence is that “another implementation might be faster,” comparable data is unavailable, or target constraints are unclear, assign at most `SUSPICIOUS`. Do not present an optimization candidate as a confirmed performance defect.

## Additional Rules for PR Reviews

In a PR review, treat whether an issue belongs to the diff separately from whether the issue itself is valid:

- Attribution to the diff may serve as scope evidence, but cannot by itself prove that the code is defective;
- For issues outside the diff, apply AscendC's “out of scope `-50%`” score and do not assign `FAIL` for the current PR;
- The complete source may be used to verify call chains and defenses, but unmodified context must not automatically be expanded into the formal review scope.

File reviews do not use “PR attribution” or “out of scope” evidence.

## Markdown Rules

Markdown uses the same status and evidence structure. Direct text, parse results, actual paths, anchors, command output, and the current implementation may all serve as evidence:

- Reproducible unclosed fences, invalid repository paths, or incorrect commands may constitute clear issue evidence;
- Terminology differences count as issues only when they create ambiguity about the subject or requirement level;
- If an external link is temporarily inaccessible or a version-specific fact cannot be verified, record only the evidence limitation and do not independently declare a definite error;
- Formatting preferences that are unspecified by the repository and do not affect meaning are not issues.

## Rule Boundaries and Result Self-Check

Each rule checks only the issue described by its body, applicability scope, review method, and exclusion rules. Before producing output, confirm all of the following:

1. Every `FAIL/SUSPICIOUS` maps to a specific issue pattern in the currently assigned rule;
2. All positive and negative evidence has a reproducible source, with no duplicate scoring of the same fact;
3. Complete defenses, unreachable paths, and rule exclusions have been applied;
4. `status`, `confidence`, and `confidence_value` conform to the mapping in this document;
5. The issue location uses an accurate one-based line number, and the code or documentation excerpt includes the original problematic text plus enough context to understand the issue;
6. The remediation recommendation addresses only the current issue and does not expand into unrelated refactoring or performance work.

Red-line issues originate from the TileLang reference rule that actually matches the issue; this document does not maintain a duplicate red-line list.
