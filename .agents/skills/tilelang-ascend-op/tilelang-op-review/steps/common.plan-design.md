# Review Plan Design

## Purpose

Determine the rules applicable to the current review from the code summary, user-specified scope, and declarations in `references/*.md`, then generate directly executable grouping and wave plans.

This step is responsible for routing and scheduling design. It does not execute rule reviews, generate issue reports, or modify files under review.

## Inputs

- Files formally under review: `{file_input}`
- File types: `{file_types}` (`Python`, `TileLang DSL`, `Markdown`, or mixed)
- Code side: `{code_side}` (`Kernel`, `Host`, mixed, or `N/A`)
- Code-summary path: `{code_summary_path}`
- Review-scope hint: `{scope_hint}` (a category explicitly specified by the user; empty if unspecified)
- Review type: `{review_type}` (`file` or `pr`)
- Review identifier: `{review_id}` (the PR number for PR review; empty for file review)
- PR diff file: `{diff_path}` (used only for PR review; empty for file review)
- Number of subagent slots available in Stage 1: `{available_agent_slots}`
- Skill resource directory: `{skill_base}` (an absolute path passed by the main workflow, used only to read references and invoke scripts)

## Dispatch Requirements

Pass these inputs to the review-planning subagent and require it to execute the "Subagent Execution Guide" in this file in full. After completion, the subagent returns `yaml_dir` and the complete review plan. Generating the final review report is prohibited.

---

## Subagent Execution Guide

### Step 0: Create the YAML Output Directory

Invoke the command corresponding to the review type:

```bash
# File review
python3 "{skill_base}/scripts/workflow.create_review_dir.py" --type file

# PR review
python3 {skill_base}/scripts/workflow.create_review_dir.py --type pr --id {review_id}
```

Capture the single absolute path line from stdout as `yaml_dir`. If the command fails, output is empty, or the path is not an existing directory that was created, stop plan design and return the actual error.

Return `yaml_dir` only to the main workflow so it can start the collector and assemble the report. Do not pass it to item-by-item review subagents.

### Step 1: Read the Summary and Reference Indexes

Read `{code_summary_path}` in full and extract:

- Files formally under review and context-only files.
- File type and code side of every file formally under review.
- Entrypoints, functions, branches, and call relationships.
- Parameter sources and defenses.
- TileLang API, Buffer, partitioning, and pipeline indexes.
- Markdown section, command, path, link, and reference indexes.
- Unconfirmed information.

Rules may be scheduled only for files formally under review. Context files may serve as defensive or call evidence but must not be added to the review scope automatically.

Enumerate `{skill_base}/references/*.md`. During planning, read only these sections from each reference:

- `<applicability>...</applicability>`;
- `<review_load>...</review_load>`;
- Content from `## Quick Index` through the next peer-level `##` heading.

Do not read complete rule bodies during planning. Subsequent item-by-item review subagents read specific rules according to their assignments.

Use this canonical identity for every rule:

```text
{reference filename without .md}/{original rule ID}
```

For example:

```text
tilelang-red-line/2
tilelang-perf/PERF-3
python-secure/4.1
doc-style/D3
```

Do not rewrite the original rule ID or use only a display alias in place of the canonical identity.

### Step 2: Declarative Reference Matching

Match each file formally under review independently; do not rely only on the aggregate code side for the complete input.

#### 2.1 Language Matching

Normalize and compare these file-type names:

- Ordinary Python: `Python`.
- Python containing TileLang DSL: matches both `Python` and `TileLang DSL`.
- Markdown: `Markdown`.

Language matching succeeds when the languages declared by the reference include the current file type.

#### 2.2 Code-Side Matching

- `Side: All`: matches both Kernel and Host.
- `Side: Kernel`: matches only Kernel files.
- `Side: Host`: matches only Host files.
- `Side: N/A`: matches only files without a code side, such as Markdown.
- Match mixed code separately according to the per-file side in the summary.

#### 2.3 Default and Domain Rules

- `Enabled by default: false`: skip unless the user explicitly selects it through `scope_hint`.
- `Domain: false`: enable after language and side match.
- `Domain: true`: the formal review content or code summary must also match real semantics or executable-code characteristics declared under `Triggers:`.

A domain trigger cannot be based only on comments, strings, unrelated variable names, or context files. When a reference declares `Excluded scenarios:`, apply it; if an exclusion matches, skip the reference and record the reason.

### Step 3: Apply `scope_hint`

`scope_hint` may come only from an explicit user request:

- Empty: continue processing every matched reference.
- Specifies a category, domain, rule ID, or reference: retain only the corresponding content.
- Specifies content with no match: return an empty plan with an explicit reason; do not silently fall back to a full review.

### Step 4: Rule-Level Code-Side Filtering

Continue filtering according to the Quick Index section or rule marker:

- `[Applicable: All]`: retain for both Kernel and Host files.
- `[Applicable: Kernel]`: retain only for Kernel files.
- `[Applicable: Host]`: retain only for Host files.
- If no rule-level marker exists, inherit the reference's global side.
- Markdown rules have side `N/A`.

Every retained rule must record the list of formal review files to which it applies. Write rules excluded by side filtering to the skipped list, explaining the actual side and required side.

### Step 5: Content Screening

Following the content-screening approach used by AscendC file review, retain only rules whose corresponding operations, data objects, or document structures occur in the formal review content.

#### File Review

Use the complete function, API, parameter, branch, Buffer, and Markdown indexes in the code summary. When evidence is insufficient, perform only limited supplementary `rg` checks against files formally under review.

#### PR Review

Screen only against added or modified diff content and functions, APIs, parameters, and constants explicitly marked as changed in the code summary. Patterns occurring only in unmodified context or deleted code cannot trigger rules. When evidence is insufficient, perform supplementary checks only in the diff and files formally under review.

#### Screening Principles

- Division-by-zero rules: retain when division, modulo, a normalization denominator, or an equivalent operation exists.
- Indexing and addressing rules: retain when Tensor/Buffer indexing, offsets, or dynamic boundaries exist.
- Data-movement rules: retain when `T.copy` or equivalent movement exists.
- Pipeline, Buffer, and performance rules: retain when corresponding APIs or execution structures exist.
- Python file, command, exception, and resource rules: retain when corresponding Host operations exist.
- MoE, TopK, and quantization rules: require both a domain trigger and the rule's specific scenario.
- After Markdown input matches `doc-style.md`, retain all D1-D4 rules; do not determine that they pass prematurely from the summary.

When a corresponding pattern is confirmed absent, write the rule to the skipped list and record "no corresponding operation found within the formal review scope," along with the summary location or search basis used for confirmation.

When evidence is insufficient, do not immediately skip a high-severity rule; first perform supplementary confirmation against the files formally under review. Upstream validation, defenses, or fixed values already recorded in the summary are not reasons to skip. Pass this information to the item-by-item review subagent as evidence and dispatch the rule normally.

### Step 6: Merge and Control Capacity

Read `core/review-load-balance.md` in full and apply its rules:

1. Ordinary rules with the same applicable files, code side, and root cause may be merged.
2. When merging across references, use the minimum capacity among all source references as the group capacity.
3. The capacity limit applies to the total number of rules in a group; do not accumulate separate capacity for each reference.
4. Red-line rules, rules spanning file types, rules spanning code sides, or rules with different evidence scopes must not be merged.
5. Split groups into independent groups when capacity is exceeded.

Every group must contain:

- A unique `group_id`.
- Reference filename.
- Canonical rule identities, original IDs, and titles.
- File types and code sides.
- List of files formally under review.
- Group capacity and actual rule count.
- Priority and its sorting basis.

The same canonical rule may appear only once within the same formal file scope. After grouping, check for duplicates and omissions; fix any issues before generating waves.

### Step 7: Generate Waves

Sort by priority according to `core/review-load-balance.md`. Each wave has this capacity:

```text
min(number of unscheduled groups, available_agent_slots, 6)
```

`available_agent_slots` must be a positive integer. If it is invalid or missing, stop and ask the main workflow to provide it; do not assume that 6 slots exist.

One new subagent subsequently executes each group. The plan must not assign the same subagent to multiple groups or reuse a subagent across waves.

Highest priority determines order only; it does not permit exceeding per-group capacity, actual slots, or the concurrency limit of 6.

For file review, plan waves record static capacity and stable ordering. During execution, the main workflow flattens them into a queue. The formatting task occupies one slot in the first batch; afterward, tasks are dispatched continuously as subagents finish. Wave boundaries do not require file review to wait for every task in a batch to complete. Future PR review continues to use the original wave-by-wave process.

### Step 8: Output the Plan

Return the following structure to the main workflow:

```text
yaml_dir: /tmp/{directory_name}
Review type: {file/pr}
Review identifier: {review_id or empty}
Review scope: {full or scope_hint}
File types: {Python/TileLang DSL/Markdown/mixed}
Code side: {Kernel/Host/mixed/N/A}

Matched rule files:
- {reference}: {match reason}, retained {N} rules

Skipped rule files:
- {reference}: {reason for language/side/domain trigger/excluded-scenario mismatch}

Skipped rules:
- {canonical rule identity}: {side or content-screening reason}

Expected rule set:
- {group_id}: {canonical rule identity} -> {list of files formally under review}

Wave 1:
- {group_id} | priority:{priority} | capacity:{capacity} | rules:{canonical identity + title} | files:{list of files formally under review}

Wave 2 (if any):
- ...

Total: {G} groups across {W} waves; retained {C} rules; skipped {S} rules.
```

If no rules need to run, still return `yaml_dir`, matched and skipped reasons, an empty expected rule set, and "Total: 0 groups across 0 waves."

## Constraints

- Do not read or modify unrelated code outside the files formally under review.
- Do not add context files to the formal review scope.
- Do not read complete reference bodies; read only declarations and Quick Indexes.
- Use only fields declared in the "Inputs" section of this file; do not depend on undeclared upstream artifacts.
- Do not generate YAML rule results or the final review report.
- Every skip decision must have an explicit, verifiable reason.
- The returned expected rule set must exactly match the groups in every wave for subsequent completeness checks.
