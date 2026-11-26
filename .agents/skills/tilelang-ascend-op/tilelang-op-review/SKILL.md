---
name: tilelang-op-review
description: Review TileLang, Python, and Markdown files. Use for code review, code audits, coding-standard checks, formatting checks, lint checks, or pre-submit validation. Run Ruff and clang-format checks on user-specified files or directories, and review correctness, safety, and verifiable performance against TileLang references. Do not modify source during review; formatting fixes require explicit user approval after the report is complete.
---

# TileLang File Review

## Workflow Routing

Based on the file path, file list, or directory explicitly supplied by the user, read and execute:

```text
workflows/file-review.md
```

Route all of the following requests to this workflow: code review, code auditing, checking for problems, checking coding standards, format checking, lint checking, and pre-submit validation.

Only the file-review workflow is currently integrated. If the user provides only a PR number, remote PR link, or diff without identifiable local files, do not pretend that the source has been obtained or invoke a nonexistent PR workflow. First obtain, or ask the user to provide, the local files to review. Underlying reserved PR fields exist only for a future PR-workflow migration.

If the user has not provided identifiable files or a directory, first ask for the paths to review; do not expand the review scope independently.

## Artifact and Resource Paths

All workflows use the same artifact-directory rules: prefer an output directory explicitly specified by the user; otherwise, use `operators/<review_id>/` under the user's working directory when this skill is invoked. When entered from the `agent/AGENTS.md` workflow, the user's working directory is the directory containing that `AGENTS.md`; for an independent invocation, it is the working directory at invocation time. Each workflow determines its own review identifier, fixes the artifact directory as an absolute path at the start, and passes it to subagents and scripts that need to write files. Do not derive the artifact directory from this skill's installation location, the repository containing the reviewed files, a temporary copy, or a subagent's working directory. The directory-creation script continues to write the collector's intermediate YAML under `/tmp`.

Locate this skill's scripts, steps, and references relative to the directory containing this `SKILL.md`. For subagents that need these resources, the main workflow passes the absolute path of this directory as `skill_base`. `skill_base` is used only to locate bundled skill resources; it does not determine the artifact directory and does not need to be supplied by the user.

## Execution Rules

1. Read `workflows/file-review.md` in full to obtain task, stage-ordering, and context-transfer requirements.
2. Execute strictly in workflow-stage order. Read the step referenced by a stage only when that stage begins; loading files for later stages in advance is prohibited.
3. Use `spawn_agent` when dispatching subtasks. Dispatch file-review tasks continuously by planned priority: as soon as one task completes and its result is confirmed, use the freed slot to dispatch the next group without waiting for other tasks in the same batch. At any time, the total number of running formatting and rule-review tasks must not exceed both the concurrency-slot limit for the current review and 6. Confirm that a slot is available before dispatching.
4. Run the formatting subagent in parallel with the first wave of rule-review subagents, counting all of them toward the same concurrency limit. The formatting check reports results only and must not fix source code.
5. For item-by-item review, use the hypothesis-testing, evidence, and confidence rules in `core/methodology.md`, and read only the rules actually assigned by plan-design.
6. Subagents submit YAML only through the local HTTP collector and must not access `yaml_dir`. The main workflow is responsible for completeness checks, report assembly, and the collector lifecycle. If the environment configures an HTTP or SOCKS proxy, all collector health checks and submission requests must bypass `127.0.0.1,localhost` while preserving proxy settings needed for external access.
7. After the collector starts, whether the review succeeds, fails, or is interrupted, stop the exact recorded PID before leaving the review stage. Do not terminate other instances in bulk by process name.
8. Whether issues are found, all checks pass, or formatting tools fail, generate and validate the final report whenever the collector and reporting steps can run. Do not output a path to a report that does not exist.

## Formatting-Fix Boundary

The file-review workflow only checks files and generates a report; it does not fix formatting automatically. After the report is complete, read and execute `steps/common.format-fix.md` only when the user explicitly agrees to fixes based on the check results:

- Fix only files in which this formatting check explicitly found issues and that the user authorized.
- Do not modify semantic code corresponding to findings from the rule review.
- After fixes, rerun the same formatting checks and update the results accurately.
- If the user does not agree, provide only the report and manual remediation guidance.

## Resource Index

| Resource | Path | Description |
|---|---|---|
| File-review orchestration | `workflows/file-review.md` | Defines code overview, planning, parallel formatting and rule review, and report generation. |
| Execution steps | `steps/` | Complete execution requirements and YAML formats for each stage. |
| Review methodology | `core/methodology.md` | Hypothesis testing, PASS/FAIL/SUSPICIOUS, and confidence rules. |
| Load balancing | `core/review-load-balance.md` | Rule-group capacity, dispatch priority, and concurrency limits. |
| Review rules | `references/*.md` | TileLang red lines, performance, domain, Python safety, and Markdown rules. |
| Tool scripts | `scripts/` | Ruff, clang-format, collector, directory creation, and report assembly. |

Do not copy complete references into this file, and do not bypass the workflow to execute individual steps directly.
