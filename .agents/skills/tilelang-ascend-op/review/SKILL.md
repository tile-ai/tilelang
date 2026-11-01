---
name: review
description: Review code changes on the current branch against main
license: MIT
compatibility: opencode
metadata:
  audience: maintainers
  workflow: code-review
---

## What I do

- Diff the current branch against `main` (default) or a given commit range
- For each changed file: summarize what changed and flag issues
- Report: correctness bugs, style inconsistencies, path/config errors, missing/broken tests, security concerns
- Output a summary table with per-file severity ratings

## Usage

- `/review` — diff current branch vs `main`
- `/review abc123..def456` — diff a specific commit range
- `/review --by-commit` — review each commit individually instead of the final diff
- `/review --by-commit abc123..def456` — review each commit in a range individually

## When to use me

Use this before merging a branch or after pulling new commits to catch issues early.

## How I work

1. Run `git diff main...HEAD` (or the specified range) to get the full diff
2. Read each changed file to understand context beyond the diff
3. Analyze changes grouped by file, checking for:
   - Logic bugs and off-by-one errors
   - Inconsistencies with existing code patterns (naming, imports, assertion style)
   - Hardcoded paths, stale references, copy-paste artifacts from other repos
   - Test coverage gaps (new code without tests, assertions that don't actually check anything)
   - Duplicated code that should be shared
4. Output per-file findings, then a summary table

## What I do NOT do

- I do not modify any files — review is read-only
- I do not block on style nitpicks if there are no real issues
- I do not review merge commits (I look at the underlying changes instead)

## Key review principles

- Focus on correctness and logic bugs over style nitpicks
- Flag inconsistencies with existing patterns in the codebase
- Check that new code has adequate test coverage
- Verify paths, config references, and imports are correct
- Look for code that was clearly copied from elsewhere and not adapted

## Project-specific things to watch for

- **torch ref / kernel alignment**: `tile_kernels/torch/` reference implementations must match the kernel's computation order (e.g. FMA grouping, reduction order). Mismatched float operation order causes divergence at boundary conditions.
- **Benchmark key determinism**: all values in `benchmark_record(params=...)` must be deterministic across runs. Random or data-dependent values (like `num_tokens` from filtered random data) break baseline matching.
- **Test three-way comparison**: correctness tests should compare kernel output against both the torch ref and the legacy library (`hfai_topk` / `hfai_fp8`). Missing either comparison weakens coverage.
