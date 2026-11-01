# Format Fixes

Shared by all review modes. A subsequent workflow may invoke this step only after the format check, required code review, and report assembly have all completed. It must not run prematurely immediately after the format check.

## Inputs

- `file_input`: The original review targets used for the format check.
- `format_check_result`: The complete output from `common.format-check.md`.
- `report_path`: The path to the generated review report; empty when no report exists.
- `completed_checks`: Every check step and status completed by the workflow.

Locate the Skill scripts from the directory containing this `SKILL.md`. In the commands below, `{skill_base}` is the absolute path to that directory; do not derive it from the repository being fixed or the current shell directory. Continue passing file arguments according to the targets used in the format check.

## Invocation Preconditions

Continue only when all of the following conditions are met:

1. The workflow explicitly places this step after all checks and report assembly.
2. `format_check_result` contains lint or formatting issues that can be handled automatically.
3. `file_input` is identical to the format-check stage input.

When any condition is unmet, do not ask and do not apply fixes; return the reason for skipping.

## Procedure

### 1. Show a Summary and Ask the User

First show the files with issues, their languages, and issue counts, then explicitly ask:

```text
Apply code-formatting fixes?
- Yes: Automatically fix the Python/C++ files within the scope confirmed by the user
- No: Do not modify files
```

Wait for an explicit user response. The user may narrow authorization to specified languages or files. Do not treat an ambiguous response as consent, and do not extend authorization for formatting fixes to semantic issues.

### 2. Apply Fixes Within the User-Authorized Scope

Python:

```bash
bash "{skill_base}/scripts/fix-python.sh" path/to/a.py path/to/b.pyi
```

The script runs:

```bash
ruff check --fix <files>
ruff format <files>
```

C/C++:

```bash
bash "{skill_base}/scripts/fix-cpp.sh" path/to/a.cc path/to/a.h
```

The script runs:

```bash
clang-format -i --style=file <files>
```

Pass to the scripts only files that were actually checked during the format-check stage, have issues, and are authorized by the user. When the user provides pasted content, do not modify repository files; return only the suggested corrected code.

### 3. Recheck

Rerun the same read-only checks from `common.format-check.md` on the fixed files. If a fix command fails or the recheck still finds issues, record the result accurately; do not claim that the fix succeeded.

### 4. Update the Report

When `report_path` exists, append `## Format Fix Results` to the end of the report, including:

- The fix scope authorized by the user.
- Each file's pre-fix issues and differences.
- Commands actually executed.
- Post-fix recheck results.
- Items not fixed or whose fixes failed, with reasons.

When `report_path` is empty, return only a structured fix result for a later reporting step to write into the final report. Do not fabricate a report path.

## Output

Return `format_fix_result`, containing at least:

- Whether the user authorized fixes and the authorized scope.
- Files modified, not modified, and unsuccessfully fixed.
- Fix commands actually executed.
- Recheck results.
- Whether the report was updated and its actual path.

## Constraints

- Do not run a fix script without explicit user consent.
- Fix only Ruff lint/formatting and clang-format issues. Do not automatically modify semantic issues found by the required code review.
- Do not process files outside the format-check scope.
- A recheck is mandatory after fixes; do not determine success from the fix command's exit code alone.
