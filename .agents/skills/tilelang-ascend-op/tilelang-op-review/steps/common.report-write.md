# Report Writing (File Review + PR Review)

## Inputs

- YAML result directory: `{yaml_dir}`
- Report output path: `{report_output_path}`
- Artifact directory: `{review_output_dir}` (fixed in Stage 0 according to the shared rules in `SKILL.md`; used to validate the report path)
- Review targets: `{file_input}`; for a PR review, this is the changed-file list
- File types: `{file_types}`
- Code side: `{side}`
- Matched rule files: `{matched_references}`
- Review time: `{timestamp}`

## Procedure

### 1. Assemble the Report Body

Run:

```bash
python3 {skill_base}/scripts/workflow.assemble_report.py \
    --dir {yaml_dir} \
    --output {report_output_path}
```

The script reads all available YAML files under `{yaml_dir}` and assembles one aggregate `type: format` result together with ordinary `type: clause` results into a single Markdown report. Every parseable `_dupN` file is included independently in the report and statistics. Unparseable YAML is skipped with a warning. When an ordinary clause's total evidence score disagrees with `confidence_value`, the script recalculates it according to `core/methodology.md` and follows the AscendC behavior of writing the corrected value back in place.

If the command fails or does not generate a nonempty report, stop this step and return the actual error. Do not fabricate a report.

### 2. Complete the Review Overview

The report's `## 1. Review Overview` contains the following placeholders. The main agent replaces each one using the actual input:

| Placeholder | Replacement Source |
|-------------|--------------------|
| `{{CODE_FILE}}` | `file_input`; use the changed-file list for a PR review |
| `{{FILE_TYPES}}` | File types returned by code-summarize |
| `{{SIDE}}` | Code side returned by code-summarize |
| `{{DOC_LIST}}` | Matched rule-file list returned by plan-design |
| `{{TIMESTAMP}}` | Timestamp of the current review |

Use `apply_patch` to replace the placeholders. After replacement, search for `{{` and confirm that no unprocessed placeholder remains in the report.

### 3. Report Path

For both file reviews and PR reviews, verify that `report_output_path` is an absolute path under the `review_output_dir` fixed in Stage 0, then pass that path to the assembler's `--output`. Do not rederive the artifact directory at this stage from the current shell directory, a temporary review repository, or the input files. Follow the AscendC report-naming convention:

- File review: `{review_output_dir}/{source_file}_review_summary.md`
- PR review: `{review_output_dir}/{review_id}_review_summary.md`

This step does not define an additional rule for deriving `{source_file}` from multi-file or directory input.

### 4. Check for an Oversized Report

After completing the header metadata, check the report's line count:

```bash
wc -l {report_output_path}
```

- No more than 5,000 lines: the report is complete.
- More than 5,000 lines: read and execute `steps/common.report-filter.md`. Following the AscendC method, condense formatting details, remove non-severe clause findings according to severity, and update statistics. The report is final only after filtering is complete.

## Report Contents

`workflow.assemble_report.py` consistently generates:

1. Review overview and check summary;
2. Ruff lint, Python formatting, and C/C++ formatting results;
3. Format-check tools, unchecked files, and tool-execution errors;
4. PASS, FAIL, and SUSPICIOUS statistics for ordinary clauses;
5. Details of HIGH-, MED-, and LOW-confidence findings;
6. Remarks outside the PR scope when `out_of_range` results exist;
7. Issue-statistics summary and next actions.

## Constraints

- The report body may be generated only by the assembler from YAML results. The main agent only completes header metadata and triggers oversized-report filtering.
- Do not rereview source code, rerun format checks, or create or modify issue conclusions in this step.
- Do not generate design, S1-S7, D8, style, API preliminary-research, or line-number-verification sections.
- Formatting issues may be fixed only after all checks complete and the user explicitly consents. This step must not modify files under review.
- Preserve the `out_of_range` presentation and PR report path required by the PR workflow.
- Follow the AscendC temporary-directory lifecycle; do not delete `{yaml_dir}` after report generation.
