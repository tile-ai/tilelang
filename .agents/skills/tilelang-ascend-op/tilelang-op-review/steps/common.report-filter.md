# Report Filtering (Condensing an Oversized Report)

## Trigger Condition

After the final report is generated and its header is completed, dispatch one general-purpose subagent to execute this step if `wc -l {report_path}` exceeds 5,000 lines; otherwise, skip it.

## Subagent Task

Pass the final report path `{report_path}`. After reading the report, the subagent condenses it in place in the following order. It does not read or modify `yaml_dir`, nor does it modify the files under review.

### 1. Condense the Format-Check Presentation

- Preserve the number of checked files, total Ruff issue count, count for each error code, number of files with formatting issues, tool versions, skipped files, and execution errors.
- Condense Python lint details to "file + Ruff code + count", removing repeated expansions of example locations, source code, and remediation suggestions.
- Condense Python and C/C++ formatting details to lists of affected files and remove diff previews.
- Formatting details are only collapsed for presentation; the total formatting-issue count does not change.

### 2. Filter Clause Findings

Filter only independent findings within `Issues Found (HIGH Confidence)`, `Needs Attention (MED Confidence)`, and `Suspected (LOW Confidence)`:

1. Do not merge or deduplicate by clause ID or canonical identity. When the same clause is triggered at different locations, each location is an independent finding.
2. Preserve issues involving correctness, memory safety, array or Buffer out-of-bounds access, null pointers, uninitialized values, race conditions, resource leaks, external-input validation, numerical overflow/underflow, division by zero, precision errors, TileLang red lines, and verifiable severe performance regressions.
3. Purely non-severe findings involving naming, missing comments, log wording, line width, blank lines, indentation, and similar matters may be deleted. Assess both MED and LOW by their actual impact; do not delete them automatically because their confidence is lower.
4. Preserve HIGH findings first. Delete one only when its content clearly falls into a non-severe category above and the report remains seriously oversized.
5. Remarks outside the PR scope do not participate in filtering.

### 3. Update Statistics and Validate

After deleting clause findings, synchronously update every affected location:

- Total clause-result count in the review overview;
- Required Review issue count in the check summary;
- PASS, FAIL, and SUSPICIOUS counts and percentages;
- FAIL and SUSPICIOUS counts for each canonical clause.

Collapsing formatting details does not change formatting statistics. Before overwriting the report, confirm that:

- Every preserved finding retains its complete title, location, confidence, issue description, and remediation suggestion;
- Counts in every section match the visible findings after filtering;
- Markdown headings, tables, and code fences remain complete;
- Findings at different locations were not mistakenly deleted by clause ID.

If statistics or structural validation cannot be completed reliably, do not overwrite the original report; return the failure reason directly.

## Output

On success, return only: `Complete: {N} lines before filtering → {M} lines after filtering; removed {K} non-severe clause findings.`

If no findings remain that meet deletion criteria but the report still exceeds 5,000 lines, preserve the remaining content and accurately return the final line count. Do not delete severe issues merely to satisfy the line-count threshold.

## Constraints

- Follow the AscendC handling method: an independent subagent assesses severity and overwrites the report in place.
- Do not save a separate filtered report or add filtering markers or process notes to the report body.
- Do not modify the aggregate format YAML or ordinary clause YAML. Complete original results remain in `{yaml_dir}` and can be used by the assembler to regenerate the unfiltered report.
- This step only condenses presentation and removes non-severe clause findings. It does not apply formatting fixes or source modifications.
