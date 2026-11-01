# Clause-by-Clause Review Message Template (File Review)

The workflow dispatches subagents wave by wave according to the review plan produced by `plan-design`. Use the following message template for each group.

## Message Template

```text
[Completed Upstream]
- Review group: {group_id}
- File type: {Python/TileLang DSL/Markdown/mixed}
- Code side: {Kernel/Host/mixed/N/A}
- Clause filtering: Complete; execute only the clauses assigned below
- Code summary: {code_summary_path}
- YAML submission endpoint: http://127.0.0.1:{collector_port}/submit

Files under formal review: {file_input}

Assigned clauses:
- {reference_file_1}/{original_clause_id_1} {clause_title_1}
- {reference_file_2}/{original_clause_id_2} {clause_title_2}

[Execution Requirements]
1. First load the `tilelang-op-review` Skill, then read that Skill's `core/methodology.md` to understand the decision method for hypothesis testing, evidence, confidence, and red-line issues.
2. Read the code summary to obtain the formal review scope, API index, parameter provenance, cross-file defenses, and cross-file relationships. Related files listed in the summary are only for verifying call chains or defenses; do not add them to the formal review scope.
3. For each assigned clause, use the reference filename and original clause ID to locate the clause title exactly, and read only the content from that clause through the point before the next clause heading. Preserve subsections within the clause, including "Issue Description", "Review Method", "Evidence Requirements", "Exclusion Rules", and examples. Do not read the entire rule file. If the clause defines a specialized review method, execute it.
4. Establish a separate hypothesis, collect positive and negative evidence, and make a separate determination for every clause. Do not substitute an aggregate judgment for the whole group for any individual clause. If the formal review input contains multiple files, every clause must cover all formal files assigned to it by the plan, and issue results must identify the actual file path.
5. Only when the current clause's decision depends on actual TileLang API semantics, inspect on demand the currently installed TileLang/PTO source, lowering implementation, or valid in-repository examples. Check only the specific APIs involved in the suspicious code. Do not scan or conduct preliminary research on every API used by the code in this review, and do not infer version-dependent behavior from memory.
6. After all clause reviews are complete, submit each clause exactly once using the YAML schema below. Submission command:

   curl -sS --noproxy 127.0.0.1,localhost -X POST "http://127.0.0.1:{collector_port}/submit?group={group_id}&rule={reference_file}&clause={original_clause_id}" --data-binary @- <<'YAML_EOF'
   <YAML content>
   YAML_EOF

7. Check the exit status and collector response for every curl invocation. Preserve `--noproxy 127.0.0.1,localhost` when accessing the local collector so that environment HTTP or SOCKS proxies cannot affect it. On HTTP 400, correct the schema or identity error from the response and resubmit. If submission cannot succeed, accurately return the failed clauses and errors; do not claim that all work is complete.
8. Do not directly write to or read from the YAML output directory. Do not generate the final review report or repeat review results in the text response. On completion, return only the submission count and any clauses that were not submitted successfully.

[Pre-Submission Self-Check]
- The complete section for every assigned clause was read and reviewed independently.
- Every assigned clause covers all formal review files designated by the plan.
- Every assigned clause has exactly one successful submission; failed requests corrected after HTTP 400 do not count as successful submissions.
- Every FAIL/SUSPICIOUS maps to a concrete issue pattern in the current clause and has reviewable source or documentation evidence.
- The YAML output directory was not accessed directly, and no aggregate report was generated.
```

## YAML Output Format

Submit every clause through a URL containing the complete `group/rule/clause` identity. The collector supplies:

```yaml
group_name: {group_id}
rule_file: {reference_file_normalized_to_md_filename}
canonical_id: {reference_filename_without_md}/{original_clause_id}
submission_key: {group_id}::{canonical_id}
```

The subagent must not specify these fields in the request body.

### PASS

```yaml
type: clause
clause_id: PERF-1
clause_title: Avoid Per-Element GM Operations in Hot Loops
status: PASS
```

A PASS result must not contain `confidence` or `evidence`.

### FAIL/SUSPICIOUS

```yaml
type: clause
clause_id: PERF-1
clause_title: Avoid Per-Element GM Operations in Hot Loops
status: FAIL              # FAIL or SUSPICIOUS
confidence: HIGH          # Enter HIGH, MED, or LOW according to core/methodology.md
problem_desc: {issue_description}
code_snippet:
  file_path: {formal_review_file_path}
  start_line: {issue_snippet_start_line}
  end_line: {issue_snippet_end_line}
  code: |
    {original_issue_text_and_context_required_to_understand_it}
evidence:
  positive:
    - type: {evidence_type}
      score: {positive_score}
      desc: {reviewable_evidence}
  negative:
    - type: {evidence_type}
      score: {negative_score}
      desc: {reviewable_evidence}
  confidence_value: {cumulative_confidence}
fix_suggestion: {remediation_suggestion_directly_corresponding_to_the_current_issue}
```

Markdown clauses use the same ordinary clause format. In `code_snippet.code`, place the problematic original document text and required context. `file_path`, `start_line`, and `end_line` identify the corresponding location in the Markdown file.

## Field Constraints

The collector parses and validates YAML. Before submission, ensure that:

1. `clause_id` exactly matches the URL's `clause` parameter, preserving original case, numbers, hyphens, and periods.
2. `file_path` contains only a file path, without line numbers, comments, or multiple locations appended.
3. `start_line` and `end_line` use actual one-based line numbers, and the snippet includes the problematic line. Include enough context to understand the issue; no fixed line count is required.
4. `code` contains only original source or Markdown text, without line-number prefixes or file-path comments.
5. `code_snippet` and `evidence` must be mappings. `evidence.positive` and `evidence.negative` must be lists; use `negative: []` when no negative evidence exists.
6. Positive and negative evidence must come from actual code, documentation, call chains, installed source, lowering, valid examples, or reproducible validation. Do not fabricate evidence when none can be found.
7. `status`, `confidence`, and `confidence_value` must follow the mapping in `core/methodology.md`.

Under normal circumstances, each clause produces exactly one successful YAML submission. For accidental duplicate submissions, the collector follows AscendC behavior by preserving the original file and using `_dup1`, `_dup2`, and subsequent suffixes. Do not intentionally exploit this mechanism to create duplicate results for one clause.

Do not generate a report file.
