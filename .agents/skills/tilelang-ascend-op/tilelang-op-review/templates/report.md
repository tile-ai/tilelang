# Code Formatting and Required Review Report

> Legacy report template retained for reference only. The current `tilelang-op-review` file-review workflow does not read this template; `scripts/workflow.assemble_report.py` generates the report body from YAML.
> The actual output path for the current file review is `{review_output_dir}/{source_file}_review_summary.md`. The artifact-directory rules in `SKILL.md` determine `review_output_dir`; do not use this template to generate an additional report under `.agents/`.

- **Generated at**: {YYYYMMDD_HHMMSS}
- **Review target**: {file path / PR / diff / pasted content}

---

## 1. Check Summary

| Language | Files | Lint issues | Formatting issues | Required review issues |
|----------|-------|-------------|-------------------|------------------------|
| Python / TileLang | {N} | {N} | {N} | {N} |
| C++ | {N} | — | {N} | — |
| Markdown | {N} | — | — | {N} |
| **Total** | {N} | {N} | {N} | {N} |

---

## 2. Python File Issue Details

> Create one `###` section for each Python file with issues. Remove this section when no issues exist.

### {path/to/file.py}

#### Lint Issues

| Line | Code | Description | Suggested fix |
|------|------|-------------|---------------|
| {N} | {CODE} | {detailed issue description} | {automatic fix suggested by Ruff, or — if unavailable} |

#### Formatting Issues

```diff
{output of ruff format --diff <file>}
```

- Specific issue: {line too long / incorrect indentation / invalid blank-line layout / etc.}

---

## 3. C++ File Issue Details

> Create one `###` section for each C++ file requiring formatting. Remove this section when no issues exist.

### {path/to/file.cc}

#### Formatting Issues

```diff
{output of clang-format --style=file <file> | diff -u <file> -}
```

- Specific issue: {alignment / line wrapping / brace style / etc.}

---

## 4. Required Code Review Details

> Read the matching rules according to the reference routing in `SKILL.md`, then list confirmed issues and items requiring confirmation by file. Write N/A when no rules apply.

### {path/to/file.py}

| Canonical rule identity | Severity | Status | Location | Evidence and impact | Suggested fix |
|---|---|---|---|---|---|
| {reference-stem/CLAUSE-ID} | {High/Medium} | {Issue/Requires confirmation/Pass} | {L<N>} | {observable evidence and impact} | {recommendation} |

---

## 5. Issue Statistics Summary

**Lint Issues (by Error Code)**

| Code | Occurrences | Meaning |
|------|-------------|---------|
| {UP035} | {N} | {description} |
| {F401} | {N} | {description} |

**Formatting Issues (by File Count)**

| Language | Files requiring formatting |
|----------|----------------------------|
| Python | {N} |
| C++ | {N} |

**Required Review Issues (by Rule ID)**

| Canonical rule identity | High | Medium | Requires confirmation |
|---|---:|---:|---:|
| {reference-stem/CLAUSE-ID} | {N} | {N} | {N} |

---

## 6. Next Steps

```bash
# View lint details for one file
ruff check <file>

# View formatting differences
ruff format --diff <file>              # Python
clang-format --style=file <file> | diff -u <file> -   # C++

# After all checks finish and approval is obtained, use common.format-fix.md to apply automatic fixes
# Note: Automatic fixes address only lint/formatting and do not modify semantic issues from the required review
```
