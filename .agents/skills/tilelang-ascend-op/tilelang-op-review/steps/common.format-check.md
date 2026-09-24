# Format Checking and Summary Submission

This step is executed by one independent general-purpose subagent. It checks only the objects explicitly included in `file_input`, consolidates the complete results into one summary YAML with `type: format`, and submits it through the collector. It must not access `yaml_dir` or modify the files under review.

## Inputs

- Objects to check: `{file_input}`
- Collector port: `{collector_port}`
- Skill resource directory: `{skill_base}` (an absolute path passed by the main workflow, used only to invoke bundled scripts)

The main agent passes these inputs to the formatting-check subagent and requires it to execute this file in full. After completion, the subagent returns only the submission status and issue count; it does not repeat issue details or diffs in its textual response.

## Execution Process

### 1. Fix the Check Scope and Normalize Paths

Process only the files explicitly listed in `file_input`. The caller has already expanded directories or PR-changed files; do not scan the repository or expand the scope independently.

Obtain the repository root with `git rev-parse --show-toplevel`. In the summary YAML, always represent files inside the repository by paths relative to the repository root; do not write absolute workspace paths. When tools must run on content pasted by the user, create temporary files outside the repository and use logical names such as `<pasted-python>` or `<pasted-cpp>` in the results. Remove the temporary files afterward.

Classify by extension:

| Type | Extensions | Handling |
|---|---|---|
| Python | `.py`, `.pyi` | Ruff lint and format checks |
| C/C++ | `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hpp`, `.hh`, `.icc` | clang-format check |
| Markdown | `.md` | Do not run a formatting tool; record under `skipped_files` for review by `references/doc-style.md` |
| Other | Other extensions | Record under `skipped_files` and explain that it is not applicable |

Inputs that do not exist, are not regular files, or cannot be read must also be written to `skipped_files`; do not count them as checked.

### 2. Detect and Install Tools When Needed

Check a tool only when files of its corresponding type are present:

```bash
ruff --version
clang-format --version
```

Automatic installation is allowed when a tool is missing:

```bash
# Ruff
curl -LsSf https://astral.sh/ruff/install.sh | sh

# clang-format: choose one method available in the current environment
sudo apt-get install clang-format-18
# or
brew install clang-format@18
# or
python3 -m pip install clang-format==18.1.8
```

After installation, resolve the executable again and recheck its version. If a required command, network connection, permission, or package manager is unavailable, do not repeatedly retry the same path. Write the tool name, failed command, and actual error to `execution_errors`, then continue with other available checks. Do not report a missing tool or execution failure as a pass.

`tools` must record the version and status of every tool involved in this run. A tool with no corresponding input may be recorded as `not_applicable`.

### 3. Check Python Files

From the repository root, explicitly pass the filtered Python files:

```bash
bash "{skill_base}/scripts/check-python.sh" path/to/a.py path/to/b.pyi
```

Parse the JSON returned by the script:

- Convert `issues` to `python.lint_issues`.
- Convert files in `format_issues` to `python.format_issues`.
- Write paths from `checked_files` to `python.files_checked`; do not substitute the original input count or a count from only the successful lint or format operation.
- Merge `execution_errors` and `skipped_files` into the same-named fields of the summary YAML. If only lint or format succeeds for a file, do not count the file in `files_checked`, but retain issues from the successful operation. Use `lint_files_checked` and `format_files_checked` to distinguish "partially succeeded" from "completely failed," setting `PARTIAL` or `ERROR`, respectively.

Every lint issue uses the following available fields. Do not invent missing information:

```yaml
file_path: path/to/file.py
line: 12
column: 5
source: original source line
code: F401
message: imported but unused
fix_suggestion: fix information provided by Ruff, or empty
```

Read the corresponding source line at the location returned by Ruff to populate `source`. For files requiring formatting, execute:

```bash
ruff format --diff path/to/file.py
```

Write the complete diff and file path as:

```yaml
file_path: path/to/file.py
message: Ruff formatting required
diff: complete diff text
```

Do not run `ruff check --fix` or `ruff format <file>`, which would modify files.

### 4. Check C/C++ Files

From the repository root, explicitly pass the filtered C/C++ files:

```bash
bash "{skill_base}/scripts/check-cpp.sh" path/to/a.cc path/to/a.h
```

Write files actually checked successfully from the script's `checked_files` to `cpp.files_checked`, and merge the script's `execution_errors` and `skipped_files`. For every issue file returned by the script, execute:

```bash
clang-format --style=file path/to/file.cc | diff -u path/to/file.cc -
```

Write the file path, `clang-format formatting required`, and the complete diff to `cpp.format_issues`. Do not use `clang-format -i`.

### 5. Determine the Summary Status

Set the top-level `status` in this priority order:

1. `ERROR`: applicable Python or C/C++ files exist, but every applicable formatting check failed to complete because of tool or execution errors.
2. `PARTIAL`: only some applicable checks completed, unchecked inputs remain, or there are no files that Ruff/clang-format can check. Markdown-only input has this status.
3. `FAIL`: every applicable input completed its checks, with no skipped files or execution errors, but at least one lint or formatting issue was found.
4. `PASS`: every applicable input completed its checks, with no skipped files, execution errors, lint issues, or formatting issues.

Retain issues found by completed checks even when the final status is `PARTIAL`.

A nonzero exit code from Ruff caused by lint findings or formatting differences, and exit code 1 from `diff` caused by a difference, are normal "issue found" results and must not be written to `execution_errors`. Treat only failure to start a tool, unparsable output, or abnormal command interruption as an execution error.

### 6. Generate and Submit One Summary YAML

Use safe YAML serialization to generate the following structure, retaining complete issues and diffs:

```yaml
type: format
check_id: file-format
status: PASS
tools:
  ruff:
    version: 0.12.0
    status: available
  clang-format:
    version: 18.1.8
    status: available
python:
  files_checked: []
  lint_issues: []
  format_issues: []
cpp:
  files_checked: []
  format_issues: []
markdown:
  files_checked: []
skipped_files: []
execution_errors: []
```

Keep `markdown.files_checked` empty. Place Markdown files in `skipped_files` with a reason stating that they are reviewed by `references/doc-style.md`.

Submit once through this endpoint:

```text
http://127.0.0.1:{collector_port}/submit?type=format
```

Submit the complete YAML using `curl -sS --noproxy 127.0.0.1,localhost --fail-with-body -X POST --data-binary`, and check both the exit status and collector response. Access to the local collector must explicitly bypass HTTP or SOCKS proxies in the environment, but do not clear global proxy settings needed for external tool downloads. HTTP 400 means the YAML structure does not match the schema; correct it according to the actual error and resubmit. Do not proactively resubmit after HTTP 200. If an ambiguous network state causes duplicate submissions, the collector follows AscendC behavior and creates `format_dupN.yaml`; every duplicate file is included in the report and statistics.

The collector does not limit request-body size, so the format YAML may retain complete results. The final report assembler limits the number of Ruff examples and displayed diff lines.

## Return

After success, return only:

```text
Format check complete: submitted 1 format YAML; status {STATUS}; Ruff issues {N}; Python formatting {P} files; C/C++ formatting {C} files.
```

On submission failure, return the actual HTTP or tool error. Do not claim completion, and do not paste the complete YAML or diff into the textual response.

## Constraints

- Modifying project files or invoking any `fix-*.sh` during the check is strictly prohibited.
- Do not mark missing tools, command failures, nonexistent files, zero matched files, or Markdown-only input as `PASS`.
- Do not read or write `yaml_dir`; submit every result only through the collector.
- This step does not generate the final report or ask whether to apply fixes. Fixes are allowed only after all reviews and reporting are complete and the user explicitly authorizes them.
