#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""Assemble format-check and clause-review YAML into one Markdown report.

Usage:
  python3 workflow.assemble_report.py --dir <yaml_dir> --output <report.md>

The input directory may contain one aggregate ``type: format`` result and any
number of ordinary ``type: clause`` results. Matching the AscendC behavior,
parseable ``_dupN`` files all participate in the report, malformed YAML is
warned about and skipped, and an incorrect ``confidence_value`` is recalculated
and written back.
"""

import argparse
from collections import Counter, defaultdict
import logging
import os
import sys
from typing import Any

import yaml


logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

CLAUSE_STATUSES = {"PASS", "FAIL", "SUSPICIOUS"}
FORMAT_STATUSES = {"PASS", "FAIL", "ERROR", "PARTIAL"}
MAX_LINT_EXAMPLES_PER_FILE_CODE = 3
MAX_FORMAT_DIFF_LINES_PER_FILE = 50


def _md_cell(value: Any) -> str:
    """Escape text for one Markdown table cell."""
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _parse_score(value: Any) -> int:
    """Parse +40%/-15%; invalid values become zero as in AscendC."""
    text = str(value or "").strip().replace("%", "").replace("+", "")
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return 0


def _confidence_total(data: dict, fname: str) -> int | None:
    """Return the clamped evidence total after validating containers."""
    evidence = data.get("evidence")
    if not evidence:
        return None
    if not isinstance(evidence, dict):
        raise ValueError(f"{fname}: evidence must be a mapping, got {type(evidence).__name__}")

    total = 0
    for key in ("positive", "negative"):
        items = evidence.get(key, []) or []
        if not isinstance(items, list):
            raise ValueError(f"{fname}: evidence.{key} must be a list, got {type(items).__name__}")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"{fname}: evidence.{key} items must be mappings, got {type(item).__name__}")
            total += _parse_score(item.get("score"))
    return max(0, min(100, total))


def _fix_confidence(data: dict, fname: str) -> bool:
    """Correct confidence_value in memory; return whether it changed."""
    total = _confidence_total(data, fname)
    if total is None:
        return False
    evidence = data["evidence"]
    correct_value = f"{total}%"
    if str(evidence.get("confidence_value", "")).strip() == correct_value:
        return False
    evidence["confidence_value"] = correct_value
    return True


def _writeback_yaml(data: dict, path: str, fname: str) -> None:
    """Write corrected confidence back, matching AscendC behavior."""
    try:
        with open(path, "w", encoding="utf-8") as output_file:
            yaml.safe_dump(
                data,
                output_file,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
    except OSError as error:
        logger.warning("Failed to write back confidence correction: %s (%s)", fname, error)


def _positive_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(value) >= 1
    except (TypeError, ValueError):
        return False


def _validate_clause(data: dict, fname: str) -> list[str]:
    """Validate a clause result after confidence correction."""
    errors = []
    required = (
        "clause_id",
        "clause_title",
        "canonical_id",
        "submission_key",
        "status",
    )
    for field in required:
        if not data.get(field):
            errors.append(f"{fname}: missing required field {field}")

    status = data.get("status")
    if status not in CLAUSE_STATUSES:
        errors.append(f"{fname}: invalid status: {status!r}")
        return errors
    if status == "PASS":
        if "evidence" in data or "confidence" in data:
            errors.append(f"{fname}: PASS must not include evidence/confidence")
        return errors

    for field in ("confidence", "problem_desc", "fix_suggestion"):
        if not data.get(field):
            errors.append(f"{fname}: {status} is missing required field {field}")

    snippet = data.get("code_snippet")
    if not isinstance(snippet, dict):
        errors.append(f"{fname}: code_snippet must be a mapping")
    else:
        if not snippet.get("file_path"):
            errors.append(f"{fname}: code_snippet.file_path is missing")
        if not snippet.get("code"):
            errors.append(f"{fname}: code_snippet.code is missing")
        start_valid = _positive_int(snippet.get("start_line"))
        end_valid = _positive_int(snippet.get("end_line"))
        if not start_valid:
            errors.append(f"{fname}: code_snippet.start_line must be a positive one-based integer")
        if not end_valid:
            errors.append(f"{fname}: code_snippet.end_line must be a positive one-based integer")
        if start_valid and end_valid and int(snippet["end_line"]) < int(snippet["start_line"]):
            errors.append(f"{fname}: code_snippet.end_line must not be less than start_line")

    try:
        total = _confidence_total(data, fname)
    except ValueError as error:
        errors.append(str(error))
        return errors
    if total is None:
        errors.append(f"{fname}: {status} is missing evidence")
        return errors

    if total >= 80:
        expected_status, expected_confidence = "FAIL", "HIGH"
    elif total >= 70:
        expected_status, expected_confidence = "SUSPICIOUS", "MED"
    else:
        expected_status, expected_confidence = "SUSPICIOUS", "LOW"
    if status != expected_status or data.get("confidence") != expected_confidence:
        errors.append(f"{fname}: {total}% must map to {expected_status}/{expected_confidence}; got {status}/{data.get('confidence')!s}")
    return errors


def _validate_format_section(section: Any, section_name: str, fname: str) -> list[str]:
    """Validate one optional language section in aggregate format YAML."""
    if section is None:
        return []
    if not isinstance(section, dict):
        return [f"{fname}: {section_name} must be a mapping"]
    errors = []
    for key in ("files_checked", "lint_issues", "format_issues"):
        if key not in section:
            continue
        value = section[key]
        if key == "files_checked" and isinstance(value, int):
            continue
        if not isinstance(value, list):
            errors.append(f"{fname}: {section_name}.{key} must be a list")
    return errors


def _validate_format(data: dict, fname: str) -> list[str]:
    """Validate an aggregate deterministic format-check result."""
    errors = []
    if not data.get("check_id"):
        errors.append(f"{fname}: format YAML is missing check_id")
    if data.get("status") not in FORMAT_STATUSES:
        errors.append(f"{fname}: invalid format status: {data.get('status')!r}")
    if "tools" in data and not isinstance(data["tools"], dict):
        errors.append(f"{fname}: tools must be a mapping")
    for section_name in ("python", "cpp", "markdown"):
        errors.extend(_validate_format_section(data.get(section_name), section_name, fname))
    for key in ("skipped_files", "execution_errors"):
        if key in data and not isinstance(data[key], list):
            errors.append(f"{fname}: {key} must be a list")
    return errors


def load_yaml_files(yaml_dir: str) -> tuple[list[dict], list[dict], list[str]]:
    """Load parseable YAML; duplicate files deliberately stay independent."""
    clause_results = []
    format_results = []
    skipped_files = []
    schema_errors = []
    corrected_count = 0

    for fname in sorted(os.listdir(yaml_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(yaml_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as input_file:
                data = yaml.safe_load(input_file)
        except (OSError, yaml.YAMLError) as error:
            logger.warning("Skipping unparsable YAML: %s (%s)", fname, error)
            skipped_files.append(fname)
            continue
        if not isinstance(data, dict):
            skipped_files.append(fname)
            continue

        result_type = data.get("type", "clause")
        if result_type == "format":
            schema_errors.extend(_validate_format(data, fname))
            format_results.append(data)
            continue
        if result_type != "clause":
            schema_errors.append(f"{fname}: unsupported YAML type: {result_type!r}")
            continue

        try:
            corrected = _fix_confidence(data, fname)
        except ValueError as error:
            schema_errors.append(str(error))
            corrected = False
        if corrected:
            _writeback_yaml(data, path, fname)
            corrected_count += 1
        schema_errors.extend(_validate_clause(data, fname))
        clause_results.append(data)

    if schema_errors:
        logger.error("%d YAML schema validations failed:", len(schema_errors))
        for error in schema_errors:
            logger.error("  %s", error)
        raise ValueError(f"{len(schema_errors)} YAML schema validations failed")
    if corrected_count:
        logger.info("Confidence validation corrected confidence_value in %d YAML files", corrected_count)
    return clause_results, format_results, skipped_files


def _pct(number: int, total: int) -> str:
    return "0%" if total == 0 else f"{number * 100 / total:.1f}%"


def _clause_stats(results: list[dict]) -> dict[str, int]:
    stats = {"total": len(results), "PASS": 0, "FAIL": 0, "SUSPICIOUS": 0}
    for result in results:
        status = result.get("status")
        if status in CLAUSE_STATUSES:
            stats[status] += 1
    return stats


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _files_checked_count(section: dict) -> int:
    files = section.get("files_checked", [])
    if isinstance(files, int):
        return max(0, files)
    return len(files) if isinstance(files, list) else 0


def _format_stats(format_results: list[dict]) -> dict[str, int]:
    stats = {
        "python_files": 0,
        "cpp_files": 0,
        "markdown_files": 0,
        "lint_issues": 0,
        "python_format_issues": 0,
        "cpp_format_issues": 0,
    }
    for result in format_results:
        python = result.get("python") if isinstance(result.get("python"), dict) else {}
        cpp = result.get("cpp") if isinstance(result.get("cpp"), dict) else {}
        markdown = result.get("markdown") if isinstance(result.get("markdown"), dict) else {}
        stats["python_files"] += _files_checked_count(python)
        stats["cpp_files"] += _files_checked_count(cpp)
        stats["markdown_files"] += _files_checked_count(markdown)
        stats["lint_issues"] += len(_as_list(python.get("lint_issues")))
        stats["python_format_issues"] += len(_as_list(python.get("format_issues")))
        stats["cpp_format_issues"] += len(_as_list(cpp.get("format_issues")))
    return stats


def _review_issue_stats(clause_results: list[dict]) -> dict[str, int]:
    stats = {"python": 0, "cpp": 0, "markdown": 0, "other": 0}
    for result in clause_results:
        if result.get("status") == "PASS":
            continue
        snippet = result.get("code_snippet")
        path = str(snippet.get("file_path", "")) if isinstance(snippet, dict) else ""
        extension = os.path.splitext(path)[1].lower()
        if extension in {".py", ".pyi"}:
            stats["python"] += 1
        elif extension in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".icc"}:
            stats["cpp"] += 1
        elif extension == ".md":
            stats["markdown"] += 1
        else:
            stats["other"] += 1
    return stats


def _issue_path(issue: Any) -> str:
    if isinstance(issue, str):
        return issue
    if not isinstance(issue, dict):
        return ""
    return str(issue.get("file_path") or issue.get("filename") or issue.get("file") or "")


def _issue_line(issue: dict) -> Any:
    if issue.get("line") is not None:
        return issue["line"]
    location = issue.get("location")
    return location.get("row", "") if isinstance(location, dict) else ""


def _issue_fix(issue: dict) -> str:
    value = issue.get("fix_suggestion") or issue.get("fix") or "—"
    if isinstance(value, dict):
        return str(value.get("message") or value.get("applicability") or value)
    return str(value)


def _collect_format_items(format_results: list[dict], section_name: str, field: str) -> list[Any]:
    items = []
    for result in format_results:
        section = result.get(section_name)
        if isinstance(section, dict):
            items.extend(_as_list(section.get(field)))
    return items


def _format_item_text(item: Any) -> str:
    """Render one skipped-file or execution-error item as readable text."""
    if not isinstance(item, dict):
        return str(item)
    path = item.get("file_path") or item.get("file") or item.get("path") or item.get("tool")
    reason = item.get("reason") or item.get("message") or item.get("error")
    if path and reason:
        return f"{path}: {reason}"
    return str(path or reason or item)


def _render_format_metadata(lines: list[str], format_results: list[dict]) -> None:
    """Render tool versions plus skipped files and execution failures."""
    tools = []
    skipped = []
    errors = []
    for result in format_results:
        tool_data = result.get("tools")
        if isinstance(tool_data, dict):
            tools.extend(tool_data.items())
        skipped.extend(_as_list(result.get("skipped_files")))
        errors.extend(_as_list(result.get("execution_errors")))

    if tools:
        lines.extend(
            [
                "### Formatting Tools",
                "",
                "| Tool | Version/Status |",
                "|---|---|",
            ]
        )
        for name, detail in tools:
            if isinstance(detail, dict):
                version = detail.get("version")
                status = detail.get("status")
                if version and status:
                    version = f"{version} ({status})"
                else:
                    version = version or status or detail
            else:
                version = detail
            lines.append(f"| {_md_cell(name)} | {_md_cell(version)} |")
        lines.append("")

    if skipped:
        lines.extend(["### Unchecked Files", ""])
        lines.extend(f"- {_format_item_text(item)}" for item in skipped)
        lines.append("")

    if errors:
        lines.extend(["### Tool Installation or Execution Failures", ""])
        lines.extend(f"- {_format_item_text(item)}" for item in errors)
        lines.append("")


def _render_format_summary(lines: list[str], format_results: list[dict], clause_results: list[dict]) -> None:
    lines.extend(["## 2. Check Summary", ""])
    review = _review_issue_stats(clause_results)
    review_total = sum(review.values())
    if not format_results:
        lines.extend(
            [
                "> This workflow received no formatting-check YAML. Formatting is recorded as “not run” and must not be treated as passing.",
                "",
                "| Language | Files | Lint issues | Formatting issues | Required review issues |",
                "|---|---:|---:|---:|---:|",
                f"| Python / TileLang | Not run | Not run | Not run | {review['python']} |",
                f"| C/C++ | Not run | — | Not run | {review['cpp']} |",
                f"| Markdown | Not run | — | — | {review['markdown']} |",
                f"| Other/Unclassified | — | — | — | {review['other']} |",
                f"| **Total** | Not run | Not run | Not run | {review_total} |",
                "",
            ]
        )
        return

    stats = _format_stats(format_results)
    statuses = Counter(str(result.get("status")) for result in format_results)
    status_text = ", ".join(f"{key} {value}" for key, value in sorted(statuses.items()))
    file_total = stats["python_files"] + stats["cpp_files"] + stats["markdown_files"]
    format_total = stats["python_format_issues"] + stats["cpp_format_issues"]
    lines.extend(
        [
            f"- Formatting-check results: {status_text}",
            "",
            "| Language | Files | Lint issues | Formatting issues | Required review issues |",
            "|---|---:|---:|---:|---:|",
            f"| Python / TileLang | {stats['python_files']} | {stats['lint_issues']} | {stats['python_format_issues']} | {review['python']} |",
            f"| C/C++ | {stats['cpp_files']} | — | {stats['cpp_format_issues']} | {review['cpp']} |",
            f"| Markdown | {stats['markdown_files']} | — | — | {review['markdown']} |",
            f"| Other/Unclassified | — | — | — | {review['other']} |",
            f"| **Total** | {file_total} | {stats['lint_issues']} | {format_total} | {review_total} |",
            "",
        ]
    )
    _render_format_metadata(lines, format_results)


def _format_description(issue: Any) -> str:
    if not isinstance(issue, dict):
        return "Formatting required"
    return str(issue.get("message") or issue.get("description") or "Formatting required")


def _issue_code(issue: Any) -> str:
    if not isinstance(issue, dict):
        return "Unclassified"
    return str(issue.get("code") or "Unclassified")


def _unique_text(values: list[str]) -> str:
    return "; ".join(dict.fromkeys(value for value in values if value)) or "—"


def _lint_example(issue: Any) -> str:
    if not isinstance(issue, dict):
        return str(issue)
    line = _issue_line(issue)
    source = issue.get("source") or issue.get("line_text") or ""
    if line and source:
        return f"L{line}: {source}"
    if line:
        return f"L{line}"
    return str(source or "Location not provided")


def _render_lint_groups(lines: list[str], issues: list[Any]) -> None:
    """Render bounded Ruff examples while retaining totals from the full YAML."""
    groups: dict[str, list[Any]] = defaultdict(list)
    for issue in issues:
        groups[_issue_code(issue)].append(issue)

    lines.extend(
        [
            "#### Lint Issues (Grouped by Ruff Code)",
            "",
            "| Ruff code | Total | Examples (up to 3) | Description | Suggested fix |",
            "|---|---:|---|---|---|",
        ]
    )
    omitted = 0
    for code, grouped in sorted(groups.items()):
        examples = grouped[:MAX_LINT_EXAMPLES_PER_FILE_CODE]
        omitted += max(0, len(grouped) - len(examples))
        descriptions = [str(item.get("message", "")) for item in examples if isinstance(item, dict)]
        fixes = [_issue_fix(item) for item in examples if isinstance(item, dict)]
        lines.append(
            f"| {_md_cell(code)} | {len(grouped)} "
            f"| {_md_cell('; '.join(_lint_example(item) for item in examples))} "
            f"| {_md_cell(_unique_text(descriptions))} "
            f"| {_md_cell(_unique_text(fixes))} |"
        )
    lines.append("")
    if omitted:
        lines.extend(
            [
                f"> This file has {omitted} additional Ruff issues not expanded here; the aggregate format YAML retains the complete results.",
                "",
            ]
        )


def _render_diff_preview(lines: list[str], diffs: list[str]) -> None:
    """Render at most 50 diff lines for one file."""
    diff_lines = []
    for diff in diffs:
        if diff:
            diff_lines.extend(str(diff).rstrip().splitlines())
    if not diff_lines:
        return

    preview = diff_lines[:MAX_FORMAT_DIFF_LINES_PER_FILE]
    lines.extend(["", "```diff", *preview, "```"])
    omitted = len(diff_lines) - len(preview)
    if omitted:
        lines.extend(
            [
                f"> The diff contains {len(diff_lines)} lines; only the first {len(preview)} are shown. The aggregate format YAML retains the remaining {omitted} lines.",
                "",
            ]
        )


def _render_python_format(lines: list[str], format_results: list[dict]) -> None:
    lines.extend(["## 3. Python Lint and Formatting Issues", ""])
    if not format_results:
        lines.extend(["Formatting checks were not run.", ""])
        return
    lint_issues = _collect_format_items(format_results, "python", "lint_issues")
    format_issues = _collect_format_items(format_results, "python", "format_issues")
    if not lint_issues and not format_issues:
        lines.extend(["No Python lint or formatting issues were found.", ""])
        return

    by_file: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"lint": [], "format": []})
    for issue in lint_issues:
        by_file[_issue_path(issue) or "Unknown file"]["lint"].append(issue)
    for issue in format_issues:
        by_file[_issue_path(issue) or "Unknown file"]["format"].append(issue)

    for file_path in sorted(by_file):
        lines.extend([f"### {file_path}", ""])
        file_lint = by_file[file_path]["lint"]
        if file_lint:
            _render_lint_groups(lines, file_lint)

        file_format = by_file[file_path]["format"]
        if file_format:
            lines.extend(["#### Formatting Issues", ""])
            for issue in file_format:
                lines.append(f"- Specific issue: {_format_description(issue)}")
            diffs = [str(issue.get("diff", "")) for issue in file_format if isinstance(issue, dict)]
            _render_diff_preview(lines, diffs)
            lines.append("")


def _render_cpp_format(lines: list[str], format_results: list[dict]) -> None:
    lines.extend(["## 4. C/C++ Formatting Issues", ""])
    if not format_results:
        lines.extend(["Formatting checks were not run.", ""])
        return
    issues = _collect_format_items(format_results, "cpp", "format_issues")
    if not issues:
        lines.extend(["No C/C++ formatting issues were found.", ""])
        return
    by_file: dict[str, list[Any]] = defaultdict(list)
    for issue in issues:
        by_file[_issue_path(issue) or "Unknown file"].append(issue)
    for file_path in sorted(by_file):
        file_issues = by_file[file_path]
        lines.extend([f"### {file_path}", ""])
        for issue in file_issues:
            lines.append(f"- Specific issue: {_format_description(issue)}")
        diffs = [str(issue.get("diff", "")) for issue in file_issues if isinstance(issue, dict)]
        _render_diff_preview(lines, diffs)
        lines.append("")


def _infer_fence_language(file_path: str) -> str:
    extension = os.path.splitext(file_path)[1].lower()
    return {
        ".py": "python",
        ".pyi": "python",
        ".md": "markdown",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
    }.get(extension, "")


def _finding_sort_key(result: dict) -> tuple[str, int, str]:
    snippet = result.get("code_snippet")
    if not isinstance(snippet, dict):
        return "", 0, str(result.get("canonical_id", ""))
    try:
        start_line = int(snippet.get("start_line", 0))
    except (TypeError, ValueError):
        start_line = 0
    return (
        str(snippet.get("file_path", "")),
        start_line,
        str(result.get("canonical_id", "")),
    )


def _render_evidence(lines: list[str], evidence: dict) -> None:
    for label, key in (("Positive Evidence", "positive"), ("Negative Evidence", "negative")):
        items = evidence.get(key, []) or []
        if not items:
            continue
        lines.extend(
            [
                f"**{label}**",
                "",
                "| Evidence type | Score | Evidence description |",
                "|---|---:|---|",
            ]
        )
        for item in items:
            lines.append(f"| {_md_cell(item.get('type', ''))} | {_md_cell(item.get('score', ''))} | {_md_cell(item.get('desc', ''))} |")
        lines.append("")
    confidence_value = evidence.get("confidence_value")
    if confidence_value:
        lines.append(f"Confidence value = clamp(Σpositive + Σnegative, 0, 100) = {confidence_value}; 80% is the FAIL threshold.")


def _render_finding(result: dict) -> str:
    canonical = result.get("canonical_id") or result.get("clause_id", "")
    lines = [f"### [{canonical}] {result.get('clause_title', '')}", ""]
    lines.append(f"- **Status**: {result.get('status', '')} | **Confidence**: {result.get('confidence', '')}")
    lines.append(f"- **Issue description**: {result.get('problem_desc', '')}")

    snippet = result.get("code_snippet")
    if isinstance(snippet, dict):
        file_path = str(snippet.get("file_path", ""))
        lines.extend(
            [
                f"- **File**: {file_path}",
                f"- **Lines**: {snippet.get('start_line', '')}-{snippet.get('end_line', '')}",
                "- **Problematic excerpt**:",
                "",
                f"```{_infer_fence_language(file_path)}",
                str(snippet.get("code", "")).rstrip(),
                "```",
            ]
        )

    evidence = result.get("evidence")
    if isinstance(evidence, dict):
        lines.extend(["", "- **Hypothesis-testing evidence**:", ""])
        _render_evidence(lines, evidence)
    lines.extend(["", f"- **Suggested fix**: {result.get('fix_suggestion', '')}"])
    return "\n".join(lines)


def _classify_findings(
    results: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    high, medium, low, out_of_range = [], [], [], []
    for result in results:
        if result.get("status") == "PASS":
            continue
        if result.get("out_of_range"):
            out_of_range.append(result)
        elif result.get("confidence") == "HIGH":
            high.append(result)
        elif result.get("confidence") == "MED":
            medium.append(result)
        else:
            low.append(result)
    for group in (high, medium, low, out_of_range):
        group.sort(key=_finding_sort_key)
    return high, medium, low, out_of_range


def _render_clause_sections(lines: list[str], clause_results: list[dict]) -> None:
    stats = _clause_stats(clause_results)
    lines.extend(
        [
            "## 5. Required Code Review Statistics",
            "",
            "| Status | Rule results | Percentage |",
            "|---|---:|---:|",
            f"| PASS | {stats['PASS']} | {_pct(stats['PASS'], stats['total'])} |",
            f"| FAIL | {stats['FAIL']} | {_pct(stats['FAIL'], stats['total'])} |",
            f"| SUSPICIOUS | {stats['SUSPICIOUS']} | {_pct(stats['SUSPICIOUS'], stats['total'])} |",
            "",
        ]
    )
    passed = sorted(
        str(result.get("canonical_id") or result.get("clause_id")) for result in clause_results if result.get("status") == "PASS"
    )
    if passed:
        lines.extend(["### Passed Rules", "", ", ".join(f"`{item}`" for item in passed), ""])

    high, medium, low, out_of_range = _classify_findings(clause_results)
    for title, findings in (
        ("## 6. Findings (HIGH Confidence)", high),
        ("## 7. Items Requiring Attention (MED Confidence)", medium),
        ("## 8. Suspected Issues (LOW Confidence)", low),
    ):
        if not findings:
            continue
        lines.extend([title, ""])
        for finding in findings:
            lines.extend([_render_finding(finding), ""])
    if out_of_range:
        lines.extend(
            [
                "## 9. Notes Outside the PR Scope",
                "",
                "> The following findings are outside the formal scope of this PR diff, but remain displayed through the compatibility field.",
                "",
            ]
        )
        for finding in out_of_range:
            lines.extend([_render_finding(finding), ""])


def _render_problem_stats(lines: list[str], format_results: list[dict], clause_results: list[dict]) -> None:
    lines.extend(["## 10. Issue Statistics Summary", ""])
    if not format_results:
        lines.extend(
            [
                "### Lint and Formatting Issues",
                "",
                "Formatting checks were not run, so lint and formatting issue statistics cannot be generated.",
            ]
        )
    else:
        lint_issues = _collect_format_items(format_results, "python", "lint_issues")
        lint_codes = Counter(str(issue.get("code") or "Unclassified") for issue in lint_issues if isinstance(issue, dict))
        lines.extend(
            [
                "### Lint Issues (by Error Code)",
                "",
                "| Code | Occurrences |",
                "|---|---:|",
            ]
        )
        if lint_codes:
            for code, count in sorted(lint_codes.items()):
                lines.append(f"| {_md_cell(code)} | {count} |")
        else:
            lines.append("| — | 0 |")

        format_stats = _format_stats(format_results)
        lines.extend(
            [
                "",
                "### Formatting Issues",
                "",
                "| Language | Issues |",
                "|---|---:|",
                f"| Python / TileLang | {format_stats['python_format_issues']} |",
                f"| C/C++ | {format_stats['cpp_format_issues']} |",
            ]
        )

    lines.extend(
        [
            "",
            "### Required Review Issues (by Rule Identity)",
            "",
            "| Canonical rule identity | FAIL | SUSPICIOUS |",
            "|---|---:|---:|",
        ]
    )
    review_counts: dict[str, Counter] = defaultdict(Counter)
    for result in clause_results:
        if result.get("status") == "PASS":
            continue
        canonical = str(result.get("canonical_id") or result.get("clause_id") or "")
        review_counts[canonical][str(result.get("status"))] += 1
    if review_counts:
        for canonical, counts in sorted(review_counts.items()):
            lines.append(f"| {_md_cell(canonical)} | {counts['FAIL']} | {counts['SUSPICIOUS']} |")
    else:
        lines.append("| — | 0 | 0 |")
    lines.append("")


def assemble_report(clause_results: list[dict], format_results: list[dict], skipped_files: list[str]) -> str:
    """Render the complete fused report."""
    stats = _clause_stats(clause_results)
    lines = [
        "# Code Formatting and Required Review Report",
        "",
        "## 1. Review Overview",
        "",
        "- **Generated at**: {{TIMESTAMP}}",
        "- **Review target**: {{CODE_FILE}}",
        "- **File types**: {{FILE_TYPES}}",
        "- **Code side**: {{SIDE}}",
        "- **Review rules**: {{DOC_LIST}}",
        f"- **Rule result count**: {stats['total']}",
        f"- **Formatting-check YAML count**: {len(format_results)}",
        "",
    ]
    if skipped_files:
        lines.extend(
            [
                f"- **Skipped invalid YAML**: {', '.join(skipped_files)}",
                "",
            ]
        )
    _render_format_summary(lines, format_results, clause_results)
    _render_python_format(lines, format_results)
    _render_cpp_format(lines, format_results)
    _render_clause_sections(lines, clause_results)
    _render_problem_stats(lines, format_results, clause_results)
    lines.extend(
        [
            "## 11. Next Steps",
            "",
            "```bash",
            "# Recheck formatting issues",
            "ruff check <file>",
            "ruff format --diff <file>",
            "clang-format --style=file <file> | diff -u <file> -",
            "```",
            "",
            "- Formatting issues may be fixed automatically only after all checks finish and the user explicitly approves the fix.",
            "- Findings from reference-based reviews are semantic issues; provide evidence and recommendations without modifying code automatically.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YAML directory → consolidated Markdown review report")
    parser.add_argument("--dir", required=True, help="Collector YAML output directory")
    parser.add_argument("--output", required=True, help="Markdown report output path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    yaml_dir = os.path.abspath(args.dir)
    output_path = os.path.abspath(args.output)
    if not os.path.isdir(yaml_dir):
        logger.error("Directory does not exist: %s", yaml_dir)
        return 2

    try:
        clause_results, format_results, skipped_files = load_yaml_files(yaml_dir)
    except ValueError:
        return 1

    report = assemble_report(clause_results, format_results, skipped_files)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as error:
            logger.error("Failed to create output directory: %s", error)
            return 2
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(report)
    except OSError as error:
        logger.error("Failed to write report: %s", error)
        return 2

    stats = _clause_stats(clause_results)
    logger.info("Report assembly complete")
    logger.info("  YAML directory: %s", yaml_dir)
    logger.info("  Report path: %s", output_path)
    logger.info("  Format YAML files: %d", len(format_results))
    logger.info(
        "  Rule statistics: total %d / PASS %d / FAIL %d / SUSPICIOUS %d",
        stats["total"],
        stats["PASS"],
        stats["FAIL"],
        stats["SUSPICIOUS"],
    )
    if skipped_files:
        logger.info("  Skipped unparsable or non-mapping YAML files: %d", len(skipped_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
