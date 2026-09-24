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
"""Receive review YAML over HTTP and write it into one review directory.

Usage:
  python3 workflow.submit_server.py <output_dir> <port>

Endpoints:
  GET /health
      Return ``ok`` when the collector is ready.

  POST /submit?group={group_id}&rule={reference_file}&clause={clause_id}
      Accept one YAML mapping, validate its schema and identity, then write
      ``{group}__{rule_stem}__{clause}.yaml`` below ``output_dir``.

  POST /submit?type=format
      Accept one aggregate format-check YAML and write ``format.yaml`` below
      ``output_dir``.

Only the parent workflow knows ``output_dir``. Review sub-agents receive the
port and submit results through this local endpoint. If a target filename
already exists, the collector preserves it and appends ``_dup1``, ``_dup2``,
and so on for both clause and format results, matching the AscendC collector
behavior. Request bodies are read from their declared ``Content-Length``
without an additional collector size limit.
"""

import http.server
import logging
import os
import signal
import sys
import urllib.parse

import yaml


logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

OUTPUT_DIR = ""
PORT = 0
FORMAT_STATUSES = {"PASS", "FAIL", "ERROR", "PARTIAL"}


class IdentityError(ValueError):
    """Raised when a submitted review identity is incomplete or unsafe."""


def _try_fix_mojibake(value: str) -> str:
    """Recover an unescaped UTF-8 query value decoded as ISO-8859-1."""
    if not value:
        return value
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def _safe_filename(name: str) -> str:
    """Remove path separators and NUL bytes from one filename component."""
    return name.replace("/", "_").replace("\\", "_").replace("\x00", "")


def _unique_fname(fname: str, output_dir: str) -> str:
    """Append _dup1, _dup2, ... when a result filename already exists."""
    candidate = fname
    index = 1
    while os.path.exists(os.path.join(output_dir, candidate)):
        base, extension = os.path.splitext(fname)
        candidate = f"{base}_dup{index}{extension}"
        index += 1
    return candidate


def _normalize_rule_file(rule_file: str) -> str:
    """Normalize a references filename without importing another helper."""
    name = str(rule_file or "").strip()
    if not name:
        raise IdentityError("rule parameter must not be empty")
    if os.path.basename(name) != name or name in {".", ".."}:
        raise IdentityError(f"rule must be a filename directly under references/: {name!r}")
    if not name.endswith(".md"):
        name = f"{name}.md"
    if not os.path.splitext(name)[0]:
        raise IdentityError("rule filename must not be empty")
    return name


def _canonical_id(rule_file: str, clause_id: str) -> str:
    """Build the canonical ``reference-stem/clause-id`` identity."""
    clause = str(clause_id or "").strip()
    if not clause:
        raise IdentityError("clause parameter must not be empty")
    return f"{os.path.splitext(rule_file)[0]}/{clause}"


def _validate_pass_schema(data: dict) -> list[str]:
    """PASS results must not carry hypothesis evidence or confidence."""
    errors = []
    if "evidence" in data:
        errors.append("PASS rule results must not include an evidence field (required only for FAIL/SUSPICIOUS)")
    if "confidence" in data:
        errors.append("PASS rule results must not include a confidence field (required only for FAIL/SUSPICIOUS)")
    return errors


def _validate_fail_snippet(data: dict) -> list[str]:
    """Validate the source snippet attached to FAIL/SUSPICIOUS results."""
    errors = []
    snippet = data.get("code_snippet")
    if snippet is None:
        errors.append("FAIL/SUSPICIOUS rule result is missing the code_snippet field")
    elif not isinstance(snippet, dict):
        errors.append(f"code_snippet must be a mapping (file_path/start_line/end_line/code); actual type: {type(snippet).__name__}")
    else:
        if not snippet.get("file_path"):
            errors.append("code_snippet.file_path is missing or empty")
        if not snippet.get("code"):
            errors.append("code_snippet.code is missing or empty (FAIL/SUSPICIOUS must include a code excerpt)")
    return errors


def _validate_evidence_list(items: list, key: str) -> list[str]:
    """Validate evidence.positive or evidence.negative entries."""
    errors = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"evidence.{key}[{index}] must be a mapping (type/score/desc); actual type: {type(item).__name__}")
        elif not item.get("score"):
            errors.append(f"evidence.{key}[{index}].score is missing or empty")
    return errors


def _validate_fail_evidence(data: dict) -> list[str]:
    """Validate hypothesis evidence attached to FAIL/SUSPICIOUS results."""
    errors = []
    evidence = data.get("evidence")
    if evidence is None:
        return ["FAIL/SUSPICIOUS rule result is missing the evidence field"]
    if not isinstance(evidence, dict):
        return [f"evidence must be a mapping (positive/negative/confidence_value); actual type: {type(evidence).__name__}"]

    for key in ("positive", "negative"):
        value = evidence.get(key)
        if value is None:
            errors.append(f"evidence.{key} is missing")
        elif not isinstance(value, list):
            errors.append(f"evidence.{key} must be a list; actual type: {type(value).__name__}")
        else:
            errors.extend(_validate_evidence_list(value, key))
    if not evidence.get("confidence_value"):
        errors.append("evidence.confidence_value is missing or empty")
    return errors


def _validate_yaml_schema(data: dict) -> list[str]:
    """Validate one ordinary TileLang, Python, or Markdown clause result."""
    errors = []
    for field in ("clause_id", "status"):
        if not data.get(field):
            errors.append(f"missing required field: {field}")

    status = data.get("status", "")
    if status == "PASS":
        errors.extend(_validate_pass_schema(data))
        return errors
    if status not in ("FAIL", "SUSPICIOUS"):
        errors.append(f"invalid status: expected PASS/FAIL/SUSPICIOUS, got '{status}'")
        return errors

    if not data.get("problem_desc"):
        if data.get("description"):
            errors.append("incorrect field name: expected 'problem_desc', got 'description'")
        else:
            errors.append("missing required field: problem_desc")
    if not data.get("fix_suggestion"):
        if data.get("suggestion"):
            errors.append("incorrect field name: expected 'fix_suggestion', got 'suggestion'")
        else:
            errors.append("missing required field: fix_suggestion")

    errors.extend(_validate_fail_snippet(data))
    errors.extend(_validate_fail_evidence(data))
    return errors


def _validate_format_section(data: dict, section_name: str, fields: tuple[str, ...]) -> list[str]:
    """Validate one required language section in aggregate format YAML."""
    section = data.get(section_name)
    if not isinstance(section, dict):
        return [f"{section_name} must be a mapping"]

    errors = []
    for field in fields:
        if field not in section:
            errors.append(f"{section_name}.{field} is missing")
            continue
        value = section[field]
        if field == "files_checked" and isinstance(value, int) and not isinstance(value, bool):
            if value < 0:
                errors.append(f"{section_name}.files_checked must not be negative")
            continue
        if not isinstance(value, list):
            errors.append(f"{section_name}.{field} must be a list")
    return errors


def _validate_format_schema(data: dict) -> list[str]:
    """Validate one aggregate Ruff/clang-format result."""
    errors = []
    if data.get("type") != "format":
        errors.append("type must be 'format'")
    if not data.get("check_id"):
        errors.append("missing required field: check_id")
    if data.get("status") not in FORMAT_STATUSES:
        errors.append(f"invalid status: expected PASS/FAIL/PARTIAL/ERROR, got {data.get('status')!r}")
    if not isinstance(data.get("tools"), dict):
        errors.append("tools must be a mapping")

    errors.extend(
        _validate_format_section(
            data,
            "python",
            ("files_checked", "lint_issues", "format_issues"),
        )
    )
    errors.extend(_validate_format_section(data, "cpp", ("files_checked", "format_issues")))
    errors.extend(_validate_format_section(data, "markdown", ("files_checked",)))
    for field in ("skipped_files", "execution_errors"):
        if not isinstance(data.get(field), list):
            errors.append(f"{field} must be a list")
    return errors


def _resolve_fname(group: str, rule_file: str, clause: str) -> str:
    """Generate a safe, unique filename from the canonical review identity."""
    group_safe = _safe_filename(group)
    rule_safe = _safe_filename(os.path.splitext(rule_file)[0])
    clause_safe = _safe_filename(clause)
    fname = f"{group_safe}__{rule_safe}__{clause_safe}.yaml"
    return _unique_fname(fname, OUTPUT_DIR)


class CollectorHandler(http.server.BaseHTTPRequestHandler):
    """Handle YAML submissions from review sub-agents."""

    def log_message(self, *args) -> None:
        """Suppress default access logs to avoid polluting workflow output."""

    def _handle_submit(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/submit":
            self.send_error(404, "not found")
            return

        params = urllib.parse.parse_qs(parsed.query)
        submission_type = str(params.get("type", ["clause"])[0]).strip() or "clause"

        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self.send_error(400, "invalid Content-Length")
            return
        try:
            body = self.rfile.read(content_length).decode("utf-8") if content_length else ""
        except UnicodeDecodeError as error:
            self.send_error(400, "body must be UTF-8", str(error))
            return

        try:
            data = yaml.safe_load(body)
        except yaml.YAMLError as error:
            self.send_error(400, "invalid yaml", str(error))
            return
        if not isinstance(data, dict):
            self.send_error(400, "yaml root must be a mapping")
            return

        if submission_type == "format":
            errors = _validate_format_schema(data)
            if errors:
                self.send_error(400, "format schema validation failed", "; ".join(errors))
                return
            fname = _unique_fname("format.yaml", OUTPUT_DIR)
            self._write_result(data, fname)
            return

        if submission_type != "clause":
            self.send_error(400, "unsupported result type", f"query type={submission_type!r}")
            return
        if data.get("type", "clause") != "clause":
            self.send_error(400, "result type mismatch", "format YAML must use /submit?type=format")
            return

        group = _try_fix_mojibake(params.get("group", [""])[0]).strip()
        rule = _try_fix_mojibake(params.get("rule", [""])[0]).strip()
        clause = _try_fix_mojibake(params.get("clause", [""])[0]).strip()
        if not group:
            self.send_error(400, "missing group parameter")
            return
        try:
            rule_file = _normalize_rule_file(rule)
            canonical = _canonical_id(rule_file, clause)
        except IdentityError as error:
            self.send_error(400, "invalid review identity", str(error))
            return

        body_clause = str(data.get("clause_id") or "").strip()
        if body_clause != clause:
            self.send_error(
                400,
                "clause identity mismatch",
                f"URL={clause!r}, body={body_clause!r}",
            )
            return

        data["group_name"] = group
        data["rule_file"] = rule_file
        data["canonical_id"] = canonical
        data["submission_key"] = f"{group}::{canonical}"

        errors = _validate_yaml_schema(data)
        if errors:
            self.send_error(400, "schema validation failed", "; ".join(errors))
            return

        fname = _resolve_fname(group, rule_file, clause)
        self._write_result(data, fname)

    def _write_result(self, data: dict, fname: str) -> None:
        """Write one validated result and return its collector filename."""
        output_path = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(output_path, "w", encoding="utf-8") as output_file:
                yaml.safe_dump(data, output_file, allow_unicode=True, sort_keys=False)
        except OSError as error:
            self.send_error(500, "write failed", str(error))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(f"ok: {fname}\n".encode())

    def _handle_health(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404, "not found")

    do_POST = _handle_submit
    do_GET = _handle_health


def main() -> int:
    global OUTPUT_DIR, PORT

    if len(sys.argv) != 3:
        logger.error("Usage: python3 workflow.submit_server.py <output_dir> <port>")
        return 1

    OUTPUT_DIR = os.path.abspath(sys.argv[1])
    try:
        PORT = int(sys.argv[2])
    except ValueError:
        logger.error("port must be an integer: %s", sys.argv[2])
        return 1

    if not os.path.isdir(OUTPUT_DIR):
        logger.error("output_dir does not exist: %s", OUTPUT_DIR)
        return 1
    if not 1 <= PORT <= 65535:
        logger.error("port is outside the valid range: %s", PORT)
        return 1

    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    try:
        server = http.server.HTTPServer(("127.0.0.1", PORT), CollectorHandler)
    except OSError as error:
        logger.error("Unable to listen on port %s: %s", PORT, error)
        return 2

    logger.info("collector listening on http://127.0.0.1:%s, output: %s", PORT, OUTPUT_DIR)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
