"""Audit TileLang C++ headers for API style cleanup candidates.

This script intentionally reports heuristics instead of enforcing a lint gate.
Use it to find review targets, then decide manually whether a finding is safe to
rename, FFI-visible, or worth allowlisting.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".cuh"}
SOURCE_SUFFIXES = {".cc", ".cpp", ".cxx", ".cu"}
DEFAULT_EXCLUDED_PARTS = {
    "3rdparty",
    "build",
    "dist",
    ".git",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
}
DEFAULT_EXCLUDED_TEMPLATE_PREFIXES = (Path("src/tl_templates"),)
DEFAULT_EXCLUDED_SHIM_PREFIXES = (
    Path("src/cuda/stubs"),
    Path("src/rocm/stubs"),
)
DEFAULT_EXCLUDED_VENDOR_PREFIXES = (
    Path("src/cuda/stubs/vendor"),
    Path("src/rocm/stubs/vendor"),
)

RULES = {
    "TLCPP001": "function name is not PascalCase",
    "TLCPP002": "ambiguous API parameter name",
    "TLCPP003": "public ObjectNode field needs compatibility review",
    "TLCPP004": "broad namespace import in header",
}

CONTROL_KEYWORDS = {
    "catch",
    "for",
    "if",
    "return",
    "sizeof",
    "switch",
    "while",
}

QUALIFIER_TOKENS = {
    "const",
    "constexpr",
    "explicit",
    "final",
    "inline",
    "mutable",
    "noexcept",
    "override",
    "static",
    "virtual",
}


@dataclass(order=True)
class Finding:
    path: str
    line: int
    rule: str
    symbol: str
    message: str
    snippet: str


@dataclass
class ClassState:
    name: str
    brace_depth: int
    access: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report C++ API style cleanup candidates in TileLang-owned code.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src"],
        help="Files or directories to audit. Defaults to src.",
    )
    parser.add_argument(
        "--include-sources",
        action="store_true",
        help="Also scan .cc/.cpp/.cu files. By default only headers are scanned.",
    )
    parser.add_argument(
        "--include-templates",
        action="store_true",
        help="Include src/tl_templates, which is excluded by default.",
    )
    parser.add_argument(
        "--include-runtime-shims",
        action="store_true",
        help="Include runtime shim headers under src/cuda/stubs and src/rocm/stubs.",
    )
    parser.add_argument(
        "--include-vendor",
        action="store_true",
        help="Include vendored shim headers under src/*/stubs/vendor.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of text.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum findings to print. 0 means no limit.",
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Return exit code 1 when findings are present.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path


def is_excluded(path: Path, args: argparse.Namespace) -> bool:
    rel = normalize_path(path)
    if any(part in DEFAULT_EXCLUDED_PARTS for part in rel.parts):
        return True
    if not args.include_templates:
        for prefix in DEFAULT_EXCLUDED_TEMPLATE_PREFIXES:
            if rel.is_relative_to(prefix):
                return True
    if not args.include_runtime_shims:
        for prefix in DEFAULT_EXCLUDED_SHIM_PREFIXES:
            if rel.is_relative_to(prefix):
                return True
    if not args.include_vendor:
        for prefix in DEFAULT_EXCLUDED_VENDOR_PREFIXES:
            if rel.is_relative_to(prefix):
                return True
    return False


def candidate_files(args: argparse.Namespace) -> list[Path]:
    suffixes = set(HEADER_SUFFIXES)
    if args.include_sources:
        suffixes |= SOURCE_SUFFIXES

    result: list[Path] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"warning: path does not exist: {raw_path}", file=sys.stderr)
            continue
        if path.is_file():
            if path.suffix in suffixes and not is_excluded(path, args):
                result.append(path)
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.suffix in suffixes and not is_excluded(child, args):
                result.append(child)
    return sorted(set(result))


def strip_line_comment(line: str) -> str:
    """Strip // comments while preserving URLs and simple string literals enough for auditing."""
    in_string = False
    escaped = False
    for index in range(len(line) - 1):
        char = line[index]
        if char == "\\" and in_string and not escaped:
            escaped = True
            continue
        if char == '"' and not escaped:
            in_string = not in_string
        escaped = False
        if not in_string and line[index : index + 2] == "//":
            return line[:index]
    return line


def strip_block_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), text, flags=re.S)


def is_pascal_case(name: str) -> bool:
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", name):
        return False
    return "__" not in name


def is_tvm_hook_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9]*_", name))


def split_parameters(params: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    depth = 0
    for char in params:
        if char in "(<[{":
            depth += 1
        elif char in ")>}]" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        result.append("".join(current).strip())
    return result


def parameter_name(param: str) -> str | None:
    param = re.sub(r"=.*$", "", param).strip()
    if not param or param == "void":
        return None
    param = re.sub(r"\[[^\]]*\]", "", param).strip()
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", param)
    if not match:
        return None
    name = match.group(1)
    if name in QUALIFIER_TOKENS:
        return None
    return name


def function_statement_name(statement: str) -> tuple[str, str] | None:
    compact = " ".join(statement.strip().split())
    compact = re.sub(r"^template\s*<[^>]+>\s*", "", compact)
    if not compact or compact.startswith("#"):
        return None
    first_token = compact.split(maxsplit=1)[0]
    if first_token in CONTROL_KEYWORDS:
        return None
    if compact.startswith(("using ", "typedef ", "static_assert", "template ")):
        return None
    if compact.startswith(("TVM_", "ICHECK", "LOG(", "CHECK")):
        return None
    first_paren = compact.find("(")
    if first_paren >= 0 and "=" in compact[:first_paren]:
        return None
    if "->" in compact and compact.find("->") < compact.find("("):
        return None

    match = re.search(
        r"(?:^|[\s:*&<>,~])([A-Za-z_][A-Za-z0-9_:~]*)\s*\(([^;{}]*)\)\s*"
        r"(?:const\b|noexcept\b|override\b|final\b|->|[;{=])",
        compact,
    )
    if not match:
        return None

    qualified_name = match.group(1)
    name = qualified_name.split("::")[-1]
    if name in CONTROL_KEYWORDS or name == "operator" or name.startswith("operator"):
        return None
    if name.startswith("~"):
        return None
    return name, match.group(2)


def statement_starting_at(lines: list[str], index: int) -> tuple[str, int]:
    pieces: list[str] = []
    depth = 0
    end_index = index
    for offset, line in enumerate(lines[index : min(index + 8, len(lines))]):
        stripped = strip_line_comment(line).strip()
        if not stripped:
            continue
        end_index = index + offset
        pieces.append(stripped)
        depth += stripped.count("(") - stripped.count(")")
        if depth <= 0 and (";" in stripped or "{" in stripped):
            break
    return " ".join(pieces), end_index


def audit_functions(path: Path, lines: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    class_stack: list[ClassState] = []
    brace_depth = 0
    processed_until = -1
    function_body_depth: int | None = None
    function_body_pending_until = -1

    for index, line in enumerate(lines):
        stripped = strip_line_comment(line).strip()
        if not stripped:
            continue

        if function_body_depth is not None:
            brace_depth += stripped.count("{") - stripped.count("}")
            if index > function_body_pending_until and brace_depth < function_body_depth:
                function_body_depth = None
                function_body_pending_until = -1
            continue

        class_match = re.search(r"\b(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b", stripped)
        if class_match and "{" in stripped and not stripped.endswith(";"):
            default_access = "public" if class_match.group(1) == "struct" else "private"
            class_stack.append(ClassState(class_match.group(2), brace_depth, default_access))

        if class_stack and re.fullmatch(r"(public|private|protected):", stripped):
            class_stack[-1].access = stripped[:-1]
            brace_depth += stripped.count("{") - stripped.count("}")
            continue

        in_direct_class_scope = bool(class_stack and brace_depth == class_stack[-1].brace_depth + 1)
        in_declaration_context = not class_stack or in_direct_class_scope

        if not in_declaration_context or index <= processed_until or stripped.startswith(("//", "#")):
            brace_depth += stripped.count("{") - stripped.count("}")
            while class_stack and brace_depth <= class_stack[-1].brace_depth:
                class_stack.pop()
            continue

        statement, end_index = statement_starting_at(lines, index)
        parsed = function_statement_name(statement)
        if not parsed:
            brace_depth += stripped.count("{") - stripped.count("}")
            while class_stack and brace_depth <= class_stack[-1].brace_depth:
                class_stack.pop()
            continue
        processed_until = end_index

        name, params = parsed
        if not is_pascal_case(name) and not is_tvm_hook_name(name) and "override" not in statement:
            findings.append(
                Finding(
                    str(normalize_path(path)),
                    index + 1,
                    "TLCPP001",
                    name,
                    f"Function or method `{name}` does not follow PascalCase.",
                    stripped,
                )
            )
        for param in split_parameters(params):
            if parameter_name(param) == "T":
                findings.append(
                    Finding(
                        str(normalize_path(path)),
                        index + 1,
                        "TLCPP002",
                        "T",
                        "Parameter `T` is ambiguous in API declarations; prefer a descriptive snake_case name.",
                        stripped,
                    )
                )

        if statement.count("{") > statement.count("}"):
            function_body_depth = brace_depth + 1
            function_body_pending_until = end_index

        brace_depth += stripped.count("{") - stripped.count("}")
        while class_stack and brace_depth <= class_stack[-1].brace_depth:
            class_stack.pop()
    return findings


def field_names_from_statement(statement: str) -> list[str]:
    statement = re.sub(r"=.*?(?=,|;)", "", statement)
    statement = statement.rstrip(";").strip()
    if not statement:
        return []
    names: list[str] = []
    for part in split_parameters(statement):
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", part.strip())
        if match:
            name = match.group(1)
            if name not in QUALIFIER_TOKENS:
                names.append(name)
    return names


def is_field_statement(stripped: str) -> bool:
    if not stripped.endswith(";"):
        return False
    if "(" in stripped or ")" in stripped:
        return False
    if stripped.startswith(
        (
            "class ",
            "enum ",
            "friend ",
            "namespace ",
            "struct ",
            "template ",
            "typedef ",
            "using ",
            "TVM_",
        )
    ):
        return False
    return not stripped.startswith("static constexpr ")


def audit_object_node_fields(path: Path, lines: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    class_stack: list[ClassState] = []
    brace_depth = 0

    for index, line in enumerate(lines):
        stripped = strip_line_comment(line).strip()
        class_match = re.search(r"\b(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*Node)\b", stripped)
        if class_match:
            default_access = "public" if class_match.group(1) == "struct" else "private"
            class_stack.append(ClassState(class_match.group(2), brace_depth, default_access))

        if class_stack and re.fullmatch(r"(public|private|protected):", stripped):
            class_stack[-1].access = stripped[:-1]

        if class_stack and class_stack[-1].access == "public" and is_field_statement(stripped):
            for name in field_names_from_statement(stripped):
                reasons: list[str] = []
                if name.endswith("_"):
                    reasons.append("has a trailing underscore")
                if re.search(r"[a-z][A-Za-z0-9]*[A-Z]", name):
                    reasons.append("uses camelCase")
                if name.startswith("_type_") or name.startswith("kTVMFFISEqHashKind"):
                    reasons = []
                if reasons:
                    findings.append(
                        Finding(
                            str(normalize_path(path)),
                            index + 1,
                            "TLCPP003",
                            name,
                            (
                                f"Public ObjectNode field `{name}` {' and '.join(reasons)}; "
                                "check FFI/reflection compatibility before renaming."
                            ),
                            stripped,
                        )
                    )

        brace_depth += stripped.count("{") - stripped.count("}")
        while class_stack and brace_depth <= class_stack[-1].brace_depth:
            class_stack.pop()

    return findings


def audit_namespace_imports(path: Path, lines: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for index, line in enumerate(lines):
        stripped = strip_line_comment(line).strip()
        match = re.fullmatch(r"using\s+namespace\s+([A-Za-z_][A-Za-z0-9_:]*);", stripped)
        if not match:
            continue
        namespace = match.group(1)
        findings.append(
            Finding(
                str(normalize_path(path)),
                index + 1,
                "TLCPP004",
                namespace,
                f"Header imports namespace `{namespace}` broadly; verify this is not a shared interface.",
                stripped,
            )
        )
    return findings


def audit_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = strip_block_comments(text)
    lines = text.splitlines()
    findings: list[Finding] = []
    findings.extend(audit_namespace_imports(path, lines))
    findings.extend(audit_functions(path, lines))
    findings.extend(audit_object_node_fields(path, lines))
    return sorted(findings)


def print_text(findings: list[Finding], file_count: int, limit: int) -> None:
    shown = findings if limit <= 0 else findings[:limit]
    print("C++ API style audit")
    print(f"Scanned {file_count} file(s); found {len(findings)} candidate(s).")
    if findings:
        counts = Counter(finding.rule for finding in findings)
        print("Findings by rule:")
        for code in sorted(counts):
            print(f"  {code}: {counts[code]}")
        print("Rules:")
        for code, description in RULES.items():
            print(f"  {code}: {description}")
        print()
    for finding in shown:
        print(f"{finding.path}:{finding.line}: {finding.rule}: {finding.message}")
        print(f"  {finding.snippet}")
    if limit > 0 and len(findings) > limit:
        print(f"... omitted {len(findings) - limit} finding(s); rerun with --limit 0 to show all.")


def main() -> int:
    args = parse_args()
    files = candidate_files(args)
    findings: list[Finding] = []
    for path in files:
        findings.extend(audit_file(path))
    findings.sort()

    if args.json:
        payload = {
            "file_count": len(files),
            "finding_count": len(findings),
            "rules": RULES,
            "findings": [asdict(finding) for finding in findings],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_text(findings, len(files), args.limit)

    return 1 if args.fail_on_findings and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
