#!/usr/bin/env python3
"""Statically discover pytest entry points in the current repository without importing repository modules.

The output supports review but does not prove that tests were collected at runtime or
that they are trustworthy. Dynamic parameters, helper assertions, aliases, and dispatch
logic may still require manual inspection or running ``pytest --collect-only``.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any
from collections.abc import Iterable


EXCLUDED_DIRS = {".git", ".venv", "__pycache__", "build", "dist"}
VALUE_ASSERT_CALLS = {
    "assert_close",
    "assert_equal",
    "allclose",
    "equal",
    "calc_diff",
}
DEVICE_LITERAL_PREFIXES = {"cpu", "cuda", "npu"}
BACKEND_HINT_LEAVES = {
    "get_device",
    "is_ascend",
    "is_cuda",
    "PlatformEnum",
    "PLATFORM",
}
PROCESS_STATE_MUTATOR_LEAVES = {
    "manual_seed",
    "manual_seed_all",
    "set_default_device",
    "set_default_dtype",
    "set_default_tensor_type",
    "set_token_alignment",
}


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def literal_case_count(node: ast.AST) -> int | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts)
    if isinstance(node, ast.Dict):
        return len(node.keys)
    return None


def parameter_names(node: ast.AST) -> list[str] | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def decorator_info(decorator: ast.AST) -> dict[str, Any]:
    call = decorator if isinstance(decorator, ast.Call) else None
    target = call.func if call else decorator
    name = dotted_name(target)
    info: dict[str, Any] = {"name": name or ast.unparse(decorator)}
    if call and name.endswith(".parametrize") and len(call.args) >= 2:
        info["parameters"] = parameter_names(call.args[0])
        info["literal_case_count"] = literal_case_count(call.args[1])
        if info["literal_case_count"] is None:
            info["case_source"] = ast.unparse(call.args[1])
    return info


def call_names(node: ast.AST) -> list[str]:
    return sorted({name for child in ast.walk(node) if isinstance(child, ast.Call) for name in [dotted_name(child.func)] if name})


def reference_identifiers(node: ast.AST) -> list[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            name = child.id
        elif isinstance(child, ast.Attribute):
            name = dotted_name(child)
        else:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf.endswith("_ref") or "golden" in leaf.lower() or "reference" in leaf.lower():
            names.add(name)
    return sorted(names)


def identifier_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(dotted_name(child))
    return names


def device_literals(node: ast.AST) -> list[str]:
    values = {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant)
        and isinstance(child.value, str)
        and child.value.split(":", 1)[0].lower() in DEVICE_LITERAL_PREFIXES
    }
    return sorted(values)


def reachable_local_helpers(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    function_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Recursively trace direct calls to same-file helpers without importing the module."""
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    pending = [node]
    while pending:
        current = pending.pop()
        for child in ast.walk(current):
            if not isinstance(child, ast.Call):
                continue
            called_name = dotted_name(child.func)
            if called_name not in function_defs or called_name == node.name or called_name in found:
                continue
            helper = function_defs[called_name]
            found[called_name] = helper
            pending.append(helper)
    return sorted(found.values(), key=lambda item: item.lineno)


def assertion_summary(node: ast.AST) -> dict[str, Any]:
    value_calls: list[str] = []
    shape_asserts = 0
    general_asserts = 0
    raises: list[str] = []
    state_checks = 0

    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            source = ast.unparse(child.test)
            calls = [dotted_name(item.func) for item in ast.walk(child.test) if isinstance(item, ast.Call)]
            matched = [name for name in calls if name.rsplit(".", 1)[-1] in VALUE_ASSERT_CALLS]
            if matched:
                value_calls.extend(matched)
            elif any(token in source.lower() for token in ("diff", "atol", "rtol", "expected", "reference")):
                value_calls.append("numeric-expression")
            elif ".shape" in source or ".ndim" in source or ".dim(" in source:
                shape_asserts += 1
            else:
                general_asserts += 1
            if any(token in source for token in ("storage", "unchanged", "padding", "stride")):
                state_checks += 1

        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
            name = dotted_name(child.value.func)
            if name.rsplit(".", 1)[-1] in VALUE_ASSERT_CALLS:
                value_calls.append(name)
            source = ast.unparse(child.value)
            if any(token in source for token in ("storage", "unchanged", "padding")):
                state_checks += 1

        if isinstance(child, (ast.With, ast.AsyncWith)):
            for item in child.items:
                context = item.context_expr
                if isinstance(context, ast.Call) and dotted_name(context.func).endswith("pytest.raises"):
                    raises.append(ast.unparse(context))

    return {
        "value_assert_calls": sorted(set(value_calls)),
        "shape_assert_count": shape_asserts,
        "general_assert_count": general_asserts,
        "exception_oracles": raises,
        "state_or_storage_check_count": state_checks,
    }


def iter_python_files(repo_root: Path, roots: Iterable[str]) -> Iterable[Path]:
    for root_name in roots:
        root = (repo_root / root_name).resolve()
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        for path in sorted(root.rglob("*.py")):
            if not any(part in EXCLUDED_DIRS for part in path.parts):
                yield path


def analyze_file(repo_root: Path, path: Path, symbols: set[str]) -> tuple[list[dict[str, Any]], str | None]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [], f"{type(exc).__name__}: {exc}"

    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    file_has_level_control = "get_test_level" in source
    function_defs = {item.name: item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    results: list[dict[str, Any]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
            continue
        helpers = reachable_local_helpers(node, function_defs)
        analysis_scope = ast.Module(body=[node, *helpers], type_ignores=[])
        calls = call_names(analysis_scope)
        identifiers = identifier_names(analysis_scope)
        referenced_names = set(calls) | identifiers
        backend_hints = sorted(name for name in referenced_names if name.rsplit(".", 1)[-1] in BACKEND_HINT_LEAVES)
        process_state_mutations = sorted(name for name in calls if name.rsplit(".", 1)[-1] in PROCESS_STATE_MUTATOR_LEAVES)
        matched_symbols = sorted(
            symbol for symbol in symbols if any(name == symbol or name.endswith(f".{symbol}") for name in referenced_names)
        )
        if symbols and not matched_symbols:
            continue

        decorators = [decorator_info(item) for item in node.decorator_list]
        decorator_names = [str(item["name"]) for item in decorators]
        oracle_calls = reference_identifiers(analysis_scope)
        results.append(
            {
                "node_hint": f"{relative}::{node.name}",
                "file": relative,
                "line": node.lineno,
                "layout": "package-test" if path.name.startswith("test_") or path.name.endswith("_test.py") else "embedded-test",
                "function": node.name,
                "matched_symbols": matched_symbols,
                "device_literals": device_literals(analysis_scope),
                "backend_hints": backend_hints,
                "process_state_mutations": process_state_mutations,
                "benchmark": any("benchmark" in name for name in decorator_names),
                "skip_decorators": [name for name in decorator_names if name.endswith("skip") or name.endswith("skipif")],
                "parametrize": [item for item in decorators if str(item["name"]).endswith(".parametrize")],
                "file_uses_test_level": file_has_level_control,
                "helper_functions": [helper.name for helper in helpers],
                "reference_like_calls": oracle_calls,
                "direct_assertions": assertion_summary(node),
                "assertions": assertion_summary(analysis_scope),
            }
        )
    return results, None


def discover(repo_root: Path, roots: list[str], symbols: set[str]) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    scanned = 0
    seen: set[Path] = set()
    for path in iter_python_files(repo_root, roots):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        scanned += 1
        found, error = analyze_file(repo_root, resolved, symbols)
        tests.extend(found)
        if error:
            parse_errors.append({"file": str(path), "error": error})
    return {
        "repo_root": str(repo_root.resolve()),
        "roots": roots,
        "symbols": sorted(symbols),
        "summary": {
            "python_files_scanned": scanned,
            "matching_test_functions": len(tests),
            "package_test_functions": sum(item["layout"] == "package-test" for item in tests),
            "embedded_test_functions": sum(item["layout"] == "embedded-test" for item in tests),
            "parse_errors": len(parse_errors),
        },
        "tests": tests,
        "parse_errors": parse_errors,
        "limitations": [
            "Static AST evidence cannot prove that pytest collected or executed these tests.",
            "Direct calls by name to same-file helpers are traced recursively; imported, aliased, dynamically called, or method-form helpers still require manual inspection.",
            "Dynamic parameter counts and runtime backend dispatch may not be determinable through static analysis.",
            "Device literals and process-state mutation calls are review hints only; they do not prove the actual runtime backend or that state leakage occurred.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# pytest Static Discovery Results",
        "",
        f"Scanned {summary['python_files_scanned']} Python files and found {summary['matching_test_functions']} matching test functions.",
        "",
        "| Test | Layout | Benchmark | Uses Test Level | Backend/Device Hints | Process-State Mutations | Traced Helpers | Reference-Like Calls | Numerical Assertions |",
        "|---|---|---:|---:|---|---|---|---|---|",
    ]
    for item in report["tests"]:
        references = ", ".join(item["reference_like_calls"]) or "—"
        value_asserts = ", ".join(item["assertions"]["value_assert_calls"]) or "—"
        helpers = ", ".join(item["helper_functions"]) or "—"
        backend_hints = ", ".join([*item["device_literals"], *item["backend_hints"]]) or "—"
        state_mutations = ", ".join(item["process_state_mutations"]) or "—"
        layout = {"package-test": "standalone test", "embedded-test": "source-embedded test"}.get(item["layout"], item["layout"])
        benchmark = "Yes" if item["benchmark"] else "No"
        level_aware = "Yes" if item["file_uses_test_level"] else "No"
        lines.append(
            f"| `{item['node_hint']}` | {layout} | {benchmark} | "
            f"{level_aware} | {backend_hints} | {state_mutations} | "
            f"{helpers} | {references} | {value_asserts} |"
        )
    if not report["tests"]:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    lines.extend(["", "Static discovery cannot prove that tests were collected, executed, or trustworthy."])
    if report["parse_errors"]:
        lines.extend(["", "## Parse Errors", ""])
        lines.extend(f"- `{item['file']}`: {item['error']}" for item in report["parse_errors"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--root", action="append", dest="roots", help="Relative file or directory to scan; may be specified repeatedly")
    parser.add_argument("--symbol", action="append", default=[], help="Public function symbol to match; may be specified repeatedly")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path, help="Write results to this file instead of standard output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = (
        args.roots
        if args.roots is not None
        else [
            name
            for name in ("testing/ascend", "examples/ascend", "operators", "agent/operators", "tests")
            if (args.repo_root / name).is_dir()
        ]
    )
    if not roots:
        raise SystemExit("No test directories found; pass --root with the target test file or directory.")
    report = discover(args.repo_root, roots, set(args.symbol))
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n" if args.format == "json" else render_markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 1 if report["parse_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
