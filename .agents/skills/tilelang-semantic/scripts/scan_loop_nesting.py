#!/usr/bin/env python3
"""Inventory lexical TileLang loop nesting in Python source files."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


LOOP_NAMES = {
    "Parallel": "Parallel",
    "parallel": "Parallel",
    "Pipelined": "Pipelined",
    "Persistent": "Persistent",
    "serial": "Serial",
    "Serial": "Serial",
    "grid": "Grid",
    "unroll": "Unroll",
    "Unroll": "Unroll",
    "vectorized": "Vectorized",
    "Vectorized": "Vectorized",
}

PIPELINE_KEYWORDS = {"num_stages", "order", "stage"}
PIPELINE_ANNOTATION_KEYS = {
    "num_stages",
    "tl_pipeline_order",
    "tl_pipeline_stage",
    "tl_pipelined_num_stages",
}


@dataclass(frozen=True)
class LoopInfo:
    kind: str
    pipeline_requested: bool = False

    @property
    def label(self) -> str:
        if self.kind != "Pipelined":
            return self.kind
        suffix = "requested" if self.pipeline_requested else "bare"
        return f"Pipelined[{suffix}]"


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_statically_empty(node: ast.expr) -> bool:
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    return False


def _is_zero(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)) and int(node.value) == 0


def _annotations_request_pipeline(node: ast.expr) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    return any(isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in PIPELINE_ANNOTATION_KEYS for key in node.keys)


def _pipeline_requested(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg == "annotations":
            if _annotations_request_pipeline(keyword.value):
                return True
            continue
        if keyword.arg not in PIPELINE_KEYWORDS:
            continue
        if keyword.arg == "num_stages" and _is_zero(keyword.value):
            continue
        if _is_statically_empty(keyword.value):
            continue
        return True
    return False


def loop_info(node: ast.For | ast.While) -> LoopInfo | None:
    if isinstance(node, ast.While):
        return LoopInfo("While")
    if not isinstance(node.iter, ast.Call):
        return None
    name = _call_name(node.iter)
    if name not in LOOP_NAMES:
        return None
    kind = LOOP_NAMES[name]
    return LoopInfo(kind, kind == "Pipelined" and _pipeline_requested(node.iter))


class NestingVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, examples_per_pair: int) -> None:
        self.path = path
        self.examples_per_pair = examples_per_pair
        self.stack: list[tuple[LoopInfo, int]] = []
        self.pairs: Counter[tuple[str, str]] = Counter()
        self.examples: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.pipeline_violations: list[str] = []

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def _visit_loop(self, node: ast.For | ast.While) -> None:
        info = loop_info(node)
        if info is None:
            self.generic_visit(node)
            return

        if self.stack:
            pair = (self.stack[-1][0].label, info.label)
            self.pairs[pair] += 1
            if len(self.examples[pair]) < self.examples_per_pair:
                self.examples[pair].append(f"{self.path}:{node.lineno}")

        if info.pipeline_requested:
            ancestor = next((entry for entry in reversed(self.stack) if entry[0].pipeline_requested), None)
            if ancestor is not None:
                ancestor_info, ancestor_line = ancestor
                self.pipeline_violations.append(
                    f"{self.path}:{node.lineno}: {info.label} is nested under {ancestor_info.label} at line {ancestor_line}"
                )

        self.stack.append((info, node.lineno))
        for child in node.body:
            self.visit(child)
        for child in node.orelse:
            self.visit(child)
        self.stack.pop()


def python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            yield root
        elif root.is_dir():
            yield from root.rglob("*.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("tilelang"), Path("testing/python"), Path("examples")],
        help="Python files or directories to scan",
    )
    parser.add_argument("--examples", type=int, default=3, help="example locations to retain per pair")
    parser.add_argument(
        "--check-pipeline-paths",
        action="store_true",
        help="exit nonzero when two pipeline-requested loops share a lexical path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    violations: list[str] = []
    parse_failures: list[str] = []

    for path in sorted(set(python_files(args.roots))):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as err:
            parse_failures.append(f"{path}: {err}")
            continue
        visitor = NestingVisitor(path, args.examples)
        visitor.visit(tree)
        pair_counts.update(visitor.pairs)
        violations.extend(visitor.pipeline_violations)
        for pair, locations in visitor.examples.items():
            remaining = args.examples - len(pair_examples[pair])
            if remaining > 0:
                pair_examples[pair].extend(locations[:remaining])

    print("Lexical TileLang loop pairs:")
    for pair, count in sorted(pair_counts.items()):
        locations = ", ".join(pair_examples[pair])
        print(f"  {pair[0]:24s} -> {pair[1]:24s} {count:5d}  {locations}")

    print("\nNested pipeline-requested paths:")
    if violations:
        for violation in violations:
            print(f"  {violation}")
    else:
        print("  none")

    if parse_failures:
        print("\nSkipped files:")
        for failure in parse_failures:
            print(f"  {failure}")

    return 1 if args.check_pipeline_paths and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
