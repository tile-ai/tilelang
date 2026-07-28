#!/usr/bin/env python3
"""Run local schedule/graph regression checks.

This script is intended for local validation after changing schedule
primitives, schedule templates, graph fusion, or graph-mode lowering.  It
rebuilds the native TileLang library by default, then runs the focused checks
that are most likely to catch regressions from aggressive scheduler changes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrimitiveTest:
    name: str
    path: str
    coverage: str


@dataclass(frozen=True)
class Suite:
    name: str
    coverage: str


PRIMITIVE_TESTS = (
    PrimitiveTest(
        "annotate_layout",
        "testing/python/primitives/test_tilelang_schedule_annotate_layout.py",
        "layout annotation attachment on the root block",
    ),
    PrimitiveTest(
        "cache_read_at",
        "testing/python/primitives/test_tilelang_schedule_cache_read_at.py",
        "global->fragment/shared cache insertion and consumer rewrite",
    ),
    PrimitiveTest(
        "cache_reduce_at",
        "testing/python/primitives/test_tilelang_schedule_cache_reduce_at.py",
        "reducer cache allocation, initialization, and write-back",
    ),
    PrimitiveTest(
        "cache_write_at",
        "testing/python/primitives/test_tilelang_schedule_cache_write_at.py",
        "fragment/shared write cache insertion and write-back behavior",
    ),
    PrimitiveTest(
        "copy_at",
        "testing/python/primitives/test_tilelang_schedule_copy_at.py",
        "copy block replacement with a tile-level T.copy call",
    ),
    PrimitiveTest(
        "fill_at",
        "testing/python/primitives/test_tilelang_schedule_fill_at.py",
        "T.fill insertion at the requested loop level",
    ),
    PrimitiveTest(
        "gemm_at",
        "testing/python/primitives/test_tilelang_schedule_gemm_at.py",
        "matmul loop replacement with a tile-level T.gemm call",
    ),
    PrimitiveTest(
        "launch_thread",
        "testing/python/primitives/test_tilelang_schedule_launch_thread.py",
        "thread launch annotation emitted by the schedule primitive",
    ),
    PrimitiveTest(
        "parallelize",
        "testing/python/primitives/test_tilelang_schedule_parallelize.py",
        "serial loop replacement with tile-level T.Parallel semantics",
    ),
    PrimitiveTest(
        "pipeline",
        "testing/python/primitives/test_tilelang_schedule_pipeline.py",
        "pipeline loop annotation and staging metadata",
    ),
    PrimitiveTest(
        "reduce_at",
        "testing/python/primitives/test_tilelang_schedule_reduce_at.py",
        "reduction loop replacement with a tile-level T.reduce call",
    ),
)

SUITES = (
    Suite(
        "primitives",
        "schedule primitive before/after IR tests",
    ),
    Suite(
        "inductor_example",
        "examples/inductor/graph_trace_example.py end-to-end graph trace smoke test",
    ),
    Suite(
        "graph_fusion_ops",
        "tests/graph/fusion_ops isolated graph fusion operator checks",
    ),
)

DEFAULT_SUITES = tuple(suite.name for suite in SUITES)
GRAPH_FUSION_DIR = "tests/graph/fusion_ops"
INDUCTOR_EXAMPLE = "examples/inductor/graph_trace_example.py"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True, timeout=timeout)


def parse_csv(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    result: set[str] = set()
    for value in values:
        result.update(part.strip() for part in value.split(",") if part.strip())
    return result


def select_suites(raw_suites: set[str]) -> list[str]:
    if not raw_suites:
        return list(DEFAULT_SUITES)
    known = {suite.name for suite in SUITES}
    unknown = sorted(raw_suites - known)
    if unknown:
        raise SystemExit(f"Unknown suite(s): {', '.join(unknown)}")
    return [suite.name for suite in SUITES if suite.name in raw_suites]


def select_primitive_tests(only: set[str]) -> list[PrimitiveTest]:
    if not only:
        return list(PRIMITIVE_TESTS)

    known = {test.name for test in PRIMITIVE_TESTS}
    known_stems = {Path(test.path).stem for test in PRIMITIVE_TESTS}
    selected: list[PrimitiveTest] = []
    for test in PRIMITIVE_TESTS:
        stem = Path(test.path).stem
        if test.name in only or stem in only:
            selected.append(test)

    missing = sorted(only - known - known_stems)
    if missing:
        raise SystemExit(f"Unknown primitive test selector(s): {', '.join(missing)}")
    return selected


def print_plan(suite_names: list[str], primitive_tests: list[PrimitiveTest]) -> None:
    suite_by_name = {suite.name: suite for suite in SUITES}
    print("Schedule/graph regression set:")
    for suite_name in suite_names:
        suite = suite_by_name[suite_name]
        print(f"  - {suite.name:<18} {suite.coverage}")
        if suite.name == "primitives":
            for test in primitive_tests:
                print(f"      * {test.name:<16} {test.path}")
                print(f"        covers: {test.coverage}")


def ensure_paths_exist(root: Path, suite_names: list[str], primitive_tests: list[PrimitiveTest]) -> None:
    missing: list[str] = []
    if "primitives" in suite_names:
        missing.extend(test.path for test in primitive_tests if not (root / test.path).is_file())
    if "inductor_example" in suite_names and not (root / INDUCTOR_EXAMPLE).is_file():
        missing.append(INDUCTOR_EXAMPLE)
    if "graph_fusion_ops" in suite_names:
        graph_fusion_path = root / GRAPH_FUSION_DIR
        if not graph_fusion_path.is_dir():
            missing.append(GRAPH_FUSION_DIR)
        elif not (graph_fusion_path / "run_all.py").is_file():
            missing.append(f"{GRAPH_FUSION_DIR}/run_all.py")
        elif not list(graph_fusion_path.glob("test_*.py")):
            missing.append(f"{GRAPH_FUSION_DIR}/test_*.py")
    if missing:
        raise SystemExit("Missing expected regression path(s):\n" + "\n".join(f"  - {path}" for path in missing))


def base_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def build_native(root: Path, build_dir: Path, jobs: int) -> None:
    if not (build_dir / "CMakeCache.txt").is_file():
        run_command(["cmake", "-S", str(root), "-B", str(build_dir)], cwd=root)
    run_command(["cmake", "--build", str(build_dir), f"-j{jobs}"], cwd=root)


def split_remainder(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def run_primitive_tests(root: Path, tests: list[PrimitiveTest], args: argparse.Namespace) -> None:
    cmd = [sys.executable, "-m", "pytest"]
    cmd.append("-vv" if args.verbose else "-q")
    if args.fail_fast:
        cmd.append("-x")
    cmd.extend(test.path for test in tests)
    cmd.extend(split_remainder(args.pytest_args))
    run_command(cmd, cwd=root, env=base_env(root))


def run_inductor_example(root: Path, args: argparse.Namespace) -> None:
    env = base_env(root)
    trace_dir = root / "examples/inductor/logs"
    trace_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("TL_GRAPH_TRACE_MODE", args.trace_mode)
    env.setdefault("TL_GRAPH_TRACE_DIR", str(trace_dir))
    env.setdefault("TL_GRAPH_TRACE_CODEGEN", str(trace_dir / "codegen.cu"))
    run_command([sys.executable, INDUCTOR_EXAMPLE], cwd=root, env=env, timeout=args.timeout)


def run_graph_fusion_ops(root: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        f"{GRAPH_FUSION_DIR}/run_all.py",
        "--log-dir",
        str(root / GRAPH_FUSION_DIR / "logs"),
        "--timeout",
        str(args.case_timeout),
    ]
    for case in args.graph_case or []:
        cmd.extend(["--case", case])
    run_command(cmd, cwd=root, env=base_env(root), timeout=args.timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        help=f"Suite(s) to run, comma-separated or repeated. Default: {', '.join(DEFAULT_SUITES)}.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the native C++ rebuild and run selected checks directly.",
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        help="CMake build directory relative to the repo root, or an absolute path.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel build jobs used for cmake --build.",
    )
    parser.add_argument(
        "--only-primitive",
        action="append",
        help="Run only selected primitive tests. Accepts names or file stems, comma-separated or repeated.",
    )
    parser.add_argument(
        "--graph-case",
        action="append",
        help="Run only graph fusion op test files containing this substring. Can be repeated.",
    )
    parser.add_argument(
        "--trace-mode",
        default="html",
        choices=("html", "terminal", "both", "0"),
        help="Trace mode used by examples/inductor/graph_trace_example.py.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for each top-level graph command.",
    )
    parser.add_argument(
        "--case-timeout",
        type=int,
        default=int(os.environ.get("TL_FUSION_TEST_TIMEOUT", "900")),
        help="Timeout in seconds for each tests/graph/fusion_ops case.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the covered suites and exit.",
    )
    parser.add_argument(
        "-x",
        "--fail-fast",
        action="store_true",
        help="Stop pytest at the first primitive-test failure.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Run primitive pytest in verbose mode.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra primitive pytest arguments after '--', for example: -- -k reduce_at",
    )
    args = parser.parse_args()

    root = repo_root()
    suite_names = select_suites(parse_csv(args.suite))
    primitive_tests = select_primitive_tests(parse_csv(args.only_primitive))
    ensure_paths_exist(root, suite_names, primitive_tests)
    print_plan(suite_names, primitive_tests)

    if args.list:
        return

    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = root / build_dir

    if not args.skip_build:
        build_native(root, build_dir, args.jobs)

    if "primitives" in suite_names:
        run_primitive_tests(root, primitive_tests, args)
    if "inductor_example" in suite_names:
        run_inductor_example(root, args)
    if "graph_fusion_ops" in suite_names:
        run_graph_fusion_ops(root, args)


if __name__ == "__main__":
    main()
