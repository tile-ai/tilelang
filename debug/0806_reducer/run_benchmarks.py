"""Compare the legacy reducer checkout with reducer v2.

The two TileLang versions are always loaded in separate worker processes so
that their Python modules, native libraries, and TVM registrations cannot mix.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from benchmark_cases import CASES, BenchmarkCase, select_cases


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_V2_REPO = SCRIPT_DIR.parents[1]
DEFAULT_LEGACY_REPO = DEFAULT_V2_REPO.parent / "tilelang_ref"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("quick", "full", "diagnostic", "all"),
        default="quick",
        help="Predefined benchmark matrix (default: quick).",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_patterns",
        help="Exact case name or shell-style pattern; repeat to select several cases.",
    )
    parser.add_argument("--legacy-repo", type=Path, default=DEFAULT_LEGACY_REPO)
    parser.add_argument("--v2-repo", type=Path, default=DEFAULT_V2_REPO)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument(
        "--order",
        choices=("legacy-first", "v2-first"),
        default="legacy-first",
        help="Execution order, useful for checking thermal/order bias.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument(
        "--benchmark-incorrect",
        action="store_true",
        help="Collect diagnostic latency even when correctness fails.",
    )
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    return parser.parse_args()


def selected_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    if not args.case_patterns:
        return select_cases(args.suite)

    selected_names: set[str] = set()
    for pattern in args.case_patterns:
        matched = [case.name for case in CASES if fnmatch.fnmatchcase(case.name, pattern)]
        if not matched:
            raise ValueError(f"Case pattern matched nothing: {pattern}")
        selected_names.update(matched)
    return [case for case in CASES if case.name in selected_names]


def print_cases(cases: list[BenchmarkCase]) -> None:
    print(f"{'name':48} {'family':16} {'shape / launch'}")
    for case in cases:
        shape = f"blocks={case.blocks}, m={case.m}, k={case.k}, threads={case.threads}"
        if case.tile_k:
            shape += f", tile_k={case.tile_k}, stages={case.num_stages}"
        if case.batch != 1:
            shape += f", batch={case.batch}"
        print(f"{case.name:48} {case.family:16} {shape}")


def validate_repo(label: str, repo: Path) -> Path:
    repo = repo.resolve()
    required = (repo / "tilelang", repo / "build" / "lib" / "libtilelang.so")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        details = "\n  ".join(missing)
        raise ValueError(f"{label} checkout is not build-ready; missing:\n  {details}")
    return repo


def make_output_dir(requested: Path | None) -> Path:
    if requested is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        requested = SCRIPT_DIR / "results" / timestamp
    requested = requested.resolve()
    if requested.exists():
        existing = list(requested.iterdir())
        if existing:
            details = "\n  ".join(str(path) for path in existing[:8])
            if len(existing) > 8:
                details += f"\n  ... and {len(existing) - 8} more"
            raise ValueError(f"Refusing to use non-empty output directory:\n  {details}")
    requested.mkdir(parents=True, exist_ok=True)
    return requested


def worker_env(repo: Path, output_dir: Path, variant: str, both_repos: tuple[Path, Path]):
    env = os.environ.copy()
    old_entries = env.get("PYTHONPATH", "").split(os.pathsep)
    excluded = {str(path.resolve()) for path in both_repos}
    retained = []
    for entry in old_entries:
        if not entry or entry == ".":
            continue
        try:
            resolved = str(Path(entry).resolve())
        except OSError:
            resolved = entry
        if resolved not in excluded:
            retained.append(entry)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), *retained])
    env["PYTHONUNBUFFERED"] = "1"
    env["TILELANG_CACHE_DIR"] = str(output_dir / "_cache" / variant)
    env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    return env


def run_worker(
    variant: str,
    repo: Path,
    cases: list[BenchmarkCase],
    args: argparse.Namespace,
    output_dir: Path,
    both_repos: tuple[Path, Path],
) -> int:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "benchmark_worker.py"),
        "--repo",
        str(repo),
        "--variant",
        variant,
        "--output",
        str(output_dir / f"{variant}.json"),
        "--source-dir",
        str(output_dir / "sources"),
        "--device",
        str(args.device),
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
    ]
    for case in cases:
        command.extend(("--case", case.name))
    if args.no_benchmark:
        command.append("--no-benchmark")
    if args.benchmark_incorrect:
        command.append("--benchmark-incorrect")

    print(f"\n=== Running {variant}: {repo} ===", flush=True)
    completed = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        env=worker_env(repo, output_dir, variant, both_repos),
        check=False,
    )
    if completed.returncode:
        print(f"{variant} worker exited with status {completed.returncode}", file=sys.stderr)
    return completed.returncode


def load_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"metadata": {}, "cases": []}
    return json.loads(path.read_text(encoding="utf-8"))


def result_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["case"]["name"]: entry for entry in payload.get("cases", [])}


def correctness(entry: dict[str, Any] | None) -> bool | None:
    if not entry or entry.get("status") != "ok":
        return None
    return entry.get("correctness", {}).get("correct")


def latency(entry: dict[str, Any] | None) -> float | None:
    if not entry or entry.get("status") != "ok":
        return None
    value = entry.get("latency_ms")
    return float(value) if value is not None else None


def nested(entry: dict[str, Any] | None, *keys: str):
    value: Any = entry
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def describe_status(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "MISSING"
    if entry.get("status") == "error":
        return "ERROR"
    value = correctness(entry)
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "UNKNOWN"


def format_number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def format_integer(value: int | None) -> str:
    return "—" if value is None else str(value)


def comparison_rows(cases: list[BenchmarkCase], legacy: dict[str, Any], v2: dict[str, Any]) -> list[dict[str, Any]]:
    legacy_by_name = result_index(legacy)
    v2_by_name = result_index(v2)
    rows = []
    for case in cases:
        old = legacy_by_name.get(case.name)
        new = v2_by_name.get(case.name)
        old_latency = latency(old)
        new_latency = latency(new)
        ratio = None
        if (
            correctness(old) is True
            and correctness(new) is True
            and old_latency is not None
            and new_latency is not None
            and new_latency > 0
        ):
            ratio = old_latency / new_latency

        notes = []
        if old and old.get("status") == "error":
            notes.append(f"legacy error: {old.get('error')}")
        if new and new.get("status") == "error":
            notes.append(f"v2 error: {new.get('error')}")
        if not case.expected_legacy_correct and correctness(old) is False:
            notes.append("expected legacy correctness failure")
        elif not case.expected_legacy_correct and correctness(old) is True:
            notes.append("legacy unexpectedly passed")
        if correctness(new) is False:
            notes.append("v2 correctness failure")
        if ratio is None and (old_latency is not None or new_latency is not None):
            notes.append("speedup suppressed unless both outputs are correct")

        rows.append(
            {
                "case": case.name,
                "family": case.family,
                "blocks": case.blocks,
                "m": case.m,
                "k": case.k,
                "tile_k": case.tile_k or "",
                "threads": case.threads,
                "batch": case.batch,
                "legacy_replication": case.legacy_replication,
                "expected_legacy_correct": case.expected_legacy_correct,
                "legacy_status": describe_status(old),
                "legacy_correct": correctness(old),
                "legacy_latency_ms": old_latency,
                "v2_status": describe_status(new),
                "v2_correct": correctness(new),
                "v2_latency_ms": new_latency,
                "legacy_over_v2": ratio,
                "legacy_compile_seconds": nested(old, "compile_seconds"),
                "v2_compile_seconds": nested(new, "compile_seconds"),
                "legacy_all_reduce_calls": nested(old, "source_metrics", "all_reduce_calls"),
                "v2_all_reduce_calls": nested(new, "source_metrics", "all_reduce_calls"),
                "legacy_source_bytes": nested(old, "source_metrics", "bytes"),
                "v2_source_bytes": nested(new, "source_metrics", "bytes"),
                "notes": "; ".join(notes),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata_lines(label: str, payload: dict[str, Any]) -> list[str]:
    metadata = payload.get("metadata", {})
    git = metadata.get("git", {})
    commit = git.get("commit", "unknown")
    if commit != "unknown":
        commit = commit[:12]
    dirty = "dirty" if git.get("dirty") else "clean"
    return [
        f"- {label}: `{metadata.get('repo', 'unknown')}`",
        f"  - TileLang {metadata.get('tilelang_version', 'unknown')}, commit `{commit}` ({dirty})",
        f"  - imported from `{metadata.get('tilelang_file', 'unknown')}`",
    ]


def render_summary(
    cases: list[BenchmarkCase],
    rows: list[dict[str, Any]],
    legacy: dict[str, Any],
    v2: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> str:
    metadata = v2.get("metadata") or legacy.get("metadata") or {}
    lines = [
        "# Reducer benchmark comparison",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "`legacy / v2 > 1.0` means reducer v2 is faster. A ratio is emitted only when both outputs pass correctness.",
        "",
        "## Environment",
        "",
        *metadata_lines("legacy", legacy),
        *metadata_lines("v2", v2),
        f"- GPU: {metadata.get('gpu_name', 'unknown')} (device {args.device}, compute capability {metadata.get('compute_capability', 'unknown')})",
        f"- Timing: median CUDA-event latency, warmup={args.warmup_ms} ms, measurement={args.rep_ms} ms",
        "- Both variants disable warp specialization and TMA lowering.",
        "",
        "## Results",
        "",
        "| case | family | legacy correct | v2 correct | legacy ms | v2 ms | legacy / v2 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {family} | {legacy_status} | {v2_status} | {old} | {new} | {ratio} |".format(
                case=row["case"],
                family=row["family"],
                legacy_status=row["legacy_status"],
                v2_status=row["v2_status"],
                old=format_number(row["legacy_latency_ms"]),
                new=format_number(row["v2_latency_ms"]),
                ratio=format_number(row["legacy_over_v2"], digits=3),
            )
        )

    lines.extend(
        [
            "",
            "## Generated-code indicators",
            "",
            "These textual counts are diagnostics, not a substitute for profiler data.",
            "",
            "| case | legacy AllReduce sites | v2 AllReduce sites | legacy source bytes | v2 source bytes |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case']} | {format_integer(row['legacy_all_reduce_calls'])} | "
            f"{format_integer(row['v2_all_reduce_calls'])} | "
            f"{format_integer(row['legacy_source_bytes'])} | "
            f"{format_integer(row['v2_source_bytes'])} |"
        )

    notable = [row for row in rows if row["notes"]]
    if notable:
        lines.extend(("", "## Notes", ""))
        lines.extend(f"- `{row['case']}`: {row['notes']}" for row in notable)

    lines.extend(
        (
            "",
            "## Artifacts",
            "",
            f"- Machine-readable comparison: `{output_dir / 'comparison.csv'}`",
            f"- Raw worker results: `{output_dir / 'legacy.json'}`, `{output_dir / 'v2.json'}`",
            f"- Generated CUDA: `{output_dir / 'sources'}`",
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        cases = selected_cases(args)
        if args.list:
            print_cases(list(CASES))
            return 0
        if not cases:
            raise ValueError(f"Suite contains no cases: {args.suite}")
        legacy_repo = validate_repo("legacy", args.legacy_repo)
        v2_repo = validate_repo("v2", args.v2_repo)
        if legacy_repo == v2_repo:
            raise ValueError("Legacy and v2 repositories must be different checkouts")
        output_dir = make_output_dir(args.output_dir)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    print("Selected cases:")
    print_cases(cases)
    print(f"Output directory: {output_dir}")
    repos = (legacy_repo, v2_repo)
    configurations = {
        "legacy": legacy_repo,
        "v2": v2_repo,
    }
    order = ("legacy", "v2") if args.order == "legacy-first" else ("v2", "legacy")
    return_codes = {}
    for variant in order:
        return_codes[variant] = run_worker(variant, configurations[variant], cases, args, output_dir, repos)

    legacy = load_result(output_dir / "legacy.json")
    v2 = load_result(output_dir / "v2.json")
    rows = comparison_rows(cases, legacy, v2)
    write_csv(output_dir / "comparison.csv", rows)
    summary = render_summary(cases, rows, legacy, v2, args, output_dir)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)

    has_worker_error = any(code != 0 for code in return_codes.values())
    has_v2_failure = any(row["v2_status"] != "PASS" for row in rows)
    unexpected_legacy_failure = any(row["expected_legacy_correct"] and row["legacy_status"] != "PASS" for row in rows)
    return 1 if has_worker_error or has_v2_failure or unexpected_legacy_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
