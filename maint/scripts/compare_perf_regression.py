from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections.abc import Mapping
from collections.abc import Sequence

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover
    tabulate = None  # type: ignore


def _load_payload(path: str | Path) -> dict[str, object]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _extract_results(payload: Mapping[str, object]) -> dict[str, float]:
    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("payload.results must be an object")

    parsed: dict[str, float] = {}
    for name, value in results.items():
        if isinstance(value, dict):
            value = value.get("latency")
        try:
            latency = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid latency for result {name!r}: {value!r}") from exc
        if math.isfinite(latency) and latency > 0.0:
            parsed[str(name)] = latency
    return parsed


def _extract_failures(payload: Mapping[str, object]) -> dict[str, object]:
    failures = payload.get("failures", {})
    if not isinstance(failures, dict):
        return {}
    return {str(k): v for k, v in failures.items()}


def _short_commit(payload: Mapping[str, object]) -> str:
    git = payload.get("git", {})
    if not isinstance(git, dict):
        return "unknown"
    commit = str(git.get("commit", "unknown"))
    dirty = bool(git.get("dirty", False))
    suffix = "-dirty" if dirty else ""
    return commit[:12] + suffix


def _format_latency(value: float) -> str:
    return f"{value:.6g}"


def _format_speedup(value: float) -> str:
    return f"{value:.4f}x"


def _format_delta(value: float) -> str:
    return f"{(value - 1.0) * 100.0:+.2f}%"


def compare_payloads(
    old_payload: Mapping[str, object],
    new_payload: Mapping[str, object],
) -> dict[str, object]:
    old_results = _extract_results(old_payload)
    new_results = _extract_results(new_payload)
    common = sorted(set(old_results) & set(new_results))

    rows: list[dict[str, object]] = []
    for name in common:
        old_latency = old_results[name]
        new_latency = new_results[name]
        speedup = old_latency / new_latency
        rows.append(
            {
                "name": name,
                "old_latency": old_latency,
                "new_latency": new_latency,
                "speedup": speedup,
                "delta_pct": (speedup - 1.0) * 100.0,
                "status": "improved" if speedup >= 1.0 else "regressed",
            }
        )
    rows.sort(key=lambda item: (float(item["speedup"]), str(item["name"])))

    return {
        "old": {
            "commit": _short_commit(old_payload),
            "result_count": len(old_results),
            "failure_count": len(_extract_failures(old_payload)),
        },
        "new": {
            "commit": _short_commit(new_payload),
            "result_count": len(new_results),
            "failure_count": len(_extract_failures(new_payload)),
        },
        "summary": {
            "common": len(common),
            "regressed": sum(1 for item in rows if item["status"] == "regressed"),
            "improved_or_equal": sum(1 for item in rows if item["status"] == "improved"),
            "only_old": len(set(old_results) - set(new_results)),
            "only_new": len(set(new_results) - set(old_results)),
        },
        "results": rows,
        "only_old": sorted(set(old_results) - set(new_results)),
        "only_new": sorted(set(new_results) - set(old_results)),
        "old_failures": _extract_failures(old_payload),
        "new_failures": _extract_failures(new_payload),
    }


def _failure_summary(failures: Mapping[str, object]) -> list[list[object]]:
    rows: list[list[object]] = []
    for case_id, item in sorted(failures.items()):
        if isinstance(item, dict):
            rows.append([case_id, item.get("status", "error"), item.get("returncode", "")])
        else:
            rows.append([case_id, "error", ""])
    return rows


def _markdown_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    if tabulate is not None:
        return tabulate(rows, headers=headers, tablefmt="github", stralign="left", numalign="decimal")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(report: Mapping[str, object], old_label: str, new_label: str) -> str:
    old = report["old"]
    new = report["new"]
    summary = report["summary"]
    assert isinstance(old, dict)
    assert isinstance(new, dict)
    assert isinstance(summary, dict)

    lines: list[str] = []
    lines.append("# Performance Regression Report")
    lines.append("")
    lines.append(f"- Baseline: {old_label} ({old.get('commit')})")
    lines.append(f"- Current: {new_label} ({new.get('commit')})")
    lines.append(f"- Common results: {summary.get('common')}")
    lines.append(f"- Regressed: {summary.get('regressed')}")
    lines.append(f"- Improved or equal: {summary.get('improved_or_equal')}")
    lines.append(f"- Only baseline: {summary.get('only_old')}")
    lines.append(f"- Only current: {summary.get('only_new')}")
    lines.append(f"- Baseline failed cases: {old.get('failure_count')}")
    lines.append(f"- Current failed cases: {new.get('failure_count')}")
    lines.append("")

    result_rows = []
    for item in report["results"]:  # type: ignore[index]
        assert isinstance(item, dict)
        speedup = float(item["speedup"])
        result_rows.append(
            [
                item["name"],
                _format_latency(float(item["old_latency"])),
                _format_latency(float(item["new_latency"])),
                _format_speedup(speedup),
                _format_delta(speedup),
                item["status"],
            ]
        )

    lines.append("## Results")
    lines.append("")
    if result_rows:
        lines.append(
            _markdown_table(
                result_rows,
                ["Result", f"{old_label} Latency", f"{new_label} Latency", "Speedup", "Delta", "Status"],
            )
        )
    else:
        lines.append("No common successful results to compare.")
    lines.append("")

    only_old = report["only_old"]
    only_new = report["only_new"]
    assert isinstance(only_old, list)
    assert isinstance(only_new, list)
    if only_old or only_new:
        lines.append("## Missing Results")
        lines.append("")
        if only_old:
            lines.append(f"Only in {old_label}:")
            lines.extend(f"- {name}" for name in only_old)
            lines.append("")
        if only_new:
            lines.append(f"Only in {new_label}:")
            lines.extend(f"- {name}" for name in only_new)
            lines.append("")

    old_failures = report["old_failures"]
    new_failures = report["new_failures"]
    assert isinstance(old_failures, dict)
    assert isinstance(new_failures, dict)
    if old_failures or new_failures:
        lines.append("## Failed Cases")
        lines.append("")
        if old_failures:
            lines.append(f"{old_label}:")
            lines.append(_markdown_table(_failure_summary(old_failures), ["Case", "Status", "Return Code"]))
            lines.append("")
        if new_failures:
            lines.append(f"{new_label}:")
            lines.append(_markdown_table(_failure_summary(new_failures), ["Case", "Status", "Return Code"]))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def draw_png(report: Mapping[str, object], output_png: str | Path) -> None:
    results = report["results"]
    assert isinstance(results, list)
    if not results:
        return

    import matplotlib.pyplot as plt

    rows = sorted(results, key=lambda item: float(item["speedup"]))  # type: ignore[index]
    names = [str(item["name"]) for item in rows]  # type: ignore[index]
    speedups = [float(item["speedup"]) for item in rows]  # type: ignore[index]
    colors = ["#2ca02c" if value >= 1.0 else "#d62728" for value in speedups]

    height = min(max(5.5, 0.35 * len(rows) + 2.0), 22.0)
    fig, ax = plt.subplots(figsize=(14.0, height))
    y = list(range(len(rows)))
    ax.barh(y, speedups, color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(1.0, linestyle="--", linewidth=1.2, color="black", alpha=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Speedup (baseline latency / current latency)")
    ax.set_title("TileLang Performance Regression")
    ax.xaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    x_min = min(speedups)
    x_max = max(speedups)
    pad = max(0.02, (x_max - x_min) * 0.12)
    ax.set_xlim(min(1.0, x_min) - pad, max(1.0, x_max) + pad)

    for idx, value in enumerate(speedups):
        label = f"{value:.3f}x ({(value - 1.0) * 100.0:+.2f}%)"
        if value >= 1.0:
            ax.text(value + 0.003, idx, label, va="center", ha="left", fontsize=9)
        else:
            ax.text(value - 0.003, idx, label, va="center", ha="right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _write_text(path: str | Path, content: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)


def _write_json(path: str | Path, content: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two TileLang performance regression JSON payloads.")
    parser.add_argument("old_json", help="Baseline JSON from run_current_regression.py")
    parser.add_argument("new_json", help="Current JSON from run_current_regression.py")
    parser.add_argument("--old-label", default="Baseline", help="Label for the baseline payload")
    parser.add_argument("--new-label", default="Current", help="Label for the current payload")
    parser.add_argument("--output-md", default=None, help="Write markdown report to this path")
    parser.add_argument("--output-json", default=None, help="Write structured comparison JSON to this path")
    parser.add_argument("--output-png", default=None, help="Write speedup plot PNG to this path")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any common result has speedup below the threshold",
    )
    parser.add_argument("--regression-threshold", type=float, default=1.0, help="Speedup threshold for regressions")
    parser.add_argument("--fail-on-failure", action="store_true", help="Exit non-zero if either payload has failures")
    parser.add_argument("--fail-on-missing", action="store_true", help="Exit non-zero if either payload has missing results")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    old_payload = _load_payload(args.old_json)
    new_payload = _load_payload(args.new_json)
    report = compare_payloads(old_payload, new_payload)
    markdown = render_markdown(report, args.old_label, args.new_label)

    if args.output_md:
        _write_text(args.output_md, markdown)
    else:
        print(markdown, end="")

    if args.output_json:
        _write_json(args.output_json, report)

    if args.output_png:
        draw_png(report, args.output_png)

    failures = []
    result_rows = report["results"]
    assert isinstance(result_rows, list)
    if args.fail_on_regression and any(float(item["speedup"]) < args.regression_threshold for item in result_rows):
        failures.append("regression")
    if args.fail_on_failure and (report["old_failures"] or report["new_failures"]):
        failures.append("failure")
    if args.fail_on_missing and (report["only_old"] or report["only_new"]):
        failures.append("missing")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
