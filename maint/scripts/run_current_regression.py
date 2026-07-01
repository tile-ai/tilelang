from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from collections.abc import Sequence

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover
    tabulate = None  # type: ignore

from regression_all import _RESULTS_JSON_PREFIX
from regression_all import discover_benchmarks
from regression_all import select_benchmarks


DEFAULT_CACHE_DIR = Path.home() / ".tilelang" / "perf-regression"
SCHEMA_VERSION = 1


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _run_git(args: Sequence[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True).strip()


def _repo_root() -> Path:
    return Path(_run_git(["rev-parse", "--show-toplevel"], Path.cwd()))


def _hash_text(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def _git_state(repo_root: Path) -> dict[str, object]:
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    status = _run_git(["status", "--porcelain=v1"], repo_root)
    dirty = bool(status)
    dirty_hash = _hash_text(status) if dirty else None
    commit_key = f"{commit}-dirty-{dirty_hash}" if dirty_hash else commit
    return {
        "commit": commit,
        "commit_key": commit_key,
        "branch": branch,
        "dirty": dirty,
        "dirty_hash": dirty_hash,
    }


def _environment_metadata() -> dict[str, object]:
    meta: dict[str, object] = {
        "platform": sys.platform,
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }

    try:
        import tilelang

        meta["tilelang"] = getattr(tilelang, "__version__", "unknown")
    except Exception as exc:
        meta["tilelang_error"] = repr(exc)

    try:
        import torch

        meta["torch"] = getattr(torch, "__version__", "unknown")
        meta["torch_cuda"] = getattr(torch.version, "cuda", None)
        meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            meta["cuda_device_index"] = device_index
            meta["cuda_device_name"] = torch.cuda.get_device_name(device_index)
            meta["cuda_device_capability"] = ".".join(str(part) for part in torch.cuda.get_device_capability(device_index))
            meta["cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:
        meta["torch_error"] = repr(exc)

    return meta


def _environment_key(metadata: dict[str, object]) -> str:
    key_fields = {
        "platform": metadata.get("platform"),
        "machine": metadata.get("machine"),
        "python": metadata.get("python"),
        "tilelang": metadata.get("tilelang"),
        "torch": metadata.get("torch"),
        "torch_cuda": metadata.get("torch_cuda"),
        "cuda_device_name": metadata.get("cuda_device_name"),
        "cuda_device_capability": metadata.get("cuda_device_capability"),
    }
    return _hash_text(json.dumps(key_fields, sort_keys=True), length=16)


def _cache_path(cache_dir: Path, env_key: str, commit_key: str) -> Path:
    return cache_dir / env_key / f"{commit_key}.json"


def _empty_cache(git_state: dict[str, object], env_meta: dict[str, object], env_key: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "git": git_state,
        "environment_key": env_key,
        "environment": env_meta,
        "results": {},
        "cases": {},
        "result_cases": {},
        "failures": {},
    }


def _load_cache(path: Path, git_state: dict[str, object], env_meta: dict[str, object], env_key: str) -> dict[str, object]:
    if not path.exists():
        return _empty_cache(git_state, env_meta, env_key)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return _empty_cache(git_state, env_meta, env_key)
    if data.get("schema_version") != SCHEMA_VERSION:
        return _empty_cache(git_state, env_meta, env_key)
    data.setdefault("results", {})
    data.setdefault("cases", {})
    data.setdefault("result_cases", {})
    data.setdefault("failures", {})
    return data


def _save_cache(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _now_iso()
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def _selected_case_ids(examples_root: str | os.PathLike[str] | None, filters: Sequence[str]) -> list[str]:
    selected = select_benchmarks(discover_benchmarks(examples_root), filters)
    case_ids: list[str] = []
    for bench_file, cases in selected:
        listed_cases = bench_file.cases if cases is None else cases
        if not listed_cases:
            case_ids.append(bench_file.rel_path)
            continue
        case_ids.extend(case.id for case in listed_cases)
    return case_ids


def _case_is_cached(cache: dict[str, object], case_id: str) -> bool:
    cases = cache.get("cases", {})
    results = cache.get("results", {})
    if not isinstance(cases, dict) or not isinstance(results, dict):
        return False
    result_names = cases.get(case_id)
    if not isinstance(result_names, list) or not result_names:
        return False
    return all(isinstance(name, str) and name in results for name in result_names)


def _cached_results_for_cases(cache: dict[str, object], case_ids: Sequence[str]) -> dict[str, float]:
    cases = cache.get("cases", {})
    results = cache.get("results", {})
    if not isinstance(cases, dict) or not isinstance(results, dict):
        return {}

    selected_results: dict[str, float] = {}
    for case_id in case_ids:
        result_names = cases.get(case_id, [])
        if not isinstance(result_names, list):
            continue
        for name in result_names:
            if not isinstance(name, str):
                continue
            item = results.get(name)
            if isinstance(item, dict) and "latency" in item:
                selected_results[name] = float(item["latency"])
    return selected_results


def _cached_failures_for_cases(cache: dict[str, object], case_ids: Sequence[str]) -> dict[str, object]:
    failures = cache.get("failures", {})
    if not isinstance(failures, dict):
        return {}
    return {case_id: failures[case_id] for case_id in case_ids if case_id in failures and not _case_is_cached(cache, case_id)}


def _parse_rich_json(output: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        if line.startswith(_RESULTS_JSON_PREFIX):
            payload = line[len(_RESULTS_JSON_PREFIX) :].strip()
            parsed = json.loads(payload)
            if not isinstance(parsed, dict) or "results" not in parsed:
                raise RuntimeError("regression_all.py did not emit rich JSON results")
            return parsed
    raise RuntimeError("regression_all.py did not emit a JSON result marker")


def _tail_text(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _run_regression_case(
    case_id: str,
    examples_root: str | os.PathLike[str] | None,
    repo_root: Path,
    use_installed_package: bool,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    script = Path(__file__).with_name("regression_all.py")
    cmd = [sys.executable, str(script), "--format", "rich-json", case_id]
    if examples_root is not None:
        cmd.extend(["--examples-root", str(examples_root)])

    child_env = os.environ.copy()
    if not use_installed_package:
        pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = str(repo_root) if not pythonpath else str(repo_root) + os.pathsep + pythonpath

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_env,
    )
    stdout_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_lines.append(line)
        if not line.startswith(_RESULTS_JSON_PREFIX):
            print(line, end="", flush=True)
    proc.wait()
    stdout = "".join(stdout_lines)
    if proc.returncode != 0:
        return None, {
            "status": "error",
            "case": case_id,
            "returncode": proc.returncode,
            "command": cmd,
            "output_tail": _tail_text(stdout),
            "updated_at": _now_iso(),
        }
    try:
        return _parse_rich_json(stdout), None
    except Exception as exc:
        return None, {
            "status": "parse_error",
            "case": case_id,
            "returncode": proc.returncode,
            "command": cmd,
            "error": repr(exc),
            "output_tail": _tail_text(stdout),
            "updated_at": _now_iso(),
        }


def _merge_run_into_cache(cache: dict[str, object], run_data: dict[str, object]) -> None:
    cache_results = cache.setdefault("results", {})
    cache_cases = cache.setdefault("cases", {})
    cache_result_cases = cache.setdefault("result_cases", {})
    cache_failures = cache.setdefault("failures", {})
    if (
        not isinstance(cache_results, dict)
        or not isinstance(cache_cases, dict)
        or not isinstance(cache_result_cases, dict)
        or not isinstance(cache_failures, dict)
    ):
        raise RuntimeError("Invalid cache structure")

    run_results = run_data.get("results", {})
    if not isinstance(run_results, dict):
        raise RuntimeError("Invalid run results")
    for name, latency in run_results.items():
        cache_results[str(name)] = {"latency": float(latency), "updated_at": _now_iso()}

    run_cases = run_data.get("cases", {})
    if isinstance(run_cases, dict):
        for case_id, result_names in run_cases.items():
            if isinstance(result_names, list):
                cache_cases[str(case_id)] = [str(name) for name in result_names]
                cache_failures.pop(str(case_id), None)

    run_result_cases = run_data.get("result_cases", {})
    if isinstance(run_result_cases, dict):
        for name, case_id in run_result_cases.items():
            cache_result_cases[str(name)] = str(case_id)


def _record_failure_in_cache(cache: dict[str, object], case_id: str, failure: dict[str, object]) -> None:
    failures = cache.setdefault("failures", {})
    if not isinstance(failures, dict):
        raise RuntimeError("Invalid cache structure")
    failures[case_id] = failure


def _format_markdown(results: dict[str, float], failures: dict[str, object] | None = None) -> str:
    rows = [[name, latency] for name, latency in sorted(results.items())]
    headers = ["Name", "Latency (ms)"]
    if tabulate is not None:
        text = tabulate(rows, headers=headers, tablefmt="github", stralign="left", numalign="decimal") + "\n"
    else:
        lines = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
        lines.extend(f"| {name} | {latency} |" for name, latency in rows)
        text = "\n".join(lines) + "\n"

    if failures:
        text += "\nFailed cases\n\n"
        for case_id, item in sorted(failures.items()):
            returncode = item.get("returncode") if isinstance(item, dict) else None
            status = item.get("status", "error") if isinstance(item, dict) else "error"
            text += f"- {case_id} ({status}, returncode={returncode})\n"
    return text


def _write_output(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current checkout TileLang performance regression with result cache.")
    parser.add_argument("patterns", nargs="*", help="Optional file/case filters, e.g. gemm flash_attention")
    parser.add_argument("-k", "--filter", action="append", default=[], help="Additional file/case filter")
    parser.add_argument("--examples-root", default=None, help="Examples directory to scan")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Perf result cache directory")
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write perf result cache")
    parser.add_argument("--refresh", action="store_true", help="Rerun selected cases and overwrite cached results")
    parser.add_argument("--list", action="store_true", help="List selected cases and cache status without running")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any selected case fails")
    parser.add_argument(
        "--use-installed-package",
        action="store_true",
        help="Do not prepend the repo root to child PYTHONPATH; use the package installed in the active environment.",
    )
    parser.add_argument("--output", default=None, help="Write final JSON payload to this path")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="Final stdout format")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    filters = [*args.patterns, *args.filter]

    selected_case_ids = _selected_case_ids(args.examples_root, filters)
    if not selected_case_ids:
        raise RuntimeError("No regression benchmark cases selected")

    if args.list:
        for case_id in selected_case_ids:
            print(case_id)
        print(f"cache_dir\t{Path(args.cache_dir).expanduser()}")
        return

    repo_root = _repo_root()
    git_state = _git_state(repo_root)
    env_meta = _environment_metadata()
    env_key = _environment_key(env_meta)
    cache_file = _cache_path(Path(args.cache_dir).expanduser(), env_key, str(git_state["commit_key"]))
    cache = _empty_cache(git_state, env_meta, env_key) if args.no_cache else _load_cache(cache_file, git_state, env_meta, env_key)

    if args.no_cache or args.refresh:
        missing_case_ids = selected_case_ids
    else:
        missing_case_ids = [case_id for case_id in selected_case_ids if not _case_is_cached(cache, case_id)]

    if missing_case_ids:
        print(f"Running {len(missing_case_ids)} missing/refresh case(s)")
        if args.no_cache:
            cache = _empty_cache(git_state, env_meta, env_key)
        for idx, case_id in enumerate(missing_case_ids, 1):
            print(f"\n[{idx}/{len(missing_case_ids)}] {case_id}")
            run_data, failure = _run_regression_case(case_id, args.examples_root, repo_root, args.use_installed_package)
            if run_data is not None:
                _merge_run_into_cache(cache, run_data)
            elif failure is not None:
                _record_failure_in_cache(cache, case_id, failure)
                print(f"FAILED: {case_id}")
            if not args.no_cache:
                _save_cache(cache_file, cache)
    else:
        print(f"Using cached results from {cache_file}")

    final_results = _cached_results_for_cases(cache, selected_case_ids)
    final_failures = _cached_failures_for_cases(cache, selected_case_ids)
    succeeded_case_count = sum(1 for case_id in selected_case_ids if _case_is_cached(cache, case_id))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "cache_file": None if args.no_cache else str(cache_file),
        "cache_used": not args.no_cache,
        "refreshed": bool(args.refresh),
        "git": git_state,
        "environment_key": env_key,
        "environment": env_meta,
        "selected_cases": selected_case_ids,
        "results": final_results,
        "failures": final_failures,
        "summary": {
            "selected": len(selected_case_ids),
            "succeeded_cases": succeeded_case_count,
            "failed_cases": len(final_failures),
            "result_count": len(final_results),
        },
    }

    if args.output:
        _write_output(Path(args.output), payload)

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_markdown(final_results, final_failures), end="")

    if args.fail_on_error and final_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
