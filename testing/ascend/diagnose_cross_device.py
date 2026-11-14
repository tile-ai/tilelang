"""Collect passive diagnostics for the original Ascend cross-device tests.

The in-process snapshot reads existing Python modules and files only. The CLI
summarizes recorded probes and copies matching process logs; neither path
initializes an NPU, submits operators, or runs additional tests.
"""

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sys


@lru_cache(maxsize=32)
def _library_hash(path, size, mtime_ns):
    # The stat fields invalidate the cache if a library changes between snapshots.
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def environment():
    """Inspect the test process without importing or calling device runtimes."""
    torch = sys.modules.get("torch")
    torch_npu = sys.modules.get("torch_npu")
    names = {"libascendcl.so", "libruntime.so", "libacl_rt.so", "libacl_rt_impl.so", "libhccl.so", "libhcomm.so"}
    paths = sorted(
        {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if line.split()[-1].split("/")[-1] in names}
    )
    roots = {Path(os.environ[key]) for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if os.environ.get(key)}
    libraries = {}
    for path in paths:
        roots.update(Path(path).resolve().parents)
        try:
            stat = Path(path).stat()
            libraries[path] = {"sha256": _library_hash(path, stat.st_size, stat.st_mtime_ns), "size": stat.st_size}
        except OSError as error:
            libraries[path] = {"error": str(error)}
    components = {}
    for root in roots:
        for component in ("runtime", "hccl", "hcomm", "opbase", "ops_math", "bisheng-compiler"):
            path = root / "share/info" / component / "version.info"
            if path.is_file():
                try:
                    components[str(path)] = path.read_text()
                except OSError as error:
                    components[str(path)] = str(error)
    driver = Path("/usr/local/Ascend/driver/version.info")
    try:
        components[str(driver)] = driver.read_text()
    except OSError as error:
        components[str(driver)] = str(error)
    return {
        "pid": os.getpid(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": getattr(torch, "__version__", None),
        "torch_git": getattr(getattr(torch, "version", None), "git_version", None),
        "torch_npu": getattr(torch_npu, "__version__", None),
        "torch_npu_git": getattr(getattr(torch_npu, "version", None), "git_version", None),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
                "ASCEND_OPP_PATH",
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_VISIBLE_DEVICES",
                "ASCEND_LAUNCH_BLOCKING",
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
            )
        },
        "libraries": libraries,
        "components": components,
    }


def collect_plogs(directory, probes):
    if not probes:
        return
    pids = {probe["pid"] for probe in probes}
    started = min(probe["time"] for probe in probes)
    roots = {Path.home() / "ascend/log/plog"}
    if os.environ.get("ASCEND_PROCESS_LOG_PATH"):
        roots.add(Path(os.environ["ASCEND_PROCESS_LOG_PATH"]))
    remaining = 20 * 1024 * 1024
    count = 0
    for root in roots:
        try:
            for path in root.rglob("*.log"):
                if remaining <= 0:
                    return
                if path.stat().st_mtime < started or not any(re.search(rf"(?<!\d){pid}(?!\d)", path.name) for pid in pids):
                    continue
                # Copy only this run's process logs, capped at 2 MiB per file.
                size = min(path.stat().st_size, 2 * 1024 * 1024, remaining)
                with path.open("rb") as source:
                    source.seek(-size, os.SEEK_END)
                    data = source.read(size)
                target = directory / "plog"
                target.mkdir(exist_ok=True)
                (target / f"{count}-{path.name}").write_bytes(data)
                remaining -= len(data)
                count += 1
        except OSError as error:
            print(f"Could not collect plog from {root}: {error}", file=sys.stderr)


def summarize(directory):
    probes = []
    rows = []
    for path in sorted(directory.rglob("probe-*.jsonl")):
        records = []
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Incomplete probe record in {path}", file=sys.stderr)
        if not records:
            continue
        probes.extend(records)
        last = records[-1]
        rows.append(f"| {last['worker']} / {last['pid']} | {last['stage']} | {last['current_device']} → {last['other_device']} |")
        for record in records:
            if record.get("traceback"):
                print(f"Original cross-device exception ({path.name}):\n{record['traceback']}", flush=True)
    (directory / "summary.json").write_text(json.dumps(probes, indent=2))
    collect_plogs(directory, probes)
    lines = ["### Ascend cross-device diagnostics", "", "| Worker / PID | Last recorded stage | Devices |", "| --- | --- | --- |", *rows]
    if not rows:
        lines.extend(["", "No cross-device probe records were produced; inspect the original unit-test log."])
    lines.extend(["", "Passive collection only; no additional device operations or tests were run."])
    summary = "\n".join(lines) + "\n"
    (directory / "summary.md").write_text(summary)
    print(summary, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as output:
            output.write(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    summarize(directory)


if __name__ == "__main__":
    main()
