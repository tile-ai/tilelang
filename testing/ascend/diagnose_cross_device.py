"""Temporary CI diagnostics for cross-device failures observed in PR #4114.

Run after the normal AscendC suite. Each comparison uses a fresh process on
the same runner; failed/unsupported operations are recorded, never emulated.
"""

import argparse
from contextlib import suppress
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
AUTOTUNE = "testing/ascend/autotune/test_tilelang_ascend_autotune_set_inputs.py::test_set_inputs_supplies_captured_inputs_on_current_device"
CAST = "testing/ascend/language/test_tilelang_ascend_simtvf_cast.py::test_simtvf_cast"
CASTS = [f"{CAST}[{case}]" for case in ("16-fp32->half", "16-half->fp32", "8-fp32->e4m3")]
CASES = {
    "A": ("standalone cast", [CASTS[0]]),
    "B": ("autotune including cross-device probe, then casts", [AUTOTUNE, *CASTS]),
    "C": ("same sequence without the cross-device case", [f"{AUTOTUNE}[npu]", f"{AUTOTUNE}[cpu]", *CASTS]),
    "D": ("direct ACL IPC event creation and cross-device wait", []),
}


def emit(directory, phase, **fields):
    record = {"time": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "phase": phase, **fields}
    with (directory / "events.jsonl").open("a") as output:
        print(json.dumps(record), file=output, flush=True)
    print(json.dumps(record), flush=True)


def loaded_libraries():
    names = {"libascendcl.so", "libruntime.so", "libacl_rt.so", "libacl_rt_impl.so", "libhccl.so", "libhcomm.so"}
    return sorted(
        {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if line.split()[-1].split("/")[-1] in names}
    )


def environment(torch, torch_npu):
    libraries = loaded_libraries()
    components = {}
    roots = {Path(os.environ[key]) for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if os.environ.get(key)}
    for library in libraries:
        roots.update(Path(library).resolve().parents)
    for root in roots:
        for component in ("runtime", "hccl", "hcomm", "opbase", "ops_math", "bisheng-compiler"):
            path = root / "share/info" / component / "version.info"
            if path.is_file():
                components[str(path)] = path.read_text()
    hashes = {}
    for path in libraries:
        with Path(path).open("rb") as source:
            hashes[path] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "torch_npu": torch_npu.__version__,
        "torch_npu_git": getattr(torch_npu.version, "git_version", None),
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
        "libraries": hashes,
        "components": components,
    }


def health(directory, torch, device):
    emit(directory, "health_begin", device=device)
    try:
        value = torch.arange(4, dtype=torch.float32, device=f"npu:{device}")
        torch.npu.synchronize(device)
        emit(directory, "health_pass", values=value.cpu().tolist())
        return True
    except Exception:
        emit(directory, "health_failed", traceback=traceback.format_exc())
        return False


class ACLFailure(RuntimeError):
    def __init__(self, stage, code):
        super().__init__(f"{stage} returned {code}")
        self.stage = stage
        self.code = code


def ipc_probe(directory, torch, target):
    if torch.npu.device_count() < 2:
        return "skipped"
    source = next(device for device in range(torch.npu.device_count()) if device != target)
    library = next(path for path in loaded_libraries() if Path(path).name == "libascendcl.so")
    acl = ctypes.CDLL(library)

    def bind(name, arguments, result=ctypes.c_int):
        function = getattr(acl, name)
        function.argtypes = arguments
        function.restype = result
        return function

    pointer = ctypes.c_void_p
    set_device = bind("aclrtSetDevice", [ctypes.c_int])
    create = bind("aclrtCreateEventExWithFlag", [ctypes.POINTER(pointer), ctypes.c_uint])
    destroy = bind("aclrtDestroyEvent", [pointer])
    record = bind("aclrtRecordEvent", [pointer, pointer])
    wait = bind("aclrtStreamWaitEvent", [pointer, pointer])
    synchronize = bind("aclrtSynchronizeStream", [pointer])
    message = bind("aclGetRecentErrMsg", [], ctypes.c_char_p)

    def call(stage, function, *args):
        emit(directory, stage + "_begin", source=source, target=target)
        code = function(*args)
        detail = message() if code else None
        emit(directory, stage + "_return", code=code, error=detail.decode(errors="replace") if detail else None)
        if code:
            raise ACLFailure(stage, code)

    event = pointer()
    created = False
    try:
        source_stream = pointer(torch.npu.current_stream(source).npu_stream)
        target_stream = pointer(torch.npu.current_stream(target).npu_stream)
        call("set_source", set_device, source)
        # Check a normal event before trying the real ACL_EVENT_IPC flag.
        call("create_normal_event", create, ctypes.byref(event), 1)
        call("destroy_normal_event", destroy, event)
        event = pointer()
        call("create_ipc_event", create, ctypes.byref(event), 64)
        created = True
        call("record_source", record, event, source_stream)
        call("set_target", set_device, target)
        call("wait_target", wait, target_stream, event)
        call("synchronize_target", synchronize, target_stream)
        return "passed"
    except ACLFailure as error:
        emit(directory, "ipc_failed", stage=error.stage, code=error.code, traceback=traceback.format_exc())
        return "unsupported" if error.stage == "create_ipc_event" and error.code == 207000 else "failed"
    finally:
        # Raw ACL device switches must not leave Torch's cached device stale.
        if created:
            set_device(source)
            emit(directory, "destroy_ipc_event", code=destroy(event))
        emit(directory, "restore_target", code=set_device(target))


def run_case(case, directory):
    result = {"case": case, "pid": os.getpid(), "status": "error"}
    try:
        import torch
        import torch_npu

        (directory / "environment.json").write_text(json.dumps(environment(torch, torch_npu), indent=2))
        device = torch.npu.current_device()
        emit(directory, "device", current=device, count=torch.npu.device_count(), name=torch.npu.get_device_name(device))
        # Device initialization may load additional runtime implementations.
        (directory / "environment.json").write_text(json.dumps(environment(torch, torch_npu), indent=2))
        if case == "D":
            result["status"] = ipc_probe(directory, torch, device)
        else:
            import pytest

            result["pytest_exit_code"] = int(
                pytest.main(["-vv", "-rs", "-n", "0", "--tb=long", f"--junitxml={directory / 'pytest.xml'}", *CASES[case][1]])
            )
            result["status"] = "passed" if result["pytest_exit_code"] == 0 else "failed"
        result["health"] = health(directory, torch, device)
        if not result["health"]:
            result["status"] = "failed"
    except Exception:
        result["traceback"] = traceback.format_exc()
        emit(directory, "case_error", traceback=result["traceback"])
    finally:
        (directory / "result.json").write_text(json.dumps(result, indent=2))
    return 0 if result["status"] in ("passed", "unsupported", "skipped") else 1


def collect_plogs(directory, pids, started):
    roots = {Path.home() / "ascend/log/plog"}
    if os.environ.get("ASCEND_PROCESS_LOG_PATH"):
        roots.add(Path(os.environ["ASCEND_PROCESS_LOG_PATH"]))
    remaining = 20 * 1024 * 1024
    for root in roots:
        try:
            for path in root.rglob("*.log"):
                if remaining <= 0:
                    return
                if path.stat().st_mtime < started or not any(re.search(rf"(?<!\d){pid}(?!\d)", path.name) for pid in pids):
                    continue
                # Keep only this run's process logs; cap the tail of each file.
                size = min(path.stat().st_size, 2 * 1024 * 1024, remaining)
                with path.open("rb") as source:
                    source.seek(-size, os.SEEK_END)
                    data = source.read(size)
                target = directory / "plog"
                target.mkdir(exist_ok=True)
                (target / f"{len(list(target.iterdir()))}-{path.name}").write_bytes(data)
                remaining -= len(data)
        except OSError as error:
            emit(directory, "plog_unavailable", path=str(root), error=str(error))


def run_all(directory, timeout):
    started = time.time()
    results = []
    pids = set()
    try:
        smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
        (directory / "npu-smi.txt").write_text(smi.stdout + smi.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        (directory / "npu-smi.txt").write_text(str(error))
    for case, (description, _) in CASES.items():
        child = directory / case
        child.mkdir(exist_ok=True)
        # A killed child must not inherit a successful result from an earlier run.
        (child / "result.json").unlink(missing_ok=True)
        env = os.environ.copy()
        env["TILELANG_ASCEND_DIAGNOSTICS_DIR"] = str(child / "probes")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--case", case, "--output-dir", str(child)]
        with (child / "stdout.log").open("w") as stdout, (child / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            pids.add(process.pid)
            timed_out = False
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=3)
                # Kill any descendants left behind by pytest or a compiler.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
        try:
            result = json.loads((child / "result.json").read_text())
        except (OSError, json.JSONDecodeError):
            result = {}
        result.update(case=case, description=description, exit_code=code, timed_out=timed_out)
        if timed_out or not result.get("status"):
            result["status"] = "timeout" if timed_out else "error"
        elif code != 0 and result["status"] in ("passed", "unsupported", "skipped"):
            result["status"] = "error"
        results.append(result)
        (directory / "summary.json").write_text(json.dumps(results, indent=2))
    for path in directory.rglob("probe-*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                probe = json.loads(line)
                pids.add(probe["pid"])
                started = min(started, probe["time"])
            except (json.JSONDecodeError, KeyError):
                emit(directory, "incomplete_probe_record", path=str(path))
    collect_plogs(directory, pids, started)
    lines = ["### Ascend cross-device diagnostics", "", "| Case | Result | Exit | Health |", "| --- | --- | --- | --- |"]
    lines.extend(
        f"| {item['case']}: {item['description']} | {item['status']} | {item['exit_code']} | {item.get('health', 'unavailable')} |"
        for item in results
    )
    summary = "\n".join(lines) + "\n"
    (directory / "summary.md").write_text(summary)
    print(summary, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as output:
            output.write(summary)
    return int(any(item["status"] not in ("passed", "unsupported", "skipped") for item in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    return run_case(args.case, directory) if args.case else run_all(directory, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
