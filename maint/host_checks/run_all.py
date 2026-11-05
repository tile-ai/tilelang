import json
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPTS = (
    "01_num_args_mismatch.py",
    "02_pointer_type_error.py",
    "03_ndim_mismatch.py",
    "04_dtype_mismatch.py",
    "05_shape_mismatch.py",
    "06_strides_mismatch.py",
    "07_device_type_mismatch.py",
    "08_device_id_mismatch.py",
    "09_null_data_pointer.py",
    "10_scalar_type_mismatch.py",
)
DEVICE_ID_SKIP = "[SKIP] Need at least 2 CUDA devices to reproduce device_id mismatch."
_MAIN_SIGNATURE = r"(?:__tl_output_(?:allocate|compute)_)?main\([^`\n]*\)"
_SCALAR_SIGNATURE = r"scalar_check\([^`\n]*\)"


def _native_message(problem, expected, signature=_MAIN_SIGNATURE):
    return re.escape(problem) + r" when calling:\n  `" + signature + r"`,\n  expected " + re.escape(expected)


# Public exception classes come from the original TileLang checks. These
# native messages identify the particular argument contract, not any failure
# that happened while importing or compiling the kernel.
EXPECTED_ERRORS = {
    SCRIPTS[0]: ("ValueError", re.escape("Kernel expected 2 inputs, but 1 are provided.")),
    SCRIPTS[1]: ("RuntimeError", _native_message("Mismatched type on argument #0", "Tensor")),
    SCRIPTS[2]: ("RuntimeError", _native_message("Mismatched A.ndim on argument #0", "2")),
    SCRIPTS[3]: ("RuntimeError", _native_message("Mismatched A.dtype on argument #0", "float16")),
    SCRIPTS[4]: ("RuntimeError", _native_message("Invalid A.shape[1] on argument #0", "128")),
    SCRIPTS[5]: ("RuntimeError", _native_message("Mismatched A.strides on argument #0", "to be compact array")),
    SCRIPTS[6]: (
        "RuntimeError",
        re.escape(
            "Check failed: anchor.device().device_type == selected->device_type (1 vs. 2) : "
            "Allocator anchor device type does not match the kernel"
        ),
    ),
    SCRIPTS[7]: ("RuntimeError", _native_message("Mismatched B.device_id on argument #1", "to match A.device_id")),
    # 09 supplies None. It does not construct a nonempty DLTensor with NULL data.
    SCRIPTS[8]: ("RuntimeError", _native_message("Mismatched type on argument #0", "Tensor")),
    # The first call raises; the later bool mismatch is not reached by 10.
    SCRIPTS[9]: ("RuntimeError", _native_message("Mismatched type on argument #0", "int", _SCALAR_SIGNATURE)),
}


def _execute_child(script, outcome_path):
    """Record the actual terminal exception, without parsing printed source."""
    root = Path(__file__).resolve().parent
    if script.parent != root or script.name not in SCRIPTS:
        raise ValueError("Unknown host-check script")
    sys.argv = [str(script)]
    sys.path[0] = str(root)
    outcome = {"schema": 1, "script": script.name}
    try:
        runpy.run_path(str(script), run_name="__main__")
    except BaseException as exc:
        outcome.update(
            outcome="exception",
            exception={"module": type(exc).__module__, "type": type(exc).__qualname__, "message": str(exc)},
        )
        outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
        raise
    else:
        outcome["outcome"] = "returned"
        outcome_path.write_text(json.dumps(outcome), encoding="utf-8")


def _read_outcome(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def classify(name, returncode, stdout, stderr, outcome):
    if name not in EXPECTED_ERRORS:
        return "FAIL", "unknown script"
    if not isinstance(outcome, dict) or outcome.get("schema") != 1 or outcome.get("script") != name:
        return "FAIL", "child did not report a valid terminal outcome"

    skip_lines = [line for line in stdout.splitlines() if "[SKIP]" in line]
    if skip_lines or "[SKIP]" in stderr:
        if (
            name == SCRIPTS[7]
            and returncode == 0
            and outcome.get("outcome") == "returned"
            and skip_lines == [DEVICE_ID_SKIP]
            and stdout.rstrip("\r\n").splitlines()[-1] == DEVICE_ID_SKIP
            and "[SKIP]" not in stderr
        ):
            return "SKIP", "fewer than two CUDA devices"
        return "FAIL", "unexpected skip marker or failed skip process"

    if returncode != 1 or outcome.get("outcome") != "exception":
        return "FAIL", "expected an uncaught argument error with exit code 1"
    error = outcome.get("exception")
    expected_type, message_pattern = EXPECTED_ERRORS[name]
    if not isinstance(error, dict) or not isinstance(error.get("message"), str):
        return "FAIL", "child exception record is incomplete"
    if error.get("module") != "builtins" or error.get("type") != expected_type:
        return "FAIL", f"expected {expected_type}, got {error.get('module')}.{error.get('type')}"
    if re.fullmatch(message_pattern, error["message"]) is None:
        return "FAIL", f"{expected_type} did not match this script's argument contract"
    return "PASS", "expected argument error reproduced"


def main():
    root = Path(__file__).resolve().parent
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    results = []
    for name in SCRIPTS:
        script_path = root / name
        if not script_path.is_file():
            results.append((name, "MISSING", None))
            print(f"[MISSING] {name}")
            continue

        print(f"\n=== Running {name} ===")
        # A fresh private result path prevents stale logs or source text printed
        # by 04 from supplying exception evidence for this child process.
        with tempfile.TemporaryDirectory(prefix="host-check-", dir=logs_dir) as scratch:
            outcome_path = Path(scratch) / "outcome.json"
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", str(script_path), str(outcome_path)],
                cwd=str(root),
                capture_output=True,
                text=True,
            )
            outcome = _read_outcome(outcome_path)
        (logs_dir / f"{name}.out").write_text(proc.stdout, encoding="utf-8")
        (logs_dir / f"{name}.err").write_text(proc.stderr, encoding="utf-8")
        status, reason = classify(name, proc.returncode, proc.stdout, proc.stderr, outcome)
        (logs_dir / f"{name}.result.json").write_text(
            json.dumps({"status": status, "reason": reason, "returncode": proc.returncode, "outcome": outcome}, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append((name, status, proc.returncode))
        print(f"[{status}] {name} (rc={proc.returncode}): {reason}")

    print("\n=== Summary ===")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "MISSING": 0}
    for name, status, _ in results:
        counts[status] += 1
        print(f"{status:7} {name}")
    print("\nTotals:")
    for status in counts:
        print(f"  {status:7}: {counts[status]}")
    return 1 if counts["FAIL"] or counts["MISSING"] else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        _execute_child(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
    elif len(sys.argv) == 1:
        sys.exit(main())
    else:
        sys.exit("Usage: python run_all.py")
