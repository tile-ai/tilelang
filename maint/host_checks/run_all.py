import re
import sys
import subprocess
from pathlib import Path


# Signatures are the validation failures documented in tensor_checks.md. Other
# failures (compiler errors, crashes, unrelated exceptions) must not count as PASS.
EXPECTED_ERRORS = {
    "01_num_args_mismatch.py": (
        r"num_args should|expected \d+ (?:input )?arguments?"
        r"|Kernel expected \d+ inputs, but \d+ are provided\."
    ),
    "02_pointer_type_error.py": (
        r"Expect (?:arg\[\d+\]|buffer \S+) to be (?:pointer|tensor)"
        r"|kernel \S+ input \S+ expected pointer or tensor handle"
    ),
    "03_ndim_mismatch.py": r"ndim is expected|mismatched ndim|ndim mismatch, expected",
    "04_dtype_mismatch.py": r"dtype is expected|incompatible dtype|dtype mismatch, expected",
    "05_shape_mismatch.py": (r"shape\[\d+\].*(?:unsatisfied constraint|violates packed ABI constraint)"),
    "06_strides_mismatch.py": (
        r"strides\[\d+\].*(?:unsatisfied constraint|violates packed ABI constraint)"
        r"|\S+\.strides: expected to be compact array, but got non-compact strides"
    ),
    "07_device_type_mismatch.py": r"device_type mismatch",
    "08_device_id_mismatch.py": r"device_id.*(?:unsatisfied constraint|violates packed ABI constraint)",
    "09_null_data_pointer.py": r"non-NULL (?:data )?pointer|Expect (?:arg\[\d+\]|buffer \S+) to be (?:pointer|tensor)",
    "10_scalar_type_mismatch.py": (
        r"Expect arg\[\d+\] to be (?:int|boolean)"
        r"|kernel \S+ scalar \S+ expected (?:integer|boolean)"
    ),
}


def classify_result(name, proc):
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return "SKIP" if any(line.startswith("[SKIP]") for line in output.splitlines()) else "FAIL"

    # Windows NTSTATUS exception codes are sometimes returned as large positive
    # integers instead of negative signal-style codes.
    if proc.returncode < 0 or 0xC0000000 <= proc.returncode <= 0xFFFFFFFF:
        return "FAIL"
    stderr = proc.stderr or ""
    if "ModuleNotFoundError:" in stderr or "ImportError:" in stderr:
        return "FAIL"

    # Check the final exception, not a warning or diagnostic earlier in stderr.
    last_line = stderr.strip().splitlines()[-1] if stderr.strip() else ""
    if last_line in (
        "RuntimeError: CUDA is not available",
        "RuntimeError: CUDA is not available; cannot build CUDA kernel for host-check repros.",
    ):
        return "SKIP"
    # The argument-count check raises ValueError; unrelated compiler TypeErrors
    # must not pass merely because they mention an expected argument count.
    if name == "01_num_args_mismatch.py" and not last_line.startswith("ValueError: "):
        return "FAIL"
    return "PASS" if re.search(EXPECTED_ERRORS[name], last_line) else "FAIL"


def main():
    root = Path(__file__).resolve().parent
    scripts = [
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
    ]

    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    results = []
    for name in scripts:
        script_path = root / name
        if not script_path.exists():
            results.append((name, "MISSING", 0))
            print(f"[MISSING] {name}")
            continue

        print(f"\n=== Running {name} ===")
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(root),
            capture_output=True,
            text=True,
            errors="replace",
        )

        # Save logs
        (logs_dir / f"{name}.out").write_text(proc.stdout, encoding="utf-8")
        (logs_dir / f"{name}.err").write_text(proc.stderr, encoding="utf-8")

        status = classify_result(name, proc)

        results.append((name, status, proc.returncode))
        print(f"[{status}] {name} (rc={proc.returncode})")

    # Summary
    print("\n=== Summary ===")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "MISSING": 0}
    for name, status, _ in results:
        counts[status] = counts.get(status, 0) + 1
        print(f"{status:7} {name}")

    print("\nTotals:")
    for k in ("PASS", "FAIL", "SKIP", "MISSING"):
        print(f"  {k:7}: {counts.get(k, 0)}")

    # Exit non-zero if any FAIL
    sys.exit(1 if counts["FAIL"] or counts["MISSING"] else 0)


if __name__ == "__main__":
    main()
