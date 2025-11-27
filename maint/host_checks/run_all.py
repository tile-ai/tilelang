import sys
import subprocess
from pathlib import Path


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
        )

        # Save logs
        (logs_dir / f"{name}.out").write_text(proc.stdout)
        (logs_dir / f"{name}.err").write_text(proc.stderr)

        out = (proc.stdout or "") + (proc.stderr or "")
        if "[SKIP]" in out:
            status = "SKIP"
        elif proc.returncode != 0:
            status = "PASS"  # error reproduced as expected
        else:
            status = "FAIL"  # no error observed

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
    sys.exit(1 if counts.get("FAIL", 0) else 0)


if __name__ == "__main__":
    main()
