"""Run all TileLang graph fusion operator examples in isolated processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEST_FILES = sorted(path.name for path in ROOT.glob("test_*.py"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", help="Run only files containing this substring.")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for per-case stdout/stderr logs.",
    )
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("TL_FUSION_TEST_TIMEOUT", "900")))
    args = parser.parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else ROOT / "logs" / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {log_dir}", flush=True)

    selected = [name for name in TEST_FILES if not args.case or any(pattern in name for pattern in args.case)]
    failures = []
    for name in selected:
        path = ROOT / name
        print(f"\n===== RUN {name} =====", flush=True)
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        (log_dir / f"{name}.stdout").write_text(proc.stdout)
        (log_dir / f"{name}.stderr").write_text(proc.stderr)
        if proc.returncode != 0:
            failures.append((name, proc.returncode))
            print(f"===== FAIL {name} rc={proc.returncode} =====", flush=True)
        else:
            print(f"===== PASS {name} =====", flush=True)

    if failures:
        print("\nFAILED CASES:")
        for name, rc in failures:
            print(f"  {name}: rc={rc}")
        raise SystemExit(1)

    print("\nALL FUSION OP TESTS PASSED")


if __name__ == "__main__":
    main()
