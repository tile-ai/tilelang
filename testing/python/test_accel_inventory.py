"""Diagnostic: report the accelerator inventory of the CI runner."""

import subprocess


def _run(cmd: str) -> str:
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # pragma: no cover - diagnostic helper
        return f"<{cmd!r} failed: {exc}>"


def test_accelerator_inventory() -> None:
    report = "\n".join(
        [
            "=== accelerator inventory ===",
            _run("nvidia-smi -L"),
            _run("nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv,noheader"),
            _run("hostname"),
            _run("id"),
            _run("rocm-smi 2>/dev/null || true"),
            "=== end accelerator inventory ===",
        ]
    )
    print(report)
    raise AssertionError(report)
