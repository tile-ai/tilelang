import subprocess
import sys
from pathlib import Path

import pytest


def test_autodd_module_help_runs_with_light_import():
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-m", "tilelang.autodd", "--help"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Delta-debug the provided Python source" in result.stdout


def test_autodd_rejects_nonpositive_jobs():
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tilelang.autodd",
            "unused.py",
            "--err-msg",
            "boom",
            "-o",
            "unused_out.py",
            "--jobs",
            "0",
        ],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2, result.stderr
    assert "--jobs must be >= 1, got 0" in result.stderr


def test_partaskmanager_rejects_nonpositive_workers():
    from tilelang.autodd import ParTaskManager

    for jobs in (0, -3):
        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            ParTaskManager(
                err_msg="boom",
                text="",
                output_file=Path("unused_out.py"),
                num_workers=jobs,
            )
