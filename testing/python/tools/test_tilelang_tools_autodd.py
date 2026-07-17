import subprocess
import sys
from pathlib import Path


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
