import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("existing_driver", [False, True])
def test_cuda_toolkit_preparation_does_not_fabricate_driver(tmp_path, existing_driver):
    """Preparing pip toolkit paths must not create or overwrite a driver library."""
    root = tmp_path / "cu13"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    nvcc = root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    nvcc.write_text("toolkit discovery fixture; never executed\n")
    driver = root / "lib/stubs/libcuda.so"
    original = None
    if existing_driver:
        driver.parent.mkdir()
        driver.write_bytes(b"existing provider must remain unchanged\n")
        original = hashlib.sha256(driver.read_bytes()).hexdigest()

    tool = Path(__file__).resolve().parents[3] / "cmake/find_pip_cuda.py"
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(tool), str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    assert Path(metadata["root"]) == root
    assert Path(metadata["nvcc"]) == nvcc
    if existing_driver:
        assert hashlib.sha256(driver.read_bytes()).hexdigest() == original
    else:
        assert not driver.exists(), "Toolkit preparation fabricated an incomplete libcuda.so"
    assert not (root / "lib/stubs/_stub.c").exists()
