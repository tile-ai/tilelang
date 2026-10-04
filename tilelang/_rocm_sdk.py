"""ROCm SDK discovery shared by CMake and early Python initialization.

This module uses only the standard library: native libraries and backend
registrations are not available yet when either caller discovers the SDK.
"""

import importlib.metadata
import os
from pathlib import Path
import shutil


def find_rocm_home() -> str:
    configured_sdk = os.environ.get("USE_ROCM", "")
    if configured_sdk and Path(configured_sdk).is_dir():
        return configured_sdk
    for name in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        if os.environ.get(name):
            return os.environ[name]
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        compiler = shutil.which("hipcc", path=directory)
        if not compiler:
            continue
        prefix = Path(compiler).resolve().parent.parent
        if (prefix / "include/hip/hip_runtime.h").is_file():
            return str(prefix)
    # Metadata works for both monolithic and split pip SDKs. Do not expand a
    # development archive or load a runtime as a side effect of discovery.
    for package in ("rocm-sdk-core", "rocm-sdk-devel"):
        try:
            files = importlib.metadata.files(package) or []
        except importlib.metadata.PackageNotFoundError:
            continue
        for file in files:
            if file.name in ("hipcc", "hipcc.exe"):
                prefix = Path(file.locate()).parent.parent
                if (prefix / "include/hip/hip_runtime.h").is_file():
                    return str(prefix)
    return "/opt/rocm" if Path("/opt/rocm").is_dir() else ""


if __name__ == "__main__":
    sdk = find_rocm_home()
    print(Path(sdk).as_posix() if sdk else "")
