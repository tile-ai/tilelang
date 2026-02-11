"""Locate pip-installed CUDA toolkit and prepare it for CMake consumption.

Used by cmake/FindPipCUDAToolkit.cmake via ``execute_process``.
Outputs a JSON object with paths on success, exits with code 1 on failure.

Usage:
    python find_pip_cuda.py              # auto-detect from current env
    python find_pip_cuda.py /path/to/cu13  # use explicit path, just prepare it
"""

import contextlib
import json
import pathlib
import subprocess
import sys


def _find_cu_dir():
    """Find the nvidia/cu<ver> directory from the nvidia pip package."""
    try:
        import nvidia
    except ImportError:
        return None

    nvidia_dir = pathlib.Path(nvidia.__path__[0])
    cu_dirs = sorted(
        (d for d in nvidia_dir.iterdir() if d.name[:2] == "cu" and d.name[2:].isdigit()),
        key=lambda d: int(d.name[2:]),
    )
    if not cu_dirs:
        return None
    cu_dir = cu_dirs[-1]
    if (cu_dir / "bin" / "nvcc").is_file():
        return cu_dir
    return None


def _ensure_lib_symlinks(cu_dir):
    """Create symlinks that CMake / nvcc expect but pip packages omit."""
    lib_dir = cu_dir / "lib"
    if not lib_dir.is_dir():
        return

    # nvcc expects lib64/ on 64-bit
    lib64 = cu_dir / "lib64"
    if not lib64.exists():
        with contextlib.suppress(OSError):
            lib64.symlink_to("lib")

    # CMake expects unversioned .so (e.g., libcudart.so)
    for so in lib_dir.glob("*.so.*"):
        base = lib_dir / (so.name.split(".so.")[0] + ".so")
        if not base.exists():
            with contextlib.suppress(OSError):
                base.symlink_to(so.name)


def _ensure_cuda_stub(cu_dir):
    """Create a minimal libcuda.so stub for build-time -lcuda linking."""
    stubs_dir = cu_dir / "lib" / "stubs"
    stub = stubs_dir / "libcuda.so"
    if stub.exists():
        return
    stubs_dir.mkdir(parents=True, exist_ok=True)
    src = stubs_dir / "_stub.c"
    try:
        src.write_text("void cuGetErrorString(void){}\n")
        subprocess.check_call(
            ["gcc", "-shared", "-o", str(stub), str(src)],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    finally:
        src.unlink(missing_ok=True)


def main():
    if len(sys.argv) > 1:
        # Explicit path provided — just prepare it
        cu_dir = pathlib.Path(sys.argv[1])
    else:
        # Auto-detect from current Python environment
        cu_dir = _find_cu_dir()

    if cu_dir is None or not (cu_dir / "bin" / "nvcc").is_file():
        sys.exit(1)

    _ensure_lib_symlinks(cu_dir)
    _ensure_cuda_stub(cu_dir)

    print(
        json.dumps(
            {
                "nvcc": str(cu_dir / "bin" / "nvcc"),
                "root": str(cu_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
