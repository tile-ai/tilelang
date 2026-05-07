"""Locate pip-installed CUDA toolkit and prepare it for CMake consumption.

Used by cmake/FindPipCUDAToolkit.cmake via ``execute_process``.
Outputs a JSON object with paths on success, exits with code 1 on failure.

Usage:
    python find_pip_cuda.py              # auto-detect from current env
    python find_pip_cuda.py /path/to/cu13  # use explicit path, just prepare it
    python find_pip_cuda.py /path/to/site-packages
    python find_pip_cuda.py /path/to/venv
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import site
import subprocess
import sys


def _nvcc_name() -> str:
    return "nvcc.exe" if os.name == "nt" else "nvcc"


def _find_nvcc(cu_dir: pathlib.Path) -> pathlib.Path | None:
    candidate = cu_dir / "bin" / _nvcc_name()
    if candidate.is_file():
        return candidate

    fallback = cu_dir / "bin" / "nvcc"
    if fallback.is_file():
        return fallback

    fallback_exe = cu_dir / "bin" / "nvcc.exe"
    if fallback_exe.is_file():
        return fallback_exe

    return None


def _candidate_cu_dirs(nvidia_dir: pathlib.Path):
    if not nvidia_dir.is_dir():
        return []

    return sorted(
        (d for d in nvidia_dir.iterdir() if d.is_dir() and d.name.startswith("cu") and d.name[2:].isdigit()),
        key=lambda d: int(d.name[2:]),
    )


def _normalize_explicit_path(path: pathlib.Path) -> pathlib.Path | None:
    path = path.resolve()

    if _find_nvcc(path):
        return path

    if path.name == "nvidia":
        cu_dirs = _candidate_cu_dirs(path)
        return cu_dirs[-1] if cu_dirs else None

    nvidia_dir = path / "nvidia"
    cu_dirs = _candidate_cu_dirs(nvidia_dir)
    if cu_dirs:
        return cu_dirs[-1]

    if path.name == "site-packages":
        cu_dirs = _candidate_cu_dirs(path / "nvidia")
        return cu_dirs[-1] if cu_dirs else None

    if os.name == "nt":
        for candidate in (path / "Lib" / "site-packages", path / ".venv" / "Lib" / "site-packages"):
            cu_dirs = _candidate_cu_dirs(candidate / "nvidia")
            if cu_dirs:
                return cu_dirs[-1]
    else:
        for candidate in sorted((path / "lib").glob("python*/site-packages")):
            cu_dirs = _candidate_cu_dirs(candidate / "nvidia")
            if cu_dirs:
                return cu_dirs[-1]

    return None


def _candidate_nvidia_dirs():
    seen: set[pathlib.Path] = set()

    def add(path: pathlib.Path):
        with contextlib.suppress(OSError):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved

    try:
        import nvidia
    except ImportError:
        pass
    else:
        for candidate in add(pathlib.Path(nvidia.__path__[0])):
            yield candidate

    for entry in sys.path:
        if not entry:
            continue
        for candidate in add(pathlib.Path(entry) / "nvidia"):
            yield candidate

    with contextlib.suppress(Exception):
        for entry in site.getsitepackages():
            for candidate in add(pathlib.Path(entry) / "nvidia"):
                yield candidate

    with contextlib.suppress(Exception):
        usersite = site.getusersitepackages()
        if usersite:
            for candidate in add(pathlib.Path(usersite) / "nvidia"):
                yield candidate


def _find_cu_dir():
    """Find the nvidia/cu<ver> directory from pip-installed CUDA packages."""
    for nvidia_dir in _candidate_nvidia_dirs():
        cu_dirs = _candidate_cu_dirs(nvidia_dir)
        for cu_dir in reversed(cu_dirs):
            if _find_nvcc(cu_dir):
                return cu_dir
    return None


def _ensure_lib_symlinks(cu_dir):
    """Create Linux symlinks that CMake / nvcc expect but pip packages omit."""
    if os.name == "nt":
        return

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
    """Create a minimal Linux libcuda.so stub for build-time -lcuda linking."""
    if os.name == "nt":
        return

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


def _library_dir(cu_dir: pathlib.Path) -> pathlib.Path:
    if os.name == "nt":
        return cu_dir / "lib" / "x64"
    lib64 = cu_dir / "lib64"
    if lib64.is_dir():
        return lib64
    return cu_dir / "lib"


def main():
    if len(sys.argv) > 1:
        cu_dir = _normalize_explicit_path(pathlib.Path(sys.argv[1]))
    else:
        cu_dir = _find_cu_dir()

    if cu_dir is None:
        sys.exit(1)

    nvcc = _find_nvcc(cu_dir)
    if nvcc is None:
        sys.exit(1)

    _ensure_lib_symlinks(cu_dir)
    _ensure_cuda_stub(cu_dir)

    print(
        json.dumps(
            {
                "nvcc": str(nvcc),
                "root": str(cu_dir),
                "library_dir": str(_library_dir(cu_dir)),
            }
        )
    )


if __name__ == "__main__":
    main()
