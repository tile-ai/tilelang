"""Windows host toolchain discovery shared by CMake and JIT (stdlib only).

This module can also run directly before TileLang's native libraries exist.
"""

from __future__ import annotations

import functools
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

_MSVC_ENV_ERROR: str | None = None


def get_env_path(compiler_env: dict[str, str]) -> str | None:
    return compiler_env.get("PATH") or compiler_env.get("Path") or compiler_env.get("path")


def _clang_cl_disabled() -> bool:
    val = os.environ.get("TILELANG_DISABLE_CLANG_CL", "")
    return val not in ("", "0", "OFF", "off", "false", "False")


def _vs_install_roots() -> list[str]:
    roots: list[str] = []
    for base in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)")):
        if not base:
            continue
        vs_root = os.path.join(base, "Microsoft Visual Studio")
        if os.path.isdir(vs_root):
            roots.append(vs_root)
    if vc_install := os.environ.get("VCINSTALLDIR"):
        # VCINSTALLDIR points to <vs_install>/VC/, so its parent is the VS install root.
        candidate = os.path.dirname(os.path.normpath(vc_install))
        if os.path.isdir(candidate):
            roots.append(candidate)
    return roots


def _candidate_clang_cl_paths(compiler_env: dict[str, str] | None) -> list[str]:
    candidates: list[str] = []

    def add(path: str | None) -> None:
        if path and os.path.exists(path) and path not in candidates:
            candidates.append(path)

    if explicit := os.environ.get("CLANG_CL"):
        add(explicit)

    if compiler_env and (vc_root := compiler_env.get("VCINSTALLDIR")):
        add(os.path.join(vc_root, "Tools", "Llvm", "x64", "bin", "clang-cl.exe"))
        add(os.path.join(vc_root, "Tools", "Llvm", "bin", "clang-cl.exe"))

    # VS-bundled LLVM (when the "C++ Clang Compiler for Windows" workload is installed).
    for vs_root in _vs_install_roots():
        for pattern in (
            os.path.join(vs_root, "*", "*", "VC", "Tools", "Llvm", "x64", "bin", "clang-cl.exe"),
            os.path.join(vs_root, "*", "*", "VC", "Tools", "Llvm", "bin", "clang-cl.exe"),
        ):
            for hit in sorted(glob.glob(pattern), reverse=True):
                add(hit)

    # Standalone LLVM installation.
    for base in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)")):
        if base:
            add(os.path.join(base, "LLVM", "bin", "clang-cl.exe"))
    for env_var in ("LLVM_HOME", "LLVM_DIR"):
        if base := os.environ.get(env_var):
            add(os.path.join(base, "bin", "clang-cl.exe"))

    # PATH lookup using the compiler's own environment when available so we
    # find the LLVM that VsDevCmd just exposed.
    env_path = get_env_path(compiler_env or {}) if compiler_env else None
    add(shutil.which("clang-cl.exe", path=env_path))
    add(shutil.which("clang-cl", path=env_path))

    return candidates


def _find_clang_cl(compiler_env: dict[str, str] | None) -> str | None:
    if _clang_cl_disabled():
        return None
    for candidate in _candidate_clang_cl_paths(compiler_env):
        return candidate
    return None


@functools.cache
def _resolve_windows_compiler(prefer_clang_cl: bool) -> tuple[str | None, str | None]:
    """Return (compiler_path, kind) where kind is 'clang-cl' or 'msvc'.

    Cached so repeated JIT compiles don't re-spawn vswhere or rescan PATH.
    """
    compiler_env = get_msvc_subprocess_env()
    if prefer_clang_cl:
        clang_cl = _find_clang_cl(compiler_env)
        if clang_cl is not None:
            return clang_cl, "clang-cl"
    cl = shutil.which("cl.exe", path=get_env_path(compiler_env or {}))
    if cl is not None:
        return cl, "msvc"
    return None, None


def get_windows_compiler() -> tuple[str | None, str | None]:
    """Resolve the preferred Windows host compiler.

    Returns ``(path, kind)`` where ``kind`` is ``"clang-cl"`` or ``"msvc"``.
    Set the ``TILELANG_DISABLE_CLANG_CL`` env var to force the legacy
    ``cl.exe`` path.
    """
    return _resolve_windows_compiler(not _clang_cl_disabled())


def _find_vsdevcmd() -> str | None:
    candidate = os.environ.get("VSDEVCMD_BAT")
    if candidate and os.path.exists(candidate):
        return candidate

    vswhere_candidates: list[str] = []
    for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES")):
        if base:
            candidate = os.path.join(base, "Microsoft Visual Studio", "Installer", "vswhere.exe")
            if os.path.exists(candidate):
                vswhere_candidates.append(candidate)

    for vswhere in vswhere_candidates:
        try:
            proc = subprocess.run(
                [
                    vswhere,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-find",
                    r"Common7\Tools\VsDevCmd.bat",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError:
            continue
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                path = line.strip()
                if path and os.path.exists(path):
                    return path

    for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES")):
        if not base:
            continue
        root = os.path.join(base, "Microsoft Visual Studio")
        for pattern in (
            os.path.join(root, "2022", "*", "Common7", "Tools", "VsDevCmd.bat"),
            os.path.join(root, "2019", "*", "Common7", "Tools", "VsDevCmd.bat"),
        ):
            matches = sorted(glob.glob(pattern), reverse=True)
            if matches:
                return matches[0]
    return None


def _import_vsdevcmd_environment(vsdevcmd: str) -> dict[str, str] | None:
    cmd_exe = os.environ.get("COMSPEC")
    if not cmd_exe:
        cmd_exe = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "cmd.exe")

    command = f'call "{vsdevcmd}" -no_logo -arch=x64 -host_arch=x64 >nul && set'
    # `set` is a cmd builtin: /u makes its output UTF-16LE regardless of the
    # console code page, preserving SDK paths under non-ASCII user directories.
    command_line = f'"{cmd_exe}" /u /d /s /c "{command}"'
    try:
        proc = subprocess.run(
            command_line,
            capture_output=True,
            text=True,
            encoding="utf-16le",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None

    compiler_env = os.environ.copy()
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name:
            if name.upper() == "PATH":
                # A subprocess environment is a plain, case-sensitive dict.
                # Duplicate Path/PATH keys make Windows ignore later SDK paths.
                for key in list(compiler_env):
                    if key.upper() == "PATH":
                        del compiler_env[key]
                compiler_env["PATH"] = value
            else:
                compiler_env[name] = value
    return compiler_env


@functools.cache
def get_msvc_subprocess_env() -> dict[str, str] | None:
    """Return the resolved MSVC subprocess environment.

    Memoized via :func:`functools.cache` (matching ``contrib/cc.py`` and
    ``cache/kernel_cache.py``). Concurrent first callers may each spawn
    ``vswhere.exe`` / ``VsDevCmd.bat`` once; ``functools.cache`` keeps only
    the winning result. Subsequent calls hit the cache directly with no
    locking.
    """
    if os.name != "nt":
        return None

    global _MSVC_ENV_ERROR
    base_env = os.environ.copy()
    if base_env.get("INCLUDE") and base_env.get("LIB") and shutil.which("cl.exe", path=get_env_path(base_env)):
        return base_env

    vsdevcmd = _find_vsdevcmd()
    if not vsdevcmd:
        _MSVC_ENV_ERROR = "Could not find VsDevCmd.bat. Install Visual Studio Build Tools or set VSDEVCMD_BAT."
        return base_env

    compiler_env = _import_vsdevcmd_environment(vsdevcmd)
    if compiler_env is None:
        _MSVC_ENV_ERROR = f"VsDevCmd.bat failed: {vsdevcmd}"
        return base_env

    if not shutil.which("cl.exe", path=get_env_path(compiler_env)):
        # cl.exe missing is only a hard failure when the caller actually needs
        # MSVC; clang-cl still benefits from the activated INCLUDE/LIB/PATH,
        # so surface the diagnostic but keep the enriched env.
        _MSVC_ENV_ERROR = f"VsDevCmd.bat did not expose cl.exe: {vsdevcmd}"
        return compiler_env

    _MSVC_ENV_ERROR = None
    return compiler_env


def get_msvc_environment_error() -> str | None:
    return _MSVC_ENV_ERROR


# Persist only the toolchain environment, never unrelated credentials from `set`.
_TOOLCHAIN_ENV_KEYS = {
    "PATH",
    "INCLUDE",
    "LIB",
    "LIBPATH",
    "VCINSTALLDIR",
    "VCTOOLSINSTALLDIR",
    "WINDOWSSDKDIR",
    "WINDOWSSDKVERSION",
    "UNIVERSALCRTSDKDIR",
    "UCRTVERSION",
    "VSCMD_VER",
    "VSCMD_ARG_TGT_ARCH",
    "VSCMD_ARG_HOST_ARCH",
}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    compiler_env = get_msvc_subprocess_env() or os.environ.copy()
    compiler, _ = get_windows_compiler()
    if compiler is None or not compiler_env.get("INCLUDE") or not compiler_env.get("LIB"):
        print(get_msvc_environment_error() or "MSVC headers and libraries are unavailable", file=sys.stderr)
        return 1
    selected = {key.upper(): value for key, value in compiler_env.items() if key.upper() in _TOOLCHAIN_ENV_KEYS}
    args.environment.write_text(json.dumps(selected), encoding="utf-8")
    print(json.dumps({"compiler": compiler, "environment": selected}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
