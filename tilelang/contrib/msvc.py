from __future__ import annotations

import functools
import hashlib
import os
import re
import glob
import shutil
import subprocess
import tempfile

from tvm.base import py_str

from tilelang.env import TL_LIBS

_MSVC_ENV_ERROR: str | None = None

_ALIGN_ATTRIBUTE_RE = re.compile(r"__attribute__\s*\(\(\s*aligned\s*\(\s*([0-9]+)\s*\)\s*\)\)")
_TVM_FFI_EXPORT_RE = re.compile(r"\bint32_t\s+(__tvm_ffi_[A-Za-z0-9_]+)\s*\(")


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
    command_line = f'"{cmd_exe}" /d /s /c "{command}"'
    try:
        proc = subprocess.run(
            command_line,
            capture_output=True,
            text=True,
            encoding="utf-8",
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
            compiler_env[name] = value
            if name.upper() == "PATH":
                compiler_env["PATH"] = value
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
    if shutil.which("cl.exe", path=get_env_path(base_env)):
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


def _normalize_option(option: str) -> str | None:
    if option.startswith("-I"):
        return "/I" + option[2:]
    if option.startswith("-D"):
        return "/D" + option[2:]
    if option in ("-g", "-fPIC"):
        return None
    if option.startswith("-std="):
        return None
    return option


def _patch_source_for_msvc(path: str, tmp_dir: str) -> str:
    if os.path.splitext(path)[1].lower() not in (".c", ".cc", ".cpp"):
        return path

    with open(path, encoding="utf-8") as src:
        source = src.read()

    patched = _ALIGN_ATTRIBUTE_RE.sub(r"__declspec(align(\1))", source)
    if patched == source:
        return path

    prefix = hashlib.md5(os.path.dirname(os.path.abspath(path)).encode()).hexdigest()[:8]
    patched_path = os.path.join(tmp_dir, f"{prefix}_{os.path.basename(path)}")
    with open(patched_path, "w", encoding="utf-8") as dst:
        dst.write(patched)
    return patched_path


def _collect_tvm_ffi_exports(path: str) -> list[str]:
    if os.path.splitext(path)[1].lower() not in (".c", ".cc", ".cpp"):
        return []
    with open(path, encoding="utf-8") as src:
        source = src.read()
    exports = set(_TVM_FFI_EXPORT_RE.findall(source))
    if "__tvm_ffi__library_ctx" in source:
        exports.add("__tvm_ffi__library_ctx,DATA")
    return sorted(exports)


def _find_import_libs() -> tuple[list[str], list[str]]:
    lib_paths: list[str] = []
    libs: list[str] = []

    def add_import_lib(lib_path: str | None):
        if not lib_path:
            return
        lib_path = os.path.abspath(lib_path)
        if not os.path.exists(lib_path):
            return
        lib_dir = os.path.dirname(lib_path)
        if lib_dir not in lib_paths:
            lib_paths.append(lib_dir)
        if lib_path not in libs:
            libs.append(lib_path)

    for lib_dir in TL_LIBS:
        if not os.path.isdir(lib_dir):
            continue
        for lib_name in ("tvm.lib", "tvm_ffi.lib"):
            lib_path = os.path.join(lib_dir, lib_name)
            add_import_lib(lib_path)

    try:
        from tvm_ffi import libinfo as tvm_ffi_libinfo

        add_import_lib(tvm_ffi_libinfo.find_windows_implib())
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("Could not locate tvm_ffi import lib: %s", e)

    return lib_paths, libs


def create_shared(output: str, objects, options=None, cc=None, cwd=None, ccache_env=None):
    if os.name != "nt":
        raise ValueError("Windows shared-library compiler is only available on Windows")

    compiler_env = get_msvc_subprocess_env()

    if cc is not None:
        compiler = cc
        kind = "clang-cl" if "clang-cl" in os.path.basename(cc).lower() else "msvc"
    else:
        compiler, kind = get_windows_compiler()

    if compiler is None:
        detail = get_msvc_environment_error()
        msg = (
            "Could not find a Windows host compiler (clang-cl or cl.exe). "
            "Install LLVM (clang-cl) or Visual Studio Build Tools, or set "
            "VSDEVCMD_BAT / CLANG_CL."
        )
        if detail:
            msg += f" ({detail})"
        raise RuntimeError(msg)

    if isinstance(objects, str):
        objects = [objects]
    options = [] if options is None else list(options)

    with tempfile.TemporaryDirectory() as tmp_dir:
        patched_objects = [_patch_source_for_msvc(path, tmp_dir) for path in objects]

        compile_options: list[str] = []
        for option in options:
            normalized = _normalize_option(str(option))
            if normalized is not None:
                compile_options.append(normalized)

        lib_paths, libs = _find_import_libs()
        link_options = ["/LIBPATH:" + lib_path for lib_path in lib_paths]
        link_options.extend(libs)
        exports: list[str] = []
        for path in patched_objects:
            exports.extend(_collect_tvm_ffi_exports(path))
        link_options.extend("/EXPORT:" + name for name in sorted(set(exports)))

        # "/Fo" isolates intermediate .obj output to this call's tmp_dir. # codespell:ignore
        # Without it cl.exe writes ``<source>.obj`` next to the cwd, so two
        # concurrent ``create_shared`` invocations (e.g. parallel autotune
        # workers) race on the same path and one fails with Permission denied.
        cmd = [
            compiler,
            "/nologo",
            "/LD",
            "/O2",
            "/utf-8",
            "/W0",
            "/Fe:" + output,
            "/Fo:" + os.path.join(tmp_dir, ""),  # codespell:ignore
        ]
        if kind == "clang-cl":
            cmd.append("-Wno-unused-command-line-argument")
        cmd += compile_options
        cmd += patched_objects
        cmd += ["/link"] + link_options

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=compiler_env,
        )
        out, _ = proc.communicate()
        if proc.returncode != 0:
            msg = "Compilation error:\n"
            msg += py_str(out)
            msg += "\nCommand line: " + " ".join(cmd)
            raise RuntimeError(msg)


create_shared.output_format = "dll"
