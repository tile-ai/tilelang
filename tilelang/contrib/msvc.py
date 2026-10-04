from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile

from tvm.base import py_str

from tilelang.env import TL_LIBS
from tilelang._host_toolchain import (
    get_env_path as get_env_path,
    get_msvc_subprocess_env,
    get_msvc_environment_error,
    get_windows_compiler,
)

_ALIGN_ATTRIBUTE_RE = re.compile(r"__attribute__\s*\(\(\s*aligned\s*\(\s*([0-9]+)\s*\)\s*\)\)")
_TVM_FFI_EXPORT_RE = re.compile(r"\bint32_t\s+(__tvm_ffi_[A-Za-z0-9_]+)\s*\(")


def _normalize_option(option: str) -> str | None:
    if option.startswith("-I"):
        return "/I" + option[2:]
    if option.startswith("-D"):
        return "/D" + option[2:]
    if option in ("-g", "-fPIC"):
        return None
    if option.startswith("-std="):
        return "/std:" + option[5:]
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
        # JIT-generated host libraries call TVM runtime C API symbols such as
        # TVMBackendGetFuncFromEnv. Split Windows builds export those from
        # tvm_runtime.dll, while older unified builds exported them from tvm.dll.
        runtime_lib = os.path.join(lib_dir, "tvm_runtime.lib")
        if os.path.exists(runtime_lib):
            add_import_lib(runtime_lib)
        else:
            add_import_lib(os.path.join(lib_dir, "tvm.lib"))
        add_import_lib(os.path.join(lib_dir, "tvm_ffi.lib"))

    try:
        from tvm_ffi import libinfo as tvm_ffi_libinfo

        add_import_lib(tvm_ffi_libinfo.find_windows_implib())
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("Could not locate tvm_ffi import lib: %s", e)

    return lib_paths, libs


def _compile(output: str, objects, options=None, cc=None, cwd=None, ccache_env=None, timeout=None, compile_shared=True):
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
            "/O2",
            "/EHsc",
            "/utf-8",
            "/W0",
            "/Fe:" + output,
            "/Fo:" + os.path.join(tmp_dir, ""),  # codespell:ignore
        ]
        if compile_shared:
            cmd.append("/LD")
        if kind == "clang-cl":
            cmd.append("-Wno-unused-command-line-argument")
        if ccache_env is not None:
            if not shutil.which("ccache"):
                raise ValueError("ccache not found")
            cmd.insert(0, "ccache")
            compiler_env = dict(compiler_env or os.environ)
            compiler_env.update(ccache_env)
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
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        if proc.returncode != 0:
            msg = "Compilation error:\n"
            msg += py_str(out)
            msg += "\nCommand line: " + " ".join(cmd)
            raise RuntimeError(msg)


def create_shared(output: str, objects, options=None, cc=None, cwd=None, ccache_env=None, timeout=None):
    _compile(output, objects, options, cc, cwd, ccache_env, timeout=timeout, compile_shared=True)


def create_executable(output: str, objects, options=None, cc=None, cwd=None, ccache_env=None):
    _compile(output, objects, options, cc, cwd, ccache_env, compile_shared=False)


create_shared.output_format = "dll"
