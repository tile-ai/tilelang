"""CuTeDSL Library Generator for TileLang.

This module provides library generation functionality for the CuTeDSL backend.
"""

from __future__ import annotations
import importlib.util
import os
import tempfile
import subprocess

from tvm.target import Target

from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.utils import is_cutedsl_target


class CuTeDSLLibraryGenerator(LibraryGenerator):
    host_func: str | None = None
    tma_cpp_init_code: str | None = None
    tma_lib_name: str | None = None
    launcher_cpp_code: str | None = None
    launcher_lib_name: str | None = None
    pymodule = None

    def __init__(self, target: Target, verbose: bool = False):
        super().__init__(target, verbose)

    @staticmethod
    def import_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def update_host_func(self, host_func: str):
        self.host_func = host_func

    def update_tma_cpp_init_code(self, tma_cpp_init_code: str):
        self.tma_cpp_init_code = tma_cpp_init_code

    def update_tma_lib_name(self, tma_lib_name: str):
        self.tma_lib_name = tma_lib_name

    def update_launcher_cpp_code(self, launcher_cpp_code: str):
        self.launcher_cpp_code = launcher_cpp_code

    def update_launcher_lib_name(self, launcher_lib_name: str):
        self.launcher_lib_name = launcher_lib_name

    def load_lib(self, lib_path: str | None = None):
        if lib_path is None:
            if self.libpath is None:
                raise RuntimeError("CuTeDSLLibraryGenerator.libpath is not set; call compile_lib() first or pass lib_path explicitly.")
            lib_path = self.libpath

        self.pymodule = self.import_from_file("kernel", lib_path)

    def compile_lib(self, timeout: float = None):
        if self.host_func is None:
            raise RuntimeError("CuTeDSLLibraryGenerator.host_func is not set; call update_host_func() before compile_lib().")
        target = self.target
        if is_cutedsl_target(target):
            # Use a dedicated temp directory per kernel so CuTeDSL artifacts (e.g. kept .cubin)
            # never pollute user CWD, and are easy to locate alongside the generated module.
            work_dir = tempfile.mkdtemp(prefix="tilelang_cutedsl_")
            src_path = os.path.join(work_dir, "kernel.py")
            with open(src_path, "w") as f:
                # Note: lib_code (containing @cute.kernel definitions) is embedded
                # inside host_func's _generate_cubin_if_needed function, so we only
                # write host_func here. This ensures cute imports are lazy-loaded.
                f.write(self.host_func)

            # Compile C++ launcher library if needed
            if self.launcher_cpp_code is not None:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".cpp",
                    delete=False,
                ) as launcher_src:
                    launcher_src.write(self.launcher_cpp_code)
                    launcher_src_path = launcher_src.name

                # Generate launcher lib under the same directory as the source file
                launcher_lib_path = os.path.join(os.path.dirname(src_path), self.launcher_lib_name)

                # Get TVM FFI compiler flags using tvm_ffi.libinfo API
                try:
                    import tvm_ffi.libinfo

                    include_paths = tvm_ffi.libinfo.include_paths()
                    tvm_cxxflags = [f"-I{path}" for path in include_paths]
                    lib_path = tvm_ffi.libinfo.find_libtvm_ffi()
                    lib_dir = os.path.dirname(lib_path)
                    tvm_ldflags = [f"-L{lib_dir}", "-ltvm_ffi"]
                except (ImportError, RuntimeError):
                    # tvm_ffi unavailable or libinfo functions failed
                    tvm_cxxflags = []
                    tvm_ldflags = []

                # Compile with nvcc (need CUDA driver API)
                compile_cmd = [
                    "nvcc",
                    "-shared",
                    "-Xcompiler=-fPIC",
                    "-lcuda",
                    *tvm_cxxflags,
                    *tvm_ldflags,
                    "-o",
                    launcher_lib_path,
                    launcher_src_path,
                ]

                result = subprocess.run(compile_cmd, check=False, capture_output=True, text=True, timeout=timeout)
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to compile C++ launcher: {result.stderr}")

                self.launcher_libpath = launcher_lib_path
                self.launcher_libname = self.launcher_lib_name

            self.srcpath = src_path
            self.libpath = src_path
        else:
            raise ValueError(f"Unsupported target: {target}")
