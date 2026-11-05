"""Persist native CuTeDSL host modules and lazily compile their device entries."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import uuid

from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.utils import is_cutedsl_target

from .runtime import CuTeDSLRuntime


class CuTeDSLLibraryGenerator(LibraryGenerator):
    """A Python host plus provider-owned cubins; no generated launcher DSO."""

    host_func = None
    pymodule = None
    runtime = None

    @staticmethod
    def import_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load generated CuTeDSL module {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def update_host_func(self, host_func):
        self.host_func = host_func

    def update_launcher_cpp_code(self, code):
        if code:
            raise ValueError("A native CuTeDSL host uses the provider launcher")

    def update_launcher_lib_name(self, name):
        if name:
            raise ValueError("A native CuTeDSL host does not require another launcher library")

    def compile_lib(self, timeout=None):
        if not is_cutedsl_target(self.target):
            raise ValueError(f"Unsupported CuTeDSL target: {self.target}")
        if self.host_func is None:
            raise RuntimeError("CuTeDSL host source is missing")
        directory = Path(tempfile.mkdtemp(prefix="tilelang_cutedsl_"))
        path = directory / "kernel.py"
        path.write_text(self.host_func)
        self.srcpath = self.libpath = str(path)

    def load_lib(self, lib_path=None):
        if lib_path is not None:
            self.libpath = lib_path
        if self.libpath is None:
            raise RuntimeError("CuTeDSL library path is missing")
        module = self.import_from_file("tilelang_cutedsl_" + uuid.uuid4().hex, self.libpath)
        if getattr(module, "_native_host_schema", None) != 1 or getattr(module, "_native_cutedsl_schema", None) != 1:
            raise ValueError("Stale CuTeDSL host cache; recompile with the native host schema")
        self.pymodule = module
        self.runtime = CuTeDSLRuntime(module, self.libpath, self.compile_flags)

    def cleanup(self):
        if self.runtime is not None:
            self.runtime.cleanup()
