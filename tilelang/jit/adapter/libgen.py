from __future__ import annotations
import ctypes
import logging
import os
import subprocess
import tempfile
from typing import Any

from tvm.target import Target

from tilelang.backend.execution import get_library_compile_spec

logger = logging.getLogger(__name__)


class LibraryGenerator:
    srcpath: str | None = None
    libpath: str | None = None
    lib_code: str | None = None
    pass_configs: dict[str, Any] | None = None
    compile_flags: list[str] | None = None

    def __init__(self, target: Target, verbose: bool = False):
        self.target = target
        self.verbose = verbose

    def assign_pass_configs(self, pass_configs: dict[str, Any] | None = None):
        self.pass_configs = pass_configs

    def assign_compile_flags(self, compile_flags: list[str] | None = None):
        if compile_flags is None:
            compile_flags = []
        self.compile_flags = compile_flags

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: str | None = None):
        if lib_path is None:
            lib_path = self.libpath
        else:
            self.libpath = lib_path
        return ctypes.CDLL(lib_path)

    def compile_lib(self, timeout: float = None):
        target = self.target
        verbose = self.verbose
        compile_spec = get_library_compile_spec(target)
        src = tempfile.NamedTemporaryFile(mode="w", suffix=compile_spec.source_suffix, delete=False)  # noqa: SIM115
        libpath = src.name[: -len(compile_spec.source_suffix)] + compile_spec.library_suffix
        command = compile_spec.command_factory(target, src.name, libpath, self.pass_configs or {})

        if self.compile_flags:
            command += [item for flag in self.compile_flags for item in flag.split() if item not in command]

        src.write(self.lib_code)
        src.flush()

        try:
            if verbose:
                print(f"compile_lib compilation command: {' '.join(command)}")
            ret = subprocess.run(command, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            raise RuntimeError(f"Compilation Failed! {command}\n {self.lib_code}")

        self.srcpath = src.name
        self.libpath = libpath

    def remove_lib(self):
        if self.libpath:
            os.remove(self.libpath)
        self.libpath = None

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath
