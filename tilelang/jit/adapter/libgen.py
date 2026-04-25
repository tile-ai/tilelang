from __future__ import annotations
import ctypes
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any

from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.transform import PassConfigKey
from tilelang.contrib.nvcc import (
    get_cuda_library_dirs,
    get_nvcc_compiler,
    get_target_arch,
    get_target_compute_version,
)
from tilelang.contrib.rocm import find_rocm_path, get_rocm_arch
from tilelang.env import TILELANG_TEMPLATE_PATH

from .utils import is_cpu_target, is_cuda_target, is_hip_target

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
        if is_cuda_target(target):
            from tilelang.env import CUTLASS_INCLUDE_DIR

            _lib_ext = ".dll" if sys.platform == "win32" else ".so"
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)  # noqa: SIM115
            target_arch = get_target_arch(get_target_compute_version(target))
            libpath = src.name.replace(".cu", _lib_ext)

            enable_fast_math = self.pass_configs.get(PassConfigKey.TL_ENABLE_FAST_MATH, False)

            ptxas_usage_level = self.pass_configs.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
            if ptxas_usage_level is not None:
                ptxas_usage_level = int(ptxas_usage_level)
            verbose_ptxas_output = self.pass_configs.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False)
            cuda_library_flags = [f"-L{lib_dir}" for lib_dir in get_cuda_library_dirs()]

            command = [
                get_nvcc_compiler(),
                "-std=c++17",
                "-w",
                "-Xcudafe",
                "--diag_suppress=177",
                "-lineinfo",
                "--shared",
                src.name,
                *cuda_library_flags,
                "-lcuda",
                "-gencode",
                f"arch=compute_{target_arch},code=sm_{target_arch}",
            ]
            if sys.platform == "win32":
                command += ["-Xcompiler", "/Zc:preprocessor"]
            else:
                command += ["--compiler-options", "-fPIC"]
            if enable_fast_math:
                command += ["--use_fast_math"]
            if ptxas_usage_level is not None:
                command += [f"--ptxas-options=--register-usage-level={ptxas_usage_level}"]
            if verbose_ptxas_output:
                command += ["--ptxas-options=--verbose"]
            command += [
                "-I" + CUTLASS_INCLUDE_DIR,
            ]

        elif is_hip_target(target):
            from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR

            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)  # noqa: SIM115
            libpath = src.name.replace(".cpp", ".so")
            rocm_path = find_rocm_path()
            arch = get_rocm_arch(rocm_path)
            command = [
                "hipcc",
                "-std=c++17",
                "-fPIC",
                f"--offload-arch={arch}",
                "--shared",
                src.name,
            ]
            command += [
                "-I" + COMPOSABLE_KERNEL_INCLUDE_DIR,
            ]
        elif is_cpu_target(target):
            from tilelang.contrib.cc import get_cplus_compiler

            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)  # noqa: SIM115
            libpath = src.name.replace(".cpp", ".so")

            command = [get_cplus_compiler(), "-std=c++17", "-fPIC", "-shared", src.name]
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
            ]
        else:
            raise ValueError(f"Unsupported target: {target}")

        command += [
            "-I" + TILELANG_TEMPLATE_PATH,
        ]

        if self.compile_flags:
            command += [item for flag in self.compile_flags for item in flag.split() if item not in command]

        command += ["-o", libpath]

        src.write(self.lib_code)
        src.flush()
        if sys.platform == "win32":
            src.close()

        # On Windows, two concerns matter for parallel autotune:
        # 1. nvcc needs MSVC's host compiler env (cl.exe, INCLUDE, LIB).
        # 2. Concurrent subprocesses sharing the parent's console handle can
        #    deadlock when their output interleaves with tqdm progress bars.
        # Pipe stdio + isolate stdin to make the launch self-contained.
        run_kwargs: dict[str, Any] = {"timeout": timeout}
        if sys.platform == "win32":
            from tilelang.contrib.nvcc import get_nvcc_subprocess_env

            run_kwargs.update(
                env=get_nvcc_subprocess_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

        try:
            if verbose:
                print(f"compile_lib compilation command: {' '.join(command)}")
            ret = subprocess.run(command, **run_kwargs)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            captured = ret.stdout.decode("utf-8", errors="replace") if ret.stdout else ""
            raise RuntimeError(f"Compilation Failed! {command}\n{captured}\n{self.lib_code}")

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
