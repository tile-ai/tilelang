# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from typing import Optional
from .utils import is_cuda_target, is_hip_target, is_cpu_target
from tilelang import tvm as tvm
from tilelang.contrib.nvcc import get_target_compute_version, get_nvcc_compiler
from tvm.target import Target
import ctypes
import os
import tempfile
import subprocess
import logging
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.contrib.rocm import find_rocm_path, get_rocm_arch

logger = logging.getLogger(__name__)


class LibraryGenerator(object):
    srcpath: Optional[str] = None
    libpath: Optional[str] = None
    lib_code: Optional[str] = None

    def __init__(self, target: Target):
        self.target = target

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self.libpath
        return ctypes.CDLL(lib_path)

    def compile_lib(self, timeout: float = None):
        target = self.target
        if is_cuda_target(target):
            from tilelang.env import CUTLASS_INCLUDE_DIR
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
            compute_version = "".join(get_target_compute_version(target).split("."))
            if compute_version == "90":
                compute_version = "90a"
            libpath = src.name.replace(".cu", ".so")

            command = [
                get_nvcc_compiler(),
                "-std=c++17",
                "-w",  # Disable all warning messages
                "-Xcudafe",
                "--diag_suppress=177",
                "--compiler-options",
                "'-fPIC'",
                "-lineinfo",
                "--shared",
                src.name,
                "-lcuda",
                "-gencode",
                f"arch=compute_{compute_version},code=sm_{compute_version}",
            ]
            command += [
                "-I" + CUTLASS_INCLUDE_DIR,
            ]

        elif is_hip_target(target):
            from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
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
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
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
        command += ["-o", libpath]

        src.write(self.lib_code)
        src.flush()
        try:
            ret = subprocess.run(command, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            raise RuntimeError(f"Compilation Failed! {command}")

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
