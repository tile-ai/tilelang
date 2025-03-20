# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from typing import Optional
from .utils import is_cuda_target, is_hip_target, is_cpu_target
from tilelang import tvm as tvm
from tilelang.contrib.nvcc import get_target_compute_version
from tvm.target import Target
import ctypes
import os
import tempfile
import subprocess
import logging
from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR

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

    def compile_lib(self, timeout: float = None, with_tl: bool = True):
        target = self.target
        if is_cuda_target(target):
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
            compute_version = "".join(get_target_compute_version(target).split("."))
            if compute_version == "90":
                compute_version = "90a"
            libpath = src.name.replace(".cu", ".so")

            command = [
                "nvcc",
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

        elif is_hip_target(target):
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
            libpath = src.name.replace(".cpp", ".so")

            command = [
                "hipcc",
                "-std=c++17",
                "-fPIC",
                "--shared",
                src.name,
            ]
        elif is_cpu_target(target):
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
            libpath = src.name.replace(".cpp", ".so")

            command = ["g++", "-std=c++17", "-fPIC", "-shared", src.name]
            with_tl = False
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
            ]
        else:
            raise ValueError(f"Unsupported target: {target}")

        if with_tl:
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
                "-I" + CUTLASS_INCLUDE_DIR,
            ]
            command += ["-diag-suppress=20013"]
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
