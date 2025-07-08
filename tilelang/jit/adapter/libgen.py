# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import ctypes
import importlib
import logging
import os
import os.path as osp
import subprocess
import tempfile
from typing import Any, Dict, Optional

from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.transform import PassConfigKey
from tilelang.contrib.nvcc import get_nvcc_compiler, get_target_compute_version
from tilelang.contrib.rocm import find_rocm_path, get_rocm_arch
from tilelang.env import TILELANG_TEMPLATE_PATH

from .utils import is_cpu_target, is_cuda_target, is_hip_target

logger = logging.getLogger(__name__)

is_nvrtc_available = False
NVRTC_UNAVAILABLE_WARNING = "cuda-python is not available, nvrtc backend cannot be used. " \
                            "Please install cuda-python via `pip install cuda-python` " \
                            "if you want to use the nvrtc backend."
try:
    import cuda.bindings.driver as cuda
    from tilelang.contrib.nvrtc import compile_cuda
    is_nvrtc_available = True
except ImportError:
    pass


class LibraryGenerator(object):
    srcpath: Optional[str] = None
    libpath: Optional[str] = None
    lib_code: Optional[str] = None
    pass_configs: Optional[Dict[str, Any]] = None

    def __init__(self, target: Target):
        self.target = target

    def assign_pass_configs(self, pass_configs: Optional[Dict[str, Any]] = None):
        self.pass_configs = pass_configs

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self.libpath
        else:
            self.libpath = lib_path
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

            disable_fast_math = self.pass_configs.get(PassConfigKey.TL_DISABLE_FAST_MATH, False)
            verbose_ptxas_output = self.pass_configs.get(
                PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False)

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
            if not disable_fast_math:
                command += ["--use_fast_math"]
            if verbose_ptxas_output:
                command += ["--ptxas_options", "-v"]
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
            raise RuntimeError(f"Compilation Failed! {command}"
                               f"\n {self.lib_code}")

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


class PyLibraryGenerator(LibraryGenerator):
    host_func: Optional[str] = None
    culib = None
    pymodule = None

    def __init__(self, target: Target):
        if not is_nvrtc_available:
            raise ImportError(NVRTC_UNAVAILABLE_WARNING)
        super().__init__(target)

    @staticmethod
    def import_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def update_host_func(self, host_func: str):
        self.host_func = host_func

    def load_lib(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self.libpath

        pypath = lib_path.replace(".cubin", ".py")
        self.pymodule = self.import_from_file("kernel", pypath)

        # Ensure the context is valid
        ctx = cuda.cuCtxGetCurrent()[1]
        if cuda.cuCtxGetApiVersion(ctx)[0] != cuda.CUresult.CUDA_SUCCESS:
            import torch
            torch.cuda.synchronize()

        result, self.culib = cuda.cuLibraryLoadFromFile(
            bytes(lib_path, "utf-8"), [], [], 0, [], [], 0)
        assert result == cuda.CUresult.CUDA_SUCCESS, f"Failed to load library: {lib_path}"

    def compile_lib(self, timeout: float = None):
        target = self.target
        if is_cuda_target(target):
            from tilelang.env import (CUDA_HOME, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH)
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
            libpath = src.name.replace(".cu", ".cubin")

            project_root = osp.join(osp.dirname(__file__), "..", "..")
            if CUTLASS_INCLUDE_DIR is None:
                cutlass_path = osp.abspath(osp.join(project_root, "3rdparty/cutlass/include"))
            else:
                cutlass_path = CUTLASS_INCLUDE_DIR

            if TILELANG_TEMPLATE_PATH is None:
                tl_template_path = osp.abspath(osp.join(project_root, "src"))
            else:
                tl_template_path = TILELANG_TEMPLATE_PATH

            cuda_home = "/usr/local/cuda" if CUDA_HOME is None else CUDA_HOME

            cubin_bytes = compile_cuda(
                self.lib_code,
                target_format="cubin",
                options=[f"-I{tl_template_path}", f"-I{cutlass_path}", f"-I{cuda_home}/include"],
                verbose=True)
            with open(libpath, "wb") as f:
                f.write(cubin_bytes)

            src.write(self.lib_code)
            src.flush()

            self.srcpath = src.name
            self.libpath = libpath

            pypath = src.name.replace(".cu", ".py")
            with open(pypath, "w") as f:
                f.write(self.host_func)
        else:
            raise ValueError(f"Unsupported target: {target}")

    def __del__(self):
        if self.culib:
            result = cuda.cuLibraryUnload(self.culib)[0]
            if result != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(f"Failed to unload library: {self.libpath}")
            self.culib = None
