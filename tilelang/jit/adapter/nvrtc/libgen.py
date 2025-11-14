from __future__ import annotations
import importlib
import logging
import os.path as osp
import tempfile

from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.utils import is_cuda_target

logger = logging.getLogger(__name__)

try:
    import cuda.bindings.driver as cuda  # noqa: F401
    from tilelang.contrib.nvrtc import compile_cuda
    is_nvrtc_available = True
except ImportError:
    is_nvrtc_available = False

class PyLibraryGenerator(LibraryGenerator):
    host_func: str | None = None
    culib = None
    pymodule = None

    def __init__(self, target: Target, verbose: bool = False):
        if not is_nvrtc_available:
            raise ImportError("cuda-python is not available, nvrtc backend cannot be used. "
                              "Please install cuda-python via `pip install cuda-python` "
                              "if you want to use the nvrtc backend.")
        super().__init__(target, verbose)

    @staticmethod
    def import_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def update_host_func(self, host_func: str):
        self.host_func = host_func

    def load_lib(self, lib_path: str | None = None):
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
        verbose = self.verbose
        if is_cuda_target(target):
            from tilelang.env import (
                CUDA_HOME, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH)
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)  # noqa: SIM115
            libpath = src.name.replace(".cu", ".cubin")

            project_root = osp.join(osp.dirname(__file__), "..", "..")
            if CUTLASS_INCLUDE_DIR is None:
                cutlass_path = osp.abspath(
                    osp.join(project_root, "3rdparty/cutlass/include"))
            else:
                cutlass_path = CUTLASS_INCLUDE_DIR

            if TILELANG_TEMPLATE_PATH is None:
                tl_template_path = osp.abspath(osp.join(project_root, "src"))
            else:
                tl_template_path = TILELANG_TEMPLATE_PATH

            cuda_home = CUDA_HOME if CUDA_HOME else "/usr/local/cuda"
            __CUDACC_VER_MAJOR__ = cuda.CUDA_VERSION // 1000

            options = [
                f"-I{tl_template_path}",
                f"-I{cutlass_path}",
                f"-I{cuda_home}/include",
                f"-I{cuda_home}/targets/x86_64-linux/include", # [TODO](zihuaw) remove temporary include path
                f"-D__CUDACC_VER_MAJOR__={__CUDACC_VER_MAJOR__}",
            ]
            if self.compile_flags:
                options += [
                    item for flag in self.compile_flags for item in flag.split()
                    if item not in options
                ]

            cubin_bytes = compile_cuda(
                self.lib_code, target_format="cubin", options=options, verbose=verbose)
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
