"""Compile NVRTC cubins and load their native host/launch metadata."""

from __future__ import annotations
import importlib
import logging
import os.path as osp
import tempfile
from types import ModuleType

from tvm_tirx.target import Target

from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.device_py_host import load_cuda_kernels
from tilelang.jit.adapter.utils import is_cuda_target
from tilelang.jit.adapter.nvrtc import is_nvrtc_available, NVRTC_UNAVAILABLE_MESSAGE
from tilelang.jit.adapter.nvrtc.include_paths import discover_cuda_include_paths

logger = logging.getLogger(__name__)

if is_nvrtc_available:
    from tilelang.contrib.nvrtc import compile_cuda, get_nvrtc_version
else:
    raise ImportError(NVRTC_UNAVAILABLE_MESSAGE)


class NVRTCLibraryGenerator(LibraryGenerator):
    """Retain source, cubin, and serializable Python host as a matched artifact.

    Packed functions own their provider CUDA modules. Device/context setup and
    launch attributes follow the same runtime path as other native adapters.
    """

    host_func: str | None = None
    kernels: dict | None = None
    pymodule: ModuleType | None = None
    pypath: str | None = None

    def __init__(self, target: Target, verbose: bool = False):
        """Initialize NVRTC library generator.

        Args:
            target: Compilation target (must be CUDA)
            verbose: Enable verbose compilation output
        """
        super().__init__(target, verbose)

    @staticmethod
    def import_from_file(module_name, file_path):
        """Dynamically import Python module from file path.

        Standard importlib pattern for loading modules outside sys.path.
        Used to import generated .py launcher code from temp directory.

        Args:
            module_name: Name to assign to imported module
            file_path: Absolute path to .py file

        Returns:
            Imported module object
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to import module from file: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def update_host_func(self, host_func: str):
        """Store generated Python launch code for later file write.

        Called by adapter after wrapper generates the launch code.
        This is the bridge between code generation and file output.

        Args:
            host_func: Python source code containing call() function
        """
        self.host_func = host_func

    def load_lib(self, lib_path: str | None = None):
        """Read the host schema and create provider modules for the cubin.

        The provider loads each device's CUDA module when it is first called;
        importing a cached artifact does not synchronize the current device.
        """
        if lib_path is None:
            lib_path = self.libpath
        else:
            self.libpath = lib_path

        self.pypath = lib_path.replace(".cubin", ".py")
        self.pymodule = self.import_from_file("kernel", self.pypath)

        if getattr(self.pymodule, "_native_host_schema", None) != 1:
            raise ValueError("Stale NVRTC host cache; recompile with the native host schema")
        self.kernels = load_cuda_kernels(lib_path, self.pymodule._kernel_specs)

    def compile_lib(self, timeout: float | None = None):
        """Compile CUDA source to cubin using NVRTC and write output files.

        Output artifacts (all in temp directory):
            - .cu: Source code (for debugging)
            - .cubin: Compiled binary (for execution)
            - .py: Python launcher (for calling kernels)

        Include paths setup:
            - TileLang templates: kernel primitives and utilities
            - CUTLASS: optimized GEMM/tensor ops
            - CUDA headers: driver/runtime APIs

        Why architecture detection:
            ARM64 servers (SBSA) have different header paths than x86_64.

        Args:
            timeout: Compilation timeout in seconds (currently unsupported by NVRTC compiler)

        Side effects:
            - Writes .cu, .cubin, .py files to temp directory
            - Sets self.srcpath, self.libpath, self.pypath
        """
        target = self.target
        verbose = self.verbose
        if is_cuda_target(target):
            from tilelang.env import CUDA_HOME, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

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

            cuda_home = CUDA_HOME if CUDA_HOME else "/usr/local/cuda"

            cuda_include_paths = discover_cuda_include_paths(cuda_home)

            __CUDACC_VER_MAJOR__ = get_nvrtc_version()[0]
            options = [
                f"-I{tl_template_path}",
                f"-I{cutlass_path}",
                *(f"-I{include_path}" for include_path in cuda_include_paths),
                f"-D__CUDACC_VER_MAJOR__={__CUDACC_VER_MAJOR__}",
            ]

            # CUDA <13 keeps cuda::std at the legacy ``cuda/std`` path. CUDA 13
            # moved it under CCCL (``cccl/cuda/std``) which is already reachable
            # via the ``-I {arch_include}/cccl`` entry above as ``<cuda/std/*>``.
            # Adding ``-I .../cccl/cuda/std`` would also expose CCCL's private
            # ``__tuple_dir/structured_bindings.h`` at the include root, which
            # collides ODR-style with cutlass's own ``tuple_size``/``tuple_element``
            # forward declarations in ``cute/container/tuple.hpp`` under NVRTC
            # (cute uses variadic packs, cccl uses a single ``_Tp``).
            if __CUDACC_VER_MAJOR__ < 13:
                options += [
                    f"-I{include_path}/cuda/std"
                    for include_path in cuda_include_paths
                    if not include_path.endswith(osp.join("include", "cccl"))
                ]

            if self.compile_flags:
                options += [item for flag in self.compile_flags for item in flag.split() if item not in options]

            with self.target:
                cubin_bytes = compile_cuda(self.lib_code, target_format="cubin", options=options, verbose=verbose)
            with open(libpath, "wb") as f:
                f.write(cubin_bytes)

            src.write(self.lib_code)
            src.close()

            self.srcpath = src.name
            self.libpath = libpath
            self.pypath = src.name.replace(".cu", ".py")
            if self.host_func is None:
                raise RuntimeError("Host function is not set, please call update_host_func() first.")
            with open(self.pypath, "w") as f:
                f.write(self.host_func)
        else:
            raise ValueError(f"Unsupported target: {target}")

    def __del__(self):
        """Release this generator's references to provider-owned functions."""
        # Native packed functions retain their CUDA modules. Releasing the
        # references uses the provider's module lifetime and unload contract.
        self.kernels = None
