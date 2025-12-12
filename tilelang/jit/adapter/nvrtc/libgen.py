"""NVRTC Library Generator for TileLang.

Compiles CUDA kernels at runtime using NVRTC and manages resulting binaries.

Why NVRTC instead of nvcc:
- No offline compilation step, enables true JIT workflows
- Works without CUDA toolkit installed (only requires driver)
- Allows kernel specialization based on runtime parameters

Key responsibilities:
- Compile CUDA source to cubin using NVRTC API
- Generate accompanying Python launcher code
- Load compiled cubin and extract kernel handles
- Manage library lifecycle (load/unload)
"""

from __future__ import annotations
import importlib
import logging
import os.path as osp
import platform
import tempfile
from types import ModuleType

from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.utils import is_cuda_target
from tilelang.jit.adapter.nvrtc import is_nvrtc_available, NVRTC_UNAVAILABLE_MESSAGE

logger = logging.getLogger(__name__)

if is_nvrtc_available:
    import cuda.bindings.driver as cuda
    from tilelang.contrib.nvrtc import compile_cuda
else:
    raise ImportError(NVRTC_UNAVAILABLE_MESSAGE)


class NVRTCLibraryGenerator(LibraryGenerator):
    """Runtime compiler and loader for NVRTC-compiled CUDA kernels.

    Lifecycle:
        1. compile_lib(): CUDA source → cubin + Python launcher
        2. load_lib(): cubin → loaded library + kernel handles
        3. pymodule.call(): Execute kernels via Python launcher
        4. __del__: Cleanup (unload library)

    Why three files (cu, cubin, py):
        - .cu: Source for debugging, kept in temp directory
        - .cubin: Compiled binary, loaded by CUDA driver
        - .py: Launch code, imported as Python module

    Attributes:
        host_func: Generated Python launch code (from wrapper)
        culib: CUDA library handle (CUlibrary)
        pymodule: Imported Python module containing call() function
    """

    host_func: str | None = None
    culib: cuda.CUlibrary | None = None
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
        """Load compiled cubin and Python launcher into memory.

        Why two loads:
            1. Import Python module for launch logic
            2. Load cubin via CUDA Driver API for kernel handles

        Context synchronization: CUDA context must be current before loading.
        If not, use torch.cuda.synchronize() to establish context.

        Args:
            lib_path: Path to .cubin file (optional, uses self.libpath if None)

        Side effects:
            - Sets self.pymodule to imported Python module
            - Sets self.culib to CUDA library handle
        """
        if lib_path is None:
            lib_path = self.libpath
        else:
            self.libpath = lib_path

        self.pypath = lib_path.replace(".cubin", ".py")
        self.pymodule = self.import_from_file("kernel", self.pypath)

        # Ensure the context is valid
        ctx = cuda.cuCtxGetCurrent()[1]
        if cuda.cuCtxGetApiVersion(ctx)[0] != cuda.CUresult.CUDA_SUCCESS:
            import torch

            torch.cuda.synchronize()

        result, self.culib = cuda.cuLibraryLoadFromFile(bytes(lib_path, "utf-8"), [], [], 0, [], [], 0)
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to load library: {lib_path}, error: {result}")

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
            __CUDACC_VER_MAJOR__ = cuda.CUDA_VERSION // 1000

            # Determine target architecture
            machine = platform.machine()
            target_arch = "sbsa-linux" if machine in ("aarch64", "arm64") else "x86_64-linux"

            options = [
                f"-I{tl_template_path}",
                f"-I{cutlass_path}",
                f"-I{cuda_home}/include",
                f"-I{cuda_home}/targets/{target_arch}/include",
                f"-I{cuda_home}/targets/{target_arch}/include/cccl",
                f"-D__CUDACC_VER_MAJOR__={__CUDACC_VER_MAJOR__}",
            ]
            if self.compile_flags:
                options += [item for flag in self.compile_flags for item in flag.split() if item not in options]

            cubin_bytes = compile_cuda(self.lib_code, target_format="cubin", options=options, verbose=verbose)
            with open(libpath, "wb") as f:
                f.write(cubin_bytes)

            src.write(self.lib_code)
            src.flush()

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
        """Cleanup: unload CUDA library when object is destroyed.

        Critical for resource management - CUDA libraries consume GPU memory.
        Failure to unload is logged but not raised (destructor can't fail).

        Why explicit unload:
            Python GC doesn't know about GPU resources, must release manually.
        """
        if self.culib:
            result = cuda.cuLibraryUnload(self.culib)[0]
            if result != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(f"Failed to unload library: {self.libpath}")
            self.culib = None
