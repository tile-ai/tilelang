"""The cache utils with class and database persistence - KernelCache Class"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import threading
import uuid
import sys
from hashlib import sha256
from typing import Callable, Literal

import cloudpickle
from tvm.target import Target
from tvm.tir import PrimFunc
from tvm.runtime import Executable
from tilelang.engine.param import KernelParam
from tilelang.utils.language import get_prim_func_name
from tilelang import env
from tilelang.jit import JITKernel
from tilelang import __version__


class KernelCache:
    """
    Caches compiled kernels using a class and database persistence to avoid redundant compilation.
    Cache files:
        kernel.cu: The compiled kernel source code
        wrapped_kernel.cu: The compiled wrapped kernel source code
        kernel_lib.so: The compiled kernel library
        params.pkl: The compiled kernel parameters
    """

    _instance = None  # For implementing singleton pattern
    _lock = threading.Lock()  # For thread safety
    _memory_cache = {}  # In-memory cache dictionary
    execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] = "tvm_ffi"
    device_kernel_path = "device_kernel.cu"
    host_kernel_path = "host_kernel.cu"
    kernel_lib_path = "kernel_lib.so"
    params_path = "params.pkl"

    @staticmethod
    @functools.cache
    def _get_compile_args() -> dict:
        if sys.platform != "darwin":
            return {}

        from torch.utils import cpp_extension

        return {"options": ["-x", "objective-c++", "-g", "-std=gnu++17"] + ["-I" + i for i in cpp_extension.include_paths()]}

    @staticmethod
    @functools.cache
    def _get_tilelang_lib_stamp() -> str | None:
        """Return a cheap build-stamp for the TileLang runtime library.

        The kernel cache key historically only depended on `tilelang.__version__`
        and the TIR script. During development, C++ pass changes can change the
        generated kernel *without* changing the input TIR, leading to stale cache
        hits. Including a library stamp (mtime+size) avoids this class of bugs
        while keeping the cost low (computed once per process).
        """
        import importlib

        lib_dirs: list[str] = []
        try:
            env_mod = importlib.import_module("tilelang.env")
            lib_dirs.extend(getattr(env_mod, "TL_LIBS", []) or [])
        except Exception:
            pass

        if sys.platform == "win32":
            lib_names = ["tilelang.dll", "libtilelang.dll"]
        elif sys.platform == "darwin":
            lib_names = ["libtilelang.dylib", "libtilelang.so"]
        else:
            lib_names = ["libtilelang.so"]

        for lib_dir in lib_dirs:
            for name in lib_names:
                path = os.path.join(lib_dir, name)
                if os.path.exists(path):
                    st = os.stat(path)
                    return f"{name}:{st.st_size}:{st.st_mtime_ns}"
        return None

    @staticmethod
    @functools.cache
    def _get_base_key() -> dict:
        base = {"version": __version__}
        lib_stamp = KernelCache._get_tilelang_lib_stamp()
        if lib_stamp:
            base["tilelang_lib"] = lib_stamp
        if sys.platform == "darwin":
            import torch

            base["torch"] = torch.__version__
        return base

    def __new__(cls):
        """
        Implements singleton pattern for KernelCache class.

        Returns:
            KernelCache: The singleton instance of KernelCache.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    instance = super().__new__(cls)
                    KernelCache._create_dirs()
                    instance.logger = logging.getLogger(__name__)
                    instance.logger.setLevel(logging.DEBUG)
                    instance._memory_cache = {}  # Initialize memory cache
                    cls._instance = instance
        return cls._instance

    @staticmethod
    def _create_dirs():
        os.makedirs(env.TILELANG_CACHE_DIR, exist_ok=True)
        os.makedirs(env.TILELANG_TMP_DIR, exist_ok=True)

    def _generate_key(
        self,
        func: Callable,
        out_idx: list[int],
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] = "tvm_ffi",
        args=None,
        target: str | Target = "auto",
        target_host: str | Target = None,
        pass_configs: dict = None,
        compile_flags: list[str] | str | None = None,
    ) -> str:
        """
        Generates a unique hash key for caching compiled kernels.

        Args:
            func (Callable): The function to be compiled.
            out_idx (List[int]): Indices specifying which outputs to return.
            execution_backend (Literal): Backend type for execution. Defaults to "tvm_ffi".
            args: Arguments passed to the function.
            target (Union[str, Target]): Compilation target platform. Defaults to "auto".
            target_host (Union[str, Target], optional): Host target platform.

        Returns:
            str: SHA256 hash key for the kernel configuration.
        """
        self.execution_backend = execution_backend
        func_binary = func.script(show_meta=True).encode()
        key_data = {
            "func": sha256(func_binary).hexdigest(),  # Use SHA256 to generate hash key
            "out_idx": (tuple(out_idx) if isinstance(out_idx, (list, tuple)) else [out_idx]),
            "args_repr": tuple(repr(arg) for arg in args),  # Use repr to serialize arguments, may need more robust serialization
            "target": str(target),
            "target_host": str(target_host) if target_host else None,
            "execution_backend": execution_backend,
            "pass_configs": pass_configs,
            "compile_flags": compile_flags,
            **self._get_base_key(),
        }
        # Sort keys to ensure consistency
        key_string = json.dumps(key_data, sort_keys=True)
        # Use SHA256 to generate hash key
        return sha256(key_string.encode()).hexdigest()

    def cached(
        self,
        func: PrimFunc = None,
        out_idx: list[int] = None,
        *args,
        target: str | Target,
        target_host: str | Target | None = None,
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] = "tvm_ffi",
        verbose: bool,
        pass_configs: dict | None = None,
        compile_flags: list[str] | str | None = None,
    ) -> JITKernel:
        """
        Caches and reuses compiled kernels to avoid redundant compilation.

        This is the ONLY place where environment variable processing, target normalization,
        and execution backend resolution should happen. All compilation paths go through here.

        Args:
            func: Function to be compiled or a prepared PrimFunc
            out_idx: Indices specifying which outputs to return
            target: Compilation target platform (None = read from TILELANG_TARGET env var)
            target_host: Host target platform
            execution_backend: Execution backend (None = read from TILELANG_EXECUTION_BACKEND)
            verbose: Enable verbose output (None = read from TILELANG_VERBOSE)
            *args: Arguments passed to func

        Returns:
            JITKernel: The compiled kernel, either freshly compiled or from cache

        Environment Variables
        ---------------------
        TILELANG_TARGET : str
            Default compilation target (e.g., "cuda", "llvm"). Defaults to "auto".
        TILELANG_EXECUTION_BACKEND : str
            Default execution backend. Defaults to "auto".
        TILELANG_VERBOSE : str
            Set to "1", "true", "yes", or "on" to enable verbose compilation by default.
        """

        if not env.is_cache_enabled():
            if verbose:
                self.logger.info("Cache is disabled; compiling kernel without caching.")
            return JITKernel(
                func,
                out_idx=out_idx,
                execution_backend=execution_backend,
                target=target,
                target_host=target_host,
                verbose=verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )

        key = self._generate_key(
            func=func,
            out_idx=out_idx,
            execution_backend=execution_backend,
            args=args,
            target=target,
            target_host=target_host,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )
        if verbose:
            self.logger.info(f"Generated cache key: {key} for kernel {get_prim_func_name(func, '<unknown>')}")
        with self._lock:
            # First check in-memory cache
            if key in self._memory_cache:
                # Include kernel name for easier debugging when hitting memory cache
                kernel_name = get_prim_func_name(func, "<unknown>")
                self.logger.warning(
                    "Found kernel '%s' in memory cache. For better performance, consider using `@tilelang.jit` instead of direct kernel caching.",
                    kernel_name,
                )
                return self._memory_cache[key]

            if verbose:
                self.logger.debug(f"Checking disk cache for kernel {get_prim_func_name(func, '<unknown>')}")

            # Then check disk cache
            kernel = self._load_kernel_from_disk(
                key, target, target_host, out_idx, execution_backend, pass_configs, compile_flags, func, verbose
            )
            if kernel is not None:
                if verbose:
                    self.logger.debug(f"Found kernel in disk cache for {get_prim_func_name(func, '<unknown>')}")
                # Populate memory cache with disk result
                self._memory_cache[key] = kernel
                return kernel

        if verbose:
            self.logger.debug(f"No cached kernel for {get_prim_func_name(func, '<unknown>')}")
        # Compile kernel if cache miss; leave critical section
        kernel = JITKernel(
            func,
            out_idx=out_idx,
            execution_backend=execution_backend,
            target=target,
            target_host=target_host,
            verbose=verbose,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )
        with self._lock:
            if env.is_cache_enabled():
                cache_path = self._get_cache_path(key)
                self._save_kernel_to_disk(key, kernel, func, verbose)
                # Set cache path on adapter so it can save cubin after first execution
                self._set_adapter_cache_path(kernel, cache_path)

        # Store in memory cache after compilation
        self._memory_cache[key] = kernel
        return kernel

    def clear_cache(self):
        """
        Clears the entire kernel cache, including both in-memory and disk cache.
        """
        with self._lock:
            self._memory_cache.clear()  # Clear in-memory cache
            self._clear_disk_cache()  # Clear disk cache

    def _get_cache_path(self, key: str) -> str:
        """
        Gets the filesystem path for a cached kernel.

        Args:
            key (str): The hash key identifying the kernel.

        Returns:
            str: Absolute path to the cache directory for this kernel.
        """
        return os.path.join(env.TILELANG_CACHE_DIR, key)

    @staticmethod
    def _load_binary(path: str):
        with open(path, "rb") as file:
            binary = file.read()
        return binary

    @staticmethod
    def _safe_write_file(path: str, mode: str, operation: Callable):
        # Random a temporary file within the same FS as the cache directory
        temp_path = os.path.join(env.TILELANG_TMP_DIR, f"{os.getpid()}_{uuid.uuid4()}")
        with open(temp_path, mode) as temp_file:
            operation(temp_file)

        # Use atomic POSIX replace, so other processes cannot see a partial write
        os.replace(temp_path, path)

    @classmethod
    def _safe_write_executable(cls, executable: Executable, path: str):
        temp_path = os.path.join(env.TILELANG_TMP_DIR, f"{os.getpid()}_{uuid.uuid4()}.so")
        executable.export_library(temp_path, **cls._get_compile_args())
        os.replace(temp_path, path)

    def _save_kernel_to_disk(self, key: str, kernel: JITKernel, func: Callable = None, verbose: bool = False):
        """
        Persists a compiled kernel to disk cache.

        Args:
            key (str): The hash key identifying the kernel.
            kernel (JITKernel): The compiled kernel to be saved.
            func (Callable, optional): The original function.
            verbose (bool): Enable verbose log messages.

        Note:
            Saves the following files:
            - kernel.cu: The compiled kernel source code
            - wrapped_kernel.cu: The wrapped kernel source code
            - kernel_lib.so: The compiled kernel library
            - params.pkl: The serialized kernel parameters
        """
        cache_path = self._get_cache_path(key)
        os.makedirs(cache_path, exist_ok=True)  # Ensure directory exists

        # Save kernel source code
        try:
            self._save_kernel_source_code_to_disk(kernel, cache_path, verbose)
        except Exception:
            self.logger.exception("Error saving kernel source code to disk")

        # Save wrapped kernel source code
        try:
            self._save_wrapper_kernel_code_to_disk(kernel, cache_path, verbose)
        except Exception:
            self.logger.exception("Error saving host kernel source code to disk")

        # Save the kernel library
        try:
            # Save CUBIN or SO file
            self._save_so_cubin_to_disk(kernel, cache_path, verbose)

        except Exception:
            self.logger.exception("Error saving kernel library to disk")

        # Save kernel parameters
        try:
            params_path = os.path.join(cache_path, self.params_path)
            if verbose:
                self.logger.debug(f"Saving kernel parameters to disk: {params_path}")
            KernelCache._safe_write_file(params_path, "wb", lambda file: cloudpickle.dump(kernel.params, file))
        except Exception:
            self.logger.exception("Error saving kernel parameters to disk")

    def _load_kernel_from_disk(
        self,
        key: str,
        target: str | Target = "auto",
        target_host: str | Target | None = None,
        out_idx: list[int] | None = None,
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] = "tvm_ffi",
        pass_configs: dict | None = None,
        compile_flags: list[str] | str | None = None,
        func: Callable | None = None,
        verbose: bool = False,
    ) -> JITKernel | None:
        """
        Loads a previously compiled kernel from disk cache.

        Args:
            key (str): The hash key identifying the kernel.
            target (Union[str, Target]): Compilation target platform. Defaults to "auto".
            target_host (Union[str, Target], optional): Host target platform.
            out_idx (List[int], optional): Indices specifying which outputs to return.
            execution_backend (Literal): Backend type for execution. Defaults to "tvm_ffi".
            pass_configs (dict, optional): Configuration for compiler passes.
            func (Callable, optional): The original function.
            verbose (bool): Enable verbose log messages.

        Returns:
            JITKernel: The loaded kernel if found, None otherwise.
        """
        cache_path = self._get_cache_path(key)
        device_kernel_path = os.path.join(cache_path, self.device_kernel_path)
        host_kernel_path = os.path.join(cache_path, self.host_kernel_path)
        kernel_lib_path = os.path.join(cache_path, self.kernel_lib_path)
        params_path = os.path.join(cache_path, self.params_path)

        required_files = self._get_required_files(cache_path)

        if not all([os.path.exists(file) for file in required_files]):
            return None

        # Load the kernel source file (optional)
        device_kernel_source, host_kernel_source = self._load_kernel_source(device_kernel_path, host_kernel_path, verbose)

        # Load kernel parameters
        kernel_params: list[KernelParam] | None = None
        try:
            if verbose:
                self.logger.debug(f"Loading kernel parameters from file: {params_path}")
            with open(params_path, "rb") as f:
                kernel_params = cloudpickle.load(f)
        except Exception:
            self.logger.exception("Error loading kernel parameters from disk")

        return self._build_kernel(
            func=func,
            host_kernel_source=host_kernel_source,
            device_kernel_source=device_kernel_source,
            kernel_lib_path=kernel_lib_path,
            kernel_params=kernel_params,
            target=target,
            target_host=target_host,
            out_idx=out_idx,
            execution_backend=execution_backend,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )

    def _clear_disk_cache(self):
        """
        Removes all cached kernels from disk.

        Note:
            This operation will delete the entire cache directory and recreate it empty.
            Use with caution as this operation cannot be undone.
        """
        try:
            # Delete the entire cache directory
            shutil.rmtree(env.TILELANG_CACHE_DIR)

            # Re-create the cache directory
            KernelCache._create_dirs()
        except Exception:
            self.logger.exception("Error clearing disk cache")

    def _save_kernel_source_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        device_kernel_path = os.path.join(cache_path, self.device_kernel_path)
        if verbose:
            self.logger.debug(f"Saving kernel source code to file: {device_kernel_path}")
        if kernel.kernel_source is not None:
            KernelCache._safe_write_file(device_kernel_path, "w", lambda file: file.write(kernel.kernel_source))

    def _save_wrapper_kernel_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        host_kernel_path = os.path.join(cache_path, self.host_kernel_path)
        if verbose:
            self.logger.debug(f"Saving wrapped kernel source code to file: {host_kernel_path}")
        KernelCache._safe_write_file(host_kernel_path, "w", lambda file: file.write(kernel.adapter.get_kernel_source()))

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        kernel_lib_path = os.path.join(cache_path, self.kernel_lib_path)
        src_lib_path = kernel.adapter.libpath
        if verbose:
            self.logger.debug(f"Saving kernel library to file: {kernel_lib_path}")
        KernelCache._safe_write_file(kernel_lib_path, "wb", lambda file: file.write(KernelCache._load_binary(src_lib_path)))

    def _get_required_files(self, cache_path: str) -> list[str]:
        kernel_lib_path = os.path.join(cache_path, self.kernel_lib_path)
        params_path = os.path.join(cache_path, self.params_path)
        return [kernel_lib_path, params_path]

    def _load_kernel_source(self, device_kernel_path: str, host_kernel_path: str, verbose: bool = False) -> tuple[str | None, str | None]:
        try:
            if verbose:
                self.logger.debug(f"Loading kernel source code from file: {device_kernel_path}")
            with open(device_kernel_path) as f:
                device_kernel_source = f.read()
        except Exception:
            device_kernel_source = None
            self.logger.exception("Error loading kernel source code from disk")
        try:
            if verbose:
                self.logger.debug(f"Loading wrapped kernel source code from file: {host_kernel_path}")
            with open(host_kernel_path) as f:
                host_kernel_source = f.read()
        except Exception:
            host_kernel_source = None
            self.logger.exception("Error loading host kernel source code from disk")
        return device_kernel_source, host_kernel_source

    def _set_adapter_cache_path(self, kernel: JITKernel, cache_path: str):
        return

    def _build_kernel(
        self,
        func: Callable | None,
        host_kernel_source: str,
        device_kernel_source: str,
        kernel_lib_path: str,
        kernel_params: list[KernelParam] | None,
        target: str | Target,
        target_host: str | Target | None,
        out_idx: list[int] | None,
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"],
        pass_configs: dict | None,
        compile_flags: list[str] | str | None,
    ) -> JITKernel | None:
        # Check all required components and report specific failures
        missing_components = []
        if not host_kernel_source:
            missing_components.append("host_kernel_source")
        if not device_kernel_source:
            missing_components.append("device_kernel_source")
        if not kernel_params:
            missing_components.append("kernel_params")

        if missing_components:
            self.logger.warning("Cannot build kernel from cache: missing required component(s): %s", ", ".join(missing_components))
            return None

        return JITKernel.from_database(
            func=func,
            host_kernel_source=host_kernel_source,
            device_kernel_source=device_kernel_source,
            kernel_lib_path=kernel_lib_path,
            params=kernel_params,
            target=target,
            target_host=target_host,
            out_idx=out_idx,
            execution_backend=execution_backend,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )
