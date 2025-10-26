"""The cache utils with class and database persistence - KernelCache Class"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from hashlib import sha256
from typing import Callable, Literal

import cloudpickle
from tvm.target import Target
from tvm.tir import PrimFunc

from tilelang.engine.param import KernelParam
from tilelang import env
from tilelang.jit import JITKernel
from tilelang import __version__

KERNEL_PATH = "kernel.cu"
WRAPPED_KERNEL_PATH = "wrapped_kernel.cu"
KERNEL_LIB_PATH = "kernel_lib.so"
KERNEL_CUBIN_PATH = "kernel.cubin"
KERNEL_PY_PATH = "kernel.py"
PARAMS_PATH = "params.pkl"


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
    execution_backend: Literal["dlpack", "ctypes", "cython", "nvrtc"] = "cython"

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
        execution_backend: Literal["dlpack", "ctypes", "cython", "nvrtc"] = "cython",
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
            execution_backend (Literal): Backend type for execution. Defaults to "cython".
            args: Arguments passed to the function.
            target (Union[str, Target]): Compilation target platform. Defaults to "auto".
            target_host (Union[str, Target], optional): Host target platform.

        Returns:
            str: SHA256 hash key for the kernel configuration.
        """
        self.execution_backend = execution_backend
        func_binary = cloudpickle.dumps(func.script(show_meta=True))
        key_data = {
            "version": __version__,
            "func": sha256(func_binary).hexdigest(),  # Use SHA256 to generate hash key
            "out_idx": (tuple(out_idx) if isinstance(out_idx, (list, tuple)) else [out_idx]),
            "args_repr": tuple(
                repr(arg) for arg in args
            ),  # Use repr to serialize arguments, may need more robust serialization
            "target": str(target),
            "target_host": str(target_host) if target_host else None,
            "execution_backend": execution_backend,
            "pass_configs": pass_configs,
            "compile_flags": compile_flags,
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
        target: str | Target = "auto",
        target_host: str | Target = None,
        execution_backend: Literal["dlpack", "ctypes", "cython", "nvrtc"] = "cython",
        verbose: bool = False,
        pass_configs: dict = None,
        compile_flags: list[str] | str | None = None,
    ) -> JITKernel:
        """
        Caches and reuses compiled kernels to avoid redundant compilation.

        Args:
            func: Function to be compiled or a prepared PrimFunc
            out_idx: Indices specifying which outputs to return
            target: Compilation target platform
            target_host: Host target platform
            *args: Arguments passed to func

        Returns:
            JITKernel: The compiled kernel, either freshly compiled or from cache
        """
        if not env.is_cache_enabled():
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
        with self._lock:
            # First check in-memory cache
            if key in self._memory_cache:
                self.logger.warning("Found kernel in memory cache. For better performance," \
                                    " consider using `@tilelang.jit` instead of direct kernel caching.")
                return self._memory_cache[key]

            if verbose:
                self.logger.debug(f"Checking disk cache for kernel {func.attrs['global_symbol']}")

            # Then check disk cache
            kernel = self._load_kernel_from_disk(key, target, target_host, out_idx,
                                                 execution_backend, pass_configs, compile_flags,
                                                 func, verbose)
            if kernel is not None:
                if verbose:
                    self.logger.debug(
                        f"Found kernel in disk cache for {func.attrs['global_symbol']}")
                # Populate memory cache with disk result
                self._memory_cache[key] = kernel
                return kernel

        if verbose:
            self.logger.debug(f"No cached kernel for {func.attrs['global_symbol']}")
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
        if execution_backend in ("dlpack", "torch"):
            self.logger.warning("DLPack or torch backend does not support cache saving to disk.")
        else:
            with self._lock:
                if env.is_cache_enabled():
                    self._save_kernel_to_disk(key, kernel, func, verbose)

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

    def _save_kernel_to_disk(self,
                             key: str,
                             kernel: JITKernel,
                             func: Callable = None,
                             verbose: bool = False):
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
            kernel_path = os.path.join(cache_path, KERNEL_PATH)
            if verbose:
                self.logger.debug(f"Saving kernel source code to file: {kernel_path}")
            if kernel.kernel_source is not None:
                KernelCache._safe_write_file(kernel_path, "w",
                                             lambda file: file.write(kernel.kernel_source))
        except Exception as e:
            self.logger.error(f"Error saving kernel source code to disk: {e}")

        # Save wrapped kernel source code
        try:
            wrapped_kernel_path = os.path.join(cache_path, WRAPPED_KERNEL_PATH)
            if verbose:
                self.logger.debug(
                    f"Saving wrapped kernel source code to file: {wrapped_kernel_path}")
            KernelCache._safe_write_file(
                wrapped_kernel_path, "w",
                lambda file: file.write(kernel.adapter.get_kernel_source()))
        except Exception as e:
            self.logger.error(f"Error saving wrapped kernel source code to disk: {e}")

        # Save the kernel library
        try:
            # Save CUBIN or SO file
            kernel_lib_path = KERNEL_CUBIN_PATH if self.execution_backend == "nvrtc" else KERNEL_LIB_PATH
            kernel_lib_path = os.path.join(cache_path, kernel_lib_path)
            src_lib_path = kernel.adapter.libpath
            if verbose:
                self.logger.debug(f"Saving kernel library to file: {kernel_lib_path}")
            KernelCache._safe_write_file(
                kernel_lib_path, "wb",
                lambda file: file.write(KernelCache._load_binary(src_lib_path)))

            # Save an extra Python file for NVRTC
            if self.execution_backend == "nvrtc":
                kernel_py_path = os.path.join(cache_path, KERNEL_PY_PATH)
                src_lib_path = src_lib_path.replace(".cubin", ".py")
                if verbose:
                    self.logger.debug(f"Saving kernel nvrtc python code to file: {kernel_py_path}")
                KernelCache._safe_write_file(
                    kernel_py_path, "wb",
                    lambda file: file.write(KernelCache._load_binary(src_lib_path)))
        except Exception as e:
            self.logger.error(f"Error saving kernel library to disk: {e}")

        # Save kernel parameters
        try:
            params_path = os.path.join(cache_path, PARAMS_PATH)
            if verbose:
                self.logger.debug(f"Saving kernel parameters to disk: {params_path}")
            KernelCache._safe_write_file(params_path, "wb",
                                         lambda file: cloudpickle.dump(kernel.params, file))
        except Exception as e:
            self.logger.error(f"Error saving kernel parameters to disk: {e}")

    def _load_kernel_from_disk(
        self,
        key: str,
        target: str | Target = "auto",
        target_host: str | Target = None,
        out_idx: list[int] = None,
        execution_backend: Literal["dlpack", "ctypes", "cython", "nvrtc"] = "cython",
        pass_configs: dict = None,
        compile_flags: list[str] | str | None = None,
        func: Callable = None,
        verbose: bool = False,
    ) -> JITKernel | None:
        """
        Loads a previously compiled kernel from disk cache.

        Args:
            key (str): The hash key identifying the kernel.
            target (Union[str, Target]): Compilation target platform. Defaults to "auto".
            target_host (Union[str, Target], optional): Host target platform.
            out_idx (List[int], optional): Indices specifying which outputs to return.
            execution_backend (Literal): Backend type for execution. Defaults to "cython".
            pass_configs (dict, optional): Configuration for compiler passes.
            func (Callable, optional): The original function.
            verbose (bool): Enable verbose log messages.

        Returns:
            JITKernel: The loaded kernel if found, None otherwise.
        """
        cache_path = self._get_cache_path(key)
        wrapped_kernel_path = os.path.join(cache_path, WRAPPED_KERNEL_PATH)
        kernel_lib_path = os.path.join(
            cache_path, KERNEL_CUBIN_PATH if self.execution_backend == "nvrtc" else KERNEL_LIB_PATH)
        params_path = os.path.join(cache_path, PARAMS_PATH)
        if not all([os.path.exists(file) for file in (kernel_lib_path, params_path)]):
            return None

        kernel_global_source: str | None = None
        kernel_params: list[KernelParam] | None = None

        # Load the kernel source file (optional)
        try:
            if verbose:
                self.logger.debug(
                    f"Loading wrapped kernel source code from file: {wrapped_kernel_path}")
            with open(wrapped_kernel_path) as f:
                kernel_global_source = f.read()
        except Exception as e:
            self.logger.error(f"Error loading wrapped kernel source code from disk: {e}")

        # Load kernel parameters
        try:
            if verbose:
                self.logger.debug(f"Loading kernel parameters from file: {params_path}")
            with open(params_path, "rb") as f:
                kernel_params = cloudpickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading kernel parameters from disk: {e}")

        if kernel_global_source and kernel_params:
            return JITKernel.from_database(
                func=func,
                kernel_global_source=kernel_global_source,
                kernel_lib_path=kernel_lib_path,
                params=kernel_params,
                target=target,
                target_host=target_host,
                out_idx=out_idx,
                execution_backend=execution_backend,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        else:
            return None

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
        except Exception as e:
            self.logger.error(f"Error clearing disk cache: {e}")
