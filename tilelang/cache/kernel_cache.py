# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils with class and database persistence - KernelCache Class"""

import os
import json
import shutil
from hashlib import sha256
from typing import Callable, List, Literal, Union, Optional
from tvm.target import Target
from tvm.tir import PrimFunc
from tilelang.jit import JITKernel
from tilelang.engine.param import KernelParam
import threading
import cloudpickle
import logging

from tilelang.env import TILELANG_CACHE_DIR, is_cache_enabled

KERNEL_PATH = "kernel.cu"
WRAPPED_KERNEL_PATH = "wrapped_kernel.cu"
KERNEL_LIB_PATH = "kernel_lib.so"
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

    def __new__(cls, cache_dir=TILELANG_CACHE_DIR):
        """
        Implements singleton pattern for KernelCache class.

        Args:
            cache_dir (str): Directory path for storing kernel cache. Defaults to TILELANG_CACHE_DIR.

        Returns:
            KernelCache: The singleton instance of KernelCache.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    instance = super().__new__(cls)
                    instance.cache_dir = cache_dir
                    os.makedirs(instance.cache_dir, exist_ok=True)

                    instance.logger = logging.getLogger(__name__)
                    instance.logger.setLevel(logging.ERROR)
                    instance._memory_cache = {}  # Initialize memory cache
                    cls._instance = instance
        return cls._instance

    def _generate_key(
        self,
        func: Callable,
        out_idx: List[int],
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        args=None,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        pass_configs: dict = None,
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
        func_binary = cloudpickle.dumps(func.script())
        key_data = {
            "func": sha256(func_binary).hexdigest(),  # Use SHA256 to generate hash key
            "out_idx": (tuple(out_idx) if isinstance(out_idx, (list, tuple)) else [out_idx]),
            "args_repr": tuple(
                repr(arg) for arg in args
            ),  # Use repr to serialize arguments, may need more robust serialization
            "target": str(target),
            "target_host": str(target_host) if target_host else None,
            "execution_backend": execution_backend,
            "pass_configs": pass_configs,
        }
        key_string = json.dumps(key_data, sort_keys=True)  # Sort keys to ensure consistency
        return sha256(key_string.encode()).hexdigest()  # Use SHA256 to generate hash key

    def cached(
        self,
        func: PrimFunc = None,
        out_idx: List[int] = None,
        *args,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        verbose: bool = False,
        pass_configs: dict = None,
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
        if not is_cache_enabled():
            return JITKernel(
                func,
                out_idx=out_idx,
                execution_backend=execution_backend,
                target=target,
                target_host=target_host,
                verbose=verbose,
                pass_configs=pass_configs,
            )

        key = self._generate_key(
            func=func,
            out_idx=out_idx,
            execution_backend=execution_backend,
            args=args,
            target=target,
            target_host=target_host,
            pass_configs=pass_configs,
        )
        with self._lock:
            # First check in-memory cache
            if key in self._memory_cache:
                return self._memory_cache[key]

            # Then check disk cache
            kernel = self._load_kernel_from_disk(key, target, target_host, out_idx,
                                                 execution_backend, pass_configs, func)
            if kernel is not None:
                # Populate memory cache with disk result
                self._memory_cache[key] = kernel
                return kernel

        # Compile kernel if cache miss; leave critical section
        kernel = JITKernel(
            func,
            out_idx=out_idx,
            execution_backend=execution_backend,
            target=target,
            target_host=target_host,
            verbose=verbose,
            pass_configs=pass_configs,
        )
        if execution_backend == "dlpack":
            self.logger.warning("DLPack backend does not support cache saving to disk.")
        else:
            with self._lock:  # enter critical section again to check and update disk cache
                disk_kernel = self._load_kernel_from_disk(
                    key,
                    target,
                    target_host,
                    out_idx,
                    execution_backend,
                    pass_configs,
                    func,
                )
                if disk_kernel is None:
                    self._save_kernel_to_disk(key, kernel, func)

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
        return os.path.join(self.cache_dir, key)

    def _save_kernel_to_disk(self, key: str, kernel: JITKernel, func: Callable = None):
        """
        Persists a compiled kernel to disk cache.

        Args:
            key (str): The hash key identifying the kernel.
            kernel (JITKernel): The compiled kernel to be saved.
            func (Callable, optional): The original function.

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
            with open(kernel_path, "w") as f:
                f.write(kernel.artifact.kernel_source)
        except Exception as e:
            self.logger.error(f"Error saving kernel source code to disk: {e}")

        # Save wrapped kernel source code
        try:
            wrapped_kernel_path = os.path.join(cache_path, WRAPPED_KERNEL_PATH)
            with open(wrapped_kernel_path, "w") as f:
                f.write(kernel.adapter.get_kernel_source())
        except Exception as e:
            self.logger.error(f"Error saving wrapped kernel source code to disk: {e}")

        # Save kernel library
        try:
            kernel_lib_path = os.path.join(cache_path, KERNEL_LIB_PATH)
            src_lib_path = kernel.adapter.libpath
            shutil.copy(src_lib_path, kernel_lib_path)
        except Exception as e:
            self.logger.error(f"Error saving kernel library to disk: {e}")

        # Save kernel parameters
        try:
            params_path = os.path.join(cache_path, PARAMS_PATH)
            with open(params_path, "wb") as f:
                cloudpickle.dump(kernel.params, f)
        except Exception as e:
            self.logger.error(f"Error saving kernel parameters to disk: {e}")

    def _load_kernel_from_disk(
        self,
        key: str,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        out_idx: List[int] = None,
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        pass_configs: dict = None,
        func: Callable = None,
    ) -> JITKernel:
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

        Returns:
            JITKernel: The loaded kernel if found, None otherwise.
        """
        cache_path = self._get_cache_path(key)
        if not os.path.exists(cache_path):
            return None

        kernel_global_source: Optional[str] = None
        kernel_params: Optional[List[KernelParam]] = None

        try:
            wrapped_kernel_path = os.path.join(cache_path, WRAPPED_KERNEL_PATH)
            with open(wrapped_kernel_path, "r") as f:
                kernel_global_source = f.read()
        except Exception as e:
            self.logger.error(f"Error loading wrapped kernel source code from disk: {e}")

        kernel_lib_path = os.path.join(cache_path, KERNEL_LIB_PATH)

        # Load kernel parameters
        try:
            params_path = os.path.join(cache_path, PARAMS_PATH)
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
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)  # Delete entire cache directory
            os.makedirs(self.cache_dir, exist_ok=True)  # Re-create cache directory
        except Exception as e:
            self.logger.error(f"Error clearing disk cache: {e}")
