# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils with class and database persistence - KernelCache Class"""

import os
import json
import shutil
from hashlib import sha256
from typing import Callable, List, Literal, Union
from tvm.target import Target
from tvm.tir import PrimFunc
from tilelang.jit import JITKernel
import threading
import cloudpickle
import logging

from tilelang.env import TILELANG_CACHE_DIR  # noqa: F401


class KernelCache:
    """
    Caches compiled kernels using a class and database persistence to avoid redundant compilation.
    """
    _instance = None  # For implementing singleton pattern
    _lock = threading.Lock()  # For thread safety

    def __new__(cls, cache_dir=TILELANG_CACHE_DIR):
        """Singleton pattern to ensure only one KernelCache instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(KernelCache, cls).__new__(cls)
                cls._instance._cache = {}  # In-memory cache
                cls._instance.cache_dir = cache_dir  # Cache directory
                os.makedirs(cls._instance.cache_dir, exist_ok=True)  # Ensure cache directory exists
                cls._instance.logger = logging.getLogger(__name__)  # Initialize logger
                cls._instance.logger.setLevel(
                    logging.ERROR)  # Set default logging level to ERROR, can be adjusted
        return cls._instance

    def _generate_key(self, func: Callable, out_idx: List[int],
                      execution_backend: Literal["dlpack", "ctypes", "cython"], args,
                      target: Union[str, Target], target_host: Union[str, Target]) -> str:
        """
        Generates a unique cache key.
        """
        func_binary = cloudpickle.dumps(func)
        key_data = {
            "func": sha256(func_binary).hexdigest(),  # Use SHA256 to generate hash key
            "out_idx": tuple(out_idx) if isinstance(out_idx, (list, tuple)) else [out_idx],
            "args_repr": tuple(
                repr(arg) for arg in args
            ),  # Use repr to serialize arguments, may need more robust serialization
            "target": str(target),
            "target_host": str(target_host) if target_host else None,
            "execution_backend": execution_backend,
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
        key = self._generate_key(func, out_idx, execution_backend, args, target, target_host)
        with self._lock:  # Thread-safe access to cache
            if key in self._cache:
                return self._cache[key]

            # Attempt to load from disk
            kernel = self._load_kernel_from_disk(key, target, target_host, out_idx,
                                                 execution_backend, pass_configs, func)
            if kernel:
                self._cache[key] = kernel  # Load to in-memory cache
                return kernel

            # Compile kernel if cache miss
            kernel = JITKernel(
                func,
                out_idx=out_idx,
                execution_backend=execution_backend,
                target=target,
                target_host=target_host,
                verbose=verbose,
                pass_configs=pass_configs,
            )
            self._cache[key] = kernel  # Store in in-memory cache
            self._save_kernel_to_disk(key, kernel, func)
            return kernel

    def clear_cache(self):
        """
        Clears the entire kernel cache, including both in-memory and disk cache.
        """
        with self._lock:  # Thread-safe operation
            self._cache.clear()  # Clear in-memory cache
            self._clear_disk_cache()  # Clear disk cache

    def _get_cache_path(self, key: str) -> str:
        """
        Gets the cache file path for a given key.
        """
        return os.path.join(self.cache_dir, key)

    def _save_kernel_to_disk(self, key: str, kernel: JITKernel, func: Callable = None):
        """
        Saves the compiled kernel to disk.
        """
        cache_path = self._get_cache_path(key)
        os.makedirs(cache_path, exist_ok=True)  # Ensure directory exists

        # Save rt_mod as a str
        try:
            artifact_path = os.path.join(cache_path, "tvm_tmp_mod.txt")
            with open(artifact_path, "w") as f:
                f.write(kernel.rt_mod.imported_modules[0].get_source())
        except Exception as e:
            self.logger.error(f"Error saving kernel module to disk: {e}")

        try:
            dump_path = os.path.join(cache_path, "tvm_params.pkl")
            with open(dump_path, "wb") as f:
                cloudpickle.dump(kernel.params, f)
        except Exception as e:
            self.logger.error(f"Error saving kernel parameters to disk: {e}")

    def _load_kernel_from_disk(self,
                               key: str,
                               target: Union[str, Target] = "auto",
                               target_host: Union[str, Target] = None,
                               out_idx: List[int] = None,
                               execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
                               pass_configs: dict = None,
                               func: Callable = None) -> JITKernel:
        """
        Loads kernel from disk.
        """
        cache_path = self._get_cache_path(key)
        if not os.path.exists(cache_path):
            return None
        rt_module = None
        rt_params = None
        try:
            artifact_path = os.path.join(cache_path, "tvm_tmp_mod.txt")
            with open(artifact_path, "r") as f:
                rt_module = f.read()
        except Exception as e:
            self.logger.error(f"Error loading kernel module from disk: {e}")
        try:
            dump_path = os.path.join(cache_path, "tvm_params.pkl")
            with open(dump_path, "rb") as f:
                rt_params = cloudpickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading kernel parameters from disk: {e}")

        if rt_module and rt_params:
            return JITKernel(
                rt_module_src=rt_module,
                rt_params=rt_params,
                execution_backend=execution_backend,
                target=target,
                target_host=target_host,
                out_idx=out_idx,
                pass_configs=pass_configs,
                func=func,
            )
        else:
            return None

    def _clear_disk_cache(self):
        """
        Clears the cache directory on disk.
        """
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)  # Delete entire cache directory
            os.makedirs(self.cache_dir, exist_ok=True)  # Re-create cache directory
        except Exception as e:
            self.logger.error(f"Error clearing disk cache: {e}")
