# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils with class and database persistence"""

import os
import json
import shutil
from hashlib import sha256
from typing import Callable, List, Union
from tvm.target import Target
from tvm.tir import PrimFunc
from tilelang import compile
from tilelang.jit import JITKernel
import threading

class KernelCache:
    """
    Caches compiled kernels using a class and database persistence to avoid redundant compilation.
    """
    _instance = None  # For implementing singleton pattern
    _lock = threading.Lock() # For thread safety

    def __new__(cls, cache_dir=".kernel_cache"):
        """Singleton pattern to ensure only one KernelCache instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(KernelCache, cls).__new__(cls)
                cls._instance._cache = {}  # In-memory cache
                cls._instance.cache_dir = cache_dir  # Cache directory
                os.makedirs(cls._instance.cache_dir, exist_ok=True) # Ensure cache directory exists
                cls._instance._load_cache_from_disk() # Load cache from disk upon initialization
        return cls._instance

    def _generate_key(self, func: Callable, out_idx: List[int], args, target, target_host) -> str:
        """
        Generates a unique cache key.
        """
        func_name = func.__name__ if hasattr(func, '__name__') else str(func) # Get function name, handle PrimFunc cases
        key_data = {
            "func_name": func_name,
            "out_idx": tuple(out_idx) if out_idx else None,
            "args_repr": tuple(repr(arg) for arg in args), # Use repr to serialize arguments, may need more robust serialization
            "target": str(target),
            "target_host": str(target_host),
        }
        key_string = json.dumps(key_data, sort_keys=True) # Sort keys to ensure consistency
        return sha256(key_string.encode()).hexdigest() # Use SHA256 to generate hash key

    def cached_kernel(
        self,
        func: Callable,
        out_idx: List[int] = None,
        *args,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
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
        key = self._generate_key(func, out_idx, args, target, target_host)

        with self._lock: # Thread-safe access to cache
            if key in self._cache:
                print(f"Loading kernel from in-memory cache: {key}") # Debugging info
                return self._cache[key]

            # Attempt to load from disk
            kernel = self._load_kernel_from_disk(key)
            if kernel:
                print(f"Loading kernel from disk cache: {key}") # Debugging info
                self._cache[key] = kernel # Load to in-memory cache
                return kernel

            # Compile kernel if cache miss
            print(f"Compiling new kernel and caching: {key}") # Debugging info
            program = func if isinstance(func, PrimFunc) else func(*args)
            kernel = compile(program, out_idx=out_idx, target=target, target_host=target_host)

            self._cache[key] = kernel # Store in in-memory cache
            self._save_kernel_to_disk(key, kernel) # Persist to disk
            return kernel

    def clear_cache(self):
        """
        Clears the entire kernel cache, including both in-memory and disk cache.
        """
        with self._lock: # Thread-safe operation
            self._cache.clear() # Clear in-memory cache
            self._clear_disk_cache() # Clear disk cache

    def _get_cache_path(self, key: str) -> str:
        """
        Gets the cache file path for a given key.
        """
        return os.path.join(self.cache_dir, key)

    def _save_kernel_to_disk(self, key: str, kernel: JITKernel):
        """
        Saves the compiled kernel to disk.
        """
        cache_path = self._get_cache_path(key)
        os.makedirs(cache_path, exist_ok=True) # Ensure directory exists
        kernel_file = os.path.join(cache_path, "kernel.o") # Use .o as kernel file extension
        try:
            kernel.export_library(kernel_file) # Export kernel to file
            config_file = os.path.join(cache_path, "config.json") # Save configuration info
            config_data = { # Save config info used to generate key for future load validation
                "key": key,
                "cache_dir": self.cache_dir
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            print(f"Kernel saved to disk: {kernel_file}") # Debugging info
        except Exception as e:
            print(f"Error saving kernel to disk: {e}") # Error handling

    def _load_kernel_from_disk(self, key: str) -> Union[JITKernel, None]:
        """
        Loads kernel from disk.
        """
        cache_path = self._get_cache_path(key)
        kernel_file = os.path.join(cache_path, "kernel.o")
        if os.path.exists(kernel_file):
            try:
                kernel = JITKernel(kernel_file) # Load kernel from file
                config_file = os.path.join(cache_path, "config.json")
                if os.path.exists(config_file): # Optionally load and validate config info
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                        if config_data.get("key") != key: # Simple config validation to ensure correct cache loaded
                            print(f"Warning: Loaded kernel config does not match requested key: {key}") # Warning info
                            return None # Config mismatch, do not use cache
                print(f"Kernel loaded from disk: {kernel_file}") # Debugging info
                return kernel
            except Exception as e:
                print(f"Error loading kernel from disk: {e}") # Error handling
                return None
        return None # Cache file not found

    def _load_cache_from_disk(self):
        """
        Loads all cached kernels from disk to memory upon initialization (optional, can be load-on-demand).
        Currently implemented as load-on-demand, checking disk cache in `cached_kernel`.
        """
        # Currently implemented as load-on-demand, pre-loading all cache is not needed
        pass # Can optionally implement pre-loading here, iterate cache dir and load all kernels to memory

    def _clear_disk_cache(self):
        """
        Clears the cache directory on disk.
        """
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir) # Delete entire cache directory
            os.makedirs(self.cache_dir, exist_ok=True) # Re-create cache directory
            print(f"Disk cache cleared, path: {self.cache_dir}") # Info message
        except Exception as e:
            print(f"Error clearing disk cache: {e}") # Error handling


# Create singleton instance of KernelCache
_kernel_cache_instance = KernelCache()

def cached(
    func: Callable,
    out_idx: List[int] = None,
    *args,
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
) -> JITKernel:
    """
    Caches and reuses compiled kernels (using KernelCache class).
    """
    return _kernel_cache_instance.cached_kernel(func, out_idx, *args, target=target, target_host=target_host)

def clear_cache():
    """
    Clears the entire kernel cache (using KernelCache class).
    """
    _kernel_cache_instance.clear_cache()