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
import re

from tilelang.env import TILELANG_CACHE_DIR, is_cache_enabled

KERNEL_PATH = "kernel.cu"
WRAPPED_KERNEL_PATH = "warpped_kernel.cu"
KERNEL_LIB_PATH = "kernel_lib.so"
PARAMS_PATH = "params.pkl"


class KernelCache:
    """
    Caches compiled kernels using a class and database persistence to avoid redundant compilation.
    Cache files:
        kernel.cu: The compiled kernel source code
        warpped_kernel.cu: The compiled wrapped kernel source code
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
        func_script = self._sort_layout_map_reindex(func.script())
        func_binary = cloudpickle.dumps(func_script)
        key_data = {
            "func": sha256(func_binary).hexdigest(),  # Use SHA256 to generate hash key
            "out_idx": (tuple(out_idx) if isinstance(out_idx, (list, tuple)) else [out_idx]),
            "args_repr": tuple(
                repr(arg) for arg in args
            ),  # Use repr to serialize arguments, may need more robust serialization
            "target": str(target),
            "target_host": str(target_host) if target_host else None,
            "execution_backend": execution_backend,
        }
        key_string = json.dumps(key_data, sort_keys=True)  # Sort keys to ensure consistency
        return sha256(key_string.encode()).hexdigest()  # Use SHA256 to generate hash key

    def _sort_layout_map_reindex(self, func_string):
        """
        Sorts the key-value pairs in the layout_map attribute of a TVM function string alphabetically,
        and re-indexes metadata["tl.Layout"][index] to start from 0 incrementally.
        Also sorts the T.handle variable declarations before layout_map alphabetically.

        Parameters:
            func_string (str): Function string containing T.block_attr({"layout_map": { ... }})

        Returns:
            str: New function string with sorted and re-indexed layout_map key-value pairs,
                and sorted T.handle variable declarations.
                Returns the original string if layout_map is not found.
        """
        start_marker = 'T.block_attr({"layout_map": {'
        end_marker = '}})'

        start_index = func_string.find(start_marker)
        if start_index == -1:
            return func_string  # Return original string if layout_map is not found

        end_index = func_string.find(end_marker, start_index)
        if end_index == -1:
            return func_string  # Return original string if closing marker is not found (error case)

        layout_map_start = start_index + len(start_marker)
        layout_map_content = func_string[layout_map_start:end_index]

        # Parse key-value pairs from layout_map
        pairs = []
        if layout_map_content.strip():
            for item in layout_map_content.strip().split(','):
                item = item.strip()
                if not item:
                    continue
                parts = item.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    pairs.append((key, value))

        # Sort layout_map key-value pairs by key
        sorted_pairs = sorted(pairs, key=lambda pair: pair[0])

        # Rebuild sorted layout_map string and re-index
        sorted_layout_map_content = ', '.join(
            [f"{key}: metadata[\"tl.Layout\"][{i}]" for i, (key, value) in enumerate(sorted_pairs)])
        if sorted_pairs:
            sorted_layout_map_str = '{' + sorted_layout_map_content + '}'
        else:
            sorted_layout_map_str = '{}'

        # --- Handle T.handle variable declarations ---
        handle_lines = []
        handle_start_index = -1
        handle_end_index = -1
        lines = func_string.splitlines(keepends=True)

        layout_map_line_index = -1
        for line_index, line_content in enumerate(lines):
            if start_marker.strip() in line_content.strip():  # find line index of layout_map_start
                layout_map_line_index = line_index
                break

        if layout_map_line_index != -1:
            for line_index in range(layout_map_line_index - 1, -1, -1):
                line = lines[line_index]
                if re.match(r'\s*(\w+)\s*=\s*T\.handle\("float16",\s*"shared\.dyn"\)\s*', line):
                    handle_lines.insert(
                        0, line
                    )  # Find handle lines in reverse order, insert at the beginning to keep relative order
                    handle_start_index = line_index
                    if handle_end_index == -1:
                        handle_end_index = line_index + 1  # Record the end line index (exclusive) of the handle block
                else:
                    if handle_start_index != -1:  # stop if already found handle lines and current line is not handle
                        break
                    elif not line.strip() or line.strip().startswith(
                            '#'):  # skip comment lines before handle block
                        continue
                    else:  # stop if non-handle line found before handle block starts
                        break

        sorted_handle_lines = sorted(
            handle_lines, key=lambda line: re.match(r'\s*(\w+)\s*=', line).group(1).strip()
        ) if handle_lines else []

        # Construct new function string
        prefix_lines = lines[:
                             handle_start_index] if handle_start_index != -1 else lines[:
                                                                                        layout_map_line_index]  # up to handle block or layout_map if no handle
        handle_separator_lines = lines[
            handle_end_index:
            layout_map_line_index] if handle_start_index != -1 and handle_end_index != -1 else [
            ]  # lines between handle and layout_map, from handle_end_index
        suffix_lines = lines[layout_map_line_index + 1:]  # lines after layout_map block line

        new_lines = prefix_lines + [line for line in sorted_handle_lines
                                   ] + handle_separator_lines + [
                                       lines[layout_map_line_index].replace(
                                           layout_map_content, sorted_layout_map_str)
                                   ] + suffix_lines
        new_func_string = "".join(new_lines)

        return new_func_string

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
            target_host=target_host)
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
