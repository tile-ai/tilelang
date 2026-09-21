from __future__ import annotations
from functools import wraps
from collections.abc import Callable
from typing import Any

import torch
from tvm import tirx

from tilelang import tvm as tvm

from ..base import BaseKernelAdapter, CachedTextSource
from tilelang.engine.param import KernelParam


class MetalKernelAdapter(BaseKernelAdapter):
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        #  target: Union[str, Target],
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        #  host_mod: Optional[tvm.IRModule] = None,
        device_mod: tvm.IRModule | None = None,
        kernel_global_source: str | None = None,
        verbose: bool = False,
        #  pass_configs: Optional[Dict[str, Any]] = None,
        #  compile_flags: Optional[List[str]] = None
    ):
        self.kernel_global_source = kernel_global_source
        if isinstance(func_or_mod, tirx.PrimFunc):
            func_name = func_or_mod.attrs["global_symbol"]
        else:
            func_name = func_or_mod.__name__
        self.kernel_name = func_name + "_kernel"
        self.verbose = verbose

        self.block_info = [1, 1, 1]
        self.grid_info = [1, 1, 1]

        for var, func in device_mod.functions.items():
            assert var.name_hint == self.kernel_name
            thread_extent = func.attrs["thread_extent"]
            for tag, extent in thread_extent.items():
                if "threadIdx" in tag:
                    self.block_info["xyz".index(tag[-1])] = int(extent)
                elif "blockIdx" in tag:
                    self.grid_info["xyz".index(tag[-1])] = int(extent)
            break
        else:
            raise AssertionError(f"no kernel with name {func_name}")

        # print(self.block_info, self.grid_info)
        super().__init__(func_or_mod, result_idx=result_idx, params=params)

    _kernel = None

    @classmethod
    def from_database(
        cls,
        params: list[KernelParam],
        result_idx: list[int],
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        device_kernel_source: CachedTextSource,
        launch_metadata: dict[str, Any],
        host_kernel_source: CachedTextSource | None = None,
        verbose: bool = False,
    ):
        """Recreate the adapter from cached Metal source and its launch metadata."""
        adapter = cls.__new__(cls)
        adapter._set_cached_text_source("kernel_global_source", "_kernel_global_source_path", device_kernel_source)
        source = adapter._load_cached_text_source("kernel_global_source", "_kernel_global_source_path")
        if source is None:
            raise ValueError("cached Metal kernel source is unavailable")
        adapter.kernel_global_source = source
        if host_kernel_source is not None:
            adapter._set_cached_text_source("host_kernel_source", "_host_kernel_source_path", host_kernel_source)
        adapter.verbose = verbose
        try:
            adapter.kernel_name = str(launch_metadata["kernel_name"])
            adapter.block_info = [int(extent) for extent in launch_metadata["block"]]
            adapter.grid_info = [int(extent) for extent in launch_metadata["grid"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid Metal launch metadata: {launch_metadata!r}") from error
        if len(adapter.block_info) != 3 or len(adapter.grid_info) != 3:
            raise ValueError(f"invalid Metal launch metadata: {launch_metadata!r}")
        BaseKernelAdapter.__init__(adapter, func_or_mod, params=params, result_idx=result_idx)
        return adapter

    @property
    def launch_metadata(self) -> dict[str, Any]:
        """Plain data needed to relaunch the cached Metal source."""
        return {
            "kernel_name": self.kernel_name,
            "block": [int(extent) for extent in self.block_info],
            "grid": [int(extent) for extent in self.grid_info],
        }

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        if kernel_only:
            # Return just the kernel function body, stripping Metal
            # module-level boilerplate (includes, structs, etc.).
            idx = self.kernel_global_source.find("kernel void ")
            if idx >= 0:
                return self.kernel_global_source[idx:]
        return self.kernel_global_source

    def get_host_source(self) -> str:
        """The lowered host program of a kernel loaded from the cache."""
        source = self._load_cached_text_source("host_kernel_source", "_host_kernel_source_path")
        if source is None:
            raise RuntimeError("the host program of this cached Metal kernel is unavailable")
        return source

    def _convert_torch_func(self) -> Callable:
        if self._kernel is None:
            _kernel = getattr(torch.mps.compile_shader(self.kernel_global_source), self.kernel_name)
            _threads = [x * y for (x, y) in zip(self.block_info, self.grid_info)]

            @wraps(_kernel)
            def launcher(*args: torch.Tensor):
                return _kernel(
                    *args,
                    threads=_threads,
                    group_size=self.block_info,
                )

            self._kernel = launcher

        return self._kernel
