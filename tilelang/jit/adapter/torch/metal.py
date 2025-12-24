from __future__ import annotations
from functools import wraps
from typing import Callable

import torch
from tvm import tir

from tilelang import tvm as tvm

from ..base import BaseKernelAdapter
from tilelang.engine.param import KernelParam


class MetalKernelAdapter(BaseKernelAdapter):
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        #  target: Union[str, Target],
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        #  host_mod: Optional[tvm.IRModule] = None,
        device_mod: tvm.IRModule | None = None,
        kernel_global_source: str | None = None,
        verbose: bool = False,
        #  pass_configs: Optional[Dict[str, Any]] = None,
        #  compile_flags: Optional[List[str]] = None
    ):
        self.kernel_global_source = kernel_global_source
        if isinstance(func_or_mod, tir.PrimFunc):
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
                    self.block_info["xyz".index(tag[-1])] = extent
                elif "blockIdx" in tag:
                    self.grid_info["xyz".index(tag[-1])] = extent
            break
        else:
            raise AssertionError(f"no kernel with name {func_name}")

        # print(self.block_info, self.grid_info)
        super().__init__(func_or_mod, result_idx=result_idx, params=params)

    _kernel = None

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
