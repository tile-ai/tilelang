from __future__ import annotations
from functools import wraps
from collections.abc import Callable

import torch
from tvm_tirx import ir, tirx

from ..base import BaseKernelAdapter
from tilelang.engine.param import KernelParam
from tilelang.utils.language import retrieve_func_from_module
from .metal_launch import MetalLaunchPlan


class MetalKernelAdapter(BaseKernelAdapter):
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        func_or_mod: tirx.Function | ir.IRModule,
        device_mod: ir.IRModule | None = None,
        kernel_global_source: str | None = None,
        verbose: bool = False,
        host_mod: ir.IRModule | None = None,
    ):
        self.kernel_global_source = kernel_global_source
        self.prim_func = func_or_mod if isinstance(func_or_mod, tirx.Function) else retrieve_func_from_module(func_or_mod)
        self.launch_plan = MetalLaunchPlan(self.prim_func, device_mod, host_mod)
        self.kernel_name = self.launch_plan.kernel_name
        self.verbose = verbose
        self.block_info = self.launch_plan.block_info
        self.grid_info = self.launch_plan.grid_info
        self._kernel = None
        super().__init__(func_or_mod, result_idx=result_idx, params=params)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        if kernel_only:
            # Return just the kernel function body, stripping Metal
            # module-level boilerplate (includes, structs, etc.).
            idx = self.kernel_global_source.find("kernel void ")
            if idx >= 0:
                return self.kernel_global_source[idx:]
        return self.kernel_global_source

    def _convert_torch_func(self) -> Callable:
        if self._kernel is None:
            shader = torch.mps.compile_shader(self.kernel_global_source)
            _kernel = getattr(shader, self.kernel_name)
            plan = self.launch_plan

            @wraps(_kernel)
            def launcher(*args: torch.Tensor):
                if len(args) != len(self.params):
                    raise ValueError(f"Expected {len(self.params)} preallocated arguments, got {len(args)}")
                return _kernel(
                    *(args[index] for index in plan.argument_order),
                    threads=plan.threads,
                    group_size=plan.block_info,
                )

            self._shader = shader
            self._kernel = launcher

        return self._kernel
