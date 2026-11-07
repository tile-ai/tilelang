from __future__ import annotations
from functools import wraps
from collections.abc import Callable

import torch
from tvm import tirx

from tilelang import tvm as tvm

from ..base import BaseKernelAdapter
from tilelang.engine.param import KernelParam


def _param_name(var, buffer_map) -> str:
    """Buffer name for a PrimFunc parameter, falling back to the variable's own name."""
    if buffer_map is not None and var in buffer_map:
        return str(buffer_map[var].name)
    return str(getattr(var, "name", None) or var.name_hint)


def _argument_order(func_or_mod, device_func) -> list[int] | None:
    """Positions of a call's arguments, in the order the kernel declares its buffers.

    Device codegen neither preserves the host signature's parameter order nor
    emits parameters the kernel body never reads, so forwarding arguments
    positionally can bind tensors to the wrong ``[[buffer(N)]]`` slots -- the
    kernel then writes into an input and reads from an output, silently and
    without an error. Returns None when the two signatures cannot be matched
    by name, which leaves the arguments untouched.
    """
    if device_func is None:
        return None
    host = func_or_mod
    if not isinstance(host, tirx.PrimFunc):
        host = next(iter(host.functions.values()), None)
    if host is None:
        return None
    declared = [_param_name(v, getattr(host, "buffer_map", None)) for v in host.params]
    emitted = [_param_name(v, getattr(device_func, "buffer_map", None)) for v in device_func.params]
    if len(set(declared)) != len(declared) or not set(emitted) <= set(declared):
        return None
    return [declared.index(name) for name in emitted]


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

        device_func = None
        for var, func in device_mod.functions.items():
            assert var.name_hint == self.kernel_name
            device_func = func
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
        # Must precede super().__init__: it calls _post_init(), which builds the launcher.
        self._argument_order = _argument_order(func_or_mod, device_func)
        super().__init__(func_or_mod, result_idx=result_idx, params=params)

    _kernel = None

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
            _kernel = getattr(torch.mps.compile_shader(self.kernel_global_source), self.kernel_name)
            _threads = [x * y for (x, y) in zip(self.block_info, self.grid_info)]

            _order = self._argument_order

            @wraps(_kernel)
            def launcher(*args: torch.Tensor):
                if _order is not None and len(args) > max(_order, default=-1):
                    args = tuple(args[index] for index in _order)
                return _kernel(
                    *args,
                    threads=_threads,
                    group_size=self.block_info,
                )

            self._kernel = launcher

        return self._kernel
